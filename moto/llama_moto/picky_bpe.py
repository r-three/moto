"""Stage 2 — PickyBPE: vocabulary pruning.

PickyBPE prunes the base tokenizer vocabulary down to the tokens that are
actually *useful* on the target corpus.  A token is considered useful if it
appears at least ``min_token_freq`` times when the corpus is tokenized with the
base tokenizer.

Invariants that are always upheld regardless of frequency:
  - All 256 byte-fallback tokens (<0x00> … <0xFF>) are kept so that the
    pruned tokenizer can still encode any Unicode text losslessly.
  - All special tokens (BOS, EOS, PAD, UNK, …) are kept.

Output (written to ``output_dir``):
  pruned_vocab.json   — {token_string: base_tokenizer_id} for kept tokens
  pruned_stats.json   — summary statistics

Usage (CLI):
    python -m moto.llama_moto.picky_bpe \\
        base_tokenizer=meta-llama/Meta-Llama-3.1-8B \\
        corpus_dataset=flexitok/flores-21langs \\
        languages='["arb_Arab","hin_Deva"]' \\
        min_token_freq=5 \\
        output_dir=picky_bpe_output
"""

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from moto.llama_moto.args import PickyBPEArgs
from moto.utils.args import load_args

logger = logging.getLogger(__name__)

_BYTE_TOKEN_RE_STR = r"^<0x[0-9A-Fa-f]{2}>$"


# ── Corpus iteration ───────────────────────────────────────────────────────────

def _iter_jsonl_dir(lang_dir: Path, text_key: str = "text") -> Iterator[str]:
    for jf in sorted(lang_dir.glob("*.jsonl")):
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = json.loads(line).get(text_key, "").strip()
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning("Skipping %s: %s", jf, exc)


def iter_corpus(
    args: PickyBPEArgs,
    seed: int = 42,
) -> Iterator[str]:
    """Yield sentences from local JSONL files or HuggingFace dataset."""
    if args.corpus_files:
        for path in args.corpus_files:
            p = Path(path)
            if p.is_dir():
                yield from _iter_jsonl_dir(p)
            elif p.suffix == ".jsonl":
                yield from _iter_jsonl_dir(p.parent)
            else:
                with open(p, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            yield line
        return

    if args.corpus_dataset and args.languages:
        # Per-language FLORES-style dataset
        from datasets import load_dataset
        for lang in args.languages:
            logger.info("Loading corpus for %s from %s", lang, args.corpus_dataset)
            ds = load_dataset(args.corpus_dataset, lang, split=args.corpus_split)
            for ex in ds:
                text = ex.get(args.corpus_text_column, "").strip()
                if text:
                    yield text
        return

    raise ValueError(
        "Provide corpus_files=[...] or (corpus_dataset + languages) in PickyBPEArgs"
    )


# ── Core pruning logic ─────────────────────────────────────────────────────────

def build_byte_token_ids(tokenizer: PreTrainedTokenizerBase) -> Set[int]:
    import re
    pattern = re.compile(_BYTE_TOKEN_RE_STR)
    return {tid for tok, tid in tokenizer.get_vocab().items() if pattern.match(tok)}


def build_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> Set[int]:
    ids: Set[int] = set()
    if hasattr(tokenizer, "all_special_ids"):
        ids.update(tokenizer.all_special_ids)
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            ids.add(val)
    return ids


def run_picky_bpe(
    tokenizer: PreTrainedTokenizerBase,
    corpus_iter: Iterator[str],
    min_token_freq: int,
    keep_byte_tokens: bool,
    keep_special_tokens: bool,
    max_sentences: int = 0,
) -> Dict[str, int]:
    """Prune vocabulary to tokens appearing >= min_token_freq times.

    Returns:
        {token_string: original_token_id} for every kept token.
    """
    protected: Set[int] = set()
    if keep_byte_tokens:
        protected |= build_byte_token_ids(tokenizer)
    if keep_special_tokens:
        protected |= build_special_token_ids(tokenizer)

    logger.info("Protected token ids (byte + special): %d", len(protected))

    # Count token frequencies on the corpus
    freq: Counter = Counter()
    n_sents = 0
    for sent in corpus_iter:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        freq.update(ids)
        n_sents += 1
        if max_sentences and n_sents >= max_sentences:
            break
        if n_sents % 50_000 == 0:
            logger.info("Scanned %d sentences, unique tokens so far: %d", n_sents, len(freq))

    logger.info("Finished scanning %d sentences. Total unique token ids seen: %d", n_sents, len(freq))

    # Determine which ids to keep
    kept_ids: Set[int] = protected | {
        tid for tid, count in freq.items() if count >= min_token_freq
    }

    # Build id → token_string reverse map
    id_to_tok = {v: k for k, v in tokenizer.get_vocab().items()}
    pruned_vocab: Dict[str, int] = {
        id_to_tok[tid]: tid for tid in kept_ids if tid in id_to_tok
    }

    total = len(tokenizer.get_vocab())
    kept = len(pruned_vocab)
    removed = total - kept
    logger.info(
        "Vocab pruning: %d / %d kept (%.1f%% removed, %d tokens dropped)",
        kept, total, 100 * removed / total, removed,
    )
    return pruned_vocab


def main(args: PickyBPEArgs):
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    corpus = iter_corpus(args, seed=args.seed)

    pruned_vocab = run_picky_bpe(
        tokenizer=tokenizer,
        corpus_iter=corpus,
        min_token_freq=args.min_token_freq,
        keep_byte_tokens=args.keep_byte_tokens,
        keep_special_tokens=args.keep_special_tokens,
        max_sentences=args.max_sentences,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = out_dir / "pruned_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(pruned_vocab, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote pruned vocab (%d tokens) to %s", len(pruned_vocab), vocab_path)

    stats = {
        "base_tokenizer": args.base_tokenizer,
        "original_vocab_size": len(tokenizer.get_vocab()),
        "pruned_vocab_size": len(pruned_vocab),
        "min_token_freq": args.min_token_freq,
        "keep_byte_tokens": args.keep_byte_tokens,
        "keep_special_tokens": args.keep_special_tokens,
    }
    stats_path = out_dir / "pruned_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Wrote stats to %s", stats_path)

    return pruned_vocab


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = load_args(PickyBPEArgs)
    main(args)
