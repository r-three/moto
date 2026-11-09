"""Stage 4 — Extend merge rules via additional BPE training.

Starting from a pruned vocabulary (output of Stage 2 / picky_bpe.py), this
stage runs additional BPE iterations on a target-language corpus to learn new
merge rules for tokens that are under-represented in the base Llama vocabulary.

The HuggingFace ``tokenizers`` library is used for fast BPE training.  The
resulting tokenizer is saved in HF format so it can be loaded with
``AutoTokenizer.from_pretrained``.

Merge conflict note
-------------------
When two extension tokenizers for different languages learn the same merge (e.g.
both Arabic and Hindi BPE produce a merge for the byte pair representing a common
Unicode boundary), the conflict is recorded in ``merge_conflicts.json`` under the
output directory.  Downstream resolution is handled in Stage 5 (moto_tokenizer).

Output (written to ``output_dir/{lang_code}/``):
    tokenizer.json          — HuggingFace fast tokenizer
    tokenizer_config.json   — config (required for AutoTokenizer.from_pretrained)
    merge_conflicts.json    — cross-language merge collisions (if any)

Usage (CLI):
    python -m moto.llama_moto.extend_merges \\
        pruned_vocab_json=picky_bpe_output/pruned_vocab.json \\
        languages='["arb_Arab","hin_Deva"]' \\
        corpus_dataset=flexitok/flores-21langs \\
        extra_merges=8000 \\
        output_dir=extended_merges_output
"""

import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from moto.llama_moto.args import ExtendMergesArgs
from moto.llama_moto.picky_bpe import _iter_jsonl_dir
from moto.utils.args import load_args

logger = logging.getLogger(__name__)


# ── Corpus helpers ─────────────────────────────────────────────────────────────

def _iter_corpus_for_lang(
    lang: str,
    args: ExtendMergesArgs,
) -> Iterator[str]:
    """Yield sentences for a single language from JSONL dirs or HF dataset."""
    if args.corpus_files:
        for path_str in args.corpus_files:
            p = Path(path_str)
            lang_dir = p / lang if p.is_dir() else p
            if lang_dir.is_dir():
                yield from _iter_jsonl_dir(lang_dir)
        return

    if args.corpus_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.corpus_dataset, lang, split=args.corpus_split)
        for ex in ds:
            text = ex.get(args.corpus_text_column, "").strip()
            if text:
                yield text
        return

    raise ValueError("Provide corpus_files=[...] or corpus_dataset in ExtendMergesArgs")


def _write_corpus_to_file(sentences: Iterator[str], path: Path, max_sentences: int) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for sent in sentences:
            fh.write(sent.strip().replace("\n", " ") + "\n")
            n += 1
            if max_sentences and n >= max_sentences:
                break
    return n


# ── BPE training ───────────────────────────────────────────────────────────────

def train_bpe_extension(
    pruned_vocab: Dict[str, int],
    corpus_path: str,
    extra_merges: int,
    pretokenizer: str,
    min_frequency: int,
) -> "tokenizers.Tokenizer":  # noqa: F821
    """Train additional BPE merges on top of a pruned vocabulary.

    Returns a ``tokenizers.Tokenizer`` object (HuggingFace tokenizers library).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    # Initialise BPE model with the pruned vocab as the starting vocabulary.
    # We provide an empty merges list; the trainer will learn new merges.
    vocab_list = sorted(pruned_vocab.items(), key=lambda x: x[1])
    vocab_dict = {tok: idx for tok, idx in vocab_list}

    model = models.BPE(vocab=vocab_dict, merges=[], byte_fallback=True)
    tokenizer = Tokenizer(model)

    # Pre-tokenizer
    if pretokenizer == "byte_level":
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    elif pretokenizer in ("gpt2", "gpt4"):
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    elif pretokenizer == "whitespace":
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Target vocab size = current size + extra merges
    target_vocab_size = len(pruned_vocab) + extra_merges

    trainer = trainers.BpeTrainer(
        vocab_size=target_vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<s>", "</s>"],
        show_progress=True,
    )

    logger.info(
        "Training BPE: start_vocab=%d, extra_merges=%d, target_vocab=%d, min_freq=%d",
        len(pruned_vocab), extra_merges, target_vocab_size, min_frequency,
    )
    tokenizer.train(files=[corpus_path], trainer=trainer)
    logger.info("BPE training done. Final vocab size: %d", tokenizer.get_vocab_size())
    return tokenizer


# ── Merge conflict detection ───────────────────────────────────────────────────

def find_merge_conflicts(
    lang_tokenizers: Dict[str, "tokenizers.Tokenizer"],  # noqa: F821
) -> Dict[str, List[str]]:
    """Return {merge_rule: [lang1, lang2, ...]} for merges learned by multiple languages."""
    merge_to_langs: Dict[str, List[str]] = defaultdict(list)
    for lang, tok in lang_tokenizers.items():
        if hasattr(tok.model, "merges"):
            for pair in tok.model.merges:
                key = f"{pair[0]} {pair[1]}" if isinstance(pair, (list, tuple)) else str(pair)
                merge_to_langs[key].append(lang)

    conflicts = {k: v for k, v in merge_to_langs.items() if len(v) > 1}
    if conflicts:
        logger.warning(
            "Found %d merge rules learned by multiple languages — "
            "see merge_conflicts.json for details",
            len(conflicts),
        )
    return conflicts


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: ExtendMergesArgs):
    if not args.pruned_vocab_json:
        raise ValueError("Provide pruned_vocab_json= (output of picky_bpe stage)")
    if not args.languages:
        raise ValueError("Provide at least one language via languages=[...]")

    with open(args.pruned_vocab_json, "r", encoding="utf-8") as fh:
        pruned_vocab: Dict[str, int] = json.load(fh)
    logger.info("Loaded pruned vocab: %d tokens", len(pruned_vocab))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    lang_tokenizers: Dict[str, object] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for lang in args.languages:
            logger.info("=== Processing language: %s ===", lang)
            corpus_path = Path(tmpdir) / f"{lang}.txt"
            sentences = _iter_corpus_for_lang(lang, args)
            n = _write_corpus_to_file(sentences, corpus_path, args.max_sentences)
            logger.info("Wrote %d sentences to %s", n, corpus_path)

            if n == 0:
                logger.warning("No sentences found for %s — skipping", lang)
                continue

            ext_tok = train_bpe_extension(
                pruned_vocab=pruned_vocab,
                corpus_path=str(corpus_path),
                extra_merges=args.extra_merges,
                pretokenizer=args.pretokenizer,
                min_frequency=args.min_frequency,
            )
            lang_tokenizers[lang] = ext_tok

            lang_out = out_root / lang
            lang_out.mkdir(exist_ok=True)
            ext_tok.save(str(lang_out / "tokenizer.json"))

            # Write a minimal tokenizer_config.json so AutoTokenizer can load it
            config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 131072,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
            with open(lang_out / "tokenizer_config.json", "w") as fh:
                json.dump(config, fh, indent=2)

            logger.info("Saved extended tokenizer for %s → %s", lang, lang_out)

    # Cross-language conflict analysis
    conflicts = find_merge_conflicts(lang_tokenizers)
    if conflicts:
        conflict_path = out_root / "merge_conflicts.json"
        with open(conflict_path, "w", encoding="utf-8") as fh:
            json.dump(conflicts, fh, ensure_ascii=False, indent=2)
        logger.info("Wrote %d merge conflicts to %s", len(conflicts), conflict_path)

    summary = {
        "languages_processed": list(lang_tokenizers.keys()),
        "extra_merges": args.extra_merges,
        "base_pruned_vocab_size": len(pruned_vocab),
        "output_dirs": [str(out_root / lang) for lang in lang_tokenizers],
    }
    with open(out_root / "extension_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Extension complete. Summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = load_args(ExtendMergesArgs)
    main(args)
