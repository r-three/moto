"""Stage 1 — Byte-fallback analysis.

For each target language, tokenize sentences with the given tokenizer
(default: Llama 3) and measure:

  - byte_fallback_rate : fraction of emitted tokens that are <0x??> byte tokens
  - fertility          : subword tokens per word
  - total_tokens       : total tokens emitted across all sample sentences
  - byte_fallback_count: raw count of <0x??> tokens

A high byte_fallback_rate for a language signals that the tokenizer has little
native coverage and is a strong candidate for vocabulary extension.

Data sources (tried in order for each language):
  1. Local JSONL files under ``data_dir/{lang_code}/*.jsonl``,
     each line: ``{"text": "..."}``
  2. FLORES dev/devtest split (fallback when local data is absent)

Usage (CLI):
    python -m moto.llama_moto.byte_fallback_analysis \\
        tokenizer=meta-llama/Meta-Llama-3.1-8B \\
        languages='["arb_Arab","cmn_Hani","hin_Deva","jpn_Jpan"]' \\
        data_dir=data/flexitok \\
        sample_size=500 \\
        output_csv=results/byte_fallback.csv
"""

import json
import logging
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from moto.llama_moto.args import ByteFallbackAnalysisArgs
from moto.utils.args import load_args

logger = logging.getLogger(__name__)

_CJK_SCRIPTS = {"Hani", "Jpan", "Kore", "Hang"}


def _script_of(lang_code: str) -> str:
    parts = lang_code.split("_")
    return parts[1] if len(parts) == 2 else ""


def _whitespace_words(text: str) -> List[str]:
    return [
        tok for tok in re.split(r"\s+", text.strip())
        if tok and any(unicodedata.category(c)[0] in ("L", "N") for c in tok)
    ]


def _character_words(text: str) -> List[str]:
    return [c for c in text if unicodedata.category(c)[0] in ("L", "N")]


def _word_tokenize(lang_code: str, text: str) -> List[str]:
    if _script_of(lang_code) in _CJK_SCRIPTS:
        return _character_words(text)
    return _whitespace_words(text)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_from_jsonl_dir(
    lang_code: str,
    data_dir: str,
    sample_size: int,
    seed: int,
    text_key: str = "text",
) -> Optional[List[str]]:
    """Load sentences from ``{data_dir}/{lang_code}/*.jsonl``.

    Returns a list of up to *sample_size* non-empty strings, or None if the
    directory does not exist or contains no readable sentences.
    """
    lang_dir = Path(data_dir) / lang_code
    if not lang_dir.is_dir():
        return None

    jsonl_files = sorted(lang_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    sentences: List[str] = []
    for jf in jsonl_files:
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get(text_key, "").strip()
                        if text:
                            sentences.append(text)
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning("Could not read %s: %s", jf, exc)

    if not sentences:
        return None

    random.seed(seed)
    if len(sentences) > sample_size:
        sentences = random.sample(sentences, sample_size)
    logger.info(
        "Loaded %d sentences for %s from %s",
        len(sentences), lang_code, lang_dir,
    )
    return sentences


def _load_from_flores(
    lang_code: str,
    dataset_name: str,
    split: str,
    sample_size: int,
    seed: int,
) -> List[str]:
    from datasets import load_dataset  # lazy import

    logger.info(
        "No local data for %s — falling back to FLORES %s split=%s",
        lang_code, dataset_name, split,
    )
    ds = load_dataset(dataset_name, lang_code, split=split)
    sents = [ex["text"] for ex in ds if ex.get("text", "").strip()]
    random.seed(seed)
    if len(sents) > sample_size:
        sents = random.sample(sents, sample_size)
    return sents


def load_sentences(
    languages: List[str],
    data_dir: Optional[str],
    flores_dataset: str,
    flores_split: str,
    sample_size: int,
    seed: int,
    text_key: str = "text",
) -> Dict[str, List[str]]:
    """Return {lang_code: [sentence, ...]} for all requested languages.

    Prefers local JSONL data under *data_dir*; falls back to FLORES.
    """
    result: Dict[str, List[str]] = {}
    for lang in languages:
        sents: Optional[List[str]] = None
        if data_dir:
            sents = _load_from_jsonl_dir(lang, data_dir, sample_size, seed, text_key)
        if sents is None:
            sents = _load_from_flores(lang, flores_dataset, flores_split, sample_size, seed)
        result[lang] = sents
    return result


# ── Core analysis ──────────────────────────────────────────────────────────────

@dataclass
class LanguageByteFallbackStats:
    lang: str
    source: str                 # "local_jsonl" or "flores"
    total_tokens: int
    byte_fallback_count: int
    byte_fallback_rate: float   # byte_fallback_count / total_tokens
    fertility: float            # subword tokens per word
    total_words: int


def analyze_byte_fallback(
    tokenizer: PreTrainedTokenizerBase,
    lang_sentences: Dict[str, List[str]],
    byte_token_re: re.Pattern,
    data_sources: Optional[Dict[str, str]] = None,
) -> Dict[str, LanguageByteFallbackStats]:
    """Compute byte-fallback stats for each language.

    Args:
        tokenizer: HuggingFace tokenizer to probe.
        lang_sentences: {lang_code: [sentence, ...]}
        byte_token_re: compiled regex matching byte-fallback token strings.
        data_sources: optional {lang_code: "local_jsonl"|"flores"} labels.

    Returns:
        {lang_code: LanguageByteFallbackStats}
    """
    vocab = tokenizer.get_vocab()
    byte_token_ids: set[int] = {
        tid for tok, tid in vocab.items() if byte_token_re.match(tok)
    }
    logger.info(
        "Found %d byte-fallback token ids in tokenizer vocab",
        len(byte_token_ids),
    )

    results: Dict[str, LanguageByteFallbackStats] = {}
    for lang, sentences in lang_sentences.items():
        total_tokens = 0
        byte_count = 0
        total_words = 0

        for sent in sentences:
            ids = tokenizer.encode(sent, add_special_tokens=False)
            total_tokens += len(ids)
            byte_count += sum(1 for i in ids if i in byte_token_ids)
            total_words += len(_word_tokenize(lang, sent))

        rate = byte_count / total_tokens if total_tokens > 0 else 0.0
        fertility = total_tokens / total_words if total_words > 0 else 0.0
        source = (data_sources or {}).get(lang, "unknown")
        results[lang] = LanguageByteFallbackStats(
            lang=lang,
            source=source,
            total_tokens=total_tokens,
            byte_fallback_count=byte_count,
            byte_fallback_rate=round(rate, 4),
            fertility=round(fertility, 4),
            total_words=total_words,
        )
        logger.info(
            "%-20s [%-12s]  fallback_rate=%.3f  fertility=%.3f  tokens=%d",
            lang, source, rate, fertility, total_tokens,
        )

    return results


def stats_to_dataframe(stats: Dict[str, LanguageByteFallbackStats]):
    import pandas as pd
    rows = [
        {
            "lang": s.lang,
            "source": s.source,
            "total_tokens": s.total_tokens,
            "byte_fallback_count": s.byte_fallback_count,
            "byte_fallback_rate": s.byte_fallback_rate,
            "fertility": s.fertility,
            "total_words": s.total_words,
        }
        for s in stats.values()
    ]
    return pd.DataFrame(rows).sort_values("byte_fallback_rate", ascending=False)


def main(args: ByteFallbackAnalysisArgs):
    if not args.languages:
        raise ValueError("Provide at least one language via languages=[...]")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    byte_token_re = re.compile(args.byte_token_pattern)

    # Track which source was used per language for reporting
    data_sources: Dict[str, str] = {}
    lang_sentences: Dict[str, List[str]] = {}

    for lang in args.languages:
        sents: Optional[List[str]] = None
        if args.data_dir:
            sents = _load_from_jsonl_dir(
                lang, args.data_dir, args.sample_size, args.seed, args.text_key
            )
            if sents is not None:
                data_sources[lang] = "local_jsonl"
        if sents is None:
            sents = _load_from_flores(
                lang, args.flores_dataset, args.flores_split, args.sample_size, args.seed
            )
            data_sources[lang] = "flores"
        lang_sentences[lang] = sents

    stats = analyze_byte_fallback(tokenizer, lang_sentences, byte_token_re, data_sources)

    df = stats_to_dataframe(stats)
    print(df.to_string(index=False))

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        logger.info("Saved byte-fallback analysis to %s", args.output_csv)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = load_args(ByteFallbackAnalysisArgs)
    main(args)
