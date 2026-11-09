"""Subword fertility metric for tokenizers.

Fertility measures how many subword tokens a tokenizer produces per word.
Lower is generally better: a score of 1.0 means every word maps to a single
token (ideal); higher values indicate more fragmentation.

    fertility(lang, tokenizer) = Σ |tokenize(word)| / Σ |words in text|

Word segmentation strategy:
  - Default (whitespace): split on Unicode whitespace, strip punctuation. Works
    well for Latin, Cyrillic, Arabic, Greek, and other space-separated scripts.
  - CJK (cmn_Hani, jpn_Jpan, zho_Hans, kor_Hang): character-level split,
    filtering non-letter/digit characters.  True word segmentation (MeCab,
    jieba) would be more accurate but adds hard dependencies.

Usage (module):
    from moto.tokenizer_intrinsic_metrics.fertility import compute_fertility

    results = compute_fertility(
        tokenizers={"my-tok": "/path/to/tokenizer"},
        languages=["ita_Latn", "tur_Latn", "arb_Arab"],
    )
    # results["my-tok"]["ita_Latn"] -> float

Usage (CLI):
    python -m moto.tokenizer_intrinsic_metrics.fertility \\
        tokenizers='["gsaltintas/flexitok-bpe_ita_Latn_8000"]' \\
        languages='["ita_Latn","tur_Latn"]'
"""

import logging
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from moto.utils.args import Args, load_args

logger = logging.getLogger(__name__)

# Default FLORES dataset settings
_FLORES_DATASET = "openlanguagedata/flores_plus"
_FLORES_DATASET = "flexitok/flores-21langs"
# flores_plus uses one config per language code (e.g. "ita_Latn"); no single "all" config.
# _FLORES_CONFIG is kept for backward-compat exports but is not used in loading.
_FLORES_CONFIG = "default"
_FLORES_SPLIT = "dev"
_FLORES_KEY_PREFIX = "text"  # column name inside each per-language config

_FLORES_SPLIT = "devtest"
_FLORES_KEY_PREFIX = "sentence"  # column name inside each per-language config
# Scripts that use character-level segmentation (no whitespace between words)
_CJK_SCRIPTS = {"Hani", "Jpan", "Kore", "Hang"}


def _script_of(lang_code: str) -> str:
    """Return the ISO 15924 script tag from a FLORES lang code (e.g. 'cmn_Hani' -> 'Hani')."""
    parts = lang_code.split("_")
    return parts[1] if len(parts) == 2 else ""


def _whitespace_words(text: str) -> List[str]:
    """Split on whitespace; keep tokens that contain at least one letter or digit."""
    return [
        tok for tok in re.split(r"\s+", text.strip())
        if tok and any(unicodedata.category(c)[0] in ("L", "N") for c in tok)
    ]


def _character_words(text: str) -> List[str]:
    """Return individual characters that are letters or digits (for CJK scripts)."""
    return [c for c in text if unicodedata.category(c)[0] in ("L", "N")]


def _word_tokenizer_for(lang_code: str) -> Callable[[str], List[str]]:
    """Return an appropriate word-tokenization function for the given language."""
    if _script_of(lang_code) in _CJK_SCRIPTS:
        return _character_words
    return _whitespace_words


def _load_tokenizer(tokenizer: Union[str, PreTrainedTokenizerBase]) -> PreTrainedTokenizerBase:
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


def _load_flores_sentences(
    languages: List[str],
    dataset_name: str,
    dataset_split: str,
    sample_size: int,
    seed: int,
) -> Dict[str, List[str]]:
    """Load and align FLORES sentences for each requested language.

    Each language is fetched as its own dataset config (one download per language),
    which is how ``openlanguagedata/flores_plus`` is structured.  Sentences are
    parallel by index across languages, so we sample the same positional indices
    for all languages to preserve alignment.

    Returns:
        {lang_code: [sentence, ...]}  — up to *sample_size* entries per language.
    """
    lang_sentences: Dict[str, List[str]] = {}
    for lang in languages:
        logger.info("Loading %s config=%s split=%s", dataset_name, lang, dataset_split)
        # import code; code.interact(local=dict(globals(), **locals()))  # DEBUG
        key = _FLORES_KEY_PREFIX
        if Path(dataset_name).exists():
            ds = load_dataset("json", data_files={lang: [x.as_posix() for x in Path(dataset_name, lang).glob("*.jsonl")][0]}, streaming=True, split=lang).take(sample_size)# split=f"train:{sample_size}")
            key = "text" # fallback
        else:
            ds = load_dataset(dataset_name, lang, split=dataset_split)

        lang_sentences[lang] = [
            ex[key] for ex in ds
            if ex.get(key, "").strip()
            # if ex.get(key, "").split("\n")[0].strip()
        ]

    n_available = min(len(sents) for sents in lang_sentences.values())
    if n_available == 0:
        import code; code.interact(local=dict(globals(), **locals()))  # DEBUG
        raise ValueError(
            f"No examples found for some languages in {dataset_name}. "
            "Check that the language codes are valid configs for this dataset."
        )

    random.seed(seed)
    indices = random.sample(range(n_available), min(sample_size, n_available))
    logger.info("Using %d / %d available examples", len(indices), n_available)

    return {lang: [sents[i] for i in indices] for lang, sents in lang_sentences.items()}


@dataclass
class FertilityArgs(Args):
    tokenizers: List[str] = field(default_factory=list)
    # HF repo IDs or local paths, e.g. ["gsaltintas/flexitok-bpe_ita_Latn_8000"]
    languages: List[str] = field(default_factory=list)
    # FLORES language codes, e.g. ["ita_Latn", "tur_Latn"]
    sample_size: int = 1000
    dataset_name: str = _FLORES_DATASET
    dataset_split: str = _FLORES_SPLIT
    output_csv: Optional[str] = "fertility_results.csv"
    seed: int = 42


def compute_fertility(
    tokenizers: Dict[str, Union[str, PreTrainedTokenizerBase]],
    languages: List[str],
    sample_size: int = 1000,
    dataset_name: str = _FLORES_DATASET,
    dataset_split: str = _FLORES_SPLIT,
    seed: int = 42,
    skip_diff_lang: bool = False,
    non_skip_patterns: List[str] = ["AllL", "script", "llama"],
) -> Dict[str, Dict[str, float]]:
    """Compute subword fertility for each tokenizer across languages.

    Args:
        tokenizers: {name: tokenizer_path_or_object}
        languages: FLORES language codes to evaluate (e.g. ["ita_Latn", "arb_Arab"])
        sample_size: number of sentences to use per language
        dataset_name: HuggingFace dataset (per-language configs, e.g. flores_plus)
        dataset_split: dataset split
        seed: random seed for sampling

    Returns:
        {tokenizer_name: {lang_code: fertility_score}}
    """
    lang_texts = _load_flores_sentences(languages, dataset_name, dataset_split, sample_size, seed)

    # Pre-segment words per language (independent of tokenizer)
    word_segs: Dict[str, List[List[str]]] = {}
    for lang in languages:
        word_tokenize = _word_tokenizer_for(lang)
        word_segs[lang] = [word_tokenize(text) for text in lang_texts[lang]]

    results: Dict[str, Dict[str, float]] = {}

    for tok_name, tok_or_path in tokenizers.items():
        logger.info("Processing tokenizer: %s", tok_name)
        tokenizer = _load_tokenizer(tok_or_path)

        fertility_scores: Dict[str, float] = {}
        for lang in languages:
            if skip_diff_lang and lang not in tok_name:
                if any(pat in tok_name for pat in non_skip_patterns):
                    # logger.info("Keeping %s for %s due to non-skip pattern match", tok_name, lang)
                    pass
                else:
                    # logger.info("Skipping %s for %s (not in tokenizer name)", tok_name, lang)
                    continue
            total_words = 0
            total_subwords = 0
            for words in word_segs[lang]:
                total_words += len(words)
                for word in words:
                    total_subwords += len(tokenizer.encode(word, add_special_tokens=False))
            score = total_subwords / total_words if total_words > 0 else 0.0
            fertility_scores[lang] = round(score, 4)

        results[tok_name] = fertility_scores

    return results


def fertility_to_dataframe(results: Dict[str, Dict[str, float]]):
    """Convert compute_fertility output to a pandas DataFrame (tokenizers as rows)."""
    import pandas as pd
    return pd.DataFrame.from_dict(results, orient="index")


def main(args: FertilityArgs):
    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")
    if not args.languages:
        raise ValueError("Provide at least one language via languages=[...]")

    tokenizers = {tok: tok for tok in args.tokenizers}
    results = compute_fertility(
        tokenizers=tokenizers,
        languages=argxs.languages,
        sample_size=args.sample_size,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seed=args.seed,
    )

    df = fertility_to_dataframe(results)
    print(df.to_string())

    if args.output_csv:
        df.to_csv(args.output_csv, index_label="tokenizer")
        logger.info("Saved to %s", args.output_csv)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_args(FertilityArgs)
    main(args)
