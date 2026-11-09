"""Proportion of Continued Words (PCW) metric for tokenizers.

PCW is the fraction of words that are split into two or more subword tokens.
A word is "continued" if the tokenizer produces ≥ 2 tokens for it.
Lower is better: 0.0 means every word maps to a single token.

    PCW(lang, tokenizer) = |{w : |tokenize(w)| ≥ 2}| / |words|

Word segmentation follows the same strategy as the fertility metric:
whitespace-based for most scripts, character-level for CJK.

Usage (module):
    from moto.tokenizer_intrinsic_metrics.pcw import compute_pcw

    results = compute_pcw(
        tokenizers={"my-tok": "/path/to/tokenizer"},
        languages=["ita_Latn", "tur_Latn", "arb_Arab"],
    )
    # results["my-tok"]["ita_Latn"] -> float in [0, 1]

Usage (CLI):
    python -m moto.tokenizer_intrinsic_metrics.pcw \\
        tokenizers='["gsaltintas/flexitok-bpe_ita_Latn_8000"]' \\
        languages='["ita_Latn","tur_Latn"]'
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from transformers import PreTrainedTokenizerBase

from moto.tokenizer_intrinsic_metrics.fertility import (
    _FLORES_DATASET,
    _FLORES_SPLIT,
    _load_flores_sentences,
    _load_tokenizer,
    _word_tokenizer_for,
)
from moto.utils.args import Args, load_args

logger = logging.getLogger(__name__)


@dataclass
class PCWArgs(Args):
    tokenizers: List[str] = field(default_factory=list)
    # HF repo IDs or local paths, e.g. ["gsaltintas/flexitok-bpe_ita_Latn_8000"]
    languages: List[str] = field(default_factory=list)
    # FLORES language codes, e.g. ["ita_Latn", "tur_Latn"]
    sample_size: int = 1000
    dataset_name: str = _FLORES_DATASET
    dataset_split: str = _FLORES_SPLIT
    output_csv: Optional[str] = "pcw_results.csv"
    seed: int = 42


def compute_pcw(
    tokenizers: Dict[str, Union[str, PreTrainedTokenizerBase]],
    languages: List[str],
    sample_size: int = 1000,
    dataset_name: str = _FLORES_DATASET,
    dataset_split: str = _FLORES_SPLIT,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute Proportion of Continued Words for each tokenizer across languages.

    Args:
        tokenizers: {name: tokenizer_path_or_object}
        languages: FLORES language codes to evaluate (e.g. ["ita_Latn", "arb_Arab"])
        sample_size: number of sentences to use per language
        dataset_name: HuggingFace dataset (per-language configs, e.g. flores_plus)
        dataset_split: dataset split
        seed: random seed for sampling

    Returns:
        {tokenizer_name: {lang_code: pcw_score}}  — scores in [0, 1]
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

        pcw_scores: Dict[str, float] = {}
        for lang in languages:
            total_words = 0
            split_words = 0
            for words in word_segs[lang]:
                total_words += len(words)
                for word in words:
                    if len(tokenizer.encode(word, add_special_tokens=False)) >= 2:
                        split_words += 1
            score = split_words / total_words if total_words > 0 else 0.0
            pcw_scores[lang] = round(score, 4)

        results[tok_name] = pcw_scores

    return results


def pcw_to_dataframe(results: Dict[str, Dict[str, float]]):
    """Convert compute_pcw output to a pandas DataFrame (tokenizers as rows)."""
    import pandas as pd
    return pd.DataFrame.from_dict(results, orient="index")


def main(args: PCWArgs):
    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")
    if not args.languages:
        raise ValueError("Provide at least one language via languages=[...]")

    tokenizers = {tok: tok for tok in args.tokenizers}
    results = compute_pcw(
        tokenizers=tokenizers,
        languages=args.languages,
        sample_size=args.sample_size,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seed=args.seed,
    )

    df = pcw_to_dataframe(results)
    print(df.to_string())

    if args.output_csv:
        df.to_csv(args.output_csv, index_label="tokenizer")
        logger.info("Saved to %s", args.output_csv)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_args(PCWArgs)
    main(args)
