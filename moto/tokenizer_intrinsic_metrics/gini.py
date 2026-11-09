"""Gini coefficient of token-length distribution per language.

For each (tokenizer, language) pair, computes the Gini coefficient of the
distribution of token counts across sentences:

    gini(lang, tokenizer) = Σᵢ Σⱼ |tᵢ - tⱼ| / (2 * n² * μ)

where tᵢ is the number of tokens for sentence i and μ is the mean.

A score of 0 means all sentences produce the same token count (perfect
equality); higher values indicate more unequal tokenization within the
language.

Usage (module):
    from moto.tokenizer_intrinsic_metrics.gini import compute_gini

    results = compute_gini(
        tokenizers={"my-tok": "/path/to/tokenizer"},
        languages=["ita_Latn", "tur_Latn", "arb_Arab"],
    )
    # results["my-tok"]["ita_Latn"] -> float

Usage (CLI):
    python -m moto.tokenizer_intrinsic_metrics.gini \\
        tokenizers='["gsaltintas/flexitok-bpe_ita_Latn_8000"]' \\
        languages='["ita_Latn","tur_Latn","arb_Arab"]'
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from transformers import PreTrainedTokenizerBase

from moto.tokenizer_intrinsic_metrics.fertility import (
    _FLORES_DATASET,
    _FLORES_SPLIT,
    _load_flores_sentences,
    _load_tokenizer,
)
from moto.utils.args import Args, load_args

logger = logging.getLogger(__name__)


@dataclass
class GiniArgs(Args):
    tokenizers: List[str] = field(default_factory=list)
    # HF repo IDs or local paths, e.g. ["gsaltintas/flexitok-bpe_ita_Latn_8000"]
    languages: List[str] = field(default_factory=list)
    # FLORES language codes, e.g. ["ita_Latn", "tur_Latn"]
    sample_size: int = 1000
    dataset_name: str = _FLORES_DATASET
    dataset_split: str = _FLORES_SPLIT
    output_csv: Optional[str] = "gini_results.csv"
    seed: int = 42


def _gini(token_counts: List[int]) -> float:
    """Gini coefficient of a list of non-negative integers."""
    n = len(token_counts)
    if n == 0:
        return float("nan")
    mu = np.mean(token_counts)
    if mu == 0:
        return 0.0
    arr = np.array(token_counts, dtype=float)
    # Efficient O(n log n) formula: sort, then use cumulative sum identity
    arr.sort()
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) - (n + 1) * arr.sum()) / (n * arr.sum()))


def compute_gini(
    tokenizers: Dict[str, Union[str, PreTrainedTokenizerBase]],
    languages: List[str],
    sample_size: int = 1000,
    dataset_name: str = _FLORES_DATASET,
    dataset_split: str = _FLORES_SPLIT,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute per-language Gini coefficient of token-length distributions.

    Args:
        tokenizers: {name: tokenizer_path_or_object}
        languages: FLORES language codes to evaluate
        sample_size: number of sentences to use per language
        dataset_name: HuggingFace dataset (per-language configs)
        dataset_split: dataset split
        seed: random seed for sampling

    Returns:
        {tokenizer_name: {lang_code: gini_score}}
    """
    lang_texts = _load_flores_sentences(languages, dataset_name, dataset_split, sample_size, seed)

    results: Dict[str, Dict[str, float]] = {}

    for tok_name, tok_or_path in tokenizers.items():
        logger.info("Processing tokenizer: %s", tok_name)
        tokenizer = _load_tokenizer(tok_or_path)

        gini_scores: Dict[str, float] = {}
        for lang in languages:
            token_counts = [
                len(tokenizer.encode(text, add_special_tokens=False))
                for text in lang_texts[lang]
                if text.strip()
            ]
            score = round(_gini(token_counts), 4)
            gini_scores[lang] = score
            logger.debug("  %s gini: %.4f", lang, score)

        results[tok_name] = gini_scores

    return results


def gini_to_dataframe(results: Dict[str, Dict[str, float]]):
    """Convert compute_gini output to a pandas DataFrame (tokenizers as rows)."""
    import pandas as pd
    return pd.DataFrame.from_dict(results, orient="index")


def main(args: GiniArgs):
    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")
    if not args.languages:
        raise ValueError("Provide at least one language via languages=[...]")

    tokenizers = {tok: tok for tok in args.tokenizers}
    results = compute_gini(
        tokenizers=tokenizers,
        languages=args.languages,
        sample_size=args.sample_size,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seed=args.seed,
    )

    df = gini_to_dataframe(results)
    print(df.to_string())

    if args.output_csv:
        df.to_csv(args.output_csv, index_label="tokenizer")
        logger.info("Saved to %s", args.output_csv)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_args(GiniArgs)
    main(args)
