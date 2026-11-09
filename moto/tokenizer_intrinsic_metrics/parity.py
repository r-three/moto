"""Parity metric for tokenizers.

Parity measures how equitably a tokenizer treats different languages relative
to a reference language (typically English). For each language it is the mean
ratio of token counts on parallel text:

    parity(lang, ref) = mean(|tokenize(text_lang)| / |tokenize(text_ref)|)

A score of 1.0 means equal tokenization efficiency to the reference language.
Scores > 1.0 mean the language requires more tokens (under-served by the vocab).

By default the denominator |tokenize(text_ref)| uses the *same* tokenizer being
evaluated. Passing `ref_tokenizer` switches the denominator to a fixed reference
tokenizer (e.g. an English-specific tokenizer) applied to the reference text, so
scores become:

    parity(lang, ref) = mean(|tokenizeA(text_lang)| / |tokenize_ref(text_ref)|)

Usage (module):
    from moto.tokenizer_intrinsic_metrics.parity import compute_parity

    results = compute_parity(
        tokenizers={"my-tok": "/path/to/tokenizer"},
        languages=["ita_Latn", "tur_Latn", "arb_Arab"],
    )
    # results["my-tok"]["ita_Latn"] -> float

Usage (CLI):
    python -m moto.tokenizer_intrinsic_metrics.parity \
        tokenizers='["gsaltintas/flexitok-bpe_ita_Latn_8000"]' \
        languages='["ita_Latn","tur_Latn"]'
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union, overload

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
class ParityArgs(Args):
    tokenizers: List[str] = field(default_factory=list)
    # HF repo IDs or local paths, e.g. ["gsaltintas/flexitok-bpe_ita_Latn_8000"]
    languages: List[str] = field(default_factory=list)
    # FLORES language codes, e.g. ["ita_Latn", "tur_Latn"]
    ref_lang: str = "eng_Latn"
    sample_size: int = 1000
    dataset_name: str = _FLORES_DATASET
    dataset_split: str = _FLORES_SPLIT
    output_csv: Optional[str] = "parity_results.csv"
    seed: int = 42
    ref_tokenizer: Optional[str] = None
    # HF repo ID or local path of a fixed reference tokenizer (e.g. an
    # English-specific tokenizer) to tokenize ref_lang text with, instead of
    # using each evaluated tokenizer on its own ref_lang text.
    # output_mode: Literal["append", "overwrite"] = "overwrite"
    output_mode: Optional[str] = "overwrite"
    return_dispersion: bool = False



def compute_parity(
    tokenizers: Dict[str, Union[str, PreTrainedTokenizerBase]],
    languages: List[str],
    ref_lang: str = "eng_Latn",
    sample_size: int = 1000,
    dataset_name: str = _FLORES_DATASET,
    dataset_split: str = _FLORES_SPLIT,
    seed: int = 42,
    skip_diff_lang: bool = False,
    non_skip_patterns: List[str] = ["AllL", "script", "llama"],
    ref_tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
    return_dispersion: bool = False,
) -> Union[Dict[str, Dict[str, float]], Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]]:
    """Compute parity scores for each tokenizer across languages.

    Args:
        tokenizers: {name: tokenizer_path_or_object}
        languages: FLORES language codes to evaluate (e.g. ["ita_Latn", "arb_Arab"])
        ref_lang: reference language code (default: "eng_Latn")
        sample_size: number of parallel sentence pairs to use
        dataset_name: HuggingFace dataset (per-language configs, e.g. flores_plus)
        dataset_split: dataset split
        seed: random seed for sampling
        skip_diff_lang: Whether to skip parity comparisons for languages not in the tokenizer name
        non_skip_patterns: Substrings in tokenizer name that override skip_diff_lang
        ref_tokenizer: optional fixed tokenizer (path or object) to tokenize ref_lang
            text with, instead of using each evaluated tokenizer on its own ref_lang
            text. Use this to define parity relative to e.g. a dedicated English
            tokenizer's token counts rather than tokenizer A's own English counts.
        return_dispersion: if True, also return the per-language standard deviation
            of the per-sentence-pair ratios (dispersion around the mean parity score).

    Returns:
        {tokenizer_name: {lang_code: parity_score}}, or if return_dispersion is True,
        a tuple (means, stds) of two such dicts.
        The reference language entry is always 1.0 (unless ref_tokenizer is set and
        differs from the evaluated tokenizer, in which case it reflects the ratio
        between the evaluated tokenizer and the reference tokenizer on ref_lang text).
    """
    all_langs = list({ref_lang} | set(languages))
    lang_texts = _load_flores_sentences(all_langs, dataset_name, dataset_split, sample_size, seed)

    ref_texts = lang_texts[ref_lang]

    fixed_ref_lengths: Optional[List[int]] = None
    if ref_tokenizer is not None:
        loaded_ref_tokenizer = _load_tokenizer(ref_tokenizer)
        fixed_ref_lengths = [
            len(loaded_ref_tokenizer.encode(t, add_special_tokens=False)) for t in ref_texts
        ]

    results: Dict[str, Dict[str, float]] = {}
    dispersions: Dict[str, Dict[str, float]] = {}

    for tok_name, tok_or_path in tokenizers.items():
        logger.info("Processing tokenizer: %s", tok_name)
        tokenizer = _load_tokenizer(tok_or_path)
        
        ## small hack
        if "parity_aware" in tok_or_path:
            import tokenizers
            import transformers
            from tokenizers import decoders, pre_tokenizers
            from tokenizers.pre_tokenizers import ByteLevel, Whitespace
            logger.info(
                "Overriding pretokenizer"
            )
            pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), ByteLevel(use_regex=False)])
            tokenizer.pre_tokenizer = pre_tokenizer
            # self.hf_tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizer



        if fixed_ref_lengths is not None:
            ref_lengths = fixed_ref_lengths
        else:
            ref_lengths = [
                len(tokenizer.encode(t, add_special_tokens=False)) for t in ref_texts
            ]

        parity_scores: Dict[str, float] = {}
        parity_stds: Dict[str, float] = {}
        if fixed_ref_lengths is None:
            parity_scores[ref_lang] = 1.0
            parity_stds[ref_lang] = 0.0
        else:
            own_ref_lengths = [
                len(tokenizer.encode(t, add_special_tokens=False)) for t in ref_texts
            ]
            ratios = [n / r for n, r in zip(own_ref_lengths, ref_lengths) if r > 0]
            parity_scores[ref_lang] = round(float(np.mean(ratios)), 4) if ratios else 0.0
            parity_stds[ref_lang] = round(float(np.std(ratios)), 4) if ratios else 0.0
        for lang in languages:
            if lang == ref_lang:
                continue
            if skip_diff_lang and lang not in tok_name:
                if any(pat in tok_name for pat in non_skip_patterns):
                    # logger.info("Keeping %s for %s due to non-skip pattern match", tok_name, lang)
                    pass
                else:
                    # logger.info("Skipping %s for %s (not in tokenizer name)", tok_name, lang)
                    continue
            lang_lengths = [
                len(tokenizer.encode(t, add_special_tokens=False)) for t in lang_texts[lang]
            ]
            ratios = [n / r for n, r in zip(lang_lengths, ref_lengths) if r > 0]
            parity_scores[lang] = round(float(np.mean(ratios)), 4) if ratios else 0.0
            parity_stds[lang] = round(float(np.std(ratios)), 4) if ratios else 0.0

        results[tok_name] = parity_scores
        dispersions[tok_name] = parity_stds

    if return_dispersion:
        return results, dispersions
    return results


def parity_to_dataframe(
    results: Dict[str, Dict[str, float]],
    dispersions: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Convert compute_parity output to a pandas DataFrame (tokenizers as rows).

    If `dispersions` (the per-language std dev returned by
    `compute_parity(..., return_dispersion=True)`) is provided, each language
    column is paired with a `{lang}_std` column.
    """
    import pandas as pd

    df = pd.DataFrame.from_dict(results, orient="index")
    if dispersions is None:
        return df

    std_df = pd.DataFrame.from_dict(dispersions, orient="index")
    std_df = std_df.add_suffix("_std")
    interleaved_cols = [c for pair in zip(df.columns, std_df.columns) for c in pair]
    return pd.concat([df, std_df], axis=1)[interleaved_cols]


def main(args: ParityArgs):
    import pandas as pd

    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")
    if not args.languages:
        raise ValueError("Provide at least one language via languages=[...]")

    tokenizers = {tok: tok for tok in args.tokenizers}
    results, dispersions = compute_parity(
        tokenizers=tokenizers,
        languages=args.languages,
        ref_lang=args.ref_lang,
        sample_size=args.sample_size,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seed=args.seed,
        ref_tokenizer=args.ref_tokenizer,
        return_dispersion=args.return_dispersion,
    )

    df = parity_to_dataframe(results, dispersions)
    print(df.to_string())

    if args.output_csv:
        if args.output_mode == "append" and pd.io.common.file_exists(args.output_csv):
            df.to_csv(args.output_csv, mode="a", header=False, index_label="tokenizer")
            logger.info("Appended to %s", args.output_csv)
        else:
            df.to_csv(args.output_csv, index_label="tokenizer")
            logger.info("Saved to %s", args.output_csv)
        logger.info("Saved to %s", args.output_csv)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_args(ParityArgs)
    main(args)
