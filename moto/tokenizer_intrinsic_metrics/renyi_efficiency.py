""""""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import tokenization_scorer
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
class RenyiArgs(Args):
    tokenizers: List[str] = field(default_factory=list)
    # HF repo IDs or local paths, e.g. ["gsaltintas/flexitok-bpe_ita_Latn_8000"]
    languages: List[str] = field(default_factory=list)
    # FLORES language codes, e.g. ["ita_Latn", "tur_Latn"]
    ref_lang: str = "eng_Latn"
    sample_size: int = 1000
    dataset_name: str = _FLORES_DATASET
    dataset_split: str = _FLORES_SPLIT
    output_csv: Optional[str] = "renyi_results.csv"
    seed: int = 42
    power: float = 2.5  # Renyi entropy order (alpha)


def _tokenize_to_strings(
    tokenizer: PreTrainedTokenizerBase, texts: List[str]
) -> List[str]:
    """Encode each sentence and return space-joined token strings.

    tokenization_scorer.score expects an iterable of strings where each string
    is a whitespace-separated sequence of tokens (one sentence per element).
    """
    result = []
    for text in texts:
        ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        if not isinstance(tokens, list):
            tokens = [tokens] if isinstance(tokens, str) else []
        result.append(" ".join(tokens))
    return result


def compute_renyi(
    tokenizers: Dict[str, Union[str, PreTrainedTokenizerBase]],
    languages: List[str],
    ref_lang: str = "eng_Latn",
    sample_size: int = 1000,
    dataset_name: str = _FLORES_DATASET,
    dataset_split: str = _FLORES_SPLIT,
    seed: int = 42,
    power: float = 2.5,
) -> Dict[str, Dict[str, float]]:
    """Compute Renyi efficiency for each tokenizer across languages.

    Args:
        tokenizers: {name: tokenizer_path_or_object}
        languages: FLORES language codes to evaluate (e.g. ["ita_Latn", "arb_Arab"])
        ref_lang: reference language included in the evaluation
        sample_size: number of sentences per language
        dataset_name: HuggingFace dataset identifier
        dataset_split: dataset split
        seed: random seed for sampling
        power: Renyi entropy order (alpha); default 2.5 following common practice

    Returns:
        {tokenizer_name: {lang_code: renyi_efficiency}}
    """
    all_langs = list({ref_lang} | set(languages))
    lang_texts = _load_flores_sentences(all_langs, dataset_name, dataset_split, sample_size, seed)

    results: Dict[str, Dict[str, float]] = {}

    for tok_name, tok_or_path in tokenizers.items():
        logger.info("Processing tokenizer: %s", tok_name)
        tokenizer = _load_tokenizer(tok_or_path)

        renyi_scores: Dict[str, float] = {}
        for lang in all_langs:
            tokenized = _tokenize_to_strings(tokenizer, lang_texts[lang])
            score = tokenization_scorer.score(iter(tokenized), metric="renyi", power=power)
            renyi_scores[lang] = round(score, 4)
            logger.info("  %s -> %.4f", lang, renyi_scores[lang])

        results[tok_name] = renyi_scores

    return results


    

def main(args: RenyiArgs):
    import pandas as pd

    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")
    if not args.languages:
        raise ValueError("Provide at least one language via languages=[...]")

    tokenizers: Dict[str, Union[str, PreTrainedTokenizerBase]] = {tok: tok for tok in args.tokenizers}
    results = compute_renyi(
        tokenizers=tokenizers,
        languages=args.languages,
        ref_lang=args.ref_lang,
        sample_size=args.sample_size,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seed=args.seed,
        power=args.power,
    )

    df = pd.DataFrame.from_dict(results, orient="index")
    print(df.to_string())

    if args.output_csv:
        df.to_csv(args.output_csv, index_label="tokenizer")
        logger.info("Saved to %s", args.output_csv)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_args(RenyiArgs)
    main(args)
