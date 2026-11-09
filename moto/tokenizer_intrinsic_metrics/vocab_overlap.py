"""Vocabulary overlap metrics between tokenizers.

Measures how much vocabulary two tokenizers share.  The primary metric is
pairwise Jaccard similarity:

    jaccard(A, B) = |vocab(A) ∩ vocab(B)| / |vocab(A) ∪ vocab(B)|

Additional per-pair statistics (intersection size, directional coverage) are
available via :func:`compute_vocab_overlap_stats`.

Unlike the corpus-based metrics (fertility, parity, PCW), vocab overlap does
not require any text data — it operates purely on the tokenizer vocabularies.

Usage (module):
    from moto.tokenizer_intrinsic_metrics.vocab_overlap import compute_vocab_overlap

    results = compute_vocab_overlap(
        tokenizers={"tok-a": "/path/a", "tok-b": "/path/b"},
    )
    # results["tok-a"]["tok-b"] -> float in [0, 1]

Usage (CLI):
    python -m moto.tokenizer_intrinsic_metrics.vocab_overlap \\
        tokenizers='["flexitok/bpe-ita_Latn-8000","meta-llama/Llama-3.2-1B"]'
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from moto.utils.args import Args, load_args

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_tokenizer(tokenizer: Union[str, PreTrainedTokenizerBase]) -> PreTrainedTokenizerBase:
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


def _get_vocab(tokenizer: PreTrainedTokenizerBase) -> set:
    return set(tokenizer.get_vocab().keys())


# ── Core metrics ───────────────────────────────────────────────────────────

def compute_vocab_overlap(
    tokenizers: Dict[str, Union[str, PreTrainedTokenizerBase]],
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise Jaccard vocabulary overlap between tokenizers.

    Args:
        tokenizers: {name: tokenizer_path_or_object}

    Returns:
        {tok_a: {tok_b: jaccard_score}}  — symmetric matrix, diagonal = 1.0
    """
    names = list(tokenizers.keys())
    logger.info("Loading %d tokenizers for vocab overlap...", len(names))

    vocabs: Dict[str, set] = {}
    for name, tok_or_path in tokenizers.items():
        tok = _load_tokenizer(tok_or_path)
        vocabs[name] = _get_vocab(tok)
        logger.info("  %s: vocab size = %d", name, len(vocabs[name]))

    results: Dict[str, Dict[str, float]] = {}
    for a in names:
        results[a] = {}
        for b in names:
            if a == b:
                results[a][b] = 1.0
            else:
                intersection = len(vocabs[a] & vocabs[b])
                union = len(vocabs[a] | vocabs[b])
                results[a][b] = round(intersection / union, 4) if union > 0 else 0.0

    return results


def compute_vocab_overlap_stats(
    tokenizers: Dict[str, Union[str, PreTrainedTokenizerBase]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute detailed pairwise vocabulary overlap statistics.

    Returns for each (A, B) pair:
        - jaccard: |A ∩ B| / |A ∪ B|
        - intersection: |A ∩ B|
        - vocab_size_a: |A|
        - vocab_size_b: |B|
        - coverage_a_in_b: |A ∩ B| / |B|  (fraction of B covered by A)
        - coverage_b_in_a: |A ∩ B| / |A|  (fraction of A covered by B)
    """
    names = list(tokenizers.keys())
    logger.info("Loading %d tokenizers for detailed vocab overlap...", len(names))

    vocabs: Dict[str, set] = {}
    for name, tok_or_path in tokenizers.items():
        tok = _load_tokenizer(tok_or_path)
        vocabs[name] = _get_vocab(tok)

    stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for a in names:
        stats[a] = {}
        for b in names:
            va, vb = vocabs[a], vocabs[b]
            inter = len(va & vb)
            union = len(va | vb)
            stats[a][b] = {
                "jaccard": round(inter / union, 4) if union > 0 else 0.0,
                "intersection": inter,
                "vocab_size_a": len(va),
                "vocab_size_b": len(vb),
                "coverage_a_in_b": round(inter / len(vb), 4) if vb else 0.0,
                "coverage_b_in_a": round(inter / len(va), 4) if va else 0.0,
            }

    return stats


# ── DataFrame helpers ──────────────────────────────────────────────────────

def vocab_overlap_to_dataframe(results: Dict[str, Dict[str, float]]):
    """Convert compute_vocab_overlap output to a pandas DataFrame (square matrix)."""
    import pandas as pd
    return pd.DataFrame.from_dict(results, orient="index")


def vocab_overlap_stats_to_dataframe(
    stats: Dict[str, Dict[str, Dict[str, Any]]],
    stat_key: str = "jaccard",
):
    """Extract a single stat from compute_vocab_overlap_stats into a DataFrame."""
    import pandas as pd
    flat = {a: {b: stats[a][b][stat_key] for b in stats[a]} for a in stats}
    return pd.DataFrame.from_dict(flat, orient="index")


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_vocab_overlap(
    results: Dict[str, Dict[str, float]],
    save_path: Path,
    title: str = "Pairwise Vocabulary Overlap (Jaccard)",
) -> None:
    """Save a heatmap of pairwise Jaccard vocabulary overlap."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results.keys())
    matrix = np.array([[results[a][b] for b in names] for a in names])

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.7), max(5, len(names) * 0.6)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=13)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] > 0.6 else "black")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved vocab overlap plot to %s", save_path)


# ── Args & CLI ─────────────────────────────────────────────────────────────

@dataclass
class VocabOverlapArgs(Args):
    tokenizers: List[str] = field(default_factory=list)
    # HF repo IDs or local paths
    output_csv: Optional[str] = "vocab_overlap_results.csv"
    output_plot: Optional[str] = "vocab_overlap.png"


def main(args: VocabOverlapArgs):
    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")

    tokenizers = {tok: tok for tok in args.tokenizers}
    results = compute_vocab_overlap(tokenizers)

    df = vocab_overlap_to_dataframe(results)
    print(df.to_string())

    if args.output_csv:
        df.to_csv(args.output_csv, index_label="tokenizer")
        logger.info("Saved to %s", args.output_csv)

    if args.output_plot:
        plot_vocab_overlap(results, Path(args.output_plot))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_args(VocabOverlapArgs)
    main(args)
