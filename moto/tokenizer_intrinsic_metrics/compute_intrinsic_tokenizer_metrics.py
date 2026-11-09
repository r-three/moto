"""Unified script for computing intrinsic tokenizer metrics.

Runs any combination of fertility, parity, PCW, vocab_overlap, and renyi across a set
of tokenizers and languages.  Results are written to per-metric CSV files and an
optional combined CSV; bar-chart and heatmap PNGs are saved alongside.

Metrics
-------
fertility      total subword tokens / total words  (lower = better)
parity         mean(|tok(lang)| / |tok(ref)|) per sentence pair  (1.0 = equal to ref)
pcw            fraction of words split into ≥ 2 subword tokens  (lower = better)
vocab_overlap  pairwise Jaccard similarity of tokenizer vocabularies  (no corpus needed)
renyi          Renyi efficiency of the tokenization  (higher = better)

Usage
-----
python -m moto.tokenizer_intrinsic_metrics.compute_intrinsic_tokenizer_metrics \
    tokenizers='["flexitok/bpe-ita_Latn-8000","meta-llama/Llama-3.2-1B"]' \
    languages='["ita_Latn","tur_Latn","arb_Arab","cmn_Hani","rus_Cyrl"]' \
    analyses='["fertility","parity","pcw","vocab_overlap"]' \
    output_dir=./results

langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
tokenizer_str="meta-llama/Llama-3.2-1B,"
tokenizer_str+="flexitok/bpe_boundless_AllL_128000,flexitok/bpe_AllL_128000,"
tokenizer_str+="flexitok/bpe_script_Germ_16000,flexitok/bpe_script_Germ_32000,flexitok/bpe_script_Roma_16000,flexitok/bpe_script_Roma_32000,flexitok/bpe_script_CmJp_16000,flexitok/bpe_script_SEAS_16000,flexitok/bpe_script_Slav_16000,flexitok/bpe_script_Arab_16000,"
lang_str="arb_Arab,ces_Latn,cmn_Hani,dan_Latn,deu_Latn,ell_Grek,fas_Arab,fra_Latn,eng_Latn,hun_Latn,ind_Latn,ita_Latn,jpn_Jpan,nld_Latn,pol_Latn,por_Latn,rus_Cyrl,spa_Latn,swe_Latn,tur_Latn,vie_Latn"
for lang in ${langs[@]}; do
for vs in 4000 8000 16000 32000; do
    tokenizer_str+="flexitok/bpe_ltr_${lang}_${vs}_v2,"
    tokenizer_str+="flexitok/bpe_${lang}_${vs},"
done; done
python -m moto.tokenizer_intrinsic_metrics.compute_intrinsic_tokenizer_metrics \
    tokenizers=[${tokenizer_str%,}] \
    languages=[$lang_str] \
    analyses=["fertility","parity","pcw","vocab_overlap","renyi"] \
    output_dir=./results/v2/paper_final \
    dataset_name=/path/to/data/flexitok ref_lang=fw_edu

## Compute it on fw2hq
langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
tokenizer_str="meta-llama/Llama-3.2-1B,"
tokenizer_str+="flexitok/bpe_boundless_AllL_128000,flexitok/bpe_AllL_128000,"
tokenizer_str+="flexitok/bpe_script_Germ_16000,flexitok/bpe_script_Germ_32000,flexitok/bpe_script_Roma_16000,flexitok/bpe_script_Roma_32000,flexitok/bpe_script_CmJp_16000,flexitok/bpe_script_SEAS_16000,flexitok/bpe_script_Slav_16000,flexitok/bpe_script_Arab_16000,"
lang_str="arb_Arab,ces_Latn,cmn_Hani,dan_Latn,deu_Latn,ell_Grek,fas_Arab,fra_Latn,eng_Latn,hun_Latn,ind_Latn,ita_Latn,jpn_Jpan,nld_Latn,pol_Latn,por_Latn,rus_Cyrl,spa_Latn,swe_Latn,tur_Latn,vie_Latn"
for lang in ${langs[@]}; do
    tokenizer_str+="flexitok/bpe_${lang}_8000,"
for vs in 8000 16000; do
    tokenizer_str+="flexitok/bpe_ltr_${lang}_${vs}_v2,"
done; done

lang_str="arb_Arab,ces_Latn,cmn_Hani,dan_Latn,deu_Latn,ell_Grek,fas_Arab,fra_Latn,fw_edu,hun_Latn,ind_Latn,ita_Latn,jpn_Jpan,nld_Latn,pol_Latn,por_Latn,rus_Cyrl,spa_Latn,swe_Latn,tur_Latn,vie_Latn"
for metric in fertility pcw vocab_overlap renyi; do
    sbatch --time 480 --cpus-per-task 4 --mem 128G --job-name=${metric} --wrap="cd $HOME/tokenizers2/ && python -m moto.tokenizer_intrinsic_metrics.compute_intrinsic_tokenizer_metrics \
        tokenizers=[${tokenizer_str%,}] \
        languages=[$lang_str] \
        analyses=[$metric] \
        output_dir=./results/v2/paper_final_fw2hq \
        dataset_name=/path/to/data/flexitok ref_lang=fw_edu"
done

#### with v3
langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
tokenizer_str="meta-llama/Llama-3.2-1B,google/byt5-small,"
tokenizer_str+="flexitok/bpe_boundless_AllL_128000,flexitok/bpe_AllL_128000,flexitok/bpe_gpt4_AllL_128000,"
tokenizer_str+="flexitok/bpe_script_Germ_16000,flexitok/bpe_script_Germ_32000,flexitok/bpe_script_Roma_16000,flexitok/bpe_script_Roma_32000,flexitok/bpe_script_CmJp_16000,flexitok/bpe_script_SEAS_16000,flexitok/bpe_script_Slav_16000,flexitok/bpe_script_Arab_16000,flexitok/bpe_ltr_monolithic_moto"
lang_str="arb_Arab,ces_Latn,cmn_Hani,dan_Latn,deu_Latn,ell_Grek,fas_Arab,fra_Latn,eng_Latn,hun_Latn,ind_Latn,ita_Latn,jpn_Jpan,nld_Latn,pol_Latn,por_Latn,rus_Cyrl,spa_Latn,swe_Latn,tur_Latn,vie_Latn"
for lang in ${langs[@]}; do
    tokenizer_str+="flexitok/bpe_ltr_${lang}_8000_v3,"
    tokenizer_str+="flexitok/bpe_ltr_${lang}_8000_v2,"
    tokenizer_str+="flexitok/bpe_ltr_${lang}_16000_v2,"
    tokenizer_str+="flexitok/bpe_${lang}_8000,"
done;
tokenizer_str+="flexitok/bpe_ltr_fw_edu_32000_v2,"
# for metric in fertility parity gini pcw vocab_overlap renyi; do
for metric in gini; do
    sbatch --time 480 --cpus-per-task 4 --mem 128G --job-name=${metric} --wrap="cd $HOME/tokenizers2/ && python -m moto.tokenizer_intrinsic_metrics.compute_intrinsic_tokenizer_metrics \
    tokenizers=[${tokenizer_str%,}] \
    languages=[$lang_str] \
    analyses=[$metric] \
    output_dir=./results/v2/paper_final "
done
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from moto.log_utils.plot_utils import setup_styles
from moto.tokenizer_intrinsic_metrics.fertility import (
    _FLORES_DATASET,
    _FLORES_SPLIT,
    compute_fertility,
)
from moto.tokenizer_intrinsic_metrics.gini import compute_gini
from moto.tokenizer_intrinsic_metrics.parity import compute_parity
from moto.tokenizer_intrinsic_metrics.pcw import compute_pcw
from moto.tokenizer_intrinsic_metrics.renyi_efficiency import compute_renyi
from moto.tokenizer_intrinsic_metrics.vocab_overlap import (
    compute_vocab_overlap,
    plot_vocab_overlap,
)
from moto.utils.args import Args, load_args

logger = logging.getLogger(__name__)

ALL_ANALYSES = ["fertility", "parity", "pcw", "vocab_overlap", "renyi", "gini"]

@dataclass
class IntrinsicMetricsArgs(Args):
    tokenizers: List[str] = field(default_factory=list)
    # HF repo IDs or local paths; name used as display label
    languages: List[str] = field(default_factory=list)
    # FLORES-200 language codes, e.g. ["ita_Latn", "tur_Latn", "arb_Arab"]
    analyses: List[str] = field(default_factory=list)
    # Subset of ["fertility","parity","pcw","vocab_overlap"]; empty = run all
    ref_lang: str = "eng_Latn"
    # Reference language for parity (must be in languages or added automatically)
    sample_size: int = 1000
    dataset_name: str = _FLORES_DATASET
    dataset_split: str = _FLORES_SPLIT
    output_dir: str = "."
    seed: int = 42
    plot_type: str = "bar"
    # "bar" or "heatmap" — heatmap scales better with many tokenizers
    power: float = 2.5
    skip_diff_lang: bool = False
    non_skip_patterns: List[str] = field(default_factory=lambda: ["AllL", "script", "llama"])  # Substrings in tokenizer name that override skip_diff_lang
    # Whether to skip parity comparisons for langx with tokenizer for langy
    # Renyi entropy order (alpha); only used when "renyi" is in analyses


# ── Plotting ───────────────────────────────────────────────────────────────

def _plot_metric(
    scores: Dict[str, Dict[str, float]],
    metric_name: str,
    ylabel: str,
    title: str,
    save_path: Path,
    sort_langs: bool = True,
    reference_line: Optional[float] = None,
    plot_type: str = "bar",  # "bar" or "heatmap"
) -> None:
    """Plot metric scores as a bar chart or heatmap.

    bar:     grouped bars, languages on x-axis, one bar group per tokenizer.
    heatmap: tokenizers on y-axis, languages on x-axis, value encoded by colour.
             Cells are annotated with the numeric value.  Useful when there are
             many tokenizers where a bar chart becomes hard to read.
    """
    tokenizers = list(scores.keys())
    common_langs = sorted(
        set.intersection(*(set(scores[t].keys()) for t in tokenizers))
    ) if sort_langs else list(set.intersection(*(set(scores[t].keys()) for t in tokenizers)))

    data = np.array([[scores[t][lang] for lang in common_langs] for t in tokenizers])

    n_toks = len(tokenizers)
    n_langs = len(common_langs)

    if plot_type == "heatmap":
        fig, ax = plt.subplots(figsize=(max(8, n_langs * 0.6), max(3, n_toks * 0.5)))
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=ylabel)

        ax.set_xticks(range(n_langs))
        ax.set_yticks(range(n_toks))
        ax.set_xticklabels(common_langs, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(tokenizers, fontsize=9)
        ax.set_title(title, fontsize=13)

        vrange = data.max() - data.min() if data.max() != data.min() else 1.0
        for i in range(n_toks):
            for j in range(n_langs):
                brightness = (data[i, j] - data.min()) / vrange
                color = "white" if brightness > 0.6 else "black"
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    else:  # bar
        group_width = 0.8
        bar_width = group_width / n_toks
        x = np.arange(n_langs)

        fig, ax = plt.subplots(figsize=(max(10, n_langs * 0.7), 5))
        for i, tok in enumerate(tokenizers):
            offset = i * bar_width - group_width / 2 + bar_width / 2
            ax.bar(x + offset, data[i], bar_width, label=tok)

        if reference_line is not None:
            ax.axhline(reference_line, color="black", linewidth=1, linestyle="--", alpha=0.5)

        ax.set_xlabel("Language", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(common_langs, rotation=45, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.set_title(title, fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved plot to %s", save_path)


def plot_fertility(
    results: Dict[str, Dict[str, float]],
    save_path: Path,
    plot_type: str = "bar",
) -> None:
    _plot_metric(
        results,
        metric_name="fertility",
        ylabel="Subword Fertility",
        title="Subword Fertility by Tokenizer and Language",
        save_path=save_path,
        plot_type=plot_type,
    )


def plot_parity(
    results: Dict[str, Dict[str, float]],
    save_path: Path,
    ref_lang: str = "eng_Latn",
    plot_type: str = "bar",
) -> None:
    _plot_metric(
        results,
        metric_name="parity",
        ylabel=f"Parity (relative to {ref_lang})",
        title=f"Parity Scores by Tokenizer and Language\n(reference: {ref_lang})",
        save_path=save_path,
        reference_line=1.0,
        plot_type=plot_type,
    )


def plot_pcw(
    results: Dict[str, Dict[str, float]],
    save_path: Path,
    plot_type: str = "bar",
) -> None:
    _plot_metric(
        results,
        metric_name="pcw",
        ylabel="Proportion of Continued Words",
        title="Proportion of Continued Words (PCW) by Tokenizer and Language",
        save_path=save_path,
        plot_type=plot_type,
    )


def plot_renyi(
    results: Dict[str, Dict[str, float]],
    save_path: Path,
    plot_type: str = "bar",
) -> None:
    _plot_metric(
        results,
        metric_name="renyi",
        ylabel="Renyi Efficiency",
        title="Renyi Efficiency by Tokenizer and Language",
        save_path=save_path,
        plot_type=plot_type,
    )


# ── CSV helpers ────────────────────────────────────────────────────────────

def _to_df(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(results, orient="index")


def save_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> pd.DataFrame:
    """Write per-metric CSVs and a combined wide CSV; return the combined DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for metric, results in all_results.items():
        df = _to_df(results)
        csv_path = output_dir / f"{metric}_results.csv"
        df.to_csv(csv_path, index_label="tokenizer")
        logger.info("Saved %s to %s", metric, csv_path)
        # rename columns to (metric, lang) for combined frame
        df.columns = pd.MultiIndex.from_tuples([(metric, col) for col in df.columns])
        dfs[metric] = df

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs.values(), axis=1)
    combined.index.name = "tokenizer"
    combined_path = output_dir / "intrinsic_metrics.csv"
    combined.to_csv(combined_path)
    logger.info("Saved combined results to %s", combined_path)
    return combined


# ── Main ───────────────────────────────────────────────────────────────────

def main(args: IntrinsicMetricsArgs):
    setup_styles()
    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")

    analyses = args.analyses if args.analyses else ALL_ANALYSES
    unknown = set(analyses) - set(ALL_ANALYSES)
    if unknown:
        raise ValueError(f"Unknown analyses: {unknown}. Valid options: {ALL_ANALYSES}")

    corpus_analyses = {"fertility", "parity", "pcw", "renyi"}
    if set(analyses) & corpus_analyses and not args.languages:
        raise ValueError(
            f"Provide at least one language via languages=[...] "
            f"for corpus-based analyses: {corpus_analyses & set(analyses)}"
        )
    
    tokenizers = {}
    for tok_name in args.tokenizers:
        try:
            tok = AutoTokenizer.from_pretrained(tok_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer '{tok_name}': {e}")
            continue
        tokenizers[tok_name] = tok

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Running analyses: %s", analyses)
    dataset_kwargs = dict(
        sample_size=args.sample_size,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seed=args.seed,
    )

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    if "fertility" in analyses:
        logger.info("Computing fertility...")
        all_results["fertility"] = compute_fertility(
            tokenizers=tokenizers,
            languages=args.languages,
            **dataset_kwargs,
            skip_diff_lang=args.skip_diff_lang,
            non_skip_patterns=args.non_skip_patterns,
        )
        plot_fertility(all_results["fertility"], output_dir / "fertility.png",
                       plot_type=args.plot_type)

    if "parity" in analyses:
        # Make sure ref_lang is included in the eval set
        parity_langs = list(
            {args.ref_lang} | set(args.languages)
        )
        logger.info("Computing parity (ref=%s)...", args.ref_lang)
        all_results["parity"] = compute_parity(
            tokenizers=tokenizers,
            languages=parity_langs,
            ref_lang=args.ref_lang,
            **dataset_kwargs,
            skip_diff_lang=args.skip_diff_lang,
            non_skip_patterns=args.non_skip_patterns,
        )
        plot_parity(all_results["parity"], output_dir / "parity.png",
                    ref_lang=args.ref_lang, plot_type=args.plot_type)

    if "pcw" in analyses:
        logger.info("Computing PCW...")
        all_results["pcw"] = compute_pcw(
            tokenizers=tokenizers,
            languages=args.languages,
            **dataset_kwargs,
        )
        plot_pcw(all_results["pcw"], output_dir / "pcw.png", plot_type=args.plot_type)

    if "vocab_overlap" in analyses:
        logger.info("Computing vocab overlap...")
        all_results["vocab_overlap"] = compute_vocab_overlap(tokenizers)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_vocab_overlap(all_results["vocab_overlap"], output_dir / "vocab_overlap.png")

    if "renyi" in analyses:
        logger.info("Computing Renyi efficiency (power=%.1f)...", args.power)
        all_results["renyi"] = compute_renyi(
            tokenizers=tokenizers,
            languages=args.languages,
            ref_lang=args.ref_lang,
            power=args.power,
            **dataset_kwargs,
        )
        plot_renyi(all_results["renyi"], output_dir / "renyi.png", plot_type=args.plot_type)

    if "gini" in analyses:
        logger.info("Computing per-language Gini coefficients...")
        all_results["gini"] = compute_gini(
            tokenizers=tokenizers,
            languages=args.languages,
            **dataset_kwargs,
        )

    combined = save_results(all_results, output_dir)
    print(combined.to_string())
    logging.info("Outputs saved at %s", output_dir)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = load_args(IntrinsicMetricsArgs)
    main(args)
