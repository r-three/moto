"""
plot_deltas.py
──────────────
Per-model benchmark-aggregated delta plots.

Layout
------
One panel per model (run_name).
Within each panel:
  y-axis  → languages (ordered by script family, sorted by mean Δ within family)
  x-axis  → Δ accuracy vs. baseline
  Each language row gets a horizontal KDE violin coloured by sign of median Δ
  (warm = positive, cool = negative, grey = near-zero).
  Individual benchmark dots are stacked at fixed y-offsets (no jitter).
  A vertical dashed line at x=0 marks the baseline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from moto.utils.task_to_lang import LANG_TO_PRETTY

# ── font sizes ────────────────────────────────────────────────────────────────
TITLE_FONT_SIZE       = 16
TICK_FONT_SIZE        = 11
AXIS_FONT_SIZE        = 11
LEGEND_FONT_SIZE      = 10

# ── panel geometry ────────────────────────────────────────────────────────────
PANEL_WIDTH           = 3.2    # inches per model panel
PANEL_HEIGHT_PER_LANG = 0.42   # inches per language row

# ── violin appearance ─────────────────────────────────────────────────────────
# Violin is coloured by sign of median Δ; neutral grey when |median| < threshold
VIOLIN_POS_COLOR  = "#57b7278a"   # warm orange  → positive median
VIOLIN_NEG_COLOR  = "#e07b39"   # cool blue    → negative median
VIOLIN_NEU_COLOR  = "#aaaaaa"   # grey         → near-zero median
VIOLIN_SIGN_THRESH = 0.3        # |median| below this → neutral colour
VIOLIN_ALPHA      = 0.40
VIOLIN_HALF_HEIGHT = 0.30
KDE_BW            = 0.4

# kept for legend handle; actual per-violin colour is chosen dynamically
DEFAULT_VIOLIN_COLOR = VIOLIN_POS_COLOR

# ── benchmark colour palette ──────────────────────────────────────────────────
_BENCH_PALETTE = [
    "#e07b39", "#3a86ff", "#57cc99", "#e63946", "#9b5de5",
    "#f4a261", "#06d6a0", "#118ab2", "#ef476f", "#ffd166",
    "#8ecae6", "#a8dadc", "#606c38", "#dda15e", "#bc6c25",
]
BENCH_COLORS: dict[str, str] = {}

# ── colour / display-name lookups ─────────────────────────────────────────────
try:
    from moto.log_utils.plot_utils import setup_styles as _setup_styles
    from moto.utils.task_to_lang import (
        LANG_COLORS,
        RUN_NAME_TO_COLOR,
        RUN_NAME_TO_DISPLAY,
    )
except ImportError:
    LANG_COLORS         = {}
    RUN_NAME_TO_COLOR   = {}
    RUN_NAME_TO_DISPLAY = {}
    def _setup_styles(): pass

# ── script-family ordering ────────────────────────────────────────────────────
SCRIPT_GROUPS = [
    ["eng", "ita", "deu", "fra", "spa", "por", "nld", "swe", "dan"],
    ["fas", "ara", "arb"],
    ["rus", "pol", "ces", "ukr", "bul"],
    ["ell"],
    ["cmn", "zho", "jpn", "kor"],
    ["tur", "kaz", "uzb", "aze"],
    ["hin", "ben", "mar"],
    ["tha", "vie", "ind", "msa"],
    ["heb"],
    ["fin", "hun", "est"],
]

META_COLS = frozenset({"run_name", "Avg", "baseline", "baseline_perf", "benchmark"})

SCRIPT_GROUPS_FLAT = [iso for group in SCRIPT_GROUPS for iso in group]

COL_TO_ISO: dict[str, str] = {
    "HellaSwag": "eng", "Hellaswag": "eng", "hellaswag": "eng",
    "PIQA": "eng", "piqa": "eng",
    "ARC-Easy": "eng", "ARC-Challenge": "eng",
    "BoolQ": "eng", "boolq": "eng",
    "CommonsenseQA": "eng", "commonsenseqa": "eng",
    "XWinograd": "eng", "xwinograd": "eng",
    "Winogrande": "eng", "WinoGrande": "eng", "winogrande": "eng",
    "OpenBookQA": "eng", "openbookqa": "eng",
    "SIQA": "eng", "siqa": "eng",
    "English": "eng", "Italian": "ita", "German": "deu",
    "French": "fra", "Spanish": "spa", "Portuguese": "por",
    "Dutch": "nld", "Swedish": "swe", "Danish": "dan",
    "Farsi": "fas", "Persian": "fas", "Arabic": "ara",
    "Russian": "rus", "Polish": "pol", "Czech": "ces",
    "Ukrainian": "ukr", "Bulgarian": "bul",
    "Greek": "ell",
    "Chinese": "cmn", "Japanese": "jpn", "Korean": "kor",
    "Turkish": "tur", "Kazakh": "kaz", "Uzbek": "uzb", "Azerbaijani": "aze",
    "Hindi": "hin", "Bengali": "ben", "Marathi": "mar",
    "Thai": "tha", "Vietnamese": "vie", "Indonesian": "ind", "Malay": "msa",
    "Hebrew": "heb",
    "Finnish": "fin", "Hungarian": "hun", "Estonian": "est",
    **{iso: iso for iso in SCRIPT_GROUPS_FLAT},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _violin_color(values: list[float]) -> str:
    """Pick violin fill colour based on sign of median Δ."""
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return VIOLIN_NEU_COLOR
    med = float(np.median(vals))
    if med > VIOLIN_SIGN_THRESH:
        return VIOLIN_POS_COLOR
    if med < -VIOLIN_SIGN_THRESH:
        return VIOLIN_NEG_COLOR
    return VIOLIN_NEU_COLOR


def _ordered_langs(
    lang_set: set[str],
    delta_map_for_model: dict[str, dict[str, float]] | None = None,
) -> list[str]:
    """
    Script-family grouped, with languages sorted by mean Δ (descending)
    within each family when delta_map_for_model is provided.
    For the global ordering (shared across all models) pass None — falls back
    to the canonical SCRIPT_GROUPS order.
    """
    ordered: list[str] = []
    for group in SCRIPT_GROUPS:
        present = [l for l in group if l in lang_set]
        if delta_map_for_model is not None:
            present.sort(
                key=lambda l: np.mean(list(delta_map_for_model.get(l, {0: 0}).values())),
                reverse=True,
            )
        ordered += present
    ordered += sorted(l for l in lang_set if l not in ordered)
    return ordered


def _lang_y_positions(ordered_langs: list[str]) -> tuple[list[float], float]:
    """
    Assign y-positions with a gap between script families.
    Returns (positions top-to-bottom, total_height).
    """
    lang_to_family: dict[str, int] = {}
    for fi, group in enumerate(SCRIPT_GROUPS):
        for l in group:
            lang_to_family[l] = fi

    GAP = 0.55
    positions: list[float] = []
    y = 0.0
    prev_family = lang_to_family.get(ordered_langs[0], -1) if ordered_langs else -1

    for lang in ordered_langs:
        fam = lang_to_family.get(lang, 999)
        if fam != prev_family and positions:
            y += GAP
        positions.append(y)
        y += 1.0
        prev_family = fam

    total_height = y - 1.0
    flipped = [total_height - p for p in positions]
    return flipped, total_height


def _normalise_row(rec: dict, bench_name: str) -> dict:
    iso_buckets: dict[str, list[float]] = defaultdict(list)
    out: dict = {"run_name": rec["run_name"], "benchmark": bench_name}

    for col, val in rec.items():
        if col in META_COLS:
            continue
        iso = COL_TO_ISO.get(col, col)
        if pd.notna(val):
            iso_buckets[iso].append(float(val))

    tmp = {k[:3].replace("arb", "ara"): v for k, v in LANG_TO_PRETTY.items()}
    for iso, vals in iso_buckets.items():
        name = tmp.get(iso, iso)
        out[name] = float(np.mean(vals))

    return out


def _merge_results(
    results: dict[str, list[dict]],
    group_code: str | None,
) -> pd.DataFrame:
    rows = []
    for key, records in results.items():
        parts = key.rsplit("_", 1)
        gc         = parts[-1] if len(parts) == 2 else ""
        bench_name = parts[0]  if len(parts) == 2 else key
        if group_code is not None and gc != group_code:
            continue
        for rec in records:
            rows.append(_normalise_row(rec, bench_name))
    return pd.DataFrame(rows)


def _build_delta_map(
    df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {model: {lang: {benchmark: delta}}}."""
    lang_cols = [c for c in df.columns if c not in META_COLS]
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for _, row in df.iterrows():
        model = row["run_name"]
        bench = row["benchmark"]
        for lang in lang_cols:
            val = row.get(lang, np.nan)
            if pd.notna(val):
                out[model][lang][bench] = float(val)
    return out


# ── KDE horizontal violin ─────────────────────────────────────────────────────

def _horiz_violin(
    ax: plt.Axes,
    y_center: float,
    values: list[float],
    color: str,
    half_height: float = VIOLIN_HALF_HEIGHT,
    bw: float = KDE_BW,
    alpha: float = VIOLIN_ALPHA,
) -> None:
    values = [v for v in values if not np.isnan(v)]
    if len(values) < 2:
        return
    if len(set(values)) == 1:
        ax.plot([values[0], values[0]],
                [y_center - half_height * 0.7, y_center + half_height * 0.7],
                color=color, linewidth=1.8, alpha=0.75)
        return

    kde    = gaussian_kde(values, bw_method=bw)
    x_grid = np.linspace(min(values), max(values), 300)
    dens   = kde(x_grid)
    dens   = dens / dens.max() * half_height

    ax.fill_between(x_grid, y_center - dens, y_center + dens,
                    color=color, alpha=alpha, linewidth=0)
    ax.plot(x_grid, y_center + dens, color=color, linewidth=0.9, alpha=0.8)
    ax.plot(x_grid, y_center - dens, color=color, linewidth=0.9, alpha=0.8)

    med   = float(np.median(values))
    med_h = kde([med])[0] / kde(x_grid).max() * half_height
    ax.plot([med, med], [y_center - med_h, y_center + med_h],
            color=color, linewidth=2.0, solid_capstyle="round", zorder=4)


# ── single model panel ────────────────────────────────────────────────────────

def _draw_panel(
    ax: plt.Axes,
    model: str,
    lang_bench_map: dict[str, dict[str, float]],
    ordered_langs: list[str],
    benchmarks: list[str],               # global sorted list for stable stacking
    bench_colors: dict[str, str],
    xlim: tuple[float, float],
    show_ylabels: bool = True,
) -> None:
    y_positions, total_height = _lang_y_positions(ordered_langs)

    lang_to_family: dict[str, int] = {}
    for fi, group in enumerate(SCRIPT_GROUPS):
        for l in group:
            lang_to_family[l] = fi

    # fixed vertical offsets for stacked dots — evenly spaced within ±0.30
    n_bench = len(benchmarks)
    if n_bench == 1:
        dot_offsets = {benchmarks[0]: 0.0}
    else:
        offsets = np.linspace(-0.28, 0.28, n_bench)
        dot_offsets = {b: float(o) for b, o in zip(benchmarks, offsets)}

    for lang, y in zip(ordered_langs, y_positions):
        bench_map = lang_bench_map.get(lang, {})
        if not bench_map:
            continue

        values = list(bench_map.values())
        color  = _violin_color(values)
        _horiz_violin(ax, y, values, color)

        # stacked dots at fixed offsets — only for benchmarks present for this lang
        for bench, val in bench_map.items():
            ax.scatter(
                val, y ,#+  dot_offsets.get(bench, 0.0),
                color=bench_colors.get(bench, "#888780"),
                edgecolor="k", linewidth=0.3, s=20, zorder=5,
            )

    # faint row lines
    for y in y_positions:
        ax.axhline(y, color="k", linewidth=0.15, alpha=0.18, zorder=1)

    # script-family separator lines
    prev_fam = lang_to_family.get(ordered_langs[0], -1) if ordered_langs else -1
    for lang, y in zip(ordered_langs[1:], y_positions[1:]):
        fam = lang_to_family.get(lang, 999)
        if fam != prev_fam:
            ax.axhline(y + 0.75, color="k", linewidth=0.6, alpha=0.35,
                       linestyle="-", zorder=2)
        prev_fam = fam

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4, zorder=2)

    ax.set_xlim(*xlim)
    ax.set_ylim(min(y_positions) - 0.55, max(y_positions) + 0.55)
    ax.set_yticks(y_positions)

    if show_ylabels:
        ax.set_yticklabels(ordered_langs, fontsize=TICK_FONT_SIZE)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(direction="out", which="both")
    ax.set_title(RUN_NAME_TO_DISPLAY.get(model, model),
                 fontsize=AXIS_FONT_SIZE, pad=4)
    sns.despine(ax=ax)


# ── main plot function ────────────────────────────────────────────────────────

def plot_benchmark_aggregated(
    results: dict[str, list[dict]],
    group_code: str,
    baseline_name: str,
    out_dir: Path,
    panel_width: float           = PANEL_WIDTH,
    panel_height_per_lang: float = PANEL_HEIGHT_PER_LANG,
    ncols: int | None            = None,
    title: str                   = "",
) -> None:
    global BENCH_COLORS

    df = _merge_results(results, group_code)
    if df.empty:
        print(f"  No data for group_code='{group_code}'")
        return

    lang_cols  = [c for c in df.columns if c not in META_COLS]
    benchmarks = sorted(df["benchmark"].unique())
    models     = [r for r in df["run_name"].unique() if r != "baseline"]

    for j, b in enumerate(benchmarks):
        if b not in BENCH_COLORS:
            BENCH_COLORS[b] = _BENCH_PALETTE[j % len(_BENCH_PALETTE)]

    delta_map = _build_delta_map(df)

    # global language ordering: family-grouped, within each family sorted by
    # mean Δ averaged across all models (most-improved language first)
    def _mean_delta(lang: str) -> float:
        vals = [
            v
            for m in models
            for v in delta_map.get(m, {}).get(lang, {}).values()
        ]
        return float(np.mean(vals)) if vals else 0.0

    all_langs     = set(lang_cols)
    # first get family order, then sort within families by cross-model mean Δ
    family_ordered = _ordered_langs(all_langs, delta_map_for_model=None)
    # rebuild with cross-model mean for sorting
    ordered_langs: list[str] = []
    for group in SCRIPT_GROUPS:
        present = [l for l in group if l in all_langs]
        present.sort(key=_mean_delta, reverse=True)
        ordered_langs += present
    ordered_langs += sorted(l for l in all_langs if l not in ordered_langs)

    all_vals = [
        v
        for model in models
        for bench_map in delta_map.get(model, {}).values()
        for v in bench_map.values()
    ]
    if not all_vals:
        print("  No valid delta values found.")
        return
    lo, hi = min(all_vals), max(all_vals)
    pad    = max(0.12 * (hi - lo), 1.0)
    xlim   = (lo - pad, hi + pad)

    n_models = len(models)
    _ncols   = ncols or n_models
    _nrows   = math.ceil(n_models / _ncols)

    _, total_height = _lang_y_positions(ordered_langs)
    fig_w = panel_width * _ncols
    fig_h = panel_height_per_lang * (total_height + 1) * _nrows + 1.2 * _nrows

    fig, axs = plt.subplots(
        _nrows, _ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )
    axs = np.array(axs).flatten() if not isinstance(axs, np.ndarray) else axs.flatten()

    for i, model in enumerate(models):
        _draw_panel(
            axs[i], model,
            lang_bench_map=delta_map.get(model, {}),
            ordered_langs=ordered_langs,
            benchmarks=benchmarks,
            bench_colors=BENCH_COLORS,
            xlim=xlim,
            show_ylabels=(i % _ncols == 0),
        )

    bottom_start = (_nrows - 1) * _ncols
    for ax in axs[bottom_start: bottom_start + min(_ncols, n_models)]:
        ax.set_xlabel(f"Δ accuracy vs. {baseline_name}", fontsize=AXIS_FONT_SIZE)

    for ax in axs[n_models:]:
        ax.set_visible(False)

    # ── legend ────────────────────────────────────────────────────────────────
    violin_handles = [
        mpatches.Patch(facecolor=VIOLIN_POS_COLOR, alpha=0.6, edgecolor="none",
                       label="KDE (median > 0)"),
        mpatches.Patch(facecolor=VIOLIN_NEG_COLOR, alpha=0.6, edgecolor="none",
                       label="KDE (median < 0)"),
        mpatches.Patch(facecolor=VIOLIN_NEU_COLOR, alpha=0.6, edgecolor="none",
                       label="KDE (median ≈ 0)"),
    ]
    bench_handles = [
        mpatches.Patch(facecolor=BENCH_COLORS[b], edgecolor="k",
                       linewidth=0.5, label=b)
        for b in benchmarks
    ]
    all_handles = violin_handles + bench_handles

    fig.legend(
        all_handles,
        [h.get_label() for h in all_handles],
        ncols=min(8, len(all_handles)),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.suptitle(
        title or rf"$\Delta$ accuracy vs. {baseline_name} — {group_code}",
        fontsize=TITLE_FONT_SIZE,
    )

    out = out_dir / "by_model" / f"{group_code}_benchmark_aggregated.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved → {out.absolute()}")


# ── convenience wrapper ───────────────────────────────────────────────────────

def plot_all(
    results: dict[str, list[dict]],
    baseline_names: dict[str, str],
    out_dir: Path,
    group_codes: list[str] | None = None,
    panel_width: float            = PANEL_WIDTH,
    panel_height_per_lang: float  = PANEL_HEIGHT_PER_LANG,
    ncols: int | None             = None,
) -> None:
    """
    Parameters
    ----------
    results         "{benchmark}_{group_code}" → list of row dicts.
    baseline_names  {group_code: display_name}.
    out_dir         Root output directory; PDFs land in out_dir/by_model/.
    group_codes     Which group_codes to plot. None → infer from result keys.
    """
    _setup_styles()

    if group_codes is None:
        group_codes = sorted({key.rsplit("_", 1)[-1] for key in results})

    for gc in group_codes:
        baseline_name = baseline_names.get(gc, "Llama")
        title = (
            rf"$\Delta$ accuracy vs. {baseline_name} "
            rf"— {gc.replace('_', ' ').title()} (distribution over benchmarks)"
        )
        print(f"\ngroup_code={gc}  baseline={baseline_name}")
        plot_benchmark_aggregated(
            results=results,
            group_code=gc,
            baseline_name=baseline_name,
            out_dir=out_dir,
            panel_width=panel_width,
            panel_height_per_lang=panel_height_per_lang,
            ncols=ncols,
            title=title,
        )