import math
from pathlib import Path
from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from moto.log_utils.plot_utils import setup_styles
from moto.utils.task_to_lang import (
    LANG_COLORS,
    RUN_NAME_TO_COLOR,
    RUN_NAME_TO_DISPLAY,
)

# NORMAL_FONT = 14
# TITLE_FONT = 24
TITLE_FONT_SIZE = 16
TICK_FONT_SIZE = 12
AXIS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
# ── script-family grouping ────────────────────────────────────────────────────
SCRIPT_GROUPS = [
    ["eng", "ita", "deu", "fra", "spa", "por", "nld", "swe", "dan"],  # Latin
    ["fas", "ara", "arb"],  # Arabic
    ["rus", "pol", "ces", "ukr", "bul"],  # Cyrillic
    ["ell"],  # Greek
    ["cmn", "zho", "jpn", "kor"],  # CJK
    ["tur", "kaz", "uzb", "aze"],  # Turkic
    ["hin", "ben", "mar"],  # Devanagari
    ["tha", "vie", "ind", "msa"],  # SE Asian / Malay
    ["heb"],  # Hebrew
    ["fin", "hun", "est"],  # Uralic
]
SCRIPT_GROUP_LABELS = [
    "Latin",
    "Arabic",
    "Cyrillic",
    "Greek",
    "CJK",
    "Turkic",
    "Devanagari",
    "SE Asian",
    "Hebrew",
    "Uralic",
]
PANEL_WIDTH = 1.8  # inches per panel
PANEL_HEIGHT = 2.5  # inches tall


def build_ordered_langs(df: pd.DataFrame) -> list[str]:
    """Return langs present in df columns, ordered by script family."""
    meta_cols = {"run_name", "Avg", "baseline", "baseline_perf"}
    present = [c for c in df.columns if c not in meta_cols]
    ordered = []
    for group in SCRIPT_GROUPS:
        ordered += [l for l in group if l in present]
    # anything not in any group goes at the end
    ordered += [l for l in present if l not in ordered]
    return ordered


def build_lang_groups(ordered_langs: list[str]) -> list[list[str]]:
    """Return only non-empty script groups, preserving order."""
    groups = []
    for group in SCRIPT_GROUPS:
        sub = [l for l in group if l in ordered_langs]
        if sub:
            groups.append(sub)
    tail = [l for l in ordered_langs if not any(l in g for g in SCRIPT_GROUPS)]
    if tail:
        groups.append(tail)
    return groups


# ── shared helpers ────────────────────────────────────────────────────────────
ERR_KW = dict(ecolor="k", elinewidth=0.7, capsize=1.5, capthick=0.7)  # standard error of the delta


def clean_up(ax: plt.Axes) -> None:
    sns.despine(ax=ax)
    ax.tick_params(direction="out", which="both")
    ax.set_xticks([])

    ylim = ax.get_ylim()
    use_floats = abs(ylim[1]) < 1 and abs(ylim[0]) < 1

    if use_floats:
        ticks = list(np.linspace(ylim[0], ylim[1], 4))
        ticks = [round(t, 2) for t in ticks]
    else:
        auto_ticks = [t for t in ax.get_yticks()
                      if ylim[0] <= t <= ylim[1] and float(t) == int(float(t))]
        # only round auto ticks (plus 0): the axis limits carry padding for the bars and error bars,
        # so a tick forced onto the limit (-8 under a -6 bar) sticks out past the data
        ticks = sorted(set(auto_ticks) | {0})
        # keep 0 and the auto ticks first and drop any tick closer than 12% of the axis to one already kept
        min_gap = 0.12 * (ylim[1] - ylim[0])
        kept = []
        for t in sorted(ticks, key=lambda t: (t != 0, t not in auto_ticks)):
            if all(abs(t - k) >= min_gap for k in kept):
                kept.append(t)
        ticks = sorted(kept)
        if len(ticks) < 2:  # nothing but 0 survived: fall back to a spread over the axis
            ticks = [round(t) for t in np.linspace(ylim[0], ylim[1], 3)]

    ax.set_yticks(ticks)
    ax.set_ylim(*ylim)  # set_yticks must not rescale the padded limits
    ax.spines["left"].set_bounds(min(ticks), max(ticks))  # axis line ends at the outermost tick
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

def _err_values(err, df: pd.DataFrame, ordered_langs: list[str]) -> np.ndarray | None:
    """err (same layout as df: run_name + language columns) -> array aligned to df rows x ordered_langs, or None."""
    if err is None:
        return None
    e = err.set_index("run_name").reindex(index=df["run_name"], columns=ordered_langs).values.astype(float)
    return None if np.isnan(e).all() else e


def _shared_ylim(df: pd.DataFrame, ordered_langs: list[str], margin: float = 0.2, err: np.ndarray | None = None):
    vals = df[ordered_langs].values.astype(float)
    lo_v, hi_v = vals, vals
    if err is not None:
        e = np.nan_to_num(err)
        lo_v, hi_v = vals - e, vals + e
    lo, hi = np.nanmin(lo_v), np.nanmax(hi_v)
    rng = (hi - lo) or 1
    return lo - margin * rng, hi + 0.15 * rng

def _axis_ylim(scores, yerr=None, margin: float = 0.2):
    """y-limits of one panel from its own bars (and error bars); always contains 0, the bars' baseline."""
    v = np.asarray(scores, dtype=float)
    e = np.zeros_like(v) if yerr is None else np.nan_to_num(np.asarray(yerr, dtype=float))
    if np.isnan(v).all():
        return -1.0, 1.0
    lo, hi = min(0.0, float(np.nanmin(v - e))), max(0.0, float(np.nanmax(v + e)))
    rng = (hi - lo) or 1.0
    return lo - (margin if lo < 0 else 0.05) * rng, hi + (0.15 if hi > 0 else 0.05) * rng


# ── shared violin helper ──────────────────────────────────────────────────────
def draw_violin(ax, positions, scores, colors, ylim):
    """
    Draw a single violin (all scores pooled) with colored scatter overlay.
    Or, if len(scores) >= 4, draw a proper violin; otherwise just strip plot.
    """
    valid = [(p, s, c) for p, s, c in zip(positions, scores, colors)
             if not np.isnan(s)]
    if not valid:
        return
    pos, vals, cols = zip(*valid)

    if len(vals) >= 4:
        parts = ax.violinplot(
            vals,
            positions=[0],
            widths=0.6,
            showmedians=True,
            showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#cccccc")
            pc.set_alpha(0.5)
            pc.set_edgecolor("k")
            pc.set_linewidth(0.8)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_edgecolor("k")
                parts[key].set_linewidth(0.8)

    # Individual colored points (jittered)
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
    for j, v, c in zip(jitter, vals, cols):
        ax.scatter(j, v, color=c, edgecolor="k", linewidth=0.4,
                   s=30, zorder=3)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

# ── by-model plot ─────────────────────────────────────────────────────────────
def plot_by_model(
    df,
    group_code,
    benchmark,
    baseline_name,
    out_dir,
    panel_width=PANEL_WIDTH,
    panel_height=PANEL_HEIGHT,
    nrows=1,
    ncols=None,
    title: str = "",
    ylabel: str = "",
    legend_ncols: int | None = None,
    file_name: str | None = None,
    plot_type: Literal["bar", "violin"] = "bar",
    err: pd.DataFrame | None = None,
):
    ordered_langs = build_ordered_langs(df)
    models = [r for r in df["run_name"].tolist() if r not in ("baseline",)]
    err_arr = _err_values(err, df, ordered_langs)
    ylim = _shared_ylim(df, ordered_langs, err=err_arr)

    fig, axs = plt.subplots(
        nrows,
        ncols or len(models),
        figsize=(panel_width * (ncols or len(models)), panel_height * nrows),
        constrained_layout=True,
    )
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, (model, ax) in enumerate(zip(models, axs)):
        row = df[df["run_name"] == model]
        scores = [
            row[l].values[0] if not row.empty and l in row.columns else np.nan
            for l in ordered_langs
        ]
        colors = [LANG_COLORS.get(l, "#888780") for l in ordered_langs]
        yerr = None if err_arr is None else np.nan_to_num(err_arr[df.index[df["run_name"] == model][0]]) if (df["run_name"] == model).any() else None
        if plot_type == "bar":
            ax.bar(
            range(len(ordered_langs)),
            scores,
            edgecolor="k",
            linewidth=0.5,
            facecolor=colors,
            width=0.7,
            yerr=yerr,
            error_kw=ERR_KW,
        )
        elif plot_type == "violin":
            draw_violin(ax, range(len(ordered_langs)), scores, colors, ylim)
        ax.set_title(RUN_NAME_TO_DISPLAY.get(model, model), fontsize=AXIS_FONT_SIZE)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_ylim(*ylim)
        if i == 0 and ylabel:
            ax.set_ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
        clean_up(ax)

    legend_handles = [
        mpatches.Patch(
            facecolor=LANG_COLORS.get(l, "#888780"),
            edgecolor="k",
            linewidth=0.5,
            label=l,
        )
        for l in ordered_langs
    ]
    ncols_legend = legend_ncols or min(7, len(ordered_langs))
    n_legend_rows = math.ceil(len(ordered_langs) / ncols_legend)
    legend_y = -0.15 - 0.1 * n_legend_rows

    fig.legend(
        legend_handles,
        ordered_langs,
        ncols=ncols_legend,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.0,
        handletextpad=0.5,
        alignment="center",
    )
    fig.suptitle(
        title,
        fontsize=TITLE_FONT_SIZE,
    )
    # fig.tight_layout(pad=0.5)

    out = out_dir / "by_model" / f"{file_name or f'{benchmark}_{group_code}'}_by_model.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    # fig.savefig(out, pad_inches=0)
    plt.close(fig)
    print(f"  saved → {out.absolute()}")


# ── by-language plot ──────────────────────────────────────────────────────────
def plot_by_lang(
    df,
    group_code,
    benchmark,
    baseline_name,
    out_dir,
    panel_width=PANEL_WIDTH,
    panel_height=PANEL_HEIGHT,
    nrows=1,
    ncols=None,
    title: str = "",
    ylabel: str = "",
    legend_ncols: int | None = None,
    file_name: str | None = None,
    plot_type: Literal["bar", "violin"] = "bar",
    err: pd.DataFrame | None = None,
    sharey: bool = False,  # one y-range for every language panel; default: each panel scales to its own bars
):
    ordered_langs = build_ordered_langs(df)
    models = [r for r in df["run_name"].tolist() if r not in ("baseline",)]
    err_arr = _err_values(err, df, ordered_langs)
    ylim = _shared_ylim(df, ordered_langs, err=err_arr)

    n_langs = len(ordered_langs)
    fig, axs = plt.subplots(
        nrows,
        ncols or n_langs,
        figsize=(panel_width * (ncols or n_langs), panel_height * nrows),
        constrained_layout=True,
        # figsize=figsize or (2.2 * n_langs, 2.5),
    )
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    model_colors = [RUN_NAME_TO_COLOR.get(m, "#888780") for m in models]

    for i, (lang, ax) in enumerate(zip(ordered_langs, axs)):
        scores = [
            df.loc[df["run_name"] == m, lang].values[0]
            if lang in df.columns and not df[df["run_name"] == m].empty
            else np.nan
            for m in models
        ]
        yerr = None
        if err_arr is not None and lang in ordered_langs:
            li = ordered_langs.index(lang)
            yerr = [float(np.nan_to_num(err_arr[df.index[df["run_name"] == m][0], li])) if (df["run_name"] == m).any() else 0.0 for m in models]
        if plot_type == "bar":
            ax.bar(
            range(len(models)),
            scores,
            edgecolor="k",
            linewidth=0.5,
            facecolor=model_colors,
            width=0.7,
            yerr=yerr,
            error_kw=ERR_KW,
        )
        elif plot_type == "violin":
            draw_violin(ax, range(len(models)), scores, model_colors, ylim if sharey else _axis_ylim(scores))
        ax.set_title(lang, fontsize=AXIS_FONT_SIZE)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_ylim(*(ylim if sharey else _axis_ylim(scores, yerr)))
        if i == 0 and ylabel:
            ax.set_ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
        clean_up(ax)

    legend_handles = [
        mpatches.Patch(
            facecolor=RUN_NAME_TO_COLOR.get(m, "#888780"),
            edgecolor="k",
            linewidth=0.5,
            label=RUN_NAME_TO_DISPLAY.get(m, m),
        )
        for m in models
    ]
    ncols_legend = legend_ncols or min(7, len(models))
    n_legend_rows = math.ceil(len(models) / ncols_legend)
    legend_y = -0.15 - 0.1 * n_legend_rows

    fig.legend(
        legend_handles,
        [RUN_NAME_TO_DISPLAY.get(m, m) for m in models],
        ncols=ncols_legend,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.0,
        handletextpad=0.5,
        alignment="center",
    )
    fig.suptitle(
        title,
        fontsize=TITLE_FONT_SIZE,
    )
    # fig.tight_layout(pad=0.5)

    out = out_dir / "by_lang" / f"{file_name or f'{benchmark}_{group_code}'}_by_lang.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    # fig.savefig(out, pad_inches=0)
    plt.close(fig)
    print(f"  saved → {out.absolute()}")


# ── table rendering (markdown / latex) ───────────────────────────────────────
def build_delta_table(df: pd.DataFrame, ordered_langs: list[str]) -> pd.DataFrame:
    """Rows = models (display names), columns = languages (+ Avg if present)."""
    models = [r for r in df["run_name"].tolist() if r not in ("baseline",)]
    rows = []
    for model in models:
        row = df[df["run_name"] == model]
        entry = {"Model": RUN_NAME_TO_DISPLAY.get(model, model)}
        for lang in ordered_langs:
            entry[lang] = (
                row[lang].values[0] if not row.empty and lang in row.columns else np.nan
            )
        if "Avg" in df.columns:
            entry["Avg"] = row["Avg"].values[0] if not row.empty else np.nan
        rows.append(entry)
    return pd.DataFrame(rows).set_index("Model")


def _cell(v, e=None, latex: bool = False) -> str:
    """'+1.23' or, with a standard error, '+1.23 ± 0.45' (LaTeX: $+1.23 \\pm 0.45$)."""
    if pd.isna(v):
        return "–"
    if e is None or pd.isna(e):
        return f"{v:+.2f}"
    return rf"${v:+.2f} \pm {e:.2f}$" if latex else f"{v:+.2f} ± {e:.2f}"


def _to_markdown_table(table: pd.DataFrame, err_table: pd.DataFrame | None = None) -> str:
    headers = ["Model", *table.columns]
    errs = err_table.values if err_table is not None else [[None] * table.shape[1]] * len(table)
    cell_rows = [
        [str(idx)] + [_cell(v, e) for v, e in zip(row, erow)]
        for idx, row, erow in zip(table.index, table.values, errs)
    ]
    widths = [
        max(len(h), *(len(r[i]) for r in cell_rows)) if cell_rows else len(h)
        for i, h in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for r in cell_rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(lines)


def write_table(
    df: pd.DataFrame,
    ordered_langs: list[str],
    group_code: str,
    benchmark: str,
    out_dir: Path,
    title: str = "",
    plot_type: Literal["markdown", "latex"] = "markdown",
    file_name: str | None = None,
    err: pd.DataFrame | None = None,  # standard error of each delta (same layout as df): cells become "+1.23 ± 0.45"
) -> None:
    table = build_delta_table(df, ordered_langs)
    err_table = build_delta_table(err, ordered_langs).reindex(index=table.index, columns=table.columns) if err is not None else None
    if err_table is not None and err_table.isna().all().all():
        err_table = None  # no standard errors for this benchmark (e.g. FLORES)
    name = file_name or f"{benchmark}_{group_code}"

    if plot_type == "markdown":
        content = (f"### {title}\n\n" if title else "") + _to_markdown_table(table, err_table) + "\n"
        ext = "md"
    elif err_table is not None:
        cells = pd.DataFrame(
            [[_cell(v, e, latex=True) for v, e in zip(row, erow)] for row, erow in zip(table.values, err_table.values)],
            index=table.index, columns=table.columns,
        )
        content = cells.to_latex(na_rep="–", caption=title or None, escape=False)
        ext = "tex"
    else:
        content = table.to_latex(float_format="%+.2f", na_rep="–", caption=title or None)
        ext = "tex"

    out = out_dir / "tables" / f"{name}.{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    print(content)
    print(f"  saved → {out.absolute()}")


# ── main loop ─────────────────────────────────────────────────────────────────
def plot_all(
    results: dict,  # {"benchmark_group": [records], ...}
    baseline_names: dict,  # {"main": "Llama", "routing": "MoTo-GPT4", ...}
    out_dir: Path,
    mode: Literal["by_model", "by_lang", "both"] = "both",
    nrows: int | None = 1,
    ncols: int | None = None,
    panel_width: float = PANEL_WIDTH,
    panel_height: float = PANEL_HEIGHT,
    plot_type: Literal["bar", "violin", "markdown", "latex"] = "bar",
    title: str = "",
    ylabel: str = "",
    legend_ncols: int | None = None,
    file_name: str | None = None,
    errors: dict | None = None,  # {"benchmark_group": [records]}: standard error of each delta, same layout as results
) -> None:
    setup_styles()
    for key, records in results.items():
        nrows_, ncols_ = nrows, ncols
        if not records:
            print(f"Skipping {key} (no records)")
            continue
        # key format: "{benchmark}_{group_code}"
        *benchmark_parts, group_code = key.split("_")
        print("Bencmark parts, group code", benchmark_parts, group_code)
        benchmark = "_".join(benchmark_parts)
        baseline_name = baseline_names.get(group_code, "Llama")

        df = pd.DataFrame(records)
        err_df = pd.DataFrame(errors[key]) if errors and errors.get(key) else None
        if err_df is not None and "run_name" not in err_df.columns and "index" in err_df.columns:
            err_df = err_df.rename(columns={"index": "run_name"})  # all_delta_release.py written before release-evals named the index
        print(f"\n{key}  ({len(df)} models, baseline={baseline_name})")

        if plot_type in ("markdown", "latex"):
            ordered_langs = build_ordered_langs(df)
            table_title = title or f"{benchmark.title()} vs. {baseline_name}"
            write_table(
                df,
                ordered_langs,
                group_code,
                benchmark,
                out_dir,
                title=table_title,
                plot_type=plot_type,
                file_name=file_name,
                err=err_df,
            )
            continue

        if mode in ("by_model", "both"):
            panel_title = title or rf"{benchmark.title()} Per-Model $\Delta$ Accuracy (%) vs. {baseline_name}"
            if not title and "flores" in benchmark.lower():
                panel_title = rf"{' '.join(benchmark_parts).title()} ($\downarrow$) FLORES-200 vs. {baseline_name}"
            plot_by_model(
                df,
                group_code,
                benchmark,
                baseline_name,
                out_dir,
                nrows=nrows,
                ncols=ncols,
                title=panel_title,
                ylabel=ylabel,
                legend_ncols=legend_ncols,
                file_name=file_name,
                panel_width=panel_width,
                panel_height=panel_height,
                plot_type=plot_type,
                err=err_df,
            )
        if mode in ("by_lang", "both"):
            panel_title = title or rf"{benchmark.title()} Per-Language $\Delta$ Accuracy (%) vs. {baseline_name}"
            if not title and "flores" in benchmark.lower():
                nrows_, ncols_ = 3, 7
                panel_title = rf"{' '.join(benchmark_parts).title()} ($\downarrow$) FLORES-200 vs. {baseline_name}"
            plot_by_lang(
                df,
                group_code,
                benchmark,
                baseline_name,
                out_dir,
                nrows=nrows_,
                ncols=ncols_,
                title=panel_title,
                ylabel=ylabel,
                legend_ncols=legend_ncols,
                file_name=file_name,
                panel_width=panel_width,
                panel_height=panel_height,
                plot_type=plot_type,
                err=err_df,
            )
