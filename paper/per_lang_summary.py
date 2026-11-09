"""Per-language summary tables and figures for sets of release models, against a reference model.

    python experiments/release-evals.py --results /path/to/evals   # writes paper/all_delta_release.py
    PYTHONPATH=. python paper/make_release_tables.py                  # all release tables, these included
    PYTHONPATH=. python paper/per_lang_summary.py                     # only these: every set in SUMMARIES
    PYTHONPATH=. python paper/per_lang_summary.py sets=[industry]     # one set
    PYTHONPATH=. python paper/per_lang_summary.py sets=[main] rows=2 sharey=false   # languages over two rows
    PYTHONPATH=. python paper/per_lang_summary.py sets=[ablations] by=script        # per writing script
    (options: by (lang|script|both), rows, width, height, rotation, sharey, xtick_size, suffix, error_scale; see PerLangArgs)

Per language, the mean accuracy over all of that language's accuracy tasks (release-evals.py's
acc_main: XNLI, XCOPA, XStoryCloze, PAWS-X, XWinograd, Okapi HellaSwag, TokSuite, English
benchmarks), read from acc_main_all (every release model, deltas against Llama). For each set, in
paper/figures/release/per_lang/<set>/:

    <set>_per_lang.tex / .md / .csv   rows = languages (+ Avg), columns = the reference + the set, accuracy (%);
                                      best per row in bold
    <set>_per_lang_delta.pdf          delta vs. the reference per language (+ Avg), one bar per model,
                                      error bars = standard error of the delta when available
    <set>_per_lang_delta_<k>rows.pdf  with rows=<k>: the languages split over k stacked panels
"""

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import all_delta_release as R  # noqa: E402

from moto.log_utils.plot_utils import setup_styles  # noqa: E402
from moto.utils.args import Args, load_args  # noqa: E402

# bar colours: a few models side by side need clearly different hues, so these figures use the
# Okabe-Ito colour-blind-safe set in order instead of the shared family palette (near-identical
# oranges for MoTo-Random / MoTo-GPT4-Reg., blues for the Llama variants)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]

KEY = "acc_main_all"
BASELINE = "Llama"
# set -> (reference model, compared models; display names as in release-evals' rename_map, caption)
SUMMARIES = {
    "main": ("Llama", ["MoTo", "MoTo-GPT4", "mBPE-GPT4", "mBPE-Boundless", "Llama-BPE-Drop.", "Parity-Aware BPE"],
             "\\moto and the main baselines, against Llama"),
    "industry": ("Llama", ["SS-Albert", "Llama-43K", "Llama-Albert"], "Superset-Albert and its controls, against Llama"),
    "routing": ("MoTo", ["MoTo-Random", "MoTo-GPT4-Reg."], "MoTo with random and regularized routing, against MoTo (oracle routing)"),
    "ablations": ("Llama", ["MoTo", "ByT5", "MoTo-16K", "MoTo-Llama-16K", "MoTo-Script", "mBPE-GPT4 (323K)"], "byte-level and larger-vocabulary ablations, against Llama"),
    "demoto_mbpe": ("MoTo", ["Demoto", "mBPE-GPT4", "mBPE-Boundless"], "MoTo against its single-vocabulary counterpart (Demoto) and multilingual BPE"),
}
# Latin script first, then the rest by script
LANG_ORDER = ["English", "French", "Spanish", "Portuguese", "Italian", "German", "Dutch", "Danish", "Swedish", "Hungarian",
              "Polish", "Czech", "Indonesian", "Vietnamese", "Turkish", "Greek", "Russian", "Arabic", "Farsi", "Chinese", "Japanese"]
ROOT = HERE / "figures" / "release" / "per_lang"


def load(ref, MODELS):
    """Accuracies from acc_main_all (deltas vs. Llama + Llama's row); deltas re-based on `ref`."""
    deltas = {r["run_name"]: r for r in getattr(R, KEY)}
    base = dict(getattr(R, f"{KEY}_baseline"))
    errs = {r["run_name"]: r for r in (getattr(R, f"{KEY}_err_paired", None) or getattr(R, f"{KEY}_err", None) or [])}
    models = [m for m in MODELS if m in deltas]
    missing = [m for m in [ref, *MODELS] if m != BASELINE and m not in deltas]
    if missing:
        print(f"no results yet for {missing}; rerun release-evals.py once their evals finish")
    langs = [l for l in LANG_ORDER if l in base] + sorted(l for l in base if l not in LANG_ORDER and l != "Avg")
    absolute = {BASELINE: {l: base[l] for l in langs + ["Avg"]}}
    for m in {*models, ref} - {BASELINE}:
        if m in deltas:
            absolute[m] = {l: base[l] + deltas[m][l] for l in langs + ["Avg"] if deltas[m].get(l) is not None}
    if ref not in absolute:
        raise SystemExit(f"no results for the reference {ref}")
    if ref != BASELINE:  # deltas against ref; the stored standard errors are of deltas vs. Llama, so none here
        deltas = {m: {l: absolute[m][l] - absolute[ref][l] for l in absolute[m] if l in absolute[ref]} for m in models}
        errs = {}
    return langs, models, absolute, deltas, errs


def write_tables(name, caption, langs, cols, absolute, unit="lang"):
    OUT = ROOT / name
    OUT.mkdir(parents=True, exist_ok=True)
    rows = langs + ["Avg"]
    # best on the printed (1-decimal) values, so ties are all bold
    absolute = {c: {l: round(v, 1) for l, v in absolute[c].items()} for c in cols}
    best = {l: max(absolute[c].get(l, -1) for c in cols) for l in rows}

    def cell(c, l, bold):
        v = absolute[c].get(l)
        if v is None:
            return "--"
        s = f"{v:.1f}"
        return (rf"\textbf{{{s}}}" if bold else s) if bold is not None else s

    tex = [r"\begin{table}[t]", r"  \centering", r"  \small",
           rf"  \caption{{Accuracy (\%) per {UNIT_TEXT[unit]}: {caption}. Best per row in bold.}}",
           rf"  \label{{tab:{name}_per_{unit}}}", rf"  \begin{{tabular}}{{l{'r' * len(cols)}}}", r"    \toprule",
           "    \\textbf{Language} & " + " & ".join(rf"\textbf{{{c}}}" for c in cols) + r" \\", r"    \midrule"]
    for l in rows:
        if l == "Avg":
            tex.append(r"    \midrule")
        label = r"\textit{Average}" if l == "Avg" else l
        tex.append(f"    {label} & " + " & ".join(cell(c, l, absolute[c].get(l) == best[l]) for c in cols) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / f"{name}_per_{unit}.tex").write_text("\n".join(tex) + "\n")

    md = [f"| {'Language' if unit == 'lang' else 'Script'} | " + " | ".join(cols) + " |", "| --- |" + " ---: |" * len(cols)]
    for l in rows:
        md.append(f"| {'**Average**' if l == 'Avg' else l} | " + " | ".join(
            (f"**{absolute[c][l]:.1f}**" if absolute[c].get(l) == best[l] else f"{absolute[c][l]:.1f}") if absolute[c].get(l) is not None else "–"
            for c in cols) + " |")
    (OUT / f"{name}_per_{unit}.md").write_text("\n".join(md) + "\n")

    with open(OUT / f"{name}_per_{unit}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["language" if unit == "lang" else "script", *cols])
        for l in rows:
            w.writerow([l, *[round(absolute[c][l], 2) if absolute[c].get(l) is not None else "" for c in cols]])
    print(open(OUT / f"{name}_per_{unit}.md").read())


def _bars(ax, xs, models, deltas, errs, avg_sep: bool, rotation: int = 45, error_scale: float = 1.0):
    n = max(len(models), 1)
    width = 0.8 / n
    for i, m in enumerate(models):
        x = np.arange(len(xs)) + (i - (n - 1) / 2) * width
        y = [deltas[m].get(l) if deltas[m].get(l) is not None else np.nan for l in xs]
        e = [errs.get(m, {}).get(l) for l in xs] if m in errs else None
        e = [error_scale * v if v is not None else 0 for v in e] if e else None
        ax.bar(x, y, width, yerr=e, capsize=1.5, error_kw={"elinewidth": 0.6}, color=PALETTE[i % len(PALETTE)],
               edgecolor="k", linewidth=0.5, label=m)
    ax.axhline(0, color="black", linewidth=0.8)
    if avg_sep:
        ax.axvline(len(xs) - 1.5, color="#BBBBBB", linewidth=0.8, linestyle=":")  # separate the average
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([("Average" if l == "Avg" else l) for l in xs], rotation=rotation, ha="right" if rotation else "center")
    ax.margins(x=0.01)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


@dataclass
class PerLangArgs(Args):
    """CLI (moto.utils.args.load_args): `python paper/per_lang_summary.py sets=[main] rows=2 sharey=false`."""

    sets: List[str] = field(default_factory=list)  # SUMMARIES keys; empty = all
    rows: int = 1  # split the languages over this many stacked panels (the average stays in the last one)
    width: Optional[float] = None  # inches; default 10 for one row, 13 for more
    height: Optional[float] = None  # inches; default 3.4 for one row, 3.25 per row for more
    rotation: Optional[int] = None  # language-name angle; default 45 for one row, 0 for more
    sharey: bool = True  # one y scale for all rows (false: each row its own, so small deltas stay readable)
    xtick_size: Optional[int] = None  # language-name font size; default the style's
    suffix: Optional[str] = None  # file name suffix; default "" for one row, "_<rows>rows" otherwise
    error_scale: float = 1.0  # error bar = error_scale x standard error (1.96: 95% confidence interval)
    by: str = "lang"  # "lang": one group per language; "script": languages averaged per writing script; "both"


def plot(name, ref, langs, models, deltas, errs, opts: PerLangArgs | None = None, unit="lang"):
    """<name>_per_lang_delta<suffix>.pdf: one panel per row of languages, the average at the end of the last."""
    from matplotlib.ticker import MultipleLocator

    opts = opts or PerLangArgs()
    OUT = ROOT / name
    OUT.mkdir(parents=True, exist_ok=True)
    setup_styles()
    n = max(len(models), 1)
    ylabel = rf"$\Delta$ acc. vs. {ref} (pp)"
    legend_cols = n if n <= 4 else -(-n // 2)
    k = max(1, opts.rows)
    per = -(-len(langs) // k)
    rows = [langs[r * per:(r + 1) * per] for r in range(k)]
    rows = [r for r in rows if r] or [[]]
    rows[-1] = rows[-1] + ["Avg"]
    width = opts.width or (10 if k == 1 else 13)
    height = opts.height or (3.4 if k == 1 else 3.25 * len(rows))
    rotation = opts.rotation if opts.rotation is not None else (45 if k == 1 else 0)

    fig, axes = plt.subplots(len(rows), 1, figsize=(width, height), sharey=opts.sharey, squeeze=False)
    axes = axes[:, 0]
    for r, (ax, xs) in enumerate(zip(axes, rows)):
        _bars(ax, xs, models, deltas, errs, avg_sep=r == len(rows) - 1, rotation=rotation, error_scale=opts.error_scale)
        if len(rows) > 1:
            ax.set_xlim(-0.6, max(len(x) for x in rows) - 0.4)  # same bar width in every row
        ax.set_ylabel(ylabel)
        if not opts.sharey:
            ax.yaxis.set_major_locator(MultipleLocator(5 if np.nanmax(np.abs(ax.get_ylim())) > 8 else 1))
        if opts.xtick_size:
            ax.tick_params(axis="x", labelsize=opts.xtick_size)
    fig.legend(*axes[0].get_legend_handles_labels(), ncols=legend_cols, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    suffix = opts.suffix if opts.suffix is not None else ("" if k == 1 else f"_{k}rows")
    path = OUT / f"{name}_per_{unit}_delta{suffix}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


UNIT_TEXT = {"lang": "language, averaged over each language's tasks", "script": "writing script, the mean of its languages' per-language averages"}
# writing script of each language (FLORES+ script codes); per-script value = mean over its languages
SCRIPTS = {
    "Latin": ["English", "French", "Spanish", "Portuguese", "Italian", "German", "Dutch", "Danish", "Swedish", "Hungarian",
              "Polish", "Czech", "Indonesian", "Vietnamese", "Turkish"],
    "Greek": ["Greek"],
    "Cyrillic": ["Russian"],
    "Arabic": ["Arabic", "Farsi"],
    "Han / Kana": ["Chinese", "Japanese"],
}


def by_script(langs, models, absolute, deltas, errs):
    """Per-language results -> per-script: values and deltas averaged over the script's languages present,
    standard errors combined as sqrt(sum se^2) / n (independent languages). Avg stays the per-language average."""
    groups = {sc: [l for l in ls if l in langs] for sc, ls in SCRIPTS.items()}
    groups = {sc: ls for sc, ls in groups.items() if ls}
    other = [l for l in langs if not any(l in ls for ls in groups.values())]
    if other:
        groups["Other"] = other

    def agg(d):
        out = {sc: sum(d[l] for l in ls) / len(ls) for sc, ls in groups.items() if all(d.get(l) is not None for l in ls)}
        if d.get("Avg") is not None:
            out["Avg"] = d["Avg"]
        return out

    def agg_err(d):
        out = {sc: sum(d[l] ** 2 for l in ls) ** 0.5 / len(ls) for sc, ls in groups.items() if all(d.get(l) is not None for l in ls)}
        if d.get("Avg") is not None:
            out["Avg"] = d["Avg"]
        return out

    names = [f"{sc} ({len(ls)})" for sc, ls in groups.items()]  # label with the number of languages
    rename = dict(zip(groups, names))

    def relabel(d):
        return {rename.get(k, k): v for k, v in d.items()}

    return (names, models, {m: relabel(agg(v)) for m, v in absolute.items()},
            {m: relabel(agg(v)) for m, v in deltas.items()}, {m: relabel(agg_err(v)) for m, v in errs.items()})


def run(names=None, opts: PerLangArgs | None = None):
    """Every set in SUMMARIES (or `names`), from all_delta_release.py; also called by make_release_tables.py."""
    opts = opts or PerLangArgs()
    units = ["lang", "script"] if opts.by == "both" else [opts.by]
    for name in names or list(SUMMARIES):
        ref, members, caption = SUMMARIES[name]
        print(f"=== {name} (vs. {ref})")
        data = load(ref, members)
        for unit in units:
            langs, models, absolute, deltas, errs = data if unit == "lang" else by_script(*data)
            write_tables(name, caption, langs, [ref] + models, absolute, unit=unit)
            plot(name, ref, langs, models, deltas, errs, opts, unit=unit)


if __name__ == "__main__":
    args = load_args(PerLangArgs)
    run(args.sets, args)
