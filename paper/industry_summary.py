"""Superset-Albert vs. its controls (Llama, Llama-43K, Llama-Albert): per-language summary.

    python experiments/release-evals.py --results /path/to/evals   # writes paper/all_delta_release.py
    PYTHONPATH=. python paper/industry_summary.py                     # -> paper/figures/release/industry/

Per language, the mean accuracy over all of that language's accuracy tasks (release-evals.py's
acc_main: XNLI, XCOPA, XStoryCloze, PAWS-X, XWinograd, Okapi HellaSwag, TokSuite, English
benchmarks), for the models of the paper's `industry` group, with Llama as the reference.

    industry_per_lang.tex / .md / .csv   rows = languages (+ Avg), columns = models, accuracy (%);
                                         best per row in bold
    industry_per_lang_delta.pdf          delta vs. Llama per language (+ Avg), one bar per model,
                                         error bars = standard error of the delta when available
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import all_delta_release as R  # noqa: E402

from moto.log_utils.plot_utils import setup_styles  # noqa: E402
from moto.utils.task_to_lang import RUN_NAME_TO_COLOR  # noqa: E402

KEY = "acc_main_industry"
BASELINE = "Llama"
MODELS = ["Llama-43K", "Llama-Albert", "SS-Albert"]  # display names (release-evals rename_map)
# Latin script first, then the rest by script
LANG_ORDER = ["English", "French", "Spanish", "Portuguese", "Italian", "German", "Dutch", "Danish", "Swedish", "Hungarian",
              "Polish", "Czech", "Indonesian", "Vietnamese", "Turkish", "Greek", "Russian", "Arabic", "Farsi", "Chinese", "Japanese"]
OUT = HERE / "figures" / "release" / "industry"


def load():
    deltas = {r["run_name"]: r for r in getattr(R, KEY)}
    base = dict(getattr(R, f"{KEY}_baseline"))
    errs = {r["run_name"]: r for r in (getattr(R, f"{KEY}_err_paired", None) or getattr(R, f"{KEY}_err", None) or [])}
    models = [m for m in MODELS if m in deltas]
    missing = [m for m in MODELS if m not in deltas]
    if missing:
        print(f"no results yet for {missing}; rerun release-evals.py once their evals finish")
    langs = [l for l in LANG_ORDER if l in base] + sorted(l for l in base if l not in LANG_ORDER and l != "Avg")
    absolute = {BASELINE: {l: base[l] for l in langs + ["Avg"]}}
    for m in models:
        absolute[m] = {l: base[l] + deltas[m][l] for l in langs + ["Avg"] if deltas[m].get(l) is not None}
    return langs, models, absolute, deltas, errs


def write_tables(langs, cols, absolute):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = langs + ["Avg"]
    best = {l: max(absolute[c].get(l, -1) for c in cols) for l in rows}

    def cell(c, l, bold):
        v = absolute[c].get(l)
        if v is None:
            return "--"
        s = f"{v:.1f}"
        return (rf"\textbf{{{s}}}" if bold else s) if bold is not None else s

    tex = [r"\begin{table}[t]", r"  \centering", r"  \small",
           r"  \caption{Accuracy (\%) per language, averaged over each language's tasks, for Superset-Albert and its controls. Best per row in bold.}",
           r"  \label{tab:industry_per_lang}", rf"  \begin{{tabular}}{{l{'r' * len(cols)}}}", r"    \toprule",
           "    \\textbf{Language} & " + " & ".join(rf"\textbf{{{c}}}" for c in cols) + r" \\", r"    \midrule"]
    for l in rows:
        if l == "Avg":
            tex.append(r"    \midrule")
        tex.append(f"    {'\\textit{Average}' if l == 'Avg' else l} & " + " & ".join(cell(c, l, absolute[c].get(l) == best[l]) for c in cols) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "industry_per_lang.tex").write_text("\n".join(tex) + "\n")

    md = ["| Language | " + " | ".join(cols) + " |", "| --- |" + " ---: |" * len(cols)]
    for l in rows:
        md.append(f"| {'**Average**' if l == 'Avg' else l} | " + " | ".join(
            (f"**{absolute[c][l]:.1f}**" if absolute[c].get(l) == best[l] else f"{absolute[c][l]:.1f}") if absolute[c].get(l) is not None else "–"
            for c in cols) + " |")
    (OUT / "industry_per_lang.md").write_text("\n".join(md) + "\n")

    with open(OUT / "industry_per_lang.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["language", *cols])
        for l in rows:
            w.writerow([l, *[round(absolute[c][l], 2) if absolute[c].get(l) is not None else "" for c in cols]])
    print(open(OUT / "industry_per_lang.md").read())


def plot(langs, models, deltas, errs):
    setup_styles()
    xs = langs + ["Avg"]
    n = max(len(models), 1)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(xs) * n / 2 + 3), 3.0))
    for i, m in enumerate(models):
        x = np.arange(len(xs)) + (i - (n - 1) / 2) * width
        y = [deltas[m].get(l) if deltas[m].get(l) is not None else np.nan for l in xs]
        e = [errs.get(m, {}).get(l) for l in xs] if m in errs else None
        e = [v if v is not None else 0 for v in e] if e else None
        ax.bar(x, y, width, yerr=e, capsize=1.5, error_kw={"elinewidth": 0.6}, color=RUN_NAME_TO_COLOR.get(m, "#888780"),
               edgecolor="k", linewidth=0.5, label=m)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(len(xs) - 1.5, color="#BBBBBB", linewidth=0.8, linestyle=":")  # separate the average
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([("Average" if l == "Avg" else l) for l in xs], rotation=45, ha="right")
    ax.set_ylabel(rf"$\Delta$ accuracy vs. {BASELINE} (pp)")
    ax.legend(ncols=n, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "industry_per_lang_delta.pdf", bbox_inches="tight")
    print(f"saved → {OUT / 'industry_per_lang_delta.pdf'}")


if __name__ == "__main__":
    langs, models, absolute, deltas, errs = load()
    write_tables(langs, [BASELINE] + models, absolute)
    plot(langs, models, deltas, errs)
