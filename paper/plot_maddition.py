"""Multilingual-addition figures (paper: mAddition), with the shared colours of task_to_lang.

    PYTHONPATH=. python paper/plot_maddition.py      # -> paper/figures/maddition_v2/{by_model,by_lang}/

Same data and panels as the create_tables_Figures.ipynb cell (per-language accuracy delta vs. ByT5,
from paper/maddition/all_delta_multilingual_addition.py).
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "maddition"))

import all_delta_multilingual_addition as M  # noqa: E402
from plot_deltas import plot_all  # noqa: E402

LANG_PANELS = [
    ["Dutch", "German", "Swedish", "Italian", "Hungarian", "Chinese", "Japanese"],
    ["English", "French", "Spanish", "Arabic", "Farsi", "Greek", "Russian", "Turkish", "Polish", "Czech", "Portuguese", "Danish"],
]
OUT = HERE / "figures" / "maddition_v2"  # updated colours; figures/maddition keeps the earlier set


def slice_langs(records, langs):
    """Keep run_name + the panel's languages; Avg recomputed over the panel."""
    out = []
    for row in records:
        row = {("Farsi" if k == "Persian" else k): v for k, v in row.items()}
        entry = {"run_name": row["run_name"], **{l: row[l] for l in langs if l in row}}
        present = [l for l in langs if l in entry]
        entry["Avg"] = round(sum(entry[l] for l in present) / len(present), 2) if present else 0.0
        out.append(entry)
    return out


results = {f"mAddition_main{i}": slice_langs(M.accuracy_per_lang_main, langs) for i, langs in enumerate(LANG_PANELS, 1)}
baselines = {f"main{i}": M.baselines["main"] for i in range(1, len(LANG_PANELS) + 1)}
base = M.baselines["main"]
for mode, what in (("by_model", "Per-Model"), ("by_lang", "Per-Language")):  # title: plot_all would print "Maddition"
    plot_all(results, baseline_names=baselines, out_dir=OUT, mode=mode, panel_width=2.4,
             title=rf"mAddition {what} $\Delta$ Accuracy (%) vs. {base}")
