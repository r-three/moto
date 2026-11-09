"""Dot plot of the TokEval relative parity (the figure version of paper/tables/tokeval/parity_against_*.md).

    PYTHONPATH=. python paper/plot_tokeval_parity.py                        # against Llama's English -> paper/figures/tokeval/
    PYTHONPATH=. python paper/plot_tokeval_parity.py against=moto show_std=true

One row per language, one marker per tokenizer, x = tokens per sentence relative to the denominator tokenizer's English
count (lower is better). Colors are the project's (moto.utils.task_to_lang); adjacent hues do not pass the CVD check
(green/teal, red/green), so each tokenizer also has its own marker shape. Persian is excluded by default.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_tokeval_tables import FULL, NAMES, RELATIVE, relative_parity  # noqa: E402

from moto.utils.args import Args, load_args  # noqa: E402
from moto.utils.task_to_lang import RUN_NAME_TO_COLOR  # noqa: E402

COLOR_KEY = {"llama32": "Llama", "joint_mbpe_gpt4": "mBPE-GPT4", "joint_mbpe_boundless": "mBPE-Boundless",
             "parity_aware_bpe": "Parity-Aware BPE", "moto_1b": "MoTo", "moto_16k_v2": "MoTo-16K", "moto_v1_oracle": "MoTo-GPT4"}
MARKER = {"llama32": "o", "joint_mbpe_gpt4": "s", "joint_mbpe_boundless": "P", "parity_aware_bpe": "^",
          "moto_1b": "D", "moto_16k_v2": "v", "moto_v1_oracle": "X"}
INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"


@dataclass
class ParityPlotArgs(Args):
    input_dir: str = "results/tokeval"
    output_dir: str = "paper/figures/tokeval"
    against: str = "llama"                                   # llama | moto | parity_aware
    settings: List[str] = field(default_factory=lambda: ["llama32", "joint_mbpe_gpt4", "parity_aware_bpe", "moto_1b", "moto_16k_v2"])
    exclude_languages: List[str] = field(default_factory=lambda: ["pes_Arab"])
    show_std: bool = False                                   # thin +-1 std (over sentences) bars: spread, not uncertainty
    width: float = 6.5
    height: float = 6.2


def main(args: ParityPlotArgs):
    for f in font_manager.findSystemFonts():
        if "Atkinson" in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Atkinson Hyperlegible", "DejaVu Sans"], "font.size": 9, "pdf.fonttype": 42})
    title, den = {n: (t, d) for n, t, d in RELATIVE}[args.against]
    counts = pd.read_csv(Path(args.input_dir) / "token_counts.csv")
    langs = [l for l in FULL if l in set(counts.language) and l not in args.exclude_languages]
    mean, std = relative_parity(counts, args.settings, den, langs)
    order = mean[args.settings[0]].sort_values(ascending=False).index.tolist()       # worst-for-reference language first
    order = [l for l in order if l != "eng_Latn"] + ["eng_Latn"]                      # English last, apart from the rest

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    ys = {l: i for i, l in enumerate(order[::-1])}                                    # first item at the top
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.axhline(0.5, color=GRID, lw=0.8, zorder=0)                                    # separates English
    for l, y in ys.items():
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
        row = mean.loc[l, args.settings]
        ax.plot([row.min(), row.max()], [y, y], color="#c9ced4", lw=1.2, zorder=1, solid_capstyle="round")
    for st in args.settings:
        c = RUN_NAME_TO_COLOR[COLOR_KEY[st]]
        y = [ys[l] for l in order]
        x = [mean.loc[l, st] for l in order]
        if args.show_std:
            ax.errorbar(x, y, xerr=[std.loc[l, st] for l in order], fmt="none", ecolor=c, elinewidth=0.7, alpha=0.5, zorder=2)
        ax.scatter(x, y, s=34, marker=MARKER[st], color=c, edgecolor=INK, linewidth=0.5, zorder=3, label=NAMES[st])
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([FULL[l] for l in ys], color=INK)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(f"Tokens per sentence relative to {title}'s English count (lower is better)", color=MUTED)
    ax.tick_params(axis="both", length=0, colors=MUTED)
    ax.tick_params(axis="y", colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(args.settings), frameon=False, handletextpad=0.2,
                    columnspacing=1.0, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"parity_against_{args.against}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {stem}.pdf/.png ({len(langs)} languages: {' '.join(order)})")


if __name__ == "__main__":
    main(load_args(ParityPlotArgs))
