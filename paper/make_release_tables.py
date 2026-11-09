"""LaTeX + Markdown delta tables (vs. each group's baseline) for the released models.

    python experiments/release-evals.py --results /path/to/evals   # writes paper/all_delta_release.py
    PYTHONPATH=. python paper/make_release_tables.py                  # -> paper/figures/release/tables/
                                                                      #  + paper/figures/release/per_lang/

Same tables as the paper notebook (plot_deltas.plot_all with plot_type latex/markdown), read
from all_delta_release.py instead of all_delta.py. Deltas are computed in release-evals.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import all_delta_release as R  # noqa: E402
from plot_deltas import plot_all  # noqa: E402

BENCH = ["acc_main", "english", "toksuite", "toksuite_canonical", "okapi", "xnli", "xcopa", "xstorycloze",
         "paws", "xwinograd", "multiblimp", "code_switching", "flores_bpb"]
# release-evals.py BASELINES (titles only; vocab_size is left out: plot_all splits keys on the last "_")
BASELINES = {"main": "Llama", "routing": "MoTo-Random", "industry": "Llama", "demoto": "Llama", "ablations": "Llama", "gpt4": "mBPE-GPT4", "boundless": "mBPE-Boundless"}
OUT = Path(__file__).parent / "figures" / "release"

results = {f"{b}_{g}": getattr(R, f"{b}_{g}") for b in BENCH for g in BASELINES if hasattr(R, f"{b}_{g}")}
ap = argparse.ArgumentParser()
ap.add_argument("--error-bars", choices=["none", "independent", "paired"], default="independent",
                help='cells as "+1.23 ± 0.45": independent = sqrt(se_model^2 + se_baseline^2) from lm-eval\'s stderr (<key>_err); '
                     "paired = per-document paired standard error (<key>_err_paired; needs LOG_SAMPLES runs and release-evals.py --paired)")
ap.add_argument("--bench", nargs="+", choices=BENCH, default=None, help="only these benchmarks (e.g. --bench multiblimp)")
ap.add_argument("--groups", nargs="+", choices=list(BASELINES), default=None, help="only these baseline groups (e.g. --groups boundless)")
ap.add_argument("--no-per-lang", action="store_true", help="skip the per-language summaries (paper/per_lang_summary.py)")
args = ap.parse_args()
if args.bench or args.groups:
    results = {k: v for k, v in results.items() if (not args.bench or k.rsplit("_", 1)[0] in args.bench) and (not args.groups or k.rsplit("_", 1)[1] in args.groups)}
_suffix = {"independent": "_err", "paired": "_err_paired"}.get(args.error_bars)
errors = {k: getattr(R, f"{k}{_suffix}") for k in results if getattr(R, f"{k}{_suffix}", None)} if _suffix else None
if errors is not None and not errors:
    print(f"no {_suffix} blocks in all_delta_release.py: run release-evals.py{' --paired' if _suffix == '_err_paired' else ''} first; tables without ±")
for kind in ("latex", "markdown"):
    plot_all(results, baseline_names=BASELINES, out_dir=OUT, plot_type=kind, errors=errors)

# per-language summary tables + figures (paper/per_lang_summary.py SUMMARIES)
if not args.no_per_lang and not (args.bench or args.groups):
    from per_lang_summary import run as per_lang_summary

    per_lang_summary()
