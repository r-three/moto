"""Delta figures (vs. each group's baseline) for the released models.

    python experiments/release-evals.py --results /path/to/evals   # writes paper/all_delta_release.py
    PYTHONPATH=. python paper/plot_release.py                         # -> paper/figures/release/{by_model,by_lang}/

Same figures as the paper notebook (plot_deltas.plot_all), read from all_delta_release.py.
Tables: paper/make_release_tables.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import all_delta_release as R  # noqa: E402
from plot_deltas import plot_all  # noqa: E402

BENCH = ["acc_main", "english", "toksuite", "toksuite_canonical", "okapi", "xnli", "xcopa", "xstorycloze",
         "paws", "xwinograd", "multiblimp", "code_switching", "flores_bpb"]
PER_LANG = ["multiblimp", "flores_bpb", "acc_main"]  # also one panel per language
# release-evals.py BASELINES (titles only; vocab_size is left out: plot_all splits keys on the last "_")
BASELINES = {"main": "Llama", "routing": "MoTo-Random", "industry": "Llama", "demoto": "Llama", "ablations": "Llama", "gpt4": "mBPE-GPT4", "boundless": "mBPE-Boundless"}
OUT = Path(__file__).parent / "figures" / "release"

results = {f"{b}_{g}": getattr(R, f"{b}_{g}") for b in BENCH for g in BASELINES if hasattr(R, f"{b}_{g}")}
ap = argparse.ArgumentParser()
ap.add_argument("--error-bars", choices=["none", "independent", "paired"], default="independent",
                help="independent: sqrt(se_model^2 + se_baseline^2) from lm-eval's stderr (<key>_err); paired: standard error of the "
                     "per-document differences, needs the LOG_SAMPLES runs and release-evals.py --paired (<key>_err_paired)")
args = ap.parse_args()
_suffix = {"independent": "_err", "paired": "_err_paired"}.get(args.error_bars)
errors = {k: getattr(R, f"{k}{_suffix}") for k in results if getattr(R, f"{k}{_suffix}", None)} if _suffix else None
if errors is not None and not errors:
    print(f"no {_suffix} blocks in all_delta_release.py: run release-evals.py{' --paired' if _suffix == '_err_paired' else ''} first; plotting without error bars")
plot_all(results, baseline_names=BASELINES, out_dir=OUT, mode="by_model", plot_type="bar", errors=errors)
plot_all({k: v for k, v in results.items() if any(k.startswith(b + "_") for b in PER_LANG)},
         baseline_names=BASELINES, out_dir=OUT, mode="by_lang", plot_type="bar", errors=errors)
