"""Which requested tasks have no result yet for one model (eval_release.sh `missing`).

    python -m moto.hf.release.eval.missing --dir /path/to/evals/moto-1b --tasks paper+toksuite

Prints the missing task names, comma-separated (empty line when complete). A task counts as done
when some non --limit results_*.json under --dir has it; a group (flores_plus_moto) when every
member task does. release-evals.py merges all results files of a model, so rerunning only the
missing tasks completes the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# no lm_eval / transformers import here: this runs on login nodes and from eval_bos_grid.py
HERE = Path(__file__).parent
TASKS_DIR = HERE / "tasks"


class _AnyTagLoader(yaml.SafeLoader):
    pass


_AnyTagLoader.add_multi_constructor("!", lambda loader, suffix, node: None)


def done_tasks(model_dir: Path, allow_limit: bool = False) -> set[str]:
    done: set[str] = set()
    for f in model_dir.glob("**/results_*.json"):
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if doc.get("config", {}).get("limit") is not None and not allow_limit:
            continue
        done |= set(doc.get("results", {}))
    return done


def bundled_groups() -> dict[str, list[str]]:
    """Groups defined in eval/tasks (flores_plus_moto) -> member tasks, read from their YAML.

    Read directly instead of through lm_eval's TaskManager, whose internals differ between
    lm-eval 0.4.x and the r-three fork. Every other paper task is a plain task.
    """
    groups = {}
    for path in TASKS_DIR.rglob("*.yaml"):
        cfg = yaml.load(path.read_text(), Loader=_AnyTagLoader) or {}
        if isinstance(cfg, dict) and cfg.get("group") and isinstance(cfg.get("task"), list):
            groups[cfg["group"]] = [t if isinstance(t, str) else t.get("task") for t in cfg["task"]]
    return groups


def members(names: list[str]) -> dict[str, list[str]]:
    """name -> the task names its results appear under (a group expands to its tasks)."""
    groups = bundled_groups()
    return {name: groups.get(name, [name]) for name in names}


def missing(model_dir: Path, spec: str, allow_limit: bool = False) -> list[str]:
    from .task_sets import expand

    done = done_tasks(model_dir, allow_limit)
    return [n for n, subs in members(expand(spec)).items() if not all(s in done for s in subs)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="eval_release.sh output dir of one model ($OUT/<model>[-<TAG>])")
    ap.add_argument("--tasks", default="paper+toksuite")
    ap.add_argument("--allow-limit", action="store_true")
    a = ap.parse_args()
    print(",".join(missing(Path(a.dir), a.tasks, a.allow_limit)))


if __name__ == "__main__":
    main()
