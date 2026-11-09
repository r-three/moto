"""Named task sets for `--tasks` / TASKS=..., defined in paper_tasks.yaml.

    paper                 the paper's tasks (key `tasks`)
    toksuite              the toksuite_* tasks
    multiblimp_moto       lm-eval's MultiBLiMP tasks for the 17 MoTo training languages it covers
    paper+toksuite        sets joined with "+" (or ","), duplicates dropped, order kept

Anything that is not a set name is passed through as a task name, so sets and single tasks mix:
`--tasks paper,multiblimp_moto,xnli_tr`. No lm_eval import: used on login nodes too.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

HERE = Path(__file__).parent


def sets() -> dict[str, list[str]]:
    cfg = yaml.safe_load((HERE / "paper_tasks.yaml").read_text())
    return {"paper" if k == "tasks" else k: list(v) for k, v in cfg.items() if isinstance(v, list)}


# lm-eval groups whose subtasks are named <group>_<variant>. Asking for a group and one of its subtasks
# (paper+toksuite+code_switching) makes lm-eval's TaskManager raise "Duplicate task", and the group
# already runs and reports the subtask, so the subtask is dropped when its group is in the request.
TOKSUITE_GROUPS = ("toksuite_chinese", "toksuite_english", "toksuite_farsi", "toksuite_italian",
                   "toksuite_math", "toksuite_stem", "toksuite_turkish")


def expand(spec: str) -> list[str]:
    """'paper+toksuite' / 'multiblimp_moto,xnli_tr' -> task names."""
    named = sets()
    out: list[str] = []
    for part in (p.strip() for p in re.split(r"[+,]", spec)):
        for task in named.get(part, [part] if part else []):
            if task not in out:
                out.append(task)
    groups = [g for g in TOKSUITE_GROUPS if g in out]
    return [t for t in out if not any(t.startswith(g + "_") for g in groups)]


def has_set(spec: str) -> bool:
    named = sets()
    return any(p.strip() in named for p in re.split(r"[+,]", spec))
