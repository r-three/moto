"""lm_eval's CLI with the `moto` model registered and the bundled tasks on the include path.

    python -m moto.hf.release.eval --model moto --model_args pretrained=r-three-moto/moto-1b \
        --tasks paper --batch_size 16 --output_path /path/to/evals

`--tasks paper` expands to the paper's task list (paper_tasks.yaml); `paper+toksuite` adds the
toksuite_* tasks. FLORES+ replaces the internal flores200 task. Every other lm_eval argument
works as usual, and a user --include_path is kept alongside the bundled tasks.

Works with stock lm-eval 0.4.9 (argparse CLI) and with the r-three fork
(lm-evaluation-harness@multilingual-tiny-benchmarks, `lm-eval run` subcommand CLI): the task
list is expanded in argv and the bundled task directory is added in lm_eval's TaskManager,
which both versions build from --include_path.
"""

from __future__ import annotations

import functools
import inspect
import sys
from pathlib import Path

import yaml

from . import (
    moto_lm,  # noqa: F401 - registers model "moto" (and patches transformers for lm-eval 0.4.x)
)
from .task_sets import expand as expand_spec
from .task_sets import has_set

HERE = Path(__file__).parent
TASKS_DIR = HERE / "tasks"
TASK_SETS = ("paper", "paper+toksuite")  # kept for older callers; any set in paper_tasks.yaml works


def paper_tasks(include_toksuite: bool = False) -> list[str]:
    return expand_spec("paper+toksuite" if include_toksuite else "paper")


def expand_tasks(argv: list[str]) -> list[str]:
    """Named task sets in --tasks (paper, toksuite, multiblimp_moto, joined with + or ,) -> task names."""
    out = list(argv)
    for i, arg in enumerate(out):
        if arg == "--tasks" and i + 1 < len(out) and has_set(out[i + 1]):
            out[i + 1] = ",".join(expand_spec(out[i + 1]))
        elif arg.startswith("--tasks=") and has_set(arg.split("=", 1)[1]):
            out[i] = "--tasks=" + ",".join(expand_spec(arg.split("=", 1)[1]))
    return out


def add_bundled_tasks() -> None:
    """Every TaskManager lm_eval builds also indexes eval/tasks (FLORES+)."""
    from lm_eval.tasks import TaskManager

    if getattr(TaskManager.__init__, "_moto_patched", False):
        return
    original = TaskManager.__init__
    sig = inspect.signature(original)

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        bound = sig.bind_partial(self, *args, **kwargs)
        extra = bound.arguments.get("include_path")
        extra = [extra] if isinstance(extra, str) else list(extra or [])
        bound.arguments["include_path"] = [str(TASKS_DIR)] + [p for p in extra if p != str(TASKS_DIR)]
        original(*bound.args, **bound.kwargs)

    __init__._moto_patched = True
    TaskManager.__init__ = __init__


def main(argv: list[str] | None = None) -> None:
    argv = expand_tasks(list(sys.argv[1:] if argv is None else argv))
    add_bundled_tasks()
    try:  # r-three fork (lm-evaluation-harness@multilingual-tiny-benchmarks): subcommand CLI
        from lm_eval._cli import HarnessCLI
    except ImportError:
        HarnessCLI = None
    if HarnessCLI is None:  # stock lm-eval 0.4.x
        from lm_eval.__main__ import cli_evaluate

        sys.argv = [sys.argv[0]] + argv
        cli_evaluate()
        return
    from lm_eval.utils import setup_logging

    setup_logging()
    cli = HarnessCLI()
    # HarnessCLI reads sys.argv and inserts the `run` subcommand when it is omitted
    sys_argv, sys.argv = sys.argv, [sys.argv[0]] + argv
    try:
        args = cli.parse_args()
    finally:
        sys.argv = sys_argv
    if getattr(args, "command", None) == "run":
        extra = args.include_path
        args.include_path = [str(TASKS_DIR)] + [x for x in ([extra] if isinstance(extra, str) else list(extra or [])) if x != str(TASKS_DIR)]
    cli.execute(args)


if __name__ == "__main__":
    main()
