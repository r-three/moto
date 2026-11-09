"""Download everything an eval job needs, so jobs can run with HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1.

Run on a login node (with internet), with the same HF_HOME / HF_DATASETS_CACHE as the jobs:

    python -m moto.hf.release.eval.prefetch --tasks paper+toksuite \
        --models r-three-moto/moto-1b r-three-moto/moto-1b-baseline-llama ...

Per model: the weights/config/tokenizer files (not training/), and the remote tokenizer
code (cached by loading it once). Per task: its dataset, by building the lm-eval task.
"""

from __future__ import annotations

import argparse
import logging

from .__main__ import TASKS_DIR, add_bundled_tasks
from .task_sets import expand

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prefetch")

# modeling_*.py: repos with remote_code (e.g. moto-1b-baseline-llama-albert ships modeling_llama_albert.py);
# the model is loaded with trust_remote_code, which fetches that file at load time and fails offline if it is not cached
MODEL_PATTERNS = ["*.json", "*.safetensors", "tokenizer.py", "modeling_*.py", "tokenizers/*", "*.model", "*.txt", "*.md", "NOTICE", "LICENSE*"]


def prefetch_models(models: list[str]) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    for spec in models:
        m, _, rev = spec.partition("@")  # repo or repo@commit
        rev = rev or None
        path = snapshot_download(m, revision=rev, allow_patterns=MODEL_PATTERNS)
        try:  # caches tokenizer.py as a module; weights-only repos (tokenizer elsewhere) have none
            tok = AutoTokenizer.from_pretrained(m, revision=rev, trust_remote_code=True)
            # superset tokenizers (moto-1b-superset-albert) load their components from the original
            # sources on first use (Hub repos, tiktoken / tokenmonster files): touch each one now
            for name in getattr(tok, "components", None) or []:
                tok.encode_superset_ids("prefetch", component=name)
                logger.info("model %s: component %s cached", m, name)
        except Exception as e:  # noqa: BLE001
            logger.info("model %s: no loadable tokenizer here (%s); fine if the tokenizer comes from another repo", m, type(e).__name__)
        logger.info("model %s -> %s", m, path)


def prefetch_tasks(spec: str) -> None:
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = expand(spec)
    add_bundled_tasks()
    tm = TaskManager(include_path=[str(TASKS_DIR)])
    failed = []
    for t in tasks:  # one at a time, so one failure (e.g. gated FLORES+ without access) doesn't hide the rest
        try:
            get_task_dict([t], tm)
            logger.info("task %s ready", t)
        except Exception as e:  # noqa: BLE001
            failed.append(t)
            logger.error("task %s: %s: %s", t, type(e).__name__, e)
    if failed:
        raise SystemExit(f"{len(failed)} tasks failed to download: {failed}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--tasks", default="paper+toksuite")
    a = ap.parse_args()
    prefetch_models(a.models)
    if a.tasks:
        prefetch_tasks(a.tasks)


if __name__ == "__main__":
    main()
