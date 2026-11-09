"""Paired standard errors of per-task accuracy deltas, from lm-eval's per-sample logs.

`eval_release.sh` run with LOG_SAMPLES=1 writes, next to results_*.json, one samples_<task>_<timestamp>.jsonl
per task: a line per document with its doc_id, doc_hash and each metric (acc, acc_norm, ...). Two models
evaluated on the same task see the same documents, so the delta of their accuracies can be tested on the
per-document differences d_i = x_model,i - x_baseline,i (in {-1, 0, 1}):

    delta    = mean(d)
    se_paired = std(d, ddof=1) / sqrt(n)

which is smaller than the independent-errors formula sqrt(se_model^2 + se_baseline^2) whenever the two models
get the same documents right or wrong (positive covariance). Only per-document 0/1 metrics qualify; rolling
likelihood metrics (FLORES word/byte perplexity, bits per byte) are not per-document means and are skipped.

No lm_eval import: runs on login nodes. Used by experiments/release-evals.py --paired.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLES_RE = re.compile(r"^samples_(?P<task>.+)_(?P<ts>\d{4}-\d{2}-\d{2}T[\d.\-]+)\.jsonl$")
FIELDS = ["model", "baseline", "task", "metric", "n", "n_hash_mismatch", "mean_model", "mean_baseline", "delta",
          "se_paired", "se_independent", "only_model_correct", "only_baseline_correct"]


def newest_samples(model_dir: Path) -> dict[str, Path]:
    """{task: newest samples file} under <model_dir>/<org>__<model>/ (newest by the timestamp in the name,
    not the path: a model can have results under two org directories)."""
    best: dict[str, tuple[str, Path]] = {}
    for f in model_dir.glob("*/samples_*.jsonl"):
        m = SAMPLES_RE.match(f.name)
        if m and (m["task"] not in best or m["ts"] > best[m["task"]][0]):
            best[m["task"]] = (m["ts"], f)
    return {task: f for task, (_, f) in best.items()}


@lru_cache(maxsize=512)
def read_binary_metrics(path: Path) -> dict[str, dict[int, tuple[str, float]]]:
    """{metric: {doc_id: (doc_hash, 0/1 value)}} for the metrics that are 0/1 for every document."""
    out: dict[str, dict[int, tuple[str, float]]] = defaultdict(dict)
    nonbinary: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("filter", "none") != "none":
                continue
            for m in r.get("metrics", []):
                v = r.get(m)
                if isinstance(v, bool):
                    v = float(v)
                if not isinstance(v, (int, float)) or v not in (0, 1):
                    nonbinary.add(m)
                    continue
                out[m][int(r["doc_id"])] = (r.get("doc_hash", ""), float(v))
    return {m: d for m, d in out.items() if m not in nonbinary}


def paired_stats(a: dict[int, tuple[str, float]], b: dict[int, tuple[str, float]]) -> dict | None:
    """Model a vs baseline b on the documents both logged with the same doc_hash."""
    common = [i for i in a if i in b]
    ids = [i for i in common if a[i][0] == b[i][0]]
    n = len(ids)
    if n < 2:
        return None
    xa = np.array([a[i][1] for i in ids])
    xb = np.array([b[i][1] for i in ids])
    d = xa - xb
    return {
        "n": n,
        "n_hash_mismatch": len(common) - n,
        "mean_model": float(xa.mean()),
        "mean_baseline": float(xb.mean()),
        "delta": float(d.mean()),
        "se_paired": float(d.std(ddof=1) / math.sqrt(n)),
        "se_independent": float(math.sqrt(xa.var(ddof=1) / n + xb.var(ddof=1) / n)),
        "only_model_correct": int((d > 0).sum()),
        "only_baseline_correct": int((d < 0).sum()),
    }


def compute_rows(results_root: Path, pairs: list[tuple[str, str]], wanted: set[tuple[str, str]]) -> list[dict]:
    """Rows for every (model, baseline) pair x wanted (task, metric) that both models have samples for."""
    rows: list[dict] = []
    files = {m: newest_samples(results_root / m) for m in {x for p in pairs for x in p}}
    for model, baseline in pairs:
        have = [t for t in {t for t, _ in wanted} if t in files[model] and t in files[baseline]]
        print(f"  paired {model} vs {baseline}: {len(have)} tasks with samples on both sides")
        for task in sorted(have):
            a_all, b_all = read_binary_metrics(files[model][task]), read_binary_metrics(files[baseline][task])
            for t, metric in sorted(wanted):
                if t != task or metric not in a_all or metric not in b_all:
                    continue
                s = paired_stats(a_all[metric], b_all[metric])
                if s:
                    rows.append({"model": model, "baseline": baseline, "task": task, "metric": metric, **s})
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_se(path: Path) -> dict[tuple[str, str, str, str], float]:
    """paired_stats.csv -> {(model, baseline, task, metric): se_paired}."""
    with open(path, newline="") as fh:
        return {(r["model"], r["baseline"], r["task"], r["metric"]): float(r["se_paired"]) for r in csv.DictReader(fh)}
