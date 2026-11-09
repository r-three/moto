"""
Sample-level tokenizer analysis.

Directory layout assumed:
  EVAL_DIR/
    englishcode-32000-consistent-v1/samples.json
    gpt-4o/samples.json
    tekken/samples.json
    flexitok/reindexed-<name>/samples.json   (one per flexitok tokenizer)

Each samples.json is a list of dicts with at minimum:
  - doc_id   : int 
  - acc      : float  (1.0 = correct, 0.0 = wrong)
  - doc      : dict   (contains the original task fields)

The task name is recovered from results.json alongside samples.json:
  results.json -> dict keyed by task name.

Usage:
    python paper/per-sample-albert-w-confidence.py \
        --eval_dir /path/to/train/eval_superset_albert_w_xglm/eval \
        --out_dir  paper/sample_analysis_out/albert_camera_ready \
        --from_cache \
"""
"""
Sample-level tokenizer analysis.

Directory layout assumed:
  EVAL_DIR/
    englishcode-32000-consistent-v1/samples.json
    gpt-4o/samples.json
    tekken/samples.json
    flexitok/reindexed-<name>/samples.json   (one per flexitok tokenizer)

Additional eval dirs (e.g. LLaMA baseline, llama-43k, llama-albert) can be
passed via --extra_eval_dirs LABEL:PATH [LABEL:PATH ...].

Usage:
  # First run (reads all samples.json files, saves cache):
  python paper/per-sample-albert-w-confidence.py \
      --eval_dir   /path/to/train/eval_superset_albert_w_xglm/eval \
      --out_dir    paper/rebuttal/sample-analysis-out \
        > paper/sample_analysis_out/albert/summary.log 2>&1
      --from_cache

  # Subsequent runs (skip re-reading, use cached all_samples.csv):
  python paper/per-sample-albert-w-confidence.py \
      --out_dir    ./sample_analysis_out \
      --from_cache
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── taxonomy ───────────────────────────────────────────────────────────────────
# (benchmark_group, task_key, language_code)
TASK_META = [
    # English reasoning
    ("English Reasoning", "arc_easy",        "en"),
    ("English Reasoning", "arc_challenge",   "en"),
    ("English Reasoning", "hellaswag",       "en"),
    ("English Reasoning", "piqa",            "en"),
    ("English Reasoning", "boolq",           "en"),
    ("English Reasoning", "commonsense_qa",  "en"),
    # XWinograd
    ("XWinograd",         "xwinograd_en",    "en"),
    ("XWinograd",         "xwinograd_fr",    "fr"),
    ("XWinograd",         "xwinograd_jp",    "ja"),
    ("XWinograd",         "xwinograd_pt",    "pt"),
    ("XWinograd",         "xwinograd_ru",    "ru"),
    ("XWinograd",         "xwinograd_zh",    "zh"),
    # XCOPA
    ("XCOPA",             "xcopa_id",        "id"),
    ("XCOPA",             "xcopa_it",        "it"),
    ("XCOPA",             "xcopa_tr",        "tr"),
    ("XCOPA",             "xcopa_vi",        "vi"),
    ("XCOPA",             "xcopa_zh",        "zh"),
    # XStoryCloze
    ("XStoryCloze",       "xstorycloze_ar",  "ar"),
    ("XStoryCloze",       "xstorycloze_en",  "en"),
    ("XStoryCloze",       "xstorycloze_es",  "es"),
    ("XStoryCloze",       "xstorycloze_id",  "id"),
    ("XStoryCloze",       "xstorycloze_ru",  "ru"),
    ("XStoryCloze",       "xstorycloze_zh",  "zh"),
    # PAWS-X
    ("PAWS-X",            "paws_de",         "de"),
    ("PAWS-X",            "paws_en",         "en"),
    ("PAWS-X",            "paws_es",         "es"),
    ("PAWS-X",            "paws_fr",         "fr"),
    ("PAWS-X",            "paws_ja",         "ja"),
    ("PAWS-X",            "paws_zh",         "zh"),
    # Multilingual HellaSwag
    ("HellaSwag-X",       "hellaswag_ar",    "ar"),
    ("HellaSwag-X",       "hellaswag_da",    "da"),
    ("HellaSwag-X",       "hellaswag_de",    "de"),
    ("HellaSwag-X",       "hellaswag_fr",    "fr"),
    ("HellaSwag-X",       "hellaswag_hu",    "hu"),
    ("HellaSwag-X",       "hellaswag_id",    "id"),
    ("HellaSwag-X",       "hellaswag_it",    "it"),
    ("HellaSwag-X",       "hellaswag_nl",    "nl"),
    ("HellaSwag-X",       "hellaswag_pt",    "pt"),
    ("HellaSwag-X",       "hellaswag_ru",    "ru"),
    ("HellaSwag-X",       "hellaswag_vi",    "vi"),
    ("HellaSwag-X",       "hellaswag_es",    "es"),
    # XNLI
    ("XNLI",              "xnli_ar",         "ar"),
    ("XNLI",              "xnli_zh",         "zh"),
    ("XNLI",              "xnli_de",         "de"),
    ("XNLI",              "xnli_el",         "el"),
    ("XNLI",              "xnli_en",         "en"),
    ("XNLI",              "xnli_es",         "es"),
    ("XNLI",              "xnli_fr",         "fr"),
    ("XNLI",              "xnli_ru",         "ru"),
    ("XNLI",              "xnli_tr",         "tr"),
    ("XNLI",              "xnli_vi",         "vi"),
    # TokSuite
    ("TokSuite",          "toksuite_english",  "en"),
    ("TokSuite",          "toksuite_chinese",  "zh"),
    ("TokSuite",          "toksuite_turkish",  "tr"),
    ("TokSuite",          "toksuite_italian",  "it"),
    ("TokSuite",          "toksuite_farsi",    "fa"),
    ("TokSuite",          "toksuite_math",     "math"),
    ("TokSuite",          "toksuite_stem",     "stem"),
]

TASKS      = [t for _, t, _ in TASK_META]
TASK_SET   = set(TASKS)
TASK_GROUP = {t: g for g, t, _ in TASK_META}
TASK_LANG  = {t: l for _, t, l in TASK_META}
# insertion-ordered unique groups
GROUPS     = list(dict.fromkeys(g for g, _, _ in TASK_META))

# ── I/O helpers ────────────────────────────────────────────────────────────────

def short_name(samples_path: Path) -> str:
    p = samples_path.parent
    if p.parent.name == "flexitok":
        return p.name.replace("reindexed-", "")
    return p.name


def _loglik(entry) -> float:
    """Unwrap one choice's loglikelihood entry.

    lm-eval-harness stores each choice as [ll_str, is_greedy_str] in
    `filtered_resps`, or as [[ll_str, is_greedy_str]] (one extra nesting
    level) in raw `resps`.
    """
    if isinstance(entry[0], list):
        entry = entry[0]
    return float(entry[0])


def confidence_margin(sample: dict) -> float:
    """Log-likelihood margin between the gold choice and the best distractor.

    Returns NaN if the sample doesn't carry per-choice loglikelihoods (e.g.
    generative tasks, or older samples.json dumps that stripped resps).
    """
    resps = sample.get("filtered_resps") or sample.get("resps")
    target = sample.get("target")
    if not resps or not isinstance(target, int):
        return float("nan")
    try:
        lls = [_loglik(r) for r in resps]
    except (TypeError, ValueError, IndexError):
        return float("nan")
    if not (0 <= target < len(lls)):
        return float("nan")
    others = lls[:target] + lls[target + 1:]
    if not others:
        return float("nan")
    return lls[target] - max(others)


def load_samples(samples_path: Path) -> dict[str, dict[int, dict]]:
    """Returns {task_name: {doc_id: {"acc": float, "margin": float}}}."""

    def _process(samples) -> dict[int, dict]:
        return {
            s["doc_id"]: {"acc": float(s["acc"]), "margin": confidence_margin(s)}
            for s in samples
        }

    with open(samples_path) as f:
        raw = json.load(f)

    out: dict[str, dict[int, dict]] = {}

    if isinstance(raw, dict):
        for task, samples in raw.items():
            if task in TASK_SET:
                out[task] = _process(samples)
        return out

    # flat list — infer task from sibling results.json
    results_path = samples_path.parent / "results.json"
    if not results_path.exists():
        print(f"  [warn] no results.json next to {samples_path}, skipping",
              file=sys.stderr)
        return {}
    with open(results_path) as f:
        res = json.load(f)
    task_results = res.get("results", res)
    matching = [t for t in task_results if t in TASK_SET]
    if not matching:
        print(f"  [warn] no matching tasks in {results_path}", file=sys.stderr)
        return {}
    if len(matching) > 1:
        print(f"  [warn] ambiguous flat samples.json ({samples_path}), "
              f"tasks={matching} — skipping", file=sys.stderr)
        return {}
    task = matching[0]
    out[task] = _process(raw)
    return out


def discover_tokenizers(
    eval_dir: Path,
    extra: list[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    found = []
    for sp in sorted(eval_dir.rglob("samples.json")):
        found.append((short_name(sp), sp))
    for label, extra_dir in extra:
        hits = sorted(extra_dir.rglob("samples.json"))
        if not hits:
            print(f"  [warn] no samples.json under {extra_dir}", file=sys.stderr)
            continue
        if len(hits) == 1:
            found.append((label, hits[0]))
        else:
            for sp in hits:
                found.append((f"{label}/{short_name(sp)}", sp))
    return found


# ── bootstrap ──────────────────────────────────────────────────────────────────

def bootstrap_random_router(
    acc_matrix: np.ndarray,   # (n_samples, n_toks), may contain NaN
    B: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Returns (mean_pct, ci_half_width_pct) for a uniform random router.
    Per sample: pick one tokenizer uniformly at random among those that have a
    non-NaN value for that sample.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_toks = acc_matrix.shape
    valid_mask = ~np.isnan(acc_matrix)

    # Fast path: if no NaNs, we can vectorise fully
    if valid_mask.all():
        # shape (B, n_samples): one random column index per sample per iter
        chosen_cols = rng.integers(0, n_toks, size=(B, n_samples))
        rows = np.arange(n_samples)
        # acc_matrix indexed per bootstrap iteration
        boot_means = np.array([
            acc_matrix[rows, chosen_cols[b]].mean() for b in range(B)
        ])
    else:
        boot_means = np.empty(B)
        for b in range(B):
            chosen_accs = np.full(n_samples, np.nan)
            for i in range(n_samples):
                valid_cols = np.where(valid_mask[i])[0]
                if len(valid_cols):
                    chosen_accs[i] = acc_matrix[i, rng.choice(valid_cols)]
            boot_means[b] = np.nanmean(chosen_accs)

    return 100 * boot_means.mean(), 100 * 1.96 * boot_means.std()


def bootstrap_pct(
    bool_arr: np.ndarray,
    B: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Returns (mean_pct, ci_half_width_pct) for the fraction of True values in
    `bool_arr`, via sample-level bootstrap resampling.
    """
    rng = np.random.default_rng(seed)
    n = len(bool_arr)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = bool_arr[idx].mean(axis=1)
    return 100 * boot_means.mean(), 100 * 1.96 * boot_means.std()


def oracle_at_least_n_table(
    all_df: pd.DataFrame,
    ns: list[int],
    n_tokenizers: int,
    B: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Oracle-correct := at least n tokenizers get the sample right, with a
    bootstrap 95% CI on the resulting percentage."""
    n_correct = all_df["n_correct"].to_numpy()
    rows = []
    for n in ns:
        if n > n_tokenizers:
            print(f"  [warn] --oracle_ns includes {n} but only "
                  f"{n_tokenizers} tokenizer(s) are loaded — skipping",
                  file=sys.stderr)
            continue
        bool_arr = (n_correct >= n).astype(float)
        boot_mean, ci = bootstrap_pct(bool_arr, B=B, seed=seed)
        rows.append({
            "at_least_n":        n,
            "n_samples":         int(bool_arr.sum()),
            "pct_of_total":      round(100 * bool_arr.mean(), 4),
            "bootstrap_mean_pct": round(boot_mean, 4),
            "ci95_halfwidth_pct": round(ci, 4),
        })
    return pd.DataFrame(rows)


def confidence_oracle_table(
    all_df: pd.DataFrame,
    tok_labels: list[str],
    thresholds: list[float],
    B: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Confident-oracle-correct := at least one tokenizer gets the sample
    right AND its answer's log-likelihood margin over the best distractor
    exceeds `threshold` nats (i.e. the correct answer is at least
    exp(threshold)x more likely than any wrong choice), with a bootstrap
    95% CI on the resulting percentage.

    Samples lacking loglikelihood data (margin is NaN for every tokenizer)
    are excluded — they carry no evidence either way.
    """
    margin_cols = [f"{t}__margin" for t in tok_labels]
    has_margin = all_df[margin_cols].notna().any(axis=1)
    scored_df = all_df[has_margin]
    n_excluded = len(all_df) - len(scored_df)
    if n_excluded:
        print(f"  [info] {n_excluded} sample(s) lack loglikelihood data "
              f"for every tokenizer — excluded from confidence-oracle analysis",
              file=sys.stderr)

    rows = []
    for thr in thresholds:
        confident_correct = np.zeros(len(scored_df), dtype=bool)
        for t in tok_labels:
            confident_correct |= (
                (scored_df[t] == 1.0) & (scored_df[f"{t}__margin"] > thr)
            ).to_numpy()
        bool_arr = confident_correct.astype(float)
        boot_mean, ci = bootstrap_pct(bool_arr, B=B, seed=seed)
        rows.append({
            "confidence_threshold_nats": thr,
            "min_likelihood_ratio":      round(float(np.exp(thr)), 4),
            "n_samples_scored":          len(scored_df),
            "n_confident_oracle":        int(bool_arr.sum()),
            "pct_of_scored":             round(100 * bool_arr.mean(), 4),
            "bootstrap_mean_pct":        round(boot_mean, 4),
            "ci95_halfwidth_pct":        round(ci, 4),
        })
    return pd.DataFrame(rows)


def oracle_at_least_n_by_group_table(
    all_df: pd.DataFrame,
    ns: list[int],
    n_tokenizers: int,
    groups: list[str],
    B: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Same as oracle_at_least_n_table, but computed separately within each
    benchmark group (plus an 'ALL' group covering every sample)."""
    frames = []
    for group in ["ALL"] + groups:
        g_df = all_df if group == "ALL" else all_df[all_df["task_group"] == group]
        if g_df.empty:
            continue
        g_table = oracle_at_least_n_table(g_df, ns, n_tokenizers, B=B, seed=seed)
        g_table.insert(0, "group", group)
        frames.append(g_table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def confidence_oracle_by_group_table(
    all_df: pd.DataFrame,
    tok_labels: list[str],
    thresholds: list[float],
    groups: list[str],
    B: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Same as confidence_oracle_table, but computed separately within each
    benchmark group (plus an 'ALL' group covering every sample)."""
    frames = []
    for group in ["ALL"] + groups:
        g_df = all_df if group == "ALL" else all_df[all_df["task_group"] == group]
        if g_df.empty:
            continue
        margin_cols = [f"{t}__margin" for t in tok_labels]
        if not g_df[margin_cols].notna().any().any():
            continue
        g_table = confidence_oracle_table(g_df, tok_labels, thresholds, B=B, seed=seed)
        g_table.insert(0, "group", group)
        frames.append(g_table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def routing_stats(
    sub_df: pd.DataFrame,
    tok_labels: list[str],
    B: int = 2_000,
) -> dict:
    mat     = sub_df[tok_labels].to_numpy().astype(float)
    oracle  = 100 * (sub_df["n_correct"] >= 1).mean()
    avg_tok = 100 * float(np.nanmean(mat))
    rand_mean, rand_ci = bootstrap_random_router(mat, B=B)
    return {
        "n_samples":          len(sub_df),
        "avg_single_tok_acc": round(avg_tok,           4),
        "random_router_acc":  round(rand_mean,         4),
        "random_router_ci95": round(rand_ci,           4),
        "oracle_acc":         round(oracle,            4),
        "oracle_gain_pp":     round(oracle - avg_tok,  4),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, default=None,
                    help="Root eval directory. Not required with --from_cache.")
    ap.add_argument(
        "--extra_eval_dirs", nargs="*", default=[], metavar="LABEL:PATH",
        help="Additional tokenizer eval dirs. e.g. "
             "llama-baseline:/path/to/eval  llama-43k:/path/to/eval2",
    )
    ap.add_argument("--out_dir", default="./sample_analysis_out", type=Path)
    ap.add_argument(
        "--from_cache", action="store_true",
        help="Load all_samples.csv from --out_dir instead of re-reading samples.json.",
    )
    ap.add_argument("--bootstrap_B", type=int, default=10_000,
                    help="Bootstrap iterations for global random-router estimate.")
    ap.add_argument(
        "--oracle_ns", type=int, nargs="*", default=[1, 2, 5, 7, 10, 13, 14],
        help="Thresholds n for the 'at least n tokenizers correct' oracle "
             "CI analysis.",
    )
    ap.add_argument(
        "--confidence_thresholds", type=float, nargs="*",
        default=[0.0, 1.0, 2.0, 5.0],
        help="Log-likelihood margin thresholds (nats) for the confidence "
             "oracle analysis. exp(threshold) is the minimum likelihood "
             "ratio between the correct answer and the best distractor.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "all_samples.csv"

    # ── 1. build or load the flat sample table ────────────────────────────────
    META_COLS = {"task", "task_group", "language", "doc_id", "n_correct"}

    if args.from_cache:
        if not cache_path.exists():
            sys.exit(f"[error] Cache not found: {cache_path}\n"
                     f"Run without --from_cache first.")
        print(f"Loading from cache: {cache_path}")
        all_df = pd.read_csv(cache_path)
        margin_cols = [c for c in all_df.columns if c.endswith("__margin")]
        tok_labels = [
            c for c in all_df.columns if c not in META_COLS and c not in margin_cols
        ]
        print(f"  {len(all_df)} rows, {len(tok_labels)} tokenizers: {tok_labels}")

    else:
        if args.eval_dir is None:
            sys.exit("[error] --eval_dir required unless --from_cache is set.")

        extra_parsed: list[tuple[str, Path]] = []
        for item in args.extra_eval_dirs:
            if ":" not in item:
                sys.exit(f"[error] --extra_eval_dirs expects LABEL:PATH, got: {item!r}")
            label, path_str = item.split(":", 1)
            extra_parsed.append((label, Path(path_str)))

        tokenizers = discover_tokenizers(args.eval_dir, extra_parsed)
        print(f"Found {len(tokenizers)} tokenizer(s):")
        for label, p in tokenizers:
            print(f"  {label:55s}  {p}")

        tok_labels = [label for label, _ in tokenizers]

        task_data: dict[str, dict[str, dict[int, dict]]] = defaultdict(dict)
        for label, sp in tokenizers:
            print(f"\nLoading {label} …")
            for task, doc_accs in load_samples(sp).items():
                task_data[task][label] = doc_accs
                print(f"  {task}: {len(doc_accs)} samples")

        all_sample_rows = []
        for task in TASKS:
            if task not in task_data:
                continue
            tok_map = task_data[task]
            if not tok_map:
                continue
            all_doc_ids = sorted(
                set().union(*[set(d.keys()) for d in tok_map.values()])
            )
            if not all_doc_ids:
                continue

            df = pd.DataFrame(index=all_doc_ids)
            for tok in tok_labels:
                if tok in tok_map:
                    df[tok] = pd.Series({d: v["acc"] for d, v in tok_map[tok].items()})
                    df[f"{tok}__margin"] = pd.Series(
                        {d: v["margin"] for d, v in tok_map[tok].items()}
                    )
            df["n_correct"] = df[tok_labels].sum(axis=1)

            margin_labels = [f"{tok}__margin" for tok in tok_labels]
            for doc_id, row in df.iterrows():
                all_sample_rows.append({
                    "task":       task,
                    "task_group": TASK_GROUP[task],
                    "language":   TASK_LANG[task],
                    "doc_id":     doc_id,
                    "n_correct":  int(row["n_correct"]),
                    **{tok: row[tok] for tok in tok_labels},
                    **{mcol: row[mcol] for mcol in margin_labels},
                })

        all_df = pd.DataFrame(all_sample_rows)
        all_df.to_csv(cache_path, index=False)
        print(f"\nSaved cache: {cache_path}  ({len(all_df)} rows)")

    N = len(tok_labels)
    B = args.bootstrap_B
    B_sub = max(500, B // 5)

    # ── 2. Coverage table: ≥n tokenizers correct ─────────────────────────────
    total = len(all_df)
    coverage_rows = []
    for n in range(1, N + 1):
        count = int((all_df["n_correct"] >= n).sum())
        coverage_rows.append({
            "at_least_n":   n,
            "n_samples":    count,
            "pct_of_total": round(100 * count / total, 1),
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(args.out_dir / "coverage_by_n_tokenizers.csv", index=False)
    print("\n── Coverage by n tokenizers ──")
    print(coverage_df.to_string(index=False))

    # ── 3. Exclusive wins ─────────────────────────────────────────────────────
    excl_rows = []
    for tok in tok_labels:
        total_correct = int((all_df[tok] == 1.0).sum())
        excl = int(((all_df["n_correct"] == 1) & (all_df[tok] == 1.0)).sum())
        excl_rows.append({
            "tokenizer":      tok,
            "total_correct":  total_correct,
            "exclusive_wins": excl,
            "pct_exclusive":  round(100 * excl / total_correct, 1)
                              if total_correct else 0,
        })
    excl_df = (pd.DataFrame(excl_rows)
               .sort_values("exclusive_wins", ascending=False))
    excl_df.to_csv(args.out_dir / "exclusive_wins.csv", index=False)
    print("\n── Exclusive wins per tokenizer ──")
    print(excl_df.to_string(index=False))

    # ── 4. Hierarchical routing summary ──────────────────────────────────────
    #   levels: overall → group → task
    #   (language level only emitted where a task covers multiple languages,
    #    which for this benchmark set never happens — each task_key is one lang)
    hier_rows = []

    def _add(level, name, group, task, language, sub):
        s = routing_stats(sub, tok_labels, B=B_sub)
        hier_rows.append({
            "level":    level,
            "name":     name,
            "group":    group,
            "task":     task,
            "language": language,
            **s,
        })

    _add("overall", "ALL", "", "", "", all_df)

    for group in GROUPS:
        g_df = all_df[all_df["task_group"] == group]
        if g_df.empty:
            continue
        _add("group", group, group, "", "", g_df)

        for task in TASKS:
            if TASK_GROUP.get(task) != group:
                continue
            t_df = g_df[g_df["task"] == task]
            if t_df.empty:
                continue
            lang = TASK_LANG[task]
            _add("task", task, group, task, lang, t_df)

    hier_df = pd.DataFrame(hier_rows)
    hier_df.to_csv(args.out_dir / "routing_hierarchy.csv", index=False)
    print(f"\nWrote routing_hierarchy.csv  ({len(hier_df)} rows)")

    # ── 5. Per-task summary with task/lang columns ────────────────────────────
    task_rows = []
    for task in TASKS:
        t_df = all_df[all_df["task"] == task]
        if t_df.empty:
            continue
        n = len(t_df)
        n_any  = int((t_df["n_correct"] >= 1).sum())
        n_all  = int((t_df["n_correct"] == N).sum())
        n_none = int((t_df["n_correct"] == 0).sum())
        task_rows.append({
            "task_group":        TASK_GROUP[task],
            "task":              task,
            "language":          TASK_LANG[task],
            "n_samples":         n,
            "pct_any_correct":   round(100 * n_any  / n, 1),
            "pct_7_or_more_correct": round(100 * (t_df["n_correct"] >= 7).sum() / n, 1),
            "pct_all_correct":   round(100 * n_all  / n, 1),
            "pct_none_correct":  round(100 * n_none / n, 1),
        })
    task_df = pd.DataFrame(task_rows)
    task_df.to_csv(args.out_dir / "task_summary.csv", index=False)
    print(f"Wrote task_summary.csv  ({len(task_df)} tasks)")

    # ── 6. LaTeX table (grouped, midrules between benchmark groups) ───────────
    latex_lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Group & Task & $N$ "
        "& $\\geq$1 tok.\\ & $\\geq$7 tok.\\ & All toks.\\ & No tok.\\ \\\\",
        "\\midrule",
    ]
    current_group = None
    for _, row in task_df.iterrows():
        if row["task_group"] != current_group:
            if current_group is not None:
                group_df = task_df[task_df["task_group"] == current_group]
                # averages
                latex_lines.append(
                    f" \\cline{{4-7}}\\noalign{{\\vskip 2pt}}"
                    f"&&& {group_df['pct_any_correct'].mean():>5.1f}\\% "
                    f"& {group_df['pct_7_or_more_correct'].mean():>5.1f}\\% "
                    f"& {group_df['pct_all_correct'].mean():>5.1f}\\% "
                    f"& {group_df['pct_none_correct'].mean():>5.1f}\\% \\\\[2pt]"
                )
                latex_lines.append("\\midrule")
            current_group = row["task_group"]
            grp_label = current_group
        else:
            grp_label = ""
        task_label = row["task"] if current_group == "English Reasoning" else row["language"]
        latex_lines.append(
            f"  {grp_label:<20s} & {task_label:<28s}"
            f"& {row['n_samples']:>6d} "
            f"& {row['pct_any_correct']:>5.1f}\\% "
            f"& {row['pct_7_or_more_correct']:>5.1f}\\% "
            f"& {row['pct_all_correct']:>5.1f}\\% "
            f"& {row['pct_none_correct']:>5.1f}\\% \\\\"
        )
    
    group_df = task_df[task_df["task_group"] == current_group]
    # averages
    latex_lines.append(
        f" \\cline{{4-7}}\\noalign{{\\vskip 2pt}}"
        f"&&& {group_df['pct_any_correct'].mean():>5.1f}\\% "
        f"& {group_df['pct_7_or_more_correct'].mean():>5.1f}\\% "
        f"& {group_df['pct_all_correct'].mean():>5.1f}\\% "
        f"& {group_df['pct_none_correct'].mean():>5.1f}\\% \\\\[2pt]"
    )
    latex_lines += ["\\bottomrule", "\\end{tabular}"]
    latex_lines = [line.replace("_", "\\_") for line in latex_lines]
    (args.out_dir / "task_coverage.tex").write_text("\n".join(latex_lines))
    print("Wrote task_coverage.tex")

    # ── 7. Global routing bounds ──────────────────────────────────────────────
    print(f"\n── Global routing bounds (B={B}) ──")
    g_stats = routing_stats(all_df, tok_labels, B=B)
    print(f"  Average single-tokenizer : {g_stats['avg_single_tok_acc']:.2f}%")
    print(f"  Random router (boot.)    : {g_stats['random_router_acc']:.2f}%"
          f" ± {g_stats['random_router_ci95']:.2f}pp")
    print(f"  Oracle router            : {g_stats['oracle_acc']:.2f}%")
    print(f"  Upper-bound gain         : +{g_stats['oracle_gain_pp']:.2f}pp")
    pd.DataFrame([g_stats]).to_csv(args.out_dir / "routing_bounds.csv", index=False)

    # ── 8. Group-level routing summary (stdout) ───────────────────────────────
    print("\n── Routing bounds by benchmark group ──")
    grp_view = hier_df[hier_df["level"] == "group"][[
        "name", "n_samples", "avg_single_tok_acc",
        "random_router_acc", "random_router_ci95",
        "oracle_acc", "oracle_gain_pp",
    ]]
    print(grp_view.to_string(index=False))

    # ── 9. Oracle CI: at least n tokenizers correct ────────────────────────────
    print(f"\n── Oracle (≥n tokenizers correct) with 95% CI (B={B_sub}) ──")
    oracle_n_df = oracle_at_least_n_table(all_df, args.oracle_ns, N, B=B_sub)
    oracle_n_df.to_csv(args.out_dir / "oracle_at_least_n.csv", index=False)
    print(oracle_n_df.to_string(index=False))

    print(f"\n── Oracle (≥n tokenizers correct) by benchmark group, "
          f"95% CI (B={B_sub}) ──")
    oracle_n_by_group_df = oracle_at_least_n_by_group_table(
        all_df, args.oracle_ns, N, GROUPS, B=B_sub,
    )
    oracle_n_by_group_df.to_csv(
        args.out_dir / "oracle_at_least_n_by_group.csv", index=False
    )
    print(oracle_n_by_group_df.to_string(index=False))

    # ── 10. Confidence oracle: correct AND significantly more likely ──────────
    margin_cols = [f"{t}__margin" for t in tok_labels]
    if all_df[margin_cols].notna().any().any():
        print(f"\n── Confidence oracle (correct + log-lik. margin > threshold), "
              f"95% CI (B={B_sub}) ──")
        conf_df = confidence_oracle_table(
            all_df, tok_labels, args.confidence_thresholds, B=B_sub,
        )
        conf_df.to_csv(args.out_dir / "oracle_confidence.csv", index=False)
        print(conf_df.to_string(index=False))

        print(f"\n── Confidence oracle by benchmark group, 95% CI (B={B_sub}) ──")
        conf_by_group_df = confidence_oracle_by_group_table(
            all_df, tok_labels, args.confidence_thresholds, GROUPS, B=B_sub,
        )
        conf_by_group_df.to_csv(
            args.out_dir / "oracle_confidence_by_group.csv", index=False
        )
        print(conf_by_group_df.to_string(index=False))
    else:
        print("\n[info] no loglikelihood data (filtered_resps/resps) found in "
              "any samples.json — skipping confidence oracle analysis.",
              file=sys.stderr)

    print(f"\nAll outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()