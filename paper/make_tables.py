"""
Generate LaTeX tables for fertility, parity, pcw metrics.
One table per metric, languages as rows, model settings as columns.

python paper/make_tables.py --dispersion std
"""

import argparse
import os

import pandas as pd

RES_DIR = "/path/to/results/v2/paper_final"
RES_DIR = "/path/to/results/rebuttal"
# ── language mappings ─────────────────────────────────────────────────────────

LANG_TO_PRETTY = {
    "eng_Latn": "English",
    "arb_Arab": "Arabic",
    "ces_Latn": "Czech",
    "dan_Latn": "Danish",
    "deu_Latn": "German",
    "ell_Grek": "Greek",
    "fra_Latn": "French",
    "hun_Latn": "Hungarian",
    "ind_Latn": "Indonesian",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "nld_Latn": "Dutch",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "spa_Latn": "Spanish",
    "swe_Latn": "Swedish",
    "tur_Latn": "Turkish",
    "vie_Latn": "Vietnamese",
    "cmn_Hani": "Chinese",
    "fas_Arab": "Farsi",
    "fw_edu":   "English",
}

# Ordered language rows
LANG_ORDER = [
    "arb_Arab", "ces_Latn", "cmn_Hani", "dan_Latn", "deu_Latn",
    "ell_Grek", "fas_Arab", "fra_Latn", "hun_Latn", "ind_Latn",
    "ita_Latn", "jpn_Jpan", "nld_Latn", "pol_Latn", "por_Latn",
    "rus_Cyrl", "spa_Latn", "swe_Latn", "tur_Latn", "vie_Latn",
    "fw_edu",
]

# fw_edu reads from eng_Latn column in the CSV
LANG_CSV_COL = {lc: lc for lc in LANG_ORDER}
LANG_CSV_COL["fw_edu"] = "eng_Latn"

# ── per-language tokenizer lookup ─────────────────────────────────────────────
# Keys: fw2hq_8k | moto_8k | script | moto_16k | llama_16k

LANG_CONFIGS = {
    "arb_Arab": {"fw2hq_8k": "flexitok/bpe_arb_Arab_8000",    "moto_8k": "flexitok/bpe_ltr_arb_Arab_8000_v3",   "script": "flexitok/bpe_script_Arab_16000",       "moto_16k": "flexitok/bpe_ltr_arb_Arab_16000_v2",   "llama_16k": "flexitok/bpe_ltr_arb_Arab_16000_v2"},
    "ces_Latn": {"fw2hq_8k": "flexitok/bpe_ces_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_ces_Latn_8000_v3",   "script": "flexitok/bpe_script_Slav_16000",       "moto_16k": "flexitok/bpe_ltr_ces_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_ces_Latn_16000_v2"},
    "cmn_Hani": {"fw2hq_8k": "flexitok/bpe_cmn_Hani_8000",    "moto_8k": "flexitok/bpe_ltr_cmn_Hani_8000_v3",   "script": "flexitok/bpe_script_CmJp_16000",       "moto_16k": "flexitok/bpe_ltr_cmn_Hani_16000_v2",   "llama_16k": "flexitok/bpe_ltr_cmn_Hani_16000_v2"},
    "dan_Latn": {"fw2hq_8k": "flexitok/bpe_dan_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_dan_Latn_8000_v3",   "script": "flexitok/bpe_script_Germ_32000",       "moto_16k": "flexitok/bpe_ltr_dan_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_dan_Latn_16000_v2"},
    "deu_Latn": {"fw2hq_8k": "flexitok/bpe_deu_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_deu_Latn_8000_v3",   "script": "flexitok/bpe_script_Germ_32000",       "moto_16k": "flexitok/bpe_ltr_deu_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_deu_Latn_16000_v2"},
    "ell_Grek": {"fw2hq_8k": "flexitok/bpe_ell_Grek_8000",    "moto_8k": "flexitok/bpe_ltr_ell_Grek_8000_v3",   "script": "flexitok/bpe_ltr_ell_Grek_8000_v2",   "moto_16k": "flexitok/bpe_ltr_ell_Grek_16000_v2",   "llama_16k": "flexitok/bpe_ltr_ell_Grek_16000_v2"},
    "fas_Arab": {"fw2hq_8k": "flexitok/bpe_fas_Arab_8000",    "moto_8k": "flexitok/bpe_ltr_fas_Arab_8000_v3",   "script": "flexitok/bpe_script_Arab_16000",       "moto_16k": "flexitok/bpe_ltr_fas_Arab_16000_v2",   "llama_16k": "flexitok/bpe_ltr_fas_Arab_16000_v2"},
    "fra_Latn": {"fw2hq_8k": "flexitok/bpe_fra_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_fra_Latn_8000_v3",   "script": "flexitok/bpe_script_Roma_32000",       "moto_16k": "flexitok/bpe_ltr_fra_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_fra_Latn_16000_v2"},
    "fw_edu":   {"fw2hq_8k": "flexitok/bpe_fw_edu_8000",      "moto_8k": "flexitok/bpe_ltr_fw_edu_8000_v3",     "script": "flexitok/bpe_ltr_fw_edu_32000_v2",     "moto_16k": "flexitok/bpe_ltr_fw_edu_16000_v2",     "llama_16k": "meta-llama/Llama-3.2-1B"},
    "hun_Latn": {"fw2hq_8k": "flexitok/bpe_hun_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_hun_Latn_8000_v3",   "script": "flexitok/bpe_ltr_hun_Latn_8000_v2",   "moto_16k": "flexitok/bpe_ltr_hun_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_hun_Latn_16000_v2"},
    "ind_Latn": {"fw2hq_8k": "flexitok/bpe_ind_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_ind_Latn_8000_v3",   "script": "flexitok/bpe_script_SEAS_16000",       "moto_16k": "flexitok/bpe_ltr_ind_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_ind_Latn_16000_v2"},
    "ita_Latn": {"fw2hq_8k": "flexitok/bpe_ita_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_ita_Latn_8000_v3",   "script": "flexitok/bpe_script_Roma_32000",       "moto_16k": "flexitok/bpe_ltr_ita_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_ita_Latn_16000_v2"},
    "jpn_Jpan": {"fw2hq_8k": "flexitok/bpe_jpn_Jpan_8000",    "moto_8k": "flexitok/bpe_ltr_jpn_Jpan_8000_v3",   "script": "flexitok/bpe_script_CmJp_16000",       "moto_16k": "flexitok/bpe_ltr_jpn_Jpan_16000_v2",   "llama_16k": "flexitok/bpe_ltr_jpn_Jpan_16000_v2"},
    "nld_Latn": {"fw2hq_8k": "flexitok/bpe_nld_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_nld_Latn_8000_v3",   "script": "flexitok/bpe_script_Germ_32000",       "moto_16k": "flexitok/bpe_ltr_nld_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_nld_Latn_16000_v2"},
    "pol_Latn": {"fw2hq_8k": "flexitok/bpe_pol_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_pol_Latn_8000_v3",   "script": "flexitok/bpe_script_Slav_16000",       "moto_16k": "flexitok/bpe_ltr_pol_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_pol_Latn_16000_v2"},
    "por_Latn": {"fw2hq_8k": "flexitok/bpe_por_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_por_Latn_8000_v3",   "script": "flexitok/bpe_script_Roma_32000",       "moto_16k": "flexitok/bpe_ltr_por_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_por_Latn_16000_v2"},
    "rus_Cyrl": {"fw2hq_8k": "flexitok/bpe_rus_Cyrl_8000",    "moto_8k": "flexitok/bpe_ltr_rus_Cyrl_8000_v3",   "script": "flexitok/bpe_ltr_rus_Cyrl_16000_v2",  "moto_16k": "flexitok/bpe_ltr_rus_Cyrl_16000_v2",   "llama_16k": "flexitok/bpe_ltr_rus_Cyrl_16000_v2"},
    "spa_Latn": {"fw2hq_8k": "flexitok/bpe_spa_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_spa_Latn_8000_v3",   "script": "flexitok/bpe_script_Roma_32000",       "moto_16k": "flexitok/bpe_ltr_spa_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_spa_Latn_16000_v2"},
    "swe_Latn": {"fw2hq_8k": "flexitok/bpe_swe_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_swe_Latn_8000_v3",   "script": "flexitok/bpe_script_Germ_32000",       "moto_16k": "flexitok/bpe_ltr_swe_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_swe_Latn_16000_v2"},
    "tur_Latn": {"fw2hq_8k": "flexitok/bpe_tur_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_tur_Latn_8000_v3",   "script": "flexitok/bpe_ltr_tur_Latn_8000_v2",   "moto_16k": "flexitok/bpe_ltr_tur_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_tur_Latn_16000_v2"},
    "vie_Latn": {"fw2hq_8k": "flexitok/bpe_vie_Latn_8000",    "moto_8k": "flexitok/bpe_ltr_vie_Latn_8000_v3",   "script": "flexitok/bpe_script_SEAS_16000",       "moto_16k": "flexitok/bpe_ltr_vie_Latn_16000_v2",   "llama_16k": "flexitok/bpe_ltr_vie_Latn_16000_v2"},
}

LANG_CONFIGS["eng_Latn"] = LANG_CONFIGS["fw_edu"]  # fw_edu reads from eng_Latn column in the CSV
# ── column definitions ────────────────────────────────────────────────────────

# Baseline: single tokenizer applied to all language columns
BASELINE_MODELS = {
    "llama":     "meta-llama/Llama-3.2-1B",
    "byt5":      "google/byt5-small",
    "boundless": "flexitok/bpe_boundless_AllL_128000",
    "gpt4":      "flexitok/bpe_gpt4_AllL_128000",
    # "parity_aware": "flexitok/parity_aware_bpe_tr_500k_val_1000",
    "parity_aware_ratio": "flexitok/parity-aware-bpe-ratio",
    "parity_aware": "flexitok/parity-aware-bpe-devset",
}
BASELINE_KEYS = set(BASELINE_MODELS)

# (col_key, LaTeX header label)
COLUMNS = [
    ("llama",     r"Llama"),
    # ("byt5",      r"ByT5"),
    # ("boundless", r"Boundless"),
    # ("gpt4",      r"GPT4"),
    # ("fw2hq_8k",  r"\moto-GPT4"),
    ("moto_8k",   r"\moto"),
    ("parity_aware",   r"Parity Aware BPE"),
    ("parity_aware_ratio",   r"Parity Aware BPE (Ratio)"),
    # ("script",    r"Script"),
    # ("moto_16k",  r"\moto\ (16K)"),
    # ("llama_16k", r"Llama+16K"),
]

# ── metric files ──────────────────────────────────────────────────────────────

METRICS = [
    # ("fertility", f"{RES_DIR}/fertility_results.csv", "Fertility"),
    # ("parity",    f"{RES_DIR}/parity_results.csv",  "Parity"),
    ("parity-against-parity-aware", f"{RES_DIR}/parity-against-parity-aware-tokenizer.csv", "Parity Against Parity-Aware Tokenizer"),
    ("parity-against-llama",    f"{RES_DIR}/parity-against-llama-tokenizer.csv",  "Parity Against Llama"),
    ("parity-llama-against-llama",    f"{RES_DIR}/parity-against-llama-tokenizer.csv",  "Llama Parity Against Llama"),
    ("parity-against-fw-edu",    f"{RES_DIR}/parity-against-fw-edu-tokenizer.csv",  "Parity Against English Tokenizer"),
    # ("pcw",       f"{RES_DIR}/pcw_results.csv",         "PCW"),
    # ("gini",       f"{RES_DIR}/gini_results.csv",         "Gini"),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def fmt(val, std=None, bold=False):
    try:
        s = f"{float(val):.2f}"
        if std is not None:
            s += f" $\\pm$ {float(std):.2f}"
        return f"\\textbf{{{s}}}" if bold else s
    except (ValueError, TypeError):
        return r"---"


def fmt_md(val, std=None, bold=False):
    try:
        s = f"{float(val):.2f}"
        if std is not None:
            s += f" ± {float(std):.2f}"
        return f"**{s}**" if bold else s
    except (ValueError, TypeError):
        return "---"


def lookup(df_idx, tokenizer_name, csv_col):
    """Return float value or None."""
    if tokenizer_name is None or tokenizer_name not in df_idx.index:
        return None
    row = df_idx.loc[tokenizer_name]
    if csv_col not in row.index:
        return None
    try:
        return float(row[csv_col])
    except (ValueError, TypeError):
        return None


def collect_table_data(df, show_std=False):
    """Gather per-language values plus the average row, shared by all output formats.

    When show_std is True and the CSV has a "{lang}_std" column, each value is
    paired with its dispersion so callers can render it as "val $\\pm$ std".
    """
    df_idx = df.set_index("tokenizer")

    col_keys   = [ck for ck, _ in COLUMNS]
    col_labels = [cl for _, cl in COLUMNS]
    n_cols = len(COLUMNS)

    missing = []
    col_sums   = [0.0] * n_cols
    col_counts = [0]   * n_cols

    rows = []  # list of (pretty_lang, [(value, std_or_None)])
    for lang_code in LANG_ORDER:
        pretty   = LANG_TO_PRETTY.get(lang_code, lang_code)
        csv_col  = LANG_CSV_COL.get(lang_code, lang_code)
        std_col  = f"{csv_col}_std"
        has_std  = show_std and std_col in df.columns

        vals = []
        for col_key in col_keys:
            if col_key in BASELINE_KEYS:
                tok = BASELINE_MODELS[col_key]
            else:
                tok = LANG_CONFIGS.get(lang_code, {}).get(col_key)
            val = lookup(df_idx, tok, csv_col)
            std = lookup(df_idx, tok, std_col) if has_std else None
            if val is None:
                missing.append((lang_code, col_key, tok))
            vals.append((val, std))

        for i, (val, _) in enumerate(vals):
            if val is not None:
                col_sums[i]   += val
                col_counts[i] += 1

        rows.append((pretty, vals))

    avgs = [col_sums[i] / col_counts[i] if col_counts[i] else None for i in range(n_cols)]

    return col_labels, rows, avgs, missing


def make_table(df, metric_label, show_std=False):
    col_labels, rows, avgs, missing = collect_table_data(df, show_std=show_std)
    n_cols = len(col_labels)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \tiny")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(f"  \\caption{{{metric_label} scores by language and tokenizer setting.}}")
    lines.append(f"  \\label{{tab:{metric_label.lower()}}}")

    col_spec = "l" + "r" * n_cols
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    lines.append(
        r"    \multicolumn{1}{l}{} & "
        r"\multicolumn{4}{c}{\textit{Baselines}} & "
        r"\multicolumn{5}{c}{\textit{Per-language}} \\"
    )
    lines.append(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-10}")

    header_cells = [r"\textbf{Language}"] + [f"\\textbf{{{c}}}" for c in col_labels]
    lines.append("    " + " & ".join(header_cells) + r" \\")
    lines.append(r"    \midrule")

    for pretty, vals in rows:
        # best = lowest (all three metrics are lower-is-better)
        numeric = [val for val, _ in vals if val is not None]
        best    = min(numeric) if numeric else None

        cells = [pretty]
        for val, std in vals:
            is_best = (val is not None) and (best is not None) and (val == best)
            cells.append(fmt(val, std=std, bold=is_best))

        lines.append("    " + " & ".join(cells) + r" \\")

    # Average row
    best_avg = min(v for v in avgs if v is not None) if any(v is not None for v in avgs) else None
    lines.append(r"    \midrule")
    avg_cells = [r"\textbf{Average}"]
    for val in avgs:
        is_best = (val is not None) and (best_avg is not None) and (val == best_avg)
        avg_cells.append(fmt(val, bold=is_best))
    lines.append("    " + " & ".join(avg_cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    result = "\n".join(lines)

    if missing:
        print(f"  [WARN] {len(missing)} missing (tokenizer/lang not found in CSV):")
        for lc, ck, tok in missing[:10]:
            print(f"    lang={lc}  col={ck}  tok={tok}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    return result


def make_csv(df, show_std=False):
    col_labels, rows, avgs, _ = collect_table_data(df, show_std=show_std)

    if show_std:
        header = ["Language"] + [c for cl in col_labels for c in (cl, f"{cl}_std")]
    else:
        header = ["Language"] + col_labels
    lines = [",".join(header)]
    for pretty, vals in rows:
        cells = [pretty]
        for val, std in vals:
            cells.append(f"{val:.4f}" if val is not None else "")
            if show_std:
                cells.append(f"{std:.4f}" if std is not None else "")
        lines.append(",".join(cells))
    avg_cells = ["Average"]
    for v in avgs:
        avg_cells.append(f"{v:.4f}" if v is not None else "")
        if show_std:
            avg_cells.append("")
    lines.append(",".join(avg_cells))

    return "\n".join(lines)


def make_markdown(df, metric_label, show_std=False):
    col_labels, rows, avgs, _ = collect_table_data(df, show_std=show_std)

    lines = [f"### {metric_label}", ""]
    header = ["Language"] + col_labels
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for pretty, vals in rows:
        numeric = [val for val, _ in vals if val is not None]
        best    = min(numeric) if numeric else None
        cells = [pretty]
        for val, std in vals:
            is_best = (val is not None) and (best is not None) and (val == best)
            cells.append(fmt_md(val, std=std, bold=is_best))
        lines.append("| " + " | ".join(cells) + " |")

    best_avg = min(v for v in avgs if v is not None) if any(v is not None for v in avgs) else None
    avg_cells = ["**Average**"]
    for val in avgs:
        is_best = (val is not None) and (best_avg is not None) and (val == best_avg)
        avg_cells.append(fmt_md(val, bold=is_best))
    lines.append("| " + " | ".join(avg_cells) + " |")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dispersion",
        choices=["none", "std"],
        default="none",
        help="Report dispersion (2 decimal points, $\\pm$ sign) alongside each value "
             "when a matching '{lang}_std' column is present in the CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    show_std = args.dispersion == "std"

    os.makedirs("tables", exist_ok=True)

    for metric_key, csv_file, metric_label in METRICS:
        if not os.path.exists(csv_file):
            print(f"[SKIP] {csv_file} not found")
            continue

        print(f"\n=== {metric_label} ===")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()

        if "tokenizer" not in df.columns:
            print(f"  [ERROR] no 'tokenizer' column in {csv_file}")
            continue

        tex = make_table(df, metric_label, show_std=show_std)
        csv_out = make_csv(df, show_std=show_std)
        md_out = make_markdown(df, metric_label, show_std=show_std)

        tex_path = os.path.join("tables", f"{metric_key}.tex")
        csv_path = os.path.join("tables", f"{metric_key}.csv")
        md_path  = os.path.join("tables", f"{metric_key}.md")

        with open(tex_path, "w") as f:
            f.write(tex + "\n")
        with open(csv_path, "w") as f:
            f.write(csv_out + "\n")
        with open(md_path, "w") as f:
            f.write(md_out + "\n")

        print(f"  wrote {tex_path}, {csv_path}, {md_path}")

    print("\nDone. Files in ./tables/")


if __name__ == "__main__":
    main()
                   