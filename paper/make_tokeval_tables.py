"""Intrinsic tokenizer tables (main-text summary + per-language appendix) from the TokEval driver outputs.

    python paper/make_tokeval_tables.py                       # reads results/tokeval, writes paper/tables/tokeval/

Persian (pes_Arab) is excluded by default: the joint mBPE baseline tokenizers were not trained on it
(train_tokenizer_on_all.sh lists 20 training sources, no fas_Arab), so it is out of distribution for them and would
dominate their spread. Pass exclude_languages=[] to keep it. Duplicate rows (Llama-43K and BPE-dropout use Llama's
tokenizer; MoTo-GPT4-Reg. uses MoTo-GPT4's) are dropped; they differ only in training.

Parity is tokens relative to the Llama-3.2 tokenizer's English count (fixed denominator, FLORES+ devtest, 1000
sentences per language); lower is better. Fertility is tokens per whitespace word and is averaged without zh/ja,
which have no whitespace words.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import json

import numpy as np
import pandas as pd

from moto.utils.args import Args, load_args

NAMES = {
    "llama32": "Llama-3.2",
    "joint_mbpe_gpt4": "mBPE-GPT4",
    "joint_mbpe_boundless": "mBPE-Boundless",
    "joint_mbpe_gpt4_323k": "mBPE-GPT4 (323K)",
    "parity_aware_bpe": "Parity-Aware BPE",
    "byt5": "ByT5",
    "moto_v1_oracle": "MoTo-GPT4",
    "moto_script": "MoTo-Script",
    "moto_1b": "MoTo-8K",
    "moto_v2": "MoTo-8K (v2)",
    "moto_16k_v2": "MoTo-16K",
    "moto_16k_plus_llama": "MoTo-16K + Llama",
}
SHORT = {"arb_Arab": "ar", "ces_Latn": "cs", "cmn_Hans": "zh", "dan_Latn": "da", "deu_Latn": "de", "ell_Grek": "el",
         "eng_Latn": "en", "fra_Latn": "fr", "hun_Latn": "hu", "ind_Latn": "id", "ita_Latn": "it", "jpn_Jpan": "ja",
         "nld_Latn": "nl", "pes_Arab": "fa", "pol_Latn": "pl", "por_Latn": "pt", "rus_Cyrl": "ru", "spa_Latn": "es",
         "swe_Latn": "sv", "tur_Latn": "tr", "vie_Latn": "vi"}
NO_WORDS = ["cmn_Hans", "jpn_Jpan"]


@dataclass
class TokEvalTablesArgs(Args):
    input_dir: str = "results/tokeval"
    output_dir: str = "paper/tables/tokeval"
    analysis_json: str = "analysis_results_full.json"       # in input_dir; for the Gini tables
    exclude_languages: List[str] = field(default_factory=lambda: ["pes_Arab"])
    settings: List[str] = field(default_factory=lambda: list(NAMES))


def gini(costs) -> float:
    """Gini coefficient of the per-language costs (0 = every language costs the same)."""
    x = np.sort(np.asarray(list(costs), float))
    n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def gini_tables(analysis: dict, keep: List[str], langs: List[str]):
    """(summary, per-language sentence cost) from TokEval's tokenizer_fairness_gini, restricted to `langs`.

    Per-sentence cost = tokens per FLORES+ sentence (the sentences are parallel, so a language's cost is directly
    comparable); per-byte cost = tokens per UTF-8 byte (TokEval's default 'global' Gini; a byte-level tokenizer is 0 by
    construction there). Both are recomputed on `langs` from the stored per-language costs."""
    per_tok = analysis["tokenizer_fairness_gini"]["per_tokenizer"]
    rows, sent = [], {}
    for k in keep:
        by_byte = {l: v for l, v in per_tok[k]["per_language"].items() if l in langs}
        by_sent = {l: v for l, v in per_tok[k]["per_line_normalization"]["language_costs"].items() if l in langs}
        sent[NAMES[k]] = by_sent
        lo, hi = min(by_sent, key=by_sent.get), max(by_sent, key=by_sent.get)
        rows.append({"Tokenizer": NAMES[k], "Gini (sentence)": gini(by_sent.values()), "Gini (byte)": gini(by_byte.values()),
                     "Tokens/sentence": float(np.mean(list(by_sent.values()))),
                     "Max/min cost": by_sent[hi] / by_sent[lo], "Costliest": SHORT[hi]})
    return pd.DataFrame(rows).set_index("Tokenizer"), pd.DataFrame(sent).loc[langs]


# ── parity against a fixed English denominator, in the layout of the rebuttal tables (tables/parity-against-*.md) ──
FULL = {"arb_Arab": "Arabic", "ces_Latn": "Czech", "cmn_Hans": "Chinese", "dan_Latn": "Danish", "deu_Latn": "German",
        "ell_Grek": "Greek", "pes_Arab": "Farsi", "fra_Latn": "French", "hun_Latn": "Hungarian", "ind_Latn": "Indonesian",
        "ita_Latn": "Italian", "jpn_Jpan": "Japanese", "nld_Latn": "Dutch", "pol_Latn": "Polish", "por_Latn": "Portuguese",
        "rus_Cyrl": "Russian", "spa_Latn": "Spanish", "swe_Latn": "Swedish", "tur_Latn": "Turkish", "vie_Latn": "Vietnamese",
        "eng_Latn": "English"}
ROW_ORDER = list(FULL)
# (output name, title, denominator setting)
RELATIVE = [("llama", "Llama", "llama32"), ("moto", "MoTo", "moto_1b"), ("parity_aware", "Parity-Aware BPE", "parity_aware_bpe")]
REL_BASELINES = ["llama32", "joint_mbpe_gpt4", "joint_mbpe_boundless", "parity_aware_bpe"]
REL_MOTO = ["moto_v1_oracle", "moto_1b", "moto_16k_v2"]
TEX_NAMES = {"moto_v1_oracle": r"\moto-GPT4", "moto_1b": r"\moto", "moto_16k_v2": r"\moto\ (16K)"}


def relative_parity(counts: pd.DataFrame, settings: List[str], denominator: str, langs: List[str]):
    """Per-sentence tokens(setting, language) / tokens(denominator, English) -> (mean, std) frames [language x setting].
    std is the population std over the sentences, as TokEval reports it."""
    p = counts.pivot_table(index=["language", "doc"], columns="setting", values="n_tokens")
    den = p.xs("eng_Latn", level="language")[denominator]
    mean, std = {}, {}
    for st in settings:
        r = {l: p[st].xs(l, level="language") / den for l in langs}
        mean[st] = pd.Series({l: v.mean() for l, v in r.items()})
        std[st] = pd.Series({l: v.std(ddof=0) for l, v in r.items()})
    return pd.DataFrame(mean), pd.DataFrame(std)


def write_relative(name: str, title: str, mean: pd.DataFrame, std: pd.DataFrame, out: Path, reference: str):
    order = [l for l in ROW_ORDER if l in mean.index]
    mean, std = mean.loc[order], std.loc[order]
    cell = lambda m, sd: f"{m:.2f} ± {sd:.2f}"
    best_row = mean.eq(mean.min(axis=1), axis=0)
    avg = mean.mean()
    md = [f"### Parity Against {title}", "", "| Language | " + " | ".join(NAMES[c] for c in mean.columns) + " |",
          "|---|" + "---|" * len(mean.columns)]
    for l in order:
        cs = [(f"**{cell(mean.loc[l, c], std.loc[l, c])}**" if round(mean.loc[l, c], 2) == round(mean.loc[l].min(), 2)
               else cell(mean.loc[l, c], std.loc[l, c])) for c in mean.columns]
        md.append(f"| {FULL[l]} | " + " | ".join(cs) + " |")
    md.append("| **Average** | " + " | ".join(f"**{avg[c]:.2f}**" if round(avg[c], 2) == round(avg.min(), 2) else f"{avg[c]:.2f}" for c in mean.columns) + " |")
    (out / f"parity_against_{name}.md").write_text("\n".join(md) + "\n")
    base = [c for c in mean.columns if c in REL_BASELINES]
    moto = [c for c in mean.columns if c in REL_MOTO]
    texn = lambda c: TEX_NAMES.get(c, NAMES[c])
    tex = ["\\begin{table*}[t]", "  \\centering", "  \\tiny", "  \\setlength{\\tabcolsep}{4pt}",
           f"  \\caption{{Parity against {title}: tokens per sentence relative to {reference}'s English count (mean $\\pm$ std over "
           f"sentences, {len(order) - ('eng_Latn' not in order)} languages; lower is better).}}",
           f"  \\label{{tab:parity-against-{name.replace('_', '-')}}}", "  \\begin{tabular}{l" + "r" * len(mean.columns) + "}",
           "    \\toprule", f"    \\multicolumn{{1}}{{l}}{{}} & \\multicolumn{{{len(base)}}}{{c}}{{\\textit{{Baselines}}}} & "
           f"\\multicolumn{{{len(moto)}}}{{c}}{{\\textit{{Per-language}}}} \\\\",
           f"    \\cmidrule(lr){{2-{1 + len(base)}}}\\cmidrule(lr){{{2 + len(base)}-{1 + len(base) + len(moto)}}}"]
    cols = base + moto
    tex.append("    \\textbf{Language} & " + " & ".join(f"\\textbf{{{texn(c)}}}" for c in cols) + " \\\\")
    tex.append("    \\midrule")
    for l in order:
        cs = []
        for c in cols:
            t = f"{mean.loc[l, c]:.2f} $\\pm$ {std.loc[l, c]:.2f}"
            cs.append(f"\\textbf{{{t}}}" if round(mean.loc[l, c], 2) == round(mean.loc[l, cols].min(), 2) else t)
        tex.append(f"    {FULL[l]} & " + " & ".join(cs) + " \\\\")
    tex.append("    \\midrule")
    tex.append("    \\textbf{Average} & " + " & ".join(f"\\textbf{{{avg[c]:.2f}}}" if round(avg[c], 2) == round(avg[cols].min(), 2) else f"{avg[c]:.2f}" for c in cols) + " \\\\")
    tex += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""]
    (out / f"parity_against_{name}.tex").write_text("\n".join(tex))
    pd.concat({"mean": mean, "std": std}, axis=1).round(4).to_csv(out / f"parity_against_{name}.csv")


def summary(parity: pd.DataFrame, per_lang: pd.DataFrame, langs: List[str]) -> pd.DataFrame:
    non_en = [l for l in langs if l != "eng_Latn"]
    word_langs = [l for l in langs if l not in NO_WORDS]
    fert = per_lang.pivot(index="tokenizer", columns="language", values="fertility")
    comp = per_lang.pivot(index="tokenizer", columns="language", values="compression_rate")
    p = parity[langs]
    return pd.DataFrame({
        "Parity (avg)": p.mean(axis=1),
        "Parity (std)": p.std(axis=1),
        "Parity (max non-En)": p[non_en].max(axis=1),
        "English": p["eng_Latn"],
        "Fertility": fert.loc[p.index, word_langs].mean(axis=1),
        "Compression": comp.loc[p.index, langs].mean(axis=1),
    })


def _dg(digits, col) -> int:
    return digits.get(col, 2) if isinstance(digits, dict) else digits


def to_md(df: pd.DataFrame, bold_min: List[str], digits=2) -> str:
    cells = df.copy().astype(object)
    for c in df.columns:
        best = df[c].min() if c in bold_min else None
        dg = _dg(digits, c)
        for i in df.index:
            s = f"{df.loc[i, c]:.{dg}f}"
            cells.loc[i, c] = f"**{s}**" if best is not None and abs(df.loc[i, c] - best) < 10 ** -dg / 2 else s
    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + list(cells.loc[i])) + " |" for i in df.index]
    return "\n".join([head, sep, *body]) + "\n"


def to_tex(df: pd.DataFrame, bold_min: List[str], caption: str, label: str, digits=2, small: bool = False) -> str:
    cols = "l" + "r" * len(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize" if small else "\\small", f"\\begin{{tabular}}{{{cols}}}",
             "\\toprule", " & ".join([""] + [c.replace("%", "\\%") for c in df.columns]) + " \\\\", "\\midrule"]
    for i in df.index:
        row = []
        for c in df.columns:
            dg = _dg(digits, c)
            s = f"{df.loc[i, c]:.{dg}f}"
            if c in bold_min and abs(df.loc[i, c] - df[c].min()) < 10 ** -dg / 2:
                s = f"\\textbf{{{s}}}"
            row.append(s)
        lines.append(" & ".join([i.replace("&", "\\&")] + row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(lines)


def main(args: TokEvalTablesArgs):
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    parity = pd.read_csv(inp / "parity_vs_llama.csv").set_index("setting")
    parity = parity[[c for c in parity.columns if not c.endswith("_std")]]
    per_lang = pd.read_csv(inp / "summary_per_language.csv")
    langs = [l for l in parity.columns if l not in args.exclude_languages]
    keep = [s for s in args.settings if s in parity.index]
    parity, per_lang = parity.loc[keep], per_lang[per_lang.tokenizer.isin(keep)]
    n = len(langs)
    note = (f"Averaged over {n} FLORES+ languages" + (f" (excluding {', '.join(SHORT[l] for l in args.exclude_languages)})"
            if args.exclude_languages else "") + "; parity is tokens relative to Llama-3.2's English count, lower is better. "
            "Fertility excludes zh and ja.")

    s = summary(parity, per_lang, langs)
    gini_df, sent_cost = None, None
    if (inp / args.analysis_json).exists():
        gini_df, sent_cost = gini_tables(json.load(open(inp / args.analysis_json)), keep, langs)
        s["Gini"] = gini_df["Gini (sentence)"].to_numpy()
    else:
        print(f"{inp / args.analysis_json} not found: no Gini column / table")
    s.index = [NAMES[i] for i in s.index]
    s.index.name = "Tokenizer"
    s.round(4).to_csv(out / "summary.csv")
    bold = ["Parity (avg)", "Parity (std)", "Parity (max non-En)", "Fertility", "Gini"]
    (out / "summary.md").write_text(to_md(s, bold, {"Gini": 3}))
    (out / "summary.tex").write_text(to_tex(s, bold, "Intrinsic tokenizer metrics. " + note + " Gini is over tokens per sentence.", "tab:tokeval-summary", {"Gini": 3}))

    counts = pd.read_csv(inp / "token_counts.csv")
    rel_cols = [c for c in REL_BASELINES + REL_MOTO if c in keep]
    for rname, rtitle, den in RELATIVE:
        if den in set(counts.setting):
            m, sd = relative_parity(counts, rel_cols, den, langs)
            ref_csv = {"llama32": "parity_vs_llama.csv", "moto_1b": "parity_vs_english_moto.csv"}.get(den)
            if ref_csv and (inp / ref_csv).exists():  # cross-check against TokEval's own fixed-denominator CSVs
                ref = pd.read_csv(inp / ref_csv).set_index("setting")
                assert abs(m.loc[langs, rel_cols[0]] - ref.loc[rel_cols[0], langs]).max() < 1e-6, f"{rname}: differs from TokEval CSV"
            write_relative(rname, rtitle, m, sd, out, rtitle)
    if gini_df is not None:
        gcap = (f"Cross-language Gini coefficient of tokenization cost over {n} languages"
                + (f" (excluding {', '.join(SHORT[l] for l in args.exclude_languages)})" if args.exclude_languages else "")
                + ": cost is tokens per FLORES+ sentence (parallel text) or per UTF-8 byte; 0 means every language costs the same. "
                "A byte-level tokenizer is 0 per byte by construction, so read the Gini next to the tokens per sentence.")
        gini_df.round(4).to_csv(out / "gini.csv")
        gbold = ["Gini (sentence)", "Gini (byte)"]
        num = gini_df.drop(columns="Costliest")
        gdg = {"Gini (sentence)": 3, "Gini (byte)": 3, "Tokens/sentence": 1, "Max/min cost": 2}
        (out / "gini.md").write_text(to_md(num, gbold, gdg))
        (out / "gini.tex").write_text(to_tex(num, gbold, gcap, "tab:tokeval-gini", gdg))
        sent_cost.round(2).to_csv(out / "gini_cost_per_language.csv")
        t = sent_cost.copy()
        t.index = [SHORT[l] for l in t.index]
        t.index.name = "Language"
        (out / "gini_cost_per_language.md").write_text(to_md(t, [], 1))
        (out / "gini_cost_per_language.tex").write_text(to_tex(
            t, [], "Tokens per FLORES+ sentence by language (the costs behind the Gini).", "tab:tokeval-gini-cost", 1, small=True))
    for name, df, caption in [
        ("parity_per_language", parity[langs], "Parity per language, relative to Llama-3.2's English token count (lower is better)."),
        ("fertility_per_language", per_lang.pivot(index="tokenizer", columns="language", values="fertility").loc[keep, langs],
         "Fertility (tokens per whitespace word) per language; zh and ja have no whitespace words."),
    ]:
        t = df.T
        t.columns = [NAMES[c] for c in t.columns]
        t.index = [SHORT[l] for l in t.index]
        t.index.name = "Language"
        t.round(4).to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(to_md(t, list(t.columns) if False else []))
        (out / f"{name}.tex").write_text(to_tex(t, [], caption, f"tab:tokeval-{name.replace('_', '-')}", small=True))
    print(f"wrote {out}/ ({n} languages: {' '.join(SHORT[l] for l in langs)})")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main(load_args(TokEvalTablesArgs))
