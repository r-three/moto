#!/usr/bin/env python
# coding: utf-8

# cd experiments
# python repeated-evals.py
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

from moto.log_utils.plot_utils import setup_styles
from moto.utils.task_to_lang import LANG_TO_PRETTY, LANGS_TO_TASKS, TASK_TO_LANG

setup_styles()
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
from pathlib import Path

# "gsaltintas/FlexiTok-Evaluations",
WANDB_RUN = "FlexiTok-Evaluations-Paper"
# WANDB_RUN = "gsaltintas/moto-multilingual-addition"

out_dir = (Path().parent / "results" / "summaries").resolve().absolute()
out_dir = (Path(__file__).resolve().absolute().parents[1] / "paper" / "results" )
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {out_dir}")

api = wandb.Api()
api.flush()
filters = {
    "tags": {"$nin": ["english-models", "ignore"]},
}
runs = api.runs(
    # "gsaltintas/FlexiTok-Evaluations-NoBootstrap",
    WANDB_RUN,
    filters=filters,
)
print(f"Loading {len(runs)} runs from {WANDB_RUN}.")
print(set(t.name for t in runs))

langs=["arb_Arab","ces_Latn","cmn_Hani","dan_Latn","deu_Latn","ell_Grek","fas_Arab","fra_Latn","eng_Latn","hun_Latn","ind_Latn","ita_Latn","jpn_Jpan","nld_Latn","pol_Latn","por_Latn","rus_Cyrl","spa_Latn","swe_Latn","tur_Latn","vie_Latn"]
english_baselines = [
    "arc_challenge/acc_norm",
    "arc_easy/acc_norm",
    "piqa/acc_norm",
    "hellaswag/acc_norm",
    "boolq/acc",
    "commonsense_qa/acc",
]
belebele = [
    "belebele_arb_Arab/acc_norm",
    "belebele_ces_Latn/acc_norm",
    "belebele_dan_Latn/acc_norm",
    "belebele_deu_Latn/acc_norm",
    "belebele_ell_Grek/acc_norm",
    "belebele_eng_Latn/acc_norm",
    "belebele_fra_Latn/acc_norm",
    "belebele_hun_Latn/acc_norm",
    "belebele_ind_Latn/acc_norm",
    "belebele_ita_Latn/acc_norm",
    "belebele_jpn_Jpan/acc_norm",
    "belebele_nld_Latn/acc_norm",
    "belebele_pes_Arab/acc_norm",
    "belebele_pol_Latn/acc_norm",
    "belebele_por_Latn/acc_norm",
    "belebele_rus_Cyrl/acc_norm",
    "belebele_spa_Latn/acc_norm",
    "belebele_swe_Latn/acc_norm",
    "belebele_tur_Latn/acc_norm",
    "belebele_vie_Latn/acc_norm",
    "belebele_zho_Hans/acc_norm",
]
include = [
    "include_base_44_arabic/acc",
    "include_base_44_chinese/acc",
    "include_base_44_dutch/acc",
    "include_base_44_french/acc",
    "include_base_44_german/acc",
    "include_base_44_greek/acc",
    "include_base_44_hungarian/acc",
    "include_base_44_indonesian/acc",
    "include_base_44_italian/acc",
    "include_base_44_japanese/acc",
    "include_base_44_persian/acc",
    "include_base_44_polish/acc",
    "include_base_44_portuguese/acc",
    "include_base_44_russian/acc",
    "include_base_44_spanish/acc",
    "include_base_44_turkish/acc",
    "include_base_44_vietnamese/acc",
]
xnli = [
    "xnli_ar/acc",
    "xnli_zh/acc",
    "xnli_de/acc",
    "xnli_el/acc",
    "xnli_en/acc",
    "xnli_es/acc",
    "xnli_fr/acc",
    "xnli_ru/acc",
    "xnli_tr/acc",
    "xnli_vi/acc",
]
flores = [f"flores200_{lang}/word_perplexity" for lang in langs]
flores += [f"flores200_{lang}/bits_per_byte" for lang in langs]
flores += [f"flores200_{lang}/byte_perplexity" for lang in langs]
others = [
    "xcopa_id/acc",
    "xcopa_it/acc",
    "xcopa_tr/acc",
    "xcopa_vi/acc",
    "xcopa_zh/acc",
    "xstorycloze_ar/acc",
    "xstorycloze_en/acc",
    "xstorycloze_es/acc",
    "xstorycloze_id/acc",
    "xstorycloze_ru/acc",
    "xstorycloze_zh/acc",
    "paws_de/acc",
    "paws_en/acc",
    "paws_es/acc",
    "paws_fr/acc",
    "paws_ja/acc",
    "paws_zh/acc",
   "xwinograd_en/acc",
   "xwinograd_fr/acc",
   "xwinograd_jp/acc",
   "xwinograd_pt/acc",
   "xwinograd_ru/acc",
   "xwinograd_zh/acc",
]
toksuite = ["toksuite_english_canonical/acc_norm", "toksuite_turkish_canonical/acc_norm", "toksuite_chinese_canonical/acc_norm", "toksuite_italian_canonical/acc_norm", "toksuite_farsi_canonical/acc_norm", "toksuite_math_canonical/acc_norm", "toksuite_stem_canonical/acc_norm"]
toksuite_all = ["toksuite_english/acc_norm", "toksuite_turkish/acc_norm", "toksuite_chinese/acc_norm", "toksuite_italian/acc_norm", "toksuite_farsi/acc_norm", "toksuite_math/acc_norm", "toksuite_stem/acc_norm"]
okapi = [
    "hellaswag_ar/acc_norm", 
    "hellaswag_da/acc_norm", 
    "hellaswag_de/acc_norm", 
    "hellaswag_fr/acc_norm", 
    "hellaswag_hu/acc_norm", 
    "hellaswag_id/acc_norm", 
    "hellaswag_it/acc_norm", 
    "hellaswag_nl/acc_norm", 
    "hellaswag_pt/acc_norm", 
    "hellaswag_ru/acc_norm", 
    # "hellaswag_sv/acc_norm", 
    "hellaswag_vi/acc_norm", 
    "hellaswag_es/acc_norm",
]


# In[273]:


summary_list = []
for run in runs:
    summary = run.summary._json_dict
    config = run.config
    entry = {
        "run_id": run.id,
        "run_name": run.name.replace("Superset_Albert-", "SA-").replace("flexitok/reindexed-", "").replace("flexitok/bpe_", "").replace("Superset", "SS").replace("Fineweb-2-HQ", "FW2HQ"),
        # "model_name": config["model_name_or_path"],
        # "tokenizer_name": config["tokenizer_name"],
        # "tags": run.tags,
    }
    for key in include + belebele + english_baselines + toksuite + toksuite_all + flores +xnli +others + okapi:
        if key in summary:
            entry[key] = summary[key]
        std_err_key = f"{key}_stderr"
        if std_err_key in summary:
            entry[std_err_key] = summary[std_err_key]
    summary_list.append(entry)

summary_df = pd.DataFrame(summary_list)
summary_df["english_avg_acc_norm"] = summary_df[english_baselines].mean(axis=1)
summary_df["belebele_avg_acc_norm"] = summary_df[belebele].mean(axis=1)
summary_df["include_base_avg_acc"] = summary_df[include].mean(axis=1)
summary_df["toksuite_avg_acc_canonical"] = summary_df[toksuite].mean(axis=1)
summary_df["toksuite_avg_acc_norm"] = summary_df[toksuite_all].mean(axis=1)
summary_df["xnli_avg_acc"] = summary_df[xnli].mean(axis=1)
summary_df["okapi_avg_acc_norm"] = summary_df[okapi].mean(axis=1)
summary_df.tail()


# In[274]:


print(f"Unique run names: {summary_df['run_name'].unique()}")
actual = ["Llama", "Llama-BPEDropout", "Llama-Albert", "Llama-43K", 
          "SS-Albert",
          "FW2HQ-v1-Oracle", 
          "FW2HQ-v1-Oracle-09", "FW2HQ-v1-Random",
          "FW2HQ-v2-Oracle", "FW2HQ-Llama-16K-Oracle", "FW2HQ-Script-Oracle",
          "ByT5", "FW2HQ-16K-Oracle", "FW2HQ-8K-v3-Oracle", 
          "Multilingual GPT4-PreTok", "Multilingual Boundless BPE",

          ]
print(f"Difference: {set(summary_df['run_name'].unique()) - set(actual)}")
summary_df["_index"] = summary_df["run_name"].apply(lambda x: actual.index(x) if x in actual else len(actual))
summary_df = summary_df.sort_values(by="_index")
# summary_df = summary_df.sort_values(by="run_name", key=lambda x: actual.index(x) if x in actual else len(actual))


## create summaries

MAIN_TEXT_LANGS = ["eng_Latn", "fas_Arab", "ita_Latn", "cmn_Hani", "tur_Latn", "jpn_Jpan", "rus_Cyrl", "ell_Grek", "deu_Latn"]

RUN_GROUPS = {
    "main":       ["Llama", "FW2HQ-8K-v3-Oracle", "FW2HQ-v1-Oracle", "Llama-BPEDropout", "Multilingual GPT4-PreTok", "Multilingual Boundless BPE"],
    "routing":    ["FW2HQ-v1-Oracle-09", "FW2HQ-v1-Random", "ByT5",],
    # "routing":    ["FW2HQ-v1-Oracle-09", "FW2HQ-v1-Random", "FW2HQ-v1-Oracle", "ByT5",],
    "vocab_size": ["FW2HQ-Llama-16K-Oracle", "FW2HQ-16K-Oracle", "FW2HQ-Script-Oracle", "mBPE-GPT4 (323K)"],
    "industry":   ["Llama-43K", "Llama-Albert", "SS-Albert", "Superset_Albert-Fixed (w/ XGLM)", "SA-Fixed (w/ XGLM)"],
}
RUN_GROUPS["all"] = [run for group in RUN_GROUPS.values() for run in group]
RUN_GROUPS["ablations"] = [run for code, group in RUN_GROUPS.items() for run in group if code in ("routing", "vocab_size", "industry")]
rename_map = {
    "FW2HQ-8K-v3-Oracle": "MoTo",
    "FW2HQ-v1-Oracle": "MoTo-GPT4",
    "FW2HQ-v1-Oracle-09": "MoTo-GPT4-Reg.",
    "FW2HQ-v1-Random": "MoTo-Random",
    "FW2HQ-Llama-16K-Oracle": "MoTo-Llama-16K",
    "FW2HQ-16K-Oracle": "MoTo-16K",
    "FW2HQ-Script-Oracle": "MoTo-Script",
    "Llama-BPEDropout": "Llama-BPE-Drop.",
"Multilingual GPT4-PreTok":   "mBPE-GPT4",
"Multilingual Boundless BPE": "mBPE-Boundless",
"Superset_Albert-Fixed (w/ XGLM)": "SS-Albert",
"SA-Fixed (w/ XGLM)": "SS-Albert",

}
BASELINES = {
    "main": "Llama",
    "routing": "FW2HQ-v1-Random",
    # "routing": "FW2HQ-v1-Oracle",
    "vocab_size": "Llama",
    # "vocab_size": "FW2HQ-Llama-16K-Oracle",
    "industry": "Llama",
    "all": "Llama",
    "ablations": "Llama",
}


########################
BENCHMARK_TASKS = {
    "english": {
        "HellaSwag":  ["hellaswag/acc_norm"],
        "PIQA": ["piqa/acc_norm"],
        "ARC-Easy": ["arc_easy/acc_norm"],
        "ARC-Challenge": ["arc_challenge/acc_norm"],
        "BoolQ": ["boolq/acc"],
        "CommonsenseQA": ["commonsense_qa/acc"],
        "XWinograd": ["xwinograd_en/acc"],
    },
    "toksuite": {
        lang: [t for t in tasks if "toksuite" in t and "canonical" not in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "toksuite_canonical": {
        lang: [t for t in tasks if "toksuite" in t and "canonical" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "okapi": {
        lang: [t for t in tasks if "hellaswag" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "xnli": {
        lang: [t for t in tasks if "xnli" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "xcopa": {
        lang: [t for t in tasks if "xcopa" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "xstorycloze": {
        lang: [t for t in tasks if "xstorycloze" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "paws": {
        lang: [t for t in tasks if "paws" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "xwinograd": {
        lang: [t for t in tasks if "xwinograd" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "flores_bpb": {
        lang: [t for t in tasks if "flores" in t and "bits_per_byte" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "flores_byte_perplexity": {
        lang: [t for t in tasks if "flores" in t and "byte_perplexity" in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
    "acc_main": {
        lang: [t for t in tasks if "acc" in t and "belebele" not in t and "include" not in t]
        for lang, tasks in LANGS_TO_TASKS.items()
    },
}
# drop empty
BENCHMARK_TASKS = {
    bname: {lang: tasks for lang, tasks in btasks.items() if tasks}
    for bname, btasks in BENCHMARK_TASKS.items()
}


def compute_group_summary_for_benchmark(
    group: str,
    benchmark: str,
    baseline: str = "Llama",
    percentage: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.Series | None]:
    group_runs = RUN_GROUPS[group]
    acc_tasks = BENCHMARK_TASKS[benchmark]
    langs = acc_tasks.keys()

    group_runs = list(set(group_runs) | {baseline})
    lang_cols = []
    for lang in langs:
        tasks = acc_tasks.get(lang, [])
        if not tasks:
            continue
        df_ = summary_df[summary_df["run_name"].isin(group_runs)]
        if df_.empty:
            continue
        lang_short = LANG_TO_PRETTY.get(lang, lang.split("_")[0])

        # import code; code.interact(local=dict(locals(), **globals()))
        if "flores" in benchmark:
            col = (
            df_.groupby("run_name")[tasks]
            .min()
            .mean(axis=1)
            .rename(lang_short)
            .reset_index()
            .set_index("run_name")
        )
        else:
            col = (
            df_.groupby("run_name")[tasks]
            .max()
            .mean(axis=1)
            .rename(lang_short)
            .reset_index()
            .set_index("run_name")
        )
        lang_cols.append(col)

    if not lang_cols:
        print(f"  No data for {group}/{benchmark}, skipping.")
        return None, None, None

    summary = pd.concat(lang_cols, axis=1, join="inner").round(4)
    summary["Avg"] = summary.mean(axis=1).round(4)
    if percentage:
        summary = summary * 100

    baseline_row = summary.loc[baseline] if baseline in summary.index else None

    delta = None
    if baseline_row is not None:
        numeric_cols = [c for c in summary.columns if c not in ("baseline",)]
        delta = (summary[numeric_cols] - baseline_row[numeric_cols]).drop(index=baseline).round(2)
        delta.index = delta.index.map(lambda x: rename_map.get(x, x))
        delta = delta.reset_index().rename(columns={"run_name": "run_name"})

    summary.index = summary.index.map(lambda x: rename_map.get(x, x))
    summary = summary.reset_index()
    return summary, delta, baseline_row


full_res = ""
baseline_performances = {}

for group_code in RUN_GROUPS:
    baseline = BASELINES[group_code]
    for benchmark in BENCHMARK_TASKS:
        summary, delta, baseline_row = compute_group_summary_for_benchmark(
            group_code, benchmark, baseline=baseline, percentage="flores" not in benchmark
        )
        if delta is None:
            print(f"  {group_code} / {benchmark}: Empty")
            continue
        print(f"  {group_code} / {benchmark}: {len(delta)}",)

        key = f"{benchmark}_{group_code}"
        baseline_performances[key] = baseline_row.round(2)

        delta_str   = delta.to_json(orient="records", indent=None)
        baseline_str = json.dumps(baseline_row.round(2).to_dict())

        block = (
            f"{key} = {delta_str}\n"
            f"{key}_baseline = {baseline_str}\n"
        )
        full_res += "\n" + block
        (out_dir / f"{key}_delta.json").write_text(block)

full_res += f"\nbaselines = {json.dumps({k: rename_map.get(v, v) for k, v in BASELINES.items()})}\n"
(out_dir.parent/ "all_delta.py").write_text(
    full_res.replace("null", "None").replace("NaN", "None")
            .replace("true", "True").replace("false", "False")
)
print(f"\nAll results saved in {out_dir.parent}/all_delta.py")

# Dump consolidated baseline performance across all tasks
unique_baselines = {}
for group_code, baseline_name in BASELINES.items():
    for benchmark in BENCHMARK_TASKS:
        key = f"{benchmark}_{group_code}"
        if key not in baseline_performances:
            continue
        row = baseline_performances[key]
        display_name = rename_map.get(baseline_name, baseline_name)
        if display_name not in unique_baselines:
            unique_baselines[display_name] = {}
        unique_baselines[display_name][benchmark] = row.to_dict()

baseline_out = {}
for baseline_name, benchmarks in unique_baselines.items():
    baseline_out[baseline_name] = {}
    for benchmark, perf in benchmarks.items():
        baseline_out[baseline_name][benchmark] = {k: round(v, 2) for k, v in perf.items()}

print("\n=== Baseline Performance Across All Tasks ===")
for baseline_name, benchmarks in baseline_out.items():
    print(f"\n--- {baseline_name} ---")
    for benchmark, perf in benchmarks.items():
        avg = perf.get("Avg", float("nan"))
        print(f"  {benchmark:30s}  Avg={avg:.2f}   {perf}")

baseline_out_path = out_dir.parent / "baseline_performances.json"
with open(baseline_out_path, "w") as f:
    json.dump(baseline_out, f, indent=2)
print(f"\nBaseline performances saved to {baseline_out_path}")