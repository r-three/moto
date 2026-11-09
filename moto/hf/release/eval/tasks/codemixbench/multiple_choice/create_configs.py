"""
Generates one lm-eval YAML task file per CodeMixBench multiple-choice
config, plus a group YAML that lets you run all of them with a single task
name: codemixbench_mc.

Usage:
    python lm_eval/tasks/codemixbench/multiple_choice/create_configs.py

Then evaluate:
    lm_eval --model hf \
        --model_args pretrained=your-model \
        --tasks codemixbench_mc \
        --include_path ./lm_eval/tasks/codemixbench/multiple_choice \
        --device cuda \
        --output_path ./results
"""

import os
from pathlib import Path


TASKS = [
    # "gsm8k",  # generation, not multiple choice -- out of scope here
    # "lid",
    "mmlu",
    # "mt",
    # "ner",
    # "pos",
    "sa",
    "truthfulqa",
]

LANGS = {
    # not verified skip
    # "lid": [
    #     {"name": "lid_chieng", "langa": "zho_Hans", "langb": "eng_Latn"},
    #     {"name": "lid_fridut", "langa": "fry_Latn", "langb": "nld_Latn"},
    #     {"name": "lid_gereng", "langa": "deu_Latn", "langb": "eng_Latn"},
    #     {"name": "lid_guaspa", "langa": "grn_Latn", "langb": "spa_Latn"},
    #     {"name": "lid_hineng", "langa": "hin_Deva", "langb": "eng_Latn"},
    #     {"name": "lid_hokman", "langa": "hok_Latn", "langb": "eng_Latn"},
    #     {"name": "lid_mareng", "langa": "mar_Deva", "langb": "eng_Latn"},
    #     {"name": "lid_msaea", "langa": "msa_Latn", "langb": "eng_Latn"},
    #     {"name": "lid_nepeng", "langa": "nep_Deva", "langb": "eng_Latn"},
    #     {"name": "lid_spaeng", "langa": "spa_Latn", "langb": "eng_Latn"},
    # ],
    "mmlu": [
        {"name": "mmlu_araeng", "langa": "arb_Arab", "langb": "eng_Latn"},
        {"name": "mmlu_beneng", "langa": "ben_Beng", "langb": "eng_Latn"},
        {"name": "mmlu_chieng", "langa": "zho_Hans", "langb": "eng_Latn"},
        {"name": "mmlu_duteng", "langa": "nld_Latn", "langb": "eng_Latn"},
        {"name": "mmlu_freeng", "langa": "fra_Latn", "langb": "eng_Latn"},
        {"name": "mmlu_gereng", "langa": "deu_Latn", "langb": "eng_Latn"},
        {"name": "mmlu_hineng", "langa": "hin_Deva", "langb": "eng_Latn"},
        {"name": "mmlu_mareng", "langa": "mar_Deva", "langb": "eng_Latn"},
        {"name": "mmlu_nepeng", "langa": "npi_Deva", "langb": "eng_Latn"},
        {"name": "mmlu_spaeng", "langa": "spa_Latn", "langb": "eng_Latn"},
        {"name": "mmlu_tameng", "langa": "tam_Taml", "langb": "eng_Latn"},
    ],
    #     "mt": [
    # {"name": "mt_araeng_eng", "langa":"", "langb": "eng_Latn"},
    # {"name": "mt_beneng_eng", "langa":"", "langb": "eng_Latn"},
    # {"name": "mt_chieng_chi", "langa":"", "langb": "eng_Latn"},
    # {"name": "mt_chieng_eng", "langa":"", "langb": "eng_Latn"},
    # {"name": "mt_hineng_eng", "langa":"", "langb": "eng_Latn"},
    # {"name": "mt_hokman_man", "langa":"", "langb": "eng_Latn"},
    # {"name": "mt_mareng_eng", "langa":"", "langb": "eng_Latn"},
    # {"name": "mt_spaeng_eng", "langa":"", "langb": "eng_Latn"},
    #     ],
    # "ner": [{"langa": "eng_Latn", "langb": "arb_Arab"}],
    # "pos": [{"langa": "eng_Latn", "langb": "arb_Arab"}],
    "sa": [
        {"name": "sa_beneng", "langa": "ben_Beng", "langb": "eng_Latn"},
        {"name": "sa_hineng", "langa": "hin_Deva", "langb": "eng_Latn"},
        {"name": "sa_maleng", "langa": "mal_Mlym", "langb": "eng_Latn"},
        {"name": "sa_mareng", "langa": "mar_Deva", "langb": "eng_Latn"},
        {"name": "sa_nepeng", "langa": "nep_Deva", "langb": "eng_Latn"},
        {"name": "sa_spaeng", "langa": "spa_Latn", "langb": "eng_Latn"},
        {"name": "sa_tameng", "langa": "tam_Taml", "langb": "eng_Latn"},
    ],
    "truthfulqa": [
        {"name": "truthfulqa_araeng", "langa": "arb_Arab", "langb": "eng_Latn"},
        {"name": "truthfulqa_chieng", "langa": "zho_Hans", "langb": "eng_Latn"},
        {"name": "truthfulqa_hineng", "langa": "hin_Deva", "langb": "eng_Latn"},
        {"name": "truthfulqa_spaeng", "langa": "spa_Latn", "langb": "eng_Latn"},
    ],
}

"""
sa: sentence, answer, positive, negative, neutral
truthfulqa: sentence, answer (11 choices: A, B, C, D, E, F, G, H, I, J, K)
mmlu: sentence, answer (4 choices: A, B, C, D)
gsm8k: sentence, answer (generation)
"""

# mmlu / truthfulqa: the question and its lettered options are embedded
# together in `sentence` (e.g. "...\n(A): ...\n(B): ..."), and the option
# count varies per doc for truthfulqa, so choices are parsed dynamically in
# utils.py rather than hardcoded here. `answer` is the gold letter (e.g. "B"),
# which lm-eval resolves against doc_to_choice's output automatically.
MC_TEMPLATE = """\
task: codemixbench_{name}
dataset_path: {repo_id}
dataset_name: {name}
test_split: test
output_type: multiple_choice
doc_to_text: !function utils.doc_to_text
doc_to_choice: !function utils.doc_to_choice
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

# sa: plain sentence, fixed 3-way sentiment label as the gold answer.
SA_CHOICES = ["positive", "negative", "neutral"]
SA_TEMPLATE = """\
task: codemixbench_{name}
dataset_path: {repo_id}
dataset_name: {name}
test_split: test
output_type: multiple_choice
doc_to_text: "{{{{sentence}}}}"
doc_to_choice: {choices}
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

GROUP_TEMPLATE = """\
group: codemixbench_mc
task:
{task_list}
metadata:
  version: 1.0
"""


def main(repo_id: str, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)

    task_names = []
    for task in TASKS:
        for entry in LANGS[task]:
            name = entry["name"]
            task_name = f"codemixbench_{name}"
            task_names.append(task_name)
            yaml_path = out_dir / f"{task_name}.yaml"
            with open(yaml_path, "w") as f:
                if task == "sa":
                    f.write(
                        SA_TEMPLATE.format(
                            name=name, repo_id=repo_id, choices=SA_CHOICES
                        )
                    )
                else:
                    f.write(MC_TEMPLATE.format(name=name, repo_id=repo_id))
            print(f"  wrote {yaml_path}")

    # Group YAML so `--tasks codemixbench_mc` runs all subsets/languages at once
    task_list_str = "\n".join(f"  - {t}" for t in task_names)
    group_path = out_dir / "_codemixbench_mc_group.yaml"
    with open(group_path, "w") as f:
        f.write(GROUP_TEMPLATE.format(task_list=task_list_str))
    print(f"  wrote {group_path}")

    print("\nDone. Run with:")
    print("  lm_eval --model hf \\")
    print("      --model_args pretrained=<your-model> \\")
    print("      --tasks codemixbench_mc \\")
    print(f"      --include_path {out_dir} \\")
    print("      --device cuda \\")
    print("      --output_path ./results")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    repo = "CodeMixBench/CodeMixBench"
    main(repo_id=repo, out_dir=out_dir)
