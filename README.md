# MoTo: Mixture of Tokenizers

**Gül Sena Altıntaş**, **Colin Raffel**
University of Toronto · Vector Institute

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Models 🤗](https://img.shields.io/badge/Models-%F0%9F%A4%97%20HuggingFace-yellow)](https://huggingface.co/r-three-moto)
[![Training ⚙️](https://img.shields.io/badge/Training-Lingua-blue)](https://github.com/r-three/lingua)

MoTo is a modular multilingual tokenization framework. Instead of one monolithic tokenizer, you train one
tokenizer per language and compose them into a shared **superset vocabulary**. Each language keeps its own
segmentation and pre-tokenization choices, while the model sees a single embedding space and an otherwise
unchanged architecture (only the tokenizer and embedding layer differ).

This repository contains the code behind the paper: per-language tokenizer training, superset construction and
reindexing, Lingua-based model utilities, tokenizer-intrinsic metrics, synthetic tasks, and tooling to collapse a MoTo
tokenizer back into a single BPE tokenizer.

## Installation

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/r-three/moto
cd moto

uv venv --python 3.13
source .venv/bin/activate
uv sync

# optional extras
uv sync --extra lingua --no-build-isolation      # xformers, for the Lingua model code
uv sync --extra conversion --inexact             # conversion / reindexing scripts (blobfile)
```

`torch` and `xformers` are pinned to the `cu126` PyTorch index in `pyproject.toml`; change the index there if you
use a different CUDA version. Evaluation uses a fork of `lm-evaluation-harness` (shipping the `toksuite_*` tasks) and
[TokEval](https://github.com/cimeister/tokenizer-intrinsic-evals); both are installed by `uv sync` as pinned git
dependencies.

## Pipeline at a glance

1. **Train per-language tokenizers**: `moto/tokenizer_training/`
2. **Build the superset vocabulary** and reindex each tokenizer into it: `moto/supertokenizer/`
3. **Create an initial model checkpoint** for the superset size: `moto/supertokenizer/create_checkpoints.py`
4. **Pretrain** with [Lingua](https://github.com/r-three/lingua) using the model code in `moto/lingua/` and configs in `configs/lingua/`
5. **Evaluate**: intrinsic tokenizer metrics (`moto/tokenizer_intrinsic_metrics/`) and downstream tasks via `lm-eval` ([Evaluating a model](#evaluating-a-model)), then **publish** ([Publishing tokenizers](#publishing-tokenizers-in-one-hugging-face-repo))

### 1. Train per-language tokenizers

Data layout: `{data_dir}/{lang}/*.jsonl`, each line a JSON object with a `"text"` field.

```bash
python -m moto.tokenizer_training.train_tokenizers \
    algorithm=bpe vocab_size=32000 langs='[ita_Latn,tur_Latn]' \
    data_dir=/path/to/fineweb2_hq output_dir=./trained_tokenizers
```

See the `train_*.sh` launch scripts in that folder. `regularized_bpe.py` implements a
regularized BPE trainer (segmentation / token-length / vocabulary penalties; standard BPE is recovered with
`λ_seg=1, λ_tok=0, λ_voc=0`).

### 2. Build the superset vocabulary

`moto/supertokenizer/supervocab.py` merges the vocabularies of the participating tokenizers into one shared
vocabulary; `reindex_tokenizers.py` maps each tokenizer into it (see `configs/tokenizers/reindex_tokenizers.yaml`).
The `create_supervocab_for_*.sh` scripts are the exact invocations used for the experiments in the paper.

### 3. Initial checkpoint

`create_checkpoints.py` builds a checkpoint for a given tokenizer on top of the superset vocabulary. It needs a Llama
base model with the superset vocabulary size, created beforehand with Lingua.

### 4. Pretraining

`moto/lingua/` holds the MoTo-specific model components that plug into Lingua (it does not re-implement the
trainer). Training configs live in `configs/lingua/` (300M, 1B, 7B). The main model variant is
`LlamaAlbertForCausalLM` (ALBERT-style factorized embeddings), which keeps the embedding parameter count manageable as
the superset vocabulary grows. See [`moto/lingua/README.md`](moto/lingua/README.md).

### 5. Evaluation

```bash
# intrinsic metrics for MoTo (routed) tokenizers, built on TokEval
python moto/tokenizer_intrinsic_metrics/tokeval/run_moto_tokeval.py   # config: configs/tokeval/moto_eval.yaml

# MorphScore
python -m moto.tokenizer_intrinsic_metrics.morphscore
```

See [`moto/tokenizer_intrinsic_metrics/tokeval/README.md`](moto/tokenizer_intrinsic_metrics/tokeval/README.md)
for the settings (`moto_repo`, `single`, `routed`) and oracle-routing details. Fertility, parity, Gini, Rényi
efficiency, PCW and vocabulary-overlap metrics are in the same package.

## Evaluating a model

Downstream evaluation uses `lm-eval` (the pinned r-three fork, installed by `uv sync`) with a custom model, `moto`,
registered by `moto/hf/release/eval`. It reproduces the paper's protocol on top of `HFLM`, so MoTo models and
single-tokenizer baselines are scored the same way.

```bash
python -m moto.hf.release.eval --model moto \
    --model_args pretrained=r-three-moto/moto-1b \
    --tasks paper+toksuite --batch_size 16 --output_path /path/to/evals
```

`python -m moto.hf.release.eval` is lm-eval's CLI with the `moto` model and the bundled tasks (`eval/tasks/`:
FLORES+ bits-per-byte and CodeMixBench) added to the include path.

**Task sets** (`moto/hf/release/eval/paper_tasks.yaml`; sets and single tasks can be mixed with `,` or `+`):

| `--tasks` | What it runs |
|---|---|
| `paper` | The paper's multilingual and English suite (XNLI, XCOPA, XStoryCloze, PAWS-X, XWinograd, Okapi HellaSwag, ARC, PIQA, BoolQ, CommonsenseQA, FLORES+ bits per byte, ...) |
| `toksuite` | The `toksuite_*` tasks (English, Chinese, Turkish, Italian, Farsi, math, STEM) |
| `multiblimp_moto` | MultiBLiMP for the MoTo training languages it covers |
| `code_switching` | The code-switching evals |

**What the `moto` model does differently from plain `HFLM`:**

- **BOS.** Every scored sequence starts with BOS.
- **Log-likelihood.** The continuation is scored given `[bos] + enc(context) + enc(continuation)`, with context and
  continuation encoded separately (no whitespace moved between them), as in Lingua's evaluation. Rolling log-likelihood
  scores `[bos] + enc(text)`.
- **Routing (MoTo models only).** Each request is routed to one of the model's tokenizers by *task name*, using a
  `task_to_tokenizer` table (`eval/routing/<repo>.yaml`, chosen automatically from the repo name): exact keys first,
  then `regex:` keys. A task missing from the table gets a uniformly drawn tokenizer (seed 42), and the log lists
  those tasks. Requests are batched per tokenizer.
- **Baselines** use the same encoding with no routing.

Useful `model_args` (in addition to any `HFLM` argument such as `batch_size`, `dtype`, `max_length`):

| Argument | Meaning |
|---|---|
| `routing=<name \| path \| none>` | Bundled routing table, a YAML with `task_to_tokenizer` / `suitable_tokenizer_probability`, or `none`. Default: the one matching the repo name. |
| `language=fra_Latn` | Use one fixed tokenizer for every task instead of routing. |
| `add_bos=false` | Do not prepend BOS (default `true`). |
| `lingua_pair_encoding=false` | Use `HFLM`'s joint context+continuation encoding. |
| `fallback=random\|default\|error` | What to do for a task with no routing entry. Default `random`. |
| `debug_tokens=N` | Log the exact ids and tokens of the first N requests per task, to check BOS and pair encoding. |

**Offline cluster nodes.** Prefetch weights, tokenizer code and task datasets on a node with internet, then run jobs
with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` and the same `HF_HOME`:

```bash
python -m moto.hf.release.eval.prefetch --tasks paper+toksuite --models r-three-moto/moto-1b
```

**Resuming and comparing.** `python -m moto.hf.release.eval.missing --dir /path/to/evals/moto-1b --tasks paper+toksuite`
prints the tasks that still have no result, so only those need rerunning. Running with `--log_samples` writes per-document
logs, and `moto/hf/release/eval/paired.py` uses them to compute paired standard errors of accuracy differences
between two models, which are tighter than independent errors when both models get the same documents right or wrong.

## Publishing tokenizers in one Hugging Face repo

A MoTo model ships as **one repo** containing the model weights, one subfolder per language tokenizer, and a small
amount of glue code. Users load it with a single call and pick the language tokenizer at load time:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("r-three-moto/moto-1b", trust_remote_code=True, language="fra_Latn")
tok.set_language("tur_Latn")            # switch component; also: with tok.using("tur_Latn"): ...
ids = tok("Merhaba dünya")["input_ids"]  # [bos] + superset ids
```

**Repo layout**

```
<repo>/
  config.json, generation_config.json, model*.safetensors   BOS/EOS ids are superset ids
  moto_config.json        vocab size, special ids, routing table, and the list of component tokenizers
  tokenizer_config.json   points AutoTokenizer at tokenizer.py
  tokenizer.py            MoToTokenizer: a copy of moto/hf/release/remote/tokenizer.py
  super_vocab.json        the superset vocabulary used in training (optional but recommended)
  tokenizers/
    fra_Latn/
      tokenizer.json            the language's ORIGINAL tokenizer, unmodified
      super_mapping.json        {original id: superset id}
    tur_Latn/
      ...
```

Each component is shipped **unmodified**, together with its id map. `MoToTokenizer` rebuilds the superset tokenizer in
memory at load time (ids remapped through `super_mapping.json`, unmapped entries dropped, superset BOS/EOS registered),
so nothing has to be re-exported when the vocabulary changes. Only `transformers`, `tokenizers` and `huggingface_hub` are needed.
If your model uses the factorized-embedding Llama variant, also ship `moto/hf/release/remote/modeling_llama_albert.py` and
set `auto_map` in `config.json` accordingly.

**Steps**

1. **Train the per-language tokenizers and the superset** (`moto/tokenizer_training/`, `moto/supertokenizer/`). The superset
   step writes `super_vocab.json` and, per tokenizer, a `<owner>--<name>_super_mapping.json` mapping each tokenizer's
   ids to superset ids. Copy each into `tokenizers/<lang>/super_mapping.json` next to that tokenizer's `tokenizer.json`.
2. **Write `tokenizer_config.json`:**

   ```json
   {
     "tokenizer_class": "MoToTokenizer",
     "auto_map": {"AutoTokenizer": [null, "tokenizer.MoToTokenizer"]},
     "language": "fra_Latn",
     "model_max_length": 4096,
     "padding_side": "left"
   }
   ```

3. **Write `moto_config.json`.** `MoToTokenizer` reads these fields:

   ```json
   {
     "vocab_size": 167000,
     "bos_token_id": 0,
     "eos_token_id": 1,
     "eos_token": "</s>",
     "add_bos": true,
     "default_tokenizer": "fra_Latn",
     "routing": {
       "mode": "oracle",
       "source_to_tokenizer": {"fra_Latn": "fra_Latn", "tur_Latn": "tur_Latn", "fw_edu": "eng_Latn"},
       "language_aliases": {"eng_Latn": "eng_Latn"}
     },
     "tokenizers": [
       {"name": "fra_Latn", "subfolder": "tokenizers/fra_Latn", "languages": ["fra_Latn"], "id_map": "super_mapping.json"},
       {"name": "tur_Latn", "subfolder": "tokenizers/tur_Latn", "languages": ["tur_Latn"], "id_map": "super_mapping.json"}
     ]
   }
   ```

   The ids and sizes above are placeholders; use your own. `routing.mode` is `oracle` (languages routed to their own tokenizer) or `random`, and is
   informational for the tokenizer class. `vocab_size` is the superset size (the model's embedding rows), `source_to_tokenizer` maps a training source or
   language code to a component `name`, and `language_aliases` lets users pass codes such as `cmn_Hans` or `pes_Arab`.
   Extra fields (source repo, revision, `tokenizer_sha256`) are optional provenance.
4. **Copy the glue and upload**, with a token that has write access to the org:

   ```python
   import shutil
   from huggingface_hub import HfApi

   shutil.copy("moto/hf/release/remote/tokenizer.py", "my_repo/tokenizer.py")

   api = HfApi()
   api.create_repo("my-org/my-model", private=True, exist_ok=True)
   api.upload_folder(folder_path="my_repo", repo_id="my-org/my-model", commit_message="Initial upload")
   ```

5. **Check before making it public.** Load it with `trust_remote_code=True` and confirm that, for a few sentences in each
   language, `tok(text)["input_ids"]` equals `[bos]` plus the ids your training pipeline produced for that language's
   tokenizer. That equality is what guarantees the published tokenizers match what the model was trained on.

Single-vocabulary baselines (e.g. a de-MoTo-ed merged BPE) are ordinary Hugging Face tokenizer repos: pass
`hf.publish_to_hf=true hf.hf_repo_id=<org>/<name>` to `python -m moto.moto_to_monolithic.demotoing`, which uploads them
(and can add them to a collection with `hf.collections=[...]`). The training and superset scripts under
`moto/tokenizer_training/` and `moto/supertokenizer/` accept the same `hf.*` options.

## Other components

| Path | What it is |
|---|---|
| `moto/llama_moto/` | End-to-end "Llama-MoTo" pipeline: extends a Llama tokenizer with per-language merges (PickyBPE, merge extension) and builds the MoTo superset. `python -m moto.llama_moto.pipeline`, defaults in `configs/llama_moto/base.yaml`. |
| `moto/moto_to_monolithic/` | "De-MoTo-ing": merges N per-language BPE tokenizers into a single monolithic BPE tokenizer (`demotoing.py`), plus analysis helpers. |
| `moto/simplified/` | Controlled synthetic tasks (multilingual addition, modulo) with their tokenizer and data generation scripts. |
| `moto/hf/release/` | `eval/`: the `moto` lm-eval model, task sets, routing tables and bundled tasks. `remote/`: `MoToTokenizer` (shipped as `tokenizer.py` in model repos) and the factorized-embedding Llama modeling file. |
| `moto/log_utils/` | Plot and table helpers shared by the analysis scripts. |
| `paper/` | Scripts that build the paper's tables and figures (`per_lang_summary.py`, `plot_deltas.py`, `plot_release.py`, ...), the result modules they read (`all_delta*.py`), and the pipeline diagrams in `paper/figures/`. Generated figures are written under `paper/figures/<name>/` and are not tracked. Run from the repo root, e.g. `PYTHONPATH=. python paper/per_lang_summary.py`. |

## Paths and cluster setup

Scripts and configs use `/path/to/...` placeholders for data, output and cache locations; many `.sh` launchers also
read cluster environment variables such as `$PROJECT`. Adjust these for your environment before running.

## Citation

```bibtex
@inproceedings{altintas2026moto,
  title     = {{MoTo}: Mixture of Tokenizers --- Towards Fair Multilingual Language Modeling},
  author    = {Alt{\i}nta\c{s}, G\"ul Sena and Raffel, Colin},
  year      = {2026},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```
