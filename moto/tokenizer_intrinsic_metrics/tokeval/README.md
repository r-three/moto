# MoTo × TokEval

Intrinsic metrics for MoTo tokenizers using
[cimeister/tokenizer-intrinsic-evals](https://github.com/cimeister/tokenizer-intrinsic-evals)
(TokEval). `run_moto_tokeval.py` is a driver on top of TokEval; it does not modify it.

## Why a driver

TokEval's `encode(text)` takes no language, so MoTo's routed tokenization cannot be a plain TokEval
tokenizer class. The driver does the routing and passes finished ids to TokEval's pre-tokenized path.
`tokenizer_analysis` in the imports is TokEval's package name.

Settings (one results row each), mixable in one run:

| mode | what it is |
|---|---|
| `moto_repo` | a MoTo release repo (`r-three-moto/moto-1b` or a local dir). Loaded with the repo's own `MoToTokenizer`, so components, id maps, routing table, language aliases and superset size are read from `moto_config.json`; ids are the model's superset ids. |
| `single` | any ordinary tokenizer (joint BPE, parity-aware BPE, a merged superset `tokenizer.json`), loaded with TokEval's own wrappers (`class`, `path`). |
| `routed` | per-language tokenizers that are not in a release repo: each language is encoded by its own tokenizer and token strings are mapped into a shared id space (union of the vocabularies, or `superset_vocab`). |

Routing is oracle for all routed settings: each corpus language is encoded by the component the repo
assigns it (`eng_Latn` resolves through the repo's alias to `fw_edu`). Encoding uses
`add_special_tokens=False`, so BOS/EOS are not counted.

### Languages come from the repo

With a `moto_repo` setting you give `corpus_dir` instead of `language_config`. The evaluated languages are
parsed from the repo's routing table (FLORES+-style codes; `fw_edu` becomes `eng_Latn` via the repo's
alias) and the driver writes `<output_dir>/language_config.generated.json`. Corpus files that are missing from
`corpus_dir` are fetched from `openlanguagedata/flores_plus` (gated: `hf auth login`, needs `datasets`) with
`corpus_split=devtest` by default and `corpus_revision` to pin the dataset; a `corpus_manifest.json` records the
split, and a `corpus_dir` written for another split is refused. `languages=[...]` restricts the
set. Baseline settings are evaluated on the same languages.

Vocabulary-dependent metrics (utilization, Rényi efficiency, entropies) are measured over the repo's
`super_vocab.json` with the model's `vocab_size` as the denominator.

`default_language` (optional): TokEval's math and code corpora have no language label. With this set they
are encoded by that language's component so the digit/operator metrics exist for that setting.

## Run

```bash
hf auth login   # FLORES+ is gated; the driver fetches the devtest split into corpus_dir on first use

cd moto   # an environment where moto and TokEval (`tokenizer_analysis`, a pyproject dependency) are both importable
export MPLBACKEND=Agg
python -m moto.tokenizer_intrinsic_metrics.tokeval.run_moto_tokeval \
    config=configs/tokeval/moto_eval.yaml output_dir=results/tokeval \
    use_builtin_math_data=true save_full_results=true generate_latex_tables=true
```

Arguments follow the moto `Args`/`load_args` convention: any field of `TokEvalArgs` can be set in the YAML
or on the command line (CLI wins), e.g. `only=[moto_1b] samples_per_lang=200 no_plots=true`. Use absolute paths
for TokEval's own config files. On the cluster, point HF caches at scratch (`/path/to/.cache`).

Outputs: `analysis_results.json` (TokEval's slim schema), `summary_global.csv` (one row per setting),
`summary_per_language.csv` (fertility and compression per setting × language), plots, optional LaTeX tables.

## Parity with fixed denominators

Set `parity_denominators` (label -> setting name) and every setting's per-document token counts are divided by
the English counts of the named setting, so all rows share one denominator:

```yaml
parity_lang: eng_Latn
parity_denominators: {english_moto: moto_1b, llama: llama32}
```

`mean over documents of tokens(setting, language) / tokens(denominator setting, English)`. A setting that is
its own denominator is its own parity (Llama's row under `llama` is Llama's own parity; `moto_1b` under
`english_moto` is 1.0 on English). A `self` column (each setting's own English, the confounded version) is
always written for comparison.

Outputs: `parity_vs_<label>.csv` (rows = settings, columns = languages, plus `_std`), `parity_long.csv`
(with `n_docs` and the count of zero-denominator documents dropped) and `token_counts.csv` (setting, language,
doc, n_tokens, n_words, n_bytes) for anything else you want to compute.

Needs a line-aligned corpus (same number of documents in every language, nothing blank); the run stops with the
counts if not. Token counts exclude special tokens, as in `parity.py`. TokEval's `fetch_flores.py` writes the
`dev` split by default, whereas the driver's own fetch and `fertility.py`/`parity.py` use `devtest`. If you use
`language_config` with files from that script, fetch them with `--split devtest`.

## What is and isn't computed for routed settings

Computed: compression, fertility, token length, vocabulary utilization, Rényi efficiency, bi/trigram
entropy, Gini and cross-lingual CoV, UTF-8 integrity, and (with `default_language`) the digit and operator
metrics. Not computed: reconstruction fidelity (the superset wrapper cannot decode), code-AST metrics,
encoding speed, MorphScore. `exact_match` is blank for routed rows in the CSV for that reason.

## Check it before trusting numbers

```bash
python tests/make_toy_fixture.py /tmp/moto_toy path/to/tokenizer.py   # builds a mock release repo
python -m moto.tokenizer_intrinsic_metrics.tokeval.run_moto_tokeval config=/tmp/moto_toy/moto_eval.yaml \
    output_dir=/tmp/moto_toy/out use_builtin_math_data=true
python tests/check_routed_matches_direct.py /tmp/moto_toy
```

The check recomputes per-language compression and fertility straight from the per-language tokenizers and
compares them with TokEval's output, and checks that `moto_repo` and `routed` agree on the same tokenizers.

## Things to watch

- **Fertility denominator.** TokEval's fertility counts whitespace words, which is not meaningful for
  CJK/Thai. Compression and the information-theoretic metrics default to bytes. Set `measurement_config` for
  anything you plan to report next to words-based numbers.
- **Superset definition.** Union-by-token-string assumes the superset vocabulary is the union of the
  per-language vocabularies with identical strings sharing an id. If your model's real id space differs, pass
  it as `superset_vocab`.
- **Hub access.** `moto_repo` downloads the repo's `tokenizers/*` and `super_vocab.json` (needs a token if the repo is private).
