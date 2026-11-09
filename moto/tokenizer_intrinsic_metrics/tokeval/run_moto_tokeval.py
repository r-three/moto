#!/usr/bin/env python
"""Evaluate MoTo tokenizers with TokEval (cimeister/tokenizer-intrinsic-evals).

TokEval's `encode(text)` has no notion of language, so MoTo's routed
tokenization cannot be expressed as a plain TokEval tokenizer class. This driver
does the routing itself and hands TokEval the finished token ids through its
pre-tokenized path (`create_analyzer_from_tokenized_data`).

Each entry under `settings` in the config is one row in the results:

  mode "moto_repo"  a MoTo release repo (r-three-moto/moto-1b or a local copy). Everything is
                 read from the repo: components, routing table, language aliases, superset
                 size and super_vocab.json. If no language config is given the evaluated
                 languages are parsed from its routing table too. Needs `transformers`.

  mode "single"  one ordinary tokenizer (joint BPE baseline, parity-aware BPE,
                 a merged superset tokenizer.json, ...). Loaded with TokEval's own
                 wrappers, so `class` and `path` mean what they do in TokEval.

  mode "routed"  oracle routing: each language is encoded by its own tokenizer
                 and token strings are mapped into one shared superset id space.
                 The id space is the union of all per-language vocabularies by
                 token string, or the vocabulary of `superset_vocab` if given.
                 Metrics that depend on the id space (vocabulary utilization,
                 Renyi efficiency, entropies) are then measured over the
                 superset, which is the vocabulary the model actually has.

Parity (optional): `parity_denominators={english_moto: moto_1b, llama: llama32}` divides every
setting's per-document token counts by the English counts of the named setting, so all rows share
one denominator. Writes parity_long.csv, parity_vs_<label>.csv and token_counts.csv.

Usage (moto Args/load_args convention; needs TokEval importable, i.e.
`pip install git+https://github.com/cimeister/tokenizer-intrinsic-evals.git` or
run inside a TokEval checkout's environment):

  python -m moto.tokenizer_intrinsic_metrics.tokeval.run_moto_tokeval \\
      config=configs/tokeval/moto_eval.yaml output_dir=results/tokeval \\
      use_builtin_math_data=true

Any field of TokEvalArgs can be given on the command line or in the YAML file
(CLI overrides the file), e.g. `only=[joint_bpe,moto_oracle] samples_per_lang=200`.
"""
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tokenizer_analysis import create_analyzer_from_tokenized_data
from tokenizer_analysis.cli.run_analysis import (
    _convert_for_json_public,
    normalize_results,
    select_results,
)
from tokenizer_analysis.config import TextMeasurementConfig
from tokenizer_analysis.config.language_metadata import LanguageMetadata
from tokenizer_analysis.core.input_types import TokenizedData
from tokenizer_analysis.core.tokenizer_wrapper import (
    PreTokenizedDataTokenizer,
    TokenizerWrapper,
    create_tokenizer_wrapper,
)
from tokenizer_analysis.loaders.multilingual_data import load_multilingual_data

from moto.utils.args import Args, load_args

logger = logging.getLogger("moto_tokeval")


FLORES_DATASET = "openlanguagedata/flores_plus"


def fetch_flores(candidates: Dict[str, List[str]], out_dir: Path, split: str, revision: Optional[str] = None):
    """Write <code>.txt (one sentence per line) from the gated Hub dataset, one file per key of `candidates`.

    `candidates` maps a key (a routing source) to FLORES+ codes in order of preference; the first one the
    dataset has is used (so cmn_Hani -> cmn_Hans, fas_Arab -> pes_Arab). Same layout as TokEval's
    scripts/fetch_flores.py, but the split is a parameter that defaults to devtest here (that script writes
    `dev`). Returns ({key: chosen code}, [keys the dataset has no candidate for]).
    Needs `datasets` and a Hub token with access to openlanguagedata/flores_plus.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Fetching FLORES+ needs the `datasets` package (pip install datasets).")
    logger.info("Fetching %s split=%s for %d languages into %s (gated: needs `hf auth login`)",
                FLORES_DATASET, split, len(candidates), out_dir)
    ds = load_dataset(FLORES_DATASET, split=split, revision=revision)
    wanted = {c for cs in candidates.values() for c in cs}
    by_lang: Dict[str, List[str]] = {}
    for row in ds:
        code = f"{row['iso_639_3']}_{row['iso_15924']}"
        if code in wanted:
            by_lang.setdefault(code, []).append(row["text"])
    chosen: Dict[str, str] = {}
    for key, cands in candidates.items():
        have = [c for c in cands if c in by_lang]
        if have:
            chosen[key] = have[0]
            if len(have) > 1:
                logger.warning("%s: dataset has several of %s; using %s", key, cands, have[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in set(chosen.values()):
        (out_dir / f"{code}.txt").write_text("\n".join(by_lang[code]) + "\n", encoding="utf-8")
    manifest_path = out_dir / "corpus_manifest.json"
    prev = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    prev.update({"dataset": FLORES_DATASET, "split": split, "revision_requested": revision})
    manifest_path.write_text(json.dumps(prev, indent=2) + "\n")
    return chosen, [k for k in candidates if k not in chosen]


def check_corpus_split(corpus_dir: Path, split: str):
    """A corpus_dir written for another split would silently change every number."""
    path = corpus_dir / "corpus_manifest.json"
    if path.exists():
        have = json.loads(path.read_text()).get("split")
        if have and have != split:
            raise SystemExit(f"{corpus_dir} holds FLORES+ split={have!r} (corpus_manifest.json) but corpus_split={split!r}. "
                             "Use another corpus_dir, or delete these files so they are fetched again.")


class SupersetVocabTokenizer(PreTokenizedDataTokenizer):
    """Vocabulary-only wrapper for a routed setting.

    Same as TokEval's PreTokenizedDataTokenizer, but it reports the special
    tokens of the per-language tokenizers instead of None, so TokEval does not
    fall back to its generic special-token list.
    """

    def __init__(self, name: str, vocab: Dict[str, int], special: Set[str],
                 encode_fn=None, vocab_size: Optional[int] = None):
        super().__init__(name, max(vocab_size or 0, max(vocab.values()) + 1), vocab)
        self._special = set(special)
        # Optional: encode language-free text (TokEval's math and code corpora,
        # which carry no language label) with one fixed per-language tokenizer.
        # encode_fn(text) -> (superset ids, offsets or None)
        self._encode_fn = encode_fn

    def get_special_token_strings(self) -> Optional[Set[str]]:
        return set(self._special)

    def can_encode(self) -> bool:
        return self._encode_fn is not None

    def encode(self, text: str) -> List[int]:
        return self.encode_with_offsets(text)[0]

    def encode_with_offsets(self, text: str):
        if self._encode_fn is None:
            raise NotImplementedError("routed setting has no default_language to encode with")
        return self._encode_fn(text)


def _encode_texts(wrapper: TokenizerWrapper, texts: List[str]):
    return wrapper.encode_batch_with_offsets(texts)


def _tokenize_single(name: str, cfg: Dict[str, Any], language_texts):
    wrapper = create_tokenizer_wrapper(name, cfg)
    data: List[TokenizedData] = []
    for lang, texts in language_texts.items():
        texts = [t for t in texts if t.strip()]
        for text, (ids, offsets) in zip(texts, _encode_texts(wrapper, texts)):
            data.append(TokenizedData(
                tokenizer_name=name, language=lang, tokens=list(ids), text=text,
                offsets=offsets, metadata={"source": "single"}))
    return data, wrapper


def _resolve_routed_tokenizers(cfg: Dict[str, Any], languages: List[str]):
    explicit = dict(cfg.get("tokenizers", {}))
    template = cfg.get("tokenizer_template")
    out: Dict[str, Dict[str, Any]] = {}
    for lang in languages:
        if lang in explicit:
            entry = explicit[lang]
            out[lang] = {"class": "huggingface", "path": entry} if isinstance(entry, str) else entry
        elif template:
            out[lang] = {"class": cfg.get("class", "huggingface"),
                         "path": template.format(lang=lang)}
    missing = [l for l in languages if l not in out]
    if missing:
        raise SystemExit(
            f"routed setting has no tokenizer for: {missing}. Add them under "
            "'tokenizers', give a 'tokenizer_template', or drop those languages "
            "from the language config.")
    return out


def _tokenize_routed(name: str, cfg: Dict[str, Any], language_texts):
    langs = list(language_texts)
    per_lang_cfg = _resolve_routed_tokenizers(cfg, langs)

    # One wrapper per distinct tokenizer path (several languages may share one).
    by_path: Dict[str, TokenizerWrapper] = {}
    per_lang: Dict[str, TokenizerWrapper] = {}
    for lang, tcfg in per_lang_cfg.items():
        key = json.dumps(tcfg, sort_keys=True)
        if key not in by_path:
            logger.info("[%s] loading tokenizer for %s: %s", name, lang, tcfg.get("path"))
            by_path[key] = create_tokenizer_wrapper(f"{name}:{lang}", tcfg)
        per_lang[lang] = by_path[key]

    special: Set[str] = set()
    for w in by_path.values():
        special |= (w.get_special_token_strings() or set())

    if cfg.get("superset_vocab"):
        sup = create_tokenizer_wrapper(f"{name}:superset", cfg["superset_vocab"])
        vocab = dict(sup.get_vocab())
        special |= (sup.get_special_token_strings() or set())
        origin = "superset_vocab"
    else:
        vocab = {}
        for w in by_path.values():
            for tok in sorted(w.get_vocab(), key=w.get_vocab().get):
                vocab.setdefault(tok, len(vocab))
        origin = "union of per-language vocabularies"
    logger.info("[%s] superset id space (%s): %d types", name, origin, len(vocab))

    data: List[TokenizedData] = []
    missing: Dict[str, int] = {}
    for lang, texts in language_texts.items():
        w = per_lang[lang]
        texts = [t for t in texts if t.strip()]
        for text, (ids, offsets) in zip(texts, _encode_texts(w, texts)):
            strings = w.convert_ids_to_tokens(list(ids))
            new_ids = []
            for s in strings:
                if s in vocab:
                    new_ids.append(vocab[s])
                else:
                    missing[s] = missing.get(s, 0) + 1
            if len(new_ids) != len(ids):
                continue
            data.append(TokenizedData(
                tokenizer_name=name, language=lang, tokens=new_ids, text=text,
                offsets=offsets, metadata={"source": "moto_routed", "router": "oracle"}))
    if missing:
        ex = sorted(missing, key=missing.get, reverse=True)[:10]
        raise SystemExit(
            f"[{name}] {len(missing)} token strings from per-language tokenizers are "
            f"not in the superset vocabulary (e.g. {ex}). The superset must contain every "
            "per-language token; check that superset_vocab matches these tokenizers "
            "(byte-level vs. rendered token strings are a common mismatch).")
    encoder = None
    default_lang = cfg.get("default_language")
    if default_lang:
        if default_lang not in per_lang:
            raise SystemExit(f"[{name}] default_language {default_lang!r} is not one of the corpus languages {langs}")
        w = per_lang[default_lang]

        def encoder(text, w=w):
            ids, offsets = w.encode_with_offsets(text)
            return [vocab[t] for t in w.convert_ids_to_tokens(list(ids))], offsets
        logger.info("[%s] math/code corpora (no language label) will be routed to %s", name, default_lang)
    return data, SupersetVocabTokenizer(name, vocab, special, encoder)


def _load_moto_tokenizer(repo: str, revision: Optional[str] = None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
    tok.model_max_length = int(1e9)  # intrinsic metrics encode whole documents; silence the length warning
    return tok


# Routing sources whose FLORES+ code differs (also in release repos' moto_config.json language_aliases,
# repeated here so an older release without them still works).
FLORES_FALLBACK = {"cmn_Hani": ["cmn_Hans"], "fas_Arab": ["pes_Arab"]}
_FLORES_CODE = None


def moto_repo_languages(repo: str, revision: Optional[str] = None) -> Dict[str, List[str]]:
    """{routing source -> candidate FLORES+ codes, best first} for a MoTo release repo (moto_config.json).

    The source itself comes first when it looks like a FLORES+ code (arb_Arab), then the repo's aliases
    that resolve to the same component (cmn_Hans, pes_Arab, ...), then FLORES_FALLBACK. The English
    source fw_edu has no code of its own and takes its alias (eng_Latn). Which candidate exists is
    decided against the dataset or the files in corpus_dir.
    """
    import re
    pat = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}$")
    tok = _load_moto_tokenizer(repo, revision)
    routing = tok.moto_config.get("routing", {})
    aliases = routing.get("language_aliases", {})
    out: Dict[str, List[str]] = {}
    for source in routing.get("source_to_tokenizer", {}):
        cands = [source] if pat.match(source) else []
        comp = tok.resolve_language(source)
        cands += [a for a in aliases if pat.match(a) and a not in cands and tok.resolve_language(a) == comp]
        cands += [a for a in FLORES_FALLBACK.get(source, []) if a not in cands]
        if not cands:
            raise SystemExit(f"{repo}: routing source {source!r} has no FLORES+-style code or alias in moto_config.json; "
                             "pass languages=[...] and language_config explicitly.")
        out[source] = cands
    return out


def _tokenize_moto_repo(name: str, cfg: Dict[str, Any], language_texts):
    """A MoTo release repo (e.g. r-three-moto/moto-1b) with oracle routing.

    The repo's own MoToTokenizer (trust_remote_code) builds each component as the
    superset tokenizer used in training, so ids are already the model's superset ids and
    no string mapping is needed. Each corpus language is encoded by the component the repo's
    routing table (and aliases) assigns it, with add_special_tokens=False (no BOS/EOS counted).
    """
    repo, rev = cfg["repo"], cfg.get("revision")
    tok = _load_moto_tokenizer(repo, rev)
    moto_cfg = tok.moto_config
    superset = int(moto_cfg["vocab_size"])
    for lang in language_texts:  # FLORES+ codes (cmn_Hans, pes_Arab) that an older repo has no alias for
        try:
            tok.resolve_language(lang)
        except ValueError:
            for src, cs in FLORES_FALLBACK.items():
                if lang in cs:
                    tok.set_mapping({lang: src})
    try:
        resolved = {l: tok.resolve_language(l) for l in language_texts}
    except ValueError as e:
        raise SystemExit(f"[{name}] {e}")
    logger.info("[%s] %s: %d components, superset %d; routing %s", name, repo, len(tok.languages), superset, resolved)

    data: List[TokenizedData] = []
    special: Set[str] = set()
    for lang, texts in language_texts.items():
        texts = [t for t in texts if t.strip()]
        tok.set_language(lang)
        special |= {t for t in (tok.bos_token, tok.eos_token, *tok.all_special_tokens) if t}
        enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        for text, ids, off in zip(texts, enc["input_ids"], enc["offset_mapping"]):
            data.append(TokenizedData(
                tokenizer_name=name, language=lang, tokens=list(ids), text=text,
                offsets=[tuple(o) for o in off],
                metadata={"source": "moto_repo", "router": "oracle", "component": resolved[lang]}))

    vocab = _superset_surfaces(tok, repo, rev)
    encoder = None
    default_lang = cfg.get("default_language")
    if default_lang:
        tok.resolve_language(default_lang)

        def encoder(text, d=default_lang):
            tok.set_language(d)
            e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            return list(e["input_ids"]), [tuple(o) for o in e["offset_mapping"]]
    return data, SupersetVocabTokenizer(name, vocab, special, encoder, vocab_size=superset)


def _superset_surfaces(tok, repo: str, revision: Optional[str]) -> Dict[str, int]:
    """surface -> superset id. super_vocab.json from the repo when it is there, else the union of components."""
    import os
    from pathlib import Path
    try:
        if os.path.isdir(repo):
            path = Path(repo) / "super_vocab.json"
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo, "super_vocab.json", revision=revision))
        return {k: int(v) for k, v in json.loads(path.read_text()).items()}
    except Exception as e:  # not shipped, offline, ...
        logger.warning("super_vocab.json unavailable for %s (%s); using the union of component vocabularies", repo, e)
        vocab: Dict[str, int] = {}
        for comp in tok.languages:
            for t, i in tok._backend(comp).get_vocab(with_added_tokens=True).items():
                vocab.setdefault(t, i)
        return vocab


def _get(d: Any, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


HEADLINES = [
    ("compression_rate", ("compression_rate", "per_tokenizer", "{t}", "global", "compression_rate")),
    ("fertility", ("fertility", "per_tokenizer", "{t}", "global", "mean")),
    ("vocab_utilization", ("vocabulary_utilization", "per_tokenizer", "{t}", "global", "utilization")),
    ("renyi_2.5", ("renyi_efficiency", "per_tokenizer", "{t}", "global", "renyi_2.5")),
    ("bigram_entropy", ("bigram_entropy", "per_tokenizer", "{t}", "global", "bigram_entropy")),
    ("gini", ("tokenizer_fairness_gini", "per_tokenizer", "{t}", "global", "gini_coefficient")),
    ("cov_vocab_util", ("vocabulary_utilization", "per_tokenizer", "{t}", "per_language_cov")),
    ("exact_match", ("reconstruction_fidelity", "per_tokenizer", "{t}", "global", "exact_match_rate")),
    ("utf8_complete", ("utf8_token_integrity", "per_tokenizer", "{t}", "global", "completeness_rate")),
]
PER_LANG = [
    ("fertility", ("fertility", "per_tokenizer", "{t}", "per_language", "{l}", "mean")),
    ("compression_rate", ("compression_rate", "per_tokenizer", "{t}", "per_language", "{l}", "compression_rate")),
]


def _fill(path, t, l=None):
    return tuple(p.replace("{t}", t).replace("{l}", l or "") for p in path)


def token_count_rows(tokenized: Dict[str, List[TokenizedData]]):
    """One row per (setting, language, document): token, word and byte counts.

    Documents keep the corpus order within a language, so row k of every language is the same
    parallel sentence when the corpus is line-aligned (FLORES+).
    """
    rows = []
    for setting, items in tokenized.items():
        seen: Dict[str, int] = {}
        for d in items:
            k = seen.get(d.language, 0)
            seen[d.language] = k + 1
            rows.append((setting, d.language, k, len(d.tokens), len((d.text or "").split()), len((d.text or "").encode("utf-8"))))
    return rows


def check_aligned(language_texts: Dict[str, List[str]]) -> int:
    """Parity needs parallel documents: same count in every language and no blank documents."""
    counts = {l: len(t) for l, t in language_texts.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"Parity needs a line-aligned corpus, but the languages differ in document count: {counts}. "
                         "TokEval drops documents under 5 characters, which breaks the pairing; fix the files or drop parity_denominators.")
    blank = {l: sum(1 for x in t if not x.strip()) for l, t in language_texts.items()}
    if any(blank.values()):
        raise SystemExit(f"Parity needs a line-aligned corpus but some documents are blank: { {l: n for l, n in blank.items() if n} }")
    return next(iter(counts.values()))


def parity_rows(counts, denominators: Dict[str, str], english: str, own: bool = True):
    """Mean (and std) over documents of tokens(setting, language) / tokens(denominator setting, english).

    counts[setting][language] is the list of per-document token counts. `denominators` maps a label
    to the setting whose `english` counts are the fixed denominator, so every row is divided by the
    same numbers. With own=True a "self" column divides by the row's own English counts (the
    confounded version, for comparison). A denominator setting's own row is its own parity.
    """
    import numpy as np
    out = []
    labels = dict(denominators)
    for setting, per_lang in counts.items():
        for label, den_setting in ([("self", setting)] if own else []) + list(labels.items()):
            den = counts.get(den_setting, {}).get(english)
            if den is None:
                raise SystemExit(f"parity denominator {label!r}: setting {den_setting!r} has no {english} documents")
            for lang, num in per_lang.items():
                if len(num) != len(den):
                    raise SystemExit(f"parity: {setting}/{lang} has {len(num)} docs but {den_setting}/{english} has {len(den)}")
                r = [n / d for n, d in zip(num, den) if d > 0]
                out.append((setting, lang, label, den_setting if label != "self" else setting,
                            float(np.mean(r)) if r else None, float(np.std(r)) if r else None, len(r), len(num) - len(r)))
    return out


def write_counts_and_parity(tokenized, args, settings, out: Path):
    rows = token_count_rows(tokenized)
    with open(out / "token_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "language", "doc", "n_tokens", "n_words", "n_bytes"])
        w.writerows(rows)
    if not args.parity_denominators:
        return
    bad = [v for v in args.parity_denominators.values() if v not in settings]
    if bad:
        raise SystemExit(f"parity_denominators refer to settings that are not in this run: {bad}. Have {list(settings)}")
    counts: Dict[str, Dict[str, List[int]]] = {}
    for setting, lang, k, n, _, _ in rows:
        counts.setdefault(setting, {}).setdefault(lang, []).append(n)
    prows = parity_rows(counts, args.parity_denominators, args.parity_lang)
    with open(out / "parity_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "language", "denominator", "denominator_setting", "mean_parity", "std_parity", "n_docs", "n_zero_denominator_docs"])
        w.writerows(prows)
    for label in ["self", *args.parity_denominators]:
        langs = sorted({r[1] for r in prows})
        table = {(r[0], r[1]): r for r in prows if r[2] == label}
        with open(out / f"parity_vs_{label}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["setting"] + langs + [f"{l}_std" for l in langs])
            for setting in settings:
                w.writerow([setting] + [table[(setting, l)][4] if (setting, l) in table else None for l in langs]
                           + [table[(setting, l)][5] if (setting, l) in table else None for l in langs])


def write_summaries(slim: Dict, tokenizers: List[str], languages: List[str], out: Path):
    with open(out / "summary_global.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer"] + [h for h, _ in HEADLINES])
        for t in tokenizers:
            w.writerow([t] + [_get(slim, *_fill(p, t)) for _, p in HEADLINES])
    with open(out / "summary_per_language.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer", "language"] + [h for h, _ in PER_LANG])
        for t in tokenizers:
            for l in languages:
                w.writerow([t, l] + [_get(slim, *_fill(p, t, l)) for _, p in PER_LANG])


@dataclass
class TokEvalArgs(Args):
    """CLI/YAML arguments. `settings` maps a row name to its setting (see module docstring)."""

    config: Any = None  # reserved by load_args: YAML file(s) merged under CLI overrides
    language_config: str = ""  # TokEval language config JSON; empty = derive from the first moto_repo setting
    corpus_dir: str = ""  # with no language_config: directory holding <code>.txt files; missing ones are fetched
    corpus_split: str = "devtest"  # FLORES+ split to fetch (TokEval's own fetch_flores.py defaults to dev)
    corpus_revision: Optional[str] = None  # pin the dataset revision for the fetch
    languages: List[str] = field(default_factory=list)  # restrict the evaluated languages (corpus codes)
    measurement_config: Optional[str] = None  # TokEval text-measurement JSON
    settings: Dict[str, Any] = field(default_factory=dict)
    only: List[str] = field(default_factory=list)  # run only these setting names
    samples_per_lang: int = 2000
    output_dir: str = "results/tokeval"
    use_builtin_math_data: bool = False  # digit metrics on TokEval's bundled math texts
    math_data: Optional[str] = None  # or your own math texts (.txt/.json)
    code_ast_config: Optional[str] = None  # JSON: programming language -> code path
    parity_denominators: Dict[str, str] = field(default_factory=dict)  # label -> setting whose English counts are the fixed denominator
    parity_lang: str = "eng_Latn"  # the reference language (its corpus code)
    no_plots: bool = False
    no_reconstruction: bool = False
    no_utf8_integrity: bool = False
    no_digit_boundary: bool = False
    no_code_ast: bool = False
    save_full_results: bool = False
    generate_latex_tables: bool = False
    verbose: bool = False


def main():
    args = load_args(TokEvalArgs)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    if not args.settings:
        raise SystemExit("settings are required (give them in config=<yaml> or on the CLI).")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    settings = {k: v for k, v in args.settings.items() if not args.only or k in args.only}
    if not settings:
        raise SystemExit(f"No settings left after only={args.only}; have {list(args.settings)}")

    language_config = args.language_config
    if not language_config:
        repos = [v for v in settings.values() if v.get("mode") == "moto_repo"]
        if not repos or not args.corpus_dir:
            raise SystemExit("Give language_config, or a moto_repo setting plus corpus_dir to derive the languages from it.")
        cands = moto_repo_languages(repos[0]["repo"], repos[0].get("revision"))
        if args.languages:  # corpus codes; a routing source name also selects its language
            cands = {src: cs for src, cs in cands.items() if src in args.languages or any(c in args.languages for c in cs)}
        corpus = Path(args.corpus_dir)
        check_corpus_split(corpus, args.corpus_split)
        chosen = {src: next((c for c in cs if (corpus / f"{c}.txt").exists()), None) for src, cs in cands.items()}
        todo = {src: cands[src] for src, c in chosen.items() if c is None}
        if todo:
            got, absent = fetch_flores(todo, corpus, args.corpus_split, args.corpus_revision)
            if absent:
                raise SystemExit(f"{ {s_: cands[s_] for s_ in absent} } (routing source: FLORES+ codes tried) are not in "
                                 f"{FLORES_DATASET} split={args.corpus_split}; drop them with languages=[...] or put the files in corpus_dir.")
            chosen.update(got)
        derived = {code: src for src, code in chosen.items()}
        gen = {"languages": {c: {"name": c, "iso_code": c[:3], "data_path": str((corpus / f"{c}.txt").resolve())}
                             for c in derived}}
        language_config = str(out / "language_config.generated.json")
        Path(language_config).write_text(json.dumps(gen, indent=2))
        logger.info("Languages from %s: %s (wrote %s)", repos[0]["repo"], {c: derived[c] for c in sorted(derived)}, language_config)

    lm = LanguageMetadata(language_config)
    n = args.samples_per_lang
    language_texts = load_multilingual_data(language_metadata=lm, max_texts_per_language=n)
    if args.languages and args.language_config:
        language_texts = {k: v for k, v in language_texts.items() if k in args.languages}
    if not language_texts:
        raise SystemExit("No corpus text loaded; check language_config and data paths.")

    if args.parity_denominators:
        if args.parity_lang not in language_texts:
            raise SystemExit(f"parity_lang {args.parity_lang} is not among the evaluated languages {sorted(language_texts)}")
        check_aligned(language_texts)

    tokenized: Dict[str, List[TokenizedData]] = {}
    vocabularies: Dict[str, Any] = {}
    for name, scfg in settings.items():
        mode = scfg.get("mode", "single")
        logger.info("Tokenizing setting %s (%s)", name, mode)
        if mode == "single":
            tokenized[name], vocabularies[name] = _tokenize_single(name, scfg, language_texts)
        elif mode == "routed":
            tokenized[name], vocabularies[name] = _tokenize_routed(name, scfg, language_texts)
        elif mode == "moto_repo":
            tokenized[name], vocabularies[name] = _tokenize_moto_repo(name, scfg, language_texts)
        else:
            raise SystemExit(f"unknown mode {mode!r} for setting {name}")

    write_counts_and_parity(tokenized, args, settings, out)

    measurement = None
    if args.measurement_config:
        measurement = TextMeasurementConfig.from_dict(json.loads(Path(args.measurement_config).read_text()))

    code_ast_config = json.loads(Path(args.code_ast_config).read_text()) if args.code_ast_config else None
    analyzer = create_analyzer_from_tokenized_data(
        tokenized, vocabularies,
        measurement_config=measurement, language_metadata=lm, plot_save_dir=str(out),
        math_data_path=args.math_data, use_builtin_math_data=args.use_builtin_math_data,
        code_ast_config=code_ast_config)
    results = analyzer.run_analysis(
        save_plots=not args.no_plots,
        include_morphscore=False,
        include_digit_boundary=not args.no_digit_boundary,
        include_code_ast=not args.no_code_ast,
        include_utf8_integrity=not args.no_utf8_integrity,
        include_reconstruction=not args.no_reconstruction,
        verbose=args.verbose,
        cer_time_budget_s=0,
    )
    results["run_metadata"] = {
        "driver": "moto.tokenizer_intrinsic_metrics.tokeval.run_moto_tokeval",
        "config": str(args.config), "samples_per_lang": n, "settings": settings,
        "corpus": {"dir": args.corpus_dir or None, "dataset": FLORES_DATASET, "split": args.corpus_split if args.corpus_dir else None,
                   "revision_requested": args.corpus_revision},
        "languages": sorted(language_texts),
        "input_path": "pre-tokenized (oracle routing for routed settings)",
    }
    normalized = normalize_results(results)
    slim = _convert_for_json_public(select_results(normalized))
    (out / "analysis_results.json").write_text(json.dumps(slim, indent=2))
    if args.save_full_results:
        (out / "analysis_results_full.json").write_text(json.dumps(_convert_for_json_public(normalized), indent=2))
    write_summaries(slim, list(settings), sorted(language_texts), out)

    if args.generate_latex_tables:
        analyzer.generate_latex_tables(results=results, output_dir=str(out / "latex_tables"),
                                       table_types=["basic", "comprehensive"])
    print(f"Wrote {out}/analysis_results.json, summary_global.csv, summary_per_language.csv, token_counts.csv"
          + (", parity_*.csv" if args.parity_denominators else ""))


if __name__ == "__main__":
    main()
