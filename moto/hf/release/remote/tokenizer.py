"""MoTo tokenizer: one tokenizer per language over a shared superset vocabulary.

Shipped as tokenizer.py in every r-three-moto MoTo repo and loaded with

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, language="fra_Latn")

The repo holds the original tokenizer of each language, unmodified, under tokenizers/<name>/
together with super_mapping.json (its id -> the model's superset id), and moto_config.json
with the routing table and special ids. At load time each component is turned into a
superset tokenizer in memory, exactly as moto's reindex_tokenizers.py did for training:
vocabulary ids remapped, unmapped entries dropped, superset BOS/EOS registered, BOS
prepended. This class wraps the active one as its backend:

  * encoding (`tok(text)`, `tok.encode`, batching, padding, offsets) is the active
    component's, which reproduces training: [bos] + superset ids;
  * `tok.set_language("tur_Latn")` (or `with tok.using("tur_Latn"):`) switches the component.
    Accepts component names, training sources (fw_edu, arb_Arab, ...) and the aliases in
    moto_config.json (eng_Latn, cmn_Hans, pes_Arab, ...);
  * `tok.set_mapping({"cat_Latn": "spa_Latn", "regex:.*_Cyrl": "rus_Cyrl"})` overrides which
    tokenizer a language (or task, or any key) uses, like lingua's eval-time
    lang_to_tokenizer / task_to_tokenizer: exact keys first, then `regex:` keys in order
    (re.match). `tok.mapping` shows the effective table; `set_mapping(None)` resets it;
  * decoding accepts ids from any component: each id is decoded by a component that owns
    it, preferring the active one, so generations that mix vocabularies decode fully;
  * len(tok) and tok.vocab_size are the superset size, i.e. the model's embedding rows.

Only depends on transformers, tokenizers and huggingface_hub. Works with transformers
4.57 and 5.x. Source: moto/hf/release/remote/tokenizer.py in r-three/moto.
"""

from __future__ import annotations

import atexit
import contextlib
import copy
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

MOTO_FILES = ("moto_config.json", "tokenizer_config.json", "tokenizer.py")
# what a hub download needs; super_vocab.json is not needed to tokenize
ALLOW_PATTERNS = ["moto_config.json", "tokenizer_config.json", "tokenizer.py", "tokenizers/*"]
INIT_KEYS = ("language", "mapping", "model_max_length", "padding_side", "truncation_side", "clean_up_tokenization_spaces")


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level alphabet (tokenizers' ByteLevel pre-tokenizer)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_DECODER = {c: b for b, c in _bytes_to_unicode().items()}


ALIGNED_BOS = "~SPECIAL~ALIGNED~BOS~SYMBOL~"
ID_PROCESSORS = ("TemplateProcessing", "BertProcessing", "RobertaProcessing")  # carry local special ids


def _tokens_with_id(tj: dict, tid: int) -> list[str]:
    out = [t["content"] for t in tj.get("added_tokens", []) if int(t["id"]) == tid]
    return out + [k for k, v in tj["model"]["vocab"].items() if v == tid and k not in out]


def _ensure_special(tj: dict, tid: int, fallback: str) -> str:
    """Name of the token with superset id `tid`, registered as a special added token."""
    names = _tokens_with_id(tj, tid)
    name = names[0] if names else fallback
    if not names:
        tj["model"]["vocab"][name] = tid
    for t in tj.setdefault("added_tokens", []):
        if t["content"] == name:
            t["id"], t["special"] = tid, True
            return name
    tj["added_tokens"].append({"id": tid, "content": name, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True})
    return name


def to_superset_json(tj: dict, id_map: dict[int, int], bos_id: int | None, eos_id: int | None, add_bos: bool, eos_fallback: str = "</s>") -> tuple[dict, str | None, str | None]:
    """An original tokenizer.json -> the superset tokenizer used in training.

    Vocabulary and added tokens are remapped as reindex_tokenizers.reindex_hf_json does
    (entries without a superset id are dropped; added tokens also go into the vocab).
    Then BOS/EOS get the superset ids, BPE-dropout is switched off, and the post-processor
    prepends BOS iff add_bos (training encoded with add_special_tokens=False and added
    [bos] itself). Returns (tokenizer json, bos token, eos token).
    """
    tj = copy.deepcopy(tj)
    model = tj["model"]
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise NotImplementedError(f"{model.get('type')} tokenizers with list vocabularies are not supported")
    model["vocab"] = {t: id_map[i] for t, i in vocab.items() if i in id_map}
    added = []
    for t in tj.get("added_tokens", []):
        if int(t["id"]) in id_map:
            t = {**t, "id": id_map[int(t["id"])]}
            added.append(t)
            model["vocab"][t["content"]] = t["id"]
    tj["added_tokens"] = added
    if model.get("dropout"):
        model["dropout"] = None
    bos = _ensure_special(tj, bos_id, ALIGNED_BOS) if bos_id is not None else None
    eos = _ensure_special(tj, eos_id, eos_fallback) if eos_id is not None else None
    pp = tj.get("post_processor")
    steps = pp["processors"] if pp and pp.get("type") == "Sequence" else ([pp] if pp else [])
    steps = [p for p in steps if p.get("type") not in ID_PROCESSORS]
    if add_bos and bos is not None:
        steps.append({
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": bos, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": bos, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": bos, "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {bos: {"id": bos, "ids": [bos_id], "tokens": [bos]}},
        })
    tj["post_processor"] = None if not steps else steps[0] if len(steps) == 1 else {"type": "Sequence", "processors": steps}
    return tj, bos, eos


def load_id_map(path: Path) -> dict[int, int]:
    return {int(k): int(v) for k, v in json.loads(Path(path).read_text()).items()}


def _is_byte_level(tokenizer_json: Path) -> bool:
    dec = json.loads(tokenizer_json.read_text()).get("decoder") or {}
    parts = dec.get("decoders", [dec]) if dec.get("type") == "Sequence" else [dec]
    return any(p.get("type") == "ByteLevel" for p in parts)


class MoToTokenizer(PreTrainedTokenizerFast):
    vocab_files_names: dict = {}
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(self, moto_dir: str | os.PathLike | None = None, language: str | None = None, mapping: dict | None = None, **kwargs):
        if moto_dir is None:
            raise ValueError("MoToTokenizer needs moto_dir; load it with AutoTokenizer.from_pretrained(repo, trust_remote_code=True)")
        moto_dir = Path(moto_dir)
        cfg = json.loads((moto_dir / "moto_config.json").read_text())
        self._moto_dir = moto_dir
        self.moto_config = cfg
        self._components = {t["name"]: t for t in cfg["tokenizers"]}
        self._backends: dict[str, Tokenizer] = {}
        self._owners: dict[int, str] | None = None
        self._byte_level: dict[str, bool] = {}
        self._mapping: dict[str, str] = {}
        self._built: dict[str, Path] = {}
        self._specials: dict[str, tuple[str | None, str | None]] = {}
        self._build_dir = Path(tempfile.mkdtemp(prefix="moto-tokenizer-"))
        atexit.register(shutil.rmtree, self._build_dir, True)
        if mapping:
            self.set_mapping(mapping)
        name = self.resolve_language(language or cfg.get("default_tokenizer") or cfg["tokenizers"][0]["name"])
        built = self._component_file(name)
        bos_name, eos_name = self._specials[name]
        for key in ("tokenizer_object", "tokenizer_file", "vocab_file", "merges_file"):
            kwargs.pop(key, None)
        kwargs.setdefault("bos_token", bos_name)
        kwargs.setdefault("eos_token", eos_name)
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        # tokenizer_file, not tokenizer_object: the latter is deep-copied, i.e. serialized, and
        # tokenizers < 0.23 then warns "OrderedVocab ... contains holes ... could be corrupted".
        # Component vocabularies are sparse by design (superset ids), so that warning is noise.
        super().__init__(tokenizer_file=str(built), **kwargs)
        self.language = name

    # ------------------------------------------------------------ languages

    @property
    def languages(self) -> list[str]:
        """Component names, in training order."""
        return list(self._components)

    def _resolve_builtin(self, language: str) -> str | None:
        """Component names, the training routing table, its aliases, and tokenizer repo names.

        Repo names (the training repo, e.g. flexitok/reindexed-..., or the shipped original) are
        matched like lingua.data._resolve_tokenizer_key against "huggingface/<repo>": exact,
        case-insensitive, or one ending in "/" + the other. That lets lingua routing tables
        (task_to_tokenizer, source_to_tokenizer) be used as they are.
        """
        routing = self.moto_config.get("routing", {})
        table = routing.get("source_to_tokenizer", {})
        aliases = routing.get("language_aliases", {})
        for key in (language, aliases.get(language)):
            if key in self._components:
                return key
            if key in table:
                return table[key]
        wanted = str(language).strip().lower()
        if "/" not in wanted:
            return None
        keys = []
        for name, comp in self._components.items():
            for repo in (comp.get("training_repo"), comp.get("source_repo")):
                if repo:
                    keys.append((f"huggingface/{repo}".lower(), name))
        for key, name in keys:
            if key == wanted or key == f"huggingface/{wanted}":
                return name
        for key, name in keys:
            if key.endswith(f"/{wanted}") or wanted.endswith(f"/{key}"):
                return name
        return None

    def resolve_language(self, language: str) -> str:
        """Component used for `language`: set_mapping overrides (exact, then regex:), then built-ins."""
        if language in self._mapping:
            return self._mapping[language]
        name = self._resolve_builtin(language)
        if name is not None:
            return name
        for key, target in self._mapping.items():
            if key.startswith("regex:") and re.match(key[len("regex:"):], language):
                return target
        options = set(self._components) | set(self.moto_config.get("routing", {}).get("source_to_tokenizer", {}))
        options |= set(self.moto_config.get("routing", {}).get("language_aliases", {})) | {k for k in self._mapping if not k.startswith("regex:")}
        raise ValueError(f"unknown language {language!r}; use one of {sorted(options)} or add it with set_mapping")

    def set_mapping(self, mapping: dict | None, replace: bool = False) -> "MoToTokenizer":
        """Route keys (languages, tasks, `regex:<pattern>`) to tokenizers.

        Values may be component names, training sources or aliases; they are resolved now,
        so a typo fails here rather than at encode time. `None` drops all overrides;
        `replace=True` drops them before adding `mapping`. Does not change the active
        language: call set_language afterwards.
        """
        if mapping is None:
            self._mapping = {}
            return self
        resolved = {}
        for key, target in mapping.items():
            name = self._resolve_builtin(str(target))
            if name is None:
                raise ValueError(f"set_mapping: {key!r} -> {target!r}: no such tokenizer; use one of {self.languages}")
            if str(key).startswith("regex:"):
                re.compile(str(key)[len("regex:"):])
            resolved[str(key)] = name
        self._mapping = resolved if replace else {**self._mapping, **resolved}
        return self

    @property
    def mapping(self) -> dict[str, str]:
        """Effective routing: training table and aliases, overridden by set_mapping."""
        routing = self.moto_config.get("routing", {})
        out = dict(routing.get("source_to_tokenizer", {}))
        out.update({a: self._resolve_builtin(a) for a in routing.get("language_aliases", {})})
        out.update(self._mapping)
        return out

    def _component_file(self, name: str) -> Path:
        """The superset tokenizer.json of a component, built on first use from the shipped files."""
        if name not in self._built:
            comp = self._components[name]
            src = self._moto_dir / comp["subfolder"]
            cfg = self.moto_config
            tj, bos, eos = to_superset_json(
                json.loads((src / "tokenizer.json").read_text()),
                load_id_map(src / comp.get("id_map", "super_mapping.json")),
                cfg.get("bos_token_id"),
                cfg.get("eos_token_id"),
                bool(cfg.get("add_bos", True)),
                cfg.get("eos_token") or "</s>",
            )
            path = self._build_dir / f"{name}.json"
            path.write_text(json.dumps(tj, ensure_ascii=False))
            self._built[name], self._specials[name] = path, (bos, eos)
        return self._built[name]

    def _backend(self, name: str) -> Tokenizer:
        """Read-only cached backend for decoding; the active one is a separate fresh load."""
        if name not in self._backends:
            self._backends[name] = Tokenizer.from_file(str(self._component_file(name)))
        return self._backends[name]

    def set_language(self, language: str) -> "MoToTokenizer":
        name = self.resolve_language(language)
        if name != getattr(self, "language", None):
            backend = Tokenizer.from_file(str(self._component_file(name)))  # fresh load; no copy, no serialization
            if self._tokenizer.truncation is not None:  # keep settings the caller made
                backend.enable_truncation(**self._tokenizer.truncation)
            if self._tokenizer.padding is not None:
                backend.enable_padding(**self._tokenizer.padding)
            self._tokenizer = backend
            pad_was_eos = self.pad_token is not None and self.pad_token == self.eos_token
            self.bos_token, self.eos_token = self._specials[name]
            if pad_was_eos:  # tok.pad_token = tok.eos_token survives a switch even if the EOS string differs
                self.pad_token = self.eos_token
            self.language = name
        return self

    @contextlib.contextmanager
    def using(self, language: str):
        previous = self.language
        self.set_language(language)
        try:
            yield self
        finally:
            self.set_language(previous)

    # ------------------------------------------------------------ superset view

    def __len__(self) -> int:
        return int(self.moto_config["vocab_size"])

    @property
    def vocab_size(self) -> int:
        return int(self.moto_config["vocab_size"])

    def _is_byte_level(self, name: str) -> bool:
        if name not in self._byte_level:
            self._byte_level[name] = _is_byte_level(self._component_file(name))
        return self._byte_level[name]

    def _owner_map(self) -> dict[int, str]:
        if self._owners is None:
            owners: dict[int, str] = {}
            for name in reversed(list(self._components)):  # earlier components win ties
                for i in self._backend(name).get_vocab(with_added_tokens=True).values():
                    owners[i] = name
            self._owners = owners
        return self._owners

    def _decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces=None, **kwargs) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        active = self._backend(self.language)
        if all(active.id_to_token(i) is not None for i in token_ids):
            return active.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
        owners = self._owner_map()
        assigned = [(i, self.language if active.id_to_token(i) is not None else owners.get(i)) for i in token_ids]
        assigned = [(i, o) for i, o in assigned if o is not None]  # drop ids outside every component
        if all(self._is_byte_level(o) for o in {o for _, o in assigned}):
            # join raw bytes across components: one character can span tokens from two vocabularies
            out = bytearray()
            for i, o in assigned:
                backend = self._backend(o)
                added = backend.get_added_tokens_decoder().get(i)
                if added is not None:
                    if not (skip_special_tokens and added.special):
                        out += added.content.encode("utf-8")
                    continue
                for ch in backend.id_to_token(i):
                    b = BYTE_DECODER.get(ch)
                    out += bytes([b]) if b is not None else ch.encode("utf-8")
            return out.decode("utf-8", errors="replace")
        parts, run, run_owner = [], [], None
        for i, owner in assigned:
            if owner != run_owner and run:
                parts.append(self._backend(run_owner).decode(run, skip_special_tokens=skip_special_tokens))
                run = []
            run_owner = owner
            run.append(i)
        if run:
            parts.append(self._backend(run_owner).decode(run, skip_special_tokens=skip_special_tokens))
        return "".join(parts)

    # ------------------------------------------------------------ loading / saving

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        path = str(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder") or ""
        if os.path.isdir(os.path.join(path, subfolder)):
            root = os.path.join(path, subfolder)
        else:
            from huggingface_hub import snapshot_download

            patterns = [f"{subfolder}/{p}" if subfolder else p for p in ALLOW_PATTERNS]
            snap = snapshot_download(
                path,
                revision=kwargs.get("revision"),
                cache_dir=kwargs.get("cache_dir"),
                token=kwargs.get("token"),
                local_files_only=kwargs.get("local_files_only", False),
                allow_patterns=patterns,
            )
            root = os.path.join(snap, subfolder)
        cfg_path = Path(root) / "tokenizer_config.json"
        root_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        init = {k: root_cfg[k] for k in INIT_KEYS if k in root_cfg}
        init.update({k: kwargs[k] for k in INIT_KEYS if k in kwargs})
        return cls(moto_dir=root, name_or_path=path, **init)

    def save_pretrained(self, save_directory, *args, **kwargs):
        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name in MOTO_FILES:
            src = self._moto_dir / name
            if src.exists() and src.resolve() != (out / name).resolve():
                shutil.copyfile(src, out / name)
                written.append(str(out / name))
        # keep the active language and set_mapping overrides
        tc_path = out / "tokenizer_config.json"
        tc = json.loads(tc_path.read_text()) if tc_path.exists() else {}
        tc["language"] = self.language
        if self._mapping:
            tc["mapping"] = dict(self._mapping)
        else:
            tc.pop("mapping", None)
        tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n")
        written.append(str(tc_path))
        src_tok = self._moto_dir / "tokenizers"
        if src_tok.resolve() != (out / "tokenizers").resolve():
            shutil.copytree(src_tok, out / "tokenizers", dirs_exist_ok=True)
        written.append(str(out / "tokenizers"))
        return tuple(written)
