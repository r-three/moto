"""Superset tokenizer: several existing tokenizers over one shared vocabulary.

Shipped as tokenizer.py in r-three-moto/moto-1b-superset-albert and loaded with

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

The model was trained with lingua.tokenizer.SupersetTokenizer: every encode call picks one of
the component tokenizers (aya-expanse, BLOOM, Llama 3.2, GPT-4o, tekken, ...) uniformly at
random from a NumPy generator seeded with 42, encodes the text with it, maps the component's
token ids to the model's superset ids (ids without a superset id are dropped) and prepends the
aligned BOS. This class does the same:

  * by default every `_tokenize` call (one per text) draws a component with
    np.random.default_rng(seed).choice(n), so the sequence of choices, and therefore of ids,
    depends on call order, exactly as in training. `tok.last_component` says which one ran;
  * `tok.set_component("llama-3.2")` (or `with tok.using("gpt-4o"):`, or
    `component="..."` at load time) pins one component; `set_component(None)` returns to random;
  * `tok.reset_rng()` restarts the random sequence from the seed.

No tokenizer files are shipped. Each component is loaded on first use from its original source
at a pinned revision: Hub repos through transformers (gated ones, e.g. Llama 3.2 and Gemma 2,
need an accepted license and a token), GPT-4o through tiktoken, tekken through mistral_common
and the TokenMonster vocabulary through tokenmonster. Those three packages are imported only
when their component is used. The repo ships what is specific to this model:
superset_tokenizer.json (components, pins, special ids), superset_id_maps.json (component id ->
superset id) and super_vocab.json (superset id -> token string).

Decoding maps ids back through the component that produced them (the last one used, else the
first that owns each id). Source: moto/hf/release/remote/superset_tokenizer.py in r-three/moto.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizer

CONFIG_FILE = "superset_tokenizer.json"
ID_MAPS_FILE = "superset_id_maps.json"
SUPER_VOCAB_FILE = "super_vocab.json"
TIKTOKEN_MAX_ENCODE_CHARS = 400_000  # lingua TikTokenTokenizer
PLACEHOLDER = "<|superset_id_{}|>"  # token string for a superset id super_vocab.json has no surface for


class _Component:
    """One component tokenizer: native ids in, native ids out. Loaded lazily."""

    def __init__(self, spec: dict, hub_kwargs: dict):
        self.spec = spec
        self.hub_kwargs = hub_kwargs
        self._impl = None

    @property
    def name(self) -> str:
        return self.spec["name"]

    def _load(self):
        s = self.spec
        backend = s["backend"]
        if backend == "hf":
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(s["repo"], revision=s.get("revision"), **self.hub_kwargs)
            # lingua SimplifiedHFTokenizer encodes with the tokenizers backend; ByT5 has none
            return ("hf_fast", tok._tokenizer) if getattr(tok, "_tokenizer", None) is not None and not s.get("slow") else ("hf_slow", tok)
        if backend == "tiktoken":
            return ("tiktoken", _optional("tiktoken", s).encoding_for_model(s["model"]))
        if backend == "tekken":
            mistral = _optional("mistral_common.tokens.tokenizers.mistral", s)
            _check_tekken_file(s)
            return ("tekken", mistral.MistralTokenizer.v3(is_tekken=True).instruct_tokenizer.tokenizer)
        if backend == "tokenmonster":
            return ("tokenmonster", _optional("tokenmonster", s).load(s["model"]))
        raise ValueError(f"unknown backend {backend!r} for component {s['name']!r}")

    @property
    def impl(self):
        if self._impl is None:
            self._impl = self._load()
        return self._impl

    def encode(self, text: str) -> list[int]:
        kind, t = self.impl
        if kind == "hf_fast":
            return list(t.encode(text, add_special_tokens=False).ids)
        if kind == "hf_slow":
            return list(t.encode(text, add_special_tokens=False))
        if kind == "tiktoken":
            subs = [text[i : i + TIKTOKEN_MAX_ENCODE_CHARS] for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS)]
            return [i for part in t.encode_ordinary_batch(subs) for i in part]
        if kind == "tekken":
            return list(t.encode(text, False, False))
        ids = t.tokenize(text)  # tokenmonster: numpy array, or None for ""
        return [] if ids is None else [int(i) for i in ids]

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        kind, t = self.impl
        if kind in ("hf_fast",):
            return t.decode(ids, skip_special_tokens=False)
        if kind == "hf_slow":
            return t.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if kind in ("tiktoken", "tekken"):
            return t.decode(ids)
        return t.decode(ids)


def _optional(module: str, spec: dict):
    """Import a component's package only when that component is used.

    importlib rather than `import`: transformers scans remote code for import statements and
    refuses to load it when any imported package is missing, even inside a function.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        pkg = spec.get("package") or module.split(".")[0]
        ver = f"=={spec['package_version']}" if spec.get("package_version") else ""
        raise ImportError(f"component {spec['name']!r} needs {pkg}: pip install {pkg}{ver}") from e


def _check_tekken_file(spec: dict) -> None:
    """mistral_common bundles the tekken vocabulary; a different version could ship a different one."""
    want = spec.get("file_sha256")
    if not want:
        return
    mistral_common = _optional("mistral_common", spec)
    path = Path(mistral_common.__file__).parent / "data" / spec.get("file", "tekken_240718.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found in mistral_common {getattr(mistral_common, '__version__', '?')}")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise ValueError(
            f"{path.name} in mistral_common {getattr(mistral_common, '__version__', '?')} differs from the one the model "
            f"was trained with (sha256 {got[:12]} != {want[:12]}); install mistral_common=={spec.get('package_version')}"
        )


class SupersetTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"config_file": CONFIG_FILE, "id_maps_file": ID_MAPS_FILE, "super_vocab_file": SUPER_VOCAB_FILE}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, config_file=None, id_maps_file=None, super_vocab_file=None, component=None, seed=None, hub_token=None, **kwargs):
        if not (config_file and id_maps_file and super_vocab_file):
            raise ValueError("load with AutoTokenizer.from_pretrained(repo, trust_remote_code=True)")
        self.superset_config = json.loads(Path(config_file).read_text())
        cfg = self.superset_config
        self._files = {"config_file": config_file, "id_maps_file": id_maps_file, "super_vocab_file": super_vocab_file}
        self.super_vocab = json.loads(Path(super_vocab_file).read_text())
        self._surface = {i: s for s, i in self.super_vocab.items()}
        maps = json.loads(Path(id_maps_file).read_text())
        hub_kwargs = {"token": hub_token} if hub_token else {}
        self._components = [_Component(spec, hub_kwargs) for spec in cfg["components"]]
        self._by_name = {c.name: i for i, c in enumerate(self._components)}
        self._maps = [{int(k): int(v) for k, v in maps[c.name].items()} for c in self._components]
        self._inverse: list[dict[int, int] | None] = [None] * len(self._components)
        self.seed = int(cfg.get("seed", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        self._fixed: int | None = None
        self.last_component: str | None = None
        kwargs.setdefault("bos_token", cfg["bos_token"])
        kwargs.setdefault("eos_token", cfg["eos_token"])
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        super().__init__(**kwargs)
        self.set_component(component)

    # ------------------------------------------------------------ components

    @property
    def components(self) -> list[str]:
        """Component names, in training order (the order the random draw indexes)."""
        return [c.name for c in self._components]

    def _index(self, component) -> int:
        if isinstance(component, (int, np.integer)):
            if not 0 <= int(component) < len(self._components):
                raise ValueError(f"component index {component} out of range 0..{len(self._components) - 1}")
            return int(component)
        key = str(component)
        if key in self._by_name:
            return self._by_name[key]
        lowered = key.strip().lower()
        for i, c in enumerate(self._components):  # also: the training key, or the source repo / model
            names = {c.name, c.spec.get("training_key"), c.spec.get("repo"), c.spec.get("model"), c.spec.get("training_path")}
            if lowered in {str(n).lower() for n in names if n}:
                return i
        raise ValueError(f"unknown component {component!r}; use one of {self.components}")

    def set_component(self, component=None) -> "SupersetTokenizer":
        """Pin a component (name, index, training key or source repo); None = random per call, as in training."""
        self._fixed = None if component is None else self._index(component)
        return self

    @property
    def component(self) -> str | None:
        return None if self._fixed is None else self._components[self._fixed].name

    @contextlib.contextmanager
    def using(self, component):
        previous = self._fixed
        self.set_component(component)
        try:
            yield self
        finally:
            self._fixed = previous

    def reset_rng(self, seed: int | None = None) -> "SupersetTokenizer":
        self.seed = self.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        return self

    def _choose(self) -> int:
        if self._fixed is not None:
            return self._fixed
        return int(self.rng.choice(len(self._components)))  # lingua SupersetTokenizer.sample_tokenizer

    # ------------------------------------------------------------ encoding

    def encode_superset_ids(self, text: str, component=None) -> list[int]:
        """Superset ids of `text`, without special tokens. Draws a component unless one is given or pinned."""
        i = self._index(component) if component is not None else self._choose()
        self.last_component = self._components[i].name
        m = self._maps[i]
        return [m[t] for t in self._components[i].encode(text) if t in m]

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return [self._id_to_token(i) for i in self.encode_superset_ids(text)]

    def _id_to_token(self, index: int) -> str:
        s = self._surface.get(int(index))
        return s if s is not None else PLACEHOLDER.format(int(index))

    _convert_id_to_token = _id_to_token

    def _convert_token_to_id(self, token: str) -> int:
        i = self.super_vocab.get(token)
        if i is not None:
            return i
        if token.startswith("<|superset_id_") and token.endswith("|>"):
            return int(token[len("<|superset_id_") : -2])
        raise KeyError(f"{token!r} is not a superset token")

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id] if self.superset_config.get("add_bos", True) else []
        out = bos + list(token_ids_0)
        if token_ids_1 is not None:
            out += bos + list(token_ids_1)
        return out

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            specials = {self.bos_token_id, self.eos_token_id}
            return [int(t in specials) for t in token_ids_0]
        n = int(bool(self.superset_config.get("add_bos", True)))
        mask = [1] * n + [0] * len(token_ids_0)
        if token_ids_1 is not None:
            mask += [1] * n + [0] * len(token_ids_1)
        return mask

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    # ------------------------------------------------------------ decoding

    def _inverse_map(self, i: int) -> dict[int, int]:
        if self._inverse[i] is None:
            self._inverse[i] = {v: k for k, v in self._maps[i].items()}
        return self._inverse[i]

    def _decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces=None, **kwargs) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        specials = {self.bos_token_id, self.eos_token_id}
        parts, run, run_owner = [], [], None
        preferred = self._by_name.get(self.last_component) if self.last_component else None

        def flush():
            if run:
                parts.append(self._components[run_owner].decode(run) if run_owner is not None else "")

        for t in map(int, token_ids):
            if t in specials:
                if not skip_special_tokens:
                    flush()
                    run.clear()
                    parts.append(self._id_to_token(t))
                continue
            owner = preferred if preferred is not None and t in self._inverse_map(preferred) else None
            if owner is None:
                owner = next((j for j in range(len(self._components)) if t in self._inverse_map(j)), None)
            if owner is None:
                continue  # an id no component emits
            if owner != run_owner and run:
                flush()
                run.clear()
            run_owner = owner
            run.append(self._inverse_map(owner)[t])
        flush()
        return "".join(parts)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._decode([self._convert_token_to_id(t) for t in tokens])

    # ------------------------------------------------------------ vocabulary / saving

    @property
    def vocab_size(self) -> int:
        return int(self.superset_config["vocab_size"])

    def get_vocab(self) -> dict[str, int]:
        return dict(self.super_vocab)

    def __len__(self) -> int:
        return self.vocab_size

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        out = []
        for key, name in self.vocab_files_names.items():
            dst = Path(save_directory) / ((filename_prefix + "-" if filename_prefix else "") + name)
            src = Path(self._files[key])
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())
            out.append(str(dst))
        return tuple(out)
