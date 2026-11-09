"""lm-eval model `moto`: evaluates r-three-moto models (MoTo and baselines) the way the paper did.

Reproduces lingua's apps/main/eval.py EvalHarnessLM on top of lm-eval's HFLM:

  * BOS: every scored sequence starts with BOS (eval_base.yaml: generator.add_bos: true).
  * loglikelihood: [bos] + encode(context) + encode(continuation), the two parts encoded
    separately with add_special_tokens=False and no whitespace moved between them
    (lingua's comment: encode(ctx + cont) != encode(ctx) + encode(cont) at the junction).
    HFLM's default instead encodes ctx + cont jointly and moves trailing spaces.
  * loglikelihood_rolling: [bos] + encode(text), every text token scored (HFLM's rolling
    windows with BOS as the prefix token).
  * MoTo routing (MoToTokenizer only), per request, from the task name, as lingua's
    _sample_source_tokenizer_choice: the task_to_tokenizer table (exact key, then `regex:` keys
    in order, re.match) names a tokenizer, used with probability suitable_tokenizer_probability,
    else one drawn uniformly. A task missing from the table, or naming a tokenizer the model
    does not have, is drawn uniformly (lingua did this silently; here it is logged). Requests
    are grouped per tokenizer and scored with that tokenizer active.

model_args (plus every HFLM argument: batch_size, dtype, max_length, device, ...):
  routing   bundled routing name (moto-1b, ...), "auto" (= the repo name, if bundled; default),
            a path to a YAML with task_to_tokenizer / suitable_tokenizer_probability, or "none"
  language  fixed tokenizer for every task instead of routing (e.g. fra_Latn)
  add_bos   default true
  lingua_pair_encoding  default true; false = HFLM's joint context+continuation encoding
  fallback  random (default, as lingua) | default (the tokenizer's default language) | error
  seed      routing RNG seed, default 42
  tokenizer_revision  commit of a `tokenizer=` repo, when it differs from the model's `revision`
            (HFLM uses one revision for both)
  debug_tokens  log the exact token sequence of the first N requests of every task (default 0):
            ids and tokens of context and continuation, so BOS / pair encoding can be checked
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import transformers
import yaml

# lm-eval 0.4.x (incl. the r-three fork) references transformers.AutoModelForVision2Seq at import
# time (models/hf_vlms.py); transformers 5 removed it in favour of AutoModelForImageTextToText.
# peft (which lm-eval imports) can re-create the transformers module object, dropping the alias,
# so import it first.
try:
    import peft  # noqa: F401
except Exception:  # noqa: BLE001 - not installed, or broken against this transformers: lm-eval reports it
    pass
transformers = sys.modules["transformers"]
if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(transformers, "AutoModelForImageTextToText"):
    transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402

logger = logging.getLogger(__name__)
ROUTING_DIR = Path(__file__).parent / "routing"


def _bool(x: Any) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in ("1", "true", "yes", "y")


def bundled_routing(name: str) -> Path | None:
    p = ROUTING_DIR / f"{name}.yaml"
    return p if p.exists() else None


def load_routing(routing: str | None, pretrained: str) -> dict[str, Any] | None:
    if routing is None or str(routing).lower() in ("auto", ""):
        p = bundled_routing(str(pretrained).rstrip("/").split("/")[-1])
        return yaml.safe_load(p.read_text()) if p else None
    if str(routing).lower() == "none":
        return None
    p = bundled_routing(routing) or Path(routing)
    if not p.exists():
        raise FileNotFoundError(f"routing {routing!r}: not bundled ({sorted(x.stem for x in ROUTING_DIR.glob('*.yaml'))}) and not a file")
    return yaml.safe_load(p.read_text())


def resolve_repo(moto_config: dict[str, Any], value: str) -> str | None:
    """A tokenizer repo name -> component, like lingua.data._resolve_tokenizer_key.

    Done here as well as in the repo's tokenizer.py, so routing does not depend on the
    version of tokenizer.py a repo was pushed with. Matches each component's training repo,
    shipped repo and (older moto_config) original_repo against "huggingface/<repo>": exact,
    case-insensitive, or one ending in "/" + the other.
    """
    wanted = str(value).strip().lower()
    keys = []
    for comp in moto_config.get("tokenizers", []):
        for field in ("training_repo", "source_repo", "original_repo"):
            if comp.get(field):
                keys.append((f"huggingface/{comp[field]}".lower(), comp["name"]))
    for key, name in keys:
        if key in (wanted, f"huggingface/{wanted}"):
            return name
    for key, name in keys:
        if key.endswith(f"/{wanted}") or wanted.endswith(f"/{key}"):
            return name
    return None


def table_lookup(table: dict[str, str], task: str | None) -> str | None:
    """lingua: exact key, then `regex:` keys in order (re.match)."""
    import re

    if task is None:
        return None
    if task in table:
        return table[task]
    for key, value in table.items():
        if key.startswith("regex:") and re.match(key[len("regex:"):], task):
            return value
    return None


class _NoKeyPair(tuple):
    """("", "") standing in for a rolling window's missing request key."""


_NO_KEY = _NoKeyPair(("", ""))


class _SkipNoKey:
    def __init__(self, hook):
        self.hook = hook

    def add_partial(self, attr, req, res):
        if isinstance(req, _NoKeyPair):
            return
        self.hook.add_partial(attr, req, res)

    def __getattr__(self, name):
        return getattr(self.hook, name)


@register_model("moto")
class MoToLM(HFLM):
    def __init__(
        self,
        pretrained: str,
        routing: str | None = "auto",
        language: str | None = None,
        add_bos: bool | str = True,
        lingua_pair_encoding: bool | str = True,
        fallback: str = "random",
        seed: int | str = 42,
        debug_tokens: int | str = 0,
        tokenizer_revision: str | None = None,
        **kwargs,
    ) -> None:
        kwargs["trust_remote_code"] = True  # MoTo repos ship tokenizer.py; harmless for baselines
        if tokenizer_revision and isinstance(kwargs.get("tokenizer"), str):
            kwargs["tokenizer"] = transformers.AutoTokenizer.from_pretrained(
                kwargs["tokenizer"], revision=tokenizer_revision, trust_remote_code=True)
        kwargs["add_bos_token"] = False  # BOS is added here, exactly once
        super().__init__(pretrained=pretrained, **kwargs)
        self.moto_add_bos = _bool(add_bos)
        self.lingua_pair_encoding = _bool(lingua_pair_encoding)
        self.fallback = fallback
        self.rng = np.random.default_rng(int(seed))
        self.debug_tokens = int(debug_tokens)
        self._debug_seen: Counter = Counter()
        self.is_moto = type(self.tokenizer).__name__ == "MoToTokenizer"
        self.fixed_language = None
        self.routing = None
        self._route_log: dict[str, Counter] = defaultdict(Counter)
        self._unrouted: dict[str, str] = {}
        if not self.is_moto:
            if language or (routing not in (None, "auto", "none")):
                logger.warning("%s is not a MoTo model; language/routing ignored", pretrained)
            return
        if language:
            self.fixed_language = self.tokenizer.resolve_language(language)
            return
        cfg = load_routing(routing, pretrained)
        if cfg is None:
            logger.warning("%s: no routing table; every task uses %s", pretrained, self.tokenizer.language)
            self.fixed_language = self.tokenizer.language
            return
        self.routing = {
            "p": float(cfg.get("suitable_tokenizer_probability", 1.0)),
            "table": dict(cfg.get("task_to_tokenizer") or {}),
        }
        unresolved = {v: None for v in set(self.routing["table"].values()) if self._resolve(v) is None}
        if unresolved:
            logger.warning(
                "routing names %d tokenizers this model does not have; their tasks are drawn uniformly, as lingua did: %s",
                len(unresolved), sorted(unresolved),
            )
        logger.info("routing: p=%s, %d task keys, from %s", self.routing["p"], len(self.routing["table"]), cfg.get("source", routing))

    # ------------------------------------------------------------------ routing

    def _choose(self, task: str | None) -> str:
        tok = self.tokenizer
        if self.fixed_language is not None:
            return self.fixed_language
        value = table_lookup(self.routing["table"], task)
        target = self._resolve(value) if value is not None else None
        if target is None:
            self._unrouted[task or "?"] = value or "<not in table>"
            if self.fallback == "error":
                raise ValueError(f"task {task!r}: no tokenizer ({value!r})")
            if self.fallback == "default":
                return tok.moto_config.get("default_tokenizer") or tok.languages[0]
            return tok.languages[int(self.rng.integers(len(tok.languages)))]
        if self.routing["p"] >= 1.0 or self.rng.random() < self.routing["p"]:
            return target
        return tok.languages[int(self.rng.integers(len(tok.languages)))]

    def _resolve(self, value: str) -> str | None:
        try:
            return self.tokenizer.resolve_language(value)
        except ValueError:
            return resolve_repo(self.tokenizer.moto_config, value)

    def _debug(self, requests, kind: str) -> None:
        """Log what is actually scored for the first `debug_tokens` requests of each task."""
        if not self.debug_tokens:
            return
        for req in requests:
            task = getattr(req, "task_name", None) or "?"
            if self._debug_seen[task] >= self.debug_tokens:
                continue
            self._debug_seen[task] += 1
            if kind == "loglikelihood":
                c_ids, k_ids = self._lingua_pair(*req.args[:2])
            else:
                ids = self.tok_encode(req.args[0], add_special_tokens=False)
                c_ids, k_ids = ([self.prefix_token_id], ids) if self.prefix_token_id is not None else (ids[:1], ids[1:])
            toks = self.tokenizer.convert_ids_to_tokens
            lang = getattr(self.tokenizer, "language", None)
            logger.info(
                "debug_tokens %s #%d %s%s add_bos=%s bos_id=%s\n  context  (%d) ids=%s ... %s\n  tokens   %s ... %s\n  continuation (%d) ids=%s\n  tokens   %s",
                task, self._debug_seen[task], kind, f" lang={lang}" if lang else "", self.moto_add_bos, self.tokenizer.bos_token_id,
                len(c_ids), c_ids[:8], c_ids[-4:], toks(c_ids[:8]), toks(c_ids[-4:]),
                len(k_ids), k_ids[:16], toks(k_ids[:16]),
            )

    def _grouped(self, requests, fn, **kwargs):
        name = fn.__name__
        kind = "rolling" if "rolling" in name else ("generate" if name == "generate_until" else "loglikelihood")
        if not self.is_moto:
            self._debug(requests, kind)
            return fn(requests, **kwargs)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, req in enumerate(requests):
            lang = self._choose(getattr(req, "task_name", None))
            groups[lang].append(i)
            self._route_log[getattr(req, "task_name", None) or "?"][lang] += 1
        results: list[Any] = [None] * len(requests)
        previous = self.tokenizer.language
        try:
            for lang, idxs in groups.items():
                self.tokenizer.set_language(lang)
                self._debug([requests[i] for i in idxs], kind)
                out = fn([requests[i] for i in idxs], **kwargs)
                for i, o in zip(idxs, out):
                    results[i] = o
        finally:
            self.tokenizer.set_language(previous)
        self._report()
        return results

    def _report(self) -> None:
        for task, counts in sorted(self._route_log.items()):
            logger.info("routing %s -> %s", task, dict(counts))
        if self._unrouted:
            logger.warning("tasks drawn uniformly (no tokenizer in the routing table): %s", self._unrouted)
        self._route_log.clear()

    def routing_summary(self) -> dict[str, Any]:
        return {"is_moto": self.is_moto, "fixed_language": self.fixed_language, "routing": self.routing, "unrouted": dict(self._unrouted)}

    # ------------------------------------------------------------------ encoding (lingua parity)

    def _bos(self) -> list[int]:
        bos = self.tokenizer.bos_token_id
        return [bos] if self.moto_add_bos and bos is not None else []

    def _encode_pair(self, context: str, continuation: str):
        if self.lingua_pair_encoding:
            ctx = self.tok_encode(context, add_special_tokens=False)
            cont = self.tok_encode(continuation, add_special_tokens=False)
        else:
            ctx, cont = super()._encode_pair(context, continuation)
        return self._bos() + ctx, cont

    @property
    def prefix_token_id(self):
        """The token scoring starts from: BOS, as lingua's [bos] + ..., or None.

        None when add_bos=false or the tokenizer has no BOS (e.g. the monolithic Demoto
        tokenizer). HFLM would then fall back to EOS; lingua instead scored the text alone,
        so the first token has no prediction. _loglikelihood_lingua / _rolling_without_prefix
        do the same.
        """
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        bos = self.tokenizer.bos_token_id
        return bos if (self.moto_add_bos and bos is not None) else None

    # ------------------------------------------------------------------ request types

    def _loglikelihood_tokens(self, requests, *args, **kwargs):
        # r-three/lm-evaluation-harness@multilingual-tiny-benchmarks prints a preview that unpacks
        # req[0] as (context, continuation); rolling windows pass None there and it crashes.
        # Give it an empty pair that the cache hook below ignores, as it ignored None.
        requests = [(r[0] if r[0] is not None else _NO_KEY, *r[1:]) for r in requests]
        return super()._loglikelihood_tokens(requests, *args, **kwargs)

    def set_cache_hook(self, cache_hook) -> None:
        super().set_cache_hook(_SkipNoKey(cache_hook))

    def _lingua_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """lingua's sequence for one loglikelihood request, as (scored-as-context, scored) ids."""
        if context != "":
            return self._encode_pair(context, continuation)
        cont = self.tok_encode(continuation, add_special_tokens=False)
        if self.prefix_token_id is not None:
            return [self.prefix_token_id], cont
        return cont[:1], cont[1:]  # no BOS: lingua cannot score the first token either

    def _loglikelihood_lingua(self, requests, disable_tqdm: bool = False):
        """HFLM.loglikelihood, with empty contexts handled as lingua (no EOS stand-in for BOS)."""
        new = [((req.args[0], req.args[1]), *self._lingua_pair(req.args[0], req.args[1])) for req in requests]
        return self._loglikelihood_tokens(new, disable_tqdm=disable_tqdm)

    def _rolling_without_prefix(self, requests, disable_tqdm: bool = False):
        """Rolling loglikelihood of enc(text) with no BOS: the first token is context, the rest scored."""
        from lm_eval import utils

        windows, owner = [], []
        for i, req in enumerate(requests):
            ids = self.tok_encode(req.args[0], add_special_tokens=False)
            if len(ids) < 2:
                continue
            for ctx, cont in map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=ids[1:], prefix_token=ids[0], max_seq_len=self.max_length, context_len=1)):
                windows.append((_NO_KEY, ctx, cont))
                owner.append(i)
        totals = [0.0] * len(requests)
        for i, (ll, _) in zip(owner, self._loglikelihood_tokens(windows, disable_tqdm=disable_tqdm)):
            totals[i] += ll
        return totals

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        return self._grouped(requests, self._loglikelihood_lingua, disable_tqdm=disable_tqdm)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        fn = super().loglikelihood_rolling if self.prefix_token_id is not None else self._rolling_without_prefix
        return self._grouped(requests, fn, disable_tqdm=disable_tqdm)

    def generate_until(self, requests, disable_tqdm: bool = False):
        return self._grouped(requests, super().generate_until, disable_tqdm=disable_tqdm)
