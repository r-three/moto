"""Stage 5 — MoTo (Mixture-of-Tokenizers) tokenizer.

Builds the MoTo superset vocabulary by delegating to the existing
``moto.supertokenizer.supervocab`` pipeline, then adds an active-vocab
layer: at inference / training time only ``active_vocab_size`` tokens are live.

Superset vocab compatibility
-----------------------------
Output files follow the exact same layout as ``supervocab.py``:

    {output_dir}/
        super_vocab.json                  — {token_str: super_id}   (supervocab-compatible)
        config.yaml                       — {model: {vocab_size: N}}
        participating_tokenizers.json     — list of tokenizer names
        {name}_vocab.json                 — per-tokenizer Vocab
        {name}_super_mapping.json         — {orig_id: super_id}
        {name}_info.json / .yaml          — tokenizer metadata
        moto_lang_extension_super_ids.json — {lang_code: [super_id, ...]}  (MoTo-specific)
        moto_config.json                  — MoTo hyperparams

This means the output can be fed directly into ``reindex_tokenizers.py`` for
any of the participating tokenizers, unchanged.

Active vocabulary
-----------------
For language ``lang`` the active set is:

    base_core_super_ids  ∪  moto_lang_extension_super_ids[lang]

where ``base_core_super_ids`` are the super-ids of all tokens that appear in
the base (Llama) vocabulary.  When the union exceeds ``active_vocab_size``,
extension ids with the highest super-id value (rarest, added latest) are
dropped first.

Usage (CLI):
    python -m moto.llama_moto.moto_tokenizer \\
        base_tokenizer=meta-llama/Meta-Llama-3.1-8B \\
        language_extensions='{"arb_Arab":"output/extended_arb","cmn_Hani":"output/extended_cmn"}' \\
        active_vocab_size=32000 \\
        output_dir=moto_output
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from moto.llama_moto.args import MoToConstructionArgs
from moto.supertokenizer.supervocab import (
    Tokenizer as SuperVocabTokenizer,
    align_to_super,
    join_vocabs,
    make_vocab,
)
from moto.utils.args import load_args

logger = logging.getLogger(__name__)


# ── Language detection (optional dep) ─────────────────────────────────────────

def _detect_lang_fasttext(text: str) -> Optional[str]:
    try:
        import fasttext  # type: ignore
    except ImportError:
        return None
    if not hasattr(_detect_lang_fasttext, "_model"):
        try:
            import fasttext.util
            fasttext.util.download_model("lid.176.bin", if_exists="ignore")
            _detect_lang_fasttext._model = fasttext.load_model("lid.176.bin")
        except Exception as exc:
            logger.debug("fasttext lid model unavailable: %s", exc)
            _detect_lang_fasttext._model = None
    model = _detect_lang_fasttext._model
    if model is None:
        return None
    labels, _ = model.predict(text.replace("\n", " "), k=1)
    return labels[0].replace("__label__", "") if labels else None


# ── Core builder ───────────────────────────────────────────────────────────────

def build_moto_vocab(
    base_tokenizer_path: str,
    language_extensions: Dict[str, str],
    output_dir: str,
    active_vocab_size: int,
) -> Dict:
    """Construct the MoTo superset vocab and write all output files.

    Delegates superset construction to ``supervocab.join_vocabs`` so the
    output is bit-for-bit compatible with the rest of the supertokenizer
    pipeline.

    Returns the moto metadata dict (also written to ``moto_config.json``).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load all tokenizers via the SuperVocab Tokenizer wrapper ──────────────
    all_names: List[str] = [base_tokenizer_path] + list(language_extensions.values())
    tokenizers: Dict[str, SuperVocabTokenizer] = {
        name: SuperVocabTokenizer.load(name) for name in all_names
    }

    # ── Build vocabs & superset ───────────────────────────────────────────────
    logger.info("Extracting per-tokenizer vocabularies")
    tokenizer_vocabs = {name: make_vocab(tok) for name, tok in tokenizers.items()}

    logger.info("Building superset vocabulary (join_vocabs — sorted by bytes)")
    super_vocab = join_vocabs(tokenizer_vocabs)
    logger.info("Superset vocab size: %d", len(super_vocab))

    # ── Identify base-tokenizer super ids (always active) ────────────────────
    base_vocab = tokenizer_vocabs[base_tokenizer_path]
    base_super_ids = sorted(super_vocab[surface] for surface in base_vocab)

    # ── Identify per-language extension super ids ─────────────────────────────
    # Extension ids = super ids that are NOT in the base vocab
    base_super_id_set = set(base_super_ids)
    lang_extension_super_ids: Dict[str, List[int]] = {}
    for lang, ext_path in language_extensions.items():
        ext_vocab = tokenizer_vocabs[ext_path]
        ext_only = sorted(
            super_vocab[surface]
            for surface in ext_vocab
            if super_vocab[surface] not in base_super_id_set
        )
        lang_extension_super_ids[lang] = ext_only
        logger.info(
            "Language '%s': %d extension-only super ids", lang, len(ext_only)
        )

    # ── Write supervocab-compatible files ─────────────────────────────────────
    _write_super_vocab_files(
        out=out,
        super_vocab=super_vocab,
        tokenizers=tokenizers,
        tokenizer_vocabs=tokenizer_vocabs,
    )

    # ── Write MoTo-specific files ─────────────────────────────────────────────
    with open(out / "moto_lang_extension_super_ids.json", "w") as fh:
        json.dump(lang_extension_super_ids, fh, indent=2)
    logger.info("Wrote moto_lang_extension_super_ids.json")

    moto_config = {
        "base_tokenizer": base_tokenizer_path,
        "language_extensions": language_extensions,
        "superset_vocab_size": len(super_vocab),
        "base_vocab_size": len(base_vocab),
        "active_vocab_size": active_vocab_size,
        "base_super_ids_count": len(base_super_ids),
        "lang_extension_counts": {
            lang: len(ids) for lang, ids in lang_extension_super_ids.items()
        },
    }
    with open(out / "moto_config.json", "w") as fh:
        json.dump(moto_config, fh, indent=2)
    logger.info("Wrote moto_config.json: %s", moto_config)

    return moto_config


def _write_super_vocab_files(
    out: Path,
    super_vocab: dict,
    tokenizers: dict,
    tokenizer_vocabs: dict,
):
    """Write supervocab.py-compatible output files."""
    with open(out / "super_vocab.json", "w") as fh:
        json.dump(super_vocab, fh)
    logger.info("Wrote super_vocab.json (%d tokens)", len(super_vocab))

    with open(out / "config.yaml", "w") as fh:
        yaml.dump({"model": {"vocab_size": len(super_vocab)}}, fh)

    with open(out / "participating_tokenizers.json", "w") as fh:
        json.dump(list(tokenizers.keys()), fh)

    for name, vocab in tokenizer_vocabs.items():
        safe_name = name.replace("/", "--")
        # super mapping
        mapping = align_to_super(super_vocab, vocab)
        with open(out / f"{safe_name}_super_mapping.json", "w") as fh:
            json.dump(mapping, fh)
        # vocab
        with open(out / f"{safe_name}_vocab.json", "w") as fh:
            json.dump(vocab, fh)
        # info
        info = tokenizers[name].info()
        with open(out / f"{safe_name}_info.json", "w") as fh:
            json.dump(info, fh)
        with open(out / f"{safe_name}.yaml", "w") as fh:
            yaml.dump(info, fh)
    logger.info("Wrote per-tokenizer vocab / mapping / info files")


# ── Runtime tokenizer class ────────────────────────────────────────────────────

class MoToTokenizer:
    """Runtime wrapper around the MoTo superset vocab.

    Encodes text using the base HuggingFace tokenizer, optionally remapping
    ids to super-vocab indices.  Provides ``active_token_ids(lang)`` to query
    which super-ids should be loaded to the accelerator for a given language.

    Load from a saved output directory::

        tok = MoToTokenizer.load("moto_output")
        ids = tok.encode("مرحبا", lang="arb_Arab")
        active = tok.active_token_ids("arb_Arab")
    """

    def __init__(
        self,
        base_hf_tokenizer: PreTrainedTokenizerBase,
        base_super_mapping: Dict[int, int],   # orig_id → super_id
        lang_extension_super_ids: Dict[str, List[int]],
        active_vocab_size: int,
        lang_detect_backend: str = "fasttext",
    ):
        self._base_tok = base_hf_tokenizer
        self._base_super_map = base_super_mapping
        self._lang_ext_ids = lang_extension_super_ids
        self._active_vocab_size = active_vocab_size
        self._lang_detect_backend = lang_detect_backend
        self._base_super_id_set = set(base_super_mapping.values())
        self._active_cache: Dict[str, List[int]] = {}

    def active_token_ids(self, lang: Optional[str] = None) -> List[int]:
        """Sorted super-ids that should be active for *lang*.

        Base vocab ids are always included.  If the budget is exceeded the
        rarest (highest-id) extension tokens are dropped.
        """
        key = lang or "__base__"
        if key in self._active_cache:
            return self._active_cache[key]

        active = set(self._base_super_id_set)
        if lang and lang in self._lang_ext_ids:
            ext = self._lang_ext_ids[lang]
            if len(active) + len(ext) > self._active_vocab_size:
                budget = self._active_vocab_size - len(active)
                # Keep the lowest (most frequent in superset ordering) ext ids
                ext = sorted(ext)[:budget]
            active |= set(ext)

        result = sorted(active)
        self._active_cache[key] = result
        return result

    def encode(
        self,
        text: str,
        lang: Optional[str] = None,
        add_special_tokens: bool = True,
        remap_to_super: bool = False,
    ) -> List[int]:
        """Encode *text* using the base tokenizer.

        Args:
            text: input string.
            lang: FLORES language code or None for auto-detection.
            add_special_tokens: whether to add BOS/EOS.
            remap_to_super: if True return super-vocab ids, else base ids.
        """
        if lang is None and self._lang_detect_backend == "fasttext":
            lang = _detect_lang_fasttext(text)
        base_ids = self._base_tok.encode(text, add_special_tokens=add_special_tokens)
        if remap_to_super:
            return [self._base_super_map.get(i, i) for i in base_ids]
        return base_ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._base_tok.decode(ids, skip_special_tokens=skip_special_tokens)

    @classmethod
    def load(cls, output_dir: str, lang_detect_backend: str = "fasttext") -> "MoToTokenizer":
        """Reconstruct a MoToTokenizer from a saved output directory."""
        out = Path(output_dir)

        with open(out / "moto_config.json") as fh:
            config = json.load(fh)

        with open(out / "moto_lang_extension_super_ids.json") as fh:
            lang_ext_ids: Dict[str, List[int]] = json.load(fh)

        base_path = config["base_tokenizer"]
        safe_name = base_path.replace("/", "--")
        with open(out / f"{safe_name}_super_mapping.json") as fh:
            raw_map: Dict[str, int] = json.load(fh)
        base_super_map = {int(k): v for k, v in raw_map.items()}

        base_hf_tok = AutoTokenizer.from_pretrained(base_path)

        return cls(
            base_hf_tokenizer=base_hf_tok,
            base_super_mapping=base_super_map,
            lang_extension_super_ids=lang_ext_ids,
            active_vocab_size=config["active_vocab_size"],
            lang_detect_backend=lang_detect_backend,
        )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(args: MoToConstructionArgs):
    if not args.language_extensions:
        raise ValueError("Provide language_extensions={'lang_code': 'tokenizer_path', ...}")

    config = build_moto_vocab(
        base_tokenizer_path=args.base_tokenizer,
        language_extensions=args.language_extensions,
        output_dir=args.output_dir,
        active_vocab_size=args.active_vocab_size,
    )

    print(f"Superset vocab size : {config['superset_vocab_size']}")
    print(f"Base vocab size     : {config['base_vocab_size']}")
    print(f"Active vocab budget : {config['active_vocab_size']}")
    for lang, count in config["lang_extension_counts"].items():
        print(f"  {lang:20s}  extension_tokens={count}")

    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = load_args(MoToConstructionArgs)
    main(args)
