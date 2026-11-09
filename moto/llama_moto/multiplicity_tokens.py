"""Stage 3 — Multiplicity token identification.

A *multiplicity token* is a token whose decoded surface form appears in more
than one tokenizer vocabulary.  When we extend the Llama vocabulary with
language-specific BPE tokens, some new tokens may duplicate existing Llama
entries.

This module identifies those duplicates and produces a resolution map that
downstream stages (Stage 4, Stage 5) use to decide what to do with them.

Resolution strategies
---------------------
"base"    — keep the base-tokenizer embedding; drop the extension's duplicate
            entry.  Safe default: no new embedding rows are added for these.
"ext"     — keep the extension embedding; the base entry is shadowed.
"average" — average the embeddings from both tokenizers (requires weight
            tensors — use only after merging model weights).
"both"    — keep both entries under distinct token ids.  The superset vocab
            grows by the number of duplicates.

Output (written to ``output_json``):
    {
        "base_tokenizer": "...",
        "extension_tokenizers": [...],
        "multiplicity_tokens": {
            "<token_surface>": {
                "base_id": 12345,
                "extension_ids": {"<ext_name>": 678, ...},
                "resolution": "base"
            },
            ...
        },
        "stats": {
            "total_base_vocab": ...,
            "total_extension_vocab": ...,
            "multiplicity_count": ...
        }
    }

Usage (CLI):
    python -m moto.llama_moto.multiplicity_tokens \\
        base_tokenizer=meta-llama/Meta-Llama-3.1-8B \\
        extension_tokenizers='["output/extended_arb","output/extended_cmn"]' \\
        resolution_strategy=base \\
        output_json=multiplicity_tokens.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from moto.llama_moto.args import MultiplicityArgs
from moto.utils.args import load_args

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decoded_vocab(tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    """Return {decoded_surface: token_id} using convert_tokens_to_string.

    This correctly handles GPT-2 byte proxies (Ġ → space), SentencePiece ▁,
    and LLaMA <0x??> byte tokens.
    """
    surface_to_id: Dict[str, int] = {}
    for token, tid in tokenizer.get_vocab().items():
        try:
            surface = tokenizer.convert_tokens_to_string([token])
        except Exception:
            surface = token
        surface_to_id[surface] = tid
    return surface_to_id


# ── Core logic ─────────────────────────────────────────────────────────────────

def find_multiplicity_tokens(
    base_tokenizer: PreTrainedTokenizerBase,
    extension_tokenizers: Dict[str, PreTrainedTokenizerBase],
    resolution_strategy: str = "base",
) -> Dict:
    """Identify tokens whose surface form appears in both base and extension vocabs.

    Args:
        base_tokenizer: the Llama (or any base) tokenizer.
        extension_tokenizers: {name: tokenizer} for each extension.
        resolution_strategy: how to resolve duplicates (see module docstring).

    Returns:
        Full result dict ready to be serialised as JSON.
    """
    base_surface_map = _decoded_vocab(base_tokenizer)
    base_surfaces: Set[str] = set(base_surface_map.keys())

    multiplicity: Dict[str, dict] = {}

    for ext_name, ext_tok in extension_tokenizers.items():
        ext_surface_map = _decoded_vocab(ext_tok)
        overlap = base_surfaces & set(ext_surface_map.keys())

        logger.info(
            "Extension '%s': vocab_size=%d, overlap with base=%d (%.1f%%)",
            ext_name,
            len(ext_surface_map),
            len(overlap),
            100 * len(overlap) / len(ext_surface_map),
        )

        for surface in overlap:
            if surface not in multiplicity:
                multiplicity[surface] = {
                    "base_id": base_surface_map[surface],
                    "extension_ids": {},
                    "resolution": resolution_strategy,
                }
            multiplicity[surface]["extension_ids"][ext_name] = ext_surface_map[surface]

    total_ext_vocab = sum(len(_decoded_vocab(t)) for t in extension_tokenizers.values())
    result = {
        "base_tokenizer": base_tokenizer.name_or_path,
        "extension_tokenizers": list(extension_tokenizers.keys()),
        "resolution_strategy": resolution_strategy,
        "multiplicity_tokens": multiplicity,
        "stats": {
            "total_base_vocab": len(base_surfaces),
            "total_extension_vocab": total_ext_vocab,
            "multiplicity_count": len(multiplicity),
        },
    }

    logger.info(
        "Total multiplicity tokens: %d (appear in base AND at least one extension)",
        len(multiplicity),
    )
    return result


def load_multiplicity_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(args: MultiplicityArgs):
    if not args.extension_tokenizers:
        raise ValueError("Provide at least one extension tokenizer via extension_tokenizers=[...]")

    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer)
    ext_toks: Dict[str, PreTrainedTokenizerBase] = {}
    for path in args.extension_tokenizers:
        name = Path(path).name
        logger.info("Loading extension tokenizer '%s' from %s", name, path)
        ext_toks[name] = AutoTokenizer.from_pretrained(path)

    result = find_multiplicity_tokens(base_tok, ext_toks, args.resolution_strategy)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote multiplicity analysis to %s", out_path)

    stats = result["stats"]
    print(
        f"Multiplicity tokens: {stats['multiplicity_count']} "
        f"(base vocab: {stats['total_base_vocab']}, "
        f"resolution: {args.resolution_strategy})"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = load_args(MultiplicityArgs)
    main(args)
