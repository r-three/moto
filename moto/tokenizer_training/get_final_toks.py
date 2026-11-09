"""
get_final_toks.py

Summarise tokenizer configurations side-by-side.  For each tokenizer the
script extracts:
  - vocab_size, added_tokens, special_tokens
  - pre_tokenizer type, normalizer type
  - number tokenisation pattern (token counts for 1- to 5-digit strings)
  - contraction handling (don't / it's / I'll  → n tokens)
  - whitespace handling (leading/trailing/internal)
  - zero-width character handling (ZWSP / ZWNJ / ZWJ)

Usage:
    langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
    tokenizer_str="meta-llama/Llama-3.2-1B,"
    # tokenizer_str+="flexitok/bpe_boundless_AllL_128000,flexitok/bpe_AllL_128000,"
    tokenizer_str+="flexitok/bpe_boundless_coverage_AllL_128000,flexitok/bpe_gpt4_alll_128000,flexitok/bpe_gpt4_AllL_323000,"
    tokenizer_str+="flexitok/bpe_script_Germ_16000,flexitok/bpe_script_Germ_32000,flexitok/bpe_script_Roma_16000,flexitok/bpe_script_Roma_32000,flexitok/bpe_script_CmJp_16000,flexitok/bpe_script_SEAS_16000,flexitok/bpe_script_Slav_16000,flexitok/bpe_script_Arab_16000,"
    for lang in ${langs[@]}; do
        tokenizer_str+="flexitok/bpe_${lang}_8000,"
        tokenizer_str+="flexitok/bpe_ltr_${lang}_8000_v3,"
    for vs in 4000 8000 16000 32000; do
        tokenizer_str+="flexitok/bpe_ltr_${lang}_${vs}_v2,"
    done; done


    python moto/tokenizer_training/get_final_toks.py \
        tokenizers=[${tokenizer_str%,}] \
        output_csv=results/tok_summary.csv

    # Or build the list in bash as in tokenizer_check.py:
    langs=(arb_Arab ces_Latn cmn_Hani ...)
    tokenizer_str="meta-llama/Llama-3.2-1B,"
    for lang in ${langs[@]}; do
    for vs in 4000 8000 16000 32000; do
        tokenizer_str+="flexitok/bpe_ltr_${lang}_${vs}_v2,"
    done; done
    python moto/tokenizer_training/get_final_toks.py \
        tokenizers=[${tokenizer_str%,}]
"""

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from moto.utils.args import Args, load_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── per-tokenizer info extraction ─────────────────────────────────────────────

def _extract_pretokenizer_components(d: dict) -> List[str]:
    """Recursively walk a pre-tokenizer JSON dict and return human-readable component strings."""
    pt_type = d.get("type", "")

    if pt_type == "Sequence":
        parts = []
        for sub in d.get("pretokenizers", []):
            parts.extend(_extract_pretokenizer_components(sub))
        return parts

    if pt_type == "Split":
        pattern = d.get("pattern", {})
        behavior = d.get("behavior", "")
        if "Regex" in pattern:
            return [f"Split(/{pattern['Regex']}/,{behavior})"]
        if "String" in pattern:
            return [f"Split({pattern['String']!r},{behavior})"]
        return [f"Split(?,{behavior})"]

    if pt_type == "ByteLevel":
        return [f"ByteLevel(add_prefix_space={d.get('add_prefix_space', False)})"]

    if pt_type == "Metaspace":
        replacement = d.get("replacement", "▁")
        add_prefix = d.get("add_prefix_space", True)
        return [f"Metaspace(replacement={replacement!r},add_prefix_space={add_prefix})"]

    if pt_type == "Whitespace":
        return ["Whitespace"]

    if pt_type == "WhitespaceSplit":
        return ["WhitespaceSplit"]

    if pt_type == "Punctuation":
        return [f"Punctuation({d.get('behavior', '')})"]

    if pt_type == "Digits":
        return [f"Digits(individual={d.get('individual_digits', False)})"]

    if pt_type == "UnicodeScripts":
        return ["UnicodeScripts"]

    extras = {k: v for k, v in d.items() if k != "type" and not isinstance(v, (dict, list))}
    if extras:
        kv = ",".join(f"{k}={v}" for k, v in extras.items())
        return [f"{pt_type}({kv})"]
    return [pt_type]


def _flatten_pt_dicts(d: dict) -> List[dict]:
    """Flatten a (possibly nested Sequence) pre_tokenizer dict into a list of leaf dicts."""
    if d.get("type") == "Sequence":
        result = []
        for sub in d.get("pretokenizers", []):
            result.extend(_flatten_pt_dicts(sub))
        return result
    return [d]


_CONTRACTION_RE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|n't", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\\p\{N\}|\\d")


def _pretokenizer_details(tokenizer: PreTrainedTokenizerBase) -> Dict:
    """Return a dict of structured pre_tokenizer columns."""
    base = {
        "pre_tokenizer": "unknown",
        "pt_byte_level": "no",
        "pt_metaspace": "no",
        "pt_unicode_scripts": "no",
        "pt_digit_split": "no",
        "pt_contraction_split": "no",
        "pt_n_word_splits": 0,
    }
    try:
        pt = tokenizer.backend_tokenizer.pre_tokenizer
        if pt is None:
            base["pre_tokenizer"] = "none"
            return base
        full = json.loads(tokenizer.backend_tokenizer.to_str())
        pt_dict = full.get("pre_tokenizer") or {}
        if not pt_dict:
            base["pre_tokenizer"] = type(pt).__name__
            return base

        leaves = _flatten_pt_dicts(pt_dict)
        word_split_patterns: List[str] = []
        word_splits = 0
        for leaf in leaves:
            t = leaf.get("type", "")
            if t == "ByteLevel":
                base["pt_byte_level"] = str(leaf.get("add_prefix_space", False))
            elif t == "Metaspace":
                base["pt_metaspace"] = f"repl={leaf.get('replacement','▁')!r},prefix={leaf.get('add_prefix_space',True)}"
            elif t == "UnicodeScripts":
                base["pt_unicode_scripts"] = "yes"
            elif t == "Split":
                regex = (leaf.get("pattern") or {}).get("Regex", "")
                has_digit = bool(_DIGIT_RE.search(regex))
                has_contraction = bool(_CONTRACTION_RE.search(regex))
                if has_digit and not has_contraction:
                    base["pt_digit_split"] = regex
                elif has_contraction and not has_digit:
                    base["pt_contraction_split"] = regex
                else:
                    # combined or pure word split — store pattern, mark embedded features
                    if has_digit:
                        base["pt_digit_split"] = "yes"
                    if has_contraction:
                        base["pt_contraction_split"] = "yes"
                    word_split_patterns.append(regex or "?")
                    word_splits += 1
            else:
                # Whitespace / Punctuation / Digits / etc — include as-is
                word_split_patterns.extend(_extract_pretokenizer_components(leaf))
                word_splits += 1
        base["pre_tokenizer"] = " | ".join(word_split_patterns) or "none"
        base["pt_n_word_splits"] = word_splits
    except Exception:
        pass
    return base


def _normalizer_info(tokenizer: PreTrainedTokenizerBase) -> str:
    try:
        norm = tokenizer.backend_tokenizer.normalizer
        if norm is None:
            return "none"
        name = type(norm).__name__
        try:
            import json as _json
            d = _json.loads(norm.__str__())
            name = d.get("type", name)
        except Exception:
            pass
        return name
    except Exception:
        return "unknown"


def _encode_len(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _check_numbers(tokenizer: PreTrainedTokenizerBase) -> str:
    """Token counts for 1- through 5-digit numbers: e.g. '1/1/1/2/2'."""
    samples = ["1", "12", "123", "1234", "12345"]
    return "/".join(str(_encode_len(tokenizer, s)) for s in samples)


def _check_contractions(tokenizer: PreTrainedTokenizerBase) -> str:
    """Token counts for common English contractions: e.g. don't=2 it's=2 I'll=2."""
    samples = [("don't", "don't"), ("it's", "it's"), ("I'll", "I'll")]
    parts = []
    for label, text in samples:
        parts.append(f"{label}={_encode_len(tokenizer, text)}")
    return " ".join(parts)


def _check_whitespace(tokenizer: PreTrainedTokenizerBase) -> str:
    """Token counts for leading / trailing / internal whitespace scenarios."""
    cases = [
        (" hello",    "lead"),
        ("hello ",    "trail"),
        (" hello ",   "both"),
        ("hello world", "mid"),
    ]
    return " ".join(f"{label}={_encode_len(tokenizer, text)}" for text, label in cases)


def _check_zerowidth(tokenizer: PreTrainedTokenizerBase) -> str:
    """How ZWSP / ZWNJ / ZWJ are tokenised (token count or UNK)."""
    chars = [("\u200b", "ZWSP"), ("\u200c", "ZWNJ"), ("\u200d", "ZWJ")]
    unk_id = tokenizer.unk_token_id
    parts = []
    for char, label in chars:
        ids = tokenizer.encode(char, add_special_tokens=False)
        has_unk = (unk_id is not None) and (unk_id in ids)
        parts.append(f"{label}={'UNK' if has_unk else len(ids)}")
    return " ".join(parts)


def extract_info(tok_path: str) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    name = tok_path.split("/")[-1]
    return {
        "name":             name,
        "path":             tok_path,
        "vocab_size":       tokenizer.vocab_size,
        "vocab_size_actual": len(tokenizer.vocab),
        "added_tokens":     len(tokenizer.added_tokens_encoder),
        "n_special":        len(tokenizer.all_special_tokens),
        "special_tokens":   ", ".join(tokenizer.all_special_tokens[:10]),
        **_pretokenizer_details(tokenizer),
        "normalizer":       _normalizer_info(tokenizer),
        "numbers_1-5d":     _check_numbers(tokenizer),
        "contractions":     _check_contractions(tokenizer),
        "whitespace":       _check_whitespace(tokenizer),
        "zerowidth":        _check_zerowidth(tokenizer),
    }


# ── pretty-print table ────────────────────────────────────────────────────────

_TABLE_FIELDS = [
    "name", "vocab_size", "vocab_size_actual", "added_tokens", "n_special",
    "pt_byte_level", "pt_metaspace", "pt_unicode_scripts",
    "pt_digit_split", "pt_contraction_split", "pt_n_word_splits",
    "normalizer",
    "numbers_1-5d", "contractions", "whitespace", "zerowidth",
]

_COL_MAX_WIDTH: Dict[str, int] = {
    "name": 30,
    "normalizer": 20,
    "contractions": 25,
    "whitespace": 30,
    "zerowidth": 20,
}
_DEFAULT_MAX_WIDTH = 20


def _trunc(value: str, width: int) -> str:
    return value if len(value) <= width else value[: width - 1] + "…"


def _print_table(results: List[Dict]) -> None:
    max_widths = {
        f: _COL_MAX_WIDTH.get(f, _DEFAULT_MAX_WIDTH) for f in _TABLE_FIELDS
    }
    widths = {
        f: min(max_widths[f], max(len(f), max(len(str(r.get(f, ""))) for r in results)))
        for f in _TABLE_FIELDS
    }
    header = "  ".join(f"{f:<{widths[f]}}" for f in _TABLE_FIELDS)
    sep    = "-" * len(header)
    print("\n" + header)
    print(sep)
    for r in results:
        row = "  ".join(
            f"{_trunc(str(r.get(f, '')), widths[f]):<{widths[f]}}"
            for f in _TABLE_FIELDS
        )
        print(row)
    print()


# ── args / main ───────────────────────────────────────────────────────────────

@dataclass
class TokenizerTable(Args):
    tokenizers: List[str] = field(default_factory=list)
    output_csv: Optional[str] = None


def main(args: TokenizerTable) -> None:
    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")

    results: List[Dict] = []
    for tok_path in args.tokenizers:
        logger.info("Loading %s", tok_path)
        try:
            info = extract_info(tok_path)
            results.append(info)
        except Exception as exc:
            logger.error("Failed to load %s: %s", tok_path, exc)

    if not results:
        logger.error("No tokenizers loaded successfully.")
        return

    _print_table(results)

    if args.output_csv:
        all_fields = list(results[0].keys())
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved to %s", args.output_csv)


if __name__ == "__main__":
    args = load_args(TokenizerTable)
    main(args)
