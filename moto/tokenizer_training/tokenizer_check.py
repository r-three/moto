"""
Written with Claude Code
tokenizer_check.py

Quick health-check for trained tokenizers:
  1. Vocabulary statistics — size, token-length distributions, per-script character
     breakdown of the vocab surface forms (mirrors assess_data_coverage.py logic).
  2. Integrity checks — corrupt byte sequences, invalid Unicode, null bytes,
     surrogates, lone combining marks, excessive repetition.
  3. Fertility and parity on a subset of the FLORES-200 dataset compared to a reference
     tokenizer (default: meta-llama/Llama-3.2-1B).

Usage:
    langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
    tokenizer_str="meta-llama/Llama-3.2-1B,"
    for lang in ${langs[@]}; do
    for vs in 4000 8000 16000; do
        tokenizer_str+="flexitok/bpe_ltr_${lang}_${vs}_v2,"
    done; done
    python moto/tokenizer_training/tokenizer_check.py \
        tokenizers=[${tokenizer_str%,}] \
        number_handling=ltr_3digit \
        ref_tokenizer=meta-llama/Meta-Llama-3.2-1B \
        flores_dataset=flexitok/flores-21langs \
        flores_split=devtest \
        sample_size=500 \
        output=explorations/results/tokenizer_check.tsv

    python moto/tokenizer_training/tokenizer_check.py \
        tokenizers=[flexitok/bpe_boundless_AllL_128000] \
        number_handling=ltr_3digit \
        ref_tokenizer=meta-llama/Meta-Llama-3.2-1B \
        flores_dataset=flexitok/flores-21langs \
        flores_split=devtest \
        sample_size=500 

        
    python moto/tokenizer_training/tokenizer_check.py \
        tokenizers=[meta-llama/Meta-Llama-3.2-1B,flexitok/bpe_ltr_tur_Latn_1000_v2,flexitok/bpe_ltr_vie_Latn_1000_v2] \
        number_handling=ltr_3digit \
        ref_tokenizer=meta-llama/Meta-Llama-3.2-1B \
        flores_dataset=flexitok/flores-21langs \
        flores_split=devtest \
        sample_size=500 \
        output=explorations/results/tokenizer_check.tsv

Notes:
    - Fertility: subword tokens per word (lower is better; 1.0 = perfect).
    - Parity: tokens(lang) / tokens(ref_lang) on parallel text; 1.0 = equal to ref.
      The parity column header shows ratios vs. the reference *tokenizer*, not just
      the reference language — so each tokenizer is compared to llama's score on the
      same language, giving a relative-to-llama fertility per language.
"""

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset  # type: ignore[import-untyped]
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from moto.tokenizer_training.train_tokenizers import get_digit_group_tokens
from moto.utils.args import Args, load_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Script patterns (reused from assess_data_coverage.py) ─────────────────────
_SCRIPT_PATTERNS: dict[str, re.Pattern] = {
    "Latin":      re.compile(r"[a-zA-Z\u00c0-\u024f]"),
    "CJK_Han":    re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]"),
    "Hiragana":   re.compile(r"[\u3041-\u3096]"),
    "Katakana":   re.compile(r"[\u30a1-\u30fa]"),
    "Cyrillic":   re.compile(r"[\u0400-\u04ff]"),
    "Arabic":     re.compile(r"[\u0600-\u06ff\u0750-\u077f]"),
    "Hebrew":     re.compile(r"[\u0590-\u05ff]"),
    # "Devanagari": re.compile(r"[\u0900-\u097f]"),
    # "Korean":     re.compile(r"[\uac00-\ud7af\u1100-\u11ff]"),
    # "Thai":       re.compile(r"[\u0e00-\u0e7f]"),
    "Greek":      re.compile(r"[\u0370-\u03ff]"),
    "Digit":      re.compile(r"[0-9]"),
}

# Integrity-check patterns
_RE_NULL      = re.compile(r"\x00")
_RE_SURROGATE = re.compile(r"[\ud800-\udfff]")
_RE_REPLACE   = re.compile(r"\ufffd")
_RE_CONTROL   = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")  # excludes \t \n \r
_RE_LONE_COMBINING = re.compile(r"^[\u0300-\u036f\u0610-\u061a\u064b-\u065f]+$")

# CJK scripts that use character-level "word" segmentation for fertility
_CJK_SCRIPTS = {"Hani", "Jpan", "Kore", "Hang"}


# ── Vocabulary analysis ────────────────────────────────────────────────────────

def _get_token_surface(tokenizer: PreTrainedTokenizerBase, token_id: int) -> Optional[str]:
    """Return the properly decoded surface for a single token id.

    Uses convert_tokens_to_string to handle:
      - GPT-2 / LLaMA byte-level encoding (Ġ, Ú, Ĩ … are byte proxies, not real chars)
      - SentencePiece ▁ prefix
      - <0x??> byte tokens (LLaMA-style fallback bytes)

    Returns None if the token decodes to something that raises an error.
    """
    token_str = tokenizer.convert_ids_to_tokens(token_id)
    if not isinstance(token_str, str):
        return None
    try:
        surface = tokenizer.convert_tokens_to_string([token_str])
    except Exception:
        return None
    return surface


def analyze_vocabulary(
    tokenizer: PreTrainedTokenizerBase,
    tok_name: str,
) -> dict:
    """Compute vocab-level stats and integrity flags."""
    vocab_size = tokenizer.vocab_size
    n_added = len(tokenizer.added_tokens_encoder)

    # Collect per-token info
    char_counts: dict[str, int] = defaultdict(int)
    total_vocab_chars = 0
    token_char_lengths: List[int] = []
    token_byte_lengths: List[int] = []

    issues: dict[str, List[int]] = {
        "null_byte":        [],
        "surrogate":        [],
        "replacement_char": [],
        "control_char":     [],
        "lone_combining":   [],
        "invalid_utf8_byte_token": [],
    }

    for tid in range(vocab_size):
        surface = _get_token_surface(tokenizer, tid)
        if surface is None:
            # Raw byte token that's not valid UTF-8 on its own — flag but don't crash
            issues["invalid_utf8_byte_token"].append(tid)
            continue

        # Skip pure special tokens (they appear only in added_tokens)
        raw = tokenizer.convert_ids_to_tokens(tid)
        if raw in tokenizer.all_special_tokens:
            continue

        char_len = len(surface)
        byte_len = len(surface.encode("utf-8", errors="replace"))
        token_char_lengths.append(char_len)
        token_byte_lengths.append(byte_len)
        total_vocab_chars += char_len

        for name, pat in _SCRIPT_PATTERNS.items():
            char_counts[name] += len(pat.findall(surface))

        # Integrity checks
        if _RE_NULL.search(surface):
            issues["null_byte"].append(tid)
        if _RE_SURROGATE.search(surface):
            issues["surrogate"].append(tid)
        if _RE_REPLACE.search(surface):
            issues["replacement_char"].append(tid)
        if _RE_CONTROL.search(surface):
            issues["control_char"].append(tid)
        if _RE_LONE_COMBINING.match(surface):
            issues["lone_combining"].append(tid)

    char_arr = np.array(token_char_lengths) if token_char_lengths else np.array([0])
    byte_arr = np.array(token_byte_lengths) if token_byte_lengths else np.array([0])

    return {
        "name":           tok_name,
        "vocab_size":     vocab_size,
        "added_tokens":   n_added,
        "char_counts":    dict(char_counts),
        "total_vocab_chars": total_vocab_chars,
        # Token length distributions
        "char_len_mean":  float(char_arr.mean()),
        "char_len_median": float(np.median(char_arr)),
        "char_len_max":   int(char_arr.max()),
        "byte_len_mean":  float(byte_arr.mean()),
        "byte_len_median": float(np.median(byte_arr)),
        "byte_len_max":   int(byte_arr.max()),
        # Single-char / multi-char counts
        "n_single_char":  int((char_arr == 1).sum()),
        "n_multi_char":   int((char_arr > 1).sum()),
        "issues":         {k: len(v) for k, v in issues.items()},
        "issue_ids":      issues,
    }


def print_vocab_report(results: List[dict]) -> None:
    scripts = list(_SCRIPT_PATTERNS.keys())
    col = max(len(r["name"]) for r in results)

    print("\n── Vocabulary Summary ──")
    hdr = (f"{'tokenizer':<{col}}  {'vocab_sz':>8}  {'added':>6}  "
           f"{'ch_mean':>7}  {'ch_med':>6}  {'ch_max':>6}  "
           f"{'by_mean':>7}  {'by_med':>6}  {'n_1ch':>7}  {'n_>1ch':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<{col}}  {r['vocab_size']:>8,}  {r['added_tokens']:>6}  "
            f"{r['char_len_mean']:>7.2f}  {r['char_len_median']:>6.1f}  {r['char_len_max']:>6}  "
            f"{r['byte_len_mean']:>7.2f}  {r['byte_len_median']:>6.1f}  "
            f"{r['n_single_char']:>7,}  {r['n_multi_char']:>7,}"
        )

    print("\n── Script Breakdown (% of vocab chars) ──")
    hdr2 = f"{'tokenizer':<{col}}" + "".join(f"  {s[:8]:>8}" for s in scripts)
    print(hdr2)
    print("-" * len(hdr2))
    for r in results:
        total = r["total_vocab_chars"] or 1
        row = f"{r['name']:<{col}}"
        for s in scripts:
            pct = r["char_counts"].get(s, 0) / total * 100
            row += f"  {pct:>7.2f}%"
        print(row)

    print("\n── Integrity Issues ──")
    issue_keys = ["null_byte", "surrogate", "replacement_char",
                  "control_char", "lone_combining", "invalid_utf8_byte_token"]
    hdr3 = f"{'tokenizer':<{col}}" + "".join(f"  {k[:14]:>14}" for k in issue_keys)
    print(hdr3)
    print("-" * len(hdr3))
    for r in results:
        row = f"{r['name']:<{col}}"
        for k in issue_keys:
            n = r["issues"].get(k, 0)
            flag = f"{n:>14,}" if n == 0 else f"{'!'+str(n):>14}"
            row += f"  {flag}"
        print(row)

    # Print example problem tokens if any issues
    for r in results:
        for k, ids in r["issue_ids"].items():
            if ids:
                tok = AutoTokenizer.from_pretrained(r.get("_path", r["name"]))
                examples = [repr(tok.convert_ids_to_tokens(i)) for i in ids[:5]]
                logger.warning("[%s] %s: %d token(s) — e.g. %s",
                               r["name"], k, len(ids), ", ".join(examples))


# ── FLORES loading ─────────────────────────────────────────────────────────────

def _script_of(lang_code: str) -> str:
    parts = lang_code.split("_")
    return parts[1] if len(parts) == 2 else ""


def _whitespace_words(text: str) -> List[str]:
    return [
        tok for tok in re.split(r"\s+", text.strip())
        if tok and any(unicodedata.category(c)[0] in ("L", "N") for c in tok)
    ]


def _character_words(text: str) -> List[str]:
    return [c for c in text if unicodedata.category(c)[0] in ("L", "N")]


def _load_local_flores(
    dataset_name: str,
    langs: List[str],
    split: str,
    sample_size: int,
    seed: int,
    text_field: str,
) -> Dict[str, List[str]]:
    """Load from a local/private HF dataset that has one config per language."""
    import random
    lang_sentences: Dict[str, List[str]] = {}
    for lang in langs:
        logger.debug("Loading %s config=%s split=%s", dataset_name, lang, split)
        try:
            ds = load_dataset(dataset_name, lang, split=split)
            texts = [ex[text_field] for ex in ds if ex.get(text_field, "").strip()]
        except Exception:
            # Fallback: dataset may store all langs in one config with a "lang" field
            logger.info("Config-per-lang failed; trying flat load with lang filter")
            ds = load_dataset(dataset_name, split=split)
            texts = [
                ex[text_field] for ex in ds
                if ex.get("lang") == lang and ex.get(text_field, "").strip()
            ]
        lang_sentences[lang] = texts

    # Align indices (parallel corpus)
    n = min((len(s) for s in lang_sentences.values()), default=0)
    if n == 0:
        raise ValueError(f"No examples found for {dataset_name}. Check lang codes and split.")

    random.seed(seed)
    import random as _random
    indices = _random.sample(range(n), min(sample_size, n))
    return {lang: [sents[i] for i in indices] for lang, sents in lang_sentences.items()}


# ── Fertility ──────────────────────────────────────────────────────────────────

def compute_fertility(
    tokenizer: PreTrainedTokenizerBase,
    lang_texts: Dict[str, List[str]],
) -> Dict[str, float]:
    """Fertility = Σ subword_tokens / Σ words.  Lower is better."""
    scores: Dict[str, float] = {}
    for lang, texts in lang_texts.items():
        word_fn = _character_words if _script_of(lang) in _CJK_SCRIPTS else _whitespace_words
        total_words = total_subtoks = 0
        for text in texts:
            words = word_fn(text)
            total_words += len(words)
            for w in words:
                total_subtoks += len(tokenizer.encode(w, add_special_tokens=False))
        scores[lang] = round(total_subtoks / total_words, 4) if total_words else 0.0
    return scores


# ── Parity ────────────────────────────────────────────────────────────────────

def compute_parity(
    tokenizer: PreTrainedTokenizerBase,
    ref_tokenizer: PreTrainedTokenizerBase,
    lang_texts: Dict[str, List[str]],
    ref_lang: str = "eng_Latn",
) -> Dict[str, float]:
    """
    Parity vs. reference tokenizer:
        parity(lang) = mean_fertility(lang, tok) / mean_fertility(lang, ref_tok)

    A score < 1.0 means this tokenizer is *more efficient* than the reference on
    that language; > 1.0 means it needs more tokens.
    """
    scores: Dict[str, float] = {}
    for lang, texts in lang_texts.items():
        tok_lengths  = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        ref_lengths  = [len(ref_tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        ratios = [t / r for t, r in zip(tok_lengths, ref_lengths) if r > 0]
        scores[lang] = round(float(np.mean(ratios)), 4) if ratios else 0.0
    return scores


# ── Output ────────────────────────────────────────────────────────────────────

def print_flores_report(
    fertility:   Dict[str, Dict[str, float]],  # {tok_name: {lang: score}}
    parity:      Dict[str, Dict[str, float]],
    ref_name:    str,
    langs:       List[str],
) -> None:

    def _table(title: str, data: Dict[str, Dict[str, float]], note: str = "") -> None:
        all_names = list(data.keys())
        col = max(len(n) for n in all_names + [ref_name])
        print(f"\n── {title} {note}──")
        hdr = f"{'tokenizer':<{col}}" + "".join(f"  {lang[:10]:>10}" for lang in langs)
        print(hdr)
        print("-" * len(hdr))
        for name in all_names:
            row = f"{name:<{col}}"
            for lang in langs:
                v = data[name].get(lang, float("nan"))
                row += f"  {v:>10.4f}"
            print(row)

    _table("Fertility (tokens/word, lower=better)", fertility)
    _table("Parity vs.", parity, note=f"[{ref_name}] (ratio; 1.0=same efficiency) ")


def write_tsv(
    vocab_results:  List[dict],
    fertility:      Dict[str, Dict[str, float]],
    parity:         Dict[str, Dict[str, float]],
    langs:          List[str],
    output_path:    str,
) -> None:
    scripts = list(_SCRIPT_PATTERNS.keys())
    issue_keys = ["null_byte", "surrogate", "replacement_char",
                  "control_char", "lone_combining", "invalid_utf8_byte_token"]
    fert_cols   = [f"fertility_{l}" for l in langs]
    parity_cols = [f"parity_{l}"   for l in langs]

    cols = (
        ["tokenizer", "vocab_size", "added_tokens",
         "char_len_mean", "char_len_median", "char_len_max",
         "byte_len_mean", "byte_len_median", "byte_len_max",
         "n_single_char", "n_multi_char"]
        + [f"vocab_pct_{s}" for s in scripts]
        + [f"issue_{k}" for k in issue_keys]
        + fert_cols + parity_cols
    )

    rows = []
    for r in vocab_results:
        total = r["total_vocab_chars"] or 1
        row = {
            "tokenizer":      r["name"],
            "vocab_size":     r["vocab_size"],
            "added_tokens":   r["added_tokens"],
            "char_len_mean":  f"{r['char_len_mean']:.4f}",
            "char_len_median": f"{r['char_len_median']:.1f}",
            "char_len_max":   r["char_len_max"],
            "byte_len_mean":  f"{r['byte_len_mean']:.4f}",
            "byte_len_median": f"{r['byte_len_median']:.1f}",
            "byte_len_max":   r["byte_len_max"],
            "n_single_char":  r["n_single_char"],
            "n_multi_char":   r["n_multi_char"],
        }
        for s in scripts:
            row[f"vocab_pct_{s}"] = f"{r['char_counts'].get(s, 0) / total * 100:.4f}"
        for k in issue_keys:
            row[f"issue_{k}"] = r["issues"].get(k, 0)
        name = r["name"]
        for l in langs:
            row[f"fertility_{l}"] = fertility.get(name, {}).get(l, "")
            row[f"parity_{l}"]    = parity.get(name, {}).get(l, "")
        rows.append(row)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    logger.info("TSV written to %s", output_path)


def check_lossless_decoding(tokenizer, lang_texts):
    """Check if tokenizer can round-trip decode the original sentences without loss."""
    issues = []
    for text in lang_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, clean_up_tokenization_spaces=False)
        if decoded != text:
            issues.append((text, decoded))
    return issues


def check_byte_fallback(tokenizer: PreTrainedTokenizerBase) -> dict:
    """Verify byte-level fallback: no input should produce UNK tokens.

    For byte-level BPE tokenizers, every byte maps to a vocab entry so UNK
    should never appear.  Failures here indicate a byte-fallback gap or a
    tokenizer type that doesn't cover the full byte range.
    """
    unk_id = tokenizer.unk_token_id
    byte_issues: List[int] = []
    # Test all 256 byte values via latin-1 (bijective 0-255 mapping)
    for byte_val in range(256):
        char = bytes([byte_val]).decode("latin-1")
        ids = tokenizer.encode(char, add_special_tokens=False)
        if unk_id in ids:
            byte_issues.append(byte_val)

    if unk_id is None:
        return {"has_unk_token": False, "byte_fallback_ok": True, "issues": []}


    string_issues: List[str] = []
    test_strings = [
        "Hello, world!",
        "こんにちは",   # Japanese
        "中文",          # Chinese
        "Привет",       # Russian / Cyrillic
        "مرحبا",        # Arabic
        "αβγδ",         # Greek
        "\t\n\r",       # ASCII control
        "🎉🤖",         # emoji (4-byte UTF-8)
    ]
    for s in test_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if unk_id in ids:
            toks = _decoded_tokens(tokenizer, ids)
            unk_pos = [i for i, id_ in enumerate(ids) if id_ == unk_id]
            string_issues.append(f"'{s}' → UNK at positions {unk_pos} tokens={toks}")

    return {
        "has_unk_token": True,
        "unk_id": unk_id,
        "byte_fallback_ok": not byte_issues and not string_issues,
        "byte_issues": byte_issues,
        "string_issues": string_issues,
    }


def _expected_number_splits(num_str: str, number_handling: str) -> Optional[List[str]]:
    """Return the expected pre-tokenizer split of a digit string.

    Returns None for "learned" (no expectation).
    """
    if number_handling == "learned":
        return None
    if number_handling == "individual":
        return list(num_str)
    m = re.match(r"ltr_(\d+)digit", number_handling)
    if m:
        n = int(m.group(1))
        return re.findall(rf"\d{{1,{n}}}", num_str)
    m = re.match(r"rtl_(\d+)digit", number_handling)
    if m:
        n = int(m.group(1))
        # Right-aligned groups: leftmost group absorbs the remainder digits
        length = len(num_str)
        first = length % n or n
        groups = [num_str[:first]]
        pos = first
        while pos < length:
            groups.append(num_str[pos : pos + n])
            pos += n
        return groups
    return None


def _decoded_tokens(tokenizer: PreTrainedTokenizerBase, ids: List[int]) -> List[str]:
    """Return surface forms for a list of token IDs, always as a list."""
    result = tokenizer.convert_ids_to_tokens(ids)
    if isinstance(result, list):
        return result
    if isinstance(result, str):
        return [result]
    return []


def check_number_handling(
    tokenizer: PreTrainedTokenizerBase,
    number_handling: str,
) -> dict:
    """Verify that numbers are split / tokenised as expected.

    For any number_handling scheme other than "learned":
    - Every expected digit-group token should be a single token (not split further
      by BPE and not produce UNK).
    - Test numbers are encoded and compared against the expected pre-tokenizer splits
      (e.g. ltr_3digit: "0987" → ["098", "7"]).

    Sample encodings for a representative set of numbers are always returned.
    """
    unk_id = tokenizer.unk_token_id

    # Samples that exercise various splitting boundaries
    sample_strings = ["1", "12", "123", "1234", "12345", "0987", "100", "1000",
                      "10000", "007", "042", "100200300"]
    sample_encodings: dict[str, dict] = {}
    for s in sample_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        toks = _decoded_tokens(tokenizer, ids)
        expected = _expected_number_splits(s, number_handling)
        match = (toks == expected) if expected is not None else None
        sample_encodings[s] = {"tokens": toks, "expected": expected, "match": match}

    if number_handling == "learned":
        return {
            "number_handling": number_handling,
            "missing_from_vocab": [],
            "unk_when_encoding": [],
            "split_mismatches": [],
            "sample_encodings": sample_encodings,
        }

    expected_tokens = get_digit_group_tokens(number_handling)

    # Check that each expected single-token digit group is indeed a single token
    # (byte-level BPE encodes without a leading space for digit-only strings).
    missing_from_vocab: List[str] = []
    unk_when_encoding: List[str] = []
    for tok in sorted(expected_tokens):
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if unk_id is not None and unk_id in ids:
            unk_when_encoding.append(tok)
        elif len(ids) != 1:
            missing_from_vocab.append(tok)  # present but splits into multiple tokens

    # Check split behaviour on the sample strings
    split_mismatches: List[dict] = [
        {"input": s, "got": info["tokens"], "expected": info["expected"]}
        for s, info in sample_encodings.items()
        if info["match"] is False
    ]

    return {
        "number_handling": number_handling,
        "expected_token_count": len(expected_tokens),
        "missing_from_vocab": missing_from_vocab,
        "unk_when_encoding": unk_when_encoding,
        "split_mismatches": split_mismatches,
        "sample_encodings": sample_encodings,
    }


def print_number_check_report(results: dict[str, dict]) -> None:
    """Print a compact number-handling check table."""
    print("\n── Number Handling Check ──")
    for tok_name, r in results.items():
        nh = r["number_handling"]
        n_missing = len(r["missing_from_vocab"])
        n_unk = len(r["unk_when_encoding"])
        n_mismatch = len(r.get("split_mismatches", []))
        ok = n_missing == 0 and n_unk == 0 and n_mismatch == 0
        status = "OK" if ok else (
            f"!{n_missing} not-single-tok  !{n_unk} UNK  !{n_mismatch} split-mismatch"
        )
        print(f"  {tok_name}  [{nh}]  {status}")
        if r["missing_from_vocab"]:
            logger.warning("[%s] %d expected single-token digit groups split into multiple tokens, e.g. %s",
                           tok_name, n_missing, r["missing_from_vocab"][:10])
        if r["unk_when_encoding"]:
            logger.warning("[%s] %d expected tokens encode to UNK, e.g. %s",
                           tok_name, n_unk, r["unk_when_encoding"][:10])
        for mm in r.get("split_mismatches", []):
            logger.warning("[%s] split mismatch: %r  got=%s  expected=%s",
                           tok_name, mm["input"], mm["got"], mm["expected"])
        # print(f"    Sample encodings:")
        # for s, info in r["sample_encodings"].items():
        #     toks = info["tokens"]
        #     match = info["match"]
        #     marker = "" if match is None else (" OK" if match else " MISMATCH")
        #     exp = f"  (expected {info['expected']})" if match is False else ""
        #     print(f"      {s!r:>12} → {toks}{marker}{exp}")


def print_byte_fallback_report(results: dict[str, dict]) -> None:
    """Print a compact byte-fallback check table."""
    print("\n── Byte Fallback / UNK Check ──")
    for tok_name, r in results.items():
        if not r["has_unk_token"]:
            print(f"  {tok_name}  no UNK token defined — skip")
            continue
        ok = r["byte_fallback_ok"]
        status = "OK" if ok else f"!{len(r['byte_issues'])} bad bytes, !{len(r['string_issues'])} bad strings"
        print(f"  {tok_name}  byte_fallback={status}")
        for b in r["byte_issues"][:16]:
            logger.warning("[%s] byte 0x%02x produces UNK", tok_name, b)
        for s in r["string_issues"]:
            logger.warning("[%s] string %s", tok_name, s)

# ── Args / main ───────────────────────────────────────────────────────────────

@dataclass
class TokenizerCheckArgs(Args):
    # One or more HF repo IDs or local paths to check
    tokenizers: List[str] = field(default_factory=list)
    # Reference tokenizer for parity (llama by default)
    ref_tokenizer: str = "meta-llama/Llama-3.2-1B"
    # FLORES dataset: private upload at flexitok/flores-21langs
    flores_dataset: str = "flexitok/flores-21langs"
    flores_split: str = "devtest"
    # FLORES field name — "sentence" in flores-200 style, "text" in flores_plus
    flores_text_field: str = "sentence"
    # Subset of langs (empty = all configs in the dataset); FLORES lang codes
    langs: List[str] = field(default_factory=list)
    ref_lang: str = "eng_Latn"
    sample_size: int = 500
    seed: int = 42
    output: str = "explorations/results/tokenizer_check.tsv"
    # Whether to print example problem tokens in the integrity report
    verbose_issues: bool = True
    # Expected number-splitting scheme (mirrors train_tokenizers.py number_handling).
    # "learned" = no expectation; any other value (e.g. "ltr_3digit", "individual")
    # checks that each expected digit group is a single token and never produces UNK.
    number_handling: str = "learned"


def main(args: TokenizerCheckArgs) -> None:
    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Vocabulary stats & integrity ───────────────────────────────────────
    logger.info("=== Vocabulary analysis ===")
    vocab_results: List[dict] = []
    loaded_tokenizers: Dict[str, PreTrainedTokenizerBase] = {}

    for tok_path in args.tokenizers:
        logger.info("Loading tokenizer: %s", tok_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except Exception as e:
            logger.error("Failed to load tokenizer from %s: %s", tok_path, e)
            continue
        tok_name = Path(tok_path).name if Path(tok_path).exists() else tok_path.split("/")[-1]
        loaded_tokenizers[tok_name] = tokenizer

        result = analyze_vocabulary(tokenizer, tok_name)
        result["_path"] = tok_path  # store path for error reporting
        vocab_results.append(result)

    print_vocab_report(vocab_results)

    # ── 2. Fertility & parity on FLORES ───────────────────────────────────────
    logger.info("=== FLORES fertility / parity ===")

    # Determine which languages to evaluate
    langs = list(args.langs)
    if not langs:
        # Default to the 21 languages in the dataset (inferred from recreate_flores200.py)
        langs = [
            "arb_Arab", "ces_Latn", "cmn_Hani", "dan_Latn", "deu_Latn", "ell_Grek",
            "fas_Arab", "fra_Latn", "eng_Latn", "hun_Latn", "ind_Latn", "ita_Latn",
            "jpn_Jpan", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl", "spa_Latn",
            "swe_Latn", "tur_Latn", "vie_Latn",
        ]

    # Load FLORES sentences (load once, reuse across tokenizers)
    logger.info("Loading FLORES data from %s (%s)", args.flores_dataset, args.flores_split)
    try:
        lang_texts = _load_local_flores(
            dataset_name=args.flores_dataset,
            langs=langs,
            split=args.flores_split,
            sample_size=args.sample_size,
            seed=args.seed,
            text_field=args.flores_text_field,
        )
    except Exception as e:
        logger.error("Failed to load FLORES dataset: %s", e)
        lang_texts = {}

    # ── 3. Check if lossless decoding happens ─────────────────────────────────
    logger.info("=== Lossless decoding check ===")
    for tok_name, tokenizer in loaded_tokenizers.items():
        logger.info("Checking lossless decoding for %s", tok_name)
        match = re.search(r"[a-z]{3}_[A-Za-z]{4}", tok_name)
        lang_code = match.group() if match else "eng_Latn"
        if "fw_edu" in tok_name:
            lang_code = "eng_Latn"
        texts_for_lang = lang_texts.get(lang_code, [])
        lossless_issues = check_lossless_decoding(tokenizer, texts_for_lang)
        if lossless_issues:
            logger.warning("[%s] Found %d/%d examples in %s where decoded text differs from original.",
                           tok_name, len(lossless_issues), len(texts_for_lang), lang_code)
        else:
            logger.info("[%s] Lossless decoding OK for %s (%d examples)",
                        tok_name, lang_code, len(texts_for_lang))

    # ── 4. Byte fallback / UNK check ──────────────────────────────────────────
    logger.info("=== Byte fallback / UNK check ===")
    byte_fallback_results: dict[str, dict] = {}
    for tok_name, tokenizer in loaded_tokenizers.items():
        byte_fallback_results[tok_name] = check_byte_fallback(tokenizer)
    print_byte_fallback_report(byte_fallback_results)

    # ── 5. Number handling check ───────────────────────────────────────────────
    logger.info("=== Number handling check (scheme=%s) ===", args.number_handling)
    number_check_results: dict[str, dict] = {}
    for tok_name, tokenizer in loaded_tokenizers.items():
        number_check_results[tok_name] = check_number_handling(tokenizer, args.number_handling)
    print_number_check_report(number_check_results)

    exit(0)

    # Load reference tokenizer
    logger.info("Loading reference tokenizer: %s", args.ref_tokenizer)
    ref_tokenizer = AutoTokenizer.from_pretrained(args.ref_tokenizer)
    ref_name = args.ref_tokenizer.split("/")[-1]

    fertility:  Dict[str, Dict[str, float]] = {}
    parity:     Dict[str, Dict[str, float]] = {}

    if lang_texts:
        for tok_name, tokenizer in loaded_tokenizers.items():
            logger.info("Computing fertility for %s", tok_name)
            fertility[tok_name] = compute_fertility(tokenizer, lang_texts)

            logger.info("Computing parity vs. %s for %s", ref_name, tok_name)
            parity[tok_name] = compute_parity(tokenizer, ref_tokenizer, lang_texts, args.ref_lang)

        fertility[args.ref_tokenizer] = compute_fertility(ref_tokenizer, lang_texts)
        # parity[args.ref_tokenizer] = 1.
        print_flores_report(fertility, parity, ref_name, langs)


    # ── 4. Write TSV ──────────────────────────────────────────────────────────
    write_tsv(vocab_results, fertility, parity, langs, args.output)


if __name__ == "__main__":
    args = load_args(TokenizerCheckArgs)
    main(args)
