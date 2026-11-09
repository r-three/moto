"""
Train BPE or UnigramLM tokenizers on per-language JSONL data.

Usage:
    python -m moto.tokenizer_training.train_tokenizers \
        algorithm=bpe vocab_size=32000 langs='[ita_Latn,tur_Latn]' \
        data_dir=/path/to/fineweb2_hq output_dir=./trained_tokenizers

Data layout expected: {data_dir}/{lang}/*.jsonl
Each JSONL line must have a "text" field.

TODO: eliminate UNK token
"""

import glob
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from os import path
from pathlib import Path
from typing import Callable, List, Literal, Optional

import regex
import tokenizers
from huggingface_hub import HfApi
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    trainers,
)
from transformers import AddedToken, PreTrainedTokenizerFast

import moto.tokenizer_training.pretokenization as pretok_patterns
from moto.utils.args import Args, HFArgs, load_args
from moto.utils.hf_utils import (
    create_or_get_collection_slug,
    create_or_get_collection_slugs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Pre-tokenizer regex patterns ───────────────────────────────────────────
# GPT-2 / BLOOM style
GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
# GPT-4 / tiktoken cl100k style (already chunks numbers ≤ 3 digits via \p{N}{1,3})
GPT4_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
# English contractions for explicit split step
CONTRACTION_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d|n't)"
SCRIPT_AWARE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"          # English contractions
    r"|[^\r\n\p{L}\p{N}]?\p{Han}+"            # CJK Han runs
    r"|[^\r\n\p{L}\p{N}]?\p{Hiragana}+"       # Hiragana runs
    r"|[^\r\n\p{L}\p{N}]?\p{Katakana}+"       # Katakana runs
    r"|[^\r\n\p{L}\p{N}]?\p{Arabic}+"         # Arabic script runs
    r"|[^\r\n\p{L}\p{N}]?\p{Cyrillic}+"       # Cyrillic runs
    r"|[^\r\n\p{L}\p{N}]?\p{Latin}+"          # Latin runs
    r"|\p{N}{1,3}"                             # numbers ≤3 digits
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"             # punctuation
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"              # whitespace
)

# Number-handling regexes for composable Split pre-tokenizer
NUMBER_REGEXES = {
    "individual": r"\p{N}",  # every digit gets its own token
    "ltr_2digit": r"\p{N}{1,2}",  # greedy left-to-right chunks of ≤ 2
    "ltr_3digit": r"\p{N}{1,3}",  # greedy left-to-right chunks of ≤ 3
    "ltr_4digit": r"\p{N}{1,4}",  # greedy left-to-right chunks of ≤ 4
    # broken
    "ltr_5digit": r"\p{N}{1,5}",  # greedy left-to-right chunks of ≤ 5
    # re.sub(r"\d{1,3}(?=(\d{3})*\b)", r"\g<0>,", "1234567")
    # '1,234,567,'
    "rtl_3digit": r"\d{1,3}(?=(\d{3})*\b)",
    "rtl_2digit": r"\d{1,2}(?=(\d{2})*\b)",
    "rtl_4digit": r"\d{1,4}(?=(\d{4})*\b)",
    "rtl_5digit": r"\d{1,5}(?=(\d{5})*\b)",
    # pre_tokenizers.Split(pattern = Regex(r"\d{1,3}(?=(\d{3})*\b)"),
    # "rtl_Ndigit" requires lookahead not supported by Rust regex → iterator-level
}
# Zero-width and invisible Unicode characters
ZERO_WIDTH_PATTERN = r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060]"
# CJK character-level isolation: Han (Chinese), Hiragana, Katakana
CJK_PATTERN = r"\p{Han}|\p{Hiragana}|\p{Katakana}"
_HAN_RUN_RE = regex.compile(r"\p{Han}+")
BOUNDLESS_PATTERN = "|".join([
r" ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+", # B-1, contraction
r"_(?:\p{Ll}\p{M}*)+", # B-2, snake_case
r" ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))", # B-3, words
r" ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+", # B-4, words
r" ?(?:\p{Lu}\p{M}*)+", # B-5, words
r" ?(?:[\p{Lt}\p{Lm}\p{Lo}]\p{M}*)+", # B-6, words
r"(?:\p{N}\p{M}*){1,3}(?=(?:(?:\p{N}\p{M}*){3})*(?:(?:\P{N}\p{M}*)|$))",# B-7
r" ?(?:[\p{P}\p{S}]\p{M}*)+", # B-8, punct and symbols
r"[^\S\r\n]*[\n\r]+|[^\S\r\n]+", # B-9, whitespace
r"(?:[\p{Z}\p{C}]\p{M}*)+", # B-10, sep or control
r"\p{M}+" # B-11, leftover marks
])


def _chunk_han_run(m: "re.Match", max_run: int) -> str:
    """Break a Han run into chunks of at most max_run chars, joined by spaces."""
    s = m.group(0)
    return " ".join(s[i : i + max_run] for i in range(0, len(s), max_run))


def make_han_chunker(max_run: int):
    """Return a text transform that caps Han runs at max_run characters."""
    return lambda text: _HAN_RUN_RE.sub(lambda m: _chunk_han_run(m, max_run), text)


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
SPECIAL_TOKENS = [ "<s>", "</s>", "<pad>"]
UNK_TOKEN = "<unk>"
UNK_TOKEN = None  # we will remove UNK token and rely on byte-level fallback instead
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"


@dataclass
class TokenizerTrainingArgs(Args):
    algorithm: str = "bpe"  # "bpe" or "unigram"
    vocab_size: int = 32000
    langs: List[str] = field(default_factory=list)  # e.g. [ita_Latn, tur_Latn]
    data_dir: str = ""  # path containing {lang}/*.jsonl
    data_shard_pattern: str = "*.jsonl"  # glob pattern for training shards, placeholder
    output_dir: str = "./trained_tokenizers"
    min_frequency: int = 2  # only used for BPE
    max_lines: int = -1  # -1 = use all data, else cap total lines read
    # ── Preprocessing ──────────────────────────────────────────────────────
    # todo: add null pretokenizer option that does no splitting at all (for character-level vocab)
    pretokenizer: Optional[str] = "byte_level"
    # "byte_level" | "gpt2" | "gpt4" | "bloom" | "sentencepiece" | "whitespace" | "bert" | custom:{lang_code}
    # byte_level: ByteLevel (current default; internally applies GPT-2 regex)
    # gpt2/bloom: explicit GPT-2 regex + ByteLevel
    # gpt4: GPT-4/tiktoken regex + ByteLevel (numbers always ≤3 digits)
    # sentencepiece: Metaspace prepend_scheme=first
    # whitespace: WhitespaceSplit  |  bert: BertPreTokenizer
    # custom:{lang_code}: use custom regex pattern from pretokenization.py for the specified language (e.g. custom:ita_Latn)
    pretok_behavior: str = "isolated"
     # "isolated" | "contiguous" | "removed"
    # isolated:   matched spans are atomic pre-tokens (current default — whole words)
    # contiguous: matched spans are boundaries only — ByteLevel encodes byte-by-byte within
    # removed:    matched spans are discarded (e.g. for whitespace stripping)
    number_handling: str = "learned"
    # "learned" | "individual" | "ltr_3digit"
    # "rtl_1digit" | "rtl_2digit" | "rtl_3digit" | "rtl_4digit" | "rtl_5digit"
    # Ignored for gpt2/gpt4/bloom/bert (number splitting baked into their pattern).
    # rtl_Ndigit variants are applied in the text iterator (Rust regex lacks lookahead)
    # add_numbers: bool = False  # whether to add individual digits to the vocab 
    add_numbers : bool = False
    byte_fallback: bool = True  # whether to enable byte-level fallback for OOV tokens (only for BPE)
    unk_token: Optional[str] = UNK_TOKEN  # UNK token to use (
    add_cjk_chars: bool = False  # whether to add common-use CJK characters to the initial alphabet (only for BPE)
    cjk_char_coverage: float = 0.999  # coverage of CJK characters to include in the initial alphabet

    handle_contractions: bool = False
    # Prepend a contraction-aware Split before the base pretokenizer.
    # Ignored for gpt2/gpt4/bloom/bert (contractions already in their pattern).
    unicode_normalization: Optional[str] = "nfc"
    # "nfc" | "nfkc" | "nfd" | "nfkd" | "none"
    strip_zero_width: bool = False
    # Strip U+200B/C/D/E/F, U+FEFF, U+00AD, U+2060 before unicode normalization.
    cjk_char_split: bool = False
    # Prepend a Split that isolates each Han/Hiragana/Katakana character before the
    # main pretokenizer.  Prevents \p{L}+ from matching entire CJK sentences as one
    # "word", which causes the BPE word-frequency table to explode in memory.
    max_han_run: int = -1
    # Cap Han runs at this many characters in the text iterator (before the tokenizer
    # sees the text), inserting spaces between chunks.  -1 = no limit.
    # Prevents BPE word-freq table from exploding on spaceless Chinese paragraphs.
    # Recommended: 16 for script_aware pretokenizer (allows multi-char merges while
    # bounding memory).
    extra_special_tokens: List[str] = field(default_factory=list)
    use_byte_level_regex: bool = True
    # Whether to apply the complex byte-level regex (GPT-2 style) that splits on Unicode character boundaries, or to disable it and treat the whole input as bytes.  Disabling it can be useful for training on data that is already mostly ASCII (e.g. numeric data), where the regex overhead may not be worth it.  Note that the byte-level regex is still applied for the "gpt2", "gpt4", and "bloom" pretokenizer options, regardless of this setting, since it's baked into their patterns.

    # Regularized BPE (algorithm=rbpe) hyperparameters
    lambda_voc: float = 0.0   # per-merge cost threshold; stop early when best score < λ_voc
    lambda_tok: float = 0.0   # token-length penalty; discourages merging long subwords
    lambda_seg: float = 1.0   # frequency reward scale; 1.0 = standard BPE ranking

    # Additional special tokens added on top of the defaults (<unk>, <s>, </s>, <pad>).
    # e.g. extra_special_tokens='["<sep>","<cls>"]'
    # HuggingFace Hub publishing
    hf_repo_prefix: Optional[str] = (
        None  # e.g. "gsaltintas/flexitok" -> pushes to "gsaltintas/flexitok-bpe_ita_Latn_32000"
    )
    hf_repo_suffix: Optional[str] = None  # e.g. "v2" -> "gsaltintas/flexitok-bpe_ita_Latn_32000-v2"
    hf: HFArgs = field(default_factory=HFArgs)
    hf_collection: Optional[str] = (
        None  # HF collection title to create/use, e.g. "Fineweb2 Tokenizers"
    )
    test_string: str = "Hello, world! 12345 This is a test. こんにちは"  # for quick smoke test of trained tokenizer


def get_chinese_chars():
    """Common-use Chinese characters (CJK Unified Ideographs block)."""
    # Level-1 and Level-2 GB2312 characters cover ~99% of modern Chinese text
    # Full CJK block is U+4E00–U+9FFF (20,902 chars) — too many for 32k vocab
    # Restrict to high-frequency subset
    chars = []
    for cp in range(0x4E00, 0x9FFF):
        chars.append(chr(cp))
    return chars
### for using max_lines as total
def balanced_text_iterator(data_dir, langs, max_lines_per_lang, shard_pattern, text_transform=None):
    """Yield text from each language up to max_lines_per_lang, interleaved."""
    import itertools
    iterators = []
    for lang in langs:
        lang_dir = Path(data_dir) / lang
        lang_files = sorted(glob.glob(str(lang_dir / shard_pattern)))
        it = jsonl_text_iterator(lang_files, max_lines=max_lines_per_lang, text_transform=text_transform)
        iterators.append(it)
    # interleave so BPE sees all languages throughout training
    # rather than all of lang1 then all of lang2
    return roundrobin(*iterators)

def roundrobin(*iterables):
    """Interleave iterables in round-robin order."""
    iterators = [iter(it) for it in iterables]
    while iterators:
        for it in list(iterators):
            try:
                yield next(it)
            except StopIteration:
                iterators.remove(it)

def jsonl_text_iterator(
    shard_files,
    max_lines: int = -1,
    text_transform: Optional[Callable[[str], str]] = None,
):
    """Yields text strings from JSONL shard files. Memory-efficient streaming."""
    count = 0
    for fpath in shard_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if text:
                    if text_transform is not None:
                        text = text_transform(text)
                    yield text
                    count += 1
                    if max_lines > 0 and count >= max_lines:
                        return


def build_normalizer(args: "TokenizerTrainingArgs"):
    """Build a normalizer from preprocessing args."""
    from tokenizers import Regex

    steps = []
    if args.strip_zero_width:
        steps.append(normalizers.Replace(Regex(ZERO_WIDTH_PATTERN), ""))
    norm_map = {
        "nfc": normalizers.NFC,
        "nfkc": normalizers.NFKC,
        "nfd": normalizers.NFD,
        "nfkd": normalizers.NFKD,
    }
    if args.unicode_normalization in norm_map:
        steps.append(norm_map[args.unicode_normalization]())
    if not steps:
        return None
    return normalizers.Sequence(steps) if len(steps) > 1 else steps[0]  # type: ignore[call-arg]


def build_pretokenizer(args: "TokenizerTrainingArgs"):
    """Build a pre-tokenizer from preprocessing args."""
    from tokenizers import Regex
    from tokenizers.pre_tokenizers import (
        BertPreTokenizer,
        ByteLevel,
        Metaspace,
        Sequence,
        Split,
        WhitespaceSplit,
    )

    fixed_pattern_pretokenizers = ("gpt2", "bloom", "gpt4")
    if args.pretokenizer in fixed_pattern_pretokenizers and (
        args.number_handling != "learned" or args.handle_contractions
    ):
        logger.warning(
            "number_handling and handle_contractions are ignored for pretokenizer=%s "
            "(number/contraction splitting is baked into its regex pattern).",
            args.pretokenizer,
        )
    cjk_splits = []
    if args.cjk_char_split:
        logger.info("CJK character split enabled: adding Split with Regex %s", CJK_PATTERN)
        cjk_splits.append(Split(Regex(CJK_PATTERN), behavior="isolated"))
    # if args.max_han_run > 0:
    #     logger.info("Max Han run length set to %d: adding Han chunking transform in pretokenizer", args.max_han_run)
    #     # This runs at inference too — it's in the tokenizer graph
    #     cjk_splits.insert(0, Split(
    #         Regex(rf"\p{{Han}}{{1,{args.max_han_run}}}"),
    #         behavior="isolated"
    #     ))

    if args.pretokenizer in ("gpt2", "bloom"):
        return Sequence(
            cjk_splits
            + [  # type: ignore[call-arg]
                Split(Regex(GPT2_PATTERN), behavior="isolated"),
                ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    if args.pretokenizer == "gpt4":
        return Sequence(
            cjk_splits
            + [  # type: ignore[call-arg]
                Split(Regex(GPT4_PATTERN), behavior="isolated"),
                ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    if args.pretokenizer == "whitespace":
        if cjk_splits:
            return Sequence(cjk_splits + [WhitespaceSplit()])  # type: ignore[call-arg]
        return WhitespaceSplit()
    if args.pretokenizer == "bert":
        if cjk_splits:
            return Sequence(cjk_splits + [BertPreTokenizer()])  # type: ignore[call-arg]
        return BertPreTokenizer()
    if args.pretokenizer == "script_aware":
        cjk_splits.append(
            Split(Regex(SCRIPT_AWARE_PATTERN), behavior=args.pretok_behavior)
        )
    if args.pretokenizer.startswith("custom:"):
        lang_code = args.pretokenizer[len("custom:"):]
        pattern_str = getattr(pretok_patterns, lang_code, None)
        if pattern_str is None:
            raise ValueError(
                f"No custom pattern for '{lang_code}' in pretokenization.py. "
                f"Available: {[k for k in dir(pretok_patterns) if not k.startswith('_')]}"
            )
        cjk_splits.append(Split(Regex(pattern_str), behavior=args.pretok_behavior))

    # Composable: byte_level | sentencepiece — prepend optional Split steps
    splits = cjk_splits[:]
    if args.handle_contractions:
        splits.append(Split(Regex(CONTRACTION_PATTERN), behavior="isolated"))
    if args.number_handling in NUMBER_REGEXES:
        splits.append(
            Split(Regex(NUMBER_REGEXES[args.number_handling]), behavior="isolated")
        )

    if args.pretokenizer == "sentencepiece":
        base = Metaspace(prepend_scheme="first")
    else:  # byte_level (default)
        base = ByteLevel(add_prefix_space=False, use_regex=args.use_byte_level_regex)

    return Sequence(splits + [base]) if splits else base  # type: ignore[call-arg]


def get_decoder(args: "TokenizerTrainingArgs"):
    """Return the decoder matching the chosen pretokenizer."""
    if args.pretokenizer in ("byte_level", "gpt2", "bloom", "gpt4", "script_aware") or args.pretokenizer.startswith("custom:"):
        return decoders.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
        # return decoders.Sequence([
        #     decoders.BPEDecoder(),
        #     decoders.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        # ])
        # return decoders.ByteLevel()
    if args.pretokenizer == "sentencepiece":
        return decoders.Metaspace(replacement="▁")
    if args.pretokenizer.startswith("custom:"):
        return decoders.WordPiece(prefix="##")  # or decoders.Sequence([])
    # whitespace / bert / script_aware / custom:*: Unicode tokens, concatenate directly
    return decoders.WordPiece(prefix="##")


def build_bpe_tokenizer(args: TokenizerTrainingArgs, initial_alphabet:Optional[list[str]]=None):
    """Configure a BPE tokenizer and trainer."""
    tokenizer = Tokenizer(models.BPE(byte_fallback=args.byte_fallback, unk_token=args.unk_token))
    tokenizer.normalizer = build_normalizer(args)  # type: ignore[assignment]
    tokenizer.pre_tokenizer = build_pretokenizer(args)  # type: ignore[assignment]
    tokenizer.decoder = get_decoder(args)  # type: ignore[assignment]
    if initial_alphabet is None:
        initial_alphabet = []

    initial_alphabet.extend(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    

    special_tokens = SPECIAL_TOKENS + list(args.extra_special_tokens)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=initial_alphabet,   # only single char
    )
    # tokenizer.add_tokens(get_digit_group_tokens(args.number_handling))  # ensure digit groups are single tokens for BPE
    return tokenizer, trainer


def build_unigram_tokenizer(args: TokenizerTrainingArgs):
    """Configure a UnigramLM tokenizer and trainer."""
    tokenizer = Tokenizer(models.Unigram([], None, False))  # type: ignore[call-arg]
    tokenizer.normalizer = build_normalizer(args)  # type: ignore[assignment]
    tokenizer.pre_tokenizer = build_pretokenizer(args)  # type: ignore[assignment]
    tokenizer.decoder = get_decoder(args)  # type: ignore[assignment]

    special_tokens = SPECIAL_TOKENS + list(args.extra_special_tokens)
    trainer = trainers.UnigramTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        unk_token=UNK_TOKEN,
        show_progress=True,
    )
    return tokenizer, trainer


def build_regularized_bpe_tokenizer(args: TokenizerTrainingArgs, **_):
    """
    Configure the shell tokenizer for regularized BPE.

    Returns (tokenizer, None) — vocab and merges are filled in after the custom
    training loop in train_langs; the HuggingFace trainer is not used here.
    """
    tokenizer = Tokenizer(models.BPE(byte_fallback=args.byte_fallback, unk_token=args.unk_token))
    tokenizer.normalizer = build_normalizer(args)  # type: ignore[assignment]
    tokenizer.pre_tokenizer = build_pretokenizer(args)  # type: ignore[assignment]
    tokenizer.decoder = get_decoder(args)  # type: ignore[assignment]
    return tokenizer, None


ALGORITHM_NAMES = {
    "bpe": "Byte-Level BPE",
    "unigram": "UnigramLM",
    "rbpe": "Regularized BPE",
}


def generate_model_card(
    algorithm: str,
    lang: List[str],
    vocab_size: int,
    final_vocab_size: int,
    num_shards: int,
    data_files: List[Path],
    pretokenizer: str = "byte_level",
    number_handling: str = "learned",
    handle_contractions: bool = False,
    unicode_normalization: Optional[str] = "nfc",
    strip_zero_width: bool = False,
    repo_id: Optional[str] = None,
    test_string: Optional[str] = None,
    sample_encoded: Optional[List[int]] = None,
    sample_tokens: Optional[List[str]] = None,
) -> str:
    algo_name = ALGORITHM_NAMES.get(algorithm, algorithm)
    normalizer_desc = str(unicode_normalization).upper()
    lang_str = ""
    # ISO 639-3 code or "und" if not identifiable
    if lang is None or lang == "numeric":
        lang_str = ["und"]  # undetermined language
    elif isinstance(lang, str):
        lang_str=[lang[:3]]
    else:
        lang_str = [l[:3] for l in lang]
    
    import yaml
    lang_str = yaml.dump({"language": lang_str})
    if strip_zero_width:
        normalizer_desc += " + zero-width strip"
    card = f"""---
license: mit
{lang_str}
tags:
- tokenizer
- {algorithm}
- flexitok
- fineweb2
---

# {algo_name} Tokenizer: {lang} ({vocab_size // 1000}K)

A **{algo_name}** tokenizer trained on **{lang}** data from Fineweb-2-HQ.

## Training Details

| Parameter | Value |
|-----------|-------|
| Algorithm | {algo_name} |
| Language | `{lang}` |
| Target Vocab Size | {vocab_size:,} |
| Final Vocab Size | {final_vocab_size:,} |
| Pre-tokenizer | {pretokenizer} |
| Number handling | {number_handling} |
| Contraction handling | {handle_contractions} |
| Normalizer | {normalizer_desc} |
| Special Tokens | `<s>`, `</s>`, `<pad>`, `<unk>` |
| Training Shards | {num_shards}, {[f.name for f in data_files]} |

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
tokens = tokenizer.encode("Hello, world!")
```

## Files

- `tokenizer.json` — Full HuggingFace tokenizer
- `vocab.json` — Vocabulary mapping
{"- `merges.txt` — BPE merge rules" if algorithm == "bpe" else "- `tokenizer.model` — SentencePiece protobuf (if available)"}
"""
    if test_string:
        card += f"""
## Sample Encoding
| Text | Tokens | Token IDs |
|------|--------|-----------|
| `{test_string}` | `{", ".join(sample_tokens) if sample_tokens else "N/A"}` | `{", ".join(map(str, sample_encoded)) if sample_encoded else "N/A"}` |

Command used to create this tokenizer:
```bash
{sys.argv}
"""
    return card


ALGORITHM_BUILDERS = {
    "bpe": build_bpe_tokenizer,
    "unigram": build_unigram_tokenizer,
    "rbpe": build_regularized_bpe_tokenizer,
}

import glob
import json
from collections import Counter

import regex


def get_cjk_char_frequencies(shard_files, max_lines=-1):
    counter = Counter()
    han = regex.compile(r"\p{Han}|\p{Hiragana}|\p{Katakana}")
    count = 0
    for fpath in shard_files:
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                counter.update(han.findall(text))
                count += 1
                if max_lines > 0 and count >= max_lines:
                    return counter
    return counter

def build_cjk_alphabet(shard_files, coverage=0.999, max_lines=500_000, max_alphabet_size=None):
    freq = get_cjk_char_frequencies(shard_files, max_lines=max_lines)
    total = sum(freq.values())
    cumsum = 0
    alphabet = []
    for char, count in freq.most_common():
        alphabet.append(char)
        cumsum += count
        if cumsum / total >= coverage:
            break
        if max_alphabet_size is not None and len(alphabet) >= max_alphabet_size:
            break
    logger.info(f"CJK alphabet: {len(alphabet)} chars cover {coverage*100}% of tokens")
    return alphabet
from tokenizers.pre_tokenizers import ByteLevel as BL


def cjk_byte_level_alphabet(chars):
    """Convert CJK chars to their ByteLevel-encoded forms for initial_alphabet."""
    bl = BL(add_prefix_space=False, use_regex=False)
    result = set()
    for char in chars:
        # pre_tokenize_str returns list of (token, offsets)
        encoded = bl.pre_tokenize_str(char)
        for token, _ in encoded:
            # each token is a single byte-mapped char like 'ĥ'
            result.update(token)  # add individual byte-mapped chars
    return list(result)

#todo
def cjk_pretokenized_tokens(chars):
    """
    Get the ByteLevel representation of each CJK char as a single
    pre-merged token string e.g. '在' -> 'åœ¨' (3 byte-mapped chars as one string).
    These get added via add_tokens() after training, but we want them
    as seeds BEFORE training so BPE treats each Han char as atomic.
    """
    bl = BL(add_prefix_space=False, use_regex=False)
    result = []
    for char in chars:
        encoded = bl.pre_tokenize_str(char)
        # encoded = [('åœ¨', (0,1))] — the whole char as one token string
        token_str = "".join(t for t, _ in encoded)
        result.append(token_str)
    return result


def get_digit_group_tokens(number_handling: str) -> list[str]:
    """Return all digit-group strings that should be guaranteed single tokens.

    For rtl_Ndigit: every integer 0 .. 10^N - 1, both in its natural string form
    ("0".."999" for N=3) and zero-padded to N digits ("000".."999"), so that
    trailing groups with leading zeros (e.g. "005" in "1 005") are also covered.
    """
    if number_handling == "individual":
        return [str(i) for i in range(10)]
    try:
        digits = int(
            number_handling.replace("ltr_", "").replace("rtl_", "").replace("digit", "")
        )
    except ValueError:
        logger.warning(f"Invalid number_handling value: {number_handling}")
        return []
    tokens: set[str] = set()
    # e.g. 1110 for ltr_3digit
    for i in range(10**digits):
        # tokens.append(str(i))
        # tokens.add(str(i))
        # adds all "1", "01", "001", "0001", etc. up to the specified digit length, to cover cases where the number is split into groups with leading zeros (e.g. "1 005" -> "1", "005")
        for j in range(1, digits+1):
            tokens.add(f"{i:0{j}d}")

    return list(tokens)
    return sorted(tokens, key=lambda s: (len(s), s))

def load_cjk_alphabet(lang_dir, coverage=0.999):
    """Load precomputed CJK alphabet, filtering by coverage threshold."""
    path = Path(lang_dir) / "cjk_alphabet.txt"
    if not path.exists():
        logger.warning(f"No cjk_alphabet.txt found in {lang_dir}, skipping CJK seeding")
        return []
    
    chars_with_freq = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            char, freq = line.split("\t")
            chars_with_freq.append((char, int(freq)))
    
    # apply coverage threshold at load time — flexible without recomputing
    total = sum(f for _, f in chars_with_freq)
    cumsum = 0
    chars = []
    for char, freq in chars_with_freq:  # already sorted by freq descending
        chars.append(char)
        cumsum += freq
        if cumsum / total >= coverage:
            break
    
    logger.info(f"Loaded {len(chars)} CJK chars ({coverage*100}% coverage) from {path}")
    return chars

def train_langs(
    args: TokenizerTrainingArgs, langs: List[str], collection_slugs: Optional[List[str]] = None
):
    """Train a tokenizer for a single language."""
    shard_files = []
    for lang in langs:
        lang_dir = Path(args.data_dir) / lang
        shard_files.extend(sorted(glob.glob(str(lang_dir / args.data_shard_pattern))))
    if not shard_files:
        raise FileNotFoundError(f"No .jsonl files found in {args.data_dir}")
    num_shards = len(shard_files)
    logger.info(f"Found {num_shards} shard files for langs={langs} in {args.data_dir}")
    ##TODO: here bypass for number tokenizers
    text_transform = None
    # text_transform = make_han_chunker(args.max_han_run) if args.max_han_run > 0 else None
    if text_transform:
        logger.info(f"Han run chunking enabled: max_han_run={args.max_han_run}")
    # iterator = jsonl_text_iterator(
    #     shard_files,
    #     max_lines=args.max_lines,
    #     text_transform=text_transform,
    # )
    iterator = balanced_text_iterator(
    args.data_dir, 
    langs, 
    max_lines_per_lang=args.max_lines,  # reuse existing arg — now means per-lang
    shard_pattern=args.data_shard_pattern,
    text_transform=text_transform,
)
    initial_alphabet=None
    cjk_seeds=None
    if args.add_cjk_chars:
        cjk_chars_cmn = []
        cjk_chars_jpn = []
        logger.info(f"Building CJK alphabet from training data with coverage={args.cjk_char_coverage} and max_lines={args.max_lines}")
        if "cmn_Hani" in langs:
            lang_dir = Path(args.data_dir) / "cmn_Hani"
            cjk_chars_cmn = load_cjk_alphabet(lang_dir, coverage=args.cjk_char_coverage)
            if len(cjk_chars_cmn) == 0:
                cjk_chars_cmn = build_cjk_alphabet(shard_files, coverage=args.cjk_char_coverage, max_lines=args.max_lines, max_alphabet_size=None)
        if "jpn_Japn" in langs:
            lang_dir = Path(args.data_dir) / "jpn_Japn"
            cjk_chars_jpn = load_cjk_alphabet(lang_dir, coverage=args.cjk_char_coverage)
            if len(cjk_chars_jpn) == 0:
                cjk_chars_jpn = build_cjk_alphabet(shard_files, coverage=args.cjk_char_coverage, max_lines=args.max_lines, max_alphabet_size=None)

        # Seed CJK chars as atomic tokens so BPE merges ON TOP of them
        n_cjk_chars_to_add = min(args.vocab_size-256, len(cjk_chars_cmn) + len(cjk_chars_jpn))  # ensure we don't exceed vocab size when adding byte-level chars, which are also in the initial alphabet
        logger.info(f"Adding {n_cjk_chars_to_add} CJK chars as atomic seed tokens")
        cjk_seeds_cmn = cjk_pretokenized_tokens(cjk_chars_cmn)[:n_cjk_chars_to_add//2]
        cjk_seeds_jpn = cjk_pretokenized_tokens(cjk_chars_jpn)[:n_cjk_chars_to_add - len(cjk_seeds_cmn)]
        cjk_seeds = cjk_seeds_cmn + cjk_seeds_jpn

    # Build tokenizer + trainer
    tokenizer, trainer = ALGORITHM_BUILDERS[args.algorithm](args, initial_alphabet=initial_alphabet)

    # Train
    logger.info(
        f"Training {args.algorithm} tokenizer: langs={langs}, vocab_size={args.vocab_size}"
    )

    if args.algorithm == "rbpe":
        from moto.tokenizer_training.regularized_bpe import (
            RegularizedBPETrainer,
            build_word_counts,
        )
        logger.info(
            f"Regularized BPE: λ_voc={args.lambda_voc}, λ_tok={args.lambda_tok}, λ_seg={args.lambda_seg}"
        )
        special_tokens = SPECIAL_TOKENS + list(args.extra_special_tokens)
        byte_alphabet = list(tokenizers.pre_tokenizers.ByteLevel.alphabet())
        rbpe_initial_alphabet = byte_alphabet + (initial_alphabet or [])

        word_counts = build_word_counts(iterator, tokenizer.pre_tokenizer, max_texts=args.max_lines)
        rbpe_trainer = RegularizedBPETrainer(
            vocab_size=args.vocab_size,
            lambda_voc=args.lambda_voc,
            lambda_tok=args.lambda_tok,
            lambda_seg=args.lambda_seg,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=rbpe_initial_alphabet,
            show_progress=True,
        )
        vocab_dict, merges = rbpe_trainer.train(word_counts)
        tokenizer.model = models.BPE(
            vocab=vocab_dict,
            merges=merges,
            unk_token=args.unk_token,
            byte_fallback=args.byte_fallback,
        )
    else:
        if cjk_seeds:
            tokenizer.add_tokens(cjk_seeds)
        tokenizer.train_from_iterator(iterator, trainer=trainer)

    if args.add_numbers:
        digits = get_digit_group_tokens(args.number_handling)
        tokenizer.add_tokens(digits)

    logger.info(f"Training complete. Vocab size: {tokenizer.get_vocab_size()}")

    # Save as HuggingFace PreTrainedTokenizerFast
    output_name = f"{args.algorithm}_{'_'.join(langs)}_{args.vocab_size}"
    repo_id = (
        args.hf.hf_repo_id
        if args.hf.hf_repo_id
        else f"{args.hf_repo_prefix}-{output_name}"
    )
    output_path = Path(args.output_dir) / repo_id.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        **(
            {"additional_special_tokens": list(args.extra_special_tokens)}
            if args.extra_special_tokens
            else {}
        ),
    )
    logger.info(f"Check hf_tokenizer.convert_ids_to_tokens(hf_tokenizer.encode('1234567890')): {hf_tokenizer.convert_ids_to_tokens(hf_tokenizer.encode('1234567890'))}")
    logger.info(f"Check hf_tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str('1234567890'): {hf_tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str('1234567890')}")

    hf_tokenizer.save_pretrained(str(output_path))
    logger.info(f"Tokenizer saved to {output_path}")

    tokenizer_config_path = output_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["number_handling"] = args.number_handling
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    # Save additional artifact files
    tokenizer_data = json.loads(tokenizer.to_str())
    vocab = tokenizer.get_vocab()
    with open(output_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    # merges.txt for BPE
    if args.algorithm == "bpe":
        merges = tokenizer_data.get("model", {}).get("merges", [])
        with open(output_path / "merges.txt", "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for merge in merges:
                f.write(f"{merge}\n")
        logger.info(f"Saved merges.txt ({len(merges)} merges) and vocab.json")
    # .model (SentencePiece protobuf) for Unigram
    if args.algorithm == "unigram":
        try:
            import sentencepiece.sentencepiece_model_pb2 as sp_pb2

            sp_model = sp_pb2.ModelProto()
            sp_model.trainer_spec.model_type = 1  # UNIGRAM
            sp_model.normalizer_spec.name = "nfc"
            pieces = tokenizer_data.get("model", {}).get("vocab", [])
            for token, score in pieces:
                piece = sp_pb2.ModelProto.SentencePiece()
                piece.piece = token
                piece.score = score
                sp_model.pieces.append(piece)
            with open(output_path / "tokenizer.model", "wb") as f:
                f.write(sp_model.SerializeToString())
            logger.info(f"Saved tokenizer.model ({len(pieces)} pieces)")
        except ImportError:
            logger.warning(
                "sentencepiece_model_pb2 not available, skipping .model file export. "
                "Install sentencepiece to enable this."
            )

    # Quick sanity check
    encoded = hf_tokenizer.encode(args.test_string)
    logger.info(f"Sample encoding: '{args.test_string}' -> {len(encoded)} tokens")
    logger.info(f"Sample tokens: '{hf_tokenizer.convert_ids_to_tokens(encoded[:20])}'")
    logger.info(f"Token IDs: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")

    # Write model card
    model_card = generate_model_card(
        algorithm=args.algorithm,
        lang=langs,
        vocab_size=args.vocab_size,
        final_vocab_size=tokenizer.get_vocab_size(),
        num_shards=num_shards,
        data_files=[Path(s) for s in shard_files],
        pretokenizer=args.pretokenizer,
        number_handling=args.number_handling,
        handle_contractions=args.handle_contractions,
        unicode_normalization=args.unicode_normalization,
        strip_zero_width=args.strip_zero_width,
        repo_id=repo_id if args.hf.publish_to_hf and args.hf_repo_prefix else None,
        test_string=args.test_string,
        sample_encoded=encoded[:20],
        sample_tokens=hf_tokenizer.convert_ids_to_tokens(encoded[:20])
        if args.test_string
        else None,
    )
    with open(output_path / "README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    # Publish to HuggingFace Hub
    if args.hf.publish_to_hf:
        repo_id = repo_id.replace("/-", "/")
        logger.info(f"Pushing tokenizer to HuggingFace Hub: https://huggingface.co/{repo_id}")
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info(f"Published to https://huggingface.co/{repo_id}")
        if collection_slugs:
            for collection_slug in collection_slugs:
                api.add_collection_item(
                    collection_slug=collection_slug,
                    item_id=repo_id,
                    item_type="model",
                    exists_ok=True,
                )
                logger.info(f"Added {repo_id} to collection {collection_slug}")


def main(args: TokenizerTrainingArgs):
    assert args.algorithm in ALGORITHM_BUILDERS, (
        f"Unknown algorithm '{args.algorithm}'. Must be one of: {list(ALGORITHM_BUILDERS.keys())}"
    )
    assert args.data_dir, "data_dir must be specified"
    assert args.langs, "langs must be specified (e.g. langs='[ita_Latn,tur_Latn]')"

    # Create HF collection once if requested
    collection_slugs = None
    if args.hf.publish_to_hf and args.hf.collections:
        try:
            collection_slugs = create_or_get_collection_slugs(args.hf.collections)
        except Exception as e:
            logger.warning(f"Failed to create/get HF collection slugs: {e}")
            collection_slugs = args.hf.collections

    logger.info(f"=== Starting training with languages: {args.langs} ===")
    train_langs(args, args.langs, collection_slugs=collection_slugs)


if __name__ == "__main__":
    args = load_args(TokenizerTrainingArgs)
    main(args)
