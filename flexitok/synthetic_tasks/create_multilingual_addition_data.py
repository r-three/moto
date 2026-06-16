"""
Written w. Claude

Generate a multilingual addition dataset and upload to HuggingFace.

Each sample is of the form `a + b =`, where a and b are written-form
numbers from 0 to 9999 in one of 21 languages, plus a 22nd "digit" split
where a and b are their numeric digit strings.

Dataset columns (per sample):
  a_str        : written-form (or digit string) of a
  a_digit      : integer value of a
  b_str        : written-form (or digit string) of b
  b_digit      : integer value of b
  answer       : written-form (or digit string) of a + b
  answer_digit : integer value of a + b
  text         : full completion string  "{a_str}+{b_str}={answer}"
  question     : prompt string           "{a_str}+{b_str}="
  lang         : language tag (e.g. "eng_Latn") or "digit" for the numeric split

Splits: one HuggingFace config per language/digit split, each with
  train.jsonl and val.jsonl shard files.

Usage:
  python flexitok/simplified/create_multilingual_addition_data.py \
    output_dir=/scratch/gsa/data/multilingual-addition \
      hf.hf_repo_id=flexitok/multilingual-addition \
      hf.publish_to_hf=true \
      a_max=999 \
      seed=42 \
      train_ratio=0.9 \
      num_workers=12
      
      max_pairs_per_lang=10

  # dry run (no upload):
  python flexitok/simplified/create_multilingual_addition_data.py. a_max=99

Requires [num2words](https://pypi.org/project/num2words/) library
"""

import argparse
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from num2words import num2words

from flexitok.utils.args import HFArgs, load_args

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

LANGUAGES: List[str] = [
    "eng_Latn", "dan_Latn", "swe_Latn", "vie_Latn", "hun_Latn",
    "fas_Arab", "tur_Latn", "ces_Latn", "arb_Arab", "ell_Grek",
    "ind_Latn", "nld_Latn", "pol_Latn", "por_Latn", "ita_Latn",
    "jpn_Jpan", "fra_Latn", "spa_Latn", "deu_Latn", "cmn_Hani",
    "rus_Cyrl", 
    # New transfer languages
    "nob_Latn", "fin_Latn", "ben_Beng", "kor_Hang",
]

# Map from FLORES-200 language tag → num2words language code
_NUM2WORDS_CODE: Dict[str, str] = {
    "eng_Latn": "en",
    "dan_Latn": "da",
    "swe_Latn": "sv",
    "vie_Latn": "vi",
    "hun_Latn": "hu",
    "fas_Arab": "fa",
    "tur_Latn": "tr",
    "ces_Latn": "cs",
    "arb_Arab": "ar",
    "ind_Latn": "id",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "por_Latn": "pt_BR",
    "ita_Latn": "it",
    "jpn_Jpan": "ja",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "deu_Latn": "de",
    "rus_Cyrl": "ru",
    # ell_Grek (Greek) and cmn_Hani (Mandarin Chinese) use custom implementations below
    # New transfer languages — all natively supported by num2words
    "nob_Latn": "no",
    "fin_Latn": "fi",
    "ben_Beng": "bn",
    "kor_Hang": "ko",
}

DIGIT_LANG = "digit"  # tag for the numeric-digit split


# ---------------------------------------------------------------------------
# Argument schema
# ---------------------------------------------------------------------------

@dataclass
class DatasetArgs:
    hf: HFArgs = field(default_factory=HFArgs)
    publish_to_hf: bool = False
    output_dir: Optional[str] = None  # where to write JSONL files; defaults to ./dataset_output
    a_min: int = 0
    a_max: int = 9999
    train_ratio: float = 0.9
    seed: int = 42
    num_shards: int = 1          # shards per split per language config
    max_pairs_per_lang: int = -1 # -1 = exhaustive (a_max-a_min+1)^2 pairs; >0 = random sample
    num_workers: int = -1        # -1 = os.cpu_count(); 1 = sequential
    languages: List[str] = field(default_factory=lambda: LANGUAGES + [DIGIT_LANG])  # languages to include; defaults to all in LANGUAGES list
# ---------------------------------------------------------------------------
# Custom number-word implementations for Greek and Mandarin Chinese
# ---------------------------------------------------------------------------

def _greek_num(n: int) -> str:
    """Convert integer 0–9999 to Modern Greek words."""
    if n == 0:
        return "μηδέν"
    ones   = ["", "ένα", "δύο", "τρία", "τέσσερα", "πέντε",
              "έξι", "επτά", "οκτώ", "εννέα"]
    teens  = ["δέκα", "έντεκα", "δώδεκα", "δεκατρία", "δεκατέσσερα",
              "δεκαπέντε", "δεκαέξι", "δεκαεπτά", "δεκαοκτώ", "δεκαεννέα"]
    tens_w = ["", "δέκα", "είκοσι", "τριάντα", "σαράντα", "πενήντα",
              "εξήντα", "εβδομήντα", "ογδόντα", "ενενήντα"]
    hunds  = ["", "εκατό", "διακόσια", "τριακόσια", "τετρακόσια",
              "πεντακόσια", "εξακόσια", "επτακόσια", "οκτακόσια", "εννιακόσια"]

    parts: List[str] = []
    if n >= 1000:
        thou = n // 1000
        parts.append("χίλια" if thou == 1 else _greek_num(thou) + " χιλιάδες")
        n %= 1000
    if n >= 100:
        h = n // 100
        # εκατόν before non-zero remainder, εκατό alone
        parts.append("εκατόν" if (h == 1 and n % 100 != 0) else hunds[h])
        n %= 100
    if n >= 20:
        t, o = divmod(n, 10)
        parts.append(tens_w[t] + (" " + ones[o] if o else ""))
    elif n >= 10:
        parts.append(teens[n - 10])
    elif n > 0:
        parts.append(ones[n])
    return " ".join(parts)


def _chinese_num(n: int) -> str:
    """Convert integer 0–9999 to Mandarin Chinese numerals (traditional counting)."""
    if n == 0:
        return "零"
    digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

    thou = n // 1000
    rem  = n % 1000
    hun  = rem // 100
    rem2 = rem % 100
    tens = rem2 // 10
    ones = rem2 % 10

    result = ""
    if thou:
        result += digits[thou] + "千"
    if hun:
        result += digits[hun] + "百"
    elif thou and (tens or ones):
        result += "零"           # single zero gap after thousands
    if tens:
        result += digits[tens] + "十"
        if ones:
            result += digits[ones]
    elif ones:
        if hun:
            result += "零"      # zero gap between hundreds and units
        result += digits[ones]
    return result

# ---------------------------------------------------------------------------
# Turkish: num2words produces unspaced compounds; split with maximal matching
# on the finite set of Turkish number words.
# ---------------------------------------------------------------------------
_TR_WORDS = [
    'milyon', 'bin', 'yüz',
    'doksan', 'seksen', 'yetmiş', 'altmış', 'elli', 'kırk', 'otuz', 'yirmi', 'on',
    'dokuz', 'sekiz', 'yedi', 'altı', 'beş', 'dört', 'üç', 'iki', 'bir',
    'sıfır',
]
_TR_SPLIT = re.compile('(' + '|'.join(re.escape(w) for w in _TR_WORDS) + ')')
 
def _turkish_num(n: int) -> str:
    from num2words import num2words as _n2w
    return ' '.join(_TR_SPLIT.findall(_n2w(n, lang='tr')))
 
# ---------------------------------------------------------------------------
# Unified number-to-word converter
# ---------------------------------------------------------------------------

def _make_converter(lang: str) -> Callable[[int], str]:
    """Return a function int → word-string for the given language tag."""
    if lang == DIGIT_LANG:
        return str
    if lang == "ell_Grek":
        return _greek_num
    if lang == "cmn_Hani":
        return _chinese_num
    if lang == "tur_Latn":
        return _turkish_num
    code = _NUM2WORDS_CODE[lang]
    return lambda n: num2words(n, lang=code)


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def _process_lang(
    lang: str,
    a_min: int,
    a_max: int,
    max_pairs: int,
    seed: int,
    train_ratio: float,
) -> Tuple[str, "DatasetDict", Dict[str, int]]:
    """Top-level worker (picklable) for ProcessPoolExecutor."""
    converter = _make_converter(lang)
    records = generate_records_for_lang(lang, converter, a_min, a_max, max_pairs, seed)
    dd = split_dataset(records, train_ratio, seed)
    stats = {"train": len(dd["train"]), "val": len(dd["val"])}
    return lang, dd, stats


def generate_records_for_lang(
    lang: str,
    converter: Callable[[int], str],
    a_min: int,
    a_max: int,
    max_pairs: int = -1,
    seed: int = 42,
) -> List[dict]:
    """
    Generate (a, b) pairs for a given language split.
    All a, b in [a_min, a_max].

    If max_pairs > 0 and the exhaustive count exceeds it, a random sample of
    max_pairs distinct (a, b) pairs is drawn (without replacement).
    """
    n = a_max - a_min + 1
    exhaustive_count = n * n

    if max_pairs > 0 and exhaustive_count > max_pairs:
        # Random sample without replacement using flat index arithmetic
        # so that the numbers are parallel across splits
        rng = random.Random(seed)
        indices = rng.sample(range(exhaustive_count), max_pairs)
        pairs: List[Tuple[int, int]] = [
            (a_min + idx // n, a_min + idx % n) for idx in indices
        ]
    else:
        pairs = [(a, b) for a in range(a_min, a_max + 1) for b in range(a_min, a_max + 1)]

    records = []
    for a, b in pairs:
        a_str      = converter(a)
        b_str      = converter(b)
        answer_int = a + b
        ans_str    = converter(answer_int)
        records.append({
            "a_str":        a_str,
            "a_digit":      a,
            "b_str":        b_str,
            "b_digit":      b,
            "answer":       ans_str,
            "answer_digit": answer_int,
            "text":         f"{a_str}+{b_str}={ans_str}",
            "question":     f"{a_str}+{b_str}=",
            "lang":         lang,
        })
    return records


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_shards(
    ds: Dataset,
    out_dir: str,
    split_name: str,
    num_shards: int,
) -> None:
    """Write dataset as `{split_name}.chunk.{i:02d}.jsonl` files under out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        path  = os.path.join(out_dir, f"{split_name}.chunk.{i:02d}.jsonl")
        shard.to_json(path, orient="records", lines=True)
        print(f"    Wrote {path}  ({len(shard):,} rows)")


def split_dataset(
    records: List[dict],
    train_ratio: float,
    seed: int,
) -> DatasetDict:
    """Shuffle and split records into train / val."""
    rng = random.Random(seed)
    rng.shuffle(records)
    n_train = int(len(records) * train_ratio)
    train_records = records[:n_train]
    val_records   = records[n_train:]
    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "val":   Dataset.from_list(val_records),
    })


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def build_card(args: DatasetArgs, lang_stats: Dict[str, Dict[str, int]]) -> DatasetCard:
    all_langs = LANGUAGES + [DIGIT_LANG]

    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": "data/*/train.chunk.*.jsonl"},
                {"split": "val",   "path": "data/*/val.chunk.*.jsonl"},
            ],
            "default": True,
        }
    ] + [
        {
            "config_name": lang,
            "data_files": [
                {"split": "train", "path": f"data/{lang}/train.chunk.*.jsonl"},
                {"split": "val",   "path": f"data/{lang}/val.chunk.*.jsonl"},
            ],
        }
        for lang in all_langs
    ]
    lang_rows = "\n".join(
        f"| `{lang}` | {lang_stats[lang]['train']:,} | {lang_stats[lang]['val']:,} |"
        for lang in all_langs
    )
 
    # pt_BR is not a valid ISO 639 code; split into ISO language list + BCP-47 list
    _iso_remap = {"pt_BR": "pt", "cmn_Hani": "zh", "ell_Grek": "el", "jpn_Jpan": "ja"}
    iso_langs = list(dict.fromkeys(  # deduplicate, preserve order
        _iso_remap.get(_NUM2WORDS_CODE.get(l, l), _NUM2WORDS_CODE.get(l, l))
        for l in LANGUAGES
    ))

    content = f"""\
---
{DatasetCardData(
    language=iso_langs,
    language_bcp47=["pt-BR"],
    task_categories=["question-answering"],
    tags=["arithmetic", "addition", "multilingual", "synthetic", "tokenization"],
    configs=configs,
).to_yaml()}
---

# Multilingual Addition Dataset

Synthetic dataset of addition problems of the form `a+b=answer`, where `a`
and `b` are written-form representations of integers in 21 languages, plus
a 22nd split using raw digit strings.

## Task format

Each sample contains:

| field | type | description |
|---|---|---|
| `a_str` | str | written-form (or digit) representation of `a` |
| `a_digit` | int | integer value of `a` |
| `b_str` | str | written-form (or digit) representation of `b` |
| `b_digit` | int | integer value of `b` |
| `answer` | str | written-form (or digit) of `a + b` |
| `answer_digit` | int | integer value of `a + b` |
| `text` | str | `"{{a_str}}+{{b_str}}={{answer}}"` (completion target) |
| `question` | str | `"{{a_str}}+{{b_str}}="` (prompt) |
| `lang` | str | language tag, e.g. `eng_Latn`, or `digit` |

Numbers range from `{args.a_min}` to `{args.a_max}` for both `a` and `b`
(answers up to `{args.a_max * 2}`).

## Languages

| lang | train | val |
|---|---|---|
{lang_rows}

## Generation

```bash
python create_multilingual_addition_data.py \\
    --hf_repo_id {args.hf.hf_repo_id or "YOUR_HF_USERNAME/multilingual-addition"} \\
    --publish_to_hf \\
    --a_min {args.a_min} --a_max {args.a_max} \\
    --seed {args.seed} --train_ratio {args.train_ratio}
```
"""
    return DatasetCard(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: DatasetArgs) -> None:
    print(f"a range   : [{args.a_min}, {args.a_max}]")
    print(f"Languages : {LANGUAGES + [DIGIT_LANG]}")
    print(f"seed      : {args.seed}")
    print(f"train_ratio: {args.train_ratio}")
    print()

    all_langs = LANGUAGES + [DIGIT_LANG]
    lang_stats: Dict[str, Dict[str, int]] = {}

    # We'll collect per-language DatasetDicts to write/upload
    lang_datasets: Dict[str, DatasetDict] = {}

    num_workers = args.num_workers if args.num_workers > 0 else (os.cpu_count() or 1)
    num_workers = min(num_workers, len(all_langs))
    print(f"num_workers: {num_workers}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_lang,
                lang, args.a_min, args.a_max,
                args.max_pairs_per_lang, args.seed, args.train_ratio,
            ): lang
            for lang in all_langs
        }
        for future in as_completed(futures):
            lang, dd, stats = future.result()
            lang_datasets[lang] = dd
            lang_stats[lang] = stats
            print(f"[{lang}] train={stats['train']:,}  val={stats['val']:,}")

    local_out = args.output_dir or os.path.join(os.getcwd(), "dataset_output")

    def _write_all(root: str) -> None:
        """Write all language splits to root/<lang>/..."""
        for lang, dd in lang_datasets.items():
            lang_dir = os.path.join(root, lang)
            for split_name, ds in dd.items():
                write_shards(ds, lang_dir, split_name, args.num_shards)

    print(f"\nWriting locally to: {local_out}")
    _write_all(local_out)
    print("Local write complete.")

    if args.hf.publish_to_hf and args.hf.hf_repo_id:
        print(f"\nPushing to hub: {args.hf.hf_repo_id} …")
        api = HfApi()
        api.create_repo(repo_id=args.hf.hf_repo_id, repo_type="dataset", exist_ok=True)

        # Upload local_out as "data/" in the repo so card paths (data/<lang>/...) resolve correctly.
        print("Uploading folder to HuggingFace …")
        api.upload_folder(
            folder_path=local_out,
            path_in_repo="data",
            repo_id=args.hf.hf_repo_id,
            repo_type="dataset",
        )

        card = build_card(args, lang_stats)
        card.push_to_hub(args.hf.hf_repo_id, repo_type="dataset")
        print("Done — dataset card pushed.")
    else:
        print("Skipping upload (pass hf.publish_to_hf=true hf.hf_repo_id=<repo> to upload).")


if __name__ == "__main__":
    args = load_args(DatasetArgs)
    main(args)