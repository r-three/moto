"""
Generate a modular arithmetic dataset and upload to HuggingFace.

Hypothesis: the "suitable tokenizer" for `a mod b` is determined by the
digit-grouping that exposes the relevant structure in `a`:

  - For b = 2^k × 5^j (no other prime factors), only the rightmost max(k, j)
    digits of a determine a mod b, because 10^max(k,j) ≡ 0 (mod b).
    → suitable_tokenizer: "digit_{max(k,j)}_rtl"  (right-to-left groups)

  - For all other moduli (3, 7, 11, …), the structure is more complex.
    → suitable_tokenizer: "complex"

Dataset columns:
  a                  : dividend (int)
  b                  : modulus (int)
  answer             : a mod b (int)
  question           : "{a} mod {b} " (str)
  text               : "{a} mod {b} = {answer}" (str)
  problem_type       : fine-grained classification of b (str)
  suitable_tokenizer : hypothesised best tokenizer type (str)

Usage:
  python -m flexitok.simplified.create_numeric_synthetic_data \
      hf.hf_repo_id=flexitok/mod-arithmetic  hf.publish_to_hf=true \
        a_max=99999 seed=42 \
        moduli=[2,3,4,5,7,8,10,16,25,32,100,125,1000] \
        num_shards=1 \
        split_by_problem_type=true


    Optionally pass `split_by_problem_type=true` to pre-split the dataset by problem type (one subfolder per type, with individual HF configs pointing to each, and a default config globbing across all).
"""

import math
import os
import tempfile
from dataclasses import dataclass, field
from itertools import product
from typing import List

from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard, DatasetCardData, HfApi

from flexitok.utils.args import Args, HFArgs, load_args

# ---------------------------------------------------------------------------
# Argument schema
# ---------------------------------------------------------------------------

@dataclass
class DatasetArgs(Args):
    hf: HFArgs = field(default_factory=HFArgs)
    a_min: int = 0
    a_max: int = 9999
    moduli: List[int] = field(default_factory=lambda: [2, 4, 5, 8, 10, 16, 25, 32, 100, 125, 1000])
    train_ratio: float = 0.9
    seed: int = 42
    num_shards: int = 1  # number of chunk files per split (e.g. train.chunk.0.jsonl … train.chunk.N-1.jsonl)
    split_by_problem_type: bool = False  # upload one subfolder per problem_type with individual HF configs


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _count_factor(n: int, p: int) -> tuple[int, int]:
    """Return (k, n // p^k) where k is the largest power of p dividing n."""
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k, n


def get_problem_type(b: int) -> str:
    """Classify the modulus b into a human-readable problem type."""
    return f"mod_{b}"  # simple type for now; could be extended to "mod_power_of_2", "mod_power_of_5", "mod_prime", etc. based on factorization
    k2, rem = _count_factor(b, 2)
    k5, rem = _count_factor(rem, 5)

    if rem == 1:
        if k5 == 0:
            return "mod_power_of_2"
        if k2 == 0:
            return "mod_power_of_5"
        if k2 == k5:
            return "mod_power_of_10"
        return "mod_mixed_2_5"

    is_prime = rem == b and all(b % i != 0 for i in range(2, math.isqrt(b) + 1))
    if is_prime:
        return "mod_prime"
    if k2 == 0 and k5 == 0:
        return "mod_composite"
    return "mod_composite_with_2_5_factors"


def get_suitable_tokenizer(b: int) -> str:
    """
    Hypothesise the suitable tokenizer for `a mod b`.

    For b = 2^k * 5^j, only the rightmost max(k, j) digits of `a` determine
    `a mod b`.  A tokenizer that groups digits right-to-left in chunks of
    that size exposes the relevant information as a single token.

    For all other b, the digit structure is more complex ("complex").
    """
    k2, rem = _count_factor(b, 2)
    k5, rem = _count_factor(rem, 5)

    if rem == 1:
        n_digits = max(k2, k5)
        if n_digits == 1:
            return "individual"
        return f"rtl_{n_digits}digit"

    return "complex"


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_records(moduli: List[int], a_min: int, a_max: int) -> list[dict]:
    records = []
    for b, a in product(moduli, range(a_min, a_max + 1)):
        records.append(
            {
                "a": a,
                "b": b,
                "answer": a % b,
                "text": f"{a} mod {b} = {a % b}",
                "question": f"{a} mod {b} = ",
                "problem_type": get_problem_type(b),
                "suitable_tokenizer": get_suitable_tokenizer(b),
            }
        )
    return records


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_shards(ds: Dataset, out_dir: str, split_name: str, num_shards: int) -> None:
    """Write `ds` as `{split_name}.chunk.{i}.jsonl` files under `out_dir`."""
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        shard.to_json(os.path.join(out_dir, f"{split_name}.chunk.{i:2d}.jsonl"))


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def build_card(
    args: DatasetArgs,
    n_train: int,
    n_test: int,
    problem_types: list[str],
    pt_counts: dict[str, dict[str, int]],
) -> DatasetCard:
    tokenizer_table_rows = "\n".join(
        f"| {b} | `{get_suitable_tokenizer(b)}` | {get_problem_type(b)} |"
        for b in sorted(args.moduli)
    )
    if args.split_by_problem_type:
        # Default config globs all per-problem-type chunk files; individual configs point to one folder each.
        configs = [
            {
                "config_name": "default",
                "data_files": [
                    {"split": "train", "path": "data/*/train.chunk.*.jsonl"},
                    {"split": "test",  "path": "data/*/test.chunk.*.jsonl"},
                ],
                "default": True,
            }
        ] + [
            {
                "config_name": pt,
                "data_files": [
                    {"split": "train", "path": f"data/{pt}/train.chunk.*.jsonl"},
                    {"split": "test",  "path": f"data/{pt}/test.chunk.*.jsonl"},
                ],
            }
            for pt in sorted(problem_types)
        ]
    else:
        configs = [
            {
                "config_name": "default",
                "data_files": [
                    {"split": "train", "path": "data/train.chunk.*.jsonl"},
                    {"split": "test",  "path": "data/test.chunk.*.jsonl"},
                ],
                "default": True,
            }
        ]
    content = f"""\
---
{DatasetCardData(
    language="en",
    task_categories=["question-answering"],
    tags=["arithmetic", "modular-arithmetic", "tokenization", "synthetic"],
    seed=args.seed,
    configs=configs,
).to_yaml()}
---

# Modular Arithmetic Dataset

Synthetic dataset of modular-arithmetic problems of the form `a mod b`,
paired with the result and a hypothesis about the most suitable tokenizer.

## Tokenizer hypothesis

For `a mod b` where `b = 2^k × 5^j` (no other prime factors), only the
rightmost `max(k, j)` digits of `a` determine the answer, because
`10^max(k,j) ≡ 0 (mod b)`.  A tokenizer that groups digits **right-to-left**
in chunks of that size exposes the relevant information as a single token.

For all other moduli (e.g. 3, 7, 11, …) the digit structure is more complex
and is labelled `complex`.

| modulus (b) | suitable_tokenizer | problem_type |
|---|---|---|
{tokenizer_table_rows}

## Dataset fields

| field | type | description |
|---|---|---|
| `a` | int | dividend |
| `b` | int | modulus |
| `answer` | int | `a mod b` |
| `text` | str | `"{{a}} mod {{b}} = {{answer}}"` |
| `question` | str | `"{{a}} mod {{b}} = "` |
| `problem_type` | str | classification of `b` |
| `suitable_tokenizer` | str | hypothesised best tokenizer type |

## Splits

| split | examples |
|---|---|
| train | {n_train:,} |
| test | {n_test:,} |

{"### Examples per subset" + chr(10) + chr(10) + "| problem_type | train_frac | train | test |" + chr(10) + "|---|---| ---| ---|" + chr(10) + chr(10).join(f"| {pt} | {pt_counts[pt]['train'] / n_train:.4f} | {pt_counts[pt]['train']:,} | {pt_counts[pt]['test']:,} |" for pt in sorted(pt_counts))}

## Generation

```
python -m flexitok.simplified.create_numeric_synthetic_data \\
    hf.hf_repo_id={args.hf.hf_repo_id} hf.publish_to_hf=true \\
    a_min={args.a_min} a_max={args.a_max} \\
    moduli="[{', '.join(str(m) for m in args.moduli)}]" \\
    seed={args.seed} train_ratio={args.train_ratio}
```
"""
    return DatasetCard(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args: DatasetArgs = load_args(DatasetArgs())  # type: ignore[assignment]

    print(f"Moduli : {args.moduli}")
    print(f"a range: [{args.a_min}, {args.a_max}]")

    records = generate_records(args.moduli, args.a_min, args.a_max)
    print(f"Generated {len(records):,} examples")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=1.0 - args.train_ratio, seed=args.seed)
    dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})

    n_train, n_test = len(dataset_dict["train"]), len(dataset_dict["test"])
    print(f"  train : {n_train:,}")
    print(f"  test  : {n_test:,}")

    # Collect unique problem types (preserving insertion order via dict)
    problem_types = list({r["problem_type"] for r in records})

    # Pre-filter each split by problem type (only when needed).
    pt_splits: dict[str, dict[str, Dataset]] = {}
    pt_counts: dict[str, dict[str, int]] = {}
    if args.split_by_problem_type:
        for pt in problem_types:
            pt_splits[pt] = {
                split_name: ds.filter(lambda ex, p=pt: ex["problem_type"] == p)
                for split_name, ds in dataset_dict.items()
            }
        pt_counts = {
            pt: {split_name: ds.num_rows for split_name, ds in splits.items()}
            for pt, splits in pt_splits.items()
        }

    if args.hf.publish_to_hf and args.hf.hf_repo_id:
        print(f"Pushing to hub: {args.hf.hf_repo_id} …")
        api = HfApi()
        api.create_repo(repo_id=args.hf.hf_repo_id, repo_type="dataset", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = os.path.join(tmp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            if args.split_by_problem_type:
                # One subfolder per problem type; default config globs across all.
                for pt, splits in pt_splits.items():
                    pt_dir = os.path.join(data_dir, pt)
                    os.makedirs(pt_dir, exist_ok=True)
                    for split_name, ds in splits.items():
                        write_shards(ds, pt_dir, split_name, args.num_shards)
            else:
                for split_name, ds in dataset_dict.items():
                    write_shards(ds, data_dir, split_name, args.num_shards)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=args.hf.hf_repo_id,
                repo_type="dataset",
            )
        card = build_card(args, n_train, n_test, problem_types, pt_counts)
        card.push_to_hub(args.hf.hf_repo_id, repo_type="dataset")
        print("Done.")
    else:
        print("Skipping upload (set publish_to_hf=true hf_repo_id=<repo> to upload).")


if __name__ == "__main__":
    main()
