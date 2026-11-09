"""
Inspect pretokenizer configs across all constituent tokenizers.

Usage:
    python inspect_pretokenizers.py --tokenizers tok1.json tok2.json ...
    python inspect_pretokenizers.py --from_hub --tokenizers flexitok/bpe_ltr_fra_Latn_8000_v3 ...
"""

# python -m moto.moto_to_monolithic.inspect_pretokenizers \
#   --from_hub \
#   --tokenizers flexitok/bpe_ltr_fw_edu_8000_v3 flexitok/bpe_ltr_fra_Latn_8000_v3 flexitok/bpe_ltr_tur_Latn_8000_v3 flexitok/bpe_ltr_cmn_Hani_8000_v3 \
#   --names eng fra tur zho
import argparse
import json
from collections import Counter


def load_tokenizer_json(path):
    with open(path) as f:
        return json.load(f)


def load_tokenizer_json_from_hub(repo_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo_id)
    return json.loads(tok.backend_tokenizer.to_str())


def describe_pretokenizer(pretok, indent=0):
    """Recursively describe a pretokenizer config."""
    pad = "  " * indent
    if pretok is None:
        return [f"{pad}(none)"]

    t = pretok.get("type", "unknown")
    lines = []

    if t == "Sequence":
        lines.append(f"{pad}Sequence:")
        for sub in pretok.get("pretokenizers", []):
            lines.extend(describe_pretokenizer(sub, indent + 1))

    elif t == "Split":
        pattern = pretok.get("pattern", {})
        pat_str = pattern.get("Regex") or pattern.get("String") or str(pattern)
        behavior = pretok.get("behavior", "")
        invert = pretok.get("invert", False)
        lines.append(f"{pad}Split  pattern={pat_str!r}  behavior={behavior}  invert={invert}")

    elif t == "ByteLevel":
        add_prefix = pretok.get("add_prefix_space", False)
        use_regex = pretok.get("use_regex", True)
        lines.append(f"{pad}ByteLevel  add_prefix_space={add_prefix}  use_regex={use_regex}")

    elif t == "Metaspace":
        replacement = pretok.get("replacement", "▁")
        add_prefix = pretok.get("add_prefix_space", True)
        lines.append(f"{pad}Metaspace  replacement={replacement!r}  add_prefix_space={add_prefix}")

    elif t == "Whitespace":
        lines.append(f"{pad}Whitespace  (split on whitespace)")

    elif t == "WhitespaceSplit":
        lines.append(f"{pad}WhitespaceSplit  (split on whitespace only)")

    elif t == "Punctuation":
        behavior = pretok.get("behavior", "Isolated")
        lines.append(f"{pad}Punctuation  behavior={behavior}")

    elif t == "Digits":
        individual = pretok.get("individual_digits", False)
        lines.append(f"{pad}Digits  individual_digits={individual}")

    elif t == "UnicodeScripts":
        lines.append(f"{pad}UnicodeScripts  (split on script boundaries)")

    else:
        lines.append(f"{pad}{t}  {json.dumps({k:v for k,v in pretok.items() if k != 'type'})}")

    return lines


def inspect_tokenizer(data, name):
    pretok = data.get("pre_tokenizer")
    normalizer = data.get("normalizer")
    model_type = data.get("model", {}).get("type", "unknown")
    n_merges = len(data.get("model", {}).get("merges", []))
    vocab_size = len(data.get("model", {}).get("vocab", {}))

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  model       : {model_type}  ({n_merges} merges, vocab {vocab_size})")

    print(f"  normalizer  : ", end="")
    if normalizer is None:
        print("(none)")
    else:
        ntype = normalizer.get("type", "unknown")
        if ntype == "Sequence":
            subtypes = [s.get("type","?") for s in normalizer.get("normalizers", [])]
            print(f"Sequence({', '.join(subtypes)})")
        else:
            print(ntype)

    print(f"  pretokenizer:")
    for line in describe_pretokenizer(pretok, indent=2):
        print(line)

    # Also show a sample tokenization if possible
    return {
        "name": name,
        "model_type": model_type,
        "n_merges": n_merges,
        "vocab_size": vocab_size,
        "pretokenizer": pretok,
        "normalizer": normalizer,
    }


def compare_pretokenizers(results):
    """Check if all pretokenizers are identical."""
    import hashlib

    def canonical(p):
        return json.dumps(p, sort_keys=True)

    hashes = {r["name"]: hashlib.md5(canonical(r["pretokenizer"]).encode()).hexdigest()
              for r in results}
    unique = Counter(hashes.values())

    print(f"\n{'─'*60}")
    print(f"  Pretokenizer compatibility summary")
    print(f"{'─'*60}")
    print(f"  Unique pretokenizer configs: {len(unique)}")

    if len(unique) == 1:
        print(f"  All tokenizers share identical pretokenizer config.")
    else:
        # Group by hash
        groups = {}
        for name, h in hashes.items():
            groups.setdefault(h, []).append(name)
        print(f"  Groups:")
        for i, (h, names) in enumerate(groups.items()):
            print(f"    Group {i+1}: {', '.join(names)}")

    return len(unique) == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizers", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--from_hub", action="store_true")
    args = parser.parse_args()

    names = args.names if args.names else args.tokenizers

    results = []
    for path, name in zip(args.tokenizers, names):
        data = load_tokenizer_json_from_hub(path) if args.from_hub else load_tokenizer_json(path)
        r = inspect_tokenizer(data, name)
        results.append(r)

    compare_pretokenizers(results)


if __name__ == "__main__":
    main()