# scripts/compute_cjk_alphabet.py
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import regex


def compute_cjk_alphabet(data_dir, lang, coverage=0.999, max_lines=-1):
    shard_files = sorted(glob.glob(str(Path(data_dir) / lang / "*.jsonl")))
    counter = Counter()
    han = regex.compile(r"\p{Han}|\p{Hiragana}|\p{Katakana}")
    count = 0
    for fpath in shard_files:
        with open(fpath) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    counter.update(han.findall(obj.get("text", "")))
                except: continue
                count += 1
                if max_lines > 0 and count >= max_lines:
                    break
    
    total = sum(counter.values())
    cumsum = 0
    chars = []
    for char, cnt in counter.most_common():
        chars.append(char)
        cumsum += cnt
        if cumsum / total >= coverage:
            break
    return chars, counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--coverage", type=float, default=0.999)
    args = parser.parse_args()
    
    chars, counter = compute_cjk_alphabet(args.data_dir, args.lang, args.coverage)
    
    out_path = Path(args.data_dir) / args.lang / "cjk_alphabet.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for char in chars:
            freq = counter[char]
            f.write(f"{char}\t{freq}\n")  # save freq too, useful for debugging
    
    print(f"Saved {len(chars)} chars ({args.coverage*100}% coverage) to {out_path}")

# python moto/tokenizer_training/precompute_cjk_chars.py --data_dir=/path/to/data/flexitok --lang=cmn_Hani --coverage=1.
# python moto/tokenizer_training/precompute_cjk_chars.py --data_dir=/path/to/data/flexitok --lang=jpn_Jpan --coverage=.999