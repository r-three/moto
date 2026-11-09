"""Independent check: TokEval's numbers for the routed setting must equal a direct
computation that never touches the driver (per-language tokenizer, raw counts).

Compression rate is total UTF-8 bytes / total tokens. Fertility is tokens per
whitespace-split word. Both are recomputed here from the tokenizers and the
corpus files, and compared with analysis_results.json.
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

root = Path(sys.argv[1] if len(sys.argv) > 1 else "moto_toy")
res = json.loads((root / "out" / "analysis_results.json").read_text())
cfg = json.loads((root / "moto_eval.yaml").read_text())  # fixture writes JSON, which is valid YAML

bad = 0
tot_bytes = tot_tokens = 0
for lang, ent in cfg["settings"]["moto_oracle"]["tokenizers"].items():
    tok = Tokenizer.from_file(ent["path"])
    texts = [l for l in (root / "parallel" / f"{lang}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    n_tok = sum(len(tok.encode(t).ids) for t in texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    # TokEval's per-language fertility is the mean of per-document tokens/words.
    fert_docs = [len(tok.encode(t).ids) / len(t.split()) for t in texts]
    tot_bytes += n_bytes
    tot_tokens += n_tok
    direct_cr = n_bytes / n_tok
    got_cr = res["compression_rate"]["per_tokenizer"]["moto_oracle"]["per_language"][lang]["compression_rate"]
    direct_f = sum(fert_docs) / len(fert_docs)
    got_f = res["fertility"]["per_tokenizer"]["moto_oracle"]["per_language"][lang]["mean"]
    ok_cr = abs(direct_cr - got_cr) < 1e-6
    ok_f = abs(direct_f - got_f) < 1e-6
    print(f"{lang}: compression direct={direct_cr:.4f} tokeval={got_cr:.4f} {'OK' if ok_cr else 'MISMATCH'} | "
          f"fertility direct={direct_f:.4f} tokeval={got_f:.4f} {'OK' if ok_f else 'MISMATCH'}")
    bad += (not ok_cr) + (not ok_f)

glob = res["compression_rate"]["per_tokenizer"]["moto_oracle"]["global"]["compression_rate"]
print(f"global compression direct={tot_bytes / tot_tokens:.4f} tokeval={glob:.4f}")
bad += abs(tot_bytes / tot_tokens - glob) > 1e-6

j = res["compression_rate"]["per_tokenizer"]["joint_bpe"]["global"]["compression_rate"]
print(f"joint_bpe global compression={j:.4f} (must differ from routed: {glob:.4f})")
bad += abs(j - glob) < 1e-9

# moto_repo (MoToTokenizer from the mock release repo) must agree with the string-mapped routed setting
if "moto_repo" in res["compression_rate"]["per_tokenizer"]:
    for metric, path in [("compression_rate", ("compression_rate",)), ("fertility", ("mean",))]:
        for lang in cfg["settings"]["moto_oracle"]["tokenizers"]:
            a = res[metric]["per_tokenizer"]["moto_oracle"]["per_language"][lang]
            b = res[metric]["per_tokenizer"]["moto_repo"]["per_language"][lang]
            for k in path:
                ok = abs(a[k] - b[k]) < 1e-9
                bad += not ok
                print(f"moto_repo vs routed {metric} {lang}: {a[k]:.4f} vs {b[k]:.4f} {'OK' if ok else 'MISMATCH'}")
    sizes = {t: res["vocabulary_utilization"]["per_tokenizer"][t]["global"] for t in ("moto_oracle", "moto_repo")}
    print("vocab utilization routed(union) vs repo(super_vocab + padded size):", {t: round(v["utilization"], 4) for t, v in sizes.items()})
# ---- parity under fixed denominators, recomputed from the raw tokenizers (never touches the driver) ----
import csv
import numpy as np
tok_of = {"joint_bpe": {l: Tokenizer.from_file(str(root / "tok" / "joint.json")) for l in cfg["settings"]["moto_oracle"]["tokenizers"]},
          "moto_oracle": {l: Tokenizer.from_file(ent["path"]) for l, ent in cfg["settings"]["moto_oracle"]["tokenizers"].items()}}
docs = {l: [x for x in (root / "parallel" / f"{l}.txt").read_text(encoding="utf-8").splitlines() if x.strip()] for l in tok_of["moto_oracle"]}
n = {s: {l: [len(tok_of[s][l].encode(t).ids) for t in docs[l]] for l in docs} for s in tok_of}
den = {"english_moto": n["moto_oracle"]["eng_Latn"], "joint": n["joint_bpe"]["eng_Latn"]}
den["self_moto"], den["self_joint"] = None, None
with open(root / "out" / "parity_long.csv") as f:
    got = {(r["setting"], r["language"], r["denominator"]): float(r["mean_parity"]) for r in csv.DictReader(f)}
for setting, direct in (("moto_oracle", "moto_repo"), ("joint_bpe", "joint_bpe")):
    for lang in docs:
        for label in ("english_moto", "joint", "self"):
            d = den[label] if label != "self" else n[setting]["eng_Latn"]
            exp = float(np.mean([a / b for a, b in zip(n[setting][lang], d) if b > 0]))
            key = (direct, lang, label)
            ok = abs(exp - got[key]) < 1e-9
            bad += not ok
            print(f"parity {direct:10s} {lang} vs {label:12s} direct={exp:.4f} driver={got[key]:.4f} {'OK' if ok else 'MISMATCH'}")
# the fixed denominator is the same for every row: english_moto, English column of moto_repo is its own parity (=1)
assert abs(got[("moto_repo", "eng_Latn", "english_moto")] - 1.0) < 1e-9
print("moto_repo English vs english_moto = 1.0; joint_bpe English vs english_moto =", round(got[("joint_bpe", "eng_Latn", "english_moto")], 4))
sys.exit(1 if bad else 0)
