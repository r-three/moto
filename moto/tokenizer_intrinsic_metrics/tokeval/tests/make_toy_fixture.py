"""Build a small fixture to smoke-test run_moto_tokeval.py without FLORES+ or HF access.

Words are random syllable strings with a Zipfian frequency profile, so that
tokenizers of different vocabulary sizes really do segment differently (a corpus
with a handful of word types makes every tokenizer learn whole words and every
setting score identically).

  uv run --project ~/tokeval python tests/make_toy_fixture.py /tmp/moto_toy [path/to/remote/tokenizer.py]
  python -m moto.tokenizer_intrinsic_metrics.tokeval.run_moto_tokeval \
      config=/tmp/moto_toy/moto_eval.yaml output_dir=/tmp/moto_toy/out no_plots=true
  uv run --project ~/tokeval python tests/check_routed_matches_direct.py /tmp/moto_toy
"""
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "moto_toy")
ROOT.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)

SYLLABLES = {
    "eng_Latn": "the ing er an on ver tion est str ough ight ble ck sh th wh ly ed al".split(),
    "tur_Latn": "lar ler ın im çı ğa şe ük ma da ki yor muş bir ço cuk kay nak".split(),
    "deu_Latn": "sch ung keit ich ein der die ge ber lich zu au ei ür hen fen str wei".split(),
}
NUMBERS = ["1234567", "2024", "3.14159", "100000", "42"]


def make_lexicon(lang, n=1500):
    syl = SYLLABLES[lang]
    return ["".join(rng.choices(syl, k=rng.randint(1, 4))) for _ in range(n)]


LEX = {l: make_lexicon(l) for l in SYLLABLES}
ZIPF = {l: [1 / (i + 1) for i in range(len(LEX[l]))] for l in LEX}


def sentences(lang, n):
    out = []
    for _ in range(n):
        ws = rng.choices(LEX[lang], weights=ZIPF[lang], k=rng.randint(6, 16))
        if rng.random() < 0.2:
            ws.insert(rng.randrange(len(ws)), rng.choice(NUMBERS))
        out.append(" ".join(ws).capitalize() + ".")
    return out


def train(texts, vocab_size, path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<unk>"]))
    tok.save(str(path))


(ROOT / "parallel").mkdir(exist_ok=True)
(ROOT / "tok").mkdir(exist_ok=True)
train_texts, langs = {}, {}
for lang in SYLLABLES:
    (ROOT / "parallel" / f"{lang}.txt").write_text("\n".join(sentences(lang, 300)) + "\n", encoding="utf-8")
    train_texts[lang] = sentences(lang, 3000)
    train(train_texts[lang], 600, ROOT / "tok" / f"{lang}.json")
    langs[lang] = {"name": lang, "iso_code": lang[:2], "data_path": str((ROOT / "parallel" / f"{lang}.txt").resolve())}
train([t for v in train_texts.values() for t in v], 1000, ROOT / "tok" / "joint.json")

# ---- mock MoTo release repo (same layout as a MoTo HF model repo), using the real tokenizer.py ----
import shutil
REPO = ROOT / "moto_repo"
SRC = {"eng_Latn": "fw_edu", "tur_Latn": "tur_Latn", "deu_Latn": "deu_Latn"}  # routing source of each corpus language
BOS, EOS = "~SPECIAL~ALIGNED~BOS~SYMBOL~", "</s>"
superset = {}
for lang in SYLLABLES:
    tj = json.loads((ROOT / "tok" / f"{lang}.json").read_text())
    for t in sorted(tj["model"]["vocab"], key=tj["model"]["vocab"].get):
        superset.setdefault(t, len(superset))
for t in (BOS, EOS):
    superset.setdefault(t, len(superset))
(REPO / "tokenizers").mkdir(parents=True, exist_ok=True)
comps = []
for lang, src in SRC.items():
    d = REPO / "tokenizers" / src
    d.mkdir(parents=True, exist_ok=True)
    tj = json.loads((ROOT / "tok" / f"{lang}.json").read_text())
    (d / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    (d / "super_mapping.json").write_text(json.dumps({str(i): superset[t] for t, i in tj["model"]["vocab"].items()}))
    comps.append({"name": src, "subfolder": f"tokenizers/{src}", "languages": [src], "id_map": "super_mapping.json"})
(REPO / "super_vocab.json").write_text(json.dumps(superset, ensure_ascii=False))
(REPO / "moto_config.json").write_text(json.dumps({
    "schema_version": 1, "default_tokenizer": "fw_edu", "vocab_size": len(superset) + 7,  # +7 padding rows, like a real embedding matrix
    "bos_token_id": superset[BOS], "eos_token_id": superset[EOS], "eos_token": EOS, "add_bos": True, "add_eos": True,
    "routing": {"mode": "oracle", "source_to_tokenizer": {v: v for v in SRC.values()}, "language_aliases": {"eng_Latn": "fw_edu"}},
    "tokenizers": comps, "super_vocab": {"file": "super_vocab.json"}}, indent=2))
(REPO / "tokenizer_config.json").write_text(json.dumps({
    "tokenizer_class": "MoToTokenizer", "auto_map": {"AutoTokenizer": [None, "tokenizer.MoToTokenizer"]},
    "language": "fw_edu", "model_max_length": 512, "clean_up_tokenization_spaces": False, "padding_side": "left"}))
real = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if real is None or not real.exists():
    sys.exit("pass the path to the MoTo remote-code tokenizer.py as 2nd argument (the mock repo needs the real MoToTokenizer)")
shutil.copyfile(real, REPO / "tokenizer.py")

(ROOT / "langs.json").write_text(json.dumps({"languages": langs}, indent=2))
(ROOT / "moto_eval.yaml").write_text(json.dumps({
    "language_config": str((ROOT / "langs.json").resolve()),
    "samples_per_lang": 300,
    "parity_denominators": {"english_moto": "moto_repo", "joint": "joint_bpe"},
    "settings": {
        "joint_bpe": {"mode": "single", "class": "huggingface", "path": str((ROOT / "tok" / "joint.json").resolve())},
        "moto_oracle": {"mode": "routed", "default_language": "eng_Latn", "tokenizers": {
            l: {"class": "huggingface", "path": str((ROOT / "tok" / f"{l}.json").resolve())} for l in SYLLABLES}},
        "moto_repo": {"mode": "moto_repo", "repo": str(REPO.resolve()), "default_language": "eng_Latn"},
    },
}, indent=2))
# no language_config: languages come from the repo's routing table, corpus from corpus_dir
(ROOT / "moto_eval_auto.yaml").write_text(json.dumps({
    "corpus_dir": str((ROOT / "parallel").resolve()),
    "samples_per_lang": 300,
    "settings": {"moto_repo": {"mode": "moto_repo", "repo": str(REPO.resolve())}},
}, indent=2))
print("fixture written to", ROOT)
