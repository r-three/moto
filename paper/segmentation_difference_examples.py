"""Illustrative segmentation comparison: MoTo vs mBPE vs Llama-3.2 on parallel FLORES sentences.

For each language the same sentence is segmented by
  * MoTo      -- the language's own 8K tokenizer (flexitok/bpe_ltr_<lang>_8000_v3; oracle routing)
  * mBPE      -- one 128K BPE trained on all languages (GPT4 or Boundless pre-tokenization)
  * Llama-3.2 -- the unmodified Llama 3.2 tokenizer (gated: needs HF_TOKEN with access)

Prints a table of segments + token counts and writes an Inkscape-friendly SVG (hex colours,
Atkinson Hyperlegible) and a txt dump. Byte-level BPE segments are decoded from bytes, so
you see readable text rather than G-dot/mojibake; a token holding part of a multi-byte
character is shown as hex, e.g. <E4 B8>.

Usage:
    uv run --no-project --with datasets==4.0.0 --with tokenizers --with "huggingface_hub<1.0" \
        python paper/segmentation_difference_examples.py --idx 0
    ... --mbpe boundless --langs eng_Latn tur_Latn      # other mBPE variant / languages
(`--no-project` because the project's CUDA torch pin doesn't resolve on macOS; inside the
project's own env, plain `python paper/segmentation_difference_examples.py` works.)
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MOTO_REPO = "flexitok/bpe_ltr_{lang}_8000_v3"
MBPE_REPOS = {
    "gpt4": "flexitok/bpe_gpt4_alll_128000",
    "boundless": "flexitok/bpe_boundless_coverage_AllL_128000",
}
LLAMA_REPO = "meta-llama/Llama-3.2-1B"
FLORES_REPO = "flexitok/flores-21langs"  # FLORES+ restricted to the 21 MoTo languages
# language code -> MoTo tokenizer name, where they differ (English is trained on FineWeb-Edu)
FLORES_TO_MOTO = {"eng_Latn": "fw_edu"}
DEFAULT_LANGS = ["eng_Latn", "tur_Latn", "fra_Latn", "rus_Cyrl", "arb_Arab", "cmn_Hani"]

PALETTE = ["#E8806A", "#9A7FC0", "#5AAAB8", "#7AB88A", "#F0C430", "#D4789C"]
FONT = "Atkinson Hyperlegible"


def load_tok(repo: str) -> Tokenizer:
    return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))


def _byte_decoder() -> dict[str, int]:
    # inverse of GPT-2 bytes_to_unicode (used by ByteLevel BPE)
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BD = _byte_decoder()


def segment(tok: Tokenizer, text: str) -> list[str]:
    """Readable segments; falls back to offsets for non-byte-level vocabularies."""
    enc = tok.encode(text, add_special_tokens=False)
    segs = []
    for t, (s, e) in zip(enc.tokens, enc.offsets):
        if t and all(c in _BD for c in t):
            raw = bytes(_BD[c] for c in t)
            try:
                segs.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                segs.append("<" + " ".join(f"{b:02X}" for b in raw) + ">")
        else:
            segs.append(text[s:e])
    return segs


def flores_sentence(lang: str, split: str, idx: int) -> str:
    from datasets import load_dataset  # datasets==4.0.0 is needed for flores (see pyproject)

    return load_dataset(FLORES_REPO, lang, split=split)[idx]["sentence"]


def svg_rows(blocks, char_w=9.5, row_h=26, label_w=150):
    """blocks: list of (title, [(tokenizer_name, [segs])])."""
    y, parts, width = 20, [], 0
    for title, rows in blocks:
        parts.append(f'<text x="8" y="{y}" font-family="{FONT}" font-size="15" font-weight="bold" fill="#333333">{html.escape(title)}</text>')
        y += row_h
        for name, segs in rows:
            parts.append(f'<text x="8" y="{y + 15}" font-family="{FONT}" font-size="12" fill="#555555">{html.escape(name)} ({len(segs)})</text>')
            x = label_w
            for i, s in enumerate(segs):
                w = max(len(s), 1) * char_w + 6
                col = PALETTE[i % len(PALETTE)]
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w - 2:.1f}" height="20" rx="3" fill="{col}" fill-opacity="0.35" stroke="{col}"/>')
                parts.append(f'<text x="{x + 3:.1f}" y="{y + 15}" font-family="{FONT}" font-size="12" fill="#222222" xml:space="preserve">{html.escape(s)}</text>')
                x += w
            width = max(width, x)
            y += row_h
        y += 10
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + 10:.0f}" height="{y}" viewBox="0 0 {width + 10:.0f} {y}">' + "".join(parts) + "</svg>"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--langs", nargs="+", default=DEFAULT_LANGS, help="FLORES language codes")
    p.add_argument("--idx", type=int, default=0, help="FLORES sentence index (parallel across languages)")
    p.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    p.add_argument("--mbpe", default="gpt4", choices=list(MBPE_REPOS))
    p.add_argument("--out-dir", default=str(Path(__file__).parent / "segmentation_out"))
    a = p.parse_args()

    llama = load_tok(LLAMA_REPO)
    mbpe = load_tok(MBPE_REPOS[a.mbpe])
    blocks, lines = [], []
    for lang in a.langs:
        text = flores_sentence(lang, a.split, a.idx)
        moto = load_tok(MOTO_REPO.format(lang=FLORES_TO_MOTO.get(lang, lang)))
        rows = [("MoTo", segment(moto, text)), (f"mBPE-{a.mbpe}", segment(mbpe, text)), ("Llama-3.2", segment(llama, text))]
        blocks.append((f"{lang}: {text}", rows))
        lines.append(f"\n## {lang}\n{text}")
        for name, segs in rows:
            lines.append(f"{name:14s} {len(segs):4d}  " + " | ".join(segs))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "segmentations.txt").write_text("\n".join(lines))
    (out / "segmentations.svg").write_text(svg_rows(blocks))
    print("\n".join(lines))
    print(f"\nWrote {out}/segmentations.txt and .svg")


if __name__ == "__main__":
    main()
