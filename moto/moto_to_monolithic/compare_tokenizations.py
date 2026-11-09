"""
Tokenize target sentences with two tokenizers and highlight where they
disagree on segmentation boundaries -- i.e. find real surface-level
consequences of merge-order differences (e.g. "relat ions" vs "rela tions").

Usage:
    python -m moto.moto_to_monolithic.compare_tokenizations \
        from_hub=true tok1=flexitok/bpe_ltr_fra_Latn_8000_v3 tok2=flexitok/bpe_ltr_fw_edu_8000_v3 \
        sentences_file=sentences.txt
"""
import json
from dataclasses import dataclass, field

from moto.utils.args import Args, load_args


def load_tokenizer(repo_id=None, path=None):
    from transformers import AutoTokenizer
    if repo_id:
        return AutoTokenizer.from_pretrained(repo_id)
    return AutoTokenizer.from_pretrained(path)


def tokenize(tok, text):
    """Return the list of token ids for a piece of text."""
    return tok.encode(text, add_special_tokens=False)


def char_spans(tok, ids):
    """
    Given token ids, decode each one individually and reconstruct character
    offsets by walking through the decoded pieces. This avoids relying on
    the raw byte-level alphabet strings (e.g. Ã©) and instead uses the
    tokenizer's own decoder, which is robust to the byte-to-unicode mapping.
    """
    spans = []
    pos = 0
    for tid in ids:
        piece = tok.decode([tid])
        length = len(piece)
        spans.append((pos, pos + length))
        pos += length
    return spans


def find_boundary_diff(tok1, ids1, tok2, ids2):
    """
    Compare two tokenizations of the same string by their character
    boundaries. Returns (agree: bool, boundaries1, boundaries2).
    """
    spans1 = char_spans(tok1, ids1)
    spans2 = char_spans(tok2, ids2)
    boundaries1 = {s[1] for s in spans1[:-1]} if spans1 else set()
    boundaries2 = {s[1] for s in spans2[:-1]} if spans2 else set()
    return boundaries1 == boundaries2, boundaries1, boundaries2


def render_segmentation(tok, ids):
    """
    Render tokens as readable text segments separated by '|', decoding each
    token id individually through the tokenizer's own decoder so byte-level
    BPE alphabet characters (e.g. Ã©) are converted back to real UTF-8 text.
    """
    pieces = []
    for tid in ids:
        piece = tok.decode([tid])
        pieces.append(piece if piece != "" else "·")
    return "|".join(pieces)


def sanity_check(tok, name, sample="Hello, world! Ça va?"):
    """Verify a tokenizer round-trips correctly before trusting its output."""
    ids = tok.encode(sample, add_special_tokens=False)
    decoded = tok.decode(ids)
    ok = decoded.strip() == sample.strip()
    print(f"  [{name}] round-trip check: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print(f"      input : {sample!r}")
        print(f"      output: {decoded!r}")
    return ok


@dataclass
class CompareArgs(Args):
    tok1: str
    tok2: str
    from_hub: bool = False
    sentences_file: str = ""
    sentences: list = field(default_factory=list)  # inline fallback


def main():
    args = load_args(CompareArgs)

    tok1 = load_tokenizer(repo_id=args.tok1) if args.from_hub else load_tokenizer(path=args.tok1)
    tok2 = load_tokenizer(repo_id=args.tok2) if args.from_hub else load_tokenizer(path=args.tok2)

    print("── Sanity checks (round-trip encode/decode) ──")
    ok1 = sanity_check(tok1, "tok1")
    ok2 = sanity_check(tok2, "tok2")
    if not (ok1 and ok2):
        print("\n  WARNING: a tokenizer failed its round-trip check. Results below "
              "may be unreliable until this is fixed.\n")

    if args.sentences_file:
        with open(args.sentences_file) as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        sentences = args.sentences

    if not sentences:
        sentences = [
            "The international relations department published a report.",
            "Les relations internationales sont importantes.",
            "He decided to submit the import documents.",
            "Il a dû soumettre les documents d'importation.",
        ]

    n_total, n_diff = 0, 0
    for sent in sentences:
        n_total += 1
        ids1 = tokenize(tok1, sent)
        ids2 = tokenize(tok2, sent)
        agree, b1, b2 = find_boundary_diff(tok1, ids1, tok2, ids2)

        print(f"\n{'─'*70}")
        print(f"  sentence: {sent!r}")
        print(f"{'─'*70}")
        print(f"  tok1: {render_segmentation(tok1, ids1)}")
        print(f"  tok2: {render_segmentation(tok2, ids2)}")

        if agree:
            print(f"  [SAME segmentation]")
        else:
            n_diff += 1
            only1 = sorted(b1 - b2)
            only2 = sorted(b2 - b1)
            print(f"  [DIFFERENT segmentation]")
            print(f"    boundaries only in tok1: {only1}")
            print(f"    boundaries only in tok2: {only2}")
            for pos in only1 + only2:
                lo, hi = max(0, pos - 8), min(len(sent), pos + 8)
                marker = " " * (pos - lo) + "^"
                print(f"      near pos {pos}: ...{sent[lo:hi]}...")
                print(f"                {' '*8}{marker}")

    print(f"\n{'─'*70}")
    print(f"  Summary: {n_diff}/{n_total} sentences tokenized differently")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()