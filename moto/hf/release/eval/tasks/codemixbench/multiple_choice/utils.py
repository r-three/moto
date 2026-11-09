import re


# Matches an option marker like "(A)" or "(A):" appearing at a token boundary.
_OPTION_RE = re.compile(r"\(([A-Z])\):?")


def _find_choices(sentence: str) -> list[str]:
    """Return the option letters (e.g. ["A", "B", "C", "D"]) for a doc.

    Each CodeMixBench multiple-choice `sentence` embeds the question
    followed by its answer options formatted as "(A): ...", "(B): ...", etc.
    The question stem can itself contain stray "(X)" style parentheticals
    (e.g. gene names, citations), so we look for the *rightmost* run of
    option markers that is fully consecutive (A, B, C, ...) through the end
    of the string -- that run is the real options block.
    """
    matches = list(_OPTION_RE.finditer(sentence))
    for i in range(len(matches) - 1, -1, -1):
        if matches[i].group(1) != "A":
            continue
        seq = [m.group(1) for m in matches[i:]]
        if seq == [chr(ord("A") + j) for j in range(len(seq))]:
            return seq
    return []


def doc_to_text(doc: dict) -> str:
    return doc["sentence"]


def doc_to_choice(doc: dict) -> list[str]:
    return _find_choices(doc["sentence"])
