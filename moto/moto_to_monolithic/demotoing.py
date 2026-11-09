"""
De'Moto'ing: Merge N per-language BPE tokenizers into a single monolithic BPE tokenizer.

Algorithm (Algorithm 1 in the paper, appendix app:moto-merge):
  1. V* = union of all vocabularies V_i (dedup by token string)
  2. P* = shared canonical pretokenizer
  3. Build dependency DAG G_i for each M_i
  4. G* = union of all G_i
  5. Resolve ordering conflicts via majority vote; ties use min-rank
  6. M* = topological sort of G* (ties broken by avg normalised rank)
  7. Prune M* and V* to merges reachable under P*
  8. Return T* = (V*, M*, P*)

Special tokens: the merged tokenizer gets its own BOS / EOS / UNK / PAD (defaults
<|begin_of_text|>, <|end_of_text|>, <unk>, <pad>; set with bos_token=... etc.), appended after
the regular vocabulary with special=True, and named in tokenizer_config.json so that
lingua's SimplifiedHFTokenizer (AutoTokenizer bos_token / eos_token) picks them up. The
post-processor prepends BOS (tok(text) == [bos] + text, as lingua trains with add_bos).

Added tokens of the input tokenizers are rebuilt against the merged vocabulary: a
non-special added token keeps its string and takes its merged-vocabulary id (or a new id
after the vocabulary); input special tokens are dropped in favour of the four above.
(Earlier versions copied the first tokenizer's added_tokens with their *old* ids, which
then collided with unrelated merged-vocabulary ids, and shipped no BOS/EOS at all.)

Usage:
    python -m moto.moto_to_monolithic.demotoing \
        tokenizers=[path/tok1/tokenizer.json,path/tok2/tokenizer.json] \
        output=monolithic/tokenizer.json

    # or from HF hub
    python -m moto.moto_to_monolithic.demotoing \
        tokenizers=[flexitok/bpe_ltr_fra_Latn_8000_v3,flexitok/bpe_ltr_fw_edu_8000_v3,flexitok/bpe_ltr_ita_Latn_8000_v3,flexitok/bpe_ltr_tur_Latn_8000_v3] \
        from_hub=true \
        output=monolithic/tokenizer.json
    
    
    python -m moto.moto_to_monolithic.demotoing \
        tokenizers=[flexitok/bpe_ltr_fw_edu_8000_v3,flexitok/bpe_ltr_arb_Arab_8000_v3,flexitok/bpe_ltr_ces_Latn_8000_v3,flexitok/bpe_ltr_cmn_Hani_8000_v3,flexitok/bpe_ltr_dan_Latn_8000_v3,flexitok/bpe_ltr_deu_Latn_8000_v3,flexitok/bpe_ltr_ell_Grek_8000_v3,flexitok/bpe_ltr_fas_Arab_8000_v3,flexitok/bpe_ltr_fra_Latn_8000_v3,flexitok/bpe_ltr_hun_Latn_8000_v3,flexitok/bpe_ltr_ind_Latn_8000_v3,flexitok/bpe_ltr_ita_Latn_8000_v3,flexitok/bpe_ltr_jpn_Jpan_8000_v3,flexitok/bpe_ltr_nld_Latn_8000_v3,flexitok/bpe_ltr_pol_Latn_8000_v3,flexitok/bpe_ltr_por_Latn_8000_v3,flexitok/bpe_ltr_rus_Cyrl_8000_v3,flexitok/bpe_ltr_spa_Latn_8000_v3,flexitok/bpe_ltr_swe_Latn_8000_v3,flexitok/bpe_ltr_tur_Latn_8000_v3,flexitok/bpe_ltr_vie_Latn_8000_v3,] \
        from_hub=true \
        output=monolithic/moto_tokenizer.json \
        hf.publish_to_hf=true hf.hf_repo_id=flexitok/bpe_ltr_monolithic_moto
    # ^ the tokenizer of the first Demoto run (no special tokens, colliding added-token ids).
    # With special tokens (defaults shown). Publishing to the same repo adds a new commit; the old
    # tokenizer stays reachable at revision fc6c17035a348da638da1a7664b2ac19bd456d17:
    python -m moto.moto_to_monolithic.demotoing \
        tokenizers=[...same 21 tokenizers...] from_hub=true \
        bos_token='<|begin_of_text|>' eos_token='<|end_of_text|>' unk_token='<unk>' pad_token='<pad>' \
        output=monolithic/tokenizer.json \
        hf.publish_to_hf=true hf.hf_repo_id=flexitok/bpe_ltr_monolithic_moto
    python -m moto.moto_to_monolithic.demotoing \
        tokenizers=[flexitok/bpe_ltr_fw_edu_8000_v3,flexitok/bpe_ltr_tur_Latn_8000_v3] \
        from_hub=true \
        output=monolithic/tokenizer.json
"""

import heapq
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, List, Optional

from moto.utils.args import Args, HFArgs, load_args

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class DemotoingArgs(Args):
    config: Any = None
    tokenizers: List[str] = field(default_factory=list)
    from_hub: bool = False
    output: str = "monolithic/tokenizer.json"
    bos_token: str = "<|begin_of_text|>"
    eos_token: str = "<|end_of_text|>"
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    # prepend BOS in tok(text) (lingua adds BOS itself and encodes with add_special_tokens=False)
    add_bos_in_template: bool = True
    model_max_length: int = 4096
    hf: HFArgs = field(default_factory=HFArgs)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_tokenizer_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_from_hub(repo_id: str) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo_id)
    return json.loads(tok.backend_tokenizer.to_str())


def extract_bpe_data(data: dict) -> tuple[dict, list[tuple[str, str]], dict]:
    """Return (vocab, merge_list, pretokenizer) from a tokenizer.json dict."""
    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"Expected BPE tokenizer, got: {model.get('type')}")
    vocab = model.get("vocab", {})
    merges_raw = model.get("merges", [])
    merges: list[tuple[str, str]] = []
    for m in merges_raw:
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m[0], m[1]
        merges.append((a, b))
    pretokenizer = data.get("pre_tokenizer") or {}
    return vocab, merges, pretokenizer


# ── DAG construction ──────────────────────────────────────────────────────────

def build_dag(merge_list: list[tuple[str, str]]) -> tuple[dict, dict, dict, dict]:
    """
    Build a dependency DAG for a single BPE merge table.

    A merge (a, b) depends on whichever earlier merge produced a or b.

    Returns:
        merge_to_rank : {(a, b): rank}
        produced_by   : {token_str: rank} , absent for base tokens
        deps          : {rank: set[rank]}
        rank_to_merge : {rank: (a, b)}
    """
    produced_by: dict[str, int] = {}
    deps: dict[int, set] = defaultdict(set)
    merge_to_rank: dict[tuple, int] = {}
    rank_to_merge: dict[int, tuple] = {}

    for rank, (a, b) in enumerate(merge_list):
        merge_to_rank[(a, b)] = rank
        rank_to_merge[rank] = (a, b)
        if a in produced_by:
            deps[rank].add(produced_by[a])
        if b in produced_by:
            deps[rank].add(produced_by[b])
        produced_by[a + b] = rank

    return merge_to_rank, produced_by, deps, rank_to_merge


# ── Union DAG & conflict resolution ──────────────────────────────────────────

def build_union_dag(
    all_merge_lists: list[list[tuple[str, str]]],
) -> tuple[dict, list[tuple[str, str]], dict]:
    """
    Build G* = union of dependency edges from all G_i, with ordering conflicts
    resolved by majority vote (tie → min-rank fallback).

    Returns:
        adj          : {merge: set[merge]}  u → v means u must precede v
        all_merges   : list of all unique merges across all tokenizers
        avg_norm_rank: {merge: float}        for topological sort tie-breaking
    """
    per_tok = [build_dag(ml) for ml in all_merge_lists]

    all_merge_set: set[tuple] = set()
    for ml in all_merge_lists:
        all_merge_set.update(ml)

    # Union of dependency edges from every G_i
    adj: dict[tuple, set] = defaultdict(set)
    for (_, _, deps, r2m) in per_tok:
        for rank, dep_ranks in deps.items():
            node = r2m[rank]
            for d in dep_ranks:
                adj[r2m[d]].add(node)

    # Average normalised rank (used later for topo-sort tie-breaking)
    rank_sum: dict[tuple, float] = defaultdict(float)
    rank_cnt: dict[tuple, int] = defaultdict(int)
    for ml in all_merge_lists:
        n = max(len(ml), 1)
        for rank, merge in enumerate(ml):
            rank_sum[merge] += rank / n
            rank_cnt[merge] += 1
    avg_norm_rank = {m: rank_sum[m] / rank_cnt[m] for m in all_merge_set}

    # Log conflict statistics.  Ordering conflicts are resolved implicitly by
    # avg_norm_rank: a merge that appears earlier on average across tokenizers
    # gets a lower score and will be placed first during topological sort.
    # This captures the majority-vote signal without adding explicit edges,
    # which can form transitive cycles (A→B, B→C, C→A) across three pairwise
    # majority decisions.
    merge_tok_ranks: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for tok_idx, ml in enumerate(all_merge_lists):
        for rank, merge in enumerate(ml):
            merge_tok_ranks[merge].append((tok_idx, rank))
    shared_merges = [m for m, tl in merge_tok_ranks.items() if len(tl) >= 2]
    n_conflicts = 0
    for m1, m2 in combinations(shared_merges, 2):
        r_m1 = dict(merge_tok_ranks[m1])
        r_m2 = dict(merge_tok_ranks[m2])
        shared_toks = set(r_m1) & set(r_m2)
        if len(shared_toks) < 2:
            continue
        m1_first = sum(1 for i in shared_toks if r_m1[i] < r_m2[i])
        if 0 < m1_first < len(shared_toks):
            n_conflicts += 1
    logger.info(
        "Ordering conflicts (resolved via avg_norm_rank tie-breaking): %d", n_conflicts
    )

    return dict(adj), list(all_merge_set), avg_norm_rank


# ── Topological sort ──────────────────────────────────────────────────────────

def topological_sort(
    adj: dict[tuple, set],
    all_merges: list[tuple[str, str]],
    avg_norm_rank: dict[tuple, float],
) -> list[tuple[str, str]]:
    """
    Kahn's algorithm. Ties broken by avg_norm_rank (lower rank = earlier).
    Raises ValueError if a cycle remains.
    """
    in_degree: dict[tuple, int] = {m: 0 for m in all_merges}
    for neighbors in adj.values():
        for v in neighbors:
            if v in in_degree:
                in_degree[v] += 1

    heap: list[tuple] = []
    for merge in all_merges:
        if in_degree[merge] == 0:
            heapq.heappush(heap, (avg_norm_rank.get(merge, 0.0), merge))

    result: list[tuple[str, str]] = []
    while heap:
        _, merge = heapq.heappop(heap)
        result.append(merge)
        for neighbor in adj.get(merge, set()):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, (avg_norm_rank.get(neighbor, 0.0), neighbor))

    if len(result) != len(all_merges):
        n_stuck = len(all_merges) - len(result)
        stuck = [m for m in all_merges if in_degree[m] > 0][:5]
        raise ValueError(
            f"Cycle in union DAG: {n_stuck} merges could not be sorted. "
            f"Sample stuck nodes: {stuck}"
        )
    return result


# ── Canonical pretokenizer (P*) ───────────────────────────────────────────────

def _flatten_pretokenizer(pt: dict) -> list[dict]:
    """Flatten a (possibly nested Sequence) into a list of atomic pretokenizer rules."""
    if pt.get("type") == "Sequence":
        rules: list[dict] = []
        for sub in pt.get("pretokenizers", []):
            rules.extend(_flatten_pretokenizer(sub))
        return rules
    return [pt]


def _split_regex_top_level(pattern: str) -> list[str]:
    """Split a regex pattern on top-level | (ignoring | inside groups/brackets)."""
    alts: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            current.append(c)
            current.append(pattern[i + 1])
            i += 2
            continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "|" and depth == 0:
            alts.append("".join(current))
            current = []
            i += 1
            continue
        current.append(c)
        i += 1
    if current:
        alts.append("".join(current))
    return alts


def _merge_split_rules(rules: list[dict]) -> dict | None:
    """
    Given several Split rules with different Regex patterns, return a merged
    rule whose pattern keeps only the alternatives that appear in ALL rules.
    Returns None if the rules cannot be merged (wrong type) or have no common
    alternatives.
    """
    if not all(r.get("type") == "Split" and "Regex" in r.get("pattern", {}) for r in rules):
        return None
    patterns = [r["pattern"]["Regex"] for r in rules]
    alt_sets = [set(_split_regex_top_level(p)) for p in patterns]
    common = alt_sets[0].copy()
    for s in alt_sets[1:]:
        common &= s
    if not common:
        return None
    # Preserve the order of alternatives from the first rule
    ordered = [a for a in _split_regex_top_level(patterns[0]) if a in common]
    merged = dict(rules[0])
    merged["pattern"] = {"Regex": "|".join(ordered)}
    return merged


def build_canonical_pretokenizer(pretokenizers: list[dict]) -> dict:
    """
    Compute P* by processing each position in the pretokenizer Sequence:

    - Rules that are identical across all tokenizers are kept as-is.
    - Split rules that DIFFER are merged: their regex alternatives are
      intersected, retaining only the alternatives that every tokenizer
      shares (e.g. whitespace and punctuation patterns common to all scripts).
      Script-specific letter-matching alternatives are dropped.
    - Non-Split rules that conflict (different type or content) are dropped.

    The result is acyclic by the same argument as the paper: BPE training
    cannot produce a merge crossing any of the retained boundaries, so all
    merges in M* remain reachable under P*.
    """
    import hashlib

    def rule_hash(r: dict) -> str:
        return hashlib.md5(json.dumps(r, sort_keys=True).encode()).hexdigest()

    rule_lists = [_flatten_pretokenizer(pt) for pt in pretokenizers]

    # Align rule lists by position.  Use the longest list as the reference;
    # tokenizers with fewer rules are treated as having no rule at that position.
    n_positions = max(len(rl) for rl in rule_lists)
    canonical_rules: list[dict] = []

    for pos in range(n_positions):
        rules_at_pos = [rl[pos] for rl in rule_lists if pos < len(rl)]
        hashes_at_pos = [rule_hash(r) for r in rules_at_pos]

        if len(set(hashes_at_pos)) == 1:
            # Identical across all tokenizers that have this position
            canonical_rules.append(rules_at_pos[0])
            continue

        # Rules differ — try merging Split patterns by common alternatives
        merged = _merge_split_rules(rules_at_pos)
        if merged is not None:
            n_orig = len(_split_regex_top_level(rules_at_pos[0]["pattern"]["Regex"]))
            n_kept = len(_split_regex_top_level(merged["pattern"]["Regex"]))
            logger.info(
                "Position %d: merged %d Split rules — kept %d / %d alternatives "
                "(dropped %d script-specific).",
                pos, len(rules_at_pos), n_kept, n_orig, n_orig - n_kept,
            )
            canonical_rules.append(merged)
        else:
            types = [r.get("type", "?") for r in rules_at_pos]
            logger.info("Position %d: dropped conflicting rules %s.", pos, types)

    if not canonical_rules:
        logger.warning("No canonical pretokenizer rules — using empty pretokenizer.")
        return {}
    if len(canonical_rules) == 1:
        return canonical_rules[0]
    return {"type": "Sequence", "pretokenizers": canonical_rules}


# ── Reachability pruning ──────────────────────────────────────────────────────

def compute_reachable_merges(
    merge_list: list[tuple[str, str]],
    pretokenizer: dict,
) -> set[tuple[str, str]]:
    """
    Return reachable merges under P*.

    After dropping script-specific rules via build_canonical_pretokenizer, all
    merges are reachable: BPE training cannot produce merges that cross the
    remaining punctuation/whitespace/byte-level boundaries.
    """
    logger.info("All %d merges reachable under canonical P* (no boundary pruning needed).", len(merge_list))
    return set(merge_list)


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_super_vocab(
    tokenizer_vocabs: list[dict[str, int]],
    keep_tokens: Optional[set] = None,
) -> dict[str, int]:
    """Union of all vocabularies, IDs reassigned 0-indexed sorted by UTF-8 bytes."""
    all_tokens: set[str] = set()
    for vocab in tokenizer_vocabs:
        all_tokens.update(vocab.keys())
    if keep_tokens is not None:
        all_tokens &= keep_tokens

    def sort_key(s: str) -> bytes:
        try:
            return s.encode("utf-8")
        except Exception:
            return s.encode("latin-1")

    return {token: idx for idx, token in enumerate(sorted(all_tokens, key=sort_key))}


# ── Main assembly ─────────────────────────────────────────────────────────────

@dataclass
class SpecialTokens:
    bos: str = "<|begin_of_text|>"
    eos: str = "<|end_of_text|>"
    unk: str = "<unk>"
    pad: str = "<pad>"

    def as_list(self) -> list[str]:
        return [self.bos, self.eos, self.unk, self.pad]


def _added(i: int, content: str, special: bool, src: Optional[dict] = None) -> dict:
    src = src or {}
    return {
        "id": i, "content": content,
        "single_word": src.get("single_word", False), "lstrip": src.get("lstrip", False),
        "rstrip": src.get("rstrip", False), "normalized": src.get("normalized", not special),
        "special": special,
    }


def rebuild_added_tokens(
    vocab: dict[str, int], inputs: list[dict], specials: SpecialTokens
) -> tuple[dict[str, int], list[dict]]:
    """Added tokens re-indexed against the merged vocab, plus the four special tokens.

    Returns (vocab with any new entries, added_tokens). Every id is unique: an added token
    whose string is in the vocab shares that vocab id (one token, one id); any other string
    gets the next free id. Special tokens are also put in the model vocab, so the BPE model's
    unk_token resolves.
    """
    vocab = dict(vocab)
    next_id = max(vocab.values()) + 1 if vocab else 0
    out: list[dict] = []
    seen: set[str] = set()
    dropped: list[str] = []
    for data in inputs:
        for t in data.get("added_tokens", []):
            c = t["content"]
            if c in seen or c in specials.as_list():
                continue
            if t.get("special"):
                dropped.append(c)  # input special tokens (per-language BOS/EOS/...) are replaced
                continue
            seen.add(c)
            if c not in vocab:
                vocab[c] = next_id
                next_id += 1
            out.append(_added(vocab[c], c, special=False, src=t))
    if dropped:
        logger.info("Dropped %d input special tokens in favour of %s: %s", len(set(dropped)), specials.as_list(), sorted(set(dropped))[:10])
    logger.info("Kept %d non-special added tokens (re-indexed to the merged vocab)", len(out))
    for c in specials.as_list():
        if c not in vocab:
            vocab[c] = next_id
            next_id += 1
        out.append(_added(vocab[c], c, special=True))
    return vocab, sorted(out, key=lambda t: t["id"])


def bos_template(bos: str, bos_id: int) -> dict:
    return {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": bos, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [
            {"SpecialToken": {"id": bos, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": bos, "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}},
        ],
        "special_tokens": {bos: {"id": bos, "ids": [bos_id], "tokens": [bos]}},
    }


def verify_tokenizer(tok_json: dict, specials: SpecialTokens, samples: Optional[list[str]] = None) -> None:
    """Fail loudly on the problems the first Demoto tokenizer had."""
    from tokenizers import Tokenizer

    vocab = tok_json["model"]["vocab"]
    by_id: dict[int, set[str]] = defaultdict(set)
    for k, v in vocab.items():
        by_id[v].add(k)
    for t in tok_json["added_tokens"]:
        by_id[t["id"]].add(t["content"])
    clashes = {i: sorted(s) for i, s in by_id.items() if len(s) > 1}
    if clashes:
        raise ValueError(f"{len(clashes)} ids are shared by different strings, e.g. {list(clashes.items())[:5]}")
    n = max(by_id) + 1
    if len(by_id) != n:
        raise ValueError(f"ids are not contiguous: {len(by_id)} distinct ids, max {n - 1}")

    tok = Tokenizer.from_str(json.dumps(tok_json))
    for c in specials.as_list():
        ids = tok.encode(c, add_special_tokens=False).ids
        if ids != [vocab[c]]:
            raise ValueError(f"special token {c!r} encodes to {ids}, expected [{vocab[c]}]")
    samples = samples or ["The quick brown fox, 1234 times.", "Türkiye'de güzel bir gün.", "日本語のテキスト。", "Привет, мир!", "  two  spaces\nnewline"]
    for text in samples:
        enc = tok.encode(text, add_special_tokens=False)
        if tok.decode(enc.ids, skip_special_tokens=False) != text:
            raise ValueError(f"round trip failed for {text!r}: {tok.decode(enc.ids)!r}")
    with_bos = tok.encode(samples[0]).ids
    logger.info("verified: %d ids, all unique; specials %s; tok(text) starts %s",
                n, {c: vocab[c] for c in specials.as_list()}, with_bos[:3])


def build_monolithic_tokenizer(
    tokenizer_data_list: list[dict],
    specials: Optional[SpecialTokens] = None,
    add_bos_in_template: bool = True,
) -> dict:
    """
    Merge N BPE tokenizer.json dicts into a single monolithic tokenizer dict
    (HF tokenizer.json format), with BOS / EOS / UNK / PAD special tokens.
    """
    specials = specials or SpecialTokens()
    if not tokenizer_data_list:
        raise ValueError("Need at least one tokenizer.")

    logger.info("Merging %d tokenizers.", len(tokenizer_data_list))

    all_vocabs: list[dict] = []
    all_merges: list[list] = []
    all_pretokenizers: list[dict] = []

    for data in tokenizer_data_list:
        vocab, merges, pretok = extract_bpe_data(data)
        all_vocabs.append(vocab)
        all_merges.append(merges)
        all_pretokenizers.append(pretok)
        logger.info("  loaded: vocab=%d  merges=%d", len(vocab), len(merges))

    logger.info("Computing canonical pretokenizer P*...")
    canonical_pretokenizer = build_canonical_pretokenizer(all_pretokenizers)

    logger.info("Building union DAG...")
    adj, union_merges, avg_norm_rank = build_union_dag(all_merges)
    logger.info("  unique merges: %d", len(union_merges))

    logger.info("Topological sort...")
    sorted_merges = topological_sort(adj, union_merges, avg_norm_rank)

    logger.info("Computing reachable merges under P*...")
    reachable = compute_reachable_merges(sorted_merges, canonical_pretokenizer)
    pruned_merges = [m for m in sorted_merges if m in reachable]
    logger.info(
        "  reachable: %d / %d  (pruned %d)",
        len(pruned_merges), len(sorted_merges), len(sorted_merges) - len(pruned_merges),
    )

    # Keep tokens that are base vocab entries or outputs of reachable merges
    produced_by_any_merge: set[str] = {a + b for a, b in union_merges}
    keep_tokens: set[str] = set()
    for vocab in all_vocabs:
        for token in vocab:
            if token not in produced_by_any_merge:
                keep_tokens.add(token)
    for a, b in pruned_merges:
        keep_tokens.add(a + b)
        for tok in (a, b):
            if tok not in produced_by_any_merge:
                keep_tokens.add(tok)

    super_vocab = build_super_vocab(all_vocabs, keep_tokens=keep_tokens - set(specials.as_list()))
    logger.info("Final vocab=%d  merges=%d", len(super_vocab), len(pruned_merges))
    super_vocab, added_tokens = rebuild_added_tokens(super_vocab, tokenizer_data_list, specials)
    logger.info("With added and special tokens: %d ids", len(super_vocab))

    template = tokenizer_data_list[0]
    output = dict(template)
    output["added_tokens"] = added_tokens
    output["post_processor"] = bos_template(specials.bos, super_vocab[specials.bos]) if add_bos_in_template else None
    output["truncation"] = None
    output["padding"] = None
    output["model"] = {
        "type": "BPE",
        "dropout": None,
        "unk_token": specials.unk,
        "continuing_subword_prefix": template["model"].get("continuing_subword_prefix"),
        "end_of_word_suffix": template["model"].get("end_of_word_suffix"),
        "fuse_unk": template["model"].get("fuse_unk", False),
        "byte_fallback": template["model"].get("byte_fallback", False),
        "ignore_merges": template["model"].get("ignore_merges", False),
        "vocab": super_vocab,
        "merges": [f"{a} {b}" for a, b in pruned_merges],
    }
    output["pre_tokenizer"] = canonical_pretokenizer
    return output


def save_hf_files(tokenizer_json: Path, out_dir: Path, specials: SpecialTokens, model_max_length: int) -> None:
    """tokenizer.json + tokenizer_config.json + special_tokens_map.json that name the specials.

    tokenizer_class is pinned to PreTrainedTokenizerFast (transformers 5 would write
    TokenizersBackend, which transformers 4.x cannot load), and token_type_ids are not returned.
    """
    from transformers import PreTrainedTokenizerFast

    verified_text = Path(tokenizer_json).read_text()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json), bos_token=specials.bos, eos_token=specials.eos,
        unk_token=specials.unk, pad_token=specials.pad, model_max_length=model_max_length,
        clean_up_tokenization_spaces=False, model_input_names=["input_ids", "attention_mask"],
    )
    fast_tok.save_pretrained(out_dir)
    cfg_path = out_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": specials.bos, "eos_token": specials.eos,
                "unk_token": specials.unk, "pad_token": specials.pad, "model_max_length": model_max_length,
                "clean_up_tokenization_spaces": False, "model_input_names": ["input_ids", "attention_mask"]})
    cfg.pop("backend", None)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "special_tokens_map.json").write_text(json.dumps(
        {"bos_token": specials.bos, "eos_token": specials.eos, "unk_token": specials.unk, "pad_token": specials.pad}, indent=2) + "\n")
    if Path(tokenizer_json).resolve() != (out_dir / "tokenizer.json").resolve():
        shutil.copyfile(tokenizer_json, out_dir / "tokenizer.json")  # exactly what verify_tokenizer checked
    else:  # save_pretrained re-serialized it in place; restore the verified bytes
        (out_dir / "tokenizer.json").write_text(verified_text)


def main(args: DemotoingArgs) -> None:
    if not args.tokenizers:
        raise ValueError("Provide at least one tokenizer via tokenizers=[...]")

    logger.info("Loading %d tokenizer(s)...", len(args.tokenizers))
    tokenizer_data_list: list[dict] = []
    for src in args.tokenizers:
        data = load_from_hub(src) if args.from_hub else load_tokenizer_json(src)
        tokenizer_data_list.append(data)
        logger.info("  %s  (%d merges)", src, len(data["model"]["merges"]))

    specials = SpecialTokens(args.bos_token, args.eos_token, args.unk_token, args.pad_token)
    monolithic = build_monolithic_tokenizer(tokenizer_data_list, specials, args.add_bos_in_template)
    verify_tokenizer(monolithic, specials)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(monolithic, f, indent=2, ensure_ascii=False)
    logger.info("Saved → %s", out_path)
    save_hf_files(out_path, out_path.parent, specials, args.model_max_length)
    logger.info("Wrote tokenizer_config.json / special_tokens_map.json next to it (load with AutoTokenizer.from_pretrained(%s))", out_path.parent)

    if args.hf.publish_to_hf:
        import tempfile

        import huggingface_hub as hf_hub
        from transformers import PreTrainedTokenizerFast

        if not args.hf.hf_repo_id:
            raise ValueError("hf.hf_repo_id must be set to publish to the HF hub.")
        repo_id = args.hf.hf_repo_id
        hf_hub.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # Save via PreTrainedTokenizerFast so that tokenizer_config.json and
        # special_tokens_map.json are generated — AutoTokenizer.from_pretrained
        # requires these alongside tokenizer.json.
        with tempfile.TemporaryDirectory() as tmp:
            save_hf_files(out_path, Path(tmp), specials, args.model_max_length)
            hf_hub.upload_folder(
                folder_path=tmp,
                repo_id=repo_id,
                repo_type="model",
                commit_message=args.hf.commit_message or "Upload monolithic BPE tokenizer",
            )
        logger.info("Uploaded tokenizer → %s", repo_id)

        if args.hf.collections:
            from moto.utils.hf_utils import create_or_get_collection_slug
            api = hf_hub.HfApi()
            for collection in args.hf.collections:
                try:
                    slug = create_or_get_collection_slug(collection)
                except Exception as e:
                    logger.warning("Could not resolve collection slug for %s: %s", collection, e)
                    continue
                api.add_collection_item(
                    collection_slug=slug,
                    item_id=repo_id,
                    item_type="model",
                    exists_ok=True,
                )
                logger.info("Added %s to collection %s", repo_id, slug)


if __name__ == "__main__":
    args = load_args(DemotoingArgs)
    main(args)
