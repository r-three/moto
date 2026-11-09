"""
Find a real ordering conflict between two tokenizers and trace its
dependency chain back to byte leaves, for use as a figure example.

Usage:
    python -m moto.moto_to_monolithic.find_dag_examples \
        from_hub=true tok1=flexitok/bpe_ltr_fra_Latn_8000_v3 tok2=flexitok/bpe_ltr_tur_Latn_8000_v3 \
        min_rank_frac=0.1 max_rank_frac=0.9 min_depth=2 max_depth=5
"""
import json
from collections import defaultdict
from dataclasses import dataclass

from moto.utils.args import Args, load_args


def load_merges(repo_id=None, path=None):
    if repo_id:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id)
        data = json.loads(tok.backend_tokenizer.to_str())
    else:
        with open(path) as f:
            data = json.load(f)
    merges = data["model"]["merges"]
    return [(m.split(" ", 1) if isinstance(m, str) else (m[0], m[1])) for m in merges]


def build_dag(merge_list):
    produced_by, deps, m2r, r2m = {}, defaultdict(set), {}, {}
    for rank, (a, b) in enumerate(merge_list):
        m2r[(a, b)] = rank
        r2m[rank] = (a, b)
        if a in produced_by: deps[rank].add(produced_by[a])
        if b in produced_by: deps[rank].add(produced_by[b])
        produced_by[a + b] = rank
    return m2r, deps, r2m


def trace_to_leaves(merge, m2r, deps, r2m):
    """Walk back from a merge to its byte-leaf ancestry, depth-limited."""
    if merge not in m2r:
        return merge  # base token / leaf
    rank = m2r[merge]
    a, b = merge
    left = trace_to_leaves(a, m2r, deps, r2m) if a in {v for v in [None]} else a
    # simpler: just recursively unwrap a and b if they themselves are merge outputs
    def unwrap(tok):
        for (x, y), r in m2r.items():
            if x + y == tok and r < rank:
                return (x, y)
        return tok
    return (unwrap(a), unwrap(b))


def find_cycle_in_depth_band(merge_list1, merge_list2, min_depth=2, max_depth=5,
                               min_rank_frac=0.1, max_rank_frac=0.9, max_cycle_len=6):
    """
    Build the union DAG (dependency edges from both tokenizers + ordering-conflict
    edges from shared merges with reversed relative rank), then find an actual
    cycle in it, restricted to merges in the given depth/rank band so the
    resulting cycle is interesting enough to draw.

    Returns: list of (a,b) merge tuples forming a cycle (last == first), or None.
    """
    m2r1, deps1, r2m1 = build_dag(merge_list1)
    m2r2, deps2, r2m2 = build_dag(merge_list2)
    shared = set(m2r1) & set(m2r2)
    n1 = len(merge_list1)

    def depth(rank, deps, memo):
        if rank in memo:
            return memo[rank]
        if not deps[rank]:
            memo[rank] = 0
        else:
            memo[rank] = 1 + max(depth(d, deps, memo) for d in deps[rank])
        return memo[rank]

    memo1, memo2 = {}, {}
    lo_rank, hi_rank = min_rank_frac * n1, max_rank_frac * n1

    # Restrict to merges within the band (using tok1's rank/depth as reference)
    band_merges = {
        m for m, r in m2r1.items()
        if lo_rank <= r <= hi_rank and min_depth <= depth(r, deps1, memo1) <= max_depth
    }

    # Build adjacency: dependency edges (within band) from both tokenizers
    graph = defaultdict(set)

    def add_dep_edges(deps, r2m):
        for rank, dep_ranks in deps.items():
            node = r2m[rank]
            if node not in band_merges:
                continue
            for d in dep_ranks:
                dep_node = r2m[d]
                if dep_node in band_merges:
                    graph[dep_node].add(node)

    add_dep_edges(deps1, r2m1)
    add_dep_edges(deps2, r2m2)

    # Add ordering-conflict edges among shared band merges
    conflict_edges = set()
    band_shared = sorted(shared & band_merges)
    for i, m1 in enumerate(band_shared):
        for m2 in band_shared[i+1:]:
            order1 = m2r1[m1] < m2r1[m2]
            order2 = m2r2[m1] < m2r2[m2]
            if order1 != order2:
                graph[m1].add(m2)
                graph[m2].add(m1)
                conflict_edges.add((m1, m2))
                conflict_edges.add((m2, m1))

    # DFS cycle detection, prefer short cycles within the band
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    found_cycles = []

    def dfs(start):
        stack = [(start, iter(graph[start]), [start])]
        color[start] = GRAY
        while stack:
            node, neighbors, path = stack[-1]
            try:
                nxt = next(neighbors)
                if color[nxt] == GRAY:
                    idx = path.index(nxt)
                    cyc = path[idx:] + [nxt]
                    if len(cyc) - 1 <= max_cycle_len:
                        found_cycles.append(cyc)
                elif color[nxt] == WHITE:
                    color[nxt] = GRAY
                    stack.append((nxt, iter(graph[nxt]), path + [nxt]))
            except StopIteration:
                color[node] = BLACK
                stack.pop()

    for node in list(band_merges):
        if color[node] == WHITE:
            dfs(node)
        if len(found_cycles) >= 50:
            break

    if not found_cycles:
        return None, conflict_edges

    # Prefer cycles that mix dependency edges and conflict edges (more interesting)
    def mixed_score(cyc):
        edges = list(zip(cyc[:-1], cyc[1:]))
        n_conflict = sum(1 for e in edges if e in conflict_edges)
        return (n_conflict > 0 and n_conflict < len(edges), -len(edges))

    found_cycles.sort(key=mixed_score, reverse=True)
    return found_cycles[0], conflict_edges


def find_conflict_in_depth_band(merge_list1, merge_list2, min_depth=3, max_depth=5,
                                  min_rank_frac=0.2, max_rank_frac=0.8):
    """
    Find a shared-merge ordering conflict where both merges have dependency
    depth in [min_depth, max_depth] and ranks in the middle band of the
    table (avoids trivial early bigrams and rare tail merges).
    """
    m2r1, deps1, r2m1 = build_dag(merge_list1)
    m2r2, deps2, r2m2 = build_dag(merge_list2)
    shared = set(m2r1) & set(m2r2)
    n1 = len(merge_list1)

    def depth(rank, deps, memo={}):
        if rank in memo:
            return memo[rank]
        if not deps[rank]:
            memo[rank] = 0
        else:
            memo[rank] = 1 + max(depth(d, deps, memo) for d in deps[rank])
        return memo[rank]

    lo_rank, hi_rank = min_rank_frac * n1, max_rank_frac * n1

    candidates = []
    shared_list = sorted(shared)
    for i, m1 in enumerate(shared_list):
        for m2 in shared_list[i+1:]:
            r1_1, r1_2 = m2r1[m1], m2r2[m1]
            r2_1, r2_2 = m2r1[m2], m2r2[m2]
            order1 = r1_1 < r2_1
            order2 = r1_2 < r2_2
            if order1 != order2:
                d1 = depth(r1_1, deps1)
                d2 = depth(r2_1, deps1)
                if (min_depth <= d1 <= max_depth and min_depth <= d2 <= max_depth
                        and lo_rank <= r1_1 <= hi_rank and lo_rank <= r2_1 <= hi_rank):
                    candidates.append((m1, m2, r1_1, r2_1, r1_2, r2_2, d1, d2))

    # Prefer the deepest, most "interesting" examples first
    candidates.sort(key=lambda c: -(c[6] + c[7]))
    return candidates


def get_ancestry_chain(merge, m2r, deps, r2m):
    """
    Return the full ancestor subtree of a merge as a dict:
    rank -> (a, b, [dependency ranks]).
    Useful for rendering the DAG figure (bytes -> ... -> this merge).
    """
    if merge not in m2r:
        return {}
    rank = m2r[merge]
    chain = {}

    def walk(r):
        if r in chain:
            return
        a, b = r2m[r]
        chain[r] = (a, b, sorted(deps[r]))
        for d in deps[r]:
            walk(d)

    walk(rank)
    return chain


@dataclass
class DagCycleArgs(Args):
    tok1: str
    tok2: str
    from_hub: bool = False
    min_depth: int = 2
    max_depth: int = 5
    min_rank_frac: float = 0.1
    max_rank_frac: float = 0.9
    max_cycle_len: int = 6


def main():
    args = load_args(DagCycleArgs)

    ml1 = load_merges(repo_id=args.tok1) if args.from_hub else load_merges(path=args.tok1)
    ml2 = load_merges(repo_id=args.tok2) if args.from_hub else load_merges(path=args.tok2)

    m2r1, deps1, r2m1 = build_dag(ml1)
    m2r2, deps2, r2m2 = build_dag(ml2)

    cycle, conflict_edges = find_cycle_in_depth_band(
        ml1, ml2,
        min_depth=args.min_depth, max_depth=args.max_depth,
        min_rank_frac=args.min_rank_frac, max_rank_frac=args.max_rank_frac,
        max_cycle_len=args.max_cycle_len,
    )

    if cycle is None:
        print("No cycle found in this depth/rank band -- try widening the band.")
        return

    print(f"── Cycle found ({len(cycle)-1} edges) ──")
    edges = list(zip(cycle[:-1], cycle[1:]))
    for u, v in edges:
        if (u, v) in conflict_edges:
            ru1, rv1 = m2r1.get(u), m2r1.get(v)
            ru2, rv2 = m2r2.get(u), m2r2.get(v)
            print(f"  {u!r} -> {v!r}   [CONFLICT (ordering)]")
            first1 = u if ru1 < rv1 else v
            first2 = u if ru2 < rv2 else v
            print(f"      tok1: {u!r}=r{ru1}, {v!r}=r{rv1}  -> {first1!r} comes first")
            print(f"      tok2: {u!r}=r{ru2}, {v!r}=r{rv2}  -> {first2!r} comes first")
        else:
            print(f"  {u!r} -> {v!r}   [dependency]")
            ru1, rv1 = m2r1.get(u), m2r1.get(v)
            print(f"      tok1 ranks: {u!r}=r{ru1}, {v!r}=r{rv1}  (agreed structural order)")

    print(f"\n── Both tokenizers' full rank for every node in the cycle ──")
    for node in cycle[:-1]:
        r1 = m2r1.get(node, "—")
        r2 = m2r2.get(node, "—")
        print(f"  {node!r:25s}  tok1 rank: {r1!s:>6}   tok2 rank: {r2!s:>6}")

    print(f"\n── Ancestry chains for each node in the cycle (tok1) ──")
    for node in cycle[:-1]:
        print(f"\n  node: {node!r}")
        chain = get_ancestry_chain(node, m2r1, deps1, r2m1)
        for rank in sorted(chain):
            a, b, dep_ranks = chain[rank]
            print(f"    rank {rank}: {a!r} + {b!r} -> {a+b!r}  (deps: {dep_ranks})")


if __name__ == "__main__":
    main()