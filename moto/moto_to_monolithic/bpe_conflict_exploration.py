"""
Explore cycle frequency when merging two BPE merge tables via DAG union.

Usage (local tokenizer.json files):
    python explore_merge_cycles.py --tok1 path/to/tok1/tokenizer.json --tok2 path/to/tok2/tokenizer.json

Usage (HF hub, if network is available):
    python explore_merge_cycles.py --tok1 gpt2 --tok2 facebook/mbart-large-cc25 --from_hub

For MoTo: pass your per-language tokenizer.json paths.
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations

# ── BPE extraction ────────────────────────────────────────────────────────────

def load_merge_list_from_file(path):
    """
    Load merge list from a HF tokenizer.json file.
    Returns: list of (a, b) tuples in merge order (index = rank).
    """
    with open(path) as f:
        data = json.load(f)

    model = data.get("model", {})
    model_type = model.get("type", "")

    if model_type != "BPE":
        raise ValueError(f"Expected BPE tokenizer, got: {model_type}")

    merges = model.get("merges", [])
    result = []
    for m in merges:
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        elif isinstance(m, list):
            a, b = m[0], m[1]
        else:
            raise ValueError(f"Unexpected merge format: {m}")
        result.append((a, b))
    return result


def load_merge_list_from_hub(repo_id):
    import json

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo_id)
    data = json.loads(tok.backend_tokenizer.to_str())
    merges_raw = data["model"]["merges"]
    return [tuple(m.split(" ", 1)) if isinstance(m, str) else (m[0], m[1]) for m in merges_raw]


# ── DAG construction ──────────────────────────────────────────────────────────

def build_dag(merge_list):
    """
    Build dependency DAG from a BPE merge list.

    merge (a, b) -> ab depends on whatever merges produced a and b.

    Returns:
        merge_to_rank : dict (a,b) -> int
        produced_by   : dict token_str -> int  (rank that produced it; base tokens absent)
        deps          : dict rank -> set of ranks it depends on
        rank_to_merge : dict rank -> (a, b)
    """
    produced_by = {}
    deps = defaultdict(set)
    merge_to_rank = {}
    rank_to_merge = {}

    for rank, (a, b) in enumerate(merge_list):
        merge_to_rank[(a, b)] = rank
        rank_to_merge[rank] = (a, b)

        if a in produced_by:
            deps[rank].add(produced_by[a])
        if b in produced_by:
            deps[rank].add(produced_by[b])

        produced_by[a + b] = rank

    return merge_to_rank, produced_by, deps, rank_to_merge


# ── Union DAG & cycle detection ───────────────────────────────────────────────

def find_cycles_in_union(merge_list1, merge_list2, max_cycles=200):
    """
    Build union DAG of two BPE merge tables and detect cycles.
    """
    m2r1, _, deps1, r2m1 = build_dag(merge_list1)
    m2r2, _, deps2, r2m2 = build_dag(merge_list2)

    set1, set2 = set(m2r1), set(m2r2)
    shared = set1 & set2
    only_1 = set1 - shared
    only_2 = set2 - shared
    all_merges = set1 | set2

    print(f"\n── Vocabulary overlap ──────────────────────────────")
    print(f"  Tok1 merges   : {len(merge_list1):>7}")
    print(f"  Tok2 merges   : {len(merge_list2):>7}")
    print(f"  Shared        : {len(shared):>7}  "
          f"({100*len(shared)/len(set1):.1f}% of tok1, "
          f"{100*len(shared)/len(set2):.1f}% of tok2)")
    print(f"  Only in tok1  : {len(only_1):>7}")
    print(f"  Only in tok2  : {len(only_2):>7}")

    # ── Build adjacency: u must come before v ──
    # graph[u].add(v) means "u is a dependency of v"
    graph = defaultdict(set)  # merge_tuple -> set of merge_tuples

    def add_dep_edges(deps, rank_to_merge):
        for rank, dep_ranks in deps.items():
            node = rank_to_merge[rank]
            for d in dep_ranks:
                dep_node = rank_to_merge[d]
                graph[dep_node].add(node)

    add_dep_edges(deps1, r2m1)
    add_dep_edges(deps2, r2m2)

    # ── Ordering conflicts among shared merges ──
    # For each pair of shared merges, check if their relative order is reversed
    # across the two tables. If so, adding both ordering edges creates a cycle.
    shared_list = sorted(shared)
    ordering_conflicts = []

    for m1, m2 in combinations(shared_list, 2):
        r1_in_1, r1_in_2 = m2r1[m1], m2r2[m1]
        r2_in_1, r2_in_2 = m2r1[m2], m2r2[m2]

        tok1_order = r1_in_1 < r2_in_1  # True: m1 before m2 in tok1
        tok2_order = r1_in_2 < r2_in_2  # True: m1 before m2 in tok2

        if tok1_order != tok2_order:
            ordering_conflicts.append((m1, m2, r1_in_1, r2_in_1, r1_in_2, r2_in_2))
            # Both orderings added -> guaranteed cycle for this pair
            graph[m1].add(m2)
            graph[m2].add(m1)

    n_shared_pairs = len(shared_list) * (len(shared_list) - 1) // 2
    print(f"\n── Ordering conflicts (shared pairs with reversed relative rank) ──")
    print(f"  Shared merge pairs     : {n_shared_pairs:>8}")
    print(f"  Conflicting pairs      : {len(ordering_conflicts):>8}  "
          f"({100*len(ordering_conflicts)/max(n_shared_pairs,1):.2f}%)")

    if ordering_conflicts:
        print(f"\n  Sample conflicts (first 10):")
        for m1, m2, r1_1, r1_2, r2_1, r2_2 in ordering_conflicts[:10]:
            print(f"    {m1!r:20s} tok1:{r1_1:5d} tok2:{r1_2:5d}")
            print(f"    {m2!r:20s} tok1:{r2_1:5d} tok2:{r2_2:5d}")
            print()

    # ── Cycle detection via iterative DFS ──
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    cycles_found = []

    def dfs(start):
        stack = [(start, iter(graph[start]), [start])]
        color[start] = GRAY

        while stack:
            node, neighbors, path = stack[-1]
            try:
                nxt = next(neighbors)
                if color[nxt] == GRAY:
                    # cycle: path from nxt back to nxt
                    idx = path.index(nxt)
                    cycles_found.append(path[idx:] + [nxt])
                    if len(cycles_found) >= max_cycles:
                        return True
                elif color[nxt] == WHITE:
                    color[nxt] = GRAY
                    stack.append((nxt, iter(graph[nxt]), path + [nxt]))
            except StopIteration:
                color[node] = BLACK
                stack.pop()
        return False

    for node in list(all_merges):
        if color[node] == WHITE:
            if dfs(node):
                break

    print(f"\n── Cycle analysis ──────────────────────────────────")
    print(f"  Cycles found (cap {max_cycles})  : {len(cycles_found)}")

    if cycles_found:
        lengths = [len(c) - 1 for c in cycles_found]
        print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.1f}")

        # Are cycles caused only by shared-merge ordering conflicts,
        # or also by dependency edges?
        conflict_set = set()
        for m1, m2, *_ in ordering_conflicts:
            conflict_set.add((m1, m2))
            conflict_set.add((m2, m1))

        dep_caused = 0
        for cycle in cycles_found:
            is_ordering_only = all(
                (cycle[i], cycle[i+1]) in conflict_set
                for i in range(len(cycle)-1)
            )
            if not is_ordering_only:
                dep_caused += 1

        print(f"  Cycles from ordering conflicts only : {len(cycles_found)-dep_caused}")
        print(f"  Cycles involving dependency edges   : {dep_caused}")

        print(f"\n  Sample cycles (first 5):")
        for cycle in cycles_found[:5]:
            print(f"    {' -> '.join(repr(x) for x in cycle)}")

    # ── Rank distance for conflicting shared merges ──
    if ordering_conflicts:
        rank_gaps_1 = [abs(r1_1 - r2_1) for _, _, r1_1, _, r2_1, _ in ordering_conflicts]
        rank_gaps_2 = [abs(r1_2 - r2_2) for _, _, _, r1_2, _, r2_2 in ordering_conflicts]
        print(f"\n── Rank distance of conflicting pairs ──────────────")
        print(f"  Tok1 rank gap: min={min(rank_gaps_1)}, max={max(rank_gaps_1)}, "
              f"avg={sum(rank_gaps_1)/len(rank_gaps_1):.1f}")
        print(f"  Tok2 rank gap: min={min(rank_gaps_2)}, max={max(rank_gaps_2)}, "
              f"avg={sum(rank_gaps_2)/len(rank_gaps_2):.1f}")

    return {
        "n_merges_1": len(merge_list1),
        "n_merges_2": len(merge_list2),
        "n_shared": len(shared),
        "n_only_1": len(only_1),
        "n_only_2": len(only_2),
        "n_shared_pairs": n_shared_pairs,
        "n_ordering_conflicts": len(ordering_conflicts),
        "n_cycles": len(cycles_found),
        "ordering_conflicts": ordering_conflicts,
        "cycles": cycles_found,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

# python -m moto.moto_to_monolithic.bpe_conflict_exploration --tok1 flexitok/bpe_ltr_fra_Latn_8000_v3 --tok2 flexitok/bpe_ltr_ita_Latn_8000_v3 --from_hub
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok1", required=True, help="Path to tokenizer.json (or HF repo ID with --from_hub)")
    parser.add_argument("--tok2", required=True, help="Path to tokenizer.json (or HF repo ID with --from_hub)")
    parser.add_argument("--from_hub", action="store_true", help="Load from HF hub instead of local file")
    parser.add_argument("--save", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    if args.from_hub:
        print("Loading from HF hub...")
        ml1 = load_merge_list_from_hub(args.tok1)
        ml2 = load_merge_list_from_hub(args.tok2)
    else:
        print(f"Loading tokenizer files...")
        ml1 = load_merge_list_from_file(args.tok1)
        ml2 = load_merge_list_from_file(args.tok2)

    print(f"  Tok1: {args.tok1}  ({len(ml1)} merges)")
    print(f"  Tok2: {args.tok2}  ({len(ml2)} merges)")

    results = find_cycles_in_union(ml1, ml2)

    if args.save:
        def serialize(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, list):
                return [serialize(x) for x in obj]
            return obj

        with open(args.save, "w") as f:
            json.dump({k: serialize(v) for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()