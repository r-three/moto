"""
Given a list of observed output token strings (as printed by
compare_tokenizations.py, e.g. 'che', 'du', 'le', 'ched', 'ul'), find the
merge(s) that produced each one in each tokenizer, and check whether any
pair of them constitutes a real ordering conflict.

Usage:
    python -m moto.moto_to_monolithic.trace_observed_tokens \
        from_hub=true tok1=flexitok/bpe_ltr_fra_Latn_8000_v3 tok2=flexitok/bpe_ltr_fw_edu_8000_v3 \
        tokens=che,du,le,ched,ul,une,x,pec,un,exp
"""
from dataclasses import dataclass

from moto.moto_to_monolithic.find_dag_example import (
    build_dag,
    check_specific_merges,
    find_merge_by_output,
    load_merges,
)
from moto.utils.args import Args, load_args


@dataclass
class TraceArgs(Args):
    tok1: str
    tok2: str
    from_hub: bool = False
    tokens: str = ""  # comma-separated list of observed token strings


def main():
    args = load_args(TraceArgs)

    ml1 = load_merges(repo_id=args.tok1) if args.from_hub else load_merges(path=args.tok1)
    ml2 = load_merges(repo_id=args.tok2) if args.from_hub else load_merges(path=args.tok2)

    observed = [t for t in args.tokens.split(",") if t]

    print(f"── Locating merges that produce each observed token ──\n")
    found_pairs = []
    for tok_str in observed:
        m1 = find_merge_by_output(ml1, tok_str)
        m2 = find_merge_by_output(ml2, tok_str)
        print(f"  {tok_str!r}:")
        print(f"    tok1 candidates: {m1}")
        print(f"    tok2 candidates: {m2}")
        # Take the first candidate from each side if present
        if m1:
            found_pairs.append(m1[0])
        if m2:
            found_pairs.append(m2[0])

    found_pairs = list(dict.fromkeys(found_pairs))  # dedupe, preserve order
    print(f"\n── Checking all located merges for ordering conflicts ──\n")
    check_specific_merges(ml1, ml2, found_pairs)


if __name__ == "__main__":
    main()