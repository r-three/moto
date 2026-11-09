#!/bin/bash
# Train and publish digit-grouping tokenizers for modular arithmetic.
#
# For `a mod b` where b = 2^k × 5^j, only the rightmost max(k,j) digits of a
# determine the answer.  One tokenizer is trained per grouping size (1–5).
#
# Training data: integers 0–99999 written as plain text, one per line.
# Also includes zero-padded trailing groups like "007" as guaranteed single tokens.
# After BPE training, every possible digit group (including zero-padded
# trailing groups like "7") is added as a guaranteed single token.
#
# Usage:
#   bash moto/tokenizer_training/train_mod_tokenizers_w_0s.sh
#   # To publish: set PUBLISH=true and HF_REPO_PREFIX before running.
#
#SBATCH --job-name=train_mod_tokenizers_w_0s
#SBATCH --output=/path/to/.slurm/%j.out
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=aip-craffel
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1

PUBLISH=${PUBLISH:-true}
HF_REPO_PREFIX=${HF_REPO_PREFIX:-"flexitok/mod-tokenizers-zero-padded"}
HF_COLLECTION=${HF_COLLECTION:-"flexitok/mod_tokenizers_zero_padded"}
OUT_DIR=${OUT_DIR:-"$SCRATCH/trained_tokenizers/mod"}
MAX_JOBS=${SLURM_NTASKS:-5}

# ── Generate training data ──────────────────────────────────────────────────
DATA_DIR=$(mktemp -d)
mkdir -p "$DATA_DIR/numeric"
echo "Generating numeric training data in $DATA_DIR/numeric ..."
touch "$DATA_DIR/numeric/numbers.jsonl"

# ── Helpers ─────────────────────────────────────────────────────────────────
wait_for_slot() {
    while (($(jobs -rp | wc -l) >= MAX_JOBS)); do
        wait -n 2>/dev/null || sleep 1
    done
}

run_train() {
    local name=$1 # should be individual, ltr_3digit, rtl_3digit etc
    local vocab_size=$2
	local repo_id=${3:-$name}
	local token_str
	local num_digits_to_pad=1
	if [[ "$name" == "ltr_2digit" || "$name" == "rtl_2digit" ]]; then
		num_digits_to_pad=2
	elif [[ "$name" == "ltr_3digit" || "$name" == "rtl_3digit" ]]; then
		num_digits_to_pad=3
	elif [[ "$name" == "ltr_4digit" || "$name" == "rtl_4digit" ]]; then
		num_digits_to_pad=4
	elif [[ "$name" == "ltr_5digit" || "$name" == "rtl_5digit" ]]; then
		num_digits_to_pad=5
	fi

	token_str=$(seq -s, 0 $((vocab_size - 8)))
	python3 -c "
import json, sys
out = '$DATA_DIR/numeric/numbers.jsonl'
num_digits_to_pad = $num_digits_to_pad

with open(out, 'w') as f:
    for i in range(${vocab_size}-7):
        f.write(json.dumps({'text': str(i)}) + '\n')
        if i > 9:
            f.write(json.dumps({'text': f\"{i:0{num_digits_to_pad}d}\"}) + '\n')

print(f'Wrote to {out}', file=sys.stderr)
"
	echo $(wc -l "$DATA_DIR/numeric/numbers.jsonl") "lines in training data for $name tokenizer (vocab_size=$vocab_size)"

    vocab_size=$((vocab_size + 10 ** (num_digits_to_pad - 1) - 1))
    echo "=== Launching $name (vocab_size=$vocab_size) ==="
    python -m moto.tokenizer_training.train_tokenizers \
        algorithm=bpe \
        vocab_size="$vocab_size" \
		min_frequency=1 \
        langs=[numeric] \
        data_dir="$DATA_DIR" \
        output_dir="$OUT_DIR" \
        number_handling="${name}" \
        unicode_normalization=none \
		use_byte_level_regex=false \
        hf.publish_to_hf="$PUBLISH" \
        "extra_special_tokens=[mod,\"=\",\" \"]" \
		hf.hf_repo_id="${HF_REPO_PREFIX}-${repo_id}" \
        ${HF_COLLECTION:+hf.collections=["$HF_COLLECTION"]} \
		test_string="103500109 mod 67" 
}

# ── Train one tokenizer per digit-group size ────────────────────────────────
# Vocab sizes are chosen to cover all possible digit groups of the given size
# (10^N) plus special tokens and BPE overhead, then rounded up.
#
#   rtl_1digit : 10 groups  → vocab ~  50
#   rtl_2digit : 100 groups → vocab ~ 200
#   rtl_3digit : 1000       → vocab ~ 2000
#   rtl_4digit : 10000      → vocab ~ 15000
#   rtl_5digit : 100000     → vocab ~ 110000

# base: special tokens (4) + mod, =, space (3)
wait_for_slot; run_train individual    17
wait_for_slot; run_train ltr_2digit  107
wait_for_slot; run_train ltr_3digit  1007
wait_for_slot; run_train ltr_4digit  10007
wait_for_slot; run_train ltr_5digit  100007
wait_for_slot; run_train rtl_2digit  107
wait_for_slot; run_train rtl_3digit  1007
wait_for_slot; run_train rtl_4digit  10007
wait_for_slot; run_train rtl_5digit  100007
wait
echo "All tokenizer training jobs completed."
echo "Output: $OUT_DIR"

# Clean up temp training data
rm -rf "$DATA_DIR"


## also create the supertokenizer
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_numbers/individual_tokenizers

python moto/supertokenizer/supervocab.py \
	tokenizers=["flexitok/mod-tokenizers-zero-padded-individual","flexitok/mod-tokenizers-zero-padded-ltr_3digit","flexitok/mod-tokenizers-zero-padded-ltr_2digit","flexitok/mod-tokenizers-zero-padded-ltr_4digit","flexitok/mod-tokenizers-zero-padded-ltr_5digit","flexitok/mod-tokenizers-zero-padded-rtl_2digit","flexitok/mod-tokenizers-zero-padded-rtl_3digit","flexitok/mod-tokenizers-zero-padded-rtl_4digit","flexitok/mod-tokenizers-zero-padded-rtl_5digit"] \
	output_dir=$OUTPUT_DIR \
	return_weights=true \
	hf_upload_repo_prefix=flexitok/supertokenizer-mod-tokenizers-zero-padded- \
	hf.collections=["flexitok/flexitok-mod-tokenizers-zero-padded"] \
	hf.publish_to_hf=true \
	hf.hf_repo_id=flexitok/supertokenizer-mod_tokenizers_zero_padded \
	upload_to_original_repo=true \
	superset_code_name="mod_tokenizers_zero_padded" 
