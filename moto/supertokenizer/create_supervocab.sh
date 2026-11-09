#!/usr/bin/env sh

# bash moto/supertokenizer/create_supervocab.sh
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer/individual_tokenizers

python moto/supertokenizer/supervocab.py \
	tokenizers=["google/byt5-small","tiktoken/gpt-4o","google-bert/bert-base-multilingual-cased","facebook/xglm-564M","google/gemma-2-2b","microsoft/Phi-3-mini-4k-instruct","tokenmonster/englishcode-32000-consistent-v1","bigscience/bloom","gpt2","mistralai/tekken","meta-llama/Llama-3.2-1B","Qwen/Qwen3-8B","CohereLabs/aya-expanse-8b","common-pile/comma-v0.1"] \
	output_dir=$OUTPUT_DIR \
	return_weights=true \
	hf_upload_repo_prefix=gsaltintas/supertokenizer-

# google-bert/bert-base-uncased \
# google-t5/t5-base \
# google/mt5-base \
# tiktoken/gpt-4 \
# mistralai/Mistral-7B-Instruct-v0.3 \
