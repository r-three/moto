#! /bin/bash

# bash moto/supertokenizer/create_scale_checkpoints.sh

OUTPUT_DIR=/path/to/tokenizers/scale-7B/

## download the original
# hf download craffel/llama1b_supertoken --include "model.pth" --local-dir=$PROJECT/aip-craffel/gsa/tokenizers/supertokenizer/checkpoints/llama_1B-original
ORIGINAL_PATH=/path/to/tokenizers/supertokenizer/checkpoints/llama_1B-original/model.pth

python moto/supertokenizer/supervocab.py \
	--tokenizers \
	tiktoken/gpt-4o \
	google-bert/bert-base-multilingual-cased \
	facebook/xglm-564M \
	google/gemma-2-2b \
	microsoft/Phi-3-mini-4k-instruct \
	tokenmonster/englishcode-32000-consistent-v1 \
	google/byt5-small \
	bigscience/bloom \
	gpt2 \
	mistralai/tekken \
	meta-llama/Llama-3.2-1B \
	Qwen/Qwen3-8B \
	CohereLabs/aya-expanse-8b \
	common-pile/comma-v0.1 \
	google-bert/bert-base-uncased \
	google-t5/t5-base \
	google/mt5-base \
	tiktoken/gpt-4 \
	mistralai/Mistral-7B-Instruct-v0.3 \
	--output_dir $OUTPUT_DIR \
	--hf_upload_repo_prefix gsaltintas/toksuite-scale-7B-

python moto/supertokenizer/create_checkpoints.py \
    --lingua_base_config="configs/lingua/llama_7B.yaml" \
	--supervocab_size=942324 \
	--hf_upload_repo_prefix gsaltintas/toksuite-scale-7B- \
	--model_dir $OUTPUT_DIR/checkpoints \
	--create_distributed \
	--vocab_dir $OUTPUT_DIR \
	--tokenizers \
	meta-llama/Llama-3.2-1B \
	tokenmonster/englishcode-32000-consistent-v1 \
	--superset $OUTPUT_DIR/checkpoints/superset.pth\
	#   --create_superset \

# 	# tiktoken/gpt-4o \
# 	# google-bert/bert-base-multilingual-cased \
# 	# facebook/xglm-564M \
# 	# google/gemma-2-2b \
# 	# microsoft/Phi-3-mini-4k-instruct \
# 	# google/byt5-small \
# 	# bigscience/bloom \
# 	# gpt2 \
# 	# mistralai/tekken \
# 	# Qwen/Qwen3-8B \
# 	# CohereLabs/aya-expanse-8b \
# 	# common-pile/comma-v0.1 \

OUTPUT_DIR=/path/to/tokenizers/scale-300M/

python moto/supertokenizer/supervocab.py \
	--tokenizers \
	tiktoken/gpt-4o \
	google-bert/bert-base-multilingual-cased \
	facebook/xglm-564M \
	google/gemma-2-2b \
	microsoft/Phi-3-mini-4k-instruct \
	tokenmonster/englishcode-32000-consistent-v1 \
	google/byt5-small \
	bigscience/bloom \
	gpt2 \
	mistralai/tekken \
	meta-llama/Llama-3.2-1B \
	Qwen/Qwen3-8B \
	CohereLabs/aya-expanse-8b \
	common-pile/comma-v0.1 \
	google-bert/bert-base-uncased \
	google-t5/t5-base \
	google/mt5-base \
	tiktoken/gpt-4 \
	mistralai/Mistral-7B-Instruct-v0.3 \
	--output_dir $OUTPUT_DIR \
	--hf_upload_repo_prefix gsaltintas/toksuite-scale-300M-

python moto/supertokenizer/create_checkpoints.py \
	--lingua_base_config="configs/lingua/llama_300M.yaml" \
	--supervocab_size=942324 \
	--hf_upload_repo_prefix gsaltintas/toksuite-scale-300M- \
	--model_dir $OUTPUT_DIR/checkpoints \
	--create_distributed \
	--vocab_dir $OUTPUT_DIR/individual_tokenizers \
	--superset $OUTPUT_DIR/checkpoints/superset.pth --tokenizers \
	meta-llama/Llama-3.2-1B \
	tokenmonster/englishcode-32000-consistent-v1 \
	--create_superset \
