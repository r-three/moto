#! /bin/bash

# bash moto/supertokenizer/create_checkpoints.sh

python moto/supertokenizer/create_checkpoints.py \
    --lingua_base_config="configs/lingua/llama_1B.yaml" \
	--supervocab_size=851586 \
	--create_superset \
	--model_dir $PROJECT/tokenizers/supertokenizer/checkpoints \
	--create_distributed \
	--vocab_dir $PROJECT/tokenizers/supertokenizer/individual_tokenizers \
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
	# --hf_upload_repo_prefix gsaltintas/supertokenizer- \
