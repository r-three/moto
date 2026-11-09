#!/usr/bin/env sh

# bash moto/supertokenizer/create_supervocab_for_individual_bpes.sh
code_name="8k_v2"
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_$code_name/individual_tokenizers

python moto/supertokenizer/supervocab.py \
	tokenizers=["flexitok/bpe_ltr_arb_Arab_8000_v2","flexitok/bpe_ltr_ces_Latn_8000_v2","flexitok/bpe_ltr_cmn_Hani_8000_v2","flexitok/bpe_ltr_dan_Latn_8000_v2","flexitok/bpe_ltr_deu_Latn_8000_v2","flexitok/bpe_ltr_ell_Grek_8000_v2","flexitok/bpe_ltr_fas_Arab_8000_v2","flexitok/bpe_ltr_fra_Latn_8000_v2","flexitok/bpe_ltr_fw_edu_8000_v2","flexitok/bpe_ltr_hun_Latn_8000_v2","flexitok/bpe_ltr_ind_Latn_8000_v2","flexitok/bpe_ltr_ita_Latn_8000_v2","flexitok/bpe_ltr_jpn_Jpan_8000_v2","flexitok/bpe_ltr_nld_Latn_8000_v2","flexitok/bpe_ltr_pol_Latn_8000_v2","flexitok/bpe_ltr_por_Latn_8000_v2","flexitok/bpe_ltr_rus_Cyrl_8000_v2","flexitok/bpe_ltr_spa_Latn_8000_v2","flexitok/bpe_ltr_swe_Latn_8000_v2","flexitok/bpe_ltr_tur_Latn_8000_v2","flexitok/bpe_ltr_vie_Latn_8000_v2"] \
	output_dir=$OUTPUT_DIR \
	return_weights=true \
	hf_upload_repo_prefix=flexitok/supertokenizer-$code_name- \
	hf.collections=["flexitok/$code_name"] \
	hf.publish_to_hf=true \
	hf.hf_repo_id=flexitok/supertokenizer-$code_name\
	upload_to_original_repo=true \
	superset_code_name="$code_name" 

## get vocab size from config.yaml
vocab_size=$(python -c "import yaml; print(yaml.safe_load(open('$OUTPUT_DIR/config.yaml'))['model']['vocab_size'])")

## Create configs



#TODO auto parse
exit(0)
# upload init ckpt
python moto/supertokenizer/create_checkpoints.py \
	hf_upload_repo_prefix=flexitok/supertokenizer-$code_name- \
	hf.collections=["flexitok/$code_name"] \
	hf.publish_to_hf=true \
	supervocab_size=$vocab_size \
	hf.hf_repo_id=flexitok/supertokenizer-$code_name \
	superset_code_name="$code_name" \
    seed=777 \
	create_superset=true
