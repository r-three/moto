#!/usr/bin/env sh

# bash moto/supertokenizer/create_supervocab_for_individual_bpes.sh
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_fineweb_2_hq/individual_tokenizers

python moto/supertokenizer/supervocab.py \
	tokenizers=["flexitok/bpe_arb_Arab_8000","flexitok/bpe_ces_Latn_8000","flexitok/bpe_cmn_Hani_8000","flexitok/bpe_dan_Latn_8000","flexitok/bpe_deu_Latn_8000","flexitok/bpe_ell_Grek_8000","flexitok/bpe_fas_Arab_8000","flexitok/bpe_fra_Latn_8000","flexitok/bpe_fw_edu_8000","flexitok/bpe_hun_Latn_8000","flexitok/bpe_ind_Latn_8000","flexitok/bpe_ita_Latn_8000","flexitok/bpe_jpn_Jpan_8000","flexitok/bpe_nld_Latn_8000","flexitok/bpe_pol_Latn_8000","flexitok/bpe_por_Latn_8000","flexitok/bpe_rus_Cyrl_8000","flexitok/bpe_spa_Latn_8000","flexitok/bpe_swe_Latn_8000","flexitok/bpe_tur_Latn_8000","flexitok/bpe_vie_Latn_8000"] \
	output_dir=$OUTPUT_DIR \
	return_weights=true \
	hf_upload_repo_prefix=flexitok/supertokenizer-fw2hq- \
	hf.collections=["flexitok/fineweb2-hq-superset-tokenizers"] \
	hf.publish_to_hf=true \
	hf.hf_repo_id=flexitok/supertokenizer-fineweb2_hq \
	upload_to_original_repo=true \
	superset_code_name="fineweb2_hq" \

# upload init ckpt
python moto/supertokenizer/create_checkpoints.py \
	hf_upload_repo_prefix=flexitok/supertokenizer-fw2hq- \
	hf.collections=["flexitok/fineweb2-hq-superset-tokenizers"] \
	hf.publish_to_hf=true \
	supervocab_size=113764 \
	hf.hf_repo_id=flexitok/supertokenizer-fineweb2_hq \
	superset_code_name="fineweb2_hq" \
    seed=777 \
	create_superset=true
