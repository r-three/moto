#!/usr/bin/env sh

# bash moto/supertokenizer/create_supervocab_for_llama_and_16k.sh
code_name="llama_and_16k_0"
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_$code_name/individual_tokenizers

# python moto/supertokenizer/supervocab.py \
# 	tokenizers=["flexitok/bpe_ltr_arb_Arab_16000_v2","flexitok/bpe_ltr_ces_Latn_16000_v2","flexitok/bpe_ltr_cmn_Hani_16000_v2","flexitok/bpe_ltr_dan_Latn_16000_v2","flexitok/bpe_ltr_deu_Latn_16000_v2","flexitok/bpe_ltr_ell_Grek_16000_v2","flexitok/bpe_ltr_fas_Arab_16000_v2","flexitok/bpe_ltr_fra_Latn_16000_v2","flexitok/llama-3.2-1b-tokenizer","flexitok/bpe_ltr_hun_Latn_16000_v2","flexitok/bpe_ltr_ind_Latn_16000_v2","flexitok/bpe_ltr_ita_Latn_16000_v2","flexitok/bpe_ltr_jpn_Jpan_16000_v2","flexitok/bpe_ltr_nld_Latn_16000_v2","flexitok/bpe_ltr_pol_Latn_16000_v2","flexitok/bpe_ltr_por_Latn_16000_v2","flexitok/bpe_ltr_rus_Cyrl_16000_v2","flexitok/bpe_ltr_spa_Latn_16000_v2","flexitok/bpe_ltr_swe_Latn_16000_v2","flexitok/bpe_ltr_tur_Latn_16000_v2","flexitok/bpe_ltr_vie_Latn_16000_v2"] \
# 	output_dir=$OUTPUT_DIR \
# 	return_weights=true \
# 	hf_upload_repo_prefix=flexitok/supertokenizer-$code_name- \
# 	hf.collections=["flexitok/$code_name"] \
# 	hf.publish_to_hf=true \
# 	hf.hf_repo_id=flexitok/supertokenizer-$code_name\
# 	upload_to_original_repo=true \
# 	superset_code_name="$code_name" 

## get vocab size from config.yaml
vocab_size=$(python -c "import yaml; print(yaml.safe_load(open('$OUTPUT_DIR/config.yaml'))['model']['vocab_size'])")

comments=$(grep "^#" $OUTPUT_DIR/config.yaml)
comments="$comments
# python -m lingua.stool script=apps.main.train config=["/fsx/craffel/lingua/apps/main/configs/flexitok/llama_1b_base.yaml","apps/main/configs/ablations/${code_name}_oracle.yaml"]
# python -m lingua.stool_ccdb ngpu=1 ncpu=1 nodes=1 launcher=bash script=apps.main.train config=["apps/main/configs/flexitok/llama_1b_base.yaml","apps/main/configs/ablations/${code_name}_oracle.yaml"]
"
python -c '
import yaml
config = yaml.safe_load(open("'"$OUTPUT_DIR"'/config.yaml"))
config["data"]["routing"] = {
    "suitable_tokenizer_probability": 1.0,
    "source_to_tokenizer": {
      "arb_Arab": "flexitok/bpe_ltr_arb_Arab_16000_v2",
      "ces_Latn": "flexitok/bpe_ltr_ces_Latn_16000_v2",
      "cmn_Hani": "flexitok/bpe_ltr_cmn_Hani_16000_v2",
      "dan_Latn": "flexitok/bpe_ltr_dan_Latn_16000_v2",
      "deu_Latn": "flexitok/bpe_ltr_deu_Latn_16000_v2",
      "ell_Grek": "flexitok/bpe_ltr_ell_Grek_16000_v2",
      "fas_Arab": "flexitok/bpe_ltr_fas_Arab_16000_v2",
      "fra_Latn": "flexitok/bpe_ltr_fra_Latn_16000_v2",
      "fw_edu": "flexitok/llama-3.2-1b-tokenizer",
      "hun_Latn": "flexitok/bpe_ltr_hun_Latn_16000_v2",
      "ind_Latn": "flexitok/bpe_ltr_ind_Latn_16000_v2",
      "ita_Latn": "flexitok/bpe_ltr_ita_Latn_16000_v2",
      "jpn_Jpan": "flexitok/bpe_ltr_jpn_Jpan_16000_v2",
      "nld_Latn": "flexitok/bpe_ltr_nld_Latn_16000_v2",
      "pol_Latn": "flexitok/bpe_ltr_pol_Latn_16000_v2",
      "por_Latn": "flexitok/bpe_ltr_por_Latn_16000_v2",
      "rus_Cyrl": "flexitok/bpe_ltr_rus_Cyrl_16000_v2",
      "spa_Latn": "flexitok/bpe_ltr_spa_Latn_16000_v2",
      "swe_Latn": "flexitok/bpe_ltr_swe_Latn_16000_v2",
      "tur_Latn": "flexitok/bpe_ltr_tur_Latn_16000_v2",
      "vie_Latn": "flexitok/bpe_ltr_vie_Latn_16000_v2"
    }
}
with open("'"$OUTPUT_DIR"'/config.yaml", "w") as f:
    f.write("""'"$comments"'""" + "\n")
    yaml.dump(config, f, sort_keys=False)
'


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
