#!/usr/bin/env sh

# bash moto/supertokenizer/create_supervocab_for_script_1.sh
code_name="script_1"
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_$code_name/individual_tokenizers


python moto/supertokenizer/supervocab.py \
	tokenizers=[flexitok/bpe_script_Arab_16000,flexitok/bpe_script_CmJp_16000,flexitok/bpe_ltr_ell_Grek_8000_v2,flexitok/bpe_ltr_fw_edu_32000_v2,flexitok/bpe_ltr_hun_Latn_8000_v2,flexitok/bpe_ltr_rus_Cyrl_16000_v2,flexitok/bpe_ltr_tur_Latn_8000_v2,flexitok/bpe_script_Germ_32000,flexitok/bpe_script_Roma_32000,flexitok/bpe_script_SEAS_16000,flexitok/bpe_script_Slav_16000] \
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
	"arb_Arab":	"flexitok/bpe_script_Arab_16000",
	"fas_Arab":	"flexitok/bpe_script_Arab_16000",
	"cmn_Hani":	"flexitok/bpe_script_CmJp_16000",
	"jpn_Jpan":	"flexitok/bpe_script_CmJp_16000",
	"ell_Grek":	"flexitok/bpe_ltr_ell_Grek_8000_v2",
	"fw_edu":	"flexitok/bpe_ltr_fw_edu_32000_v2",
	"hun_Latn":	"flexitok/bpe_ltr_hun_Latn_8000_v2",
	"rus_Cyrl":	"flexitok/bpe_ltr_rus_Cyrl_16000_v2",
	"tur_Latn":	"flexitok/bpe_ltr_tur_Latn_8000_v2",
	"dan_Latn":	"flexitok/bpe_script_Germ_32000",
	"deu_Latn":	"flexitok/bpe_script_Germ_32000",
	"nld_Latn":	"flexitok/bpe_script_Germ_32000",
	"swe_Latn":	"flexitok/bpe_script_Germ_32000",
	"fra_Latn":	"flexitok/bpe_script_Roma_32000",
	"ita_Latn":	"flexitok/bpe_script_Roma_32000",
	"por_Latn":	"flexitok/bpe_script_Roma_32000",
	"spa_Latn":	"flexitok/bpe_script_Roma_32000",
	"ind_Latn":	"flexitok/bpe_script_SEAS_16000",
	"vie_Latn":	"flexitok/bpe_script_SEAS_16000",
	"ces_Latn":	"flexitok/bpe_script_Slav_16000",
	"pol_Latn":	"flexitok/bpe_script_Slav_16000",
    }
}
with open("'"$OUTPUT_DIR"'/config.yaml", "w") as f:
    f.write("""'"$comments"'""" + "\n")
    yaml.dump(config, f, sort_keys=False)
'
cp "$OUTPUT_DIR/config.yaml" "/path/to/lingua/apps/main/configs/ablations/${code_name}.yaml"

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
