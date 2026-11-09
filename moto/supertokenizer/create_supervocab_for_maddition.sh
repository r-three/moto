#!/usr/bin/env sh

# bash moto/supertokenizer/create_supervocab_for_maddition.sh
vs=2000
vs=360
code_name="maddition_${vs}"
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_$code_name/individual_tokenizers

python moto/supertokenizer/supervocab.py \
	tokenizers=[flexitok/maddition_jpn_Jpan_${vs},flexitok/maddition_cmn_Hani_${vs},flexitok/maddition_ita_Latn_${vs},flexitok/maddition_por_Latn_${vs},flexitok/maddition_swe_Latn_${vs},flexitok/maddition_deu_Latn_${vs},flexitok/maddition_ind_Latn_${vs},flexitok/maddition_fas_Arab_${vs},flexitok/maddition_vie_Latn_${vs},flexitok/maddition_nld_Latn_${vs},flexitok/maddition_fra_Latn_${vs},flexitok/maddition_spa_Latn_${vs},flexitok/maddition_dan_Latn_${vs},flexitok/maddition_eng_Latn_${vs},flexitok/maddition_arb_Arab_${vs},flexitok/maddition_ces_Latn_${vs},flexitok/maddition_hun_Latn_${vs},flexitok/maddition_tur_Latn_${vs},flexitok/maddition_pol_Latn_${vs},flexitok/maddition_ell_Grek_${vs},flexitok/maddition_rus_Cyrl_${vs},flexitok/maddition_digit_${vs},] \
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

comments=$(grep "^#" $OUTPUT_DIR/config.yaml)
comments="$comments
# python -m lingua.stool script=apps.main.train_answer_only config=["apps/main/configs/flexitok/multilingual-addition/addition_base.yaml","apps/main/configs/ablations/${code_name}_oracle.yaml"]
# python -m lingua.stool_ccdb ngpu=1 ncpu=1 nodes=1 launcher=bash script=apps.main.train_answer_only config=["apps/main/configs/flexitok/multilingual-addition/addition_base.yaml","apps/main/configs/ablations/${code_name}_oracle.yaml"]
"
python -c '
import yaml
config = yaml.safe_load(open("'"$OUTPUT_DIR"'/config.yaml"))
config["data"].update({
    "suitable_tokenizer_probability": 1.0,
    "suitable_tokenizer_map": {
		"arb_Arab":	"flexitok/maddition_arb_Arab_${vs}",
		"fas_Arab":	"flexitok/maddition_fas_Arab_${vs}",
		"cmn_Hani":	"flexitok/maddition_cmn_Hani_${vs}",
		"jpn_Jpan":	"flexitok/maddition_jpn_Jpan_${vs}",
		"ell_Grek":	"flexitok/maddition_ell_Grek_${vs}",
		"eng_Latn":	"flexitok/maddition_eng_Latn_${vs}",
		"hun_Latn":	"flexitok/maddition_hun_Latn_${vs}",
		"rus_Cyrl":	"flexitok/maddition_rus_Cyrl_${vs}",
		"tur_Latn":	"flexitok/maddition_tur_Latn_${vs}",
		"dan_Latn":	"flexitok/maddition_dan_Latn_${vs}",
		"deu_Latn":	"flexitok/maddition_deu_Latn_${vs}",
		"nld_Latn":	"flexitok/maddition_nld_Latn_${vs}",
		"swe_Latn":	"flexitok/maddition_swe_Latn_${vs}",
		"fra_Latn":	"flexitok/maddition_fra_Latn_${vs}",
		"ita_Latn":	"flexitok/maddition_ita_Latn_${vs}",
		"por_Latn":	"flexitok/maddition_por_Latn_${vs}",
		"spa_Latn":	"flexitok/maddition_spa_Latn_${vs}",
		"ind_Latn":	"flexitok/maddition_ind_Latn_${vs}",
		"vie_Latn":	"flexitok/maddition_vie_Latn_${vs}",
		"ces_Latn":	"flexitok/maddition_ces_Latn_${vs}",
		"pol_Latn":	"flexitok/maddition_pol_Latn_${vs}",
		"digit":	"flexitok/maddition_digit_${vs}",
    }
})
with open("'"$OUTPUT_DIR"'/config.yaml", "w") as f:
    f.write("""'"$comments"'""" + "\n")
    yaml.dump(config, f, sort_keys=False)
'
cp "$OUTPUT_DIR/config.yaml" "/path/to/lingua/apps/main/configs/flexitok/multilingual-addition/${code_name}.yaml"



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


