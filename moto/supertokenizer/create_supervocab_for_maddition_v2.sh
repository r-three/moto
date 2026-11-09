#!/usr/bin/env sh

## Since the digit tokenizer within the 360 regime was stuck at ~60% accuracy, recreating the supervocab with a new digit tokenizer trained with "individual" number handling strategy instead of "ltr_3digit".

# bash moto/supertokenizer/create_supervocab_for_maddition_v2.sh
vs=2000
vs=360
code_name="maddition_${vs}_v2"
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_$code_name/individual_tokenizers

digit_tokenizer="flexitok/maddition_digit_${vs}"
digit_tokenizer="flexitok/maddition_digit_v2_${vs}"

python moto/supertokenizer/supervocab.py \
	tokenizers=[flexitok/maddition_jpn_Jpan_${vs},flexitok/maddition_cmn_Hani_${vs},flexitok/maddition_ita_Latn_${vs},flexitok/maddition_por_Latn_${vs},flexitok/maddition_swe_Latn_${vs},flexitok/maddition_deu_Latn_${vs},flexitok/maddition_ind_Latn_${vs},flexitok/maddition_fas_Arab_${vs},flexitok/maddition_vie_Latn_${vs},flexitok/maddition_nld_Latn_${vs},flexitok/maddition_fra_Latn_${vs},flexitok/maddition_spa_Latn_${vs},flexitok/maddition_dan_Latn_${vs},flexitok/maddition_eng_Latn_${vs},flexitok/maddition_arb_Arab_${vs},flexitok/maddition_ces_Latn_${vs},flexitok/maddition_hun_Latn_${vs},flexitok/maddition_tur_Latn_${vs},flexitok/maddition_pol_Latn_${vs},flexitok/maddition_ell_Grek_${vs},flexitok/maddition_rus_Cyrl_${vs},$digit_tokenizer,] \
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
# python -m lingua.stool script=apps.main.train_answer_only config=["/apps/main/configs/flexitok/multilingual-addition/addition_base.yaml","apps/main/configs/ablations/${code_name}_oracle.yaml"]
# python -m lingua.stool_ccdb ngpu=1 ncpu=1 nodes=1 launcher=bash script=apps.main.train_answer_only config=["apps/main/configs/flexitok/multilingual-addition/addition_base.yaml","apps/main/configs/ablations/${code_name}_oracle.yaml"]
"
python -c '
import yaml
vs = '"${vs}"'
digit_tokenizer = '"${digit_tokenizer}"'
config = yaml.safe_load(open("'"$OUTPUT_DIR"'/config.yaml"))
config["data"].update({
    "suitable_tokenizer_probability": 1.0,
    "suitable_tokenizer_map": {
		"arb_Arab":	f"flexitok/maddition_arb_Arab_{vs}",
		"fas_Arab":	f"flexitok/maddition_fas_Arab_{vs}",
		"cmn_Hani":	f"flexitok/maddition_cmn_Hani_{vs}",
		"jpn_Jpan":	f"flexitok/maddition_jpn_Jpan_{vs}",
		"ell_Grek":	f"flexitok/maddition_ell_Grek_{vs}",
		"eng_Latn":	f"flexitok/maddition_eng_Latn_{vs}",
		"hun_Latn":	f"flexitok/maddition_hun_Latn_{vs}",
		"rus_Cyrl":	f"flexitok/maddition_rus_Cyrl_{vs}",
		"tur_Latn":	f"flexitok/maddition_tur_Latn_{vs}",
		"dan_Latn":	f"flexitok/maddition_dan_Latn_{vs}",
		"deu_Latn":	f"flexitok/maddition_deu_Latn_{vs}",
		"nld_Latn":	f"flexitok/maddition_nld_Latn_{vs}",
		"swe_Latn":	f"flexitok/maddition_swe_Latn_{vs}",
		"fra_Latn":	f"flexitok/maddition_fra_Latn_{vs}",
		"ita_Latn":	f"flexitok/maddition_ita_Latn_{vs}",
		"por_Latn":	f"flexitok/maddition_por_Latn_{vs}",
		"spa_Latn":	f"flexitok/maddition_spa_Latn_{vs}",
		"ind_Latn":	f"flexitok/maddition_ind_Latn_{vs}",
		"vie_Latn":	f"flexitok/maddition_vie_Latn_{vs}",
		"ces_Latn":	f"flexitok/maddition_ces_Latn_{vs}",
		"pol_Latn":	f"flexitok/maddition_pol_Latn_{vs}",
		"digit":	digit_tokenizer,
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


