#!/usr/bin/env sh

# bash moto/supertokenizer/create_supervocab_for_safe_v2.sh
code_name="safe_v2"
eval_conf_name="2.6.2-eval-${code_name}"
wandb_name="FW2HQ-Safe-V2"
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_$code_name/individual_tokenizers

CREATE_SS="false"
REINDEX_TOKS="false"

if [[ $CREATE_SS == "true" ]]; then
python moto/supertokenizer/supervocab.py \
	tokenizers=[flexitok/bpe_arb_Arab_8000,flexitok/bpe_ces_Latn_8000,flexitok/bpe_ltr_cmn_Hani_8000_v2,flexitok/bpe_dan_Latn_8000,flexitok/bpe_deu_Latn_8000,flexitok/bpe_ell_Grek_8000,flexitok/bpe_fas_Arab_8000,flexitok/bpe_fra_Latn_8000,flexitok/bpe_fw_edu_8000,flexitok/bpe_hun_Latn_8000,flexitok/bpe_ind_Latn_8000,flexitok/bpe_ita_Latn_8000,flexitok/bpe_ltr_jpn_Jpan_8000_v2,flexitok/bpe_nld_Latn_8000,flexitok/bpe_pol_Latn_8000,flexitok/bpe_por_Latn_8000,flexitok/bpe_rus_Cyrl_8000,flexitok/bpe_spa_Latn_8000,flexitok/bpe_swe_Latn_8000,flexitok/bpe_ltr_tur_Latn_8000_v2,flexitok/bpe_vie_Latn_8000] \
	output_dir=$OUTPUT_DIR \
	return_weights=true \
	hf_upload_repo_prefix=flexitok/supertokenizer-$code_name- \
	hf.collections=["flexitok/$code_name"] \
	hf.publish_to_hf=true \
	hf.hf_repo_id=flexitok/supertokenizer-$code_name\
	upload_to_original_repo=true \
	superset_code_name="$code_name" 
fi

## get vocab size from config.yaml
vocab_size=$(python -c "import yaml; print(yaml.safe_load(open('$OUTPUT_DIR/config.yaml'))['model']['vocab_size'])")


echo """

tokenizers:
  - name: huggingface
    path: flexitok/bpe_arb_Arab_8000
  - name: huggingface
    path: flexitok/bpe_ces_Latn_8000
  - name: huggingface
    path: flexitok/bpe_ltr_cmn_Hani_8000_v2
  - name: huggingface
    path: flexitok/bpe_dan_Latn_8000
  - name: huggingface
    path: flexitok/bpe_deu_Latn_8000
  - name: huggingface
    path: flexitok/bpe_ell_Grek_8000
  - name: huggingface
    path: flexitok/bpe_fas_Arab_8000
  - name: huggingface
    path: flexitok/bpe_fra_Latn_8000
  - name: huggingface
    path: flexitok/bpe_fw_edu_8000
  - name: huggingface
    path: flexitok/bpe_hun_Latn_8000
  - name: huggingface
    path: flexitok/bpe_ind_Latn_8000
  - name: huggingface
    path: flexitok/bpe_ita_Latn_8000
  - name: huggingface
    path: flexitok/bpe_ltr_jpn_Jpan_8000_v2
  - name: huggingface
    path: flexitok/bpe_nld_Latn_8000
  - name: huggingface
    path: flexitok/bpe_pol_Latn_8000
  - name: huggingface
    path: flexitok/bpe_por_Latn_8000
  - name: huggingface
    path: flexitok/bpe_rus_Cyrl_8000
  - name: huggingface
    path: flexitok/bpe_spa_Latn_8000
  - name: huggingface
    path: flexitok/bpe_swe_Latn_8000
  - name: huggingface
    path: flexitok/bpe_ltr_tur_Latn_8000_v2
  - name: huggingface
    path: flexitok/bpe_vie_Latn_8000
input_dir: $OUTPUT_DIR
hf_upload_repo_prefix: flexitok/reindexed-
""" > $OUTPUT_DIR/reindex.yaml

if [[ $REINDEX_TOKS == "true" ]]; then
	python moto/supertokenizer/reindex_tokenizers.py  config=$OUTPUT_DIR/reindex.yaml
fi
## Create configs


comments=$(grep "^#" $OUTPUT_DIR/config.yaml)
comments="$comments
# python -m lingua.stool script=apps.main.train config=["/fsx/craffel/lingua/apps/main/configs/flexitok/llama_1b_base.yaml","apps/main/configs/ablations/${code_name}_oracle.yaml"]
# python -m lingua.stool_ccdb ngpu=1 ncpu=1 nodes=1 launcher=bash script=apps.main.train config=["apps/main/configs/flexitok/llama_1b_base.yaml","apps/main/configs/ablations/${code_name}_oracle.yaml"]
"
eval_comments="# python -m lingua.stool_ccdb ngpu=1 ncpu=1 mem=64G time=800 nodes=1 gpu_type=l40s launcher=sbatch script=apps.main.eval config=["apps/main/configs/flexitok/eval/eval_base.yaml",apps/main/configs/flexitok/eval/${eval_conf_name}_oracle.yaml]"

python -c '
import yaml
config = yaml.safe_load(open("'"$OUTPUT_DIR"'/config.yaml"))
setting = {
	"arb_Arab": "flexitok/bpe_arb_Arab_8000",
	"ces_Latn": "flexitok/bpe_ces_Latn_8000",
	"cmn_Hani": "flexitok/bpe_ltr_cmn_Hani_8000",
	"dan_Latn": "flexitok/bpe_dan_Latn_8000",
	"deu_Latn": "flexitok/bpe_deu_Latn_8000",
	"ell_Grek": "flexitok/bpe_ell_Grek_8000",
	"fas_Arab": "flexitok/bpe_fas_Arab_8000",
	"fra_Latn": "flexitok/bpe_fra_Latn_8000",
	"fw_edu": "flexitok/bpe_fw_edu_8000",
	"hun_Latn": "flexitok/bpe_hun_Latn_8000",
	"ind_Latn": "flexitok/bpe_ind_Latn_8000",
	"ita_Latn": "flexitok/bpe_ita_Latn_8000",
	"jpn_Jpan": "flexitok/bpe_ltr_jpn_Jpan_8000_v2",
	"nld_Latn": "flexitok/bpe_nld_Latn_8000",
	"pol_Latn": "flexitok/bpe_pol_Latn_8000",
	"por_Latn": "flexitok/bpe_por_Latn_8000",
	"rus_Cyrl": "flexitok/bpe_rus_Cyrl_8000",
	"spa_Latn": "flexitok/bpe_spa_Latn_8000",
	"swe_Latn": "flexitok/bpe_swe_Latn_8000",
	"tur_Latn": "flexitok/bpe_ltr_tur_Latn_8000_v2",
	"vie_Latn": "flexitok/bpe_vie_Latn_8000"
}

config["data"]["routing"] = {
    "suitable_tokenizer_probability": 1.0,
    "source_to_tokenizer": setting,
}
eval_routing = {
  "suitable_tokenizer_probability": 1.0,
  "task_to_tokenizer": {
    "arc_easy": setting["fw_edu"],
    "arc_challenge": setting["fw_edu"],
    "hellaswag": setting["fw_edu"],
    "piqa": setting["fw_edu"],

    # --- MMLU / Base 44 ---
    "include_base_44_arabic": setting["arb_Arab"],
    # no czech, swedish, danish
    "include_base_44_chinese": setting["cmn_Hani"],
    "include_base_44_german": setting["deu_Latn"],
    "include_base_44_greek": setting["ell_Grek"],
    "include_base_44_persian": setting["fas_Arab"],
    "include_base_44_french": setting["fra_Latn"],
    "include_base_44_hungarian": setting["hun_Latn"],
    "include_base_44_indonesian": setting["ind_Latn"],
    "include_base_44_italian": setting["ita_Latn"],
    "include_base_44_japanese": setting["jpn_Jpan"],
    "include_base_44_dutch": setting["nld_Latn"],
    "include_base_44_polish": setting["pol_Latn"],
    "include_base_44_portuguese": setting["por_Latn"],
    "include_base_44_russian": setting["rus_Cyrl"],
    "include_base_44_spanish": setting["spa_Latn"],
    "include_base_44_turkish": setting["tur_Latn"],
    "include_base_44_vietnamese": setting["vie_Latn"],

    # --- Belebele ---
    "belebele_arb_Arab": setting["arb_Arab"],
    "belebele_ces_Latn": setting["ces_Latn"],
    "belebele_zho_Hans": setting["cmn_Hani"],
    "belebele_dan_Latn": setting["dan_Latn"],
    "belebele_deu_Latn": setting["deu_Latn"],
    "belebele_ell_Grek": setting["ell_Grek"],
    "belebele_pes_Arab": setting["fas_Arab"],
    "belebele_fra_Latn": setting["fra_Latn"],
    "belebele_hun_Latn": setting["hun_Latn"],
    "belebele_ind_Latn": setting["ind_Latn"],
    "belebele_ita_Latn": setting["ita_Latn"],
    "belebele_jpn_Jpan": setting["jpn_Jpan"],
    "belebele_nld_Latn": setting["nld_Latn"],
    "belebele_pol_Latn": setting["pol_Latn"],
    "belebele_por_Latn": setting["por_Latn"],
    "belebele_rus_Cyrl": setting["rus_Cyrl"],
    "belebele_spa_Latn": setting["spa_Latn"],
    "belebele_swe_Latn": setting["swe_Latn"],
    "belebele_tur_Latn": setting["tur_Latn"],
    "belebele_vie_Latn": setting["vie_Latn"],
    "belebele_eng_Latn": setting["fw_edu"],

    # --- XNLI ---
    "xnli_ar": setting["arb_Arab"],
    "xnli_zh": setting["cmn_Hani"],
    "xnli_de": setting["deu_Latn"],
    "xnli_el": setting["ell_Grek"],
    "xnli_en": setting["fw_edu"],
    "xnli_es": setting["spa_Latn"],
    "xnli_fr": setting["fra_Latn"],
    "xnli_ru": setting["rus_Cyrl"],
    "xnli_tr": setting["tur_Latn"],
    "xnli_vi": setting["vie_Latn"],
    "regex:toksuite_turkish*": setting["tur_Latn"],
    "regex:toksuite_english*": setting["fw_edu"],
    "regex:toksuite_farsi*": setting["fas_Arab"],
    "regex:toksuite_italian*": setting["ita_Latn"],
    "regex:toksuite_chinese*": setting["cmn_Hani"],
    "regex:toksuite_math*": setting["fw_edu"],
    "regex:toksuite_stem*": setting["fw_edu"],

    # flores
    "flores200_arb_Arab": setting["arb_Arab"],
    "flores200_ces_Latn": setting["ces_Latn"],
    "flores200_cmn_Hani": setting["cmn_Hani"],
    "flores200_dan_Latn": setting["dan_Latn"],
    "flores200_deu_Latn": setting["deu_Latn"],
    "flores200_ell_Grek": setting["ell_Grek"],
    "flores200_fas_Arab": setting["fas_Arab"],
    "flores200_fra_Latn": setting["fra_Latn"],
    "flores200_fw_edu": setting["fw_edu"],
    "flores200_eng_Latn": setting["fw_edu"],
    "flores200_hun_Latn": setting["hun_Latn"],
    "flores200_ind_Latn": setting["ind_Latn"],
    "flores200_ita_Latn": setting["ita_Latn"],
    "flores200_jpn_Jpan": setting["jpn_Jpan"],
    "flores200_nld_Latn": setting["nld_Latn"],
    "flores200_pol_Latn": setting["pol_Latn"],
    "flores200_por_Latn": setting["por_Latn"],
    "flores200_rus_Cyrl": setting["rus_Cyrl"],
    "flores200_spa_Latn": setting["spa_Latn"],
    "flores200_swe_Latn": setting["swe_Latn"],
    "flores200_tur_Latn": setting["tur_Latn"],
    "flores200_vie_Latn": setting["vie_Latn"],


    "xcopa_id": setting["ind_Latn"],
    "xcopa_it": setting["ita_Latn"],
    "xcopa_tr": setting["tur_Latn"],
    "xcopa_vi": setting["vie_Latn"],
    "xcopa_zh": setting["cmn_Hani"],
    "xstorycloze_ar": setting["arb_Arab"],
    "xstorycloze_en": setting["fw_edu"],
    "xstorycloze_es": setting["spa_Latn"],
    "xstorycloze_id": setting["ind_Latn"],
    "xstorycloze_ru": setting["rus_Cyrl"],
    "xstorycloze_zh": setting["cmn_Hani"],
    "paws_de": setting["deu_Latn"],
    "paws_en": setting["fw_edu"],
    "paws_es": setting["spa_Latn"],
    "paws_fr": setting["fra_Latn"],
    "paws_ja": setting["jpn_Jpan"],
    "paws_zh": setting["cmn_Hani"],

    "hellaswag_ar": setting["arb_Arab"],
    "hellaswag_da": setting["dan_Latn"],
    "hellaswag_de": setting["deu_Latn"],
    "hellaswag_fr": setting["fra_Latn"],
    "hellaswag_hu": setting["hun_Latn"],
    "hellaswag_id": setting["ind_Latn"],
    "hellaswag_it": setting["ita_Latn"],
    "hellaswag_nl": setting["nld_Latn"],
    "hellaswag_pt": setting["por_Latn"],
    "hellaswag_ru": setting["rus_Cyrl"],
    "hellaswag_sv": setting["swe_Latn"],
    "hellaswag_vi": setting["vie_Latn"],
    "hellaswag_es": setting["spa_Latn"],
}
}
config["eval"] = {"routing": eval_routing}

with open("'"$OUTPUT_DIR"'/config.yaml", "w") as f:
    f.write("""'"$comments"'""" + "\n")
    yaml.dump(config, f, sort_keys=False)

## also create the eval config
dump_dir = config["dump_dir"] + "/eval/"
eval_dct = {"name": config["name"], "dump_dir": dump_dir, "ckpt_dir": f"{dump_dir}/checkpoints/0000100000/"}
eval_dct["wandb"] = {"name": """'"$wandb_name"'"""}
eval_dct["routing"] = eval_routing

with open("'"$OUTPUT_DIR"'/eval_config.yaml", "w") as f:
	f.write("""'"$eval_comments"'""" + "\n")
	yaml.dump(eval_dct, f, sort_keys=False)
'

# because we reindexed already
sed -i 's/load_supermapping: true/load_supermapping: false/g' $OUTPUT_DIR/config.yaml
# sed -i 's/flexitok\/bpe_ltr/flexitok\/reindexed-flexitok-bpe_ltr/g' $OUTPUT_DIR/config.yaml
sed -i 's/flexitok\/bpe_/flexitok\/reindexed-flexitok-bpe_/g' $OUTPUT_DIR/config.yaml
## replace it in eval_conf too
sed -i 's/flexitok\/bpe_/flexitok\/reindexed-flexitok-bpe_/g' $OUTPUT_DIR/eval_config.yaml
cp $OUTPUT_DIR/config.yaml "/path/to/lingua/apps/main/configs/ablations/${code_name}_oracle.yaml"
cp $OUTPUT_DIR/eval_config.yaml "/path/to/lingua/apps/main/configs/flexitok/eval/${eval_conf_name}_oracle.yaml"


echo $OUTPUT_DIR

#TODO auto parse
exit 0
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
