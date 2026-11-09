#!/usr/bin/env sh

# bash moto/supertokenizer/create_supervocab_for_safe.sh
code_name="safe"
OUTPUT_DIR=$PROJECT/tokenizers/supertokenizer_$code_name/individual_tokenizers

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

python moto/supertokenizer/reindex_tokenizers.py  config=$OUTPUT_DIR/reindex.yaml
## Create configs


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
}
with open("'"$OUTPUT_DIR"'/config.yaml", "w") as f:
    f.write("""'"$comments"'""" + "\n")
    yaml.dump(config, f, sort_keys=False)
'

# because we reindexed already
sed -i 's/load_supermapping: true/load_supermapping: false/g' $OUTPUT_DIR/config.yaml
# sed -i 's/flexitok\/bpe_ltr/flexitok\/reindexed-flexitok-bpe_ltr/g' $OUTPUT_DIR/config.yaml
sed -i 's/flexitok\/bpe_/flexitok\/reindexed-flexitok-bpe_/g' $OUTPUT_DIR/config.yaml
cp $OUTPUT_DIR/config.yaml "/path/to/lingua/apps/main/configs/ablations/${code_name}_oracle.yaml"

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
