#! /bin/bash
#SBATCH --job-name=intrinsic_metrics
#SBATCH --output=/path/to/.slurm/%j.out
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=aip-craffel
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1


# bash  moto/tokenizer_intrinsic_metrics/comprehensive_parity_for_8k_v3.sh
# sbatch $HOME/moto/moto/tokenizer_intrinsic_metrics/comprehensive_parity_for_8k_v3.sh

# flores has hans not hani
# pes not fas
langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek pes_Arab fra_Latn eng_Latn hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn eng_Latn hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
# langs=(arb_Arab ces_Latn dan_Latn deu_Latn ell_Grek pes_Arab fra_Latn eng_Latn hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
lang_str=$(IFS=,; echo "${langs[*]}") && echo $lang_str

langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
langs_prefixed=("${langs[@]/#/flexitok/bpe_ltr_}")
tokenizers=("${langs_prefixed[@]/%/_8000_v3}")
tokenizer_str=$(IFS=,; echo "${tokenizers[*]}") && echo $tokenizer_str

# python -m moto.tokenizer_intrinsic_metrics.compute_intrinsic_tokenizer_metrics \
# tokenizers=[$tokenizer_str] \
# languages=[$lang_str] \
# analyses=['fertility','parity'] 

cd $HOME/tokenizers2
source .venv/bin/activate

mkdir -p results/rebuttal
# for i in "${!langs[@]}"; do
# 	lang="${langs[$i]}"
# 	tok="${tokenizers[$i]}"
# 	echo "Computing parity for language: ${lang} with ${tok}"
# 	python -m moto.tokenizer_intrinsic_metrics.parity \
# 		tokenizers=[$tok] \
# 		languages=[$lang] \
# 		ref_tokenizer=flexitok/bpe_ltr_fw_edu_8000_v3 \
# 		ref_lang=eng_Latn \
# 		output_csv=./results/parity-against-fw-edu-tokenizer.csv \
# 		output_mode=append 

	
# 	python -m moto.tokenizer_intrinsic_metrics.parity \
# 		tokenizers=[$tok] \
# 		languages=[$lang] \
# 		ref_tokenizer=meta-llama/Llama-3.2-1B \
# 		ref_lang=eng_Latn \
# 		output_csv=./results/parity-against-llama-tokenizer.csv \
# 		output_mode=append &[]
# done
ref="flexitok/parity_aware_bpe_tr_500k_val_1000"
ref="flexitok/bpe-parity-aware-ratio"
ref="flexitok/parity-aware-bpe-devset"

# wait
tokenizer_str="${tokenizer_str},flexitok/parity-aware-bpe-devset,flexitok/parity-aware-bpe-ratio"

python -m moto.tokenizer_intrinsic_metrics.parity \
    tokenizers=[$tokenizer_str,meta-llama/Llama-3.2-1B,$ref,] \
    languages=[$lang_str] \
	ref_tokenizer=meta-llama/Llama-3.2-1B \
	ref_lang=eng_Latn \
	output_csv=./results/rebuttal/parity-against-llama-tokenizer.csv \
	return_dispersion=true

python -m moto.tokenizer_intrinsic_metrics.parity \
    tokenizers=[$tokenizer_str,meta-llama/Llama-3.2-1B,$ref,] \
    languages=[$lang_str] \
	ref_tokenizer=flexitok/bpe_ltr_fw_edu_8000_v3 \
	ref_lang=eng_Latn \
	output_csv=./results/rebuttal/parity-against-fw-edu-tokenizer.csv \
	return_dispersion=true

# python -m moto.tokenizer_intrinsic_metrics.parity \
#     tokenizers=[$tokenizer_str,meta-llama/Llama-3.2-1B,$ref,] \
#     languages=[$lang_str] \
# 	ref_tokenizer=flexitok/bpe_ltr_fw_edu_8000_v3 \
# 	ref_lang=eng_Latn \
# 	output_csv=./results/rebuttal/parity-against-fw-edu-tokenizer.csv \
# 	return_dispersion=true


python -m moto.tokenizer_intrinsic_metrics.parity \
    tokenizers=[$tokenizer_str,meta-llama/Llama-3.2-1B,$ref,] \
    languages=[$lang_str] \
	ref_tokenizer=$ref \
	ref_lang=eng_Latn \
	output_csv=./results/rebuttal/parity-against-parity-aware-tokenizer.csv \
	return_dispersion=true

# python -m moto.tokenizer_intrinsic_metrics.parity \
#     tokenizers=[meta-llama/Llama-3.2-1B] \
#     languages=[$lang_str] \
# 	ref_tokenizer=meta-llama/Llama-3.2-1B \
# 	ref_lang=eng_Latn \
# 	output_csv=./results/parity-llama-against-llama-tokenizer.csv \
# 	return_dispersion=true



