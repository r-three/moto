#! /bin/bash
#SBATCH --job-name=intrinsic_metrics
#SBATCH --output=/path/to/.slurm/%j.out
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=aip-craffel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# bash moto/tokenizer_intrinsic_metrics/compute_instrinsic_metrics_for_fineweb2_hq_toks.sh
# sbatch $HOME/moto/moto/tokenizer_intrinsic_metrics/compute_instrinsic_metrics_for_fineweb2_hq_toks.sh

# flores has hans not hani
# pes not fas
langs=(arb_Arab ces_Latn cmn_Hans dan_Latn deu_Latn ell_Grek pes_Arab fra_Latn eng_Latn hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
# langs=(arb_Arab ces_Latn dan_Latn deu_Latn ell_Grek pes_Arab fra_Latn eng_Latn hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
lang_str=$(IFS=,; echo "${langs[*]}") && echo $lang_str

langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
tokenizer_str=$(IFS=,; langs_prefixed=("${langs[@]/#/flexitok/bpe_}"); langs_suffixed=("${langs_prefixed[@]/%/_8000}"); echo "${langs_suffixed[*]}") && echo $tokenizer_str

# python -m moto.tokenizer_intrinsic_metrics.compute_intrinsic_tokenizer_metrics \
# tokenizers=[$tokenizer_str] \
# languages=[$lang_str] \
# analyses=['fertility','parity'] 

cd $HOME/tokenizers2
source .venv/bin/activate

python -m moto.tokenizer_intrinsic_metrics.compute_intrinsic_tokenizer_metrics \
    tokenizers=[$tokenizer_str] \
    languages=[$lang_str] \
    analyses='["fertility","parity","pcw","vocab_overlap"]' \
    output_dir=./results \
	plot_type=heatmap
