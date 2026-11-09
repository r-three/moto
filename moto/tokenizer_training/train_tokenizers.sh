#!/bin/bash
#SBATCH --job-name=train_tokenizers
#SBATCH --output=/path/to/.slurm/%j.out
#SBATCH --mem=340G
#SBATCH --time=12:00:00
#SBATCH --account=aip-craffel
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1

###SBATCH --ntasks=1
###SBATCH --cpus-per-task=16
# salloc -A $ACCOUNT --mem 160G --ntasks=16 --cpus-per-task=1 --nodes=1 --time=12:00:00
# bash moto/tokenizer_training/train_tokenizers.sh bash
# bash moto/tokenizer_training/train_tokenizers.sh sbatch
# sbatch $HOME/moto/moto/tokenizer_training/train_tokenizers.sh 
launcher=${1:-sbatch}

MEM="256G"
MAX_JOBS=${SLURM_NTASKS:-5}
echo "Using up to $MAX_JOBS parallel jobs for tokenizer training."
module load slurm/killarney/24.05.7 StdEnv/2023 gcc/13.3 openmpi/5.0.3 python/3.10.13

cd $HOME/tokenizers2
source .venv/bin/activate

OUT_DIR="$SCRATCH/tokenizer_training_2"
mkdir -p "$OUT_DIR/logs"

langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
# langs=(cmn_Hani jpn_Jpan)
# langs=(cmn_Hani jpn_Jpan fw_edu tur_Latn vie_Latn rus_Cyrl)
# langs=(deu_Latn)
# langs=(arb_Arab ces_Latn  dan_Latn  ell_Grek fas_Arab fra_Latn  hun_Latn ind_Latn ita_Latn  nld_Latn pol_Latn por_Latn  spa_Latn swe_Latn )
# langs=(pol_Latn nld_Latn ces_Latn hun_Latn tur_Latn ind_Latn dan_Latn swe_Latn vie_Latn ell_Grek)
# langs=(cmn_Hani)

# non chinese/japanese
langs=(arb_Arab ces_Latn dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)

job_cnt=0
running=0

cjk_char_split=false

# Waits until the number of background jobs drops below MAX_JOBS
wait_for_slot() {
	while (($(jobs -rp | wc -l) >= MAX_JOBS)); do
		wait -n 2>/dev/null || sleep 1
	done
}
cd $PROJECT
# for vs in 8000; do
# for vs in 4000; do
for vs in 16000; do
	# for vs in 16000 32000 64000 128000; do
	for algo in bpe; do
		# for algo in bpe unigram; do
		for lang in ${langs[@]}; do
			if [[ "$lang" == "cmn_Hani" || "$lang" == "jpn_Jpan" ]]; then
				cjk_char_split=false
				max_lines=5000000
				max_han_run=16
				use_byte_level_regex=false
				pretokenizer=script_aware
				echo "Using script_aware pretokenizer for CJK (Han runs grouped, capped at ${max_han_run} chars)."
			else
				cjk_char_split=false
				max_lines=-1
				max_han_run=-1
				pretokenizer=gpt4
				use_byte_level_regex=true
				echo "CJK isolation disabled."
			fi
			job_cnt=$((job_cnt + 1))
			run_name="$algo-$vs-$lang"
			echo "$job_cnt Launching $run_name"

			# wait_for_slot
			# -o "$OUT_DIR/logs/${run_name}_%j_%s.log" \
			# -e "$OUT_DIR/logs/${run_name}_%j_%s.err" \

			# srun  \
			# echo 
			cmd="cd $HOME/tokenizers2 && python -m moto.tokenizer_training.train_tokenizers \
				algorithm=$algo vocab_size=$vs \
				langs=[$lang] \
				data_dir=$SCRATCH/data/flexitok/ output_dir=$SCRATCH/trained_tokenizers \
				pretokenizer=$pretokenizer \
				number_handling=individual \
				handle_contractions=true \
				unicode_normalization=nfc \
				use_byte_level_regex=$use_byte_level_regex \
				strip_zero_width=false \
				cjk_char_split=$cjk_char_split \
				max_han_run=$max_han_run \
				max_lines=$max_lines \
				hf.publish_to_hf=true hf_repo_prefix=flexitok/ \
				hf.collections=[flexitok/fineweb2-hq-tokenizers,flexitok/$vs-vocab]"
			
			if [[ "$launcher" == "sbatch" ]]; then
				sbatch -o "/path/to/.slurm/%j.out" -e "/path/to/.slurm/%j.out" --ntasks=1 --cpus-per-task=4 --time 24:00:00 \
					--mem=$MEM  --job-name=$run_name \
					--wrap="$cmd"  &
			else
				echo "Running locally: $cmd"
				eval "$cmd" &
			fi
		done
	done
done

wait
echo "All tokenizer training jobs completed."
