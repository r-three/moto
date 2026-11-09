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
# bash moto/simplified/create_multilingual_addition_tokenizers_held_out.sh
# sbatch $HOME/moto/moto/simplified/create_multilingual_addition_tokenizers_held_out.sh 
# sbatch $HOME/moto/moto/simplified/create_multilingual_addition_tokenizers.sh 
launcher=${1:-sbatch}

MEM="64G"
MAX_JOBS=${SLURM_NTASKS:-5}
echo "Using up to $MAX_JOBS parallel jobs for tokenizer training."
module load slurm/killarney/24.05.7 StdEnv/2023 gcc/13.3 openmpi/5.0.3 python/3.10.13

cd $HOME/tokenizers2
source .venv/bin/activate

OUT_DIR="$SCRATCH/tokenizer_training_2"
mkdir -p "$OUT_DIR/logs"

pretokenizer=custom:addition
pretokenizer=custom:addition_split_on_hyphen
langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn eng_Latn hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn digit AllL)
langs=(kor_Hang ben_Beng fin_Latn nob_Latn)

job_cnt=0
running=0

declare -A job_map   # job_id (sbatch) or pid (local) -> run_name
submitted_jobs=()
failed_submissions=()

cjk_char_split=false

# Waits until the number of background jobs drops below MAX_JOBS
wait_for_slot() {
	while (($(jobs -rp | wc -l) >= MAX_JOBS)); do
		wait -n 2>/dev/null || sleep 1
	done
}

cd $PROJECT
# "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million"
for vs in 360; do
# for vs in 2000; do
	for algo in bpe; do
		# for algo in bpe unigram; do
		for lang in ${langs[@]}; do
			normalization="nfc"
			add_cjk_chars=false
			use_byte_level_regex=false
            cjk_char_split=false
			max_lines=-1
			max_han_run=-1
			use_byte_level_regex=false
			extra_special_tokens="+,="
            number_handling="ltr_3digit"
			hf_repo_id="flexitok/maddition_${lang}_${vs}"
            if [[ "$lang" == "cmn_Hani" || "$lang" == "jpn_Jpan"  || "$lang" == "AllL" ]]; then
                cjk_char_split=true
            fi
            if [[ "$vs" == 360 && "$lang" == "digit" ]]; then
                number_handling="individual"
                hf_repo_id="flexitok/maddition_${lang}_${vs}"
            fi

			job_cnt=$((job_cnt + 1))
			run_name="$algo-$vs-$lang"
			echo "$job_cnt Launching $run_name"

			cmd="cd $HOME/tokenizers2 && python -m moto.tokenizer_training.train_tokenizers \
				algorithm=$algo vocab_size=$vs \
				langs=[$lang] \
				data_dir=$SCRATCH/data/multilingual-addition/ output_dir=$SCRATCH/trained_tokenizers/multilingual_addition \
				pretokenizer=$pretokenizer  \
				number_handling=$number_handling \
				add_numbers=false \
				handle_contractions=false \
				unicode_normalization=$normalization \
				use_byte_level_regex=$use_byte_level_regex \
				byte_fallback=false \
				strip_zero_width=false \
				cjk_char_split=$cjk_char_split \
				add_cjk_chars=false \
				max_lines=$max_lines \
				test_string=\"22+9=31\nyirmi iki+dokuz=otuz bir\ntwenty two+nine=thirty one\" \
				hf.publish_to_hf=true hf_repo_prefix=flexitok/ \
				hf.hf_repo_id=$hf_repo_id \
				hf.collections=[flexitok/multilingual_addition_tokenizers_held_out]"
			# echo $cmd
			if [[ "$launcher" == "sbatch" ]]; then
				sbatch_out=$(sbatch -o "/path/to/.slurm/%j.out" -e "/path/to/.slurm/%j.out" \
					--ntasks=1 --cpus-per-task=4 --time 24:00:00 \
					--mem=$MEM --job-name=$run_name \
					--wrap="$cmd" 2>&1)
				job_id=$(echo "$sbatch_out" | grep -oP '(?<=Submitted batch job )\d+')
				if [[ -z "$job_id" ]]; then
					echo "  ERROR submitting $run_name: $sbatch_out"
					failed_submissions+=("$run_name")
				else
					echo "  -> SLURM job $job_id"
					job_map[$job_id]="$run_name"
					submitted_jobs+=("$job_id")
				fi
			else
				echo "Running locally: $cmd"
				eval "$cmd" &
				pid=$!
				job_map[$pid]="$run_name"
				submitted_jobs+=("$pid")
			fi
		done
	done
done



jobs_file=$(mktemp)
failed_file=$(mktemp)
trap 'rm -f "$jobs_file" "$failed_file"' EXIT

for job_id in "${submitted_jobs[@]}"; do
	echo "$job_id ${job_map[$job_id]}" >> "$jobs_file"
done
for name in "${failed_submissions[@]}"; do
	echo "$name" >> "$failed_file"
done

bash "$HOME/moto/moto/wait_and_report.sh" "$launcher" "$jobs_file" "$failed_file"
