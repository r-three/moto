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
# bash moto/tokenizer_training/train_tokenizers_feat2.sh bash
# bash moto/tokenizer_training/train_tokenizers_feat2.sh sbatch
# sbatch $HOME/moto/moto/tokenizer_training/train_tokenizers_feat2.sh 
launcher=${1:-sbatch}

MEM="256G"
MAX_JOBS=${SLURM_NTASKS:-5}
echo "Using up to $MAX_JOBS parallel jobs for tokenizer training."
module load slurm/killarney/24.05.7 StdEnv/2023 gcc/13.3 openmpi/5.0.3 python/3.10.13

cd $HOME/tokenizers2
source .venv/bin/activate

OUT_DIR="$SCRATCH/tokenizer_training_2"
mkdir -p "$OUT_DIR/logs"

langs=(arb_Arab ces_La	tn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)
# langs=(cmn_Hani jpn_Jpan)
# langs=(cmn_Hani jpn_Jpan fw_edu tur_Latn vie_Latn rus_Cyrl)
# langs=(deu_Latn)
# langs=(arb_Arab ces_Latn  dan_Latn  ell_Grek fas_Arab fra_Latn  hun_Latn ind_Latn ita_Latn  nld_Latn pol_Latn por_Latn  spa_Latn swe_Latn )
# langs=(pol_Latn nld_Latn ces_Latn hun_Latn tur_Latn ind_Latn dan_Latn swe_Latn vie_Latn ell_Grek)
# langs=(tur_Latn)
# langs=(arb_Arab ces_Latn cmn_Hani dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn jpn_Jpan nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn  vie_Latn)

# non chinese/japanese
# langs=(arb_Arab ces_Latn dan_Latn deu_Latn ell_Grek fas_Arab fra_Latn fw_edu hun_Latn ind_Latn ita_Latn nld_Latn pol_Latn por_Latn rus_Cyrl spa_Latn swe_Latn tur_Latn vie_Latn)

langs=(cmn_Hani jpn_Jpan)
# langs=(cmn_Hani)

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
for vs in 1000 4000 8000 16000 32000; do
# for vs in 1000; do
# for vs in 4000; do
# for vs in 8000; do
# for vs in 4000; do
# for vs in 16000; do
	# for vs in 16000 32000 64000 128000; do
	for algo in bpe; do
		# for algo in bpe unigram; do
		for lang in ${langs[@]}; do
			normalization="nfc"
			add_cjk_chars=false
			cjk_char_coverage=0
			if [[ "$lang" == "cmn_Hani" || "$lang" == "jpn_Jpan" ]]; then
				cjk_char_split=false
				max_lines=5_000_000
				max_han_run=16
				# echo "Using script_aware pretokenizer for CJK (Han runs grouped, capped at ${max_han_run} chars)."
				use_byte_level_regex=false
				cjk_char_split=false
				max_lines=-1
				max_han_run=-1
				max_han_run=16
				add_cjk_chars=true
				cjk_char_coverage=0.99
				# each shard has around 6.7M lines, that's too much, so we will cap it at 5K lines per shard
				max_lines=1_000_000
				if [[ $vs == 1000 ]]; then
					cjk_char_coverage=0.90
				elif [[ $vs -le 4000 ]]; then
					cjk_char_coverage=0.99
				else
					cjk_char_coverage=0.999
				fi
			else
				cjk_char_split=false
				max_lines=-1
				max_han_run=-1
				use_byte_level_regex=false
				echo "CJK isolation disabled."
			fi
			if [[ $lang == "fas_Arab" || $lang == "arb_Arab" ]]; then
				normalization="null"
				echo "Using null normalization for Arabic languages."
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
				pretokenizer=custom:${lang}  \
				number_handling=ltr_3digit \
				add_numbers=true \
				handle_contractions=true \
				unicode_normalization=$normalization \
				use_byte_level_regex=$use_byte_level_regex \
				byte_fallback=false \
				strip_zero_width=false \
				cjk_char_split=$cjk_char_split \
				cjk_char_coverage=$cjk_char_coverage \
				add_cjk_chars=$add_cjk_chars \
				max_han_run=$max_han_run \
				max_lines=$max_lines \
				hf.publish_to_hf=true hf_repo_prefix=flexitok/ \
				hf.hf_repo_id=flexitok/bpe_ltr_${lang}_${vs}_v2 \
				hf.collections=[flexitok/fineweb2-hq-tokenizers-v2,flexitok/$vs-vocab-v2]"
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

echo ""
echo "=== Submission Summary ==="
echo "Submitted ${#submitted_jobs[@]} jobs, ${#failed_submissions[@]} submission failures."
if [[ ${#failed_submissions[@]} -gt 0 ]]; then
	echo "Failed submissions:"
	for name in "${failed_submissions[@]}"; do echo "  - $name"; done
fi

if [[ "$launcher" == "sbatch" ]]; then
	echo "Waiting for SLURM jobs to finish..."
	while true; do
		still_running=$(squeue -j "$(IFS=,; echo "${submitted_jobs[*]}")" -h 2>/dev/null | wc -l)
		[[ "$still_running" -eq 0 ]] && break
		echo "  $still_running job(s) still running, checking again in 120s..."
		sleep 120
	done

	sleep 30  # allow sacct to catch up for jobs that just finished

	echo ""
	printf "%-12s %-40s %-12s %s\n" "JOBID" "RUN_NAME" "STATE" "EXIT_CODE"
	printf "%-12s %-40s %-12s %s\n" "-----" "--------" "-----" "---------"
	failed_jobs=()
	for job_id in "${submitted_jobs[@]}"; do
		# Retry up to 5 times (75s total) if sacct hasn't caught up yet
		for attempt in 1 2 3 4 5; do
			read -r state exit_code < <(sacct -j "$job_id" --format=State,ExitCode --noheader -P 2>/dev/null | head -1 | tr '|' ' ')
			state="${state:-UNKNOWN}"
			[[ "$state" != "UNKNOWN" ]] && break
			sleep 15
		done
		exit_code="${exit_code:-?}"
		printf "%-12s %-40s %-12s %s\n" "$job_id" "${job_map[$job_id]}" "$state" "$exit_code"
		[[ "$state" != "COMPLETED" ]] && failed_jobs+=("$job_id: ${job_map[$job_id]} [$state]")
	done
	echo ""
	echo "Completed: $((${#submitted_jobs[@]} - ${#failed_jobs[@]})) / ${#submitted_jobs[@]}"
	if [[ ${#failed_jobs[@]} -gt 0 ]]; then
		echo "Failed jobs:"
		for entry in "${failed_jobs[@]}"; do echo "  - $entry"; done
	fi
else
	# Local: wait for each PID and collect exit codes
	failed_jobs=()
	for pid in "${submitted_jobs[@]}"; do
		wait "$pid"
		rc=$?
		status="SUCCESS"
		[[ $rc -ne 0 ]] && status="FAILED(rc=$rc)" && failed_jobs+=("${job_map[$pid]}")
		printf "%-8s %-40s %s\n" "$pid" "${job_map[$pid]}" "$status"
	done
	echo ""
	echo "Completed: $((${#submitted_jobs[@]} - ${#failed_jobs[@]})) / ${#submitted_jobs[@]}"
	if [[ ${#failed_jobs[@]} -gt 0 ]]; then
		echo "Failed jobs:"
		for name in "${failed_jobs[@]}"; do echo "  - $name"; done
	fi
fi
echo "Done."
