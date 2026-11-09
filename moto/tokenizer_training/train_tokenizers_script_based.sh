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
# bash moto/tokenizer_training/train_tokenizers_script_based.sh bash
# bash moto/tokenizer_training/train_tokenizers_script_based.sh sbatch
# sbatch $HOME/moto/moto/tokenizer_training/train_tokenizers_script_based.sh 
launcher=${1:-sbatch}

MEM="256G"
MAX_JOBS=${SLURM_NTASKS:-5}
echo "Using up to $MAX_JOBS parallel jobs for tokenizer training."
module load slurm/killarney/24.05.7 StdEnv/2023 gcc/13.3 openmpi/5.0.3 python/3.10.13

cd $HOME/tokenizers2
source .venv/bin/activate

OUT_DIR="$SCRATCH/tokenizer_training_2"
mkdir -p "$OUT_DIR/logs"

# Groups: each entry is "group_name|vs|primary_lang|lang1,lang2,...|hf_repo_id"
# primary_lang is used for the pretokenizer pattern (all langs in a group share the same script)
groups=(
	"Arab|16000|arb_Arab|arb_Arab,fas_Arab|flexitok/bpe_script_Arab_16000"
	"CmJp|16000|jpn_Jpan|cmn_Hani,jpn_Jpan|flexitok/bpe_script_CmJp_16000"
	"Germ|16000|dan_Latn|dan_Latn,deu_Latn,nld_Latn,swe_Latn|flexitok/bpe_script_Germ_16000"
	"Germ|32000|dan_Latn|dan_Latn,deu_Latn,nld_Latn,swe_Latn|flexitok/bpe_script_Germ_32000"
	"Roma|16000|fra_Latn|fra_Latn,ita_Latn,por_Latn,spa_Latn|flexitok/bpe_script_Roma_16000"
	"Roma|32000|fra_Latn|fra_Latn,ita_Latn,por_Latn,spa_Latn|flexitok/bpe_script_Roma_32000"
	"SEAS|16000|ind_Latn|ind_Latn,vie_Latn|flexitok/bpe_script_SEAS_16000"
	"Slav|16000|ces_Latn|ces_Latn,pol_Latn|flexitok/bpe_script_Slav_16000"
)

job_cnt=0
running=0

declare -A job_map   # job_id (sbatch) or pid (local) -> run_name
submitted_jobs=()
failed_submissions=()

# Waits until the number of background jobs drops below MAX_JOBS
wait_for_slot() {
	while (($(jobs -rp | wc -l) >= MAX_JOBS)); do
		wait -n 2>/dev/null || sleep 1
	done
}

cd $PROJECT
for algo in bpe; do
	for group_entry in "${groups[@]}"; do
		IFS='|' read -r group_name vs primary_lang langs_csv hf_repo_id <<< "$group_entry"

		normalization="nfc"
		add_cjk_chars=false
		cjk_char_coverage=0
		cjk_char_split=false
		use_byte_level_regex=false

		if [[ "$group_name" == "CmJp" ]]; then
			max_han_run=16
			add_cjk_chars=true
			cjk_char_coverage=0.999
			max_lines=1_000_000
		else
			max_lines=-1
			max_han_run=-1
		fi

		if [[ "$group_name" == "Arab" ]]; then
			normalization="null"
			echo "Using null normalization for Arab group."
		fi

		job_cnt=$((job_cnt + 1))
		run_name="$algo-$vs-$group_name"
		echo "$job_cnt Launching $run_name (langs: $langs_csv)"

		cmd="cd $HOME/tokenizers2 && python -m moto.tokenizer_training.train_tokenizers \
			algorithm=$algo vocab_size=$vs \
			langs=[$langs_csv] \
			data_dir=$SCRATCH/data/flexitok/ output_dir=$SCRATCH/trained_tokenizers \
			pretokenizer=custom:${primary_lang}  \
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
			hf.hf_repo_id=$hf_repo_id \
			hf.collections=[flexitok/script-based-tokenizers]"
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
