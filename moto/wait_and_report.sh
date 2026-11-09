#!/bin/bash
# Waits for submitted SLURM jobs (or local background processes) to finish,
# then prints a summary table with final states.
#
# Usage:
#   wait_and_report.sh <launcher> <jobs_file> [failed_submissions_file]
#
# Arguments:
#   launcher               "sbatch" or "bash"
#   jobs_file              Path to a file where each line is: <job_id> <run_name>
#   failed_submissions_file  (optional) Path to a file listing failed submission names,
#                          one per line. Used only for the summary header count.
#
# The script exits 0 if all jobs completed successfully, 1 otherwise.

set -euo pipefail

launcher="${1:?Usage: $0 <launcher> <jobs_file> [failed_submissions_file]}"
jobs_file="${2:?Usage: $0 <launcher> <jobs_file> [failed_submissions_file]}"
failed_submissions_file="${3:-}"

if [[ ! -f "$jobs_file" ]]; then
	echo "ERROR: jobs_file '$jobs_file' not found." >&2
	exit 1
fi

# Load jobs into arrays
declare -A job_map
submitted_jobs=()
while IFS=' ' read -r job_id run_name; do
	[[ -z "$job_id" ]] && continue
	submitted_jobs+=("$job_id")
	job_map["$job_id"]="$run_name"
done < "$jobs_file"

# Count failed submissions if file provided
n_failed_submissions=0
failed_submission_names=()
if [[ -n "$failed_submissions_file" && -f "$failed_submissions_file" ]]; then
	while IFS= read -r name; do
		[[ -z "$name" ]] && continue
		failed_submission_names+=("$name")
		n_failed_submissions=$((n_failed_submissions + 1))
	done < "$failed_submissions_file"
fi

echo ""
echo "=== Submission Summary ==="
echo "Submitted ${#submitted_jobs[@]} jobs, $n_failed_submissions submission failures."
if [[ $n_failed_submissions -gt 0 ]]; then
	echo "Failed submissions:"
	for name in "${failed_submission_names[@]}"; do echo "  - $name"; done
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
		echo "Done."
		exit 1
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
		echo "Done."
		exit 1
	fi
fi
echo "Done."
