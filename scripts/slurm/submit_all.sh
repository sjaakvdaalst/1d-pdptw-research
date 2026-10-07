#!/bin/bash
# Submits one Slurm job per line of instances.txt, each with its own wall time and memory.
#
#   ACCOUNT=<your_project> ./submit_all.sh            # submit
#   DRYRUN=1 ACCOUNT=x ./submit_all.sh                # only print the sbatch commands
#
# instances.txt format (lines starting with # are ignored):
#   path_relative_to_ROOT | gen_limit_s | mip_limit_s | memory

set -euo pipefail
ROOT="${ROOT:-$HOME/1d-pdptw-research}"
LIST="${LIST:-$ROOT/scripts/slurm/instances.txt}"
RUNNER="$ROOT/scripts/slurm/run_extended.sh"
BUFFER=1200                              # s for parsing, heuristics, JSON export
: "${ACCOUNT:?set ACCOUNT=<your CLAIX project id>}"

mkdir -p "$ROOT/logs"
cd "$ROOT"

fmt_time () {                            # seconds -> [D-]HH:MM:SS
    local s=$1 d h m
    d=$(( s / 86400 )); h=$(( s % 86400 / 3600 )); m=$(( s % 3600 / 60 )); s=$(( s % 60 ))
    if (( d > 0 )); then printf "%d-%02d:%02d:%02d" "$d" "$h" "$m" "$s"
    else printf "%02d:%02d:%02d" "$h" "$m" "$s"; fi
}

tr -d '\r' < "$LIST" | while IFS='|' read -r path gen mip mem; do
    path=$(echo "$path" | xargs); gen=$(echo "${gen:-}" | xargs); mip=$(echo "${mip:-}" | xargs); mem=$(echo "${mem:-}" | xargs)
    [[ -z "$path" || "$path" == \#* ]] && continue
    [[ -f "$path" ]] || { echo "MISSING instance file: $path" >&2; continue; }

    wall=$(fmt_time $(( gen + mip + BUFFER )))
    name=$(basename "$path" .json | tr -c 'A-Za-z0-9_\n-' '_' | cut -c1-60)   # '%' etc. break sbatch name patterns

    cmd=(sbatch --account="$ACCOUNT" --time="$wall" --mem="${mem:-32G}" --job-name="$name"
         --output="$ROOT/logs/%x_%j.out" "$RUNNER" "$path" "$gen" "$mip")
    if [[ "${DRYRUN:-0}" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
