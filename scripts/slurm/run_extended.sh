#!/bin/bash
#SBATCH --cpus-per-task=8            # = Gurobi threads=8 in the solver
#SBATCH --signal=TERM@300            # SIGTERM 5 min before the wall-time limit
# Everything else (--account, --time, --mem, --job-name, --output) is set by submit_all.sh.
#
# Direct use:  sbatch --account=<proj> --time=02:30:00 --mem=32G run_extended.sh <instance> <gen_s> <mip_s>
# NOTE: <gen_s> comes first here; the script maps it to the solver's own argument order.

set -euo pipefail

INSTANCE="$1"                         # path relative to $ROOT (or absolute)
GEN="$2"                              # route-generation time limit [s]
MIP="$3"                              # Gurobi time limit [s]
MODE="${4:-0}"                        # MIP setting mode 0/1/2
NHEUR="${5:-100}"                      # heuristic runs
NOISE="${6:-0.15}"                    # insertion noise

ROOT="${ROOT:-$HOME/1d-pdptw-research}"
SOLVER="${SOLVER:-1d-pdptw_extended_v4.py}"       # file inside $ROOT/src
VENV="${VENV:-$HOME/venvs/pdptw}"
PY_MODULES="${PY_MODULES:-GCCcore Python}"        # TODO: exact names/versions from `module spider Python`

module purge
module load $PY_MODULES               # deliberately unquoted: several modules
source "$VENV/bin/activate"

cd "$ROOT"                            # results_dir()/relative paths resolve from here
export PYTHONUNBUFFERED=1             # live log output

echo "Job $SLURM_JOB_ID on $(hostname) | $(date)"
echo "Instance: $INSTANCE | gen=${GEN}s mip=${MIP}s mode=$MODE heur=$NHEUR noise=$NOISE"
echo "Python  : $(python --version 2>&1) | solver: $SOLVER"

# solver argument order: <instance> <MIP limit> <gen limit> <mode> <heur runs> <noise> [--single-tw]
srun python "src/$SOLVER" "$INSTANCE" "$MIP" "$GEN" "$MODE" "$NHEUR" "$NOISE" --single-tw

echo "Finished: $(date)"
