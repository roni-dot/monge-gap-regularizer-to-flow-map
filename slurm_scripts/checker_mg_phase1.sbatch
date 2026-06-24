#!/bin/bash
#SBATCH --job-name=checker_mg_p1
#SBATCH --partition=rtx6000
#SBATCH --time=2-00:00:00             # 2 days; lower if cluster max is less
#SBATCH --output=checker_mg_phase1-%A_%a.out
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --tasks=1
#SBATCH --array=0-3                   # 4 jobs: lambda in {0.0, 0.1, 1.0, 10.0}

# slurm_id -> experiment (sinkhorn_eps=0.05 for all):
#   0  baseline  lambda=0.0  (no Monge gap)
#   1            lambda=0.1
#   2            lambda=1.0
#   3            lambda=10.0

echo "=============================="
echo "Checker Monge Gap -- Phase 1 (lambda sweep)"
echo "Job:   ${SLURM_JOB_ID}"
echo "Array: ${SLURM_ARRAY_TASK_ID}"
echo "Node:  ${SLURM_JOB_NODELIST}"
echo "=============================="

# ── paths -- update to match your cluster ──────────────────────────────────
REPO_DIR=/home/ronimaor/flow-maps
PY_DIR=${REPO_DIR}/py
OUTPUT_DIR=/home/ronimaor/experiments/checker_mg/phase1
# ───────────────────────────────────────────────────────────────────────────

# ── resume -- set LOAD_PATH if continuing a killed job ─────────────────────
# Checkpoints are saved every 10k steps as:
#   <OUTPUT_DIR>/<wandb_name>_<N>.pkl   (N = step / 10000)
# Example: checker_mg_lam1p0_eps0p05_15.pkl  = step 150k
#
# To resume a killed job:
#   1. Find the latest .pkl in OUTPUT_DIR
#   2. Set LOAD_PATH to that file
#   3. Re-submit: sbatch checker_mg_phase1.sbatch
LOAD_PATH=""   # e.g. "${OUTPUT_DIR}/checker_mg_lam1p0_eps0p05_15.pkl"
# ───────────────────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flow_maps              # <-- update env name if different

export WANDB_ENTITY=ronimaor
export PYTHONNOUSERSITE=1

SLURM_ID=${SLURM_ARRAY_TASK_ID}
cd "${PY_DIR}" || exit 1

# Build resume args if a checkpoint is specified
RESUME_ARGS=""
if [ -n "${LOAD_PATH}" ]; then
    echo "Resuming from: ${LOAD_PATH}"
    RESUME_ARGS="--load_path ${LOAD_PATH} --no_reset_optimizer"
fi

echo "Starting (slurm_id=${SLURM_ID}) ..."
python launchers/learn.py \
    --cfg_path      configs.checker_mg \
    --slurm_id      ${SLURM_ID} \
    --output_folder "${OUTPUT_DIR}" \
    ${RESUME_ARGS}

echo "Done."
