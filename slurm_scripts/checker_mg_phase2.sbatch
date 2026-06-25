#!/bin/bash
#SBATCH --job-name=checker_mg_p2
#SBATCH --partition=rtx6000
#SBATCH --time=2-00:00:00
#SBATCH --output=checker_mg_phase2-%A_%a.out
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --tasks=1
#SBATCH --array=4-6                   # 3 jobs: eps in {0.01, 0.1, 0.5}

# !! BEFORE RUNNING !!
# 1. Inspect Phase 1 WandB results (project: self-distill-flow-maps)
# 2. Find which lambda gave the lowest KL at N=1 and N=2
# 3. Open py/configs/checker_mg.py and set BEST_LAMBDA to that value
# 4. Then submit: sbatch checker_mg_phase2.sbatch
#
# slurm_id -> experiment (lambda = BEST_LAMBDA from checker_mg.py):
#   4  eps=0.01  (tight, more accurate, slower)
#   5  eps=0.10
#   6  eps=0.50  (loose, faster, less accurate)
# (eps=0.05 at BEST_LAMBDA was already run in Phase 1 -- no need to re-run)

echo "=============================="
echo "Checker Monge Gap -- Phase 2 (eps sweep)"
echo "Job:   ${SLURM_JOB_ID}"
echo "Array: ${SLURM_ARRAY_TASK_ID}"
echo "Node:  ${SLURM_JOB_NODELIST}"
echo "=============================="

# ── paths ──────────────────────────────────────────────────────────────────
REPO_DIR=/home/ronimaor/monge-gap-regularizer-to-flow-map
PY_DIR=${REPO_DIR}/py
OUTPUT_DIR=/home/ronimaor/experiments/checker_mg/phase2
# ───────────────────────────────────────────────────────────────────────────

# ── resume (same logic as phase1) ──────────────────────────────────────────
LOAD_PATH=""
# ───────────────────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"

module load anaconda
source activate flow_maps

export WANDB_ENTITY=ronimaor
export PYTHONNOUSERSITE=1

SLURM_ID=${SLURM_ARRAY_TASK_ID}
cd "${PY_DIR}" || exit 1

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
