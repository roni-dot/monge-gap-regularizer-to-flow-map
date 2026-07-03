#!/bin/bash
#SBATCH --partition rtx6000
#SBATCH --time 2-00:00:00
#SBATCH --job-name checker_mg_seedchk
#SBATCH --output checker_mg_seedchk-%A_%a.out
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --tasks=1
#SBATCH --array=7-8

# slurm_id -> experiment (see py/configs/checker_mg.py):
#   7  lambda=BEST_LAMBDA  eps=0.01  seed=7   (seed-repeat of slurm_id 4)
#   8  lambda=BEST_LAMBDA  eps=0.01  seed=13  (seed-repeat of slurm_id 4)
#
# Purpose: check whether slurm_id 4's KL improvement over baseline survives
# seed variance. Everything (lambda, eps, steps, network, interpolant) is
# identical to slurm_id 4 except cfg.training.seed.
#
# Written to a separate seed_check/ folder (not phase1/ or phase2/), and
# output_name carries a _seed{N} suffix, so checkpoints cannot collide with
# anything already in phase1/ or phase2/ even if folders were shared:
#   seed_check/checker_mg_lam0p1_eps0p01_seed7_{0..25}.pkl
#   seed_check/checker_mg_lam0p1_eps0p01_seed13_{0..25}.pkl
# (25 = final checkpoint at total_steps=250_000 / save_freq=10_000)

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID

REPO_DIR=/home/ronimaor/monge-gap-regularizer-to-flow-map
OUTPUT_DIR=${REPO_DIR}/experiments/checker_mg/seed_check
mkdir -p "${OUTPUT_DIR}"

# To resume a killed job, set LOAD_PATH to the latest .pkl in OUTPUT_DIR:
#   e.g. LOAD_PATH="${OUTPUT_DIR}/checker_mg_lam0p1_eps0p01_seed7_15.pkl"
LOAD_PATH=""

module load anaconda
source activate flow_maps
export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_MODE=disabled

cd "${REPO_DIR}/py" || exit 1

RESUME_ARGS=""
if [ -n "${LOAD_PATH}" ]; then
    RESUME_ARGS="--load_path ${LOAD_PATH} --no_reset_optimizer"
fi

python launchers/learn.py \
    --cfg_path      configs.checker_mg \
    --slurm_id      ${SLURM_ARRAY_TASK_ID} \
    --output_folder "${OUTPUT_DIR}" \
    ${RESUME_ARGS}
