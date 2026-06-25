#!/bin/bash
#SBATCH --partition rtx6000
#SBATCH --time 2-00:00:00
#SBATCH --job-name checker_mg_p1
#SBATCH --output checker_mg_phase1-%A_%a.out
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --tasks=1
#SBATCH --array=0-3

# slurm_id -> experiment (sinkhorn_eps=0.05 for all):
#   0  baseline  lambda=0.0  (no Monge gap)
#   1            lambda=0.1
#   2            lambda=1.0
#   3            lambda=10.0

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID

REPO_DIR=/home/ronimaor/monge-gap-regularizer-to-flow-map
OUTPUT_DIR=/home/ronimaor/experiments/checker_mg/phase1
mkdir -p "${OUTPUT_DIR}"

# To resume a killed job, set LOAD_PATH to the latest .pkl in OUTPUT_DIR:
#   e.g. LOAD_PATH="${OUTPUT_DIR}/checker_mg_lam1p0_eps0p05_15.pkl"
LOAD_PATH=""

module load anaconda
source activate flow_maps
export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

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
