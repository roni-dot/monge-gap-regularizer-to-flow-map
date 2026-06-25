#!/bin/bash
#SBATCH --partition rtx6000
#SBATCH --time 2-00:00:00
#SBATCH --job-name checker_mg_p2
#SBATCH --output checker_mg_phase2-%A_%a.out
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --tasks=1
#SBATCH --array=4-6

# !! Run AFTER Phase 1 is done !!
# 1. Check WandB for best lambda at N=1/N=2 KL
# 2. Set BEST_LAMBDA in py/configs/checker_mg.py
# 3. Then: sbatch slurm_scripts/checker_mg_phase2.sbatch
#
# slurm_id -> experiment (lambda = BEST_LAMBDA from checker_mg.py):
#   4  eps=0.01
#   5  eps=0.10
#   6  eps=0.50

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID

REPO_DIR=/home/ronimaor/monge-gap-regularizer-to-flow-map
OUTPUT_DIR=/home/ronimaor/experiments/checker_mg/phase2
mkdir -p "${OUTPUT_DIR}"

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
