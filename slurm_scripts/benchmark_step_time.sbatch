#!/bin/bash
#SBATCH --partition rtx6000
#SBATCH --time 00:30:00
#SBATCH --job-name checker_mg_bench
#SBATCH --output checker_mg_bench-%A_%a.out
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --tasks=1
#SBATCH --array=0,1,4

# slurm_id -> config being timed (see py/configs/checker_mg.py):
#   0  lambda=0.0             eps=0.05  (no-op Monge gap branch; baseline)
#   1  lambda=0.1             eps=0.05
#   4  lambda=BEST_LAMBDA     eps=0.01  (BEST_LAMBDA currently 0.1 in checker_mg.py)
#
# Note: lambda=0 is a static no-op (Sinkhorn isn't even traced), so eps has no
# effect there -- id 0 alone covers every "lambda=0" case regardless of eps.

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_ARRAY_TASK_ID"=$SLURM_ARRAY_TASK_ID

REPO_DIR=/home/ronimaor/monge-gap-regularizer-to-flow-map

module load anaconda
source activate flow_maps
export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_MODE=disabled

cd "${REPO_DIR}/py" || exit 1

python launchers/benchmark_step_time.py \
    --cfg_path configs.checker_mg \
    --slurm_id ${SLURM_ARRAY_TASK_ID} \
    --n_warmup 10 \
    --n_steps 100
