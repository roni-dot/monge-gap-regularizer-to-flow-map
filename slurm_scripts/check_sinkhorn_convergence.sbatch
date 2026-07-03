#!/bin/bash
#SBATCH --partition rtx6000
#SBATCH --time 00:20:00
#SBATCH --job-name checker_mg_sinkhorn_diag
#SBATCH --output checker_mg_sinkhorn_diag-%j.out
#SBATCH --mail-user=ronimaor@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --qos=pakman
#SBATCH --tasks=1

# Read-only diagnostic: sweeps eps in [0.05, 0.01, 0.005, 0.001] through the
# EXACT Sinkhorn solver used by the Monge-gap regularizer (see
# common/monge_gap_reg.py), on one mg-batch built the same way as
# common/losses.py's Monge-gap branch. Does not modify training code.
#
# slurm_id 4 = lambda=BEST_LAMBDA, eps=0.01 in configs/checker_mg.py's
# mg_experiments sweep (BEST_LAMBDA=0.1 currently), i.e. lambda=0.1, eps=0.01
# -- matches the CHECKPOINT below (phase2's checker_mg_lam0p1_eps0p01 run).
#
# Runs the diagnostic once per seed (0, 1, 2) in this same job/allocation, so
# convergence at small eps can be checked across independent mg batches
# rather than a single lucky/unlucky draw.

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST

REPO_DIR=/home/ronimaor/monge-gap-regularizer-to-flow-map
CHECKPOINT=${REPO_DIR}/experiments/checker_mg/phase2/checker_mg_lam0p1_eps0p01_25.pkl

module load anaconda
source activate flow_maps
export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_MODE=disabled

cd "${REPO_DIR}/py" || exit 1

for SEED in 0 1 2; do
    echo "===== SEED ${SEED} ====="
    python launchers/check_sinkhorn_convergence.py \
        --cfg_path configs.checker_mg \
        --slurm_id 4 \
        --checkpoint ${CHECKPOINT} \
        --seed ${SEED}
done
