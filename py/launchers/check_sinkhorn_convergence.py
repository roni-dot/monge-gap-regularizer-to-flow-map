"""
Read-only diagnostic: does the Monge-gap Sinkhorn solver converge at small eps?

The Monge gap regularizer (common/monge_gap_reg.py:sinkhorn_ot_cost) couples
one shared-(s,t) batch of interpolant points I_s against their flow-map images
X_st = X_{s,t}(I_s), via:

    geom   = pointcloud.PointCloud(I_s, X_st, epsilon=eps_rel, relative_epsilon="mean")
    prob   = linear_problem.LinearProblem(geom)          # uniform marginals a=b=1/n
    solver = sinkhorn.Sinkhorn(max_iterations=max_iter, threshold=1e-3)  # lse_mode=True (default)

i.e. cost_fn = squared Euclidean (PointCloud default), log-domain (lse_mode
defaults to True and is never overridden), epsilon is RELATIVE to the batch's
own mean cost matrix (relative_epsilon="mean") with no annealing/schedule
across iterations, max_iterations/threshold/batch size come straight from
cfg.training.{sinkhorn_max_iter, mg_batch}. This script builds one such batch
exactly the way common/losses.py's Monge-gap branch does (same batch size,
same _sample_mg_st / interpolant / flow-map-apply calls), then re-runs the
IDENTICAL solver call once per candidate epsilon.

This script does not import or modify common/monge_gap_reg.py's training
path — it reconstructs the same geom/prob/solver calls locally so it can
inspect the full SinkhornOutput (converged, n_iters, marginals), which
sinkhorn_ot_cost() does not expose (it only returns the scalar ent_reg_cost).

Usage (from py/):
    python launchers/check_sinkhorn_convergence.py \
        --cfg_path configs.checker_mg --slurm_id 0

    # against a real (trained) batch instead of a freshly initialized network:
    python launchers/check_sinkhorn_convergence.py \
        --cfg_path configs.checker_mg --slurm_id 0 \
        --checkpoint ../experiments/checker_mg/phase1/checker_mg_lam0p0_eps0p05_25.pkl
"""

import os
import sys
import argparse
import importlib

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
from flax.serialization import from_bytes
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

import common.datasets as datasets
import common.flow_map as flow_map
import common.interpolant as interpolant
import common.loss_args as loss_args
import common.state_utils as state_utils
from ml_collections import config_dict

EPS_LIST = [0.05, 0.01, 0.005, 0.001]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_path", type=str, required=True)
    p.add_argument("--slurm_id", type=int, required=True)
    p.add_argument("--dataset_location", type=str, default="")
    p.add_argument("--output_folder", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="",
                    help="Optional .pkl checkpoint; omit to use a freshly "
                         "initialized network (as at the start of training).")
    p.add_argument("--ema_fac", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0,
                    help="Seed for the mg batch draw (independent of cfg.training.seed).")
    return p.parse_args()


def build_mg_batch(cfg, net, params, statics_ds, sample_rho0, interp, prng_key):
    """Exactly mirrors the Monge-gap branch in common/losses.py:loss()."""
    prng_key, mg_key, mg_x0key = jax.random.split(prng_key, 3)

    mg_s, mg_t = loss_args._sample_mg_st(
        mg_key, cfg.training.tmin, cfg.training.tmax, cfg.training.mg_min_gap
    )
    mg_x0 = sample_rho0(cfg.training.mg_batch, mg_x0key)
    mg_x1 = next(statics_ds)[: cfg.training.mg_batch]

    I_s_mg = jax.vmap(lambda x0i, x1i: interp.calc_It(mg_s, x0i, x1i))(mg_x0, mg_x1)
    X_st_mg = jax.vmap(
        lambda xi: net.apply(params, mg_s, mg_t, xi, None, train=False)
    )(I_s_mg)

    return I_s_mg, X_st_mg, mg_s, mg_t, prng_key


def run_sinkhorn(x, y, eps_rel, max_iter):
    """IDENTICAL construction to common/monge_gap_reg.py:sinkhorn_ot_cost,
    but keeping the full SinkhornOutput instead of just out.ent_reg_cost."""
    geom = pointcloud.PointCloud(x, y, epsilon=eps_rel, relative_epsilon="mean")
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(max_iterations=max_iter, threshold=1e-3)
    out = solver(prob)
    return out, geom, solver


def main():
    args = parse_args()

    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, args.dataset_location, args.output_folder)
    cfg, ds, prng_key = datasets.setup_target(cfg, jax.random.PRNGKey(cfg.training.seed))
    interp = interpolant.setup_interpolant(cfg)
    cfg = config_dict.FrozenConfigDict(cfg)

    ex_input = jnp.zeros((cfg.problem.d,))
    sample_rho0 = datasets.setup_base(cfg, ex_input)

    net, params, _ = flow_map.initialize_flow_map(cfg.network, ex_input, jax.random.PRNGKey(0))
    if args.checkpoint:
        tx, _ = state_utils.setup_optimizer(cfg)
        ema_init = {fac: params for fac in cfg.training.ema_facs}
        state = state_utils.EMATrainState.create(
            apply_fn=net.apply, params=params, ema_params=ema_init, tx=tx
        )
        print(f"Loading checkpoint: {args.checkpoint}")
        with open(args.checkpoint, "rb") as f:
            state = from_bytes(state, f.read())
        params = state.ema_params.get(args.ema_fac, state.params)
        print(f"Checkpoint step: {int(state.step)}, using ema_fac={args.ema_fac}")
    else:
        print("No --checkpoint given: using a freshly initialized network "
              "(mg batch will look like the very start of training).")

    print("\n=== Training-time Sinkhorn configuration (common/monge_gap_reg.py) ===")
    print(f"  max_iterations   = {cfg.training.sinkhorn_max_iter}")
    print(f"  threshold        = 1e-3  (hardcoded in monge_gap_reg.py, matches ott default)")
    print(f"  mg_batch (n=m)   = {cfg.training.mg_batch}")
    print(f"  cost_fn          = squared Euclidean (PointCloud default)")
    print(f"  relative_epsilon = 'mean'  (eps_actual = sinkhorn_eps * mean(cost_matrix))")
    print(f"  epsilon schedule = none (single flat eps per call, no annealing)")
    print(f"  lse_mode         = True (log-domain; default, never overridden)")
    print(f"  sinkhorn_eps in training config = {cfg.training.sinkhorn_eps}")

    prng_key, batch_key = jax.random.split(jax.random.PRNGKey(args.seed))
    I_s_mg, X_st_mg, mg_s, mg_t, _ = build_mg_batch(
        cfg, net, params, ds, sample_rho0, interp, batch_key
    )
    print(f"\nMonge-gap batch: mg_s={float(mg_s):.4f}, mg_t={float(mg_t):.4f}, "
          f"n={I_s_mg.shape[0]}, d={I_s_mg.shape[1]}")

    print("\n=== Sinkhorn convergence sweep ===")
    header = (f"{'eps':>8} | {'eps_actual':>11} | {'converged':>9} | {'n_iters':>7} | "
              f"{'marginal_err':>13} | {'min exp(-C/eps)':>17} | {'lse_mode':>8}")
    print(header)
    print("-" * len(header))

    for eps in EPS_LIST:
        out, geom, solver = run_sinkhorn(I_s_mg, X_st_mg, eps, cfg.training.sinkhorn_max_iter)

        converged = bool(out.converged)
        n_iters = int(out.n_iters)

        # |P @ 1 - a|_1 : row-marginal violation of the transport plan.
        row_marginal = out.marginal(1)
        marginal_err = float(jnp.sum(jnp.abs(row_marginal - out.a)))

        eps_actual = float(geom.epsilon)
        min_kernel = float(jnp.min(jnp.exp(-geom.cost_matrix / eps_actual)))

        print(f"{eps:>8.4f} | {eps_actual:>11.6f} | {str(converged):>9} | {n_iters:>7d} | "
              f"{marginal_err:>13.3e} | {min_kernel:>17.3e} | {str(solver.lse_mode):>8}")

        if min_kernel == 0.0:
            print(f"    -> WARNING: kernel exp(-C/eps) underflowed to exactly 0 at eps={eps}. "
                  f"In lse_mode this doesn't break Sinkhorn directly (updates use log-sum-exp "
                  f"on C/eps, not the kernel), but it signals eps is very small relative to the "
                  f"cost scale and iterations may be needed near max_iterations to converge.")
        if not converged:
            print(f"    -> WARNING: did NOT converge within max_iterations="
                  f"{cfg.training.sinkhorn_max_iter} at eps={eps} (threshold=1e-3).")

    print("\nDone.")


if __name__ == "__main__":
    main()
