"""
Evaluate KL divergence of checkerboard samples at multiple NFE values.

Usage (run from py/ directory):
    python launchers/eval_checker_kl.py \
        --cfg_path configs.checker_mg \
        --slurm_id 0 \
        --checkpoint /path/to/checkpoint.pkl \
        --output_folder /path/to/results \
        --n_samples 100000

Outputs a JSON file: <output_folder>/kl_results_<run_name>.json
"""

import os
import sys
import json
import argparse
import importlib

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes

import common.datasets as datasets
import common.flow_map as flow_map
import common.interpolant as interpolant
import common.state_utils as state_utils
from ml_collections import config_dict


# ── KL computation ────────────────────────────────────────────────────────────

def checkerboard_kl(model_samples: np.ndarray, n_bins: int = 64, eps: float = 1e-8) -> float:
    """
    KL(p_true || p_model) via histogram quadrature on a 2D grid.

    model_samples: (N, 2) array of generated points
    n_bins:        grid resolution (64 gives stable estimates for N >= 50k)
    """
    edges = np.linspace(-1.25, 1.25, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centers, centers, indexing="ij")

    # 4x4 checkerboard on [-1, 1]: cell index in [0,3]
    x_idx = np.floor((xx + 1.0) * 2.0).astype(int).clip(0, 3)
    y_idx = np.floor((yy + 1.0) * 2.0).astype(int).clip(0, 3)
    is_white = ((x_idx + y_idx) % 2 == 0) & (np.abs(xx) <= 1.0) & (np.abs(yy) <= 1.0)

    p_true = is_white.astype(float)
    p_true /= p_true.sum()

    hist, _, _ = np.histogram2d(
        model_samples[:, 0], model_samples[:, 1], bins=edges
    )
    p_model = hist / hist.sum()

    mask = p_true > 0
    kl = float(np.sum(p_true[mask] * np.log(p_true[mask] / (p_model[mask] + eps))))
    return kl


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_path", type=str, required=True)
    p.add_argument("--slurm_id", type=int, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pkl checkpoint")
    p.add_argument("--output_folder", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=100_000)
    p.add_argument("--nfe_list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--ema_fac", type=float, default=0.999, help="Which EMA params to use")
    p.add_argument("--dataset_location", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, args.dataset_location, args.output_folder)
    cfg, ds, prng_key = datasets.setup_target(cfg, jax.random.PRNGKey(cfg.training.seed))
    cfg = config_dict.FrozenConfigDict(cfg)

    # Build model and load checkpoint
    ex_input = jnp.zeros((cfg.problem.d,))
    net, params, prng_key = flow_map.initialize_flow_map(cfg.network, ex_input, prng_key)

    tx, _ = state_utils.setup_optimizer(cfg)
    ema_params_init = {fac: params for fac in cfg.training.ema_facs}
    train_state = state_utils.EMATrainState.create(
        apply_fn=net.apply, params=params, ema_params=ema_params_init, tx=tx
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        train_state = from_bytes(train_state, f.read())

    step = int(train_state.step)
    print(f"Checkpoint step: {step}")

    # Select EMA params for evaluation
    if args.ema_fac in train_state.ema_params:
        eval_params = train_state.ema_params[args.ema_fac]
        print(f"Using EMA {args.ema_fac} params")
    else:
        eval_params = train_state.params
        print("EMA factor not found, using instantaneous params")

    # Sample base noise
    prng_key, key = jax.random.split(prng_key)
    sample_rho0 = datasets.setup_base(cfg, ex_input)
    x0s = sample_rho0(args.n_samples, key)

    # Evaluate KL at each NFE
    results = {"step": step, "n_samples": args.n_samples, "kl": {}}

    for nfe in args.nfe_list:
        print(f"Sampling at N={nfe}...", end=" ", flush=True)
        samples = flow_map.batch_sample(net.apply, eval_params, x0s, nfe, -jnp.ones(args.n_samples))
        samples = np.array(samples)
        kl = checkerboard_kl(samples, n_bins=64)
        results["kl"][str(nfe)] = kl
        print(f"KL = {kl:.4f}")

    # Print summary table
    print("\n=== KL Summary ===")
    print(f"{'NFE':>5} | {'KL':>8}")
    print("-" * 16)
    for nfe in args.nfe_list:
        print(f"{nfe:>5} | {results['kl'][str(nfe)]:>8.4f}")

    # Save results
    os.makedirs(args.output_folder, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(args.output_folder, f"kl_{run_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
