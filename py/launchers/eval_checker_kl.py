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
import common.state_utils as state_utils
from ml_collections import config_dict


# ── KL computation — matches nmboffi/flow-maps notebooks/checker.ipynb exactly ─

def eval_checkerboard(x: np.ndarray, n_squares: int = 4) -> np.ndarray:
    """True density of the checkerboard: 0.5 on white squares, 0 elsewhere."""
    x_unit = (x[..., 0] + 1) / 2
    y_unit = (x[..., 1] + 1) / 2
    x_idx = np.floor(x_unit * n_squares).astype(int)
    y_idx = np.floor(y_unit * n_squares).astype(int)
    is_white = (x_idx + y_idx) % 2 == 0
    return np.where(is_white, 0.5, 0.0)


def compute_kl_quadrature(
    samples: np.ndarray,
    n_bins: int = 50,
    n_squares: int = 4,
    eps: float = 1e-10,
) -> float:
    """
    KL(p_true || q_model) via histogram quadrature.

    Matches the released reference implementation exactly:
    - density=True in histogram2d (no manual normalization)
    - eps=1e-10 added to q before log (empty white bins ARE penalized)
    - mask on p > 0 only (not on q > 0)
    """
    edges = np.linspace(-1, 1, n_bins + 1)
    bin_area = (edges[1] - edges[0]) ** 2
    q, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=[edges, edges], density=True
    )
    c = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(c, c, indexing="ij")
    p = eval_checkerboard(np.stack([xx, yy], axis=-1), n_squares)
    q = q + eps
    return float(np.sum(np.where(p > 0, p * np.log(p / q) * bin_area, 0.0)))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_path", type=str, required=True)
    p.add_argument("--slurm_id", type=int, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pkl checkpoint")
    p.add_argument("--output_folder", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=100_000)
    p.add_argument("--n_bins", type=int, default=50, help="Histogram bins per axis (paper uses 50)")
    p.add_argument("--nfe_list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--ema_fac", type=float, default=0.999, help="Which EMA params to use")
    p.add_argument("--dataset_location", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, args.dataset_location, args.output_folder)
    cfg, _, prng_key = datasets.setup_target(cfg, jax.random.PRNGKey(cfg.training.seed))
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
    results = {"step": step, "n_samples": args.n_samples, "n_bins": args.n_bins, "kl": {}}

    first = True
    for nfe in args.nfe_list:
        print(f"Sampling at N={nfe}...", end=" ", flush=True)
        samples = flow_map.batch_sample(net.apply, eval_params, x0s, nfe, -jnp.ones(args.n_samples))
        samples = np.array(samples)

        # Sanity check printed once: coverage of [-1,1]^2 and per-axis std
        if first:
            inside = np.all(np.abs(samples) <= 1.0, axis=-1).mean()
            print(f"\n  [sanity] frac in [-1,1]^2={inside:.4f}  "
                  f"std_x={samples[:,0].std():.4f}  std_y={samples[:,1].std():.4f}")
            first = False

        # Save raw samples for offline re-analysis
        np.save(os.path.join(args.output_folder, f"samples_N{nfe}.npy"), samples)

        kl = compute_kl_quadrature(samples, n_bins=args.n_bins)
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
