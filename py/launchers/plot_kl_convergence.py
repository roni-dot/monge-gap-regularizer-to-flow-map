"""
Fig D: KL vs training step for baseline vs Monge-gap regularized flow map.
(Re-MeanFlow Fig. 6 analog: convergence under equal training budget.)

Evaluates every intermediate checkpoint with the SAME eval protocol as
eval_checker_kl.py (n_bins=50 quadrature, identical base noise across all
checkpoints and both models), caches per-checkpoint results to JSON so
re-plotting is instant, then draws KL-vs-step curves per NFE.

Checkpoints are written by common/logging.py:save_state directly into
cfg.logging.output_folder (no per-run subdirectory) as:
    {output_name}_{step // save_freq}.pkl
where output_name = cfg.logging.wandb_name, e.g. "checker_mg_lam0p1_eps0p01".
So the glob patterns below should point at that prefix directly.

Usage (from py/):
    python launchers/plot_kl_convergence.py \
        --cfg_path configs.checker_mg --slurm_id 0 \
        --ckpts_baseline "../experiments/checker_mg/phase1/checker_mg_lam0p0_eps0p05_*.pkl" \
        --ckpts_reg      "../experiments/checker_mg/phase1/checker_mg_lam0p1_eps0p01_*.pkl" \
        --output_folder  ../experiments/checker_mg/figs \
        --n_samples 64000

Quote the globs so the shell doesn't expand them.
ADAPT: uses the same load pattern as eval_checker_kl.py; nothing model-specific
beyond flow_map.batch_sample, which you already use.
"""

import os
import sys
import json
import glob
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax.serialization import from_bytes

import common.datasets as datasets
import common.flow_map as flow_map
import common.state_utils as state_utils
from ml_collections import config_dict


# ── KL: identical to the fixed eval_checker_kl.py ──────────────────────────────

def eval_checkerboard(x, n_squares=4):
    xi = np.floor((x[..., 0] + 1) / 2 * n_squares).astype(int)
    yi = np.floor((x[..., 1] + 1) / 2 * n_squares).astype(int)
    return np.where((xi + yi) % 2 == 0, 0.5, 0.0)


def compute_kl_quadrature(samples, n_bins=50, n_squares=4, eps=1e-10):
    edges = np.linspace(-1, 1, n_bins + 1)
    bin_area = (edges[1] - edges[0]) ** 2
    q, _, _ = np.histogram2d(samples[:, 0], samples[:, 1],
                             bins=[edges, edges], density=True)
    c = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(c, c, indexing="ij")
    p = eval_checkerboard(np.stack([xx, yy], axis=-1), n_squares)
    q = q + eps
    return float(np.sum(np.where(p > 0, p * np.log(p / q) * bin_area, 0.0)))


# ── plumbing ────────────────────────────────────────────────────────────────────

def build_template_state(cfg):
    ex_input = jnp.zeros((cfg.problem.d,))
    net, params, _ = flow_map.initialize_flow_map(cfg.network, ex_input, jax.random.PRNGKey(0))
    tx, _ = state_utils.setup_optimizer(cfg)
    ema_init = {fac: params for fac in cfg.training.ema_facs}
    state = state_utils.EMATrainState.create(apply_fn=net.apply, params=params,
                                             ema_params=ema_init, tx=tx)
    return net, state


def eval_ckpt(net, template_state, ckpt_path, x0s, nfe_list, ema_fac, n_bins):
    with open(ckpt_path, "rb") as f:
        state = from_bytes(template_state, f.read())
    step = int(state.step)
    params = state.ema_params.get(ema_fac, state.params)
    kls = {}
    for nfe in nfe_list:
        s = flow_map.batch_sample(net.apply, params, x0s, nfe,
                                  -jnp.ones(x0s.shape[0]))
        kls[str(nfe)] = compute_kl_quadrature(np.array(s), n_bins=n_bins)
    return step, kls


def eval_run(name, pattern, net, template_state, x0s, nfe_list, ema_fac,
             n_bins, cache_path):
    """Evaluate all checkpoints matching pattern; cache by filename."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No checkpoints match: {pattern}")
    print(f"[{name}] {len(paths)} checkpoints")
    for pth in paths:
        key = os.path.basename(pth)
        if key in cache:
            continue
        step, kls = eval_ckpt(net, template_state, pth, x0s, nfe_list,
                              ema_fac, n_bins)
        cache[key] = {"step": step, "kl": kls}
        print(f"  step {step:>7}: " + "  ".join(
            f"N={n}:{kls[str(n)]:.4f}" for n in nfe_list))
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    rows = sorted(cache.values(), key=lambda r: r["step"])
    steps = np.array([r["step"] for r in rows])
    kl = {n: np.array([r["kl"][str(n)] for r in rows]) for n in nfe_list}
    return steps, kl


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_path", required=True)
    p.add_argument("--slurm_id", type=int, required=True)
    p.add_argument("--ckpts_baseline", required=True,
                    help='glob (quoted), e.g. ".../checker_mg_lam0p0_eps0p05_*.pkl"')
    p.add_argument("--ckpts_reg", required=True,
                    help='glob (quoted), e.g. ".../checker_mg_lam0p1_eps0p01_*.pkl"')
    p.add_argument("--output_folder", required=True)
    p.add_argument("--n_samples", type=int, default=64000)
    p.add_argument("--n_bins", type=int, default=50)
    p.add_argument("--nfe_list", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--ema_fac", type=float, default=0.999)
    p.add_argument("--dataset_location", default="")
    p.add_argument("--label_reg", default=r"Monge gap ($\lambda{=}0.1,\varepsilon{=}0.01$)")
    args = p.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, args.dataset_location, args.output_folder)
    cfg, _, key = datasets.setup_target(cfg, jax.random.PRNGKey(cfg.training.seed))
    cfg = config_dict.FrozenConfigDict(cfg)

    net, template_state = build_template_state(cfg)

    # ONE fixed noise set reused for every checkpoint and both models,
    # so curve wiggles are the model changing, not the draw changing.
    ex_input = jnp.zeros((cfg.problem.d,))
    sample_rho0 = datasets.setup_base(cfg, ex_input)
    key, k = jax.random.split(key)
    x0s = sample_rho0(args.n_samples, k)

    runs = {
        "LSD baseline": args.ckpts_baseline,
        args.label_reg: args.ckpts_reg,
    }
    colors = {"LSD baseline": "tab:red", args.label_reg: "tab:green"}
    curves = {}
    for name, pattern in runs.items():
        cache_path = os.path.join(
            args.output_folder,
            f"klcurve_cache_{'baseline' if 'baseline' in name else 'reg'}.json")
        curves[name] = eval_run(name, pattern, net, template_state, x0s,
                                args.nfe_list, args.ema_fac, args.n_bins,
                                cache_path)

    # ---------- plot: one panel per NFE ----------
    n_panels = len(args.nfe_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.8),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]
    for ax, nfe in zip(axes, args.nfe_list):
        for name, (steps, kl) in curves.items():
            ax.plot(steps, kl[nfe], marker="o", ms=3, lw=1.4,
                    color=colors[name], label=name)
        ax.set_title(f"N = {nfe}")
        ax.set_xlabel("training step")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"KL$(\rho_1 \,\|\, \hat\rho_1)$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Convergence: KL vs training step (fixed eval: "
                 f"{args.n_bins}-bin quadrature, {args.n_samples} samples)")
    fig.tight_layout()
    out = os.path.join(args.output_folder, "fig_kl_convergence.png")
    fig.savefig(out, dpi=160)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
