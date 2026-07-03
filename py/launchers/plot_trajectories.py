"""
Trajectory & transport figures: baseline (lam=0) vs Monge-gap regularized flow map.

Produces (all baseline-vs-regularized side by side or overlaid):
  fig_trajectories.png   - particle paths X_{0,t}(z) over the checkerboard   [Re-MeanFlow Fig.1/2 analog]
  fig_straightness.png   - path-length/chord-length distribution + curvature vs t [Curv(r,t) analog]
  fig_transport_hist.png - histogram of one-step displacement ||X_{0,1}(z)-z||   [Re-MeanFlow Fig.3 analog]
  fig_samples_N1.png     - N=1 sample scatter, both models                    [Boffi Fig.3 analog]

Usage (from py/):
    python launchers/plot_trajectories.py \
        --cfg_path configs.checker_mg --slurm_id 0 \
        --ckpt_baseline ../experiments/checker_mg/phase1/checker_mg_lam0p0_eps0p05_25.pkl \
        --ckpt_reg      ../experiments/checker_mg/phase1/checker_mg_lam0p1_eps0p01_25.pkl \
        --output_folder ../experiments/checker_mg/figs
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax.serialization import from_bytes

import common.datasets as datasets
import common.flow_map as flow_map
import common.state_utils as state_utils
from ml_collections import config_dict


# ── model plumbing (mirrors eval_checker_kl.py) ────────────────────────────────

def load_model(cfg, ckpt_path, ema_fac):
    ex_input = jnp.zeros((cfg.problem.d,))
    net, params, _ = flow_map.initialize_flow_map(cfg.network, ex_input, jax.random.PRNGKey(0))
    tx, _ = state_utils.setup_optimizer(cfg)
    ema_init = {fac: params for fac in cfg.training.ema_facs}
    state = state_utils.EMATrainState.create(apply_fn=net.apply, params=params,
                                             ema_params=ema_init, tx=tx)
    with open(ckpt_path, "rb") as f:
        state = from_bytes(state, f.read())
    p = state.ema_params.get(ema_fac, state.params)
    return net, p


def flow_map_apply(net, params, x, s, t, label=-1.0):
    """X_{s,t}(x): direct single-jump evaluation of the flow map.

    FlowMap.__call__ (see common/network_utils.py) already returns
    X_{s,t}(x) = x + (t - s) * phi(s,t,x) internally, so calling net.apply
    directly gives X_{s,t}(x) with no extra affine combination needed here
    (unlike a raw velocity-field parameterization). Matches the label
    convention (-1 = unconditional) used by flow_map.batch_sample callers
    elsewhere in this codebase, e.g. eval_checker_kl.py.
    """
    apply_one = lambda xi: net.apply(params, s, t, xi, label=label, train=False,
                                      calc_weight=False, return_X_and_phi=False,
                                      init_weights=False)
    return jax.vmap(apply_one)(x)


def trace_trajectories(net, params, z, n_t=101):
    """Paths gamma(t) = X_{0,t}(z) for t on a fine grid. Returns (n_t, n_particles, 2)."""
    ts = np.linspace(0.0, 1.0, n_t)
    path = [np.array(z)]
    for t in ts[1:]:
        path.append(np.array(flow_map_apply(net, params, z, 0.0, float(t))))
    return ts, np.stack(path, axis=0)


# ── statistics ──────────────────────────────────────────────────────────────

def straightness_ratio(path):
    """path: (n_t, n, 2). arclength / chord length per particle; 1.0 = straight."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=-1)          # (n_t-1, n)
    arclen = seg.sum(axis=0)                                       # (n,)
    chord = np.linalg.norm(path[-1] - path[0], axis=-1)            # (n,)
    return arclen / np.maximum(chord, 1e-12)


def curvature_vs_t(ts, path):
    """Mean second-difference norm (discrete curvature proxy) per interior time."""
    acc = np.diff(path, n=2, axis=0) / (ts[1] - ts[0]) ** 2        # (n_t-2, n, 2)
    return ts[1:-1], np.linalg.norm(acc, axis=-1).mean(axis=1)     # (n_t-2,)


def checker_background(ax):
    xs = np.linspace(-1, 1, 201)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    white = ((np.floor((xx + 1) / 2 * 4).astype(int)
              + np.floor((yy + 1) / 2 * 4).astype(int)) % 2 == 0)
    ax.imshow(white.T, origin="lower", extent=[-1, 1, -1, 1], cmap="Greys",
              alpha=0.18, zorder=0)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_path", required=True)
    p.add_argument("--slurm_id", type=int, required=True)
    p.add_argument("--ckpt_baseline", required=True)
    p.add_argument("--ckpt_reg", required=True)
    p.add_argument("--output_folder", required=True)
    p.add_argument("--n_traj", type=int, default=40, help="particles for trajectory panel")
    p.add_argument("--n_stats", type=int, default=20000, help="particles for histograms/stats")
    p.add_argument("--ema_fac", type=float, default=0.999)
    p.add_argument("--dataset_location", default="")
    p.add_argument("--label_reg", default=r"Monge gap ($\lambda{=}0.1,\varepsilon{=}0.01$)")
    args = p.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, args.dataset_location, args.output_folder)
    cfg, _, key = datasets.setup_target(cfg, jax.random.PRNGKey(cfg.training.seed))
    cfg = config_dict.FrozenConfigDict(cfg)

    models = {
        "LSD baseline": load_model(cfg, args.ckpt_baseline, args.ema_fac),
        args.label_reg: load_model(cfg, args.ckpt_reg, args.ema_fac),
    }
    colors = {"LSD baseline": "tab:red", args.label_reg: "tab:green"}

    # SAME noise for both models -> differences are the model, not the draw.
    ex_input = jnp.zeros((cfg.problem.d,))
    sample_rho0 = datasets.setup_base(cfg, ex_input)
    key, k1, k2 = jax.random.split(key, 3)
    z_traj = sample_rho0(args.n_traj, k1)
    z_stat = sample_rho0(args.n_stats, k2)

    # ---------- Fig A: trajectories ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (name, (net, params)) in zip(axes, models.items()):
        checker_background(ax)
        ts, path = trace_trajectories(net, params, z_traj)
        for i in range(path.shape[1]):
            ax.plot(path[:, i, 0], path[:, i, 1], lw=0.8, alpha=0.7,
                    color=colors[name], zorder=2)
        ax.scatter(path[0, :, 0], path[0, :, 1], s=8, c="k", zorder=3, label=r"$z\sim\rho_0$")
        ax.scatter(path[-1, :, 0], path[-1, :, 1], s=8, c="tab:blue", zorder=3,
                   label=r"$X_{0,1}(z)$")
        sr = straightness_ratio(path)
        ax.set_title(f"{name}\nmean arclen/chord = {sr.mean():.4f}")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(r"Particle trajectories $t \mapsto X_{0,t}(z)$ (same noise for both models)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "fig_trajectories.png"), dpi=160)
    plt.close(fig)

    # ---------- Fig B: straightness distribution + curvature vs t ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (net, params) in models.items():
        ts, path = trace_trajectories(net, params, z_stat, n_t=51)
        sr = straightness_ratio(path)
        ax1.hist(sr, bins=80, range=(1.0, np.quantile(sr, 0.995)), density=True,
                 alpha=0.5, color=colors[name], label=f"{name} (mean {sr.mean():.4f})")
        tt, curv = curvature_vs_t(ts, path)
        ax2.plot(tt, curv, color=colors[name], label=name)
    ax1.set_xlabel("arc length / chord length  (1 = straight)")
    ax1.set_ylabel("density"); ax1.legend(fontsize=8)
    ax1.set_title("Trajectory straightness")
    ax2.set_xlabel("t"); ax2.set_ylabel(r"mean $\|\ddot{\gamma}(t)\|$")
    ax2.set_title("Curvature along trajectories"); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "fig_straightness.png"), dpi=160)
    plt.close(fig)

    # ---------- Fig C: transport-cost histogram (Re-MeanFlow Fig.3 analog) ----------
    # N=1 sample: reuse flow_map.batch_sample (same helper eval_checker_kl.py uses)
    # rather than a manual s=0,t=1 call, so this matches the tested sampling path.
    label_stat = -jnp.ones(z_stat.shape[0])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, (net, params) in models.items():
        x1 = np.array(flow_map.batch_sample(net.apply, params, z_stat, 1, label_stat))
        d = np.linalg.norm(x1 - np.array(z_stat), axis=-1)
        ax.hist(d, bins=100, density=True, alpha=0.5, color=colors[name],
                label=f"{name} (mean {d.mean():.4f})")
    ax.set_xlabel(r"$\|X_{0,1}(z) - z\|_2$ (one-step displacement)")
    ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title("Transport distance of the learned one-step map")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "fig_transport_hist.png"), dpi=160)
    plt.close(fig)

    # ---------- Fig D: N=1 samples (Boffi Fig.3-style) ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, (net, params)) in zip(axes, models.items()):
        checker_background(ax)
        x1 = np.array(flow_map.batch_sample(net.apply, params, z_stat, 1, label_stat))
        ax.scatter(x1[:, 0], x1[:, 1], s=0.5, alpha=0.25, color=colors[name])
        ax.set_title(f"{name}  (N=1 samples)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_folder, "fig_samples_N1.png"), dpi=160)
    plt.close(fig)

    print(f"Wrote 4 figures to {args.output_folder}")


if __name__ == "__main__":
    main()
