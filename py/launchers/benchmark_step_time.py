"""
Benchmark per-step training wall time, to measure the Sinkhorn/Monge-gap
overhead as a function of sinkhorn_eps (and lambda_reg=0 as a no-op baseline).

Runs cfg.training.lambda_reg == 0.0 as a true no-op (see common/losses.py), so
comparing against a lambda_reg>0 config isolates the added Sinkhorn cost.

Mirrors the state/loss/train_step setup in launchers/learn.py:setup_state,
minus wandb and FID (neither needed for timing).

Usage (from py/):
    python launchers/benchmark_step_time.py \
        --cfg_path configs.checker_mg --slurm_id 0 \
        --n_warmup 10 --n_steps 100
"""

import os
import sys
import argparse
import importlib
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import jax
import numpy as np
from ml_collections import config_dict

import common.datasets as datasets
import common.dist_utils as dist_utils
import common.interpolant as interpolant
import common.loss_args as loss_args
import common.losses as losses
import common.state_utils as state_utils
import common.updates as updates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_path", required=True)
    p.add_argument("--slurm_id", type=int, required=True)
    p.add_argument("--dataset_location", default="")
    p.add_argument("--output_folder", default="")
    p.add_argument("--n_warmup", type=int, default=10)
    p.add_argument("--n_steps", type=int, default=100)
    return p.parse_args()


def setup_state(cfg, prng_key):
    """Same construction as launchers/learn.py:setup_state, minus wandb/FID."""
    cfg, ds, prng_key = datasets.setup_target(cfg, prng_key)
    ex_input = next(ds)
    ex_input = ex_input["image"][0] if isinstance(ex_input, dict) else ex_input[0]
    interp = interpolant.setup_interpolant(cfg)
    cfg = config_dict.FrozenConfigDict(cfg)

    train_state, net, schedule, prng_key = state_utils.setup_training_state(
        cfg, ex_input, prng_key
    )
    loss = losses.setup_loss(cfg, net, interp)

    statics = state_utils.StaticArgs(
        net=net,
        schedule=schedule,
        loss=loss,
        get_loss_fn_args=loss_args.get_loss_fn_args,
        train_step=updates.setup_train_step(cfg),
        update_ema_params=updates.setup_ema_update(cfg),
        ds=ds,
        interp=interp,
        sample_rho0=datasets.setup_base(cfg, ex_input),
        inception_fn=None,
    )

    train_state = dist_utils.safe_replicate(cfg, train_state)
    return cfg, statics, train_state, prng_key


def run_steps(cfg, statics, train_state, prng_key, n):
    """Run n train steps (+ EMA update), blocking on device work after each."""
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        loss_fn_args, prng_key = statics.get_loss_fn_args(
            cfg, statics, train_state, prng_key
        )
        train_state, loss_value, grads, aux = statics.train_step(
            train_state, statics.loss, loss_fn_args
        )
        train_state = statics.update_ema_params(train_state)
        jax.block_until_ready((train_state.params, loss_value))
        times[i] = time.perf_counter() - t0
    return times, train_state, prng_key


def main():
    args = parse_args()

    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, args.dataset_location, args.output_folder)
    cfg.training.ndevices = jax.device_count()
    print(f"lambda_reg={cfg.training.lambda_reg}  sinkhorn_eps={cfg.training.sinkhorn_eps}  "
          f"ndevices={cfg.training.ndevices}")

    prng_key = jax.random.PRNGKey(cfg.training.seed)
    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)

    print(f"Compiling + warming up ({args.n_warmup} steps)...")
    _, train_state, prng_key = run_steps(cfg, statics, train_state, prng_key, args.n_warmup)

    print(f"Timing {args.n_steps} steps...")
    times, train_state, prng_key = run_steps(
        cfg, statics, train_state, prng_key, args.n_steps
    )

    print(f"\nmean step time: {times.mean() * 1e3:.3f} ms  "
          f"(std {times.std() * 1e3:.3f} ms, min {times.min() * 1e3:.3f} ms, "
          f"max {times.max() * 1e3:.3f} ms) over {args.n_steps} steps")


if __name__ == "__main__":
    main()
