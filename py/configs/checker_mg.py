"""
Monge gap regularizer sweep on the 2D checkerboard.

slurm_id mapping:
  Phase 1 — lambda sweep (sinkhorn_eps=0.05 fixed):
    0  baseline  lambda=0.0
    1            lambda=0.1
    2            lambda=1.0
    3            lambda=10.0

  Phase 2 — eps sweep (update BEST_LAMBDA below after inspecting Phase 1 WandB):
    4  eps=0.01
    5  eps=0.1
    6  eps=0.5
    (eps=0.05 at BEST_LAMBDA is already covered by Phase 1, no need to re-run)

  Seed-repeat check (same lambda/eps as slurm_id 4, different cfg.training.seed,
  to see whether the KL improvement over baseline survives seed variance):
    7  seed=7   (else identical to slurm_id 4)
    8  seed=13  (else identical to slurm_id 4)

All experiments: LSD loss, convex stopgrad.

NOTE on seeding: cfg.training.seed controls network init and every JAX-random
draw (base noise, s/t/u/h sampling, dropout keys, the Monge-gap mg_s/mg_t/mg_x0
draws) via the prng_key threaded through datasets.setup_target ->
state_utils.setup_training_state -> flow_map.initialize_flow_map — there is no
separate hardcoded key anywhere in that path. It does NOT control the
checkerboard target pool itself: common/datasets.py:sample_checkerboard
receives a JAX key but immediately does `del key` and draws via np.random.rand
(NumPy's global legacy RNG), and np.random.seed() is never called anywhere in
this repo. So the 1e7-point target pool differs from process to process
regardless of cfg.training.seed — true for every run in this sweep, not just
the new ones. Given 1e7 iid samples from an exact checkerboard density, this
shouldn't matter statistically, but the runs are not bit-for-bit reproducible
on that axis.
"""

import os
import ml_collections

# ── UPDATE THIS after Phase 1 results are in ────────────────────────────────
BEST_LAMBDA = 0.1   # the lambda_reg that gave lowest KL at N=1/N=2 in Phase 1
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEED = 42

mg_experiments = [
    # Phase 1: lambda sweep, eps fixed at 0.05
    {"lambda_reg": 0.0,         "sinkhorn_eps": 0.05},  # 0  baseline
    {"lambda_reg": 0.1,         "sinkhorn_eps": 0.05},  # 1
    {"lambda_reg": 1.0,         "sinkhorn_eps": 0.05},  # 2
    {"lambda_reg": 10.0,        "sinkhorn_eps": 0.05},  # 3
    # Phase 2: eps sweep at BEST_LAMBDA (eps=0.05 skipped — already in Phase 1)
    {"lambda_reg": BEST_LAMBDA, "sinkhorn_eps": 0.01},  # 4
    {"lambda_reg": BEST_LAMBDA, "sinkhorn_eps": 0.1},   # 5
    {"lambda_reg": BEST_LAMBDA, "sinkhorn_eps": 0.5},   # 6
    # Seed-repeat of slurm_id 4 (lambda=BEST_LAMBDA, eps=0.01), different seed.
    # output_name gets a _seed{N} suffix (see below) so these can't collide
    # with slurm_id 4's checkpoints even if pointed at the same output_folder.
    {"lambda_reg": BEST_LAMBDA, "sinkhorn_eps": 0.01, "seed": 7},   # 7
    {"lambda_reg": BEST_LAMBDA, "sinkhorn_eps": 0.01, "seed": 13},  # 8
]


def get_config(
    slurm_id: int, dataset_location: str = "", output_folder: str = ""
) -> ml_collections.ConfigDict:
    import jax

    del dataset_location  # not needed for checker

    exp = mg_experiments[slurm_id % len(mg_experiments)]
    lambda_reg   = exp["lambda_reg"]
    sinkhorn_eps = exp["sinkhorn_eps"]
    seed         = exp.get("seed", DEFAULT_SEED)

    config = ml_collections.ConfigDict()

    # training
    config.training = ml_collections.ConfigDict()
    config.training.shuffle        = True
    config.training.conditional    = False
    config.training.class_dropout  = 0.0
    config.training.stopgrad_type  = "convex"
    config.training.psd_type       = None
    config.training.loss_type      = "lsd"
    config.training.tmin           = 0.0
    config.training.tmax           = 1.0
    config.training.seed           = seed
    config.training.ema_facs       = [0.999, 0.9999]
    config.training.ndevices       = jax.device_count()

    # Monge gap knobs
    config.training.lambda_reg        = lambda_reg
    config.training.sinkhorn_eps      = sinkhorn_eps
    config.training.sinkhorn_max_iter = 200
    config.training.mg_batch          = 512
    config.training.mg_min_gap        = 0.0

    # problem
    config.problem = ml_collections.ConfigDict()
    config.problem.n               = int(1e7)
    config.problem.d               = 2
    config.problem.image_dims      = None
    config.problem.num_classes     = None
    config.problem.target          = "checker"
    config.problem.dataset_location = None
    config.problem.interp_type     = "linear"
    config.problem.base            = "gaussian"
    config.problem.gaussian_scale  = "adaptive"

    # optimization
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs             = 100_000
    config.optimization.diag_fraction  = 0.75
    config.optimization.learning_rate  = 1e-3
    config.optimization.clip           = 10.0
    config.optimization.total_steps    = 250_000
    config.optimization.total_samples  = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps    = 35_000
    config.optimization.schedule_type  = "sqrt"

    # logging
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs       = 25_000
    config.logging.visual_freq   = 5_000
    config.logging.save_freq     = 10_000
    config.logging.wandb_project = "self-distill-flow-maps"

    lam_str = f"{lambda_reg:.1f}".replace(".", "p")   # e.g. "1.0" -> "1p0"
    eps_str = f"{sinkhorn_eps:.2f}".replace(".", "p") # e.g. "0.05" -> "0p05"
    wandb_name = f"checker_mg_lam{lam_str}_eps{eps_str}"
    if seed != DEFAULT_SEED:
        # Seed-repeat runs (slurm_id 7, 8, ...): suffix so checkpoints can't
        # collide with the DEFAULT_SEED run of the same (lambda, eps), even if
        # written to the same output_folder.
        wandb_name += f"_seed{seed}"
    config.logging.wandb_name    = wandb_name
    config.logging.wandb_entity  = os.getenv("WANDB_ENTITY", "ronimaor")
    config.logging.output_folder = output_folder
    config.logging.output_name   = config.logging.wandb_name

    config.logging.fid_freq          = 0
    config.logging.fid_stats_path    = None
    config.logging.fid_n_samples     = None
    config.logging.fid_batch_size    = None
    config.logging.fid_n_steps_flow  = None
    config.logging.fid_ema_factor    = None
    config.logging.visual_ema_factor = None

    # network — 4-layer 512-neuron MLP
    config.network = ml_collections.ConfigDict()
    config.network.network_type   = "mlp"
    config.network.n_hidden       = 4
    config.network.n_neurons      = 512
    config.network.output_dim     = 2
    config.network.act            = "gelu"
    config.network.use_residual   = False
    config.network.use_weight     = False
    config.network.use_bfloat16   = False
    config.network.rescale        = 0.5
    config.network.load_path      = ""
    config.network.input_dims     = (2,)
    config.network.load_ema_fac   = None
    config.network.img_resolution = None
    config.network.img_channels   = None
    config.network.label_dim      = None
    config.network.logvar_channels = None
    config.network.reset_optimizer = True
    config.network.unet_kwargs    = None

    return config
