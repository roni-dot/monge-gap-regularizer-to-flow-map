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

All experiments: LSD loss, convex stopgrad.
"""

import os
import ml_collections

# ── UPDATE THIS after Phase 1 results are in ────────────────────────────────
BEST_LAMBDA = 0.1   # the lambda_reg that gave lowest KL at N=1/N=2 in Phase 1
# ─────────────────────────────────────────────────────────────────────────────

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
]


def get_config(
    slurm_id: int, dataset_location: str = "", output_folder: str = ""
) -> ml_collections.ConfigDict:
    import jax

    del dataset_location  # not needed for checker

    exp = mg_experiments[slurm_id % len(mg_experiments)]
    lambda_reg   = exp["lambda_reg"]
    sinkhorn_eps = exp["sinkhorn_eps"]

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
    config.training.seed           = 42
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
    config.logging.wandb_name    = f"checker_mg_lam{lam_str}_eps{eps_str}"
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
