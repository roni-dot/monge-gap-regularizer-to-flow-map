"""
Sanity checks for the Monge gap regularizer.
Run with: python py/tests/test_monge_gap.py

TensorFlow is mocked because TF's Windows DLLs may be unavailable in dev environments;
training runs on Linux where TF works normally.
"""

import os, sys, unittest.mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock TF before any import that transitively pulls it in
_tf_mock = unittest.mock.MagicMock()
_tf_mock.data.Dataset = object  # StaticArgs type hint
sys.modules.setdefault("tensorflow", _tf_mock)
sys.modules.setdefault("tensorflow_datasets", unittest.mock.MagicMock())

import jax
import jax.numpy as jnp
import ml_collections


# ── helpers ──────────────────────────────────────────────────────────────────

def make_dummy_cfg(lambda_reg: float = 0.0):
    cfg = ml_collections.ConfigDict()
    cfg.training = ml_collections.ConfigDict()
    cfg.training.loss_type = "lsd"
    cfg.training.stopgrad_type = "convex"
    cfg.training.psd_type = None
    cfg.training.tmin = 0.0
    cfg.training.tmax = 1.0
    cfg.training.seed = 42
    cfg.training.ema_facs = [0.999]
    cfg.training.ndevices = 1
    cfg.training.lambda_reg = lambda_reg
    cfg.training.sinkhorn_eps = 0.05
    cfg.training.sinkhorn_max_iter = 200
    cfg.training.mg_batch = 64
    cfg.training.mg_min_gap = 0.0

    cfg.optimization = ml_collections.ConfigDict()
    cfg.optimization.bs = 128
    cfg.optimization.diag_fraction = 0.75

    cfg.problem = ml_collections.ConfigDict()
    cfg.problem.d = 2
    cfg.problem.interp_type = "linear"

    cfg.network = ml_collections.ConfigDict()
    cfg.network.network_type = "mlp"
    cfg.network.n_hidden = 2
    cfg.network.n_neurons = 64
    cfg.network.output_dim = 2
    cfg.network.act = "gelu"
    cfg.network.use_residual = False
    cfg.network.use_weight = False
    cfg.network.use_bfloat16 = False
    cfg.network.rescale = 0.5
    cfg.network.load_path = ""
    cfg.network.input_dims = (2,)
    cfg.network.load_ema_fac = None
    cfg.network.img_resolution = None
    cfg.network.img_channels = None
    cfg.network.label_dim = None
    cfg.network.logvar_channels = None
    cfg.network.reset_optimizer = True
    cfg.network.unet_kwargs = None

    cfg.logging = ml_collections.ConfigDict()
    return cfg


def _make_loss_inputs(cfg, key):
    bs = cfg.optimization.bs
    mg_batch = cfg.training.mg_batch
    key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
    x0 = jax.random.normal(k1, (bs, 2)) * 0.5
    x1 = jax.random.normal(k2, (bs, 2))
    mg_x0 = jax.random.normal(k3, (mg_batch, 2)) * 0.5
    mg_x1 = jax.random.normal(k4, (mg_batch, 2))
    s = jax.random.uniform(k5, (bs,))
    t = jax.random.uniform(k6, (bs,))
    t = jnp.maximum(t, s)  # ensure t >= s
    u = None
    h = None
    dropout_keys = jax.random.split(key, bs).reshape((bs, -1))
    mg_s = jnp.array(0.3)
    mg_t = jnp.array(0.7)
    return x0, x1, mg_x0, mg_x1, s, t, u, h, dropout_keys, mg_s, mg_t


# ── Test 1: Monge gap >= 0 for identity and random maps ───────────────────

def test_monge_gap_nonnegative():
    from common.monge_gap_reg import monge_gap

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (64, 2))
    y = jax.random.normal(jax.random.PRNGKey(1), (64, 2))

    gap_id = float(monge_gap(x, x, 0.05, 200))
    gap_rnd = float(monge_gap(x, y, 0.05, 200))

    print(f"  identity gap = {gap_id:.4f}  (expect > 0)")
    print(f"  random map gap = {gap_rnd:.4f}  (expect > 0)")
    assert gap_id >= 0, f"Identity gap should be >= 0, got {gap_id}"
    assert gap_rnd >= 0, f"Random map gap should be >= 0, got {gap_rnd}"
    assert gap_rnd > gap_id, "Random map gap should exceed identity gap"
    print("  PASSED")


# ── Test 2: Gradient flows through Monge gap ────────────────────────────

def test_monge_gap_gradient():
    from common.monge_gap_reg import monge_gap

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (64, 2))
    y = jax.random.normal(jax.random.PRNGKey(1), (64, 2))

    val, g = jax.value_and_grad(lambda y_: monge_gap(x, y_, 0.05, 200))(y)
    print(f"  gap = {float(val):.4f}, grad norm = {float(jnp.linalg.norm(g)):.4f}")
    assert g.shape == y.shape
    assert jnp.isfinite(g).all(), "Gradient contains NaN or Inf"
    print("  PASSED")


# ── Test 3: lambda_reg=0 makes loss return mg_val=0, total==lsd ─────────

def test_lambda_zero_noop():
    from common import flow_map, interpolant, losses

    cfg = ml_collections.FrozenConfigDict(make_dummy_cfg(lambda_reg=0.0))
    key = jax.random.PRNGKey(42)

    ex_input = jnp.zeros((2,))
    net, params, key = flow_map.initialize_flow_map(cfg.network, ex_input, key)
    interp = interpolant.setup_interpolant(cfg)
    loss_fn = losses.setup_loss(cfg, net, interp)

    x0, x1, mg_x0, mg_x1, s, t, u, h, dropout_keys, mg_s, mg_t = _make_loss_inputs(cfg, key)

    total_loss, aux = loss_fn(params, params, x0, x1, None, s, t, u, h, dropout_keys,
                               mg_x0, mg_x1, mg_s, mg_t)
    mg_val = float(aux["monge_gap"])
    lsd_val = float(aux["lsd_loss"])

    print(f"  total={float(total_loss):.4f}, lsd={lsd_val:.4f}, mg={mg_val:.4f}")
    assert mg_val == 0.0, f"lambda_reg=0 should give mg_val=0, got {mg_val}"
    assert abs(float(total_loss) - lsd_val) < 1e-6, \
        f"total_loss should equal lsd_loss when lambda_reg=0, diff={abs(float(total_loss)-lsd_val)}"
    print("  PASSED (lambda_reg=0 is a true no-op)")


# ── Test 4: lambda_reg>0 adds Monge gap to the loss ─────────────────────

def test_lambda_nonzero_adds_mg():
    from common import flow_map, interpolant, losses

    cfg = ml_collections.FrozenConfigDict(make_dummy_cfg(lambda_reg=1.0))
    key = jax.random.PRNGKey(42)

    ex_input = jnp.zeros((2,))
    net, params, key = flow_map.initialize_flow_map(cfg.network, ex_input, key)
    interp = interpolant.setup_interpolant(cfg)
    loss_fn = losses.setup_loss(cfg, net, interp)

    x0, x1, mg_x0, mg_x1, s, t, u, h, dropout_keys, mg_s, mg_t = _make_loss_inputs(cfg, key)

    total_loss, aux = loss_fn(params, params, x0, x1, None, s, t, u, h, dropout_keys,
                               mg_x0, mg_x1, mg_s, mg_t)
    mg_val = float(aux["monge_gap"])
    lsd_val = float(aux["lsd_loss"])

    print(f"  total={float(total_loss):.4f}, lsd={lsd_val:.4f}, mg={mg_val:.4f}")
    assert mg_val >= 0, f"Monge gap should be >= 0, got {mg_val}"
    assert abs(float(total_loss) - (lsd_val + 1.0 * mg_val)) < 1e-4, \
        f"total != lsd + lambda*mg: {float(total_loss)} vs {lsd_val + mg_val}"
    print("  PASSED (lambda*mg_val added to loss)")


# ── Test 5: Gradient flows through total loss w.r.t. params ─────────────

def test_loss_gradient_wrt_params():
    from common import flow_map, interpolant, losses

    cfg = ml_collections.FrozenConfigDict(make_dummy_cfg(lambda_reg=1.0))
    key = jax.random.PRNGKey(42)

    ex_input = jnp.zeros((2,))
    net, params, key = flow_map.initialize_flow_map(cfg.network, ex_input, key)
    interp = interpolant.setup_interpolant(cfg)
    loss_fn = losses.setup_loss(cfg, net, interp)

    x0, x1, mg_x0, mg_x1, s, t, u, h, dropout_keys, mg_s, mg_t = _make_loss_inputs(cfg, key)

    def scalar_loss(p):
        total, _ = loss_fn(p, p, x0, x1, None, s, t, u, h, dropout_keys,
                           mg_x0, mg_x1, mg_s, mg_t)
        return total

    val, grads = jax.value_and_grad(scalar_loss)(params)
    # Check grads are finite and non-zero
    flat_grads = jax.flatten_util.ravel_pytree(grads)[0]
    print(f"  loss={float(val):.4f}, grad norm={float(jnp.linalg.norm(flat_grads)):.4f}")
    assert jnp.isfinite(flat_grads).all(), "Gradient contains NaN/Inf"
    assert jnp.linalg.norm(flat_grads) > 0, "Gradient is all zeros"
    print("  PASSED (gradient flows through network params)")


# ── Test 6: Larger sinkhorn_eps changes cost ────────────────────────────

def test_sinkhorn_eps_affects_cost():
    from common.monge_gap_reg import sinkhorn_ot_cost

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (64, 2))
    y = jax.random.normal(jax.random.PRNGKey(1), (64, 2))

    cost_small = float(sinkhorn_ot_cost(x, y, 0.01, 500))
    cost_large = float(sinkhorn_ot_cost(x, y, 0.5, 500))
    print(f"  ent_reg_cost(eps=0.01)={cost_small:.4f}  ent_reg_cost(eps=0.5)={cost_large:.4f}")
    # Larger eps → more entropy subtracted → smaller ent_reg_cost
    assert cost_small > cost_large, \
        f"Larger eps should give smaller ent_reg_cost: {cost_small:.4f} vs {cost_large:.4f}"
    print("  PASSED (larger eps -> smaller ent_reg_cost)")


if __name__ == "__main__":
    print("=" * 60)
    print("Monge gap sanity checks")
    print("=" * 60)

    print("\n[1] Monge gap non-negativity (identity and random maps)")
    test_monge_gap_nonnegative()

    print("\n[2] Gradient flows through Monge gap w.r.t. X_st")
    test_monge_gap_gradient()

    print("\n[3] lambda_reg=0 is a true no-op")
    test_lambda_zero_noop()

    print("\n[4] lambda_reg>0 adds mg to loss")
    test_lambda_nonzero_adds_mg()

    print("\n[5] Gradient flows through total loss w.r.t. params")
    test_loss_gradient_wrt_params()

    print("\n[6] Larger sinkhorn_eps gives smaller ent_reg_cost")
    test_sinkhorn_eps_affects_cost()

    print("\n" + "=" * 60)
    print("All sanity checks PASSED.")
    print("=" * 60)
