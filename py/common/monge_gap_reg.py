"""
Monge gap regularizer via OTT-JAX Sinkhorn (generic, Section 4.1 of Uscidda & Cuturi 2023).

Uses ent_reg_cost (negative-entropy convention) so M_eps >= M_0 >= 0.
OTT-JAX 0.6.0 API: relative_epsilon='mean' (not True).
"""

import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def sinkhorn_ot_cost(x: jnp.ndarray, y: jnp.ndarray, eps_rel: float, max_iter: int) -> jnp.ndarray:
    """
    Regularized OT cost W_{c,eps}(x, y) using negative-entropy Sinkhorn.

    Uses relative_epsilon='mean': actual epsilon = eps_rel * mean_cost_matrix.
    Returns ent_reg_cost = <C, P*> - eps * H(P*), which satisfies W_{c,eps} <= W_c.
    This guarantees monge_gap = E[c(x,T(x))] - W_{c,eps} >= 0.
    """
    geom = pointcloud.PointCloud(x, y, epsilon=eps_rel, relative_epsilon="mean")
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(max_iterations=max_iter, threshold=1e-3)
    out = solver(prob)
    return out.ent_reg_cost


def monge_gap(
    I_s: jnp.ndarray,
    X_st: jnp.ndarray,
    eps_rel: float,
    max_iter: int,
) -> jnp.ndarray:
    """
    Generic Monge gap: M(T) = E[c(x, T(x))] - W_{c,eps}(rho, T_# rho) >= 0.

    I_s:  source interpolant points, shape (mg_batch, d)
    X_st: T(I_s) = X_{s,t}(I_s), shape (mg_batch, d)

    All points share the same (s, t) — required for Sinkhorn to couple them meaningfully.
    Cost c is squared Euclidean.
    """
    transport_cost = jnp.mean(jnp.sum((I_s - X_st) ** 2, axis=-1))
    ot_cost = sinkhorn_ot_cost(I_s, X_st, eps_rel, max_iter)
    return transport_cost - ot_cost
