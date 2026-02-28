"""KKT-based backward differentiation for WEC NLP sensitivity.

Differentiates through the KKT optimality conditions of the full NLP
to compute ``dx*/dp`` and ``dφ*/dp`` without perturbation solves or
active-set stability requirements.

Given a solved NLP with primal ``x*`` and dual ``(λ*, μ*)``:

1. Builds the full-space KKT matrix from the Lagrangian Hessian,
   dynamics Jacobian, and active inequality constraint Jacobians.

2. Solves a single linear system (adjoint KKT) to map an output seed
   ``v`` to the parametric gradient ``v^T dx*/dp``.

3. Chains through :func:`residual_parametric` via ``jax.vjp`` for the
   actual parameter gradient.

This replaces FFO's fragile perturbed-solve backward with a single
factorisation + back-substitution.  No re-solves, no delta-halving,
no active-set stability checks.

The user-facing API is unchanged — this is wired into
:func:`make_differentiable_solver` as ``backward_strategy="kkt"``.
"""

from __future__ import annotations

__all__ = [
    "kkt_vjp",
]

import logging

import numpy as np
import jax
import jax.numpy as jnp
from scipy import linalg as la

jax.config.update("jax_enable_x64", True)

_log = logging.getLogger(__name__)


def kkt_vjp(wec, res, wave, obj_fun, wave_data, params,
            active_tol=1e-6, sign=1.0):
    r"""Build a VJP through the NLP's KKT conditions.

    At the NLP optimum ``(x*, λ*, μ*)``, the KKT conditions are:

    .. math::

        \nabla_x \mathcal{L} = 0, \quad
        r(x; p) = 0, \quad
        g_{\mathcal{A}}(x) = 0

    Implicit differentiation of this system w.r.t. parameters ``p`` gives
    the sensitivity ``dx*/dp``.  For a VJP with seed ``v``:

    .. math::

        v^\top \frac{dx^*}{dp}
        = -w_\lambda^\top \frac{\partial r}{\partial p}

    where ``w`` solves the adjoint KKT system ``K^\top w = [v; 0; 0]``.

    Parameters
    ----------
    wec : WEC_IPOPT
    res : OptimizeResult
        IPOPT solution with ``x``, ``dynamics_mult_g``,
        ``constraint_multipliers``, ``constraint_info``.
    wave : DataArray
        Single-realisation wave.
    obj_fun : callable
        ``obj_fun(wec, x_wec, x_opt, wave) -> scalar``.
    wave_data : WaveData
        Pre-extracted wave data for the parametric residual VJP.
    params : pytree
        BEM parameters (or joint params).
    active_tol : float
        Threshold for identifying active inequality constraints.
    sign : float
        +1.0 for minimisation problems; -1.0 if IPOPT maximised.

    Returns
    -------
    vjp_fn : callable
        ``vjp_fn(v_seed) -> params_grad`` where ``v_seed`` has shape
        ``(n_opt,)`` and ``params_grad`` is a pytree matching *params*.
    x_opt_star : ndarray
    x_wec_star : ndarray
    info : dict
        Diagnostic information (KKT conditioning, active set size, etc.).
    """
    from .parametric import residual_parametric

    x_star = np.array(res.x)
    x_wec, x_opt = wec.decompose_state(x_star)
    n_wec = len(x_wec)
    n_opt = len(x_opt)
    n_x = n_wec + n_opt
    x_j = jnp.array(x_star)
    x_wec_j = jnp.array(x_wec)
    x_opt_j = jnp.array(x_opt)

    lam = np.array(res.dynamics_mult_g)
    ci = res.constraint_info

    # ── 1. Lagrangian Hessian ∇²_xx L ────────────────────────────────
    # L = sign·f(x) + λᵀr(x) + Σ μᵢ gᵢ(x)

    def lagrangian(x):
        xw, xo = x[:n_wec], x[n_wec:]
        L = sign * obj_fun(wec, xw, xo, wave)
        L = L + jnp.dot(jnp.array(lam), wec.residual(xw, xo, wave))
        for i, icons in enumerate(wec.constraints):
            cname = f"user_constraint_{i}"
            mu_i = res.constraint_multipliers.get(
                cname, np.zeros(ci[cname]["size"]))
            g_i = jnp.atleast_1d(icons["fun"](wec, xw, xo, wave))
            L = L + jnp.dot(jnp.array(mu_i), g_i)
        return L

    H_L = np.array(jax.hessian(lagrangian)(x_j))

    # ── 2. Dynamics Jacobian J_r ─────────────────────────────────────
    J_r = np.array(jax.jacobian(
        lambda x: wec.residual(x[:n_wec], x[n_wec:], wave))(x_j))
    n_dyn = J_r.shape[0]

    # ── 3. Active inequality constraint Jacobians ────────────────────
    active_rows = []
    n_active_ineq = 0

    for i, icons in enumerate(wec.constraints):
        cname = f"user_constraint_{i}"
        cinfo = ci[cname]
        g_vals = res.constraint_values[cinfo["slice"]]

        if cinfo["type"] != "ineq":
            # Equality constraints — always active, already in dynamics
            # (user eq constraints are separate from dynamics)
            J_eq = np.array(jax.jacobian(
                lambda x, _fn=icons["fun"]: jnp.atleast_1d(
                    _fn(wec, x[:n_wec], x[n_wec:], wave)))(x_j))
            for j in range(len(g_vals)):
                active_rows.append(J_eq[j])
            continue

        J_ineq = np.array(jax.jacobian(
            lambda x, _fn=icons["fun"]: jnp.atleast_1d(
                _fn(wec, x[:n_wec], x[n_wec:], wave)))(x_j))

        for j in range(len(g_vals)):
            if abs(g_vals[j]) < active_tol:
                active_rows.append(J_ineq[j])
                n_active_ineq += 1

    n_active = len(active_rows)
    if n_active > 0:
        J_active = np.array(active_rows)
    else:
        J_active = np.zeros((0, n_x))

    _log.info(
        "KKT backward: n_x=%d, n_dyn=%d, n_active_ineq=%d, n_active_total=%d",
        n_x, n_dyn, n_active_ineq, n_active)

    # ── 4. Assemble full KKT matrix ──────────────────────────────────
    #
    #  [ H_L    J_r^T    J_A^T ] [ dx  ]       [ ∂(∇L)/∂p ]
    #  [ J_r      0        0   ] [ dλ  ] = -   [ ∂r/∂p     ]
    #  [ J_A      0        0   ] [ dμ_A]       [ ∂g_A/∂p   ]

    kkt_size = n_x + n_dyn + n_active
    KKT = np.zeros((kkt_size, kkt_size))

    KKT[:n_x, :n_x] = H_L
    KKT[:n_x, n_x:n_x+n_dyn] = J_r.T
    KKT[n_x:n_x+n_dyn, :n_x] = J_r
    if n_active > 0:
        KKT[:n_x, n_x+n_dyn:] = J_active.T
        KKT[n_x+n_dyn:, :n_x] = J_active

    sv = la.svdvals(KKT)
    kkt_cond = sv[0] / sv[-1] if sv[-1] > 1e-30 else float('inf')
    _log.info("KKT condition number: %.2e", kkt_cond)

    if kkt_cond > 1e10:
        reg = 1e-10 * sv[0]
        _log.info("Regularising KKT system (reg=%.2e) to improve conditioning.", reg)
        KKT[:n_x, :n_x] += reg * np.eye(n_x)
        KKT[n_x:, n_x:] -= reg * np.eye(kkt_size - n_x)
        sv2 = la.svdvals(KKT)
        kkt_cond = sv2[0] / sv2[-1] if sv2[-1] > 1e-30 else float('inf')
        _log.info("KKT condition after regularisation: %.2e", kkt_cond)

    if kkt_cond > 1e14:
        _log.warning(
            "KKT system severely ill-conditioned (%.2e). "
            "Sensitivity may be inaccurate.", kkt_cond)

    # Factor once for repeated VJPs
    kkt_lu = la.lu_factor(KKT)

    # ── 5. Build VJP closure ─────────────────────────────────────────

    def vjp_fn(v_seed):
        """Compute v^T (dx*/dp) given seed v of shape (n_x,) = (n_wec + n_opt,).

        The seed covers the full state vector ``[x_wec; x_opt]``.
        Returns a pytree matching *params*.
        """
        v_np = np.asarray(v_seed, dtype=np.float64)
        if len(v_np) == n_opt:
            v_full = np.zeros(n_x)
            v_full[n_wec:] = v_np
        else:
            v_full = v_np

        # Adjoint RHS: [v_full; 0; 0]
        rhs = np.zeros(kkt_size)
        rhs[:n_x] = v_full

        # Solve adjoint: KKT^T w = rhs
        w = la.lu_solve(kkt_lu, rhs, trans=1)

        # Extract adjoint multiplier for dynamics
        w_lam = jnp.array(w[n_x:n_x+n_dyn])

        # grad_p = -w_λ^T ∂r/∂p  via jax.vjp through parametric residual
        def r_of_p(p):
            return residual_parametric(x_wec_j, x_opt_j, wave_data, p, wec)

        _, vjp_r = jax.vjp(r_of_p, params)
        (grad_p,) = vjp_r(-w_lam)

        from .solver_ipopt import _fix_complex_grad
        return _fix_complex_grad(grad_p)

    info = {
        "kkt_size": kkt_size,
        "kkt_cond": kkt_cond,
        "n_active_ineq": n_active_ineq,
        "n_active_total": n_active,
        "n_dyn": n_dyn,
    }

    return vjp_fn, np.array(x_opt), np.array(x_wec), info
