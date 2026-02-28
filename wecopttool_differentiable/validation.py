"""Validation utilities for post-optimality sensitivity analysis.

Finite-difference validation (three levels):

1. :func:`fd_validate` — **end-to-end**: re-solve the NLP at perturbed
   parameters and compare ``(phi*(p+eps) - phi*(p-eps)) / 2eps`` against
   the analytical Fiacco gradient.  The user supplies a ``re_solve_fn``.

2. :func:`fd_check_residual` — verify ``lambda^T (dr/dp)`` against FD
   of the residual.  No re-solve needed.

3. :func:`fd_check_objective` — verify ``df/dp`` against FD of the
   objective.  No re-solve needed.

Cross-method consistency (solver-independent):

4. :func:`cross_check_fiacco_kkt` — verify Fiacco and KKT agree via the
   total-derivative identity
   ``d phi*/dp = df/dp + (df/dx) @ (dx*/dp)``.
   No finite-difference re-solve needed.

NLP regularity checks (required for Fiacco theorem validity):

5. :func:`check_regularity` — verify LICQ, strict complementarity,
   SOSC, and active-set stability at the optimal solution.
"""

from __future__ import annotations

__all__ = [
    "fd_validate",
    "fd_check_residual",
    "fd_check_objective",
    "cross_check_fiacco_kkt",
    "make_re_solve_fn",
    "validate_sensitivity",
    "check_regularity",
]

import logging
from typing import Callable, Dict, Optional, Sequence, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy import linalg as la

_log = logging.getLogger(__name__)


class FDResult(NamedTuple):
    """Result for a single parameter from finite-difference check."""
    name: str
    analytical: float
    fd: float
    rel_error: float
    passed: bool


class CrossCheckResult(NamedTuple):
    """Per-parameter result from Fiacco vs KKT cross-consistency check.

    Attributes
    ----------
    name : str
        Parameter field name.
    fiacco : float
        ``d phi*/dp`` from Fiacco (envelope theorem).
    kkt_chain : float
        ``df/dp + (df/dx) @ (dx*/dp)`` reconstructed via KKT.
    rel_error : float
        Relative error between the two.
    passed : bool
        Whether rel_error < tolerance.
    """
    name: str
    fiacco: float
    kkt_chain: float
    rel_error: float
    passed: bool


def fd_validate(
    analytical_grad,
    params,
    re_solve_fn: Callable[[dict], float],
    *,
    fields: Optional[Sequence[str]] = None,
    eps_scale: float = 1e-4,
    tol: float = 0.10,
    verbose: bool = True,
) -> Dict[str, FDResult]:
    """End-to-end FD validation of analytical sensitivity.

    Perturbs each parameter, re-solves the NLP via *re_solve_fn*, and
    compares central finite differences against *analytical_grad*.

    Parameters
    ----------
    analytical_grad : namedtuple / pytree
        Gradient returned by :func:`sensitivity`.
    params : namedtuple
        Nominal parameter values (same structure as *analytical_grad*).
    re_solve_fn : callable
        ``re_solve_fn(params_dict) -> float`` — re-solves the NLP with
        the given parameter dict and returns the optimal objective value.
        Must accept a plain ``dict`` keyed by field name.
    fields : sequence of str, optional
        Which fields to check.  Defaults to all ``params._fields``.
    eps_scale : float
        Relative perturbation: ``eps = eps_scale * |val|`` for each
        parameter.  Falls back to ``eps_scale`` when ``|val| < 1``.
    tol : float
        Relative error threshold for pass/fail (default 10%).
    verbose : bool
        Print a comparison table.

    Returns
    -------
    dict[str, FDResult]
        Per-parameter results with analytical, fd, rel_error, and pass/fail.
    """
    if fields is None:
        fields = params._fields

    nominal = {f: float(getattr(params, f)) for f in fields}
    results = {}

    for field in fields:
        val = nominal[field]
        eps = eps_scale * abs(val) if abs(val) > 1.0 else eps_scale

        d_plus = dict(nominal)
        d_plus[field] = val + eps
        phi_plus = re_solve_fn(d_plus)

        d_minus = dict(nominal)
        d_minus[field] = val - eps
        phi_minus = re_solve_fn(d_minus)

        fd_grad = (phi_plus - phi_minus) / (2 * eps)
        a_grad = float(getattr(analytical_grad, field))

        denom = max(abs(a_grad), abs(fd_grad), 1e-20)
        rel_err = abs(a_grad - fd_grad) / denom
        passed = rel_err < tol

        results[field] = FDResult(
            name=field,
            analytical=a_grad,
            fd=fd_grad,
            rel_error=rel_err,
            passed=passed,
        )

    if verbose:
        _print_table(results, tol)

    return results


def fd_check_residual(
    wec,
    res,
    waves,
    params,
    *,
    parametric_forces=None,
    additional_forces=None,
    fields: Optional[Sequence[str]] = None,
    eps_scale: float = 1e-5,
    tol: float = 0.05,
    verbose: bool = True,
) -> Dict[str, FDResult]:
    r"""Check :math:`\lambda^\top \partial r/\partial p` via FD of the residual.

    Evaluates the residual at ``p +/- eps`` and compares the numerical
    ``lambda^T (r(p+eps) - r(p-eps)) / 2eps`` against the JAX VJP result.

    No NLP re-solve is needed — this runs at the fixed optimal point
    ``(x_wec*, x_opt*)`` from *res*.

    Parameters
    ----------
    wec : WEC_IPOPT
        WEC instance with ``_hydro_data``.
    res : OptimizeResult
        Single result from :meth:`wec.solve`.
    waves : xarray.DataArray
        Wave data used in the solve.
    params : namedtuple
        Nominal parameter values.
    parametric_forces : dict, optional
        Same as passed to :func:`sensitivity`.
    additional_forces : dict, optional
        Same as passed to :func:`sensitivity`.
    fields, eps_scale, tol, verbose
        Same as :func:`fd_validate`.

    Returns
    -------
    dict[str, FDResult]
    """
    from .parametric import extract_wave_data, residual_parametric
    from .solver_ipopt import _extract_all_realizations

    if fields is None:
        fields = params._fields

    x_wec, x_opt = wec.decompose_state(res.x)
    lam = jnp.array(res.dynamics_mult_g)
    x_wec_jax = jnp.array(x_wec)
    x_opt_jax = jnp.array(x_opt)

    wave_data_list, wave_list = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    wd = wave_data_list[0]
    wave_i = wave_list[0]

    def _wrap_force(f, wave):
        return lambda wec, xw, xo, wd_: f(wec, xw, xo, wave)

    add_wrapped = None
    if additional_forces is not None:
        param_keys = set(
            parametric_forces.keys()) if parametric_forces else set()
        add_wrapped = {
            k: _wrap_force(f, wave_i)
            for k, f in additional_forces.items()
            if k not in param_keys
        }
        if not add_wrapped:
            add_wrapped = None

    def r_of_p(p):
        return residual_parametric(
            x_wec_jax, x_opt_jax, wd, p, wec,
            additional_forces=add_wrapped,
            parametric_forces=parametric_forces,
        )

    _, vjp_fn = jax.vjp(r_of_p, params)
    (grad_analytical,) = vjp_fn(lam)

    results = {}
    r_nom = np.array(r_of_p(params))

    for field in fields:
        val = float(getattr(params, field))
        eps = eps_scale * max(abs(val), 1.0)

        p_plus = params._replace(**{field: val + eps})
        p_minus = params._replace(**{field: val - eps})

        r_plus = np.array(r_of_p(p_plus))
        r_minus = np.array(r_of_p(p_minus))
        dr_dp_fd = (r_plus - r_minus) / (2 * eps)

        lam_np = np.array(lam)
        fd_val = float(lam_np @ dr_dp_fd)
        a_val = float(getattr(grad_analytical, field))

        denom = max(abs(a_val), abs(fd_val), 1e-20)
        rel_err = abs(a_val - fd_val) / denom
        passed = rel_err < tol

        results[field] = FDResult(
            name=field,
            analytical=a_val,
            fd=fd_val,
            rel_error=rel_err,
            passed=passed,
        )

    if verbose:
        print("Residual Jacobian check: lambda^T (dr/dp)")
        _print_table(results, tol)

    return results


def fd_check_objective(
    wec,
    res,
    waves,
    params,
    obj_fn: Callable,
    *,
    fields: Optional[Sequence[str]] = None,
    eps_scale: float = 1e-5,
    tol: float = 0.05,
    verbose: bool = True,
) -> Dict[str, FDResult]:
    r"""Check :math:`\partial f/\partial p` via FD of the objective.

    Evaluates the objective at ``p +/- eps`` (at the fixed optimal point)
    and compares numerical gradient against ``jax.grad``.

    Parameters
    ----------
    wec : WEC_IPOPT
        WEC instance.
    res : OptimizeResult
        Single result from :meth:`wec.solve`.
    waves : xarray.DataArray
        Wave data used in the solve.
    params : namedtuple
        Nominal parameter values.
    obj_fn : callable
        Parametric objective: ``obj_fn(wec, x_wec, x_opt, wave, params) -> scalar``.
    fields, eps_scale, tol, verbose
        Same as :func:`fd_validate`.

    Returns
    -------
    dict[str, FDResult]
    """
    from .solver_ipopt import _extract_all_realizations

    if fields is None:
        fields = params._fields

    x_wec, x_opt = wec.decompose_state(res.x)
    x_wec_jax = jnp.array(x_wec)
    x_opt_jax = jnp.array(x_opt)

    _, wave_list = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    wave_i = wave_list[0]

    def f_of_p(p):
        return obj_fn(wec, x_wec_jax, x_opt_jax, wave_i, p)

    grad_analytical = jax.grad(f_of_p)(params)

    results = {}

    for field in fields:
        val = float(getattr(params, field))
        eps = eps_scale * max(abs(val), 1.0)

        p_plus = params._replace(**{field: val + eps})
        p_minus = params._replace(**{field: val - eps})

        f_plus = float(f_of_p(p_plus))
        f_minus = float(f_of_p(p_minus))
        fd_val = (f_plus - f_minus) / (2 * eps)
        a_val = float(getattr(grad_analytical, field))

        denom = max(abs(a_val), abs(fd_val), 1e-20)
        rel_err = abs(a_val - fd_val) / denom
        passed = rel_err < tol

        results[field] = FDResult(
            name=field,
            analytical=a_val,
            fd=fd_val,
            rel_error=rel_err,
            passed=passed,
        )

    if verbose:
        print("Objective gradient check: df/dp")
        _print_table(results, tol)

    return results


def cross_check_fiacco_kkt(
    wec,
    res,
    waves,
    obj_fun,
    nstate_opt: int,
    fiacco_grad,
    *,
    params=None,
    parametric_forces=None,
    obj_fn=None,
    fields: Optional[Sequence[str]] = None,
    tol: float = 0.05,
    active_tol: float = 1e-6,
    verbose: bool = True,
) -> Dict[str, CrossCheckResult]:
    r"""Cross-consistency check between Fiacco and KKT -- no FD re-solve.

    Exploits the total-derivative identity:

    .. math::

        \frac{d\varphi^*}{dp}
        = \frac{\partial f}{\partial p}
          + \frac{\partial f}{\partial x}\,\frac{dx^*}{dp}

    The left-hand side comes from Fiacco (``fiacco_grad``).  The
    right-hand side is reconstructed by:

    1. Computing :math:`\partial f / \partial x` at the optimum (JAX grad).
    2. Using :func:`kkt_vjp` with seed :math:`v = \nabla_x f` to get
       :math:`v^\top dx^*/dp`.
    3. Adding :math:`\partial f / \partial p` (JAX grad).

    If both sides match, it is a **solver-independent proof** that
    Fiacco and KKT are mutually consistent.

    Parameters
    ----------
    wec : WEC_IPOPT
        WEC instance.
    res : OptimizeResult
        Single result from :meth:`wec.solve`.
    waves : xarray.DataArray
        Wave data used in the solve.
    obj_fun : callable
        NLP objective ``obj_fun(wec, x_wec, x_opt, wave) -> scalar``.
    nstate_opt : int
        Number of optimisation state variables.
    fiacco_grad : namedtuple / pytree
        Gradient from :func:`sensitivity`.
    params : namedtuple, optional
        Parameter pytree.  Defaults to BEM-only.
    parametric_forces : dict, optional
        Parametric force functions (same as passed to :func:`sensitivity`).
    obj_fn : callable, optional
        Parametric objective ``obj_fn(wec, x_wec, x_opt, wave, params)``.
    fields : sequence of str, optional
        Which parameter fields to compare.  Defaults to all.
    tol : float
        Relative error threshold for pass/fail.
    active_tol : float
        Active constraint tolerance for the KKT system.
    verbose : bool
        Print a comparison table.

    Returns
    -------
    dict[str, CrossCheckResult]
        Per-parameter results.
    """
    from .parametric import extract_bem_params, extract_wave_data
    from .solver_ipopt import _extract_all_realizations
    from .qp_kkt import kkt_vjp

    if params is None:
        params = extract_bem_params(wec._hydro_data)

    if fields is None:
        fields = params._fields

    # ---- 1. Compute df/dx at the optimum --------------------------------
    _, wave_list = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    wave_0 = wave_list[0]

    x_wec, x_opt = wec.decompose_state(res.x)
    x_wec_j = jnp.array(x_wec)
    x_opt_j = jnp.array(x_opt)

    def f_of_x(x_full):
        nw = len(x_wec)
        return obj_fun(wec, x_full[:nw], x_full[nw:], wave_0)

    x_star = jnp.concatenate([x_wec_j, x_opt_j])
    df_dx = jax.grad(f_of_x)(x_star)
    seed = np.asarray(df_dx, dtype=np.float64)

    # ---- 2. KKT VJP with seed = df/dx --> (df/dx)^T (dx*/dp) -----------
    wave_data_list, _ = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    wd_0 = wave_data_list[0]

    vjp_fn, _, _, info = kkt_vjp(
        wec, res, wave_0, obj_fun, wd_0, params,
        active_tol=active_tol, sign=1.0)

    kkt_grad = vjp_fn(seed)

    if verbose:
        _log.info("KKT cross-check: cond=%.2e, n_active=%d",
                  info["kkt_cond"], info["n_active_ineq"])

    # ---- 3. Compute df/dp at the optimum --------------------------------
    if obj_fn is not None:
        def f_of_p(p):
            return obj_fn(wec, x_wec_j, x_opt_j, wave_0, p)
        df_dp = jax.grad(f_of_p)(params)
    else:
        df_dp = jax.tree_util.tree_map(jnp.zeros_like, params)

    # ---- 4. Reconstruct: d phi*/dp = df/dp + (df/dx)^T (dx*/dp) --------
    kkt_total = jax.tree_util.tree_map(jnp.add, df_dp, kkt_grad)

    # ---- 5. Compare against Fiacco --------------------------------------
    results = {}
    for field in fields:
        fiacco_arr = jnp.asarray(getattr(fiacco_grad, field))
        kkt_arr = jnp.asarray(getattr(kkt_total, field))
        fiacco_val = float(jnp.real(jnp.sum(fiacco_arr)))
        kkt_val = float(jnp.real(jnp.sum(kkt_arr)))

        denom = max(abs(fiacco_val), abs(kkt_val), 1e-20)
        rel_err = abs(fiacco_val - kkt_val) / denom
        passed = rel_err < tol

        results[field] = CrossCheckResult(
            name=field,
            fiacco=fiacco_val,
            kkt_chain=kkt_val,
            rel_error=rel_err,
            passed=passed,
        )

    if verbose:
        _print_cross_table(results, tol)

    return results


def _print_cross_table(results: Dict[str, CrossCheckResult], tol: float):
    """Pretty-print a Fiacco vs KKT cross-check table."""
    hdr = (f"  {'Parameter':25s} {'Fiacco':>14s} "
           f"{'KKT chain':>14s} {'Rel Error':>12s}")
    sep = "  " + "-" * 70
    print(hdr)
    print(sep)
    all_passed = True
    for r in results.values():
        status = "OK" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        print(f"  {r.name:25s} {r.fiacco:14.4e} {r.kkt_chain:14.4e} "
              f"{r.rel_error:12.2e}  {status}")
    print(sep)
    if all_passed:
        print(f"  All parameters passed (tol={tol:.0%})")
    else:
        n_fail = sum(1 for r in results.values() if not r.passed)
        print(f"  {n_fail} parameter(s) FAILED (tol={tol:.0%})")
    print()


def _print_table(results: Dict[str, FDResult], tol: float):
    """Pretty-print a validation comparison table."""
    hdr = f"  {'Parameter':25s} {'Analytical':>14s} {'FD':>14s} {'Rel Error':>12s}"
    sep = "  " + "-" * 70
    print(hdr)
    print(sep)
    all_passed = True
    for r in results.values():
        status = "OK" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        print(f"  {r.name:25s} {r.analytical:14.4e} {r.fd:14.4e} "
              f"{r.rel_error:12.2e}  {status}")
    print(sep)
    if all_passed:
        print(f"  All parameters passed (tol={tol:.0%})")
    else:
        n_fail = sum(1 for r in results.values() if not r.passed)
        print(f"  {n_fail} parameter(s) FAILED (tol={tol:.0%})")
    print()


def make_re_solve_fn(wec_factory, waves, obj_fun, nstate_opt,
                     res=None, **solve_kwargs):
    """Build a ``re_solve_fn`` for use with :func:`fd_validate`.

    Wraps the boilerplate of rebuilding a WEC and re-solving the NLP
    at perturbed parameters into a single callable.

    Parameters
    ----------
    wec_factory : callable
        ``wec_factory(params_dict) -> WEC_IPOPT`` — creates a new
        WEC instance from a parameter dict (keyed by field name).
    waves : xarray.DataArray
        Wave data.
    obj_fun : callable
        Objective function passed to :meth:`WEC_IPOPT.solve`.
    nstate_opt : int
        Number of optimisation state variables.
    res : OptimizeResult, optional
        Baseline result for warm-starting the perturbed solve.
        When provided, both primal (``x_wec_0``, ``x_opt_0``) and
        dual variables (``mult_g_0``, ``mult_x_L_0``, ``mult_x_U_0``)
        are passed, helping the solver converge to the same local
        optimum.
    **solve_kwargs
        Extra keyword arguments forwarded to :meth:`WEC_IPOPT.solve`
        (e.g. ``scale_x_wec``, ``optim_options``).

    Returns
    -------
    callable
        ``re_solve_fn(params_dict) -> float`` suitable for
        :func:`fd_validate`.
    """
    def re_solve(params_dict):
        wec_new = wec_factory(params_dict)
        kw = dict(solve_kwargs)
        if res is not None:
            x_w, x_o = wec_new.decompose_state(res.x)
            kw.setdefault("x_wec_0", x_w)
            kw.setdefault("x_opt_0", x_o)
            kw.setdefault("mult_g_0", res.mult_g)
            kw.setdefault("mult_x_L_0", res.mult_x_L)
            kw.setdefault("mult_x_U_0", res.mult_x_U)
        results = wec_new.solve(waves, obj_fun, nstate_opt, **kw)
        return float(results[0].fun)
    return re_solve


def validate_sensitivity(
    wec,
    results,
    waves,
    analytical_grad,
    params=None,
    *,
    parametric_forces=None,
    additional_forces=None,
    obj_fn=None,
    re_solve_fn=None,
    obj_fun=None,
    nstate_opt=None,
    cross_check_tol: float = 0.05,
    fields=None,
    residual_tol: float = 0.05,
    objective_tol: float = 0.05,
    resolve_tol: float = 0.10,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run all applicable validation checks and return a summary.

    Automatically selects which checks to run based on the arguments
    provided:

    - **Residual check** -- runs when *params* and *parametric_forces*
      are given (validates :math:`\\lambda^T \\partial r / \\partial p`).
    - **Objective check** -- runs when *params* and *obj_fn* are given
      (validates :math:`\\partial f / \\partial p`).
    - **Cross-consistency** -- runs when *obj_fun* and *nstate_opt* are
      given.  Compares Fiacco's :math:`d\\varphi^*/dp` against the
      chain-rule reconstruction via KKT
      (:math:`\\partial f/\\partial p + (\\partial f/\\partial x) \\cdot
      dx^*/dp`).  **Solver-independent** -- no FD re-solve needed.
    - **Full NLP re-solve** -- runs when *re_solve_fn* is given
      (validates the complete Fiacco gradient via central FD).

    Parameters
    ----------
    wec : WEC_IPOPT
        WEC instance.
    results : OptimizeResult or list[OptimizeResult]
        Result(s) from :meth:`WEC_IPOPT.solve`.
    waves : xarray.DataArray
        Wave data.
    analytical_grad : namedtuple / pytree
        Gradient returned by :func:`sensitivity`.
    params : namedtuple, optional
        Nominal parameter values.
    parametric_forces : dict, optional
        Parametric force dict (same as passed to :func:`sensitivity`).
    additional_forces : dict, optional
        Non-parametric forces.
    obj_fn : callable, optional
        Parametric objective ``obj_fn(wec, x_wec, x_opt, wave, params)``.
    re_solve_fn : callable, optional
        ``re_solve_fn(params_dict) -> float`` for full NLP re-solve
        (see :func:`make_re_solve_fn`).
    obj_fun : callable, optional
        NLP objective ``obj_fun(wec, x_wec, x_opt, wave) -> scalar``.
        Required for cross-consistency check.
    nstate_opt : int, optional
        Number of optimisation state variables.
        Required for cross-consistency check.
    cross_check_tol : float
        Tolerance for the Fiacco vs KKT cross-check.
    fields : sequence of str, optional
        Parameter fields to check (default: all).
    residual_tol, objective_tol, resolve_tol : float
        Tolerance for each FD check level.
    verbose : bool
        Print comparison tables.

    Returns
    -------
    dict[str, dict]
        Nested dict keyed by check name (``'residual'``, ``'objective'``,
        ``'cross_check'``, ``'resolve'``), each containing per-parameter
        :class:`FDResult` or :class:`CrossCheckResult`.
    """
    from scipy.optimize import OptimizeResult

    if isinstance(results, OptimizeResult):
        results = [results]
    res = results[0]

    report: Dict[str, Dict] = {}

    if params is not None and parametric_forces is not None:
        if verbose:
            print("=" * 74)
            print("  Residual Jacobian check: lambda^T (dr/dp)")
            print("=" * 74)
        report["residual"] = fd_check_residual(
            wec, res, waves, params,
            parametric_forces=parametric_forces,
            additional_forces=additional_forces,
            fields=fields,
            tol=residual_tol,
            verbose=verbose,
        )

    if params is not None and obj_fn is not None:
        if verbose:
            print("=" * 74)
            print("  Objective gradient check: df/dp")
            print("=" * 74)
        report["objective"] = fd_check_objective(
            wec, res, waves, params,
            obj_fn=obj_fn,
            fields=fields,
            tol=objective_tol,
            verbose=verbose,
        )

    if obj_fun is not None and nstate_opt is not None:
        if verbose:
            print("=" * 74)
            print("  Cross-consistency: Fiacco vs KKT (chain rule)")
            print("=" * 74)
        report["cross_check"] = cross_check_fiacco_kkt(
            wec, res, waves,
            obj_fun=obj_fun,
            nstate_opt=nstate_opt,
            fiacco_grad=analytical_grad,
            params=params,
            parametric_forces=parametric_forces,
            obj_fn=obj_fn,
            fields=fields,
            tol=cross_check_tol,
            verbose=verbose,
        )

    if re_solve_fn is not None and params is not None:
        if verbose:
            print("=" * 74)
            print("  Full NLP re-solve check: d(phi*)/dp")
            print("=" * 74)
        report["resolve"] = fd_validate(
            analytical_grad, params, re_solve_fn,
            fields=fields,
            tol=resolve_tol,
            verbose=verbose,
        )

    if verbose:
        total = sum(len(v) for v in report.values())
        failed = sum(
            1 for checks in report.values()
            for r in checks.values() if not r.passed
        )
        print("=" * 74)
        if failed == 0:
            print(f"  SUMMARY: All {total} checks passed.")
        else:
            print(f"  SUMMARY: {failed}/{total} checks FAILED.")
        print("=" * 74)
        print()

    return report


# ════════════════════════════════════════════════════════════════════════
# NLP regularity checks for Fiacco sensitivity validity
# ════════════════════════════════════════════════════════════════════════

class RegularityResult(NamedTuple):
    """Result of NLP regularity checks at an optimal solution.

    Attributes
    ----------
    licq : bool
        LICQ holds — active constraint gradients are linearly independent.
    licq_sigma_min : float
        Smallest singular value of the active constraint Jacobian.
        Values near zero indicate near-violation of LICQ.
    strict_complementarity : bool
        Every active inequality constraint has a nonzero multiplier.
    sc_min_active_mult : float
        Smallest |multiplier| among active inequality constraints.
    sc_degenerate : list[str]
        Names of constraints that are active but have near-zero multiplier.
    sosc : bool
        Reduced Hessian of the Lagrangian is positive (semi)definite
        on the null space of the active constraint Jacobian.
    sosc_min_eigenvalue : float
        Smallest eigenvalue of the reduced Hessian.
    active_set_summary : dict
        ``{'n_active_ineq': int, 'n_total_ineq': int,
        'n_dynamics': int, 'active_names': list}``.
    all_passed : bool
        True if LICQ, strict complementarity, and SOSC all hold.
    """
    licq: bool
    licq_sigma_min: float
    strict_complementarity: bool
    sc_min_active_mult: float
    sc_degenerate: list
    sosc: bool
    sosc_min_eigenvalue: float
    active_set_summary: dict
    all_passed: bool


def check_regularity(
    wec,
    res,
    waves,
    obj_fun=None,
    *,
    active_tol: float = 1e-6,
    licq_tol: float = 1e-8,
    sc_tol: float = 1e-8,
    sosc_tol: float = -1e-6,
    verbose: bool = True,
) -> RegularityResult:
    """Check NLP regularity conditions required by Fiacco's theorem.

    Evaluates at the optimal solution ``(x*, lambda*)`` stored in *res*:

    1. **LICQ** — active constraint gradients are linearly independent
       (smallest singular value of active Jacobian > *licq_tol*).
    2. **Strict complementarity** — every active inequality has
       ``|mu_i| > sc_tol``.
    3. **SOSC** — the reduced Hessian of the Lagrangian (projected onto
       the null space of the active constraint Jacobian) is positive
       definite (min eigenvalue > *sosc_tol*).

    Parameters
    ----------
    wec : WEC_IPOPT
        WEC instance (provides ``residual``, ``constraints``, etc.).
    res : OptimizeResult
        Result from :meth:`WEC_IPOPT.solve` (carries multipliers).
    waves : xarray.DataArray
        Wave data used in the solve.
    obj_fun : callable, optional
        Objective function ``obj_fun(wec, x_wec, x_opt, wave) -> scalar``.
        Required for SOSC check. If None, SOSC uses only constraint terms
        (may undercount curvature).
    active_tol : float
        Constraint is "active" if ``|g_i(x*)| < active_tol``
        (for inequality constraints, ``|g_i(x*) - bound| < active_tol``).
    licq_tol : float
        Minimum singular value threshold for LICQ.
    sc_tol : float
        Minimum |multiplier| for strict complementarity.
    sosc_tol : float
        Minimum eigenvalue of the reduced Hessian for SOSC.
        Slightly negative default allows for numerical noise.
    verbose : bool
        Print a detailed report.

    Returns
    -------
    RegularityResult
        Named tuple with per-check results and ``all_passed``.
    """
    from .solver_ipopt import _extract_all_realizations

    _, wave_list = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    wave = wave_list[0]

    x_star = np.array(res.x)
    n = len(x_star)
    x_wec, x_opt = wec.decompose_state(x_star)
    ci = res.constraint_info

    # ── 1. Build full constraint Jacobian and identify active set ─────

    # Dynamics equality constraints (always active)
    dyn_jac_fn = jax.jacobian(
        lambda x: wec.residual(*wec.decompose_state(x), wave))
    J_dyn = np.array(dyn_jac_fn(jnp.array(x_star)))     # (n_dyn, n)
    n_dyn = J_dyn.shape[0]

    # User inequality constraints
    active_ineq_rows = []
    active_ineq_names = []
    active_ineq_mults = []
    n_total_ineq = 0

    for i, icons in enumerate(wec.constraints):
        cname = f"user_constraint_{i}"
        cinfo = ci[cname]
        g_vals = res.constraint_values[cinfo["slice"]]
        mu_vals = res.constraint_multipliers.get(cname, np.zeros_like(g_vals))
        n_total_ineq += len(g_vals)

        jac_fn = jax.jacobian(
            lambda x, _ic=icons: jnp.atleast_1d(
                _ic["fun"](wec, *wec.decompose_state(x), wave)))

        for j in range(len(g_vals)):
            is_ineq = cinfo["type"] == "ineq"
            if is_ineq and abs(g_vals[j]) < active_tol:
                row = np.array(jac_fn(jnp.array(x_star)))[j]
                active_ineq_rows.append(row)
                active_ineq_names.append(f"{cname}[{j}]")
                active_ineq_mults.append(float(mu_vals[j]))

    # Variable bound constraints
    if hasattr(res, 'mult_x_L') and hasattr(res, 'mult_x_U'):
        for k in range(n):
            if abs(res.mult_x_L[k]) > sc_tol:
                row = np.zeros(n)
                row[k] = -1.0  # x_k >= lb  →  -x_k + lb <= 0
                active_ineq_rows.append(row)
                active_ineq_names.append(f"x_lower[{k}]")
                active_ineq_mults.append(float(res.mult_x_L[k]))
            if abs(res.mult_x_U[k]) > sc_tol:
                row = np.zeros(n)
                row[k] = 1.0  # x_k <= ub  →  x_k - ub <= 0
                active_ineq_rows.append(row)
                active_ineq_names.append(f"x_upper[{k}]")
                active_ineq_mults.append(float(res.mult_x_U[k]))

    n_active_ineq = len(active_ineq_rows)

    # Stack active constraint Jacobian: [J_dynamics; J_active_ineq]
    if n_active_ineq > 0:
        J_ineq = np.array(active_ineq_rows)  # (n_active_ineq, n)
        J_active = np.vstack([J_dyn, J_ineq])
    else:
        J_active = J_dyn

    # ── 2. LICQ check ────────────────────────────────────────────────
    if J_active.shape[0] > 0:
        sv = la.svdvals(J_active)
        sigma_min = float(sv[-1]) if len(sv) > 0 else float('inf')
    else:
        sigma_min = float('inf')

    licq_ok = sigma_min > licq_tol

    # ── 3. Strict complementarity ────────────────────────────────────
    sc_degenerate = []
    sc_min_active = float('inf')
    for name, mu in zip(active_ineq_names, active_ineq_mults):
        if abs(mu) < sc_tol:
            sc_degenerate.append(name)
        sc_min_active = min(sc_min_active, abs(mu))

    if n_active_ineq == 0:
        sc_min_active = float('inf')

    sc_ok = len(sc_degenerate) == 0

    # ── 4. SOSC — reduced Hessian on null space of active Jacobian ───
    lag_fn = _build_lagrangian(wec, wave, res, obj_fun=obj_fun)
    H_lag = np.array(jax.hessian(lag_fn)(jnp.array(x_star)))

    if J_active.shape[0] < n:
        Z = la.null_space(J_active)  # (n, n-m) where m = # active constraints
        if Z.shape[1] > 0:
            H_red = Z.T @ H_lag @ Z
            eigvals = la.eigvalsh(H_red)
            min_eig = float(eigvals[0])
        else:
            min_eig = float('inf')
    else:
        min_eig = float('inf')

    sosc_ok = min_eig > sosc_tol

    # ── 5. Assemble result ───────────────────────────────────────────
    active_summary = {
        "n_active_ineq": n_active_ineq,
        "n_total_ineq": n_total_ineq,
        "n_dynamics": n_dyn,
        "n_bound_active": sum(1 for n in active_ineq_names if n.startswith("x_")),
        "active_names": active_ineq_names,
    }

    result = RegularityResult(
        licq=licq_ok,
        licq_sigma_min=sigma_min,
        strict_complementarity=sc_ok,
        sc_min_active_mult=sc_min_active,
        sc_degenerate=sc_degenerate,
        sosc=sosc_ok,
        sosc_min_eigenvalue=min_eig,
        active_set_summary=active_summary,
        all_passed=licq_ok and sc_ok and sosc_ok,
    )

    if verbose:
        _print_regularity(result)

    return result


def _build_lagrangian(wec, wave, res, obj_fun=None):
    """Build the KKT Lagrangian L(x) = f(x) + λ^T h(x) + μ^T g(x)."""
    ci = res.constraint_info
    dyn_slice = ci["dynamics"]["slice"]
    lam_dyn = jnp.array(res.mult_g[dyn_slice])

    ineq_slices = {}
    for cname, cinfo in ci.items():
        if cname == "dynamics":
            continue
        ineq_slices[cname] = (cinfo["slice"],
                              jnp.array(res.mult_g[cinfo["slice"]]))

    def lagrangian(x):
        x_wec, x_opt = wec.decompose_state(x)

        # Objective
        L = jnp.float64(0.0)
        if obj_fun is not None:
            L = L + obj_fun(wec, x_wec, x_opt, wave)

        # Dynamics equality constraints
        r = wec.residual(x_wec, x_opt, wave)
        L = L + jnp.dot(lam_dyn, r)

        # User inequality constraints
        for i, icons in enumerate(wec.constraints):
            cname = f"user_constraint_{i}"
            if cname in ineq_slices:
                sl, mu = ineq_slices[cname]
                g = jnp.atleast_1d(icons["fun"](wec, x_wec, x_opt, wave))
                L = L + jnp.dot(mu, g)

        return L

    return lagrangian


def _print_regularity(r: RegularityResult):
    """Print a formatted regularity report."""
    W = 74
    print("=" * W)
    print("  NLP Regularity Checks (Fiacco Sensitivity Prerequisites)")
    print("=" * W)

    s = r.active_set_summary
    print(f"\n  Active set:")
    print(f"    Dynamics (equality):     {s['n_dynamics']} constraints")
    print(f"    Inequality active:       {s['n_active_ineq']} / "
          f"{s['n_total_ineq']} constraints")
    print(f"    Variable bounds active:  {s['n_bound_active']}")
    if s['active_names']:
        shown = s['active_names'][:10]
        print(f"    Active inequality names: {shown}"
              + (" ..." if len(s['active_names']) > 10 else ""))

    print(f"\n  {'Check':35s} {'Status':8s} {'Value':>14s} {'Threshold':>14s}")
    print("  " + "-" * (W - 4))

    _print_check("LICQ (σ_min of active Jacobian)",
                 r.licq, f"{r.licq_sigma_min:.4e}", f"> {1e-8:.0e}")
    _print_check("Strict complementarity (min |μ|)",
                 r.strict_complementarity,
                 f"{r.sc_min_active_mult:.4e}" if r.sc_min_active_mult < float('inf')
                 else "n/a (no active ineq)",
                 f"> {1e-8:.0e}")
    _print_check("SOSC (min eigenvalue reduced H_L)",
                 r.sosc, f"{r.sosc_min_eigenvalue:.4e}"
                 if r.sosc_min_eigenvalue < float('inf') else "n/a (fully constrained)",
                 f"> {-1e-6:.0e}")

    if r.sc_degenerate:
        print(f"\n  ⚠ Degenerate constraints (active with ~zero multiplier):")
        for name in r.sc_degenerate[:10]:
            print(f"    - {name}")

    print("\n  " + "-" * (W - 4))
    if r.all_passed:
        print("  RESULT: All regularity conditions satisfied — "
              "Fiacco sensitivity is valid.")
    else:
        failures = []
        if not r.licq:
            failures.append("LICQ")
        if not r.strict_complementarity:
            failures.append("Strict Complementarity")
        if not r.sosc:
            failures.append("SOSC")
        print(f"  RESULT: FAILED — {', '.join(failures)}. "
              "Sensitivity gradients may be unreliable.")
    print("=" * W)
    print()


def _print_check(name: str, passed: bool, value: str, threshold: str):
    status = "PASS" if passed else "FAIL"
    print(f"  {name:35s} {status:8s} {value:>14s} {threshold:>14s}")

