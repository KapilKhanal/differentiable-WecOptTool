"""Finite-difference validation utilities for sensitivity analysis.

Three levels of checking:

1. :func:`fd_validate` — **end-to-end**: re-solve the NLP at perturbed
   parameters and compare ``(phi*(p+eps) - phi*(p-eps)) / 2eps`` against
   the analytical Fiacco gradient.  The user supplies a ``re_solve_fn``.

2. :func:`fd_check_residual` — verify ``lambda^T (dr/dp)`` against FD
   of the residual.  No re-solve needed.

3. :func:`fd_check_objective` — verify ``df/dp`` against FD of the
   objective.  No re-solve needed.
"""

from __future__ import annotations

__all__ = [
    "fd_validate",
    "fd_check_residual",
    "fd_check_objective",
    "make_re_solve_fn",
    "validate_sensitivity",
]

import logging
from typing import Callable, Dict, Optional, Sequence, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

_log = logging.getLogger(__name__)


class FDResult(NamedTuple):
    """Result for a single parameter."""
    name: str
    analytical: float
    fd: float
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


def make_re_solve_fn(wec_factory, waves, obj_fun, nstate_opt, **solve_kwargs):
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
    **solve_kwargs
        Extra keyword arguments forwarded to :meth:`WEC_IPOPT.solve`
        (e.g. ``scale_x_wec``, ``optim_options``, ``x_wec_0``).

    Returns
    -------
    callable
        ``re_solve_fn(params_dict) -> float`` suitable for
        :func:`fd_validate`.
    """
    def re_solve(params_dict):
        wec_new = wec_factory(params_dict)
        results = wec_new.solve(waves, obj_fun, nstate_opt, **solve_kwargs)
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
    fields=None,
    residual_tol: float = 0.05,
    objective_tol: float = 0.05,
    resolve_tol: float = 0.10,
    verbose: bool = True,
) -> Dict[str, Dict[str, FDResult]]:
    """Run all applicable FD validation checks and return a summary.

    Automatically selects which checks to run based on the arguments
    provided:

    - **Residual check** — runs when *params* and *parametric_forces*
      are given (validates :math:`\\lambda^T \\partial r / \\partial p`).
    - **Objective check** — runs when *params* and *obj_fn* are given
      (validates :math:`\\partial f / \\partial p`).
    - **Full NLP re-solve** — runs when *re_solve_fn* is given
      (validates the complete Fiacco gradient).

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
        Parametric objective function.
    re_solve_fn : callable, optional
        ``re_solve_fn(params_dict) -> float`` for full NLP re-solve
        (see :func:`make_re_solve_fn`).
    fields : sequence of str, optional
        Parameter fields to check (default: all).
    residual_tol, objective_tol, resolve_tol : float
        Tolerance for each check level.
    verbose : bool
        Print comparison tables.

    Returns
    -------
    dict[str, dict[str, FDResult]]
        Nested dict keyed by check name (``'residual'``, ``'objective'``,
        ``'resolve'``), each containing per-parameter :class:`FDResult`.
    """
    from scipy.optimize import OptimizeResult

    if isinstance(results, OptimizeResult):
        results = [results]
    res = results[0]

    report: Dict[str, Dict[str, FDResult]] = {}

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
