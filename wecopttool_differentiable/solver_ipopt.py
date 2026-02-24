"""IPOPT-based solver for WecOptTool with Lagrange multiplier extraction.

This module provides :class:`WEC_IPOPT`, a subclass of
:class:`~wecopttool.WEC` that replaces the default SLSQP solver with
IPOPT (via `cyipopt <https://cyipopt.readthedocs.io>`_).

The key advantage is that IPOPT returns **Lagrange multipliers** for
every constraint, which are needed for  parametric
sensitivity I am considering here see Fiacco/ Sobieski  sensitivity:

.. math::

    \\frac{d\\varphi^*}{dh}
    = \\lambda^\\top \\frac{\\partial r}{\\partial h}

where :math:`\\lambda` are the multipliers for the dynamics equality
constraint :math:`r = m\\ddot{x} - \\Sigma f = 0`.

**Quick-start**::

    from wecopttool_differentiable import WEC_IPOPT, sensitivity

    wec = WEC_IPOPT.from_bem(hydro_data, friction=friction)
    results = wec.solve(waves, obj_fun, nstate_opt)
    res = results[0]
    grad_h = sensitivity(wec, res, waves)

Requires
--------
``pip install cyipopt``
"""

from __future__ import annotations

__all__ = [
    "WEC_IPOPT",
    "make_differentiable_solver",
    "make_differentiable_state_solver",
    "ffo_sensitivity",
    "sensitivity",
    "sensitivity_parametric",
]

import logging
import warnings
from typing import Optional, Mapping, Any, Iterable, Union
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from numpy import ndarray
from scipy.optimize import OptimizeResult, Bounds
from xarray import DataArray, Dataset
import cyipopt

from wecopttool.core import (
    WEC,
    TStateFunction,
    TIForceDict,
    FloatOrArray,
    scale_dofs,
    add_linear_friction,
    frequency_parameters,
    check_radiation_damping,
    standard_forces,
    read_netcdf,
)
from .parametric import (
    extract_bem_params,
    extract_wave_data,
    residual_parametric,
)

jax.config.update("jax_enable_x64", True)

_log = logging.getLogger(__name__)

_default_min_damping = 1e-6


class WEC_IPOPT(WEC):
    """A :class:`~wecopttool.WEC` subclass that uses IPOPT instead of
    SLSQP, and exposes Lagrange multipliers on the result.

    Construction is identical to :class:`~wecopttool.WEC` — use
    :meth:`from_bem`, :meth:`from_floating_body`, or
    :meth:`from_impedance`.  Only :meth:`solve` is overridden.
    """

    # ── from_bem override (returns WEC_IPOPT, not WEC) ────────────────
    @staticmethod
    def from_bem(
        bem_data: Union[Dataset, Union[str, Path]],
        friction: Optional[ndarray] = None,
        f_add: Optional[TIForceDict] = None,
        constraints: Optional[Iterable[Mapping]] = None,
        min_damping: Optional[float] = _default_min_damping,
        uniform_shift: Optional[bool] = False,
        dof_names: Optional[Iterable[str]] = None,
    ) -> "WEC_IPOPT":
        """Create a :class:`WEC_IPOPT` from BEM data.

        Identical to :meth:`wecopttool.WEC.from_bem` but returns a
        :class:`WEC_IPOPT` instance.
        """
        if isinstance(bem_data, (str, Path)):
            bem_data = read_netcdf(bem_data)
        hydro_data = add_linear_friction(bem_data, friction)
        inertia_matrix = hydro_data["inertia_matrix"].values
        f1, nfreq = frequency_parameters(
            hydro_data.omega.values / (2 * np.pi), False)
        if min_damping is not None:
            hydro_data = check_radiation_damping(
                hydro_data, min_damping, uniform_shift)
        linear_force_functions = standard_forces(hydro_data)
        f_add = f_add if (f_add is not None) else {}
        forces = linear_force_functions | f_add
        constraints = constraints if (constraints is not None) else []
        instance = WEC_IPOPT(
            f1, nfreq, forces, constraints, inertia_matrix,
            dof_names=dof_names)
        instance._hydro_data = hydro_data
        return instance

    # ── solve with IPOPT ──────────────────────────────────────────────
    def solve(
        self,
        waves: DataArray,
        obj_fun: TStateFunction,
        nstate_opt: int,
        x_wec_0: Optional[ndarray] = None,
        x_opt_0: Optional[ndarray] = None,
        scale_x_wec: Optional[list] = None,
        scale_x_opt: Optional[FloatOrArray] = 1.0,
        scale_obj: Optional[float] = 1.0,
        optim_options: Optional[Mapping[str, Any]] = {},
        use_grad: Optional[bool] = True,
        maximize: Optional[bool] = False,
        bounds_wec: Optional[Bounds] = None,
        bounds_opt: Optional[Bounds] = None,
        callback: Optional[TStateFunction] = None,
        mult_g_0: Optional[ndarray] = None,
        mult_x_L_0: Optional[ndarray] = None,
        mult_x_U_0: Optional[ndarray] = None,
    ) -> list[OptimizeResult]:
        """Solve the pseudo-spectral problem using IPOPT.

        The interface is identical to :meth:`wecopttool.WEC.solve`
        except for:

        * ``optim_options`` is forwarded to IPOPT.
        * The returned :class:`~scipy.optimize.OptimizeResult` objects
          carry extra attributes: ``dynamics_mult_g``, ``mult_g``,
          ``constraint_multipliers``, etc.
        * ``mult_g_0`` enables full dual warm-start (pass
          ``res.mult_g`` from a previous solve).
        """
        results = []

        # ── scaling vectors ───────────────────────────────────────────
        if scale_x_wec is None:
            scale_x_wec = [1.0] * self.ndof
        elif isinstance(scale_x_wec, (float, int)):
            scale_x_wec = [scale_x_wec] * self.ndof
        scale_x_wec = scale_dofs(scale_x_wec, self.ncomponents)

        if isinstance(scale_x_opt, (float, int)):
            if nstate_opt is None:
                raise ValueError(
                    "'nstate_opt' must be provided when 'scale_x_opt' "
                    "is a scalar.")
            scale_x_opt = scale_dofs([scale_x_opt], nstate_opt)

        scale = jnp.concatenate([
            jnp.array(scale_x_wec), jnp.array(scale_x_opt)])

        # ── initial guess ─────────────────────────────────────────────
        key = jax.random.PRNGKey(0)
        if x_wec_0 is None:
            x_wec_0 = jax.random.normal(
                key, [self.nstate_wec], dtype=np.float64)
        if x_opt_0 is None:
            x_opt_0 = jax.random.normal(
                key, [nstate_opt], dtype=np.float64)
        x0 = np.asarray(
            jnp.concatenate([jnp.array(x_wec_0), jnp.array(x_opt_0)])
            * scale)

        # ── bounds ────────────────────────────────────────────────────
        if (bounds_wec is None) and (bounds_opt is None):
            bounds_list_ipopt = None
        else:
            bounds_in = [bounds_wec, bounds_opt]
            for idx, bii in enumerate(bounds_in):
                if isinstance(bii, tuple):
                    bounds_in[idx] = Bounds(
                        lb=[xibs[0] for xibs in bii],
                        ub=[xibs[1] for xibs in bii])
            inf_wec = jnp.ones(self.nstate_wec) * jnp.inf
            inf_opt = jnp.ones(nstate_opt) * jnp.inf
            bounds_dflt = [
                Bounds(lb=-inf_wec, ub=inf_wec),
                Bounds(lb=-inf_opt, ub=inf_opt)]
            bounds_collected = []
            for bi, bd in zip(bounds_in, bounds_dflt):
                bounds_collected.append(bi if bi is not None else bd)
            lb = np.asarray(
                jnp.hstack([b.lb for b in bounds_collected]) * scale)
            ub = np.asarray(
                jnp.hstack([b.ub for b in bounds_collected]) * scale)
            bounds_list_ipopt = list(zip(lb, ub))

        # ── per-realisation solve loop ────────────────────────────────
        for realization, wave in waves.groupby("realization"):
            _log.info("Solving (IPOPT) for realization %s.", realization)

            try:
                wave = wave.squeeze(dim="realization")
            except KeyError:
                pass

            sign = -1.0 if maximize else 1.0

            def obj_fun_scaled(x):
                x_wec, x_opt = self.decompose_state(x / scale)
                return float(
                    obj_fun(self, x_wec, x_opt, wave) * scale_obj * sign)

            obj_grad = jax.jit(jax.grad(
                lambda x: obj_fun(
                    self,
                    *self.decompose_state(x / scale),
                    wave) * scale_obj * sign
            )) if use_grad else None

            constraints_ipopt = []
            constraint_info = {}
            offset = 0

            for i, icons in enumerate(self.constraints):
                def _make_cfun(ic):
                    def cfun(x):
                        x_wec, x_opt = self.decompose_state(x / scale)
                        return np.atleast_1d(np.asarray(
                            ic["fun"](self, x_wec, x_opt, wave)))
                    return cfun

                cfun = _make_cfun(icons)
                c_dict = {"type": icons["type"], "fun": cfun}
                if use_grad:
                    c_dict["jac"] = jax.jit(jax.jacobian(
                        lambda x, _ic=icons: jnp.atleast_1d(
                            _ic["fun"](
                                self,
                                *self.decompose_state(x / scale),
                                wave))))
                constraints_ipopt.append(c_dict)

                n_c = len(cfun(x0))
                cname = f"user_constraint_{i}"
                constraint_info[cname] = {
                    "type": icons["type"],
                    "slice": slice(offset, offset + n_c),
                    "size": n_c,
                }
                offset += n_c

            def scaled_resid_fun(x):
                x_s = x / scale
                x_wec, x_opt = self.decompose_state(x_s)
                return np.asarray(self.residual(x_wec, x_opt, wave))

            dynamics_dict = {"type": "eq", "fun": scaled_resid_fun}
            if use_grad:
                dynamics_dict["jac"] = jax.jit(jax.jacobian(
                    lambda x: self.residual(
                        *self.decompose_state(x / scale), wave)))
            constraints_ipopt.append(dynamics_dict)

            n_dyn = self.nstate_wec
            constraint_info["dynamics"] = {
                "type": "eq",
                "slice": slice(offset, offset + n_dyn),
                "size": n_dyn,
            }

            ipopt_opts = {
                "mu_strategy": "adaptive",
                "tol": 1e-7,
                "print_level": 5,
                "max_iter": 3000,
            }
            ipopt_opts.update(optim_options)

            if mult_g_0 is None:
                _log.info("Calling cyipopt.minimize_ipopt ...")
                result = cyipopt.minimize_ipopt(
                    fun=obj_fun_scaled,
                    x0=x0,
                    jac=obj_grad,
                    bounds=bounds_list_ipopt,
                    constraints=constraints_ipopt,
                    options=ipopt_opts,
                )
                info = result.info
                r_x = np.asarray(result.x)
                r_fun = float(result.fun)
                r_success = result.success
                r_message = result.message
                r_status = result.status
                r_nit = result.nit
                r_nfev = result.get("nfev", None)
            else:
                _log.info(
                    "Calling cyipopt.Problem.solve with dual warm-start ...")
                from cyipopt.scipy_interface import (
                    IpoptProblemWrapper,
                    get_constraint_bounds,
                    get_constraint_dimensions,
                    _get_sparse_jacobian_structure,
                )

                ipopt_opts.setdefault("warm_start_init_point", "yes")
                ipopt_opts.setdefault("warm_start_bound_push", 1e-9)
                ipopt_opts.setdefault("warm_start_bound_frac", 1e-9)
                ipopt_opts.setdefault("warm_start_mult_bound_push", 1e-9)

                cl, cu = get_constraint_bounds(constraints_ipopt, x0)
                con_dims = get_constraint_dimensions(constraints_ipopt, x0)
                sparse_jacs, jac_r, jac_c = (
                    _get_sparse_jacobian_structure(constraints_ipopt, x0))

                if bounds_list_ipopt is not None:
                    lb = [b[0] for b in bounds_list_ipopt]
                    ub = [b[1] for b in bounds_list_ipopt]
                else:
                    lb = [-1e19] * len(x0)
                    ub = [1e19] * len(x0)

                pw = IpoptProblemWrapper(
                    fun=obj_fun_scaled,
                    jac=obj_grad,
                    constraints=constraints_ipopt,
                    con_dims=con_dims,
                    sparse_jacs=sparse_jacs,
                    jac_nnz_row=jac_r,
                    jac_nnz_col=jac_c,
                )

                nlp = cyipopt.Problem(
                    n=len(x0), m=len(cl), problem_obj=pw,
                    lb=lb, ub=ub, cl=cl, cu=cu,
                )
                nlp.add_option(b"hessian_approximation",
                               b"limited-memory")
                for k, v in ipopt_opts.items():
                    try:
                        nlp.add_option(k, v)
                    except TypeError:
                        pass

                zl = list(mult_x_L_0) if mult_x_L_0 is not None \
                    else list(np.zeros(len(x0)))
                zu = list(mult_x_U_0) if mult_x_U_0 is not None \
                    else list(np.zeros(len(x0)))
                x_sol, info = nlp.solve(
                    x0, lagrange=list(mult_g_0), zl=zl, zu=zu)
                r_x = np.asarray(x_sol)
                r_fun = float(info["obj_val"])
                r_success = info["status"] == 0
                r_message = info["status_msg"]
                r_status = info["status"]
                r_nit = pw.nit
                r_nfev = pw.nfev

            optim_res = OptimizeResult()
            optim_res.x = r_x / np.asarray(scale)
            optim_res.fun = r_fun / scale_obj
            optim_res.success = r_success
            optim_res.message = r_message
            optim_res.status = r_status
            optim_res.nit = r_nit
            optim_res.nfev = r_nfev
            optim_res.mult_g = np.asarray(info["mult_g"])
            optim_res.mult_x_L = np.asarray(info["mult_x_L"])
            optim_res.mult_x_U = np.asarray(info["mult_x_U"])
            optim_res.constraint_values = np.asarray(info["g"])
            optim_res.constraint_info = constraint_info

            dyn_slice = constraint_info["dynamics"]["slice"]
            raw_dyn_mult = optim_res.mult_g[dyn_slice]
            optim_res.dynamics_mult_g = raw_dyn_mult / (scale_obj * sign)

            optim_res.constraint_multipliers = {}
            for cname, cinfo in constraint_info.items():
                if cname == "dynamics":
                    continue
                raw_mu = optim_res.mult_g[cinfo["slice"]]
                optim_res.constraint_multipliers[cname] = (
                    raw_mu / (scale_obj * sign)
                )

            msg = f"{r_message}  (status {r_status})"
            if r_success:
                _log.info(msg)
            else:
                _log.warning(msg)

            results.append(optim_res)

        return results

    def compute_sensitivity(self, results, waves, **kwargs):
        """Compute Fiacco sensitivity :math:`d\\varphi^*/dp`.

        Convenience wrapper around the module-level :func:`sensitivity`
        function so that users can call ``wec.compute_sensitivity(res, waves)``
        without a separate import.

        Parameters
        ----------
        results : OptimizeResult or list[OptimizeResult]
            Result(s) from :meth:`solve`.
        waves : xarray.DataArray
            Wave data used in the solve.
        **kwargs
            Forwarded to :func:`sensitivity` (e.g. *params*,
            *parametric_forces*, *obj_fn*).

        Returns
        -------
        pytree
            Gradient with the same structure as *params*.
        """
        return sensitivity(self, results, waves, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Standalone sensitivity functions (Option B)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_all_realizations(waves, exc_coeff_da):
    """Extract WaveData for every realisation in *waves*."""
    if "realization" in waves.dims:
        realizations = waves["realization"].values
        wave_data_list = []
        wave_list = []
        for r in realizations:
            w = waves.sel(realization=r)
            wave_list.append(w)
            wave_data_list.append(extract_wave_data(w, exc_coeff_da))
        return wave_data_list, wave_list
    else:
        wd = extract_wave_data(waves, exc_coeff_da)
        return [wd], [waves]


def _fix_complex_grad(grad):
    """Conjugate complex leaves so Im(grad) = d/d(Im param).

    JAX's VJP convention for complex inputs with real-valued output is
    ``bar_z = df/d(Re z) − i·df/d(Im z)``.  Users expect
    ``grad = df/d(Re z) + i·df/d(Im z)``, so we conjugate complex leaves.
    """
    return jax.tree_util.tree_map(
        lambda x: jnp.conj(x) if jnp.iscomplexobj(x) else x,
        grad,
    )


def sensitivity(
    wec,
    results,
    waves,
    params=None,
    parametric_forces=None,
    additional_forces=None,
    obj_fn=None,
    residual_fn=None,
    constraint_fns=None,
    obj_fun_parametric=None,
):
    r"""Compute the Fiacco sensitivity :math:`d\varphi^*/dp`.

    Standalone function (Option B). Requires *wec* with ``_hydro_data``
    and *results* with ``dynamics_mult_g`` (from :class:`WEC_IPOPT.solve`).

    Unified API for BEM, PTO, or joint (BEM + PTO) sensitivity.
    Averaged over wave realisations when multiple are present.

    Parameters
    ----------
    wec : WEC_IPOPT
        WEC instance built via :meth:`WEC_IPOPT.from_bem`.
    results : OptimizeResult or list[OptimizeResult]
        Result(s) from :meth:`wec.solve`.
    waves : xarray.DataArray
        The wave data used in the solve.
    params, parametric_forces, additional_forces, obj_fn, residual_fn,
    constraint_fns, obj_fun_parametric
        Same as the original wecopttool sensitivity API (see source).

    Returns
    -------
    pytree
        Gradient with same structure as *params* (BEMParams when params=None).
        For complex-valued parameters (e.g. Froude-Krylov, diffraction),
        ``Re(grad)`` = sensitivity to real part, ``Im(grad)`` = sensitivity
        to imaginary part.
    """
    if not hasattr(wec, "_hydro_data"):
        raise AttributeError(
            "sensitivity() requires _hydro_data.  "
            "Build the WEC_IPOPT via from_bem().")

    if isinstance(results, OptimizeResult):
        results = [results]

    for idx, res in enumerate(results):
        if not hasattr(res, "dynamics_mult_g"):
            raise AttributeError(
                f"results[{idx}] is missing 'dynamics_mult_g'. "
                "Use WEC_IPOPT.solve() (not WEC.solve) to obtain "
                "Lagrange multipliers.")

    if params is not None and isinstance(params, dict):
        raise TypeError(
            "params must be a namedtuple (JAX pytree), not a plain dict. "
            "Use collections.namedtuple or make_joint_params() to create "
            "a structured parameter container.")

    if parametric_forces is not None:
        for k, v in parametric_forces.items():
            if not callable(v):
                raise TypeError(
                    f"parametric_forces['{k}'] is not callable. "
                    "Each value must be a function with signature "
                    "(wec, x_wec, x_opt, wave_data, params).")

    if (params is not None
            and parametric_forces is None
            and residual_fn is None):
        raise ValueError(
            "When params is provided, you must also pass either "
            "'parametric_forces' (preferred) or 'residual_fn' (legacy). "
            "Example: sensitivity(wec, res, waves, params=my_params, "
            "parametric_forces={'PTO': f_pto_parametric})")

    wave_data_list, wave_list = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])

    nreal = len(results)
    if nreal != len(wave_data_list):
        raise ValueError(
            f"Number of results ({nreal}) does not match number of "
            f"wave realisations ({len(wave_data_list)}).")

    if params is None:
        bp = extract_bem_params(wec._hydro_data)
        total_grad = None
        for i in range(nreal):
            res = results[i]
            wd_i = wave_data_list[i]
            x_wec, x_opt = wec.decompose_state(res.x)
            lam = jnp.array(res.dynamics_mult_g)
            x_wec_jax = jnp.array(x_wec)
            x_opt_jax = jnp.array(x_opt)

            def r_of_p(p, _xw=x_wec_jax, _xo=x_opt_jax, _wd=wd_i):
                return residual_parametric(_xw, _xo, _wd, p, wec)

            _, vjp_fn = jax.vjp(r_of_p, bp)
            (grad_c,) = vjp_fn(lam)

            if obj_fun_parametric is not None:
                def f_of_p(p, _xw=x_wec_jax, _xo=x_opt_jax, _wd=wd_i):
                    return obj_fun_parametric(_xw, _xo, _wd, p, wec)
                grad_f = jax.grad(f_of_p)(bp)
                grad_i = jax.tree_util.tree_map(jnp.add, grad_c, grad_f)
            else:
                grad_i = grad_c

            if total_grad is None:
                total_grad = grad_i
            else:
                total_grad = jax.tree_util.tree_map(
                    jnp.add, total_grad, grad_i)

        avg = jax.tree_util.tree_map(lambda x: x / nreal, total_grad)
        return _fix_complex_grad(avg)

    if residual_fn is not None:
        total_grad = None
        for i in range(nreal):
            res = results[i]
            wave_i = wave_list[i]
            x_wec, x_opt = wec.decompose_state(res.x)
            lam = jnp.array(res.dynamics_mult_g)
            x_wec_jax = jnp.array(x_wec)
            x_opt_jax = jnp.array(x_opt)

            def r_of_p(p, _w=wave_i):
                return residual_fn(wec, x_wec_jax, x_opt_jax, _w, p)

            _, vjp_fn = jax.vjp(r_of_p, params)
            (grad_total,) = vjp_fn(lam)

            if obj_fn is not None:
                def f_of_p(p, _w=wave_i):
                    return obj_fn(wec, x_wec_jax, x_opt_jax, _w, p)
                grad_f = jax.grad(f_of_p)(params)
                grad_total = jax.tree_util.tree_map(
                    jnp.add, grad_total, grad_f)

            if constraint_fns is not None:
                for fn, cname in constraint_fns:
                    if cname not in res.constraint_multipliers:
                        raise KeyError(
                            f"Constraint '{cname}' not found in "
                            f"result[{i}].constraint_multipliers.")
                    mu_i = jnp.array(res.constraint_multipliers[cname])

                    def g_of_p(p, _fn=fn, _w=wave_i):
                        return _fn(wec, x_wec_jax, x_opt_jax, _w, p)

                    _, vjp_g = jax.vjp(g_of_p, params)
                    (grad_g,) = vjp_g(mu_i)
                    grad_total = jax.tree_util.tree_map(
                        jnp.add, grad_total, grad_g)

            if total_grad is None:
                total_grad = grad_total
            else:
                total_grad = jax.tree_util.tree_map(
                    jnp.add, total_grad, grad_total)

        avg = jax.tree_util.tree_map(lambda x: x / nreal, total_grad)
        return _fix_complex_grad(avg)

    if parametric_forces is None:
        raise ValueError(
            "sensitivity(params=...) requires parametric_forces or residual_fn.  "
            "Pass parametric_forces dict or residual_fn for legacy API.")

    if additional_forces is None and hasattr(wec, 'forces'):
        additional_forces = {
            k: v for k, v in wec.forces.items()
            if k not in parametric_forces
        }

    def _wrap_force(f, wave):
        return lambda wec, xw, xo, wd: f(wec, xw, xo, wave)

    total_grad = None
    for i in range(nreal):
        res = results[i]
        wd_i = wave_data_list[i]
        wave_i = wave_list[i]
        x_wec, x_opt = wec.decompose_state(res.x)
        lam = jnp.array(res.dynamics_mult_g)
        x_wec_jax = jnp.array(x_wec)
        x_opt_jax = jnp.array(x_opt)

        add_i = None
        if additional_forces is not None:
            param_keys = set(parametric_forces.keys())
            add_i = {
                k: _wrap_force(f, wave_i)
                for k, f in additional_forces.items()
                if k not in param_keys
            }
            if not add_i:
                add_i = None

        def r_of_p(p, _xw=x_wec_jax, _xo=x_opt_jax, _wd=wd_i, _add=add_i):
            return residual_parametric(
                _xw, _xo, _wd, p, wec,
                additional_forces=_add,
                parametric_forces=parametric_forces,
            )

        _, vjp_fn = jax.vjp(r_of_p, params)
        (grad_total,) = vjp_fn(lam)

        if obj_fn is not None:
            def f_of_p(p, _w=wave_i):
                return obj_fn(wec, x_wec_jax, x_opt_jax, _w, p)
            grad_f = jax.grad(f_of_p)(params)
            grad_total = jax.tree_util.tree_map(
                jnp.add, grad_total, grad_f)

        if constraint_fns is not None:
            for fn, cname in constraint_fns:
                if cname not in res.constraint_multipliers:
                    raise KeyError(
                        f"Constraint '{cname}' not found in "
                        f"result[{i}].constraint_multipliers.")
                mu_i = jnp.array(res.constraint_multipliers[cname])

                def g_of_p(p, _fn=fn, _w=wave_i):
                    return _fn(wec, x_wec_jax, x_opt_jax, _w, p)

                _, vjp_g = jax.vjp(g_of_p, params)
                (grad_g,) = vjp_g(mu_i)
                grad_total = jax.tree_util.tree_map(
                    jnp.add, grad_total, grad_g)

        if total_grad is None:
            total_grad = grad_total
        else:
            total_grad = jax.tree_util.tree_map(
                jnp.add, total_grad, grad_total)

    avg = jax.tree_util.tree_map(lambda x: x / nreal, total_grad)
    return _fix_complex_grad(avg)


def sensitivity_parametric(
    wec, results, waves, params, residual_fn,
    obj_fn=None, constraint_fns=None,
):
    r"""Compute Fiacco sensitivity for arbitrary parameters (legacy API).

    .. deprecated::
        Use :func:`sensitivity` with ``parametric_forces`` instead.
        This wrapper will be removed in a future release.
    """
    warnings.warn(
        "sensitivity_parametric() is deprecated. "
        "Use sensitivity(wec, results, waves, params=..., "
        "parametric_forces=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sensitivity(
        wec, results, waves, params=params,
        residual_fn=residual_fn, obj_fn=obj_fn, constraint_fns=constraint_fns,
    )


# ═══════════════════════════════════════════════════════════════════════════
# JAX-transparent differentiable solver
# ═══════════════════════════════════════════════════════════════════════════

def make_differentiable_solver(
    wec: WEC_IPOPT,
    waves: DataArray,
    obj_fun: TStateFunction,
    nstate_opt: int,
    obj_fun_parametric=None,
    **solve_kwargs,
):
    r"""Return a JAX-differentiable function ``f(bem_params) -> phi_star``.

    The returned callable runs IPOPT in the **forward** pass and uses
    the Fiacco / envelope-theorem formula in the **backward** pass so
    that :func:`jax.grad` works transparently.
    """
    if not hasattr(wec, "_hydro_data"):
        raise AttributeError(
            "make_differentiable_solver requires _hydro_data on the WEC.  "
            "Build the WEC_IPOPT via from_bem().")

    wave_data_list, _ = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    nreal = len(wave_data_list)

    _warm = {"x_wec_0": None, "x_opt_0": None}

    def _resolve_kwargs():
        kw = dict(solve_kwargs)
        if _warm["x_wec_0"] is not None and "x_wec_0" not in kw:
            kw["x_wec_0"] = _warm["x_wec_0"]
        if _warm["x_opt_0"] is not None and "x_opt_0" not in kw:
            kw["x_opt_0"] = _warm["x_opt_0"]
        return kw

    def _update_warm(results):
        res = results[-1]
        x_w, x_o = wec.decompose_state(res.x)
        _warm["x_wec_0"] = np.asarray(x_w)
        _warm["x_opt_0"] = np.asarray(x_o)

    @jax.custom_vjp
    def solve(bem_params):
        results = wec.solve(
            waves, obj_fun, nstate_opt, **_resolve_kwargs())
        _update_warm(results)
        phi_vals = jnp.array(
            [r.fun for r in results], dtype=jnp.float64)
        return jnp.mean(phi_vals)

    def solve_fwd(bem_params):
        results = wec.solve(
            waves, obj_fun, nstate_opt, **_resolve_kwargs())
        _update_warm(results)

        x_wecs, x_opts, lams, phi_vals = [], [], [], []
        for res in results:
            x_w, x_o = wec.decompose_state(res.x)
            phi_vals.append(res.fun)
            x_wecs.append(jnp.array(x_w, dtype=jnp.float64))
            x_opts.append(jnp.array(x_o, dtype=jnp.float64))
            lams.append(jnp.array(res.dynamics_mult_g, dtype=jnp.float64))

        phi_mean = jnp.mean(
            jnp.array(phi_vals, dtype=jnp.float64))
        residuals = (
            bem_params,
            jnp.stack(x_wecs),
            jnp.stack(x_opts),
            jnp.stack(lams),
        )
        return phi_mean, residuals

    def solve_bwd(residuals, g):
        bp, x_wecs, x_opts, lams_all = residuals
        total_grad = None

        for i in range(nreal):
            x_w = x_wecs[i]
            x_o = x_opts[i]
            lam_i = lams_all[i]
            wd_i = wave_data_list[i]

            def r_of_h(h, _xw=x_w, _xo=x_o, _wd=wd_i):
                return residual_parametric(_xw, _xo, _wd, h, wec)

            _, vjp_fn = jax.vjp(r_of_h, bp)
            (grad_c,) = vjp_fn(lam_i)

            if obj_fun_parametric is not None:
                def f_of_h(h, _xw=x_w, _xo=x_o, _wd=wd_i):
                    return obj_fun_parametric(_xw, _xo, _wd, h, wec)
                grad_f = jax.grad(f_of_h)(bp)
                grad_i = jax.tree_util.tree_map(jnp.add, grad_c, grad_f)
            else:
                grad_i = grad_c

            if total_grad is None:
                total_grad = grad_i
            else:
                total_grad = jax.tree_util.tree_map(
                    jnp.add, total_grad, grad_i)

        avg = jax.tree_util.tree_map(
            lambda x: g * x / nreal, total_grad)
        return (_fix_complex_grad(avg),)

    solve.defvjp(solve_fwd, solve_bwd)
    solve.warm_start_state = _warm

    return solve


# ═══════════════════════════════════════════════════════════════════════════
# FFO: Fully First-Order Differentiable Optimization adjoint for dx*/dp
# ═══════════════════════════════════════════════════════════════════════════

def ffo_sensitivity(
    wec: WEC_IPOPT,
    res: OptimizeResult,
    waves: DataArray,
    obj_fun: TStateFunction,
    nstate_opt: int,
    v: np.ndarray,
    *,
    delta: float = 1e-4,
    active_tol: float = 1e-6,
    max_retries: int = 3,
    **solve_kwargs,
) -> "BEMParams":
    r"""Compute :math:`v^\top \, \partial x^*/\partial p` via FFO.

    Uses the Fully First-Order Differentiable Optimization (FFO) method:
    perturb the objective by :math:`\delta \, v^\top x`, re-solve the NLP,
    and finite-difference the Lagrangian gradient w.r.t. parameters.

    This gives the *state* sensitivity (how optimal design variables
    change with parameters), contracted with seed vector *v*.  Combined
    with ``jax.custom_vjp``, this enables differentiating any downstream
    function of :math:`x^*` through the NLP solve.

    Parameters
    ----------
    wec : WEC_IPOPT
        WEC instance with ``_hydro_data``.
    res : OptimizeResult
        Result from :meth:`WEC_IPOPT.solve` (carries x*, λ*).
    waves : xarray.DataArray
        Wave data used in the solve.
    obj_fun : callable
        Objective function ``obj_fun(wec, x_wec, x_opt, wave) -> scalar``.
    nstate_opt : int
        Number of optimisation state variables.
    v : array
        Seed vector, same length as ``res.x``.  In reverse-mode AD
        this is :math:`\partial J_{\text{outer}} / \partial x^*`.
    delta : float
        Perturbation size for FFO.  Typically 1e-4 to 1e-6.
    active_tol : float
        Tolerance for active-set stability check.
    max_retries : int
        Number of times to halve delta if active set changes.
    **solve_kwargs
        Extra keyword arguments forwarded to :meth:`WEC_IPOPT.solve`.

    Returns
    -------
    BEMParams (or pytree matching wec's parameter structure)
        The contraction :math:`v^\top \partial x^*/\partial p`,
        with the same pytree structure as ``extract_bem_params()``.
    """
    if not hasattr(wec, "_hydro_data"):
        raise AttributeError(
            "ffo_sensitivity requires _hydro_data on the WEC.  "
            "Build the WEC_IPOPT via from_bem().")

    x_star = np.array(res.x)
    v = np.asarray(v, dtype=np.float64)
    assert len(v) == len(x_star), (
        f"Seed v has length {len(v)}, expected {len(x_star)}")

    x_wec_star, x_opt_star = wec.decompose_state(x_star)

    wave_data_list, wave_list = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    wave_0 = wave_list[0]
    wd_0 = wave_data_list[0]

    bp = extract_bem_params(wec._hydro_data)

    # Warm-start the perturbed solve from the original solution
    kw = dict(solve_kwargs)
    kw.setdefault("x_wec_0", x_wec_star)
    kw.setdefault("x_opt_0", x_opt_star)
    if "optim_options" not in kw:
        kw["optim_options"] = {}
    kw["optim_options"].setdefault("print_level", 0)
    kw["optim_options"].setdefault("max_iter", 2000)
    kw["optim_options"].setdefault("tol", 1e-8)
    # Warm-start duals
    kw.setdefault("mult_g_0", res.mult_g)
    kw.setdefault("mult_x_L_0", res.mult_x_L)
    kw.setdefault("mult_x_U_0", res.mult_x_U)

    current_delta = delta

    for attempt in range(max_retries):
        v_scaled = current_delta * v

        def perturbed_obj(wec_obj, x_wec, x_opt, wave):
            base = obj_fun(wec_obj, x_wec, x_opt, wave)
            x_full = jnp.concatenate([
                jnp.ravel(x_wec), jnp.ravel(x_opt)])
            return base + jnp.dot(jnp.array(v_scaled), x_full)

        results_pert = wec.solve(
            waves, perturbed_obj, nstate_opt, **kw)
        res_pert = results_pert[0]

        if not res_pert.success:
            _log.warning(
                "FFO perturbed solve did not converge (attempt %d). "
                "Halving delta.", attempt + 1)
            current_delta /= 2
            continue

        # Active-set stability check
        ci = res.constraint_info
        stable = True
        for cname, cinfo in ci.items():
            if cname == "dynamics":
                continue
            g_orig = res.constraint_values[cinfo["slice"]]
            g_pert = res_pert.constraint_values[cinfo["slice"]]
            active_orig = np.abs(g_orig) < active_tol
            active_pert = np.abs(g_pert) < active_tol
            if not np.array_equal(active_orig, active_pert):
                n_changed = int(np.sum(active_orig != active_pert))
                _log.warning(
                    "FFO active set changed for %s (%d constraints). "
                    "Halving delta (attempt %d).",
                    cname, n_changed, attempt + 1)
                stable = False
                break

        if stable:
            break
        current_delta /= 2
    else:
        _log.warning(
            "FFO active set unstable after %d retries. "
            "Proceeding with delta=%.2e.", max_retries, current_delta)

    # Lagrangian gradient w.r.t. parameters at original and perturbed solutions
    x_wec_pert, x_opt_pert = wec.decompose_state(res_pert.x)
    lam_star = jnp.array(res.dynamics_mult_g)
    lam_pert = jnp.array(res_pert.dynamics_mult_g)

    def lagrangian_grad_p(x_w, x_o, lam, wd):
        """∇_p L = λ^T ∂r/∂p  (dynamics Lagrangian w.r.t. BEM params)."""
        def r_of_p(p):
            return residual_parametric(
                jnp.array(x_w), jnp.array(x_o), wd, p, wec)
        _, vjp_fn = jax.vjp(r_of_p, bp)
        (grad,) = vjp_fn(lam)
        return grad

    grad_orig = lagrangian_grad_p(x_wec_star, x_opt_star, lam_star, wd_0)
    grad_pert = lagrangian_grad_p(x_wec_pert, x_opt_pert, lam_pert, wd_0)

    # FFO finite difference: v^T dx*/dp ≈ (∇_p L_pert - ∇_p L_orig) / delta
    result = jax.tree_util.tree_map(
        lambda gp, go: (gp - go) / current_delta, grad_pert, grad_orig)

    return _fix_complex_grad(result)


def make_differentiable_state_solver(
    wec: WEC_IPOPT,
    waves: DataArray,
    obj_fun: TStateFunction,
    nstate_opt: int,
    *,
    ffo_delta: float = 1e-4,
    active_tol: float = 1e-6,
    **solve_kwargs,
):
    r"""Return a JAX-differentiable function ``f(bem_params) -> x_star``.

    The returned callable runs IPOPT in the **forward** pass and uses
    the FFO (First-order Forward-mode Optimization) method in the
    **backward** pass, enabling :func:`jax.grad` of any function that
    consumes the optimal design variables :math:`x^*`.

    Unlike :func:`make_differentiable_solver` (which returns only
    :math:`\varphi^*` and uses Fiacco for the backward), this returns
    the full state vector and uses one extra NLP re-solve per backward
    call.

    Parameters
    ----------
    wec : WEC_IPOPT
        WEC instance with ``_hydro_data``.
    waves : xarray.DataArray
        Wave data.
    obj_fun : callable
        Objective ``obj_fun(wec, x_wec, x_opt, wave) -> scalar``.
    nstate_opt : int
        Number of optimisation state variables.
    ffo_delta : float
        Perturbation size for FFO backward pass.
    active_tol : float
        Tolerance for active-set stability check.
    **solve_kwargs
        Extra keyword arguments for :meth:`WEC_IPOPT.solve`.

    Returns
    -------
    callable
        ``f(bem_params) -> x_star`` (JAX array of length nstate_wec + nstate_opt).
        Has a custom VJP so ``jax.grad(lambda p: g(f(p)))(p0)`` works.
    """
    if not hasattr(wec, "_hydro_data"):
        raise AttributeError(
            "make_differentiable_state_solver requires _hydro_data.  "
            "Build the WEC_IPOPT via from_bem().")

    wave_data_list, _ = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    nreal = len(wave_data_list)

    _warm = {"x_wec_0": None, "x_opt_0": None}

    def _resolve_kwargs():
        kw = dict(solve_kwargs)
        if _warm["x_wec_0"] is not None and "x_wec_0" not in kw:
            kw["x_wec_0"] = _warm["x_wec_0"]
        if _warm["x_opt_0"] is not None and "x_opt_0" not in kw:
            kw["x_opt_0"] = _warm["x_opt_0"]
        return kw

    def _update_warm(results):
        res = results[-1]
        x_w, x_o = wec.decompose_state(res.x)
        _warm["x_wec_0"] = np.asarray(x_w)
        _warm["x_opt_0"] = np.asarray(x_o)

    @jax.custom_vjp
    def solve(bem_params):
        results = wec.solve(
            waves, obj_fun, nstate_opt, **_resolve_kwargs())
        _update_warm(results)
        return jnp.array(results[0].x, dtype=jnp.float64)

    def solve_fwd(bem_params):
        results = wec.solve(
            waves, obj_fun, nstate_opt, **_resolve_kwargs())
        _update_warm(results)
        res = results[0]
        x_star = jnp.array(res.x, dtype=jnp.float64)

        residuals = (
            bem_params,
            x_star,
            jnp.array(res.dynamics_mult_g, dtype=jnp.float64),
            jnp.array(res.mult_g, dtype=jnp.float64),
            jnp.array(res.mult_x_L, dtype=jnp.float64),
            jnp.array(res.mult_x_U, dtype=jnp.float64),
            jnp.array(res.constraint_values, dtype=jnp.float64),
        )
        return x_star, residuals

    def solve_bwd(residuals, g):
        """FFO backward: g is the seed dJ/dx* from downstream."""
        bp, x_star, lam_dyn, mult_g, mult_x_L, mult_x_U, g_vals = residuals
        v = np.asarray(g)

        x_wec_star, x_opt_star = wec.decompose_state(np.asarray(x_star))
        wd_0 = wave_data_list[0]

        # Warm-start perturbed solve from original solution
        kw = dict(solve_kwargs)
        kw["x_wec_0"] = x_wec_star
        kw["x_opt_0"] = x_opt_star
        if "optim_options" not in kw:
            kw["optim_options"] = {}
        kw["optim_options"].setdefault("print_level", 0)
        kw["optim_options"].setdefault("max_iter", 2000)
        kw["optim_options"].setdefault("tol", 1e-8)
        kw["mult_g_0"] = np.asarray(mult_g)
        kw["mult_x_L_0"] = np.asarray(mult_x_L)
        kw["mult_x_U_0"] = np.asarray(mult_x_U)

        v_scaled = ffo_delta * v

        def perturbed_obj(wec_obj, x_wec, x_opt, wave):
            base = obj_fun(wec_obj, x_wec, x_opt, wave)
            x_full = jnp.concatenate([
                jnp.ravel(x_wec), jnp.ravel(x_opt)])
            return base + jnp.dot(jnp.array(v_scaled), x_full)

        results_pert = wec.solve(
            waves, perturbed_obj, nstate_opt, **kw)
        res_pert = results_pert[0]

        x_wec_pert, x_opt_pert = wec.decompose_state(res_pert.x)
        lam_pert = jnp.array(res_pert.dynamics_mult_g)

        def lagrangian_grad_p(x_w, x_o, lam):
            def r_of_p(p):
                return residual_parametric(
                    jnp.array(x_w), jnp.array(x_o), wd_0, p, wec)
            _, vjp_fn = jax.vjp(r_of_p, bp)
            (grad,) = vjp_fn(lam)
            return grad

        grad_orig = lagrangian_grad_p(
            x_wec_star, x_opt_star, jnp.array(lam_dyn))
        grad_pert = lagrangian_grad_p(
            x_wec_pert, x_opt_pert, lam_pert)

        result = jax.tree_util.tree_map(
            lambda gp, go: (gp - go) / ffo_delta, grad_pert, grad_orig)

        return (_fix_complex_grad(result),)

    solve.defvjp(solve_fwd, solve_bwd)
    solve.warm_start_state = _warm

    return solve
