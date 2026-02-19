"""Parametric factory utilities for common force and objective patterns.

Provides generic factories that return callables with the correct signatures
for :func:`~wecopttool_differentiable.solver_ipopt.sensitivity` (parametric_forces,
obj_fn). Users pass device-specific constants; factories return JAX-differentiable
functions.
"""

from __future__ import annotations

__all__ = [
    "make_linear_mooring_parametric",
    "make_pto_passive_parametric",
    "make_electrical_power_obj_parametric",
]

from typing import Dict

import jax.numpy as jnp


def make_linear_mooring_parametric():
    r"""Return a parametric mooring force :math:`f = -K x`.

    Supports both mooring-only params (``params.K``) and joint-style
    (``params.mooring.K``).

    Returns
    -------
    callable
        ``(wec, x_wec, x_opt, wave_data, params) -> (ncomponents, ndof)``
        force matrix.
    """
    def f_mooring_parametric(wec, x_wec, x_opt, wave_data, params):
        moor = params.mooring if hasattr(params, "mooring") else params
        pos = wec.vec_to_dofmat(x_wec)
        time_matrix = wec.time_mat_nsubsteps(1)
        return jnp.dot(time_matrix, -moor.K * pos)
    return f_mooring_parametric


def make_pto_passive_parametric(
    gear_ratios: Dict[str, float],
    friction_dict: Dict[str, float],
):
    r"""Return a parametric PTO passive force (spring + friction + inertia).

    Uses device-specific ``gear_ratios`` and ``friction_dict`` for fixed
    coefficients (e.g. pneumatic spring static friction). The differentiable
    parameters come from ``params.pto``: ``spring_stiffness``, ``friction_pto``,
    ``inertia_pto``.

    Parameters
    ----------
    gear_ratios : dict
        Must contain ``'spring'``; used for friction scaling.
    friction_dict : dict
        Must contain ``'Bpneumatic_spring_static1'`` (or equivalent);
        fixed friction coefficient scaled by gear_ratios['spring'].

    Returns
    -------
    callable
        ``(wec, x_wec, x_opt, wave_data, params) -> (ncomponents, ndof)``
        force matrix.
    """
    def f_pto_passive_parametric(wec, x_wec, x_opt, wave_data, params):
        pto_p = params.pto if hasattr(params, "pto") else params
        pos = wec.vec_to_dofmat(x_wec)
        vel = jnp.dot(wec.derivative_mat, pos)
        acc = jnp.dot(wec.derivative_mat, vel)
        time_matrix = wec.time_mat_nsubsteps(1)

        spring = -pto_p.spring_stiffness * pos
        f_spring = jnp.dot(time_matrix, spring)

        fric = -(
            pto_p.friction_pto
            + friction_dict["Bpneumatic_spring_static1"] * gear_ratios["spring"]
        ) * vel
        f_fric = jnp.dot(time_matrix, fric)

        inertia = pto_p.inertia_pto * acc
        f_inertia = jnp.dot(time_matrix, inertia)

        return f_spring + f_fric + f_inertia
    return f_pto_passive_parametric


def make_electrical_power_obj_parametric(pto, nsubsteps: int = 1):
    r"""Return a parametric objective for average electrical power.

    Uses WecOptTool PTO's :meth:`velocity` and :meth:`force`. Electrical power
    = mechanical power + resistive loss :math:`R (T/k_T)^2`. Parameters from
    ``params.pto``: ``winding_resistance`` (R), ``torque_coefficient`` (k_T).

    Parameters
    ----------
    pto : wecopttool.pto.PTO
        PTO object with :meth:`velocity` and :meth:`force`.
    nsubsteps : int, optional
        Substeps for PTO evaluation (default 1).

    Returns
    -------
    callable
        ``(wec, x_wec, x_opt, wave, params) -> scalar``
        average electrical power (positive = extracted).
    """
    def obj_pto_parametric(wec, x_wec, x_opt, wave, params):
        pto_p = params.pto if hasattr(params, "pto") else params
        vel_td = pto.velocity(wec, x_wec, x_opt, wave, nsubsteps=nsubsteps)
        force_td = pto.force(wec, x_wec, x_opt, wave, nsubsteps=nsubsteps)
        power_mech = vel_td * force_td
        loss = pto_p.winding_resistance * (
            force_td / pto_p.torque_coefficient
        ) ** 2

        power_elec = power_mech + loss
        energy = jnp.sum(power_elec) * wec.dt
        return energy / wec.tf
    return obj_pto_parametric
