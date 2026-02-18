"""Parametric residual — JAX-differentiable w.r.t. BEM parameters.

This module provides a *parallel differentiable path* alongside the
existing WecOptTool solver.  The NLP solver (scipy/SLSQP) continues to
use the original closure-based forces.  After the NLP solves, this
module lets you compute

.. math::

    \\lambda^\\top \\, \\frac{\\partial r}{\\partial h}

via :func:`jax.vjp` for parametric sensitivity in a
bilevel optimisation setup.

No changes to the existing :class:`~wecopttool.WEC` class or solver
are required.

**Quick-start**::

    import wecopttool as wot
    from wecopttool_differentiable import (
        WEC_IPOPT, sensitivity, extract_bem_params, extract_wave_data,
        residual_parametric,
    )

    wec = WEC_IPOPT.from_bem(hydro_data, friction=friction)
    results = wec.solve(waves, obj_fun, nstate_opt)
    res = results[0]
    grad_h = sensitivity(wec, res, waves)
"""

from __future__ import annotations

__all__ = [
    "BEMParams",
    "WaveData",
    "extract_bem_params",
    "extract_wave_data",
    "residual_parametric",
]

from collections import namedtuple
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from xarray import Dataset, DataArray

from wecopttool.core import (
    mimo_transfer_mat,
    block_diag_jax,
    vec_to_dofmat,
    dofmat_to_vec,
    complex_to_real,
    frequency,
    ncomponents,
    subset_close,
)

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# Data containers  (namedtuples are automatic JAX pytrees)
# ═══════════════════════════════════════════════════════════════════════════

BEMParams = namedtuple("BEMParams", [
    "added_mass",              # (nfreq, ndof, ndof)  real
    "radiation_damping",       # (nfreq, ndof, ndof)  real
    "hydrostatic_stiffness",   # (ndof, ndof)         real
    "friction",                # (ndof, ndof)         real
    "Froude_Krylov_force",     # (nfreq, ndir, ndof)  complex
    "diffraction_force",       # (nfreq, ndir, ndof)  complex
    "inertia_matrix",          # (ndof, ndof)         real
])


WaveData = namedtuple("WaveData", [
    "wave_elev",    # (nfreq, ndir) complex  — Fourier amplitudes
    "sub_ind",      # list[int]              — direction indices into exc_coeff
])


# ═══════════════════════════════════════════════════════════════════════════
# Extraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def extract_bem_params(hydro_data: Dataset) -> BEMParams:
    """Extract BEM parameters from an :class:`xarray.Dataset` as JAX arrays.

    The input should be the *final* hydrodynamic dataset — i.e. **after**
    :func:`~wecopttool.add_linear_friction` and
    :func:`~wecopttool.check_radiation_damping` have been applied — so
    that the extracted arrays exactly match what the
    :class:`~wecopttool.WEC` object uses internally.

    Parameters
    ----------
    hydro_data
        Linear hydrodynamic dataset (output of ``run_bem`` /
        ``add_linear_friction`` / ``check_radiation_damping``).

    Returns
    -------
    BEMParams
        Named tuple whose fields are :class:`jax.Array` instances.
    """
    return BEMParams(
        added_mass=jnp.array(
            hydro_data["added_mass"]
            .transpose("omega", "radiating_dof", "influenced_dof").values),
        radiation_damping=jnp.array(
            hydro_data["radiation_damping"]
            .transpose("omega", "radiating_dof", "influenced_dof").values),
        hydrostatic_stiffness=jnp.array(
            hydro_data["hydrostatic_stiffness"].values),
        friction=jnp.array(
            hydro_data["friction"].values),
        Froude_Krylov_force=jnp.array(
            hydro_data["Froude_Krylov_force"].values),
        diffraction_force=jnp.array(
            hydro_data["diffraction_force"].values),
        inertia_matrix=jnp.array(
            hydro_data["inertia_matrix"].values),
    )


def extract_wave_data(wave: DataArray, exc_coeff: DataArray) -> WaveData:
    """Pre-extract wave arrays and direction indices for the parametric path.

    This must be called **outside** any ``jax.vjp`` / ``jax.jacrev``
    scope because it touches xarray metadata.

    Parameters
    ----------
    wave
        Single-realisation wave :class:`xarray.DataArray`
        (omega x wave_direction).
    exc_coeff
        Any one of the excitation-coefficient DataArrays from
        *hydro_data* (used only to resolve direction indices).

    Returns
    -------
    WaveData
        Named tuple with ``wave_elev`` (JAX array) and ``sub_ind``
        (list of int).
    """
    dir_w = wave["wave_direction"].values
    dir_e = exc_coeff["wave_direction"].values
    subset, sub_ind = subset_close(dir_w, dir_e)
    if not subset:
        raise ValueError(
            "Some wave directions are not in excitation coefficients.\n"
            f"  Wave dirs : {np.rad2deg(dir_w)} deg\n"
            f"  BEM  dirs : {np.rad2deg(dir_e)} deg")
    return WaveData(
        wave_elev=jnp.array(wave.values),
        sub_ind=sub_ind,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Parametric force functions  (pure JAX — no closures over BEM data)
# ═══════════════════════════════════════════════════════════════════════════

def _radiation_force(time_mat, x_wec, ndof, omega,
                     added_mass, radiation_damping):
    r"""Radiation force: :math:`-(j\omega B - \omega^2 A)` on position state.

    Matches ``standard_forces`` radiation entry with sign convention
    :math:`m\ddot{x} = \Sigma f` (negated, ``zero_freq=False``).
    """
    w = jnp.expand_dims(omega, [1, 2])
    rao = -1.0 * (1j * w * radiation_damping + -1.0 * w**2 * added_mass)
    transfer = mimo_transfer_mat(rao, zero_freq=False)
    force_fd = vec_to_dofmat(jnp.dot(transfer, x_wec), ndof)
    return jnp.dot(time_mat, force_fd)


def _friction_force(time_mat, x_wec, ndof, omega, friction):
    r"""Friction force: :math:`-(j\omega B_f)`."""
    w = jnp.expand_dims(omega, [1, 2])
    Bf = jnp.broadcast_to(friction, (len(omega), *friction.shape))
    rao = -1.0 * (1j * w * Bf)
    transfer = mimo_transfer_mat(rao, zero_freq=False)
    force_fd = vec_to_dofmat(jnp.dot(transfer, x_wec), ndof)
    return jnp.dot(time_mat, force_fd)


def _hydrostatic_force(time_mat, x_wec, ndof, nfreq,
                       hydrostatic_stiffness):
    r"""Hydrostatic restoring force: :math:`-K` (includes zero-frequency).

    The RAO is constant across frequencies (stiffness matrix), with an
    explicit zero-frequency row prepended.
    """
    K = hydrostatic_stiffness + 0j
    K_row = jnp.expand_dims(K, 0)
    rao_body = jnp.tile(K_row, (nfreq, 1, 1))
    rao_full = jnp.concatenate([K_row, rao_body], axis=0)
    rao_full = -1.0 * rao_full
    transfer = mimo_transfer_mat(rao_full, zero_freq=True)
    force_fd = vec_to_dofmat(jnp.dot(transfer, x_wec), ndof)
    return jnp.dot(time_mat, force_fd)


def _wave_excitation_parametric(exc_coeff, wave_data):
    r"""Pure-JAX wave excitation: :math:`\sum_\theta \eta(\omega,\theta) X(\omega,\theta,\text{dof})`.

    Returns complex Fourier coefficients of shape ``(nfreq, ndof)``.
    """
    wave_expanded = jnp.expand_dims(wave_data.wave_elev, -1)
    return jnp.sum(
        wave_expanded * exc_coeff[:, wave_data.sub_ind, :], axis=1)


def _excitation_force(time_mat, exc_coeff, wave_data):
    """Excitation force in time-domain from excitation coefficients.

    Uses :func:`~wecopttool.complex_to_real` with ``zero_freq=False``
    (excitation has no DC component).
    """
    exc_fd = _wave_excitation_parametric(exc_coeff, wave_data)
    force_fd = complex_to_real(exc_fd, zero_freq=False)
    return jnp.dot(time_mat[:, 1:], force_fd)


def _inertia_force(time_mat, x_wec, ndof, omega, inertia_matrix):
    r"""Inertia 'force': :math:`-\omega^2 M` (``zero_freq=False``).

    This is the left-hand side :math:`m \ddot{x}` expressed as an RAO
    acting on the position state.
    """
    w = jnp.expand_dims(omega, [1, 2])
    M = jnp.expand_dims(inertia_matrix, 0)
    rao = -1.0 * w**2 * M + 0j
    transfer = mimo_transfer_mat(rao, zero_freq=False)
    force_fd = vec_to_dofmat(jnp.dot(transfer, x_wec), ndof)
    return jnp.dot(time_mat, force_fd)


# ═══════════════════════════════════════════════════════════════════════════
# Parametric Residual   r(x_wec, x_opt ; params)
# ═══════════════════════════════════════════════════════════════════════════

def _get_bem_params(params):
    """Extract BEMParams from params — supports BEMParams or structured pytree."""
    if hasattr(params, "_fields") and "added_mass" in getattr(params, "_fields", ()):
        return params
    return params.bem


def residual_parametric(x_wec, x_opt, wave_data, params, wec,
                        additional_forces=None,
                        parametric_forces=None):
    r"""Dynamics residual :math:`r = m\ddot{x} - \Sigma f` as a
    *pure function* of parameters (BEM and/or PTO).

    This function is fully compatible with :func:`jax.vjp`,
    :func:`jax.jacrev`, :func:`jax.jvp`, etc. for computing
    :math:`\partial r / \partial p` over the full *params* pytree.

    Supports BEM-only, PTO-only, or joint (BEM + PTO) sensitivity
    via a single unified residual.

    Parameters
    ----------
    x_wec : jax.Array
        WEC state vector (Fourier coefficients of position).
    x_opt : jax.Array
        Optimisation (control) state vector.
    wave_data : WaveData
        Pre-extracted wave amplitudes and direction indices
        (see :func:`extract_wave_data`).
    params : BEMParams or pytree
        For BEM-only: pass :class:`BEMParams` directly.
        For PTO or joint: pass a structure with a ``bem`` field
        (e.g. ``Params(bem=bp, pto=pto_p)``).
    wec : wecopttool.WEC
        The WEC object — used **only** for static configuration
        (``nfreq``, ``ndof``, ``time_mat``, etc.).  Its force closures
        are NOT called.
    additional_forces : dict[str, callable] | None
        Non-parametric forces to add.  These are **not** differentiated
        w.r.t. *params* — they are closure-based and opaque to
        sensitivity.  Signature:
        ``fn(wec, x_wec, x_opt, wave_data) -> (ncomponents, ndof)``.
        Exclude any keys that are in *parametric_forces*.
    parametric_forces : dict[str, callable] | None
        Forces differentiable w.r.t. *params* (unlike additional_forces).
        Can include any force whose parameters you want to differentiate
        (e.g. mooring stiffness, buoyancy displacement), not only PTO.
        Signature: ``fn(wec, x_wec, x_opt, wave_data, params) ->
        (ncomponents, ndof)``.  These replace the corresponding
        closure-based forces from *additional_forces*.

    Returns
    -------
    jax.Array
        Residual vector of length ``ncomponents * ndof``.
    """
    bem_params = _get_bem_params(params)
    ndof  = wec.ndof
    nfreq = wec.nfreq
    omega = jnp.array(wec.omega[1:])
    tmat  = jnp.array(wec.time_mat)

    # inertia  (LHS:  m * a)
    ri = _inertia_force(tmat, x_wec, ndof, omega,
                        bem_params.inertia_matrix)

    # subtract standard linear forces (RHS)
    ri = ri - _radiation_force(tmat, x_wec, ndof, omega,
                               bem_params.added_mass,
                               bem_params.radiation_damping)
    ri = ri - _hydrostatic_force(tmat, x_wec, ndof, nfreq,
                                 bem_params.hydrostatic_stiffness)
    ri = ri - _friction_force(tmat, x_wec, ndof, omega,
                              bem_params.friction)
    ri = ri - _excitation_force(tmat, bem_params.Froude_Krylov_force,
                                wave_data)
    ri = ri - _excitation_force(tmat, bem_params.diffraction_force,
                                wave_data)

    # additional (non-parametric) forces — exclude keys in parametric_forces
    if additional_forces is not None:
        param_keys = parametric_forces.keys() if parametric_forces else ()
        for k, f in additional_forces.items():
            if k not in param_keys:
                ri = ri - f(wec, x_wec, x_opt, wave_data)

    # parametric forces — differentiable w.r.t. params
    if parametric_forces is not None:
        for f in parametric_forces.values():
            ri = ri - f(wec, x_wec, x_opt, wave_data, params)

    return dofmat_to_vec(ri)
