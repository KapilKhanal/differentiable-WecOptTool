"""Tests for the wecopttool_differentiable parametric module.

Validates that the parametric (JAX-differentiable) residual matches the
original closure-based residual, and that ``jax.jacrev`` / ``jax.vjp``
produce finite, non-zero derivatives w.r.t. BEM parameters.

Run with::

    pytest tests/test_parametric.py -v
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import xarray as xr

import wecopttool as wot
from wecopttool_differentiable.parametric import (
    BEMParams,
    WaveData,
    extract_bem_params,
    extract_wave_data,
    residual_parametric,
    _radiation_force,
    _friction_force,
    _hydrostatic_force,
    _excitation_force,
    _inertia_force,
    _wave_excitation_parametric,
)

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures  (synthetic data — no Capytaine mesh required)
# ═══════════════════════════════════════════════════════════════════════════

F1 = 0.12
NFREQ = 5
NDOF = 2
NDIR = 3
FRICTION = np.eye(NDOF) * 0.5


@pytest.fixture(scope="module")
def bem_data():
    coords = {
        "omega": [2 * np.pi * (i + 1) * F1 for i in range(NFREQ)],
        "influenced_dof": ["DOF_1", "DOF_2"],
        "radiating_dof": ["DOF_1", "DOF_2"],
        "wave_direction": [0.0, 1.5, 2.1],
    }
    radiation_dims = ["omega", "radiating_dof", "influenced_dof"]
    excitation_dims = ["omega", "wave_direction", "influenced_dof"]
    hydrostatics_dims = ["radiating_dof", "influenced_dof"]

    rng = np.random.default_rng(42)
    added_mass = rng.random((NFREQ, NDOF, NDOF)) + 0.5
    radiation_damping = rng.random((NFREQ, NDOF, NDOF)) + 0.5
    diffraction_force = (rng.random((NFREQ, NDIR, NDOF))
                         + 1j * rng.random((NFREQ, NDIR, NDOF)))
    Froude_Krylov_force = (rng.random((NFREQ, NDIR, NDOF))
                           + 1j * rng.random((NFREQ, NDIR, NDOF)))
    excitation_force = diffraction_force + Froude_Krylov_force
    inertia_matrix = np.eye(NDOF) * 2.0
    hydrostatic_stiffness = np.eye(NDOF) * 3.0

    data_vars = {
        "added_mass": (radiation_dims, added_mass),
        "radiation_damping": (radiation_dims, radiation_damping),
        "diffraction_force": (excitation_dims, diffraction_force),
        "Froude_Krylov_force": (excitation_dims, Froude_Krylov_force),
        "excitation_force": (excitation_dims, excitation_force),
        "inertia_matrix": (hydrostatics_dims, inertia_matrix),
        "hydrostatic_stiffness": (hydrostatics_dims, hydrostatic_stiffness),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.fixture(scope="module")
def hydro_data(bem_data):
    """Replicate the exact pipeline that WEC.from_bem uses internally."""
    hd = wot.add_linear_friction(bem_data, FRICTION)
    hd = wot.check_radiation_damping(hd, min_damping=1e-6)
    return hd


@pytest.fixture(scope="module")
def wec(bem_data):
    return wot.WEC.from_bem(bem_data, friction=FRICTION, min_damping=1e-6)


@pytest.fixture(scope="module")
def wave():
    freq = 2 * F1
    w = wot.waves.regular_wave(F1, NFREQ, freq, 1.0, 30.0)
    return w.sel(realization=0)


@pytest.fixture(scope="module")
def bp(hydro_data):
    return extract_bem_params(hydro_data)


@pytest.fixture(scope="module")
def wd(wave, hydro_data):
    return extract_wave_data(wave, hydro_data["Froude_Krylov_force"])


@pytest.fixture(scope="module")
def x_wec(wec):
    rng = np.random.default_rng(7)
    return jnp.array(rng.standard_normal(wec.nstate_wec))


@pytest.fixture(scope="module")
def x_opt():
    return jnp.array([])


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestExtraction:

    def test_bem_params_shapes(self, bp):
        assert bp.added_mass.shape == (NFREQ, NDOF, NDOF)
        assert bp.radiation_damping.shape == (NFREQ, NDOF, NDOF)
        assert bp.hydrostatic_stiffness.shape == (NDOF, NDOF)
        assert bp.friction.shape == (NDOF, NDOF)
        assert bp.Froude_Krylov_force.shape == (NFREQ, NDIR, NDOF)
        assert bp.diffraction_force.shape == (NFREQ, NDIR, NDOF)
        assert bp.inertia_matrix.shape == (NDOF, NDOF)

    def test_bem_params_are_jax_arrays(self, bp):
        for name in BEMParams._fields:
            arr = getattr(bp, name)
            assert isinstance(arr, jnp.ndarray), f"{name} is {type(arr)}"

    def test_wave_data_shape(self, wd):
        assert wd.wave_elev.shape[0] == NFREQ
        assert isinstance(wd.wave_elev, jnp.ndarray)
        assert len(wd.sub_ind) > 0


class TestForceMatch:
    """Each parametric force must match the original closure-based force."""

    def test_radiation(self, wec, wave, bp, x_wec, x_opt):
        original = wec.forces["radiation"](wec, x_wec, x_opt, wave)
        omega = jnp.array(wec.omega[1:])
        tmat = jnp.array(wec.time_mat)
        param = _radiation_force(tmat, x_wec, wec.ndof, omega,
                                 bp.added_mass, bp.radiation_damping)
        assert jnp.allclose(original, param, atol=1e-10)

    def test_friction(self, wec, wave, bp, x_wec, x_opt):
        original = wec.forces["friction"](wec, x_wec, x_opt, wave)
        omega = jnp.array(wec.omega[1:])
        tmat = jnp.array(wec.time_mat)
        param = _friction_force(tmat, x_wec, wec.ndof, omega, bp.friction)
        assert jnp.allclose(original, param, atol=1e-10)

    def test_hydrostatic(self, wec, wave, bp, x_wec, x_opt):
        original = wec.forces["hydrostatics"](wec, x_wec, x_opt, wave)
        tmat = jnp.array(wec.time_mat)
        param = _hydrostatic_force(tmat, x_wec, wec.ndof, wec.nfreq,
                                   bp.hydrostatic_stiffness)
        assert jnp.allclose(original, param, atol=1e-10)

    @pytest.mark.parametrize("name,coeff", [
        ("Froude_Krylov", "Froude_Krylov_force"),
        ("diffraction", "diffraction_force"),
    ])
    def test_excitation(self, wec, wave, bp, wd, x_wec, x_opt,
                        name, coeff):
        original = wec.forces[name](wec, x_wec, x_opt, wave)
        tmat = jnp.array(wec.time_mat)
        param = _excitation_force(tmat, getattr(bp, coeff), wd)
        assert jnp.allclose(original, param, atol=1e-10)


class TestResidual:

    def test_matches_original(self, wec, wave, bp, wd, x_wec, x_opt):
        original = wec.residual(x_wec, x_opt, wave)
        param = residual_parametric(x_wec, x_opt, wd, bp, wec)
        assert jnp.allclose(original, param, atol=1e-10)


class TestDifferentiation:

    def test_jacrev_finite_nonzero(self, wec, bp, wd, x_wec, x_opt):
        def r_of_h(h):
            return residual_parametric(x_wec, x_opt, wd, h, wec)

        jac = jax.jacrev(r_of_h)(bp)
        for name in BEMParams._fields:
            J = getattr(jac, name)
            assert jnp.all(jnp.isfinite(J)), f"Non-finite in d_r/d_{name}"

        any_nonzero = any(
            float(jnp.max(jnp.abs(getattr(jac, n)))) > 0
            for n in BEMParams._fields)
        assert any_nonzero, "All Jacobians are zero"

    def test_vjp_finite(self, wec, bp, wd, x_wec, x_opt):
        def r_of_h(h):
            return residual_parametric(x_wec, x_opt, wd, h, wec)

        r_val, vjp_fn = jax.vjp(r_of_h, bp)
        rng = np.random.default_rng(7)
        lam = jnp.array(rng.standard_normal(r_val.shape))
        (grad_h,) = vjp_fn(lam)

        for name in BEMParams._fields:
            g = getattr(grad_h, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"

    def test_jit_matches(self, wec, bp, wd, x_wec, x_opt):
        @jax.jit
        def r_jitted(xw, xo, h):
            return residual_parametric(xw, xo, wd, h, wec)

        r1 = r_jitted(x_wec, x_opt, bp)
        r2 = residual_parametric(x_wec, x_opt, wd, bp, wec)
        assert jnp.allclose(r1, r2, atol=1e-12)
