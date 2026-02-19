"""Tests for parametric_utils factory functions.

Run with::

    pytest tests/test_parametric_utils.py -v
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import xarray as xr
from collections import namedtuple

import wecopttool as wot
from wecopttool_differentiable.solver_ipopt import WEC_IPOPT
from wecopttool_differentiable.parametric_utils import (
    make_linear_mooring_parametric,
    make_pto_passive_parametric,
    make_electrical_power_obj_parametric,
)

jax.config.update("jax_enable_x64", True)

F1 = 0.12
NFREQ = 5
NDOF = 1
FRICTION = np.eye(NDOF) * 0.5


@pytest.fixture(scope="module")
def bem_data():
    coords = {
        "omega": [2 * np.pi * (i + 1) * F1 for i in range(NFREQ)],
        "influenced_dof": ["Heave"],
        "radiating_dof": ["Heave"],
        "wave_direction": [0.0],
    }
    radiation_dims = ["omega", "radiating_dof", "influenced_dof"]
    excitation_dims = ["omega", "wave_direction", "influenced_dof"]
    hydrostatics_dims = ["radiating_dof", "influenced_dof"]

    rng = np.random.default_rng(42)
    added_mass = rng.random((NFREQ, NDOF, NDOF)) + 0.5
    radiation_damping = rng.random((NFREQ, NDOF, NDOF)) + 0.5
    diffraction_force = (rng.random((NFREQ, 1, NDOF))
                        + 1j * rng.random((NFREQ, 1, NDOF)))
    Froude_Krylov_force = (rng.random((NFREQ, 1, NDOF))
                           + 1j * rng.random((NFREQ, 1, NDOF)))
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
def wec(bem_data):
    return WEC_IPOPT.from_bem(bem_data, friction=FRICTION, min_damping=1e-6)


@pytest.fixture(scope="module")
def wave():
    freq = 2 * F1
    w = wot.waves.regular_wave(F1, NFREQ, freq, 1.0, 30.0)
    return w


class TestMakeLinearMooringParametric:
    def test_returns_callable(self):
        fn = make_linear_mooring_parametric()
        assert callable(fn)

    def test_output_shape_mooring_only(self, wec, wave):
        MooringParams = namedtuple("MooringParams", ["K"])
        params = MooringParams(K=jnp.float64(1e4))
        fn = make_linear_mooring_parametric()

        rng = np.random.default_rng(1)
        x_wec = jnp.array(rng.standard_normal(wec.nstate_wec))
        nstate_opt = wec.ncomponents * wec.ndof
        x_opt = jnp.array(rng.standard_normal(nstate_opt))
        from wecopttool_differentiable.parametric import extract_wave_data
        wav = wave.sel(realization=0)
        wd = extract_wave_data(wav, wec._hydro_data["Froude_Krylov_force"])

        force = fn(wec, x_wec, x_opt, wd, params)
        assert force.shape == (wec.ncomponents, wec.ndof)

    def test_differentiable_wrt_K(self, wec, wave):
        MooringParams = namedtuple("MooringParams", ["K"])
        params = MooringParams(K=jnp.float64(1e4))
        fn = make_linear_mooring_parametric()

        from wecopttool_differentiable.parametric import extract_wave_data
        wav = wave.sel(realization=0)
        wd = extract_wave_data(wav, wec._hydro_data["Froude_Krylov_force"])
        rng = np.random.default_rng(1)
        x_wec = jnp.array(rng.standard_normal(wec.nstate_wec))
        nstate_opt = wec.ncomponents * wec.ndof
        x_opt = jnp.array(rng.standard_normal(nstate_opt))

        def scalar_out(p):
            f = fn(wec, x_wec, x_opt, wd, p)
            return jnp.sum(f ** 2)

        grad = jax.grad(scalar_out)(params)
        assert jnp.isfinite(grad.K)
        assert jnp.abs(grad.K) > 0


class TestMakePtoPassiveParametric:
    def test_returns_callable(self):
        fn = make_pto_passive_parametric(
            gear_ratios={"spring": 1.0},
            friction_dict={"Bpneumatic_spring_static1": 0.1},
        )
        assert callable(fn)

    def test_output_shape(self, wec, wave):
        PTOParams = namedtuple("PTOParams", [
            "friction_pto", "inertia_pto", "spring_stiffness",
            "winding_resistance", "torque_coefficient",
        ])
        params = PTOParams(
            friction_pto=jnp.float64(100.0),
            inertia_pto=jnp.float64(50.0),
            spring_stiffness=jnp.float64(5000.0),
            winding_resistance=jnp.float64(0.4),
            torque_coefficient=jnp.float64(1.5),
        )
        fn = make_pto_passive_parametric(
            gear_ratios={"spring": 1.0},
            friction_dict={"Bpneumatic_spring_static1": 0.0},
        )

        from wecopttool_differentiable.parametric import extract_wave_data
        wav = wave.sel(realization=0)
        wd = extract_wave_data(wav, wec._hydro_data["Froude_Krylov_force"])
        rng = np.random.default_rng(1)
        x_wec = jnp.array(rng.standard_normal(wec.nstate_wec))
        nstate_opt = wec.ncomponents * wec.ndof
        x_opt = jnp.array(rng.standard_normal(nstate_opt))

        force = fn(wec, x_wec, x_opt, wd, params)
        assert force.shape == (wec.ncomponents, wec.ndof)


class TestMakeElectricalPowerObjParametric:
    def test_returns_callable(self):
        class MockPTO:
            def velocity(self, wec, x_wec, x_opt, wave, nsubsteps=1):
                return jnp.zeros((wec.ncomponents * nsubsteps, wec.ndof))

            def force(self, wec, x_wec, x_opt, wave, nsubsteps=1):
                return jnp.zeros((wec.ncomponents * nsubsteps, wec.ndof))

        pto = MockPTO()
        fn = make_electrical_power_obj_parametric(pto)
        assert callable(fn)

    def test_returns_scalar(self, wec, wave):
        class MockPTO:
            def velocity(self, wec, x_wec, x_opt, wave, nsubsteps=1):
                return jnp.ones((wec.ncomponents * nsubsteps, wec.ndof))

            def force(self, wec, x_wec, x_opt, wave, nsubsteps=1):
                return jnp.ones((wec.ncomponents * nsubsteps, wec.ndof))

        PTOParams = namedtuple("PTOParams", [
            "friction_pto", "inertia_pto", "spring_stiffness",
            "winding_resistance", "torque_coefficient",
        ])
        params = PTOParams(
            friction_pto=jnp.float64(1.0),
            inertia_pto=jnp.float64(1.0),
            spring_stiffness=jnp.float64(1.0),
            winding_resistance=jnp.float64(0.4),
            torque_coefficient=jnp.float64(1.5),
        )
        pto = MockPTO()
        fn = make_electrical_power_obj_parametric(pto)

        wav = wave.sel(realization=0)
        rng = np.random.default_rng(1)
        x_wec = jnp.array(rng.standard_normal(wec.nstate_wec))
        nstate_opt = wec.ncomponents * wec.ndof
        x_opt = jnp.array(rng.standard_normal(nstate_opt))

        out = fn(wec, x_wec, x_opt, wav, params)
        assert jnp.ndim(out) == 0
        assert jnp.isfinite(out)
