"""Tests for validation utilities.

Covers fd_validate, fd_check_residual, fd_check_objective, and
CrossCheckResult.  The full cross_check_fiacco_ffo integration test
lives in test_wavebot_sensitivity.py (requires Capytaine + converging
IPOPT solve).

Run with::

    pytest tests/test_validation.py -v
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import xarray as xr
from collections import namedtuple

import wecopttool as wot
from wecopttool_differentiable import (
    WEC_IPOPT,
    sensitivity,
    extract_bem_params,
    fd_validate,
    fd_check_residual,
    fd_check_objective,
    FDResult,
    CrossCheckResult,
)
from wecopttool_differentiable.parametric import extract_wave_data
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

PTOParams = namedtuple("PTOParams", [
    "friction_pto", "inertia_pto", "spring_stiffness",
    "winding_resistance", "torque_coefficient",
])

MooringParams = namedtuple("MooringParams", ["K"])


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
def wave():
    return wot.waves.regular_wave(F1, NFREQ, 2 * F1, 1.0, 30.0)


@pytest.fixture(scope="module")
def wec_and_res(bem_data, wave):
    """Build a simple WEC and solve to get a result for validation tests."""
    MooringParams_ = namedtuple("MooringParams_", ["K"])
    K_mooring = 1e3

    def f_mooring(wec, x_wec, x_opt, w, nsubsteps=1):
        pos = wec.vec_to_dofmat(x_wec)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        return jnp.dot(time_matrix, -K_mooring * pos)

    f_add = {"mooring": f_mooring}

    wec = WEC_IPOPT.from_bem(
        bem_data, friction=FRICTION, f_add=f_add, min_damping=1e-6)

    nstate_opt = 2 * NFREQ
    controller = wot.controllers.unstructured_controller()
    kinematics = np.eye(NDOF)
    pto = wot.pto.PTO(NDOF, kinematics, controller, None, None, ["PTO"])

    res = wec.solve(
        wave,
        pto.average_power,
        nstate_opt,
        x_wec_0=np.ones(wec.nstate_wec) * 1e-3,
        x_opt_0=np.ones(nstate_opt) * 1e-3,
        scale_x_wec=1.0,
        scale_x_opt=1.0,
        scale_obj=1e-2,
        optim_options={"max_iter": 500, "tol": 1e-7, "print_level": 0},
    )[0]

    return wec, res, pto


class TestFDResult:
    def test_namedtuple(self):
        r = FDResult("test", 1.0, 1.01, 0.01, True)
        assert r.name == "test"
        assert r.passed is True


class TestCrossCheckResult:
    def test_namedtuple(self):
        r = CrossCheckResult("K", 1.23e-3, 1.24e-3, 8.1e-3, True)
        assert r.name == "K"
        assert r.fiacco == 1.23e-3
        assert r.ffo_chain == 1.24e-3
        assert r.passed is True


class TestFdCheckResidual:
    def test_mooring_residual_matches_fd(self, wec_and_res, wave):
        wec, res, _ = wec_and_res
        params = MooringParams(K=jnp.float64(1e3))
        f_mooring_p = make_linear_mooring_parametric()

        results = fd_check_residual(
            wec, res, wave,
            params=params,
            parametric_forces={"mooring": f_mooring_p},
            additional_forces={
                k: v for k, v in wec.forces.items() if k != "mooring"},
            tol=0.01,
            verbose=True,
        )

        for r in results.values():
            assert r.passed, f"{r.name}: rel_error={r.rel_error:.2e}"


class TestFdCheckObjective:
    def test_simple_parametric_obj(self, wec_and_res, wave):
        wec, res, pto = wec_and_res

        SimpleParams = namedtuple("SimpleParams", ["scale"])
        params = SimpleParams(scale=jnp.float64(1.0))

        def obj_fn(wec, x_wec, x_opt, wave, p):
            vel = pto.velocity(wec, x_wec, x_opt, wave)
            force = pto.force(wec, x_wec, x_opt, wave)
            return p.scale * jnp.sum(vel * force) * wec.dt / wec.tf

        results = fd_check_objective(
            wec, res, wave,
            params=params,
            obj_fn=obj_fn,
            tol=0.01,
            verbose=True,
        )

        for r in results.values():
            assert r.passed, f"{r.name}: rel_error={r.rel_error:.2e}"


class TestFdValidate:
    def test_returns_dict_of_fdresults(self):
        Params = namedtuple("Params", ["a", "b"])
        grad = Params(a=2.0, b=-3.0)
        params = Params(a=1.0, b=2.0)

        def re_solve_fn(d):
            return d["a"] ** 2 - 3 * d["b"]

        results = fd_validate(grad, params, re_solve_fn, verbose=True)

        assert isinstance(results, dict)
        assert set(results.keys()) == {"a", "b"}
        for r in results.values():
            assert isinstance(r, FDResult)
            assert r.passed, f"{r.name}: rel_error={r.rel_error:.2e}"


