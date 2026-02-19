"""Tests for make_differentiable_solver (JAX custom_vjp path).

Validates:
1. Forward pass returns the same objective value as a direct wec.solve().
2. jax.grad through the solver returns a gradient matching sensitivity().
3. Warm-start state is populated after the first call.

Uses the WaveBot geometry (faster than AquaHarmonics).

Run with::

    pytest tests/test_differentiable_solver.py -v
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import wecopttool as wot
from wecopttool_differentiable import (
    WEC_IPOPT,
    make_differentiable_solver,
    sensitivity,
    BEMParams,
    extract_bem_params,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def wb_setup():
    """Build WaveBot, solve, and prepare differentiable solver."""
    import capytaine as cpy
    from capytaine.io.meshio import load_from_meshio

    wavefreq = 0.3
    f1 = wavefreq
    nfreq = 10
    freq = wot.frequency(f1, nfreq, False)
    amplitude = 0.0625
    phase = 30
    wavedir = 0
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

    wb = wot.geom.WaveBot()
    mesh = wb.mesh(0.5)
    mesh_obj = load_from_meshio(mesh, "WaveBot")
    lid_mesh = mesh_obj.generate_lid(-2e-2)
    fb = cpy.FloatingBody(mesh=mesh_obj, lid_mesh=lid_mesh, name="WaveBot")
    fb.add_translation_dof(name="Heave")
    ndof = fb.nb_dofs
    bem_data = wot.run_bem(fb, freq)

    name = ["PTO_Heave"]
    kinematics = np.eye(ndof)
    controller = wot.controllers.unstructured_controller()
    pto = wot.pto.PTO(ndof, kinematics, controller, None, None, name)
    f_add = {"PTO": pto.force_on_wec}

    f_max = 750.0
    nsubsteps = 4

    def const_f_pto(wec, x_wec, x_opt, wave):
        f = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
        return f_max - jnp.abs(f.flatten())

    constraints = [{"type": "ineq", "fun": const_f_pto}]

    wec = WEC_IPOPT.from_bem(bem_data, constraints=constraints,
                              friction=None, f_add=f_add)

    obj_fun = pto.mechanical_average_power
    nstate_opt = 2 * nfreq
    scale_x_wec = 1e1
    scale_x_opt = 1e-3
    scale_obj = 1e-2

    solve_kw = dict(
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        optim_options={"max_iter": 1000, "tol": 1e-8, "print_level": 0},
    )

    results = wec.solve(waves, obj_fun, nstate_opt, **solve_kw)
    res = results[0]
    assert res.success, f"IPOPT failed: {res.message}"

    hydro_data = wot.add_linear_friction(bem_data, friction=None)
    hydro_data = wot.check_radiation_damping(hydro_data)
    bp = extract_bem_params(hydro_data)

    return dict(
        wec=wec, waves=waves, res=res, bem_data=bem_data,
        hydro_data=hydro_data, bp=bp,
        obj_fun=obj_fun, nstate_opt=nstate_opt,
        solve_kw=solve_kw,
    )


class TestDifferentiableSolverForward:
    """Forward pass of make_differentiable_solver."""

    def test_forward_matches_solve(self, wb_setup):
        """Forward pass objective matches direct wec.solve()."""
        s = wb_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            **s["solve_kw"],
        )
        phi_diff = float(f(s["bp"]))
        phi_direct = s["res"].fun

        np.testing.assert_allclose(
            phi_diff, phi_direct, rtol=1e-4,
            err_msg="Forward pass objective differs from direct solve",
        )

    def test_forward_returns_scalar(self, wb_setup):
        s = wb_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            **s["solve_kw"],
        )
        result = f(s["bp"])
        assert result.shape == (), "Forward pass should return a scalar"
        assert jnp.isfinite(result), "Forward pass returned non-finite"


class TestDifferentiableSolverBackward:
    """Backward pass (jax.grad) of make_differentiable_solver."""

    def test_grad_matches_sensitivity(self, wb_setup):
        """jax.grad through the solver matches standalone sensitivity()."""
        s = wb_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            **s["solve_kw"],
        )
        grad_vjp = jax.grad(f)(s["bp"])
        grad_sens = sensitivity(s["wec"], s["res"], s["waves"])

        for name in BEMParams._fields:
            g_vjp = np.array(getattr(grad_vjp, name))
            g_sens = np.array(getattr(grad_sens, name))
            np.testing.assert_allclose(
                g_vjp, g_sens, atol=1e-5,
                err_msg=f"Gradient mismatch for {name}",
            )

    def test_grad_all_finite(self, wb_setup):
        s = wb_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            **s["solve_kw"],
        )
        grad_vjp = jax.grad(f)(s["bp"])

        for name in BEMParams._fields:
            g = getattr(grad_vjp, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"


class TestDifferentiableSolverWarmStart:
    """Warm-start state is populated after calls."""

    def test_warm_start_populated(self, wb_setup):
        s = wb_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            **s["solve_kw"],
        )
        f(s["bp"])

        assert f.warm_start_state["x_wec_0"] is not None, \
            "x_wec_0 not set after forward pass"
        assert f.warm_start_state["x_opt_0"] is not None, \
            "x_opt_0 not set after forward pass"

    def test_warm_start_shapes(self, wb_setup):
        s = wb_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            **s["solve_kw"],
        )
        f(s["bp"])

        x_wec_0 = f.warm_start_state["x_wec_0"]
        x_opt_0 = f.warm_start_state["x_opt_0"]
        assert x_wec_0.shape == (s["wec"].nstate_wec,)
        assert x_opt_0.shape == (s["nstate_opt"],)
