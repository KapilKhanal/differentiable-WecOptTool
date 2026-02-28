"""Test KKT backward differentiation against Fiacco on Tutorial 1 WaveBot.

Tutorial 1 is a pure convex QP (force constraint only, no power constraint),
so the KKT backward should produce gradients matching Fiacco exactly.
"""
from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import wecopttool as wot

jax.config.update("jax_enable_x64", True)

from wecopttool_differentiable import (
    WEC_IPOPT,
    sensitivity,
    make_differentiable_solver,
    BEMParams,
    extract_bem_params,
    extract_wave_data,
    kkt_vjp,
)
from wecopttool_differentiable.solver_ipopt import _extract_all_realizations


@pytest.fixture(scope="module")
def tutorial1_setup():
    """Build and solve Tutorial 1 WaveBot."""
    import capytaine as cpy
    from capytaine.io.meshio import load_from_meshio

    wavefreq = 0.3
    f1, nfreq = wavefreq, 10
    freq = wot.frequency(f1, nfreq, False)
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, 0.0625, 30, 0)

    wb = wot.geom.WaveBot()
    mesh = wb.mesh(0.5)
    fb = cpy.FloatingBody(
        mesh=load_from_meshio(mesh, "WaveBot"),
        lid_mesh=load_from_meshio(mesh, "WaveBot").generate_lid(-2e-2),
        name="WaveBot",
    )
    fb.add_translation_dof(name="Heave")
    ndof = fb.nb_dofs
    bem_data = wot.run_bem(fb, freq)

    pto = wot.pto.PTO(
        ndof, np.eye(ndof), wot.controllers.unstructured_controller(),
        None, None, ["PTO_Heave"],
    )

    f_max = 750.0
    nsubsteps = 4

    def const_f_pto(wec, x_wec, x_opt, wave):
        f = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
        return f_max - jnp.abs(f.flatten())

    wec = WEC_IPOPT.from_bem(
        bem_data,
        constraints=[{"type": "ineq", "fun": const_f_pto}],
        friction=None,
        f_add={"PTO": pto.force_on_wec},
    )

    obj_fun = pto.mechanical_average_power
    nstate_opt = 2 * nfreq

    results = wec.solve(
        waves, obj_fun, nstate_opt,
        scale_x_wec=1e1, scale_x_opt=1e-3, scale_obj=1e-2,
        optim_options={"max_iter": 1000, "tol": 1e-8, "print_level": 0},
    )
    res = results[0]
    assert res.success, f"IPOPT failed: {res.message}"

    bp = extract_bem_params(wec._hydro_data)

    return {
        "wec": wec, "res": res, "waves": waves,
        "obj_fun": obj_fun, "nstate_opt": nstate_opt,
        "bp": bp, "bem_data": bem_data,
    }


class TestKKTVJP:
    """Test kkt_vjp directly."""

    def test_kkt_vjp_returns_gradient(self, tutorial1_setup):
        s = tutorial1_setup
        wec, res, waves = s["wec"], s["res"], s["waves"]
        bp = s["bp"]

        _, wave_list = _extract_all_realizations(
            waves, wec._hydro_data["Froude_Krylov_force"])
        wd_list, _ = _extract_all_realizations(
            waves, wec._hydro_data["Froude_Krylov_force"])

        wave_0 = wave_list[0]
        wd_0 = wd_list[0]

        vjp_fn, x_opt_star, x_wec_star, info = kkt_vjp(
            wec, res, wave_0, s["obj_fun"], wd_0, bp,
            active_tol=1e-6, sign=1.0)

        assert info["kkt_size"] > 0
        assert info["kkt_cond"] < 1e14

        n_opt = len(x_opt_star)
        v_seed = np.ones(n_opt)
        grad_p = vjp_fn(v_seed)

        assert isinstance(grad_p, BEMParams)
        for name in BEMParams._fields:
            g = getattr(grad_p, name)
            assert jnp.all(jnp.isfinite(g)), f"{name} has non-finite values"

    def test_kkt_agrees_with_fiacco_on_objective(self, tutorial1_setup):
        """KKT state-level sensitivity, contracted to dφ*/dp, should
        agree with Fiacco's dφ*/dp.

        dφ*/dp = (dφ/dx)^T dx*/dp, where x = [x_wec; x_opt].
        """
        s = tutorial1_setup
        wec, res, waves = s["wec"], s["res"], s["waves"]
        bp = s["bp"]

        grad_fiacco = sensitivity(wec, res, waves)

        _, wave_list = _extract_all_realizations(
            waves, wec._hydro_data["Froude_Krylov_force"])
        wd_list, _ = _extract_all_realizations(
            waves, wec._hydro_data["Froude_Krylov_force"])
        wave_0 = wave_list[0]
        wd_0 = wd_list[0]

        x_star = np.array(res.x)
        x_wec, x_opt = wec.decompose_state(x_star)
        n_wec = len(x_wec)

        # Full-space objective gradient: dφ/dx = [dφ/dx_wec; dφ/dx_opt]
        grad_f_x = np.array(jax.grad(
            lambda x: s["obj_fun"](wec, x[:n_wec], x[n_wec:], wave_0)
        )(jnp.array(x_star)))

        vjp_fn, _, _, _ = kkt_vjp(
            wec, res, wave_0, s["obj_fun"], wd_0, bp,
            active_tol=1e-6, sign=1.0)
        grad_kkt = vjp_fn(grad_f_x)

        for name in BEMParams._fields:
            g_f = getattr(grad_fiacco, name)
            g_k = getattr(grad_kkt, name)
            norm_f = float(jnp.linalg.norm(g_f))
            if norm_f < 1e-12:
                continue
            rel_diff = float(jnp.linalg.norm(g_f - g_k)) / norm_f
            assert rel_diff < 0.05, (
                f"KKT disagrees with Fiacco on {name}: rel_diff={rel_diff:.4e}")


class TestMakeDifferentiableSolverKKT:
    """Test make_differentiable_solver with backward_strategy='kkt'."""

    def test_forward_returns_state(self, tutorial1_setup):
        s = tutorial1_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            return_state=True, backward_strategy="kkt",
            scale_x_wec=1e1, scale_x_opt=1e-3, scale_obj=1e-2,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-8},
        )
        bp = s["bp"]
        x_star = f(bp)
        assert x_star.shape == s["res"].x.shape

    def test_grad_returns_bemparams(self, tutorial1_setup):
        s = tutorial1_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            return_state=True, backward_strategy="kkt",
            scale_x_wec=1e1, scale_x_opt=1e-3, scale_obj=1e-2,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-8},
        )
        bp = s["bp"]

        def scalar_of_state(p):
            x = f(p)
            return jnp.sum(x ** 2)

        grad_p = jax.grad(scalar_of_state)(bp)
        assert isinstance(grad_p, BEMParams)
        for name in BEMParams._fields:
            g = getattr(grad_p, name)
            assert jnp.all(jnp.isfinite(g)), f"{name} not finite"

    def test_auto_selects_kkt(self, tutorial1_setup):
        """backward_strategy='auto' should use KKT (the default)."""
        s = tutorial1_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            return_state=True, backward_strategy="auto",
            scale_x_wec=1e1, scale_x_opt=1e-3, scale_obj=1e-2,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-8},
        )
        bp = s["bp"]
        x_star = f(bp)
        assert x_star.shape == s["res"].x.shape
