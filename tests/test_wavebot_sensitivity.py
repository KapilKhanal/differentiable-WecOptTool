"""Integration tests: WaveBot Fiacco sensitivity with real BEM geometry.

These tests use the actual WaveBot mesh + Capytaine BEM solver,
so they validate the full pipeline on a real geometry — not synthetic data.

The two key validations are:

1. **VJP validation** — perturb one scalar BEM entry, compute the
   residual-level finite difference, and compare ``lambda^T dr/dh``
   with the analytical VJP.  Expected: machine-precision agreement.

2. **NLP pipeline validation** — perturb one scalar BEM entry,
   rebuild the WEC, re-solve IPOPT from a warm start, and compare
   ``(phi*_pert - phi*) / eps`` with the analytical gradient.
   Expected: relative error < 5 % (finite-difference limited).

Run with::

    pytest tests/test_wavebot_sensitivity.py -v
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
    extract_wave_data,
    residual_parametric,
)
from wecopttool.core import frequency_parameters, standard_forces

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — real WaveBot geometry
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def wavebot_setup():
    """Build everything once: BEM, PTO, WEC, solve, extract grads."""
    import capytaine as cpy
    from capytaine.io.meshio import load_from_meshio

    # -- wave environment --
    wavefreq = 0.3
    f1 = wavefreq
    nfreq = 10
    freq = wot.frequency(f1, nfreq, False)
    amplitude = 0.0625
    phase = 30
    wavedir = 0
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

    # -- WaveBot mesh + BEM --
    wb = wot.geom.WaveBot()
    mesh = wb.mesh(0.5)
    mesh_obj = load_from_meshio(mesh, "WaveBot")
    lid_mesh = mesh_obj.generate_lid(-2e-2)
    fb = cpy.FloatingBody(mesh=mesh_obj, lid_mesh=lid_mesh, name="WaveBot")
    fb.add_translation_dof(name="Heave")
    ndof = fb.nb_dofs
    bem_data = wot.run_bem(fb, freq)

    # -- PTO --
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

    # -- WEC_IPOPT --
    wec = WEC_IPOPT.from_bem(bem_data, constraints=constraints,
                              friction=None, f_add=f_add)

    # -- solve --
    obj_fun = pto.mechanical_average_power
    nstate_opt = 2 * nfreq
    scale_x_wec = 1e1
    scale_x_opt = 1e-3
    scale_obj = 1e-2
    ipopt_options = {"max_iter": 1000, "tol": 1e-8, "print_level": 0}

    results = wec.solve(
        waves, obj_fun, nstate_opt,
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt,
        scale_obj=scale_obj, optim_options=ipopt_options,
    )
    res = results[0]
    assert res.success, f"IPOPT failed: {res.message}"

    # -- extract differentiable quantities --
    hydro_data = wot.add_linear_friction(bem_data, friction=None)
    hydro_data = wot.check_radiation_damping(hydro_data)
    bp = extract_bem_params(hydro_data)
    wave = waves.sel(realization=0)
    wd = extract_wave_data(wave, hydro_data["Froude_Krylov_force"])

    x_wec, x_opt = wec.decompose_state(res.x)
    x_wec = jnp.array(x_wec)
    x_opt = jnp.array(x_opt)
    lam = jnp.array(res.dynamics_mult_g)

    def r_of_h(h):
        return residual_parametric(x_wec, x_opt, wd, h, wec)

    _, vjp_fn = jax.vjp(r_of_h, bp)
    (grad_h,) = vjp_fn(lam)
    # Fix JAX complex VJP convention: conjugate complex leaves
    grad_h = jax.tree_util.tree_map(
        lambda x: jnp.conj(x) if jnp.iscomplexobj(x) else x, grad_h)

    return dict(
        wec=wec, waves=waves, wave=wave, res=res, bem_data=bem_data,
        hydro_data=hydro_data, bp=bp, wd=wd, grad_h=grad_h,
        x_wec=x_wec, x_opt=x_opt, lam=lam,
        obj_fun=obj_fun, nstate_opt=nstate_opt, pto=pto,
        constraints=constraints,
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
    )


# All 7 BEM parameter fields with one representative scalar index each
TEST_CASES = [
    ("added_mass",            (0, 0, 0)),
    ("radiation_damping",     (0, 0, 0)),
    ("hydrostatic_stiffness", (0, 0)),
    ("friction",              (0, 0)),
    ("Froude_Krylov_force",   (0, 0, 0)),
    ("diffraction_force",     (0, 0, 0)),
    ("inertia_matrix",        (0, 0)),
]


# ═══════════════════════════════════════════════════════════════════════════
# VJP validation (residual-level central finite difference)
# ═══════════════════════════════════════════════════════════════════════════

class TestWaveBotVJPValidation:
    """Residual-level FD vs analytical VJP for each BEM parameter."""

    @pytest.mark.parametrize("param_name,idx", TEST_CASES,
                             ids=[t[0] for t in TEST_CASES])
    def test_vjp_fd(self, wavebot_setup, param_name, idx):
        s = wavebot_setup
        bp, wec, wd = s["bp"], s["wec"], s["wd"]
        x_wec, x_opt, lam = s["x_wec"], s["x_opt"], s["lam"]

        arr = getattr(bp, param_name)
        h0_real = float(jnp.real(arr[idx]))
        eps = 1e-5 * max(abs(h0_real), 1.0)

        def _r_pert(delta):
            a = getattr(bp, param_name)
            a_new = a.at[idx].set(a[idx] + delta)
            bp_new = bp._replace(**{param_name: a_new})
            return residual_parametric(x_wec, x_opt, wd, bp_new, wec)

        dr_dh = (_r_pert(+eps) - _r_pert(-eps)) / (2.0 * eps)
        fd_val = float(jnp.dot(lam, dr_dh))
        anal_val = float(jnp.real(getattr(s["grad_h"], param_name)[idx]))

        if abs(anal_val) > 1e-15:
            rel_err = abs(fd_val - anal_val) / abs(anal_val)
        else:
            rel_err = abs(fd_val - anal_val)

        assert rel_err < 1e-3, (
            f"VJP validation failed for {param_name}{list(idx)}: "
            f"analytical={anal_val:.6e}, FD={fd_val:.6e}, rel_err={rel_err:.2e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# NLP pipeline validation (full IPOPT re-solve with perturbation)
# ═══════════════════════════════════════════════════════════════════════════

# Dimension transpose map (must match extract_bem_params ordering)
_DIMS_MAP = {
    "added_mass":        ("omega", "radiating_dof", "influenced_dof"),
    "radiation_damping": ("omega", "radiating_dof", "influenced_dof"),
}


def _perturb_and_solve(setup, param_name, idx, epsilon):
    """Perturb one scalar entry in hydro_data, rebuild WEC, re-solve."""
    s = setup
    hd = s["hydro_data"].copy(deep=True)

    dims = _DIMS_MAP.get(param_name)
    if dims is not None:
        da = hd[param_name].transpose(*dims)
    else:
        da = hd[param_name]
    vals = da.values.copy()
    vals[idx] += epsilon
    hd[param_name] = da.copy(data=vals)

    inertia_p = hd["inertia_matrix"].values
    f1_p, nfreq_p = frequency_parameters(hd.omega.values / (2 * np.pi), False)
    forces_p = standard_forces(hd) | {"PTO": s["pto"].force_on_wec}
    wec_p = WEC_IPOPT(f1_p, nfreq_p, forces_p, s["constraints"], inertia_p)

    x_wec_0 = np.array(s["res"].x[:s["wec"].nstate_wec])
    x_opt_0 = np.array(s["res"].x[s["wec"].nstate_wec:])

    res_p = wec_p.solve(
        s["waves"], s["obj_fun"], s["nstate_opt"],
        x_wec_0=x_wec_0, x_opt_0=x_opt_0,
        scale_x_wec=s["scale_x_wec"], scale_x_opt=s["scale_x_opt"],
        scale_obj=s["scale_obj"],
        optim_options={
            "max_iter": 5000, "tol": 1e-8, "print_level": 0,
            "warm_start_init_point": "yes", "mu_init": 1e-6,
        },
    )[0]
    return res_p


# Exclude friction (zero baseline → large nonlinear FD error, VJP still valid)
NLP_TEST_CASES = [t for t in TEST_CASES if t[0] != "friction"]


class TestWaveBotNLPValidation:
    """Full IPOPT re-solve FD vs analytical gradient for each BEM parameter."""

    @pytest.mark.validation
    @pytest.mark.parametrize("param_name,idx", NLP_TEST_CASES,
                             ids=[t[0] for t in NLP_TEST_CASES])
    def test_nlp_fd(self, wavebot_setup, param_name, idx):
        s = wavebot_setup
        bp = s["bp"]
        phi_star = s["res"].fun

        h0_real = float(jnp.real(getattr(bp, param_name)[idx]))
        eps = 1e-4 * max(abs(h0_real), 1.0)

        res_p = _perturb_and_solve(s, param_name, idx, eps)
        fd_grad = (res_p.fun - phi_star) / eps
        anal_grad = float(jnp.real(getattr(s["grad_h"], param_name)[idx]))

        if abs(anal_grad) > 1e-12:
            rel_err = abs(fd_grad - anal_grad) / abs(anal_grad)
        else:
            rel_err = abs(fd_grad - anal_grad)

        assert rel_err < 0.05, (
            f"NLP validation failed for {param_name}{list(idx)}: "
            f"analytical={anal_grad:.6e}, FD={fd_grad:.6e}, "
            f"rel_err={rel_err:.4e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API tests on real geometry
# ═══════════════════════════════════════════════════════════════════════════

class TestWaveBotConvenienceAPI:
    """Test sensitivity() and make_differentiable_solver on WaveBot."""

    def test_sensitivity_matches_manual(self, wavebot_setup):
        s = wavebot_setup
        grad_sens = sensitivity(s["wec"], s["res"], s["waves"])

        for name in BEMParams._fields:
            np.testing.assert_allclose(
                np.array(getattr(grad_sens, name)),
                np.array(getattr(s["grad_h"], name)),
                atol=1e-12,
                err_msg=f"sensitivity() mismatch for {name}",
            )

    def test_make_differentiable_solver_grad(self, wavebot_setup):
        s = wavebot_setup
        f = make_differentiable_solver(
            s["wec"], s["waves"], s["obj_fun"], s["nstate_opt"],
            scale_x_wec=s["scale_x_wec"], scale_x_opt=s["scale_x_opt"],
            scale_obj=s["scale_obj"],
            optim_options={"max_iter": 1000, "tol": 1e-8, "print_level": 0},
        )
        grad_vjp = jax.grad(f)(s["bp"])

        for name in BEMParams._fields:
            g = getattr(grad_vjp, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"
            # Fresh IPOPT solve -> x*, lambda* differ at solver tolerance
            np.testing.assert_allclose(
                np.array(g),
                np.array(getattr(s["grad_h"], name)),
                atol=1e-5,
                err_msg=f"make_differentiable_solver mismatch for {name}",
            )
