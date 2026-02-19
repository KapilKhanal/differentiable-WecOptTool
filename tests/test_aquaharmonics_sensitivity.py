"""Integration tests: AquaHarmonics BEM + PTO sensitivity with real geometry.

These tests use the actual AquaHarmonics hull mesh + Capytaine BEM solver,
a realistic drive-train / generator model, and active inequality constraints
(tension, torque, speed, power limits). They validate:

1. **BEM-only sensitivity** — ``sensitivity(wec, res, waves)`` (default BEMParams).
2. **PTO-only sensitivity** — Fiacco formula including residual, objective, and
   constraint multiplier terms for the 5 PTO parameters.
3. **Joint BEM+PTO sensitivity** — single ``sensitivity()`` call on a
   ``JointParams(bem=..., pto=...)`` namedtuple.
4. **VJP residual-level FD** — central FD on λᵀ ∂r/∂p for PTO parameters
   that enter the dynamics (friction, inertia, spring).
5. **Objective FD** — central FD on ∂f/∂p for generator parameters
   (winding_resistance, torque_coefficient).
6. **Full NLP re-solve FD** — perturb each PTO parameter, warm-start IPOPT
   with ``mult_g_0`` (primal + dual), and compare Δφ*/Δp with analytical.

Run with::

    pytest tests/test_aquaharmonics_sensitivity.py -v
"""

import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from collections import namedtuple

import wecopttool as wot
from wecopttool_differentiable import (
    WEC_IPOPT,
    sensitivity,
    sensitivity_parametric,
    BEMParams,
    extract_bem_params,
    extract_wave_data,
    residual_parametric,
    make_joint_params,
    make_linear_mooring_parametric,
    make_pto_passive_parametric,
    make_electrical_power_obj_parametric,
    fd_check_residual,
    fd_check_objective,
    validate_sensitivity,
)
from wecopttool.core import frequency_parameters, standard_forces

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# Named types
# ═══════════════════════════════════════════════════════════════════════════

PTOParams = namedtuple("PTOParams", [
    "friction_pto", "inertia_pto", "spring_stiffness",
    "winding_resistance", "torque_coefficient",
])

JointParams = namedtuple("JointParams", ["bem", "pto"])


# ═══════════════════════════════════════════════════════════════════════════
# Fixture — real AquaHarmonics geometry (module-scoped, built once)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def ah_setup():
    """Build AquaHarmonics device, solve, extract all quantities."""
    import capytaine as cpy
    from capytaine.io.meshio import load_from_meshio

    # -- geometry --
    ah_hull = wot.geom.AquaHarmonics()
    mesh = ah_hull.mesh(mesh_size_factor=0.25)
    mesh_obj = load_from_meshio(mesh, "AquaHarmonics")
    lid_mesh = mesh_obj.generate_lid(-5e-2)
    fb = cpy.FloatingBody(mesh=mesh_obj, lid_mesh=lid_mesh, name="AquaHarmonics")
    fb.add_translation_dof(name="Heave")
    ndof = fb.nb_dofs

    g = 9.81
    rho = 1025
    fb.center_of_mass = [0, 0, 0]
    fb.rotation_center = fb.center_of_mass
    displaced_mass = fb.compute_rigid_body_inertia(rho=rho).values
    displacement = displaced_mass / rho
    fb.mass = np.atleast_2d(5e3)

    # -- waves --
    wavefreq = 0.24 / 2
    f1 = wavefreq
    nfreq = 10
    amplitude = 0.5
    phase = 30
    wavedir = 0
    freq = wot.frequency(f1, nfreq, False)
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

    # -- BEM --
    bem_data = wot.run_bem(fb, freq, rho=rho, g=g)

    # -- drive-train parameters --
    radii = {
        "S1": 0.02, "S2": 0.795, "S3": 0.1595, "S4": 0.200525, "S5": 0.40105,
        "S6": 0.12575, "S7": 0.103,
    }
    inertias = {
        "Igen": 3.9, "I1": 0.029, "I2": 25.6, "I3": 1.43, "I4": 1.165,
        "I5": 4.99, "I6": 1.43, "I7": 1.5, "mps": 40,
    }
    friction_dict = {
        "Bgen": 7, "Bdrivetrain": 40, "Bshaft": 40, "Bspring_pulley": 80,
        "Bpneumatic_spring": 700, "Bpneumatic_spring_static1": 0,
        "Bpspneumatic_spring_static2": 0,
    }
    airspring = {
        "gamma": 1.4, "height": 1, "diameter": 3, "area": 0.0709676,
        "press_init": 854e3, "vol_init": 1,
    }
    gear_ratios = {
        "R21": radii["S2"] / radii["S1"],
        "R45": radii["S4"] / radii["S5"],
        "R67": radii["S6"] / radii["S7"],
        "spring": radii["S6"] * (radii["S4"] / radii["S5"]),
    }

    inertia_pto = (
        (inertias["Igen"] + inertias["I1"]) * gear_ratios["R21"] ** 2
        + (inertias["I2"] + inertias["I3"] + inertias["I4"])
        + gear_ratios["R45"] ** 2 * (
            inertias["I5"] + inertias["I6"]
            + inertias["I7"] * gear_ratios["R67"] ** 2
            + inertias["mps"] * radii["S6"] ** 2
        )
    )
    friction_pto = (
        friction_dict["Bgen"] * gear_ratios["R21"] ** 2
        + friction_dict["Bdrivetrain"]
        + gear_ratios["R45"] ** 2 * (
            friction_dict["Bshaft"]
            + friction_dict["Bspring_pulley"] * gear_ratios["R67"] ** 2
            + friction_dict["Bpneumatic_spring"] * radii["S6"] ** 2
        )
    )
    spring_stiffness = (
        gear_ratios["spring"] * airspring["gamma"]
        * airspring["area"] * airspring["press_init"] / airspring["vol_init"]
    )
    winding_resistance = 0.4
    torque_coefficient = 1.5

    pto_params_nominal = PTOParams(
        friction_pto=jnp.float64(friction_pto),
        inertia_pto=jnp.float64(inertia_pto),
        spring_stiffness=jnp.float64(spring_stiffness),
        winding_resistance=jnp.float64(winding_resistance),
        torque_coefficient=jnp.float64(torque_coefficient),
    )

    # -- PTO and generator --
    def power_loss(speed, torque):
        return winding_resistance * (torque / torque_coefficient) ** 2

    name = ["PTO_Heave"]
    gear_ratio_generator = gear_ratios["R21"] / radii["S3"]
    kinematics = gear_ratio_generator * np.eye(ndof)
    controller = wot.controllers.unstructured_controller()
    nstate_opt = 2 * nfreq
    pto = wot.pto.PTO(ndof, kinematics, controller, None, power_loss, name)
    obj_fun = pto.average_power

    # -- forces --
    def f_buoyancy(wec, x_wec, x_opt, wave, nsubsteps=1):
        return displacement * rho * g * jnp.ones(
            [wec.ncomponents * nsubsteps, wec.ndof])

    def f_gravity(wec, x_wec, x_opt, wave, nsubsteps=1):
        return -1 * wec.inertia_matrix.item() * g * jnp.ones(
            [wec.ncomponents * nsubsteps, wec.ndof])

    def f_pretension_wec(wec, x_wec, x_opt, wave, nsubsteps=1):
        return -1 * (
            f_buoyancy(wec, x_wec, x_opt, wave, nsubsteps)
            + f_gravity(wec, x_wec, x_opt, wave, nsubsteps)
        )

    def f_pto_passive(wec, x_wec, x_opt, wave, nsubsteps=1):
        pos = wec.vec_to_dofmat(x_wec)
        vel = jnp.dot(wec.derivative_mat, pos)
        acc = jnp.dot(wec.derivative_mat, vel)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        spring = -spring_stiffness * pos
        f_spring = jnp.dot(time_matrix, spring)
        fric = -(friction_pto
                 + friction_dict["Bpneumatic_spring_static1"]
                 * gear_ratios["spring"]) * vel
        f_fric = jnp.dot(time_matrix, fric)
        inertia = inertia_pto * acc
        f_inertia = jnp.dot(time_matrix, inertia)
        return f_spring + f_fric + f_inertia

    def f_pto_line(wec, x_wec, x_opt, wave, nsubsteps=1):
        return (pto.force_on_wec(wec, x_wec, x_opt, wave, nsubsteps)
                + f_pretension_wec(wec, x_wec, x_opt, wave, nsubsteps))

    K_mooring = 1e4
    def f_mooring(wec, x_wec, x_opt, wave, nsubsteps=1):
        pos = wec.vec_to_dofmat(x_wec)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        return jnp.dot(time_matrix, -K_mooring * pos)

    f_add = {
        "PTO": f_pto_line,
        "PTO_passive": f_pto_passive,
        "mooring": f_mooring,
        "buoyancy": f_buoyancy,
        "gravity": f_gravity,
    }

    # -- constraints --
    torque_peak_max = 280
    rot_speed_max = 10000 * 2 * np.pi / 60
    min_line_tension = -1000
    power_max = 80e3
    nsubsteps = 2

    def constrain_min_tension(wec, x_wec, x_opt, wave):
        total_tension = -1 * f_pto_line(wec, x_wec, x_opt, wave, nsubsteps)
        return total_tension.flatten() + min_line_tension

    def const_peak_torque(wec, x_wec, x_opt, wave):
        torque = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
        return torque_peak_max - jnp.abs(torque.flatten())

    def const_speed(wec, x_wec, x_opt, wave):
        rot_vel = pto.velocity(wec, x_wec, x_opt, wave, nsubsteps)
        return rot_speed_max - jnp.abs(rot_vel.flatten())

    def const_power(wec, x_wec, x_opt, wave):
        power_mech = (
            pto.velocity(wec, x_wec, x_opt, wave, nsubsteps)
            * pto.force(wec, x_wec, x_opt, wave, nsubsteps)
        )
        return power_max - jnp.abs(power_mech.flatten())

    def zero_mean_pos(wec, x_wec, x_opt, wave):
        return x_wec[0]

    constraints = [
        {"type": "ineq", "fun": constrain_min_tension},
        {"type": "ineq", "fun": const_peak_torque},
        {"type": "ineq", "fun": const_speed},
        {"type": "ineq", "fun": const_power},
        {"type": "eq", "fun": zero_mean_pos},
    ]

    # -- build WEC and solve --
    wec = WEC_IPOPT.from_bem(bem_data, constraints=constraints, f_add=f_add)

    scale_x_wec = 1e1
    scale_x_opt = 50e-2
    scale_obj = 1e-3
    solve_kw = dict(
        x_wec_0=np.ones(wec.nstate_wec) * 1e-3,
        x_opt_0=np.ones(nstate_opt) * 1e-3,
        scale_x_wec=scale_x_wec,
        scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        optim_options={"max_iter": 2000, "tol": 1e-8, "print_level": 0},
    )
    res = wec.solve(waves, obj_fun, nstate_opt, **solve_kw)[0]
    assert res.success, f"IPOPT failed: {res.message}"

    # -- parametric factories --
    f_pto_passive_parametric = make_pto_passive_parametric(gear_ratios, friction_dict)
    obj_pto_parametric = make_electrical_power_obj_parametric(pto)

    # -- extract BEM params for joint sensitivity --
    hydro_data = wot.add_linear_friction(bem_data, friction=None)
    hydro_data = wot.check_radiation_damping(hydro_data)
    bem_params = extract_bem_params(hydro_data)

    return dict(
        wec=wec, waves=waves, res=res, bem_data=bem_data,
        hydro_data=hydro_data, bem_params=bem_params,
        pto_params_nominal=pto_params_nominal,
        obj_fun=obj_fun, nstate_opt=nstate_opt, pto=pto,
        constraints=constraints, f_add=f_add,
        f_pto_passive_parametric=f_pto_passive_parametric,
        obj_pto_parametric=obj_pto_parametric,
        scale_x_wec=scale_x_wec, scale_x_opt=scale_x_opt,
        scale_obj=scale_obj,
        gear_ratios=gear_ratios, friction_dict=friction_dict,
        ndof=ndof, kinematics=kinematics, controller=controller, name=name,
        f_pretension_wec=f_pretension_wec, f_mooring=f_mooring,
        f_buoyancy=f_buoyancy, f_gravity=f_gravity,
        nsubsteps=nsubsteps, min_line_tension=min_line_tension,
        torque_peak_max=torque_peak_max, rot_speed_max=rot_speed_max,
        power_max=power_max, zero_mean_pos=zero_mean_pos,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. BEM-only sensitivity
# ═══════════════════════════════════════════════════════════════════════════

class TestAquaHarmonicsBEMSensitivity:
    """Default BEM-only sensitivity returns finite gradients for all fields."""

    def test_bem_sensitivity_finite(self, ah_setup):
        s = ah_setup
        grad_bem = sensitivity(s["wec"], s["res"], s["waves"])
        for name in BEMParams._fields:
            g = getattr(grad_bem, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in BEM grad_{name}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. PTO-only sensitivity
# ═══════════════════════════════════════════════════════════════════════════

class TestAquaHarmonicsPTOSensitivity:
    """PTO sensitivity via parametric_forces + obj_fn."""

    def test_pto_sensitivity_finite(self, ah_setup):
        s = ah_setup
        grad_pto = sensitivity(
            s["wec"], s["res"], s["waves"],
            params=s["pto_params_nominal"],
            parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            additional_forces={
                k: v for k, v in s["wec"].forces.items()
                if k != "PTO_passive"
            },
            obj_fn=s["obj_pto_parametric"],
        )
        for name in PTOParams._fields:
            g = float(getattr(grad_pto, name))
            assert np.isfinite(g), f"Non-finite PTO grad for {name}: {g}"
            assert g != 0.0, f"Zero PTO gradient for {name}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Joint BEM + PTO sensitivity
# ═══════════════════════════════════════════════════════════════════════════

class TestAquaHarmonicsJointSensitivity:
    """Joint BEM+PTO sensitivity in a single call."""

    def test_joint_sensitivity_has_both(self, ah_setup):
        s = ah_setup
        params_joint = JointParams(
            bem=s["bem_params"], pto=s["pto_params_nominal"])

        grad_joint = sensitivity(
            s["wec"], s["res"], s["waves"],
            params=params_joint,
            parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            additional_forces={
                k: v for k, v in s["wec"].forces.items()
                if k != "PTO_passive"
            },
            obj_fn=s["obj_pto_parametric"],
        )

        for name in BEMParams._fields:
            g = getattr(grad_joint.bem, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in joint BEM grad_{name}"

        for name in PTOParams._fields:
            g = float(getattr(grad_joint.pto, name))
            assert np.isfinite(g), f"Non-finite in joint PTO grad_{name}"

    def test_joint_matches_separate(self, ah_setup):
        """Joint grad.pto should equal the standalone PTO sensitivity."""
        s = ah_setup
        force_kw = dict(
            parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            additional_forces={
                k: v for k, v in s["wec"].forces.items()
                if k != "PTO_passive"
            },
            obj_fn=s["obj_pto_parametric"],
        )

        grad_pto_only = sensitivity(
            s["wec"], s["res"], s["waves"],
            params=s["pto_params_nominal"], **force_kw,
        )

        params_joint = JointParams(
            bem=s["bem_params"], pto=s["pto_params_nominal"])
        grad_joint = sensitivity(
            s["wec"], s["res"], s["waves"],
            params=params_joint, **force_kw,
        )

        for name in PTOParams._fields:
            np.testing.assert_allclose(
                float(getattr(grad_joint.pto, name)),
                float(getattr(grad_pto_only, name)),
                rtol=1e-10,
                err_msg=f"Joint PTO grad mismatch for {name}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. VJP residual-level FD (PTO params that enter dynamics)
# ═══════════════════════════════════════════════════════════════════════════

RESIDUAL_PTO_PARAMS = ["friction_pto", "inertia_pto", "spring_stiffness"]


class TestAquaHarmonicsResidualVJP:
    """Central FD on λᵀ ∂r/∂p for each PTO param in the dynamics residual."""

    @pytest.mark.parametrize("param_name", RESIDUAL_PTO_PARAMS)
    def test_residual_vjp_fd(self, ah_setup, param_name):
        s = ah_setup
        wec = s["wec"]
        res = s["res"]
        pp = s["pto_params_nominal"]

        x_wec, x_opt = wec.decompose_state(res.x)
        lam = jnp.array(res.dynamics_mult_g)
        x_wec_j = jnp.array(x_wec)
        x_opt_j = jnp.array(x_opt)

        wave = s["waves"].sel(realization=0)
        wd = extract_wave_data(wave, s["hydro_data"]["Froude_Krylov_force"])

        add_forces = {
            k: (lambda _f: (lambda w, xw, xo, _wd: _f(w, xw, xo, wave)))(f)
            for k, f in wec.forces.items() if k != "PTO_passive"
        }

        def r_of_p(p):
            return residual_parametric(
                x_wec_j, x_opt_j, wd, p, wec,
                additional_forces=add_forces,
                parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            )

        _, vjp_fn = jax.vjp(r_of_p, pp)
        (grad_pp,) = vjp_fn(lam)
        anal = float(getattr(grad_pp, param_name))

        val = float(getattr(pp, param_name))
        eps = 1e-5 * max(abs(val), 1.0)

        def r_perturbed(delta):
            pp_new = pp._replace(
                **{param_name: getattr(pp, param_name) + delta})
            return r_of_p(pp_new)

        dr_dh = (r_perturbed(+eps) - r_perturbed(-eps)) / (2 * eps)
        fd = float(jnp.dot(lam, dr_dh))

        if abs(anal) > 1e-15:
            rel_err = abs(fd - anal) / abs(anal)
        else:
            rel_err = abs(fd - anal)

        assert rel_err < 1e-3, (
            f"VJP FD failed for {param_name}: "
            f"analytical={anal:.6e}, FD={fd:.6e}, rel_err={rel_err:.2e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Objective FD (generator params that enter objective)
# ═══════════════════════════════════════════════════════════════════════════

OBJ_PTO_PARAMS = ["winding_resistance", "torque_coefficient"]


class TestAquaHarmonicsObjectiveFD:
    """Central FD on ∂f/∂p for generator params in the objective."""

    @pytest.mark.parametrize("param_name", OBJ_PTO_PARAMS)
    def test_objective_fd(self, ah_setup, param_name):
        s = ah_setup
        wec = s["wec"]
        res = s["res"]
        pp = s["pto_params_nominal"]
        obj_param = s["obj_pto_parametric"]

        x_wec, x_opt = wec.decompose_state(res.x)
        x_wec_j = jnp.array(x_wec)
        x_opt_j = jnp.array(x_opt)
        wave = s["waves"].sel(realization=0)

        def f_of_p(p):
            return obj_param(wec, x_wec_j, x_opt_j, wave, p)

        anal = float(jax.grad(f_of_p)(pp).__getattribute__(param_name))

        val = float(getattr(pp, param_name))
        eps = 1e-6 * max(abs(val), 1.0)

        pp_plus = pp._replace(**{param_name: getattr(pp, param_name) + eps})
        pp_minus = pp._replace(**{param_name: getattr(pp, param_name) - eps})
        fd = float((f_of_p(pp_plus) - f_of_p(pp_minus)) / (2 * eps))

        if abs(anal) > 1e-12:
            rel_err = abs(fd - anal) / abs(anal)
        else:
            rel_err = abs(fd - anal)

        assert rel_err < 1e-4, (
            f"Objective FD failed for {param_name}: "
            f"analytical={anal:.6e}, FD={fd:.6e}, rel_err={rel_err:.2e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Full NLP re-solve FD with dual warm-start (mult_g_0)
# ═══════════════════════════════════════════════════════════════════════════

class TestAquaHarmonicsNLPResolve:
    """Full IPOPT re-solve FD vs analytical for each PTO parameter.

    Uses the new ``mult_g_0`` dual warm-start to keep perturbed solves
    converging reliably on this tightly-constrained problem.
    """

    @pytest.mark.validation
    @pytest.mark.parametrize("param_name", PTOParams._fields)
    def test_nlp_fd_pto(self, ah_setup, param_name):
        s = ah_setup
        wec = s["wec"]
        res = s["res"]
        pp = s["pto_params_nominal"]
        phi_star = res.fun

        grad_pto = sensitivity(
            wec, res, s["waves"],
            params=pp,
            parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            additional_forces={
                k: v for k, v in wec.forces.items()
                if k != "PTO_passive"
            },
            obj_fn=s["obj_pto_parametric"],
        )
        anal = float(getattr(grad_pto, param_name))

        val = float(getattr(pp, param_name))
        eps = 1e-4 * max(abs(val), 1.0)

        phi_pert = self._perturb_and_solve(s, param_name, eps)
        fd = (phi_pert - phi_star) / eps

        if abs(anal) > 1e-12:
            rel_err = abs(fd - anal) / abs(anal)
        else:
            rel_err = abs(fd - anal)

        assert rel_err < 0.10, (
            f"NLP FD failed for {param_name}: "
            f"analytical={anal:.6e}, FD={fd:.6e}, rel_err={rel_err:.2e}"
        )

    @staticmethod
    def _perturb_and_solve(setup, param_name, eps):
        """Rebuild WEC with one perturbed PTO param, re-solve with warm-start."""
        s = setup
        pp = s["pto_params_nominal"]
        pp_pert = pp._replace(
            **{param_name: getattr(pp, param_name) + eps})

        fric_p = float(pp_pert.friction_pto)
        inert_p = float(pp_pert.inertia_pto)
        spring_p = float(pp_pert.spring_stiffness)
        Rw_p = float(pp_pert.winding_resistance)
        kT_p = float(pp_pert.torque_coefficient)

        def power_loss_p(speed, torque):
            return Rw_p * (torque / kT_p) ** 2

        pto_p = wot.pto.PTO(
            s["ndof"], s["kinematics"], s["controller"],
            None, power_loss_p, s["name"])

        gear_ratios = s["gear_ratios"]
        friction_dict = s["friction_dict"]
        nsubsteps = s["nsubsteps"]

        def f_pto_passive_p(wec, x_wec, x_opt, wave, nsubsteps=1):
            pos = wec.vec_to_dofmat(x_wec)
            vel = jnp.dot(wec.derivative_mat, pos)
            acc = jnp.dot(wec.derivative_mat, vel)
            time_matrix = wec.time_mat_nsubsteps(nsubsteps)
            spring = -spring_p * pos
            f_spring = jnp.dot(time_matrix, spring)
            fric = -(fric_p
                     + friction_dict["Bpneumatic_spring_static1"]
                     * gear_ratios["spring"]) * vel
            f_fric = jnp.dot(time_matrix, fric)
            inertia = inert_p * acc
            f_inertia = jnp.dot(time_matrix, inertia)
            return f_spring + f_fric + f_inertia

        def f_pto_line_p(wec, x_wec, x_opt, wave, nsubsteps=1):
            return (pto_p.force_on_wec(wec, x_wec, x_opt, wave, nsubsteps)
                    + s["f_pretension_wec"](wec, x_wec, x_opt, wave, nsubsteps))

        f_add_p = {
            "PTO": f_pto_line_p,
            "PTO_passive": f_pto_passive_p,
            "mooring": s["f_mooring"],
            "buoyancy": s["f_buoyancy"],
            "gravity": s["f_gravity"],
        }

        def constrain_min_tension_p(wec, x_wec, x_opt, wave):
            total_tension = -1 * f_pto_line_p(wec, x_wec, x_opt, wave, nsubsteps)
            return total_tension.flatten() + s["min_line_tension"]

        def const_peak_torque_p(wec, x_wec, x_opt, wave):
            torque = pto_p.force(wec, x_wec, x_opt, wave, nsubsteps)
            return s["torque_peak_max"] - jnp.abs(torque.flatten())

        def const_speed_p(wec, x_wec, x_opt, wave):
            rot_vel = pto_p.velocity(wec, x_wec, x_opt, wave, nsubsteps)
            return s["rot_speed_max"] - jnp.abs(rot_vel.flatten())

        def const_power_p(wec, x_wec, x_opt, wave):
            power_mech = (
                pto_p.velocity(wec, x_wec, x_opt, wave, nsubsteps)
                * pto_p.force(wec, x_wec, x_opt, wave, nsubsteps)
            )
            return s["power_max"] - jnp.abs(power_mech.flatten())

        constraints_p = [
            {"type": "ineq", "fun": constrain_min_tension_p},
            {"type": "ineq", "fun": const_peak_torque_p},
            {"type": "ineq", "fun": const_speed_p},
            {"type": "ineq", "fun": const_power_p},
            {"type": "eq", "fun": s["zero_mean_pos"]},
        ]

        wec_p = WEC_IPOPT.from_bem(
            s["bem_data"], constraints=constraints_p, f_add=f_add_p)

        res = s["res"]
        x_wec_nom, x_opt_nom = s["wec"].decompose_state(res.x)

        fd_optim_options = {
            "max_iter": 3000,
            "tol": 1e-6,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 5,
            "print_level": 0,
        }

        res_p = wec_p.solve(
            s["waves"], pto_p.average_power, s["nstate_opt"],
            x_wec_0=x_wec_nom,
            x_opt_0=x_opt_nom,
            scale_x_wec=s["scale_x_wec"],
            scale_x_opt=s["scale_x_opt"],
            scale_obj=s["scale_obj"],
            optim_options=fd_optim_options,
            mult_g_0=res.mult_g,
            mult_x_L_0=res.mult_x_L,
            mult_x_U_0=res.mult_x_U,
        )[0]

        assert res_p.status in (0, 1), (
            f"Perturbed solve failed for {param_name}: "
            f"status={res_p.status}, {res_p.message}"
        )
        return res_p.fun


# ═══════════════════════════════════════════════════════════════════════════
# 7. Auto-partition forces (no explicit additional_forces)
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoPartitionForces:
    """sensitivity() auto-computes additional_forces from wec.forces."""

    def test_auto_partition_matches_explicit(self, ah_setup):
        s = ah_setup
        force_kw_explicit = dict(
            parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            additional_forces={
                k: v for k, v in s["wec"].forces.items()
                if k != "PTO_passive"
            },
            obj_fn=s["obj_pto_parametric"],
        )
        grad_explicit = sensitivity(
            s["wec"], s["res"], s["waves"],
            params=s["pto_params_nominal"], **force_kw_explicit,
        )

        grad_auto = sensitivity(
            s["wec"], s["res"], s["waves"],
            params=s["pto_params_nominal"],
            parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            obj_fn=s["obj_pto_parametric"],
        )

        for name in PTOParams._fields:
            np.testing.assert_allclose(
                float(getattr(grad_auto, name)),
                float(getattr(grad_explicit, name)),
                rtol=1e-10,
                err_msg=f"Auto-partition mismatch for {name}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 8. WEC_IPOPT.compute_sensitivity() convenience method
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeSensitivityMethod:
    """wec.compute_sensitivity() delegates to module-level sensitivity()."""

    def test_method_matches_function(self, ah_setup):
        s = ah_setup
        grad_fn = sensitivity(s["wec"], s["res"], s["waves"])
        grad_method = s["wec"].compute_sensitivity(s["res"], s["waves"])

        for name in BEMParams._fields:
            np.testing.assert_allclose(
                np.array(getattr(grad_method, name)),
                np.array(getattr(grad_fn, name)),
                atol=1e-12,
                err_msg=f"compute_sensitivity() mismatch for {name}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 9. make_joint_params helper
# ═══════════════════════════════════════════════════════════════════════════

class TestMakeJointParams:
    """make_joint_params creates a valid JAX pytree."""

    def test_joint_params_fields(self, ah_setup):
        s = ah_setup
        joint = make_joint_params(
            s["bem_params"], pto=s["pto_params_nominal"])
        assert hasattr(joint, "bem")
        assert hasattr(joint, "pto")

    def test_joint_params_in_sensitivity(self, ah_setup):
        s = ah_setup
        joint = make_joint_params(
            s["bem_params"], pto=s["pto_params_nominal"])

        grad_joint = sensitivity(
            s["wec"], s["res"], s["waves"],
            params=joint,
            parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            obj_fn=s["obj_pto_parametric"],
        )
        assert hasattr(grad_joint, "bem")
        assert hasattr(grad_joint, "pto")
        for name in PTOParams._fields:
            g = float(getattr(grad_joint.pto, name))
            assert np.isfinite(g), f"Non-finite in joint.pto.{name}"


# ═══════════════════════════════════════════════════════════════════════════
# 10. Constraint multipliers are present and structured
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraintMultipliers:
    """Result carries constraint_multipliers dict with expected keys."""

    def test_constraint_multipliers_populated(self, ah_setup):
        s = ah_setup
        res = s["res"]
        assert hasattr(res, "constraint_multipliers")
        assert len(res.constraint_multipliers) > 0

    def test_dynamics_multiplier_shape(self, ah_setup):
        s = ah_setup
        res = s["res"]
        assert res.dynamics_mult_g.shape == (s["wec"].nstate_wec,)

    def test_user_constraint_keys(self, ah_setup):
        s = ah_setup
        res = s["res"]
        for i in range(5):
            key = f"user_constraint_{i}"
            assert key in res.constraint_multipliers or key == "user_constraint_4", \
                f"Missing constraint multiplier for {key}"


# ═══════════════════════════════════════════════════════════════════════════
# 11. Validation utilities integration (fd_check_residual, fd_check_objective)
# ═══════════════════════════════════════════════════════════════════════════

class TestValidationUtilitiesIntegration:
    """fd_check_residual and fd_check_objective pass on real data."""

    @pytest.mark.validation
    def test_fd_check_residual_passes(self, ah_setup):
        s = ah_setup
        results = fd_check_residual(
            s["wec"], s["res"], s["waves"],
            s["pto_params_nominal"],
            parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            fields=["friction_pto", "inertia_pto", "spring_stiffness"],
            tol=0.05,
            verbose=True,
        )
        for name, r in results.items():
            assert r.passed, (
                f"fd_check_residual failed for {name}: "
                f"analytical={r.analytical:.6e}, fd={r.fd:.6e}, "
                f"rel_error={r.rel_error:.2e}"
            )

    @pytest.mark.validation
    def test_fd_check_objective_passes(self, ah_setup):
        s = ah_setup
        results = fd_check_objective(
            s["wec"], s["res"], s["waves"],
            s["pto_params_nominal"],
            obj_fn=s["obj_pto_parametric"],
            fields=["winding_resistance", "torque_coefficient"],
            tol=0.05,
            verbose=True,
        )
        for name, r in results.items():
            assert r.passed, (
                f"fd_check_objective failed for {name}: "
                f"analytical={r.analytical:.6e}, fd={r.fd:.6e}, "
                f"rel_error={r.rel_error:.2e}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 12. Deprecation warning for sensitivity_parametric
# ═══════════════════════════════════════════════════════════════════════════

class TestDeprecationWarning:
    """sensitivity_parametric emits DeprecationWarning."""

    def test_sensitivity_parametric_warns(self, ah_setup):
        s = ah_setup
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sensitivity_parametric(
                s["wec"], s["res"], s["waves"],
                params=s["pto_params_nominal"],
                residual_fn=lambda wec, xw, xo, wave, p:
                    jnp.zeros(wec.nstate_wec),
            )
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "sensitivity_parametric" in str(w[0].message)


# ═══════════════════════════════════════════════════════════════════════════
# 13. Input validation error messages
# ═══════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    """sensitivity() raises clear errors on bad inputs."""

    def test_missing_dynamics_mult_g(self, ah_setup):
        from scipy.optimize import OptimizeResult
        s = ah_setup
        bad_res = OptimizeResult(x=s["res"].x, fun=s["res"].fun)
        with pytest.raises(AttributeError, match="dynamics_mult_g"):
            sensitivity(s["wec"], bad_res, s["waves"])

    def test_dict_params_rejected(self, ah_setup):
        s = ah_setup
        with pytest.raises(TypeError, match="namedtuple"):
            sensitivity(
                s["wec"], s["res"], s["waves"],
                params={"friction_pto": 100.0},
                parametric_forces={"PTO_passive": s["f_pto_passive_parametric"]},
            )

    def test_non_callable_parametric_force(self, ah_setup):
        s = ah_setup
        with pytest.raises(TypeError, match="not callable"):
            sensitivity(
                s["wec"], s["res"], s["waves"],
                params=s["pto_params_nominal"],
                parametric_forces={"PTO_passive": "not_a_function"},
            )

    def test_params_without_forces_rejected(self, ah_setup):
        s = ah_setup
        with pytest.raises(ValueError, match="parametric_forces"):
            sensitivity(
                s["wec"], s["res"], s["waves"],
                params=s["pto_params_nominal"],
            )
