#!/usr/bin/env python3
"""Check FFO and Fiacco prerequisites for each tutorial problem.

Verifies the mathematical assumptions from:
- Zhao et al. (2025), "A Fully First-Order Layer for Differentiable
  Optimization" — Assumptions 4.2, 4.3, 4.6
- Fiacco (1983) — LICQ, strict complementarity, SOSC

Usage
=====
    python check_ffo_conditions.py          # run all problems
    python check_ffo_conditions.py 1        # Tutorial 1 only
    python check_ffo_conditions.py 1 2      # Tutorials 1 and 2
"""
from __future__ import annotations

import sys
import time
import logging
import numpy as np
import jax
import jax.numpy as jnp
import wecopttool as wot

jax.config.update("jax_enable_x64", True)

from wecopttool_differentiable import WEC_IPOPT, sensitivity, cross_check_fiacco_ffo
from wecopttool_differentiable.validation import (
    check_ffo_conditions,
    check_regularity,
    FFOConditionsResult,
)

logging.getLogger("wecopttool").setLevel(logging.WARNING)
logging.getLogger("capytaine").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════
# Problem builders
# ═══════════════════════════════════════════════════════════════════════

def _build_tutorial_1():
    """Tutorial 1: WaveBot (heave, PTO force constraint)."""
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

    name = ["PTO_Heave"]
    pto = wot.pto.PTO(
        ndof, np.eye(ndof), wot.controllers.unstructured_controller(),
        None, None, name,
    )

    f_max = 750.0
    nsubsteps = 4

    def const_f_pto(wec, x_wec, x_opt, wave):
        f = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
        return f_max - jnp.abs(f.flatten())

    constraints = [{"type": "ineq", "fun": const_f_pto}]

    wec = WEC_IPOPT.from_bem(
        bem_data, constraints=constraints, friction=None,
        f_add={"PTO": pto.force_on_wec},
    )
    obj_fun = pto.mechanical_average_power
    nstate_opt = 2 * nfreq

    return {
        "name": "Tutorial 1 — WaveBot (heave, PTO force constraint)",
        "wec": wec, "waves": waves,
        "obj_fun": obj_fun, "nstate_opt": nstate_opt,
        "scale_x_wec": 1e1, "scale_x_opt": 1e-3, "scale_obj": 1e-2,
        "ipopt": {"max_iter": 1000, "tol": 1e-8, "print_level": 0},
    }


def _build_tutorial_2():
    """Tutorial 2: AquaHarmonics (loss map, 5 constraints)."""
    import capytaine as cpy
    from capytaine.io.meshio import load_from_meshio

    # -- geometry & BEM --
    ah_hull = wot.geom.AquaHarmonics()
    mesh = ah_hull.mesh(mesh_size_factor=0.25)
    mesh_obj = load_from_meshio(mesh, "WaveBot")
    lid_mesh = mesh_obj.generate_lid(-5e-2)
    fb = cpy.FloatingBody(
        mesh=mesh_obj, lid_mesh=lid_mesh, name="AquaHarmonics")
    fb.add_translation_dof(name="Heave")
    ndof = fb.nb_dofs

    g = 9.81
    rho = 1025
    fb.center_of_mass = [0, 0, 0]
    fb.rotation_center = fb.center_of_mass
    displaced_mass = fb.compute_rigid_body_inertia(rho=rho).values
    displacement = displaced_mass / rho
    fb.mass = np.atleast_2d(5e3)

    wavefreq = 0.24 / 2
    f1, nfreq = wavefreq, 10
    amplitude = 0.5
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, 30, 0)

    freq = wot.frequency(f1, nfreq, False)
    bem_data = wot.run_bem(fb, freq, rho=rho, g=g)

    # -- drivetrain --
    radii = {
        "S1": 0.02, "S2": 0.795, "S3": 0.1595, "S4": 0.200525,
        "S5": 0.40105, "S6": 0.12575, "S7": 0.103,
    }
    inertias = {
        "Igen": 3.9, "I1": 0.029, "I2": 25.6, "I3": 1.43,
        "I4": 1.165, "I5": 4.99, "I6": 1.43, "I7": 1.5, "mps": 40,
    }
    friction_vals = {
        "Bgen": 7, "Bdrivetrain": 40, "Bshaft": 40,
        "Bspring_pulley": 80, "Bpneumatic_spring": 700,
        "Bpneumatic_spring_static1": 0, "Bpspneumatic_spring_static2": 0,
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
        (inertias["Igen"] + inertias["I1"]) * gear_ratios["R21"]**2
        + (inertias["I2"] + inertias["I3"] + inertias["I4"])
        + gear_ratios["R45"]**2 * (
            inertias["I5"] + inertias["I6"]
            + inertias["I7"] * gear_ratios["R67"]**2
            + inertias["mps"] * radii["S6"]**2
        )
    )
    friction_pto = (
        friction_vals["Bgen"] * gear_ratios["R21"]**2
        + friction_vals["Bdrivetrain"]
        + gear_ratios["R45"]**2 * (
            friction_vals["Bshaft"]
            + friction_vals["Bspring_pulley"] * gear_ratios["R67"]**2
            + friction_vals["Bpneumatic_spring"] * radii["S6"]**2
        )
    )

    winding_resistance = 0.4
    torque_coefficient = 1.5

    def power_loss(speed, torque):
        return winding_resistance * (torque / torque_coefficient)**2

    # -- PTO --
    name = ["PTO_Heave"]
    gear_ratio_generator = gear_ratios["R21"] / radii["S3"]
    kinematics = gear_ratio_generator * np.eye(ndof)
    nstate_opt = 2 * nfreq
    pto = wot.pto.PTO(
        ndof, kinematics, wot.controllers.unstructured_controller(),
        None, power_loss, name,
    )

    # -- additional forces --
    def f_buoyancy(wec, x_wec, x_opt, wave, nsubsteps=1):
        return (displacement * rho * g
                * jnp.ones([wec.ncomponents * nsubsteps, wec.ndof]))

    def f_gravity(wec, x_wec, x_opt, wave, nsubsteps=1):
        return (-1 * wec.inertia_matrix.item() * g
                * jnp.ones([wec.ncomponents * nsubsteps, wec.ndof]))

    def f_pretension_wec(wec, x_wec, x_opt, wave, nsubsteps=1):
        return -1 * (f_buoyancy(wec, x_wec, x_opt, wave, nsubsteps)
                     + f_gravity(wec, x_wec, x_opt, wave, nsubsteps))

    def f_pto_passive(wec, x_wec, x_opt, wave, nsubsteps=1):
        pos = wec.vec_to_dofmat(x_wec)
        vel = jnp.dot(wec.derivative_mat, pos)
        acc = jnp.dot(wec.derivative_mat, vel)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        spring = -(gear_ratios["spring"] * airspring["gamma"]
                   * airspring["area"] * airspring["press_init"]
                   / airspring["vol_init"]) * pos
        f_spring = jnp.dot(time_matrix, spring)
        fric = -(friction_pto
                 + friction_vals["Bpneumatic_spring_static1"]
                 * gear_ratios["spring"]) * vel
        f_fric = jnp.dot(time_matrix, fric)
        f_inertia = jnp.dot(time_matrix, inertia_pto * acc)
        return f_spring + f_fric + f_inertia

    def f_pto_line(wec, x_wec, x_opt, wave, nsubsteps=1):
        return (pto.force_on_wec(wec, x_wec, x_opt, wave, nsubsteps)
                + f_pretension_wec(wec, x_wec, x_opt, wave, nsubsteps))

    f_add = {
        "PTO": f_pto_line,
        "PTO_passive": f_pto_passive,
        "buoyancy": f_buoyancy,
        "gravity": f_gravity,
    }

    # -- constraints --
    torque_peak_max = 280
    rot_speed_max = 10000 * 2 * np.pi / 60
    power_max = 80e3
    min_line_tension = -1000
    nsubsteps = 2

    def const_peak_torque_pto(wec, x_wec, x_opt, wave):
        torque = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
        return torque_peak_max - jnp.abs(torque.flatten())

    def const_speed_pto(wec, x_wec, x_opt, wave):
        rot_vel = pto.velocity(wec, x_wec, x_opt, wave, nsubsteps)
        return rot_speed_max - jnp.abs(rot_vel.flatten())

    def const_power_pto(wec, x_wec, x_opt, wave):
        power_mech = (pto.velocity(wec, x_wec, x_opt, wave, nsubsteps)
                      * pto.force(wec, x_wec, x_opt, wave, nsubsteps))
        return power_max - jnp.abs(power_mech.flatten())

    def constrain_min_tension(wec, x_wec, x_opt, wave):
        total_tension = -1 * f_pto_line(wec, x_wec, x_opt, wave, nsubsteps)
        return total_tension.flatten() + min_line_tension

    def zero_mean_pos(wec, x_wec, x_opt, wave):
        return x_wec[0]

    constraints = [
        {"type": "ineq", "fun": constrain_min_tension},
        {"type": "ineq", "fun": const_peak_torque_pto},
        {"type": "ineq", "fun": const_speed_pto},
        {"type": "ineq", "fun": const_power_pto},
        {"type": "eq",   "fun": zero_mean_pos},
    ]

    wec = WEC_IPOPT.from_bem(bem_data, constraints=constraints, f_add=f_add)
    obj_fun = pto.average_power

    return {
        "name": "Tutorial 2 — AquaHarmonics (loss map, 5 constraints)",
        "wec": wec, "waves": waves,
        "obj_fun": obj_fun, "nstate_opt": nstate_opt,
        "scale_x_wec": 1e1, "scale_x_opt": 50e-2, "scale_obj": 1e-3,
        "x_wec_0": np.ones(wec.nstate_wec) * 1e-3,
        "x_opt_0": np.ones(nstate_opt) * 1e-3,
        "ipopt": {"max_iter": 2000, "tol": 1e-8, "print_level": 0},
    }


TUTORIAL_BUILDERS = {
    1: _build_tutorial_1,
    2: _build_tutorial_2,
}


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def run_check(tutorial_id: int) -> dict:
    builder = TUTORIAL_BUILDERS[tutorial_id]
    setup = builder()

    print("\n" + "#" * 78)
    print(f"  {setup['name']}")
    print("#" * 78)

    ipopt_opts = setup.get("ipopt", {"max_iter": 1000, "tol": 1e-8,
                                     "print_level": 0})

    solve_kw = dict(
        scale_x_wec=setup["scale_x_wec"],
        scale_x_opt=setup["scale_x_opt"],
        scale_obj=setup["scale_obj"],
        optim_options=ipopt_opts,
    )
    if "x_wec_0" in setup:
        solve_kw["x_wec_0"] = setup["x_wec_0"]
    if "x_opt_0" in setup:
        solve_kw["x_opt_0"] = setup["x_opt_0"]

    print("\n  Solving NLP ...")
    t0 = time.time()
    results = setup["wec"].solve(
        setup["waves"], setup["obj_fun"], setup["nstate_opt"], **solve_kw)
    res = results[0]
    elapsed = time.time() - t0
    print(f"  Solved in {elapsed:.1f}s  —  obj = {res.fun:.4f}, "
          f"success = {res.success}")

    if not res.success:
        status = getattr(res, "status", "?")
        print(f"  WARNING: NLP did not converge (status {status}) — "
              "checks may be unreliable at a non-optimal point.")

    # -- Fiacco regularity (LICQ + SC + SOSC) --
    print()
    reg_result = check_regularity(
        setup["wec"], res, setup["waves"],
        obj_fun=setup["obj_fun"],
        verbose=True,
    )

    # -- FFO conditions (strong convexity + smoothness + kappa + LICQ + SC) --
    ffo_result = check_ffo_conditions(
        setup["wec"], res, setup["waves"],
        obj_fun=setup["obj_fun"],
        verbose=True,
    )

    # -- Cross-consistency: Fiacco vs FFO chain rule --
    print("=" * 78)
    print("  Cross-Consistency: Fiacco dφ*/dp  vs  FFO ∂f/∂p + (∂f/∂x)·dx*/dp")
    print("=" * 78)

    print("  Computing Fiacco gradient (target='objective') ...")
    t0 = time.time()
    fiacco_grad = sensitivity(
        setup["wec"], res, setup["waves"],
        target="objective",
        obj_fun=setup["obj_fun"],
        nstate_opt=setup["nstate_opt"],
    )
    print(f"  Fiacco done in {time.time() - t0:.1f}s")

    print("  Running cross-check (FFO target='state' internally) ...")
    t0 = time.time()

    cross_solve_kw = dict(
        scale_x_wec=setup["scale_x_wec"],
        scale_x_opt=setup["scale_x_opt"],
        scale_obj=setup["scale_obj"],
        optim_options={
            "max_iter": 2000, "tol": 1e-7, "print_level": 0,
            "warm_start_init_point": "yes",
        },
    )
    if "x_wec_0" in setup:
        cross_solve_kw["x_wec_0"] = setup["x_wec_0"]
    if "x_opt_0" in setup:
        cross_solve_kw["x_opt_0"] = setup["x_opt_0"]

    cross_results = cross_check_fiacco_ffo(
        setup["wec"], res, setup["waves"],
        obj_fun=setup["obj_fun"],
        nstate_opt=setup["nstate_opt"],
        fiacco_grad=fiacco_grad,
        tol=0.10,
        delta=1e-4,
        verbose=True,
        **cross_solve_kw,
    )
    print(f"  Cross-check done in {time.time() - t0:.1f}s")

    n_pass = sum(1 for r in cross_results.values() if r.passed)
    n_total = len(cross_results)
    print(f"\n  Cross-check: {n_pass}/{n_total} parameters passed")
    print("=" * 78)

    return {"fiacco": reg_result, "ffo": ffo_result, "cross": cross_results}


def main():
    if len(sys.argv) > 1:
        ids = [int(x) for x in sys.argv[1:]]
    else:
        ids = sorted(TUTORIAL_BUILDERS.keys())

    print("=" * 78)
    print("  FFO & Fiacco Prerequisite Diagnostics")
    print("=" * 78)

    results = {}
    for tid in ids:
        if tid not in TUTORIAL_BUILDERS:
            print(f"\n  Skipping tutorial {tid} — no builder defined.")
            continue
        results[tid] = run_check(tid)

    # Summary table
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    hdr = (f"  {'Problem':34s} │ {'Fiacco':^18s} │ {'FFO':^20s} │ {'Cross':^7s}")
    print(hdr)
    sub = (f"  {'':34s} │ {'LICQ SC SOSC':^18s} │ "
           f"{'SC kappa Active':^20s} │ {'Pass':^7s}")
    print(sub)
    print("  " + "─" * 34 + "┼" + "─" * 20 + "┼" + "─" * 22 + "┼" + "─" * 9)

    for tid in ids:
        if tid not in results:
            continue
        fr = results[tid]["fiacco"]
        ff = results[tid]["ffo"]
        cr = results[tid].get("cross", {})
        builder = TUTORIAL_BUILDERS[tid]
        name = builder.__doc__.split(".", 1)[0] if builder.__doc__ else f"T{tid}"
        name = name[:34]

        f_licq = "Y" if fr.licq else "N"
        f_sc   = "Y" if fr.strict_complementarity else "N"
        f_sosc = "Y" if fr.sosc else "N"

        o_sc   = "Y" if ff.strong_convexity else "N"
        kstr   = (f"{ff.kappa_g:.0f}" if np.isfinite(ff.kappa_g)
                  and ff.kappa_g < 1e8 else
                  f"{ff.kappa_g:.0e}" if np.isfinite(ff.kappa_g) else "inf")
        o_act  = str(ff.n_active_ineq)

        if cr:
            n_pass = sum(1 for r in cr.values() if r.passed)
            c_str = f"{n_pass}/{len(cr)}"
        else:
            c_str = "—"

        fiacco_col = f" {f_licq:>3s}  {f_sc:>2s}  {f_sosc:>3s}  "
        ffo_col = f" {o_sc:>2s} {kstr:>8s} {o_act:>5s}  "
        print(f"  {name:34s} │{fiacco_col:^20s}│{ffo_col:^22s}│ {c_str:^7s} ")

    print("  " + "─" * 34 + "┴" + "─" * 20 + "┴" + "─" * 22 + "┴" + "─" * 9)
    print()


if __name__ == "__main__":
    main()
