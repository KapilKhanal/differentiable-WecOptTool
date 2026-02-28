#!/usr/bin/env python3
"""Verify the convex QP reformulation recovers the IPOPT optimum.

Strategy
--------
1. Solve the original NLP with IPOPT  →  x* = [x_wec*, x_opt*], φ*
2. Extract the affine dynamics:  A_wec @ x_wec + A_opt @ x_opt = b
3. Eliminate x_wec:  x_wec(x_opt) = A_wec⁻¹ (b - A_opt @ x_opt)
4. Build the *reduced* objective:
       f_red(x_opt) = obj_fun(wec, x_wec(x_opt), x_opt, wave)
   which is quadratic (bilinear power + quadratic loss).
5. Re-express  |F| ≤ F_max  as two-sided linear bounds on x_opt.
6. Solve the reduced QP with scipy.optimize.minimize (trust-constr).
7. Compare φ_QP vs φ_IPOPT and ||x_opt_QP - x_opt_IPOPT||.
8. Run cross-consistency (Fiacco vs FFO) on the IPOPT solution.
"""
from __future__ import annotations

import sys
import time
import logging

import numpy as np
import jax
import jax.numpy as jnp
from scipy import linalg as la
from scipy.optimize import minimize, LinearConstraint
import wecopttool as wot

jax.config.update("jax_enable_x64", True)

from wecopttool_differentiable import WEC_IPOPT
from wecopttool_differentiable.solver_ipopt import (
    sensitivity, _extract_all_realizations,
)
from wecopttool_differentiable.validation import (
    check_regularity, check_ffo_conditions, cross_check_fiacco_ffo,
)

logging.getLogger("wecopttool").setLevel(logging.WARNING)
logging.getLogger("capytaine").setLevel(logging.WARNING)

W = 78


# ═══════════════════════════════════════════════════════════════════════
#  Reduced QP solver
# ═══════════════════════════════════════════════════════════════════════

def solve_reduced_qp(wec, res, waves, obj_fun, nstate_opt, setup,
                     verbose=True):
    """Solve the reduced (x_opt-only) QP and compare with IPOPT solution.

    Returns dict with comparison metrics.
    """
    _, wave_list = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    wave = wave_list[0]

    x_star = np.array(res.x)
    x_wec_star, x_opt_star = wec.decompose_state(x_star)
    n_wec = len(x_wec_star)
    n_opt = len(x_opt_star)
    n = n_wec + n_opt

    if verbose:
        print("\n" + "=" * W)
        print("  Reduced QP Verification")
        print("=" * W)
        print(f"\n  IPOPT solution: obj = {res.fun:.6f}")
        print(f"  Dimensions: n_wec={n_wec}, n_opt={n_opt}")

    # ── 1. Extract affine dynamics  A_wec @ x_wec + A_opt @ x_opt = b ──
    x_j = jnp.array(x_star)

    J_r_fn = jax.jacobian(
        lambda x: wec.residual(*wec.decompose_state(x), wave))
    J_r = np.array(J_r_fn(x_j))
    n_dyn = J_r.shape[0]

    A_wec = J_r[:, :n_wec]
    A_opt = J_r[:, n_wec:]

    r_at_star = np.array(
        wec.residual(jnp.array(x_wec_star), jnp.array(x_opt_star), wave))
    b = -(r_at_star - A_wec @ x_wec_star - A_opt @ x_opt_star)

    residual_at_star = np.linalg.norm(r_at_star)
    if verbose:
        print(f"\n  Dynamics residual at IPOPT x*: {residual_at_star:.2e}")
        print(f"  A_wec cond: {la.svdvals(A_wec)[0]/la.svdvals(A_wec)[-1]:.1f}")

    A_wec_inv = la.inv(A_wec)
    S = A_wec_inv @ A_opt       # x_wec = P - S @ x_opt
    P = A_wec_inv @ b

    # Verify elimination recovers x_wec*
    x_wec_recovered = P - S @ x_opt_star
    recovery_err = la.norm(x_wec_recovered - x_wec_star) / max(la.norm(x_wec_star), 1e-20)
    if verbose:
        print(f"  x_wec recovery error at x_opt*: {recovery_err:.2e}")

    # ── 2. Build reduced objective f_red(x_opt) ─────────────────────────
    def x_wec_of(x_opt_np):
        return P - S @ x_opt_np

    def f_red(x_opt_np):
        xw = jnp.array(x_wec_of(x_opt_np))
        xo = jnp.array(x_opt_np)
        return float(obj_fun(wec, xw, xo, wave))

    def f_red_grad(x_opt_np):
        xw = jnp.array(x_wec_of(x_opt_np))
        xo = jnp.array(x_opt_np)
        full_grad = np.array(jax.grad(
            lambda x: obj_fun(wec, x[:n_wec], x[n_wec:], wave)
        )(jnp.concatenate([xw, xo])))
        # chain rule: df/dx_opt_reduced = df/dx_opt + df/dx_wec * (-S)
        df_dxw = full_grad[:n_wec]
        df_dxo = full_grad[n_wec:]
        return df_dxo - S.T @ df_dxw

    # Verify at IPOPT solution
    f_red_at_star = f_red(x_opt_star)
    if verbose:
        print(f"\n  f_red(x_opt*) = {f_red_at_star:.6f}  (should match IPOPT obj)")
        print(f"  Difference: {abs(f_red_at_star - res.fun):.2e}")

    # ── 3. Build linear constraints from original inequality constraints ──
    linear_constraints = []

    for i, cons in enumerate(wec.constraints):
        ctype = cons.get("type", "ineq")
        g_fn = cons["fun"]

        # Check Jacobian structure at the optimum
        J_g_fn = jax.jacobian(
            lambda x, _gfn=g_fn: jnp.atleast_1d(
                _gfn(wec, x[:n_wec], x[n_wec:], wave)))
        J_g = np.array(J_g_fn(x_j))
        J_g_xwec = J_g[:, :n_wec]
        J_g_xopt = J_g[:, n_wec:]

        # Reduced constraint Jacobian: dg/dx_opt_red = dg/dx_opt - dg/dx_wec @ S
        A_red = J_g_xopt - J_g_xwec @ S
        g_val = np.array(jnp.atleast_1d(
            g_fn(wec, jnp.array(x_wec_star), jnp.array(x_opt_star), wave)))
        # For affine g: g(x_opt) = g(x*) + A_red @ (x_opt - x_opt*)
        # i.e. g(x_opt) = (g(x*) - A_red @ x_opt*) + A_red @ x_opt
        b_const = g_val - A_red @ x_opt_star

        n_c = len(g_val)

        if ctype == "ineq":
            # g(x) >= 0  →  A_red @ x_opt >= -b_const
            linear_constraints.append(LinearConstraint(
                A_red, lb=-b_const, ub=np.full(n_c, np.inf)))
            if verbose:
                n_active = np.sum(np.abs(g_val) < 1e-4)
                print(f"  constraint_{i} (ineq, {n_c} outputs, "
                      f"{n_active} active): linearized")
        elif ctype == "eq":
            # g(x) = 0  →  A_red @ x_opt = -b_const
            linear_constraints.append(LinearConstraint(
                A_red, lb=-b_const, ub=-b_const))
            if verbose:
                print(f"  constraint_{i} (eq, {n_c} outputs): linearized")

    # ── 4. Solve reduced QP ─────────────────────────────────────────────
    if verbose:
        print(f"\n  Solving reduced QP (scipy trust-constr) ...")

    t0 = time.time()
    qp_res = minimize(
        f_red,
        x0=x_opt_star,
        jac=f_red_grad,
        method="trust-constr",
        constraints=linear_constraints,
        options={"maxiter": 2000, "gtol": 1e-10, "xtol": 1e-14,
                 "verbose": 0},
    )
    t_qp = time.time() - t0

    x_opt_qp = qp_res.x
    x_wec_qp = x_wec_of(x_opt_qp)
    obj_qp = qp_res.fun

    # Verify dynamics residual with QP solution
    r_qp = np.array(
        wec.residual(jnp.array(x_wec_qp), jnp.array(x_opt_qp), wave))
    dyn_res_qp = la.norm(r_qp)

    # Also evaluate the original objective at the QP solution
    obj_orig_at_qp = float(obj_fun(
        wec, jnp.array(x_wec_qp), jnp.array(x_opt_qp), wave))

    # ── 5. Comparison ───────────────────────────────────────────────────
    obj_diff = abs(obj_qp - res.fun)
    obj_rel = obj_diff / max(abs(res.fun), 1e-20)
    x_opt_diff = la.norm(x_opt_qp - x_opt_star)
    x_opt_rel = x_opt_diff / max(la.norm(x_opt_star), 1e-20)
    x_wec_diff = la.norm(x_wec_qp - x_wec_star)
    x_wec_rel = x_wec_diff / max(la.norm(x_wec_star), 1e-20)

    if verbose:
        print(f"\n  ┌{'─'*74}┐")
        print(f"  │ {'':36s} {'IPOPT':>14s} {'QP':>14s} │")
        print(f"  ├{'─'*74}┤")
        print(f"  │ {'Objective':36s} {res.fun:>14.6f} {obj_qp:>14.6f} │")
        print(f"  │ {'||x_opt||':36s} "
              f"{la.norm(x_opt_star):>14.6f} "
              f"{la.norm(x_opt_qp):>14.6f} │")
        print(f"  │ {'||x_wec||':36s} "
              f"{la.norm(x_wec_star):>14.6f} "
              f"{la.norm(x_wec_qp):>14.6f} │")
        print(f"  │ {'Dynamics residual':36s} "
              f"{residual_at_star:>14.2e} "
              f"{dyn_res_qp:>14.2e} │")
        print(f"  └{'─'*74}┘")
        print()
        print(f"  Objective difference:  abs = {obj_diff:.2e}, "
              f"rel = {obj_rel:.2e}")
        print(f"  x_opt difference:      abs = {x_opt_diff:.2e}, "
              f"rel = {x_opt_rel:.2e}")
        print(f"  x_wec difference:      abs = {x_wec_diff:.2e}, "
              f"rel = {x_wec_rel:.2e}")
        print(f"  QP solve time: {t_qp:.2f}s")

        match = obj_rel < 1e-3
        print(f"\n  Objective match: {'PASS' if match else 'FAIL'} "
              f"(tol = 0.1%)")

    return {
        "obj_ipopt": res.fun,
        "obj_qp": obj_qp,
        "obj_rel_err": obj_rel,
        "x_opt_rel_err": x_opt_rel,
        "x_wec_rel_err": x_wec_rel,
        "dyn_residual_qp": dyn_res_qp,
        "qp_success": qp_res.success,
        "qp_result": qp_res,
        "x_opt_qp": x_opt_qp,
        "x_wec_qp": x_wec_qp,
        "obj_match": obj_rel < 1e-3,
        "A_wec_inv": A_wec_inv,
        "S": S, "P": P,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Tutorial 2 builder (AquaHarmonics — strongly convex reduced QP)
# ═══════════════════════════════════════════════════════════════════════

def _build_tutorial_2():
    """Tutorial 2: AquaHarmonics (electrical power with loss map)."""
    import capytaine as cpy
    from capytaine.io.meshio import load_from_meshio

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
    waves = wot.waves.regular_wave(f1, nfreq, wavefreq, 0.5, 30, 0)
    freq = wot.frequency(f1, nfreq, False)
    bem_data = wot.run_bem(fb, freq, rho=rho, g=g)

    radii = {"S1": 0.02, "S2": 0.795, "S3": 0.1595, "S4": 0.200525,
             "S5": 0.40105, "S6": 0.12575, "S7": 0.103}
    inertias = {"Igen": 3.9, "I1": 0.029, "I2": 25.6, "I3": 1.43,
                "I4": 1.165, "I5": 4.99, "I6": 1.43, "I7": 1.5, "mps": 40}
    friction_vals = {"Bgen": 7, "Bdrivetrain": 40, "Bshaft": 40,
                     "Bspring_pulley": 80, "Bpneumatic_spring": 700,
                     "Bpneumatic_spring_static1": 0,
                     "Bpspneumatic_spring_static2": 0}
    airspring = {"gamma": 1.4, "height": 1, "diameter": 3,
                 "area": 0.0709676, "press_init": 854e3, "vol_init": 1}
    gear_ratios = {
        "R21": radii["S2"]/radii["S1"],
        "R45": radii["S4"]/radii["S5"],
        "R67": radii["S6"]/radii["S7"],
        "spring": radii["S6"]*(radii["S4"]/radii["S5"]),
    }
    inertia_pto = ((inertias["Igen"]+inertias["I1"])*gear_ratios["R21"]**2
        + (inertias["I2"]+inertias["I3"]+inertias["I4"])
        + gear_ratios["R45"]**2*(inertias["I5"]+inertias["I6"]
            + inertias["I7"]*gear_ratios["R67"]**2
            + inertias["mps"]*radii["S6"]**2))
    friction_pto = (friction_vals["Bgen"]*gear_ratios["R21"]**2
        + friction_vals["Bdrivetrain"]
        + gear_ratios["R45"]**2*(friction_vals["Bshaft"]
            + friction_vals["Bspring_pulley"]*gear_ratios["R67"]**2
            + friction_vals["Bpneumatic_spring"]*radii["S6"]**2))

    def power_loss(speed, torque):
        return 0.4 * (torque / 1.5)**2

    gear_ratio_gen = gear_ratios["R21"] / radii["S3"]
    kinematics = gear_ratio_gen * np.eye(ndof)
    nstate_opt = 2 * nfreq
    pto = wot.pto.PTO(ndof, kinematics,
        wot.controllers.unstructured_controller(), None, power_loss,
        ["PTO_Heave"])

    def f_buoyancy(wec, x_wec, x_opt, wave, nsubsteps=1):
        return displacement*rho*g*jnp.ones([wec.ncomponents*nsubsteps, wec.ndof])
    def f_gravity(wec, x_wec, x_opt, wave, nsubsteps=1):
        return -wec.inertia_matrix.item()*g*jnp.ones([wec.ncomponents*nsubsteps, wec.ndof])
    def f_pretension(wec, x_wec, x_opt, wave, nsubsteps=1):
        return -(f_buoyancy(wec,x_wec,x_opt,wave,nsubsteps)
                 + f_gravity(wec,x_wec,x_opt,wave,nsubsteps))
    def f_pto_passive(wec, x_wec, x_opt, wave, nsubsteps=1):
        pos = wec.vec_to_dofmat(x_wec)
        vel = jnp.dot(wec.derivative_mat, pos)
        acc = jnp.dot(wec.derivative_mat, vel)
        tm = wec.time_mat_nsubsteps(nsubsteps)
        spring = -(gear_ratios["spring"]*airspring["gamma"]*airspring["area"]
                   *airspring["press_init"]/airspring["vol_init"])*pos
        fric = -(friction_pto + friction_vals["Bpneumatic_spring_static1"]
                 *gear_ratios["spring"])*vel
        return jnp.dot(tm,spring) + jnp.dot(tm,fric) + jnp.dot(tm,inertia_pto*acc)
    def f_pto_line(wec, x_wec, x_opt, wave, nsubsteps=1):
        return (pto.force_on_wec(wec,x_wec,x_opt,wave,nsubsteps)
                + f_pretension(wec,x_wec,x_opt,wave,nsubsteps))

    f_add = {"PTO": f_pto_line, "PTO_passive": f_pto_passive,
             "buoyancy": f_buoyancy, "gravity": f_gravity}

    nsubsteps = 2
    def const_torque(wec,x_wec,x_opt,wave):
        return 280 - jnp.abs(pto.force(wec,x_wec,x_opt,wave,nsubsteps).flatten())
    def const_speed(wec,x_wec,x_opt,wave):
        return 10000*2*np.pi/60 - jnp.abs(pto.velocity(wec,x_wec,x_opt,wave,nsubsteps).flatten())
    def const_power(wec,x_wec,x_opt,wave):
        return 80e3 - jnp.abs((pto.velocity(wec,x_wec,x_opt,wave,nsubsteps)
                               *pto.force(wec,x_wec,x_opt,wave,nsubsteps)).flatten())
    def const_tension(wec,x_wec,x_opt,wave):
        return (-f_pto_line(wec,x_wec,x_opt,wave,nsubsteps)).flatten() + (-1000)
    def zero_mean(wec,x_wec,x_opt,wave):
        return x_wec[0]

    constraints = [
        {"type":"ineq","fun":const_tension},
        {"type":"ineq","fun":const_torque},
        {"type":"ineq","fun":const_speed},
        {"type":"ineq","fun":const_power},
        {"type":"eq","fun":zero_mean},
    ]
    wec = WEC_IPOPT.from_bem(bem_data, constraints=constraints, f_add=f_add)

    return {
        "name": "Tutorial 2 — AquaHarmonics (electrical power, loss map)",
        "wec": wec, "waves": waves,
        "obj_fun": pto.average_power,
        "nstate_opt": nstate_opt,
        "scale_x_wec": 1e1, "scale_x_opt": 50e-2, "scale_obj": 1e-3,
        "x_wec_0": np.ones(wec.nstate_wec)*1e-3,
        "x_opt_0": np.ones(nstate_opt)*1e-3,
        "ipopt": {"max_iter": 2000, "tol": 1e-8, "print_level": 0},
        "pto": pto,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Tutorial 1 builder (WaveBot — needs regularization for strong convex)
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
        bem_data, constraints=[{"type": "ineq", "fun": const_f_pto}],
        friction=None, f_add={"PTO": pto.force_on_wec},
    )

    return {
        "name": "Tutorial 1 — WaveBot (mechanical power, PTO constraint)",
        "wec": wec, "waves": waves,
        "obj_fun": pto.mechanical_average_power,
        "nstate_opt": 2 * nfreq,
        "scale_x_wec": 1e1, "scale_x_opt": 1e-3, "scale_obj": 1e-2,
        "ipopt": {"max_iter": 1000, "tol": 1e-8, "print_level": 0},
        "pto": pto,
    }


BUILDERS = {1: _build_tutorial_1, 2: _build_tutorial_2}


def run_verification(tid, run_cross_check=True):
    """Run complete verification for one tutorial."""
    setup = BUILDERS[tid]()
    wec = setup["wec"]

    # ── Solve original NLP with IPOPT ────────────────────────────────
    solve_kw = dict(
        scale_x_wec=setup["scale_x_wec"],
        scale_x_opt=setup["scale_x_opt"],
        scale_obj=setup["scale_obj"],
        optim_options=setup.get("ipopt",
            {"max_iter": 1000, "tol": 1e-8, "print_level": 0}),
    )
    if "x_wec_0" in setup:
        solve_kw["x_wec_0"] = setup["x_wec_0"]
    if "x_opt_0" in setup:
        solve_kw["x_opt_0"] = setup["x_opt_0"]

    print("\n" + "=" * W)
    print(f"  TUTORIAL {tid}: {setup['name']}")
    print("=" * W)
    print(f"\n  Solving NLP with IPOPT ...")
    t0 = time.time()
    res = wec.solve(
        setup["waves"], setup["obj_fun"],
        setup["nstate_opt"], **solve_kw)[0]
    t_ipopt = time.time() - t0
    print(f"  IPOPT: obj = {res.fun:.6f}, success = {res.success}, "
          f"time = {t_ipopt:.1f}s")

    # ── Solve reduced QP ─────────────────────────────────────────────
    qp_result = solve_reduced_qp(
        wec, res, setup["waves"], setup["obj_fun"],
        setup["nstate_opt"], setup, verbose=True)

    if not run_cross_check:
        return {"qp": qp_result}

    # ── Cross-consistency check ──────────────────────────────────────
    print("\n" + "=" * W)
    print("  Cross-Consistency: Fiacco dφ*/dp  vs  FFO chain rule")
    print("=" * W)

    print("\n  Computing Fiacco gradient ...")
    t0 = time.time()
    fiacco_grad = sensitivity(
        wec, res, setup["waves"],
        target="objective",
        obj_fun=setup["obj_fun"],
        nstate_opt=setup["nstate_opt"],
    )
    print(f"  Fiacco done in {time.time() - t0:.1f}s")

    print("  Running cross-check (FFO internally) ...")
    t0 = time.time()
    cross_kw = dict(
        scale_x_wec=setup["scale_x_wec"],
        scale_x_opt=setup["scale_x_opt"],
        scale_obj=setup["scale_obj"],
        optim_options={
            "max_iter": 2000, "tol": 1e-7, "print_level": 0,
            "warm_start_init_point": "yes",
        },
    )
    if "x_wec_0" in setup:
        cross_kw["x_wec_0"] = setup["x_wec_0"]
    if "x_opt_0" in setup:
        cross_kw["x_opt_0"] = setup["x_opt_0"]

    cross_results = cross_check_fiacco_ffo(
        wec, res, setup["waves"],
        obj_fun=setup["obj_fun"],
        nstate_opt=setup["nstate_opt"],
        fiacco_grad=fiacco_grad,
        tol=0.10, delta=1e-4, verbose=True,
        **cross_kw,
    )
    print(f"  Cross-check done in {time.time() - t0:.1f}s")

    n_pass = sum(1 for r in cross_results.values() if r.passed)
    n_total = len(cross_results)
    print(f"\n  Cross-check: {n_pass}/{n_total} parameters passed (tol=10%)")

    return {"qp": qp_result, "cross": cross_results}


def main():
    skip_cross = "--no-cross" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ids = [int(x) for x in args] if args else sorted(BUILDERS.keys())

    print("=" * W)
    print("  QP Reformulation Verification & Cross-Consistency")
    print("=" * W)

    results = {}
    for tid in ids:
        if tid not in BUILDERS:
            print(f"\n  Skipping tutorial {tid} — no builder.")
            continue
        results[tid] = run_verification(tid, run_cross_check=not skip_cross)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  SUMMARY")
    print("=" * W)
    print(f"  {'Tutorial':30s} {'φ_IPOPT':>10s} {'φ_QP':>10s} "
          f"{'Rel Err':>10s} {'Match':>6s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*6}")
    for tid, r in results.items():
        qp = r["qp"]
        match_str = "PASS" if qp["obj_match"] else "FAIL"
        print(f"  Tutorial {tid:2d}{' '*20} "
              f"{qp['obj_ipopt']:>10.4f} {qp['obj_qp']:>10.4f} "
              f"{qp['obj_rel_err']:>10.2e} {match_str:>6s}")
        if "cross" in r:
            cr = r["cross"]
            n_pass = sum(1 for v in cr.values() if v.passed)
            print(f"  {'  Cross-consistency':30s} "
                  f"{'':>10s} {'':>10s} "
                  f"{'':>10s} {n_pass}/{len(cr)}")
    print("=" * W)


if __name__ == "__main__":
    main()
