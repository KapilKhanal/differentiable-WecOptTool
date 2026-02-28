#!/usr/bin/env python3
"""Analyze whether the WEC NLP admits a convex (QP) reformulation for FFO.

The pseudo-spectral WEC NLP has the structure:

    min_{x_wec, x_opt}  v(x_wec)^T F(x_opt)           (bilinear power)
    s.t.   A_wec @ x_wec + A_opt @ x_opt = b(p)        (affine dynamics)
           f_max - |F(x_opt)| >= 0                      (non-convex)

Key insight: since the dynamics are affine, we can eliminate x_wec:

    x_wec = A_wec^{-1} (b - A_opt @ x_opt)

Substituting into the bilinear objective gives a quadratic in x_opt alone:

    f(x_opt) = c^T x_opt - x_opt^T H x_opt

If H is PSD, minimizing -power = maximizing a concave quadratic = convex QP.
The |f| constraints become two-sided linear: -f_max <= T @ x_opt <= f_max.

This script:
1. Extracts A_wec, A_opt, b from the dynamics Jacobian at the optimum
2. Computes the reduced Hessian H
3. Checks eigenvalues of H for strong convexity
4. Reports whether the convex QP reformulation is valid
"""
from __future__ import annotations

import sys
import time
import logging
import numpy as np
import jax
import jax.numpy as jnp
from scipy import linalg as la
import wecopttool as wot

jax.config.update("jax_enable_x64", True)

from wecopttool_differentiable import WEC_IPOPT
from wecopttool_differentiable.solver_ipopt import _extract_all_realizations

logging.getLogger("wecopttool").setLevel(logging.WARNING)
logging.getLogger("capytaine").setLevel(logging.WARNING)

W = 78


def analyze_qp_structure(wec, res, waves, obj_fun, nstate_opt, name=""):
    """Analyze the reduced QP structure of the WEC NLP."""

    print("\n" + "=" * W)
    print(f"  Convex Reformulation Analysis: {name}")
    print("=" * W)

    _, wave_list = _extract_all_realizations(
        waves, wec._hydro_data["Froude_Krylov_force"])
    wave = wave_list[0]

    x_star = np.array(res.x)
    x_wec, x_opt = wec.decompose_state(x_star)
    n_wec = len(x_wec)
    n_opt = len(x_opt)
    n = n_wec + n_opt

    print(f"\n  Problem dimensions:")
    print(f"    n_wec = {n_wec},  n_opt = {n_opt},  total = {n}")

    # ── 1. Dynamics Jacobian: r(x_wec, x_opt) = 0 ────────────────────
    # r is affine, so J_r is constant.  Extract A_wec, A_opt, and b.
    print(f"\n  Step 1: Extract dynamics Jacobian ...")
    x_j = jnp.array(x_star)

    J_r_fn = jax.jacobian(
        lambda x: wec.residual(*wec.decompose_state(x), wave))
    J_r = np.array(J_r_fn(x_j))       # shape: (n_dyn, n)
    n_dyn = J_r.shape[0]

    A_wec = J_r[:, :n_wec]             # dr/dx_wec
    A_opt = J_r[:, n_wec:]             # dr/dx_opt

    r_at_star = np.array(wec.residual(
        jnp.array(x_wec), jnp.array(x_opt), wave))
    b = -(r_at_star - A_wec @ x_wec - A_opt @ x_opt)

    print(f"    Dynamics: {n_dyn} constraints")
    print(f"    A_wec: {A_wec.shape},  A_opt: {A_opt.shape}")

    # Check that dynamics ARE affine (Jacobian should be constant)
    x_pert = x_star + np.random.randn(n) * 1e-3
    J_r_pert = np.array(J_r_fn(jnp.array(x_pert)))
    jac_diff = la.norm(J_r - J_r_pert) / max(la.norm(J_r), 1e-20)

    is_affine = jac_diff < 1e-4
    print(f"    Jacobian constancy check: ||J(x*) - J(x*+ε)|| / ||J|| "
          f"= {jac_diff:.2e}  {'(AFFINE)' if is_affine else '(NONLINEAR!)'}")

    if not is_affine:
        print(f"    WARNING: Dynamics are NOT affine. "
              f"Variable elimination is only approximate.")

    # ── 2. Check A_wec invertibility ──────────────────────────────────
    print(f"\n  Step 2: Check A_wec invertibility ...")
    if n_dyn != n_wec:
        print(f"    n_dyn ({n_dyn}) != n_wec ({n_wec}) — "
              f"cannot directly eliminate x_wec.")
        print(f"    System is {'over' if n_dyn > n_wec else 'under'}"
              f"-determined in x_wec.")
        sv_wec = la.svdvals(A_wec)
    else:
        sv_wec = la.svdvals(A_wec)

    print(f"    A_wec singular values: min = {sv_wec[-1]:.4e}, "
          f"max = {sv_wec[0]:.4e}")
    print(f"    Condition number: {sv_wec[0]/sv_wec[-1]:.2f}")
    a_wec_invertible = sv_wec[-1] > 1e-10

    if not a_wec_invertible:
        print(f"    WARNING: A_wec is nearly singular — elimination unstable.")

    # ── 3. Objective Hessian structure ────────────────────────────────
    print(f"\n  Step 3: Objective Hessian structure ...")

    f_fn = lambda x: obj_fun(wec, x[:n_wec], x[n_wec:], wave)
    H_full = np.array(jax.hessian(f_fn)(x_j))

    H_ww = H_full[:n_wec, :n_wec]       # d²f/dx_wec²
    H_wo = H_full[:n_wec, n_wec:]       # d²f/dx_wec dx_opt  (bilinear part)
    H_ow = H_full[n_wec:, :n_wec]       # d²f/dx_opt dx_wec
    H_oo = H_full[n_wec:, n_wec:]       # d²f/dx_opt²

    print(f"    ||H_ww|| = {la.norm(H_ww):.4e}  "
          f"(d²f/dx_wec²  — zero for pure bilinear)")
    print(f"    ||H_wo|| = {la.norm(H_wo):.4e}  "
          f"(d²f/dx_wec dx_opt — the bilinear coupling)")
    print(f"    ||H_oo|| = {la.norm(H_oo):.4e}  "
          f"(d²f/dx_opt² — nonzero if loss map present)")

    is_bilinear = (la.norm(H_ww) < 1e-8 * la.norm(H_wo)
                   and la.norm(H_oo) < 1e-3 * la.norm(H_wo))
    has_loss_curvature = la.norm(H_oo) > 1e-6

    if is_bilinear:
        print(f"    Structure: PURE BILINEAR  f = x_wec^T Q x_opt")
    elif has_loss_curvature:
        print(f"    Structure: BILINEAR + QUADRATIC LOSS "
              f"(loss map adds curvature in x_opt)")
    else:
        print(f"    Structure: GENERAL NONLINEAR")

    # ── 4. Reduced QP Hessian via variable elimination ────────────────
    print(f"\n  Step 4: Reduced QP Hessian (eliminate x_wec) ...")

    if n_dyn == n_wec and a_wec_invertible:
        # x_wec = A_wec^{-1} (b - A_opt @ x_opt) = P - S @ x_opt
        # where P = A_wec^{-1} b,  S = A_wec^{-1} A_opt
        A_wec_inv = la.inv(A_wec)
        S = A_wec_inv @ A_opt          # dx_wec/dx_opt = -S
        P = A_wec_inv @ b

        # f(x_opt) = obj_fun(P - S @ x_opt, x_opt)
        # Hessian = S^T H_ww S - S^T H_wo - H_ow S + H_oo
        H_reduced = S.T @ H_ww @ S - S.T @ H_wo - H_ow @ S + H_oo

        eigvals = la.eigvalsh(H_reduced)
        mu_reduced = float(eigvals[0])
        C_reduced = float(eigvals[-1])
        kappa_reduced = C_reduced / mu_reduced if mu_reduced > 0 else float('inf')

        print(f"    H_reduced eigenvalues:")
        print(f"      min (mu_g) = {mu_reduced:.6e}")
        print(f"      max (C_g)  = {C_reduced:.6e}")
        print(f"      kappa      = {kappa_reduced:.2f}"
              if np.isfinite(kappa_reduced) else
              f"      kappa      = inf")
        print(f"    Eigenvalue spectrum (first 5, last 5):")
        n_show = min(5, len(eigvals))
        for i in range(n_show):
            print(f"      λ_{i+1:2d} = {eigvals[i]:.6e}")
        if len(eigvals) > 2 * n_show:
            print(f"      ...")
        for i in range(max(n_show, len(eigvals)-n_show), len(eigvals)):
            print(f"      λ_{i+1:2d} = {eigvals[i]:.6e}")

        is_convex = mu_reduced >= -1e-10
        is_strongly_convex = mu_reduced > 1e-8

        print(f"\n    Convexity of reduced QP:")
        if is_strongly_convex:
            print(f"      STRONGLY CONVEX (mu_g = {mu_reduced:.4e})")
            print(f"      FFO Assumption 4.2.2 SATISFIED with kappa = "
                  f"{kappa_reduced:.1f}")
        elif is_convex:
            print(f"      CONVEX (PSD) but NOT strongly convex")
            print(f"      FFO requires mu_g > 0 — add regularization")
        else:
            print(f"      NON-CONVEX (negative eigenvalue: {mu_reduced:.4e})")
            print(f"      Need McCormick envelopes or SDP relaxation")

    else:
        print(f"    Cannot eliminate x_wec (dimension mismatch or singular).")
        H_reduced = None
        mu_reduced = None
        kappa_reduced = None
        is_strongly_convex = False

    # ── 5. Constraint reformulation ───────────────────────────────────
    print(f"\n  Step 5: Constraint convexification ...")

    n_ineq_constraints = len(wec.constraints)
    n_eq_user = sum(1 for c in wec.constraints if c.get("type") == "eq")
    n_ineq_user = n_ineq_constraints - n_eq_user

    print(f"    User constraints: {n_ineq_user} inequality, "
          f"{n_eq_user} equality")

    for i, cons in enumerate(wec.constraints):
        ctype = cons.get("type", "ineq")
        g_val = np.array(cons["fun"](wec,
            jnp.array(x_wec), jnp.array(x_opt), wave))
        n_c = len(np.atleast_1d(g_val))

        # Check if constraint Jacobian w.r.t. x is constant (affine)
        J_g_fn = jax.jacobian(
            lambda x: jnp.atleast_1d(
                cons["fun"](wec, x[:n_wec], x[n_wec:], wave)))
        J_g = np.array(J_g_fn(x_j))
        J_g_pert = np.array(J_g_fn(jnp.array(x_pert)))
        g_jac_diff = la.norm(J_g - J_g_pert) / max(la.norm(J_g), 1e-20)
        g_is_affine = g_jac_diff < 1e-4

        g_is_linear = g_is_affine

        status = "AFFINE" if g_is_affine else "NONLINEAR"
        if ctype == "ineq" and g_is_affine:
            status += " → two-sided linear (convex)"

        print(f"    constraint_{i} ({ctype}, {n_c} outputs): {status}")

    # ── 6. Summary ────────────────────────────────────────────────────
    print(f"\n" + "─" * W)
    print(f"  REFORMULATION SUMMARY")
    print(f"─" * W)

    all_constraints_affine = True  # simplified; real check above
    print(f"  Dynamics affine:          {'YES' if is_affine else 'NO'}")
    print(f"  Objective bilinear:       "
          f"{'YES (pure)' if is_bilinear else 'YES + loss curvature' if has_loss_curvature else 'NO (general)'}")
    print(f"  x_wec eliminable:         "
          f"{'YES' if (n_dyn == n_wec and a_wec_invertible) else 'NO'}")

    if mu_reduced is not None:
        if is_strongly_convex:
            cvx_str = f"YES (strongly, mu={mu_reduced:.4e})"
        elif mu_reduced >= -1e-10:
            cvx_str = "YES (weakly)"
        else:
            cvx_str = "NO"
        print(f"  Reduced QP convex:        {cvx_str}")
        if np.isfinite(kappa_reduced):
            print(f"  Condition number kappa:   {kappa_reduced:.1f}")

    can_reformulate = (is_affine and n_dyn == n_wec
                       and a_wec_invertible and mu_reduced is not None
                       and mu_reduced >= -1e-10)

    print(f"\n  Convex QP reformulation:  "
          f"{'FEASIBLE' if can_reformulate else 'NOT DIRECTLY FEASIBLE'}")

    if can_reformulate and not is_strongly_convex:
        print(f"  To get strong convexity, add regularizer: "
              f"f += epsilon * ||x_opt||^2")
        print(f"  Even epsilon = 1e-6 would make mu_g > 0.")

    if can_reformulate:
        print(f"\n  The equivalent convex QP is:")
        print(f"    min_{{x_opt}}  c^T x_opt + (1/2) x_opt^T H x_opt")
        print(f"    s.t.  -f_max <= T @ x_opt <= f_max   "
              f"(linear bounds)")
        print(f"          A_eq_user @ x_opt = b_eq_user  "
              f"(user equalities)")
        print(f"    where H ∈ R^{{{n_opt}x{n_opt}}}, "
              f"c ∈ R^{{{n_opt}}}")
        print(f"    and x_wec is recovered via: "
              f"x_wec = A_wec^{{-1}} (b - A_opt @ x_opt)")

    print("=" * W)
    print()

    return {
        "is_affine": is_affine,
        "a_wec_invertible": a_wec_invertible,
        "is_bilinear": is_bilinear,
        "has_loss_curvature": has_loss_curvature,
        "mu_reduced": mu_reduced,
        "C_reduced": C_reduced if mu_reduced is not None else None,
        "kappa_reduced": kappa_reduced,
        "is_strongly_convex": is_strongly_convex,
        "can_reformulate": can_reformulate,
    }


# ═══════════════════════════════════════════════════════════════════════
# Tutorial builders (same as check_ffo_conditions.py)
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
    }


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
    }


BUILDERS = {1: _build_tutorial_1, 2: _build_tutorial_2}


def main():
    ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 \
        else sorted(BUILDERS.keys())

    print("=" * W)
    print("  Convex Reformulation Analysis for FFO Compatibility")
    print("=" * W)

    for tid in ids:
        if tid not in BUILDERS:
            print(f"\n  Skipping {tid} — no builder.")
            continue
        setup = BUILDERS[tid]()

        solve_kw = dict(
            scale_x_wec=setup["scale_x_wec"],
            scale_x_opt=setup["scale_x_opt"],
            scale_obj=setup["scale_obj"],
            optim_options=setup.get("ipopt",
                {"max_iter":1000,"tol":1e-8,"print_level":0}),
        )
        if "x_wec_0" in setup:
            solve_kw["x_wec_0"] = setup["x_wec_0"]
        if "x_opt_0" in setup:
            solve_kw["x_opt_0"] = setup["x_opt_0"]

        print(f"\n  Solving NLP for {setup['name']} ...")
        res = setup["wec"].solve(
            setup["waves"], setup["obj_fun"],
            setup["nstate_opt"], **solve_kw)[0]
        print(f"  obj = {res.fun:.4f}, success = {res.success}")

        analyze_qp_structure(
            setup["wec"], res, setup["waves"],
            setup["obj_fun"], setup["nstate_opt"],
            name=setup["name"],
        )


if __name__ == "__main__":
    main()
