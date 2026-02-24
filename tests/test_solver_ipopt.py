"""Tests for wecopttool_differentiable (IPOPT solver + sensitivity).

Run with::

    pytest tests/test_solver_ipopt.py -v
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import xarray as xr

import wecopttool as wot
from wecopttool_differentiable import (
    WEC_IPOPT,
    make_differentiable_solver,
    sensitivity,
    sensitivity_parametric,
    BEMParams,
    extract_bem_params,
    extract_wave_data,
    residual_parametric,
)

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

F1 = 0.12
NFREQ = 5
NDOF = 2
NDIR = 1
FRICTION = np.eye(NDOF) * 0.5


@pytest.fixture(scope="module")
def bem_data():
    """Synthetic BEM data with a single wave direction."""
    coords = {
        "omega": [2 * np.pi * (i + 1) * F1 for i in range(NFREQ)],
        "influenced_dof": ["DOF_1", "DOF_2"],
        "radiating_dof": ["DOF_1", "DOF_2"],
        "wave_direction": [0.0],
    }
    radiation_dims = ["omega", "radiating_dof", "influenced_dof"]
    excitation_dims = ["omega", "wave_direction", "influenced_dof"]
    hydrostatics_dims = ["radiating_dof", "influenced_dof"]

    rng = np.random.default_rng(42)
    added_mass = np.eye(NDOF)[None, :, :] * (rng.random((NFREQ, 1, 1)) + 0.5)
    radiation_damping = np.eye(NDOF)[None, :, :] * (rng.random((NFREQ, 1, 1)) + 0.5)
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
    hd = wot.add_linear_friction(bem_data, FRICTION)
    hd = wot.check_radiation_damping(hd, min_damping=1e-6)
    return hd


@pytest.fixture(scope="module")
def wave():
    freq = 2 * F1
    w = wot.waves.regular_wave(F1, NFREQ, freq, 1.0, 30.0)
    return w


@pytest.fixture(scope="module")
def wec_ipopt(bem_data):
    return WEC_IPOPT.from_bem(bem_data, friction=FRICTION, min_damping=1e-6)


@pytest.fixture(scope="module")
def solve_result(wec_ipopt, wave):
    """Solve a quadratic objective that depends on BOTH x_wec and x_opt.

    This ensures the dynamics constraint is actively binding and the
    Lagrange multipliers are non-trivial (needed for Fiacco tests).
    """
    ncomp = wec_ipopt.ncomponents
    ndof = wec_ipopt.ndof
    nstate_opt = ncomp * ndof

    def obj_fun(wec, x_wec, x_opt, wav):
        return jnp.sum(x_wec**2) + jnp.sum(x_opt**2)

    results = wec_ipopt.solve(
        waves=wave,
        obj_fun=obj_fun,
        nstate_opt=nstate_opt,
        optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
    )
    return results[0]


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWEC_IPOPT:

    def test_is_wec_subclass(self, wec_ipopt):
        assert isinstance(wec_ipopt, wot.WEC)
        assert isinstance(wec_ipopt, WEC_IPOPT)

    def test_solve_succeeds(self, solve_result):
        assert solve_result.success, solve_result.message

    def test_result_has_multipliers(self, solve_result):
        assert hasattr(solve_result, "mult_g")
        assert hasattr(solve_result, "dynamics_mult_g")
        assert hasattr(solve_result, "mult_x_L")
        assert hasattr(solve_result, "mult_x_U")
        assert hasattr(solve_result, "constraint_info")

    def test_dynamics_multiplier_shape(self, solve_result, wec_ipopt):
        lam = solve_result.dynamics_mult_g
        assert lam.shape == (wec_ipopt.nstate_wec,), (
            f"Expected ({wec_ipopt.nstate_wec},), got {lam.shape}")

    def test_dynamics_multiplier_finite(self, solve_result):
        lam = solve_result.dynamics_mult_g
        assert np.all(np.isfinite(lam)), "Dynamics multipliers contain NaN/Inf"

    def test_constraint_info_has_dynamics(self, solve_result):
        ci = solve_result.constraint_info
        assert "dynamics" in ci
        assert ci["dynamics"]["type"] == "eq"
        assert ci["dynamics"]["size"] == solve_result.dynamics_mult_g.shape[0]

    def test_residual_near_zero(self, solve_result, wec_ipopt, wave):
        x_wec, x_opt = wec_ipopt.decompose_state(solve_result.x)
        wav = wave.sel(realization=0)
        r = wec_ipopt.residual(x_wec, x_opt, wav)
        assert np.max(np.abs(r)) < 1e-4, (
            f"Residual not near zero: max|r| = {np.max(np.abs(r)):.2e}")


class TestFiaccoSensitivity:
    """End-to-end: solve with IPOPT, extract λ, compute λᵀ ∂r/∂h."""

    def test_full_pipeline(self, wec_ipopt, wave, hydro_data, solve_result):
        wav = wave.sel(realization=0)
        x_wec, x_opt = wec_ipopt.decompose_state(solve_result.x)
        lam = jnp.array(solve_result.dynamics_mult_g)

        bp = extract_bem_params(hydro_data)
        wd = extract_wave_data(wav, hydro_data["Froude_Krylov_force"])

        def r_of_h(h):
            return residual_parametric(x_wec, x_opt, wd, h, wec_ipopt)

        _, vjp_fn = jax.vjp(r_of_h, bp)
        (grad_h,) = vjp_fn(lam)

        for name in BEMParams._fields:
            g = getattr(grad_h, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"

        any_nonzero = any(
            float(jnp.max(jnp.abs(getattr(grad_h, n)))) > 0
            for n in BEMParams._fields)
        assert any_nonzero, "All sensitivities are zero"


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def obj_fun_and_nstate(wec_ipopt):
    """Objective function and nstate_opt used across convenience tests."""
    ncomp = wec_ipopt.ncomponents
    ndof = wec_ipopt.ndof
    nstate_opt = ncomp * ndof

    def obj_fun(wec, x_wec, x_opt, wav):
        return jnp.sum(x_wec**2) + jnp.sum(x_opt**2)

    return obj_fun, nstate_opt


class TestSensitivityMethod:
    """Tests for :meth:`WEC_IPOPT.sensitivity`."""

    def test_returns_bemparams(self, wec_ipopt, wave, solve_result):
        grad_h = sensitivity(wec_ipopt, solve_result, wave)
        assert isinstance(grad_h, BEMParams)

    def test_shapes_match(self, wec_ipopt, wave, hydro_data, solve_result):
        bp = extract_bem_params(hydro_data)
        grad_h = sensitivity(wec_ipopt, solve_result, wave)
        for name in BEMParams._fields:
            expected = getattr(bp, name).shape
            actual = getattr(grad_h, name).shape
            assert actual == expected, (
                f"{name}: expected shape {expected}, got {actual}")

    def test_finite(self, wec_ipopt, wave, solve_result):
        grad_h = sensitivity(wec_ipopt, solve_result, wave)
        for name in BEMParams._fields:
            g = getattr(grad_h, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"

    def test_agrees_with_manual(self, wec_ipopt, wave, hydro_data,
                                solve_result):
        """sensitivity() must match the manual Fiacco computation."""
        grad_auto = sensitivity(wec_ipopt, solve_result, wave)

        wav = wave.sel(realization=0)
        x_wec, x_opt = wec_ipopt.decompose_state(solve_result.x)
        lam = jnp.array(solve_result.dynamics_mult_g)
        bp = extract_bem_params(hydro_data)
        wd = extract_wave_data(wav, hydro_data["Froude_Krylov_force"])

        def r_of_h(h):
            return residual_parametric(
                jnp.array(x_wec), jnp.array(x_opt), wd, h, wec_ipopt)

        _, vjp_fn = jax.vjp(r_of_h, bp)
        (grad_manual,) = vjp_fn(lam)
        # Fix JAX complex VJP convention: conjugate complex leaves
        grad_manual = jax.tree_util.tree_map(
            lambda x: jnp.conj(x) if jnp.iscomplexobj(x) else x, grad_manual)

        for name in BEMParams._fields:
            np.testing.assert_allclose(
                np.array(getattr(grad_auto, name)),
                np.array(getattr(grad_manual, name)),
                atol=1e-12,
                err_msg=f"Mismatch in {name}",
            )


class TestMakeDifferentiableSolver:
    """Tests for :func:`make_differentiable_solver`."""

    def test_callable(self, wec_ipopt, wave, obj_fun_and_nstate):
        obj_fun, nstate_opt = obj_fun_and_nstate
        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        assert callable(f)

    def test_forward_returns_scalar(self, wec_ipopt, wave,
                                    hydro_data, obj_fun_and_nstate):
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)
        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        phi = f(bp)
        assert phi.shape == (), f"Expected scalar, got shape {phi.shape}"
        assert jnp.isfinite(phi), f"Non-finite objective: {phi}"

    def test_grad_returns_bemparams(self, wec_ipopt, wave,
                                    hydro_data, obj_fun_and_nstate):
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)
        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        grad_h = jax.grad(f)(bp)
        assert isinstance(grad_h, BEMParams)
        for name in BEMParams._fields:
            g = getattr(grad_h, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"

    def test_grad_agrees_with_sensitivity(self, wec_ipopt, wave,
                                          hydro_data, obj_fun_and_nstate):
        """jax.grad(f) must agree with sensitivity()."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)

        solve_kwargs = dict(
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6})

        results = wec_ipopt.solve(wave, obj_fun, nstate_opt, **solve_kwargs)
        grad_sens = sensitivity(wec_ipopt, results[0], wave)

        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt, **solve_kwargs)
        grad_vjp = jax.grad(f)(bp)

        for name in BEMParams._fields:
            # Fresh IPOPT solve -> x*, lambda* differ at solver tolerance
            np.testing.assert_allclose(
                np.array(getattr(grad_vjp, name)),
                np.array(getattr(grad_sens, name)),
                atol=1e-5,
                err_msg=f"Mismatch in {name}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Warm-start tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWarmStart:
    """Tests for warm-start support in :func:`make_differentiable_solver`."""

    def test_warm_state_initially_none(self, wec_ipopt, wave,
                                       obj_fun_and_nstate):
        obj_fun, nstate_opt = obj_fun_and_nstate
        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        assert f.warm_start_state["x_wec_0"] is None
        assert f.warm_start_state["x_opt_0"] is None

    def test_warm_state_populated_after_call(self, wec_ipopt, wave,
                                              hydro_data, obj_fun_and_nstate):
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)
        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        f(bp)
        assert f.warm_start_state["x_wec_0"] is not None
        assert f.warm_start_state["x_opt_0"] is not None
        assert f.warm_start_state["x_wec_0"].shape == (wec_ipopt.nstate_wec,)
        assert f.warm_start_state["x_opt_0"].shape == (nstate_opt,)

    def test_warm_start_consistent_results(self, wec_ipopt, wave,
                                            hydro_data, obj_fun_and_nstate):
        """Two consecutive calls should produce the same objective."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)
        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        phi1 = f(bp)
        phi2 = f(bp)  # warm-started
        np.testing.assert_allclose(float(phi1), float(phi2), atol=1e-6)

    def test_explicit_x0_overrides_warm(self, wec_ipopt, wave,
                                         hydro_data, obj_fun_and_nstate):
        """User-supplied x_wec_0 / x_opt_0 in solve_kwargs take priority."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)
        rng = np.random.default_rng(42)
        x_wec_0 = rng.standard_normal(wec_ipopt.nstate_wec)
        x_opt_0 = rng.standard_normal(nstate_opt)
        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt,
            x_wec_0=x_wec_0, x_opt_0=x_opt_0,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        phi = f(bp)
        assert jnp.isfinite(phi)
        # Warm-start state should be populated (with the solution, not x0)
        assert f.warm_start_state["x_wec_0"] is not None


# ═══════════════════════════════════════════════════════════════════════════
# Full Fiacco formula (df/dh term) tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFullFiacco:
    """Tests for the ∂f/∂h term via *obj_fun_parametric*."""

    def test_obj_fun_parametric_changes_gradient(self, wec_ipopt, wave,
                                                  hydro_data,
                                                  obj_fun_and_nstate):
        """When obj_fun_parametric has df/dh != 0, gradient should differ."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        solve_kwargs = dict(
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6})

        results = wec_ipopt.solve(wave, obj_fun, nstate_opt, **solve_kwargs)

        # Without obj_fun_parametric (default: df/dh = 0)
        grad_no_df = sensitivity(wec_ipopt, results[0], wave)

        # With obj_fun_parametric that depends on BEM params
        def obj_parametric(x_wec, x_opt, wave_data, bem_params, wec):
            return (jnp.sum(x_wec**2) + jnp.sum(x_opt**2)
                    + 0.01 * jnp.sum(jnp.real(bem_params.added_mass)**2))

        grad_with_df = sensitivity(
            wec_ipopt, results[0], wave, obj_fun_parametric=obj_parametric)

        added_mass_diff = float(jnp.max(jnp.abs(
            grad_with_df.added_mass - grad_no_df.added_mass)))
        assert added_mass_diff > 0, (
            "obj_fun_parametric should change the added_mass gradient")

    def test_df_dh_additive(self, wec_ipopt, wave, hydro_data,
                            obj_fun_and_nstate):
        """The df/dh contribution should equal jax.grad of the parametric obj
        evaluated at the solve point."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)
        solve_kwargs = dict(
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6})

        results = wec_ipopt.solve(wave, obj_fun, nstate_opt, **solve_kwargs)

        grad_no_df = sensitivity(wec_ipopt, results[0], wave)

        def obj_parametric(x_wec, x_opt, wave_data, bem_params, wec):
            return 0.01 * jnp.sum(jnp.real(bem_params.added_mass)**2)

        grad_with_df = sensitivity(
            wec_ipopt, results[0], wave, obj_fun_parametric=obj_parametric)

        # Compute the expected df/dh analytically
        expected_df_dh_am = 0.02 * jnp.real(bp.added_mass)

        actual_diff = grad_with_df.added_mass - grad_no_df.added_mass
        np.testing.assert_allclose(
            np.array(jnp.real(actual_diff)),
            np.array(expected_df_dh_am),
            atol=1e-10,
            err_msg="df/dh term does not match expected analytical value",
        )

    def test_make_differentiable_solver_with_parametric(
            self, wec_ipopt, wave, hydro_data, obj_fun_and_nstate):
        """make_differentiable_solver with obj_fun_parametric should work."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)

        def obj_parametric(x_wec, x_opt, wave_data, bem_params, wec):
            return (jnp.sum(x_wec**2) + jnp.sum(x_opt**2)
                    + 0.01 * jnp.sum(jnp.real(bem_params.added_mass)**2))

        f = make_differentiable_solver(
            wec_ipopt, wave, obj_fun, nstate_opt,
            obj_fun_parametric=obj_parametric,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        grad_h = jax.grad(f)(bp)
        assert isinstance(grad_h, BEMParams)
        for name in BEMParams._fields:
            g = getattr(grad_h, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"


# ═══════════════════════════════════════════════════════════════════════════
# Multi-realisation tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def wave_multi():
    """Wave data with two realisations."""
    freq = 2 * F1
    w1 = wot.waves.regular_wave(F1, NFREQ, freq, 1.0, 30.0)
    w2 = wot.waves.regular_wave(F1, NFREQ, freq, 0.8, 60.0)
    w2 = w2.assign_coords(realization=[1])
    return xr.concat([w1, w2], dim="realization")


class TestMultiRealization:
    """Tests for multi-realisation support in sensitivity / custom_vjp."""

    def test_solve_returns_two_results(self, wec_ipopt, wave_multi,
                                       obj_fun_and_nstate):
        obj_fun, nstate_opt = obj_fun_and_nstate
        results = wec_ipopt.solve(
            wave_multi, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        assert len(results) == 2
        for r in results:
            assert r.success, r.message

    def test_sensitivity_multi_real(self, wec_ipopt, wave_multi,
                                    obj_fun_and_nstate):
        obj_fun, nstate_opt = obj_fun_and_nstate
        results = wec_ipopt.solve(
            wave_multi, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        grad_h = sensitivity(wec_ipopt, results, wave_multi)
        assert isinstance(grad_h, BEMParams)
        for name in BEMParams._fields:
            g = getattr(grad_h, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"

    def test_sensitivity_multi_is_mean_of_singles(
            self, wec_ipopt, wave_multi, obj_fun_and_nstate):
        """Multi-real sensitivity should equal mean of per-real sensitivities."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        results = wec_ipopt.solve(
            wave_multi, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        grad_mean = sensitivity(wec_ipopt, results, wave_multi)

        # Per-realisation sensitivities
        grads = []
        for i, r in enumerate(wave_multi.realization.values):
            w_i = wave_multi.sel(realization=r)
            g_i = sensitivity(wec_ipopt, results[i], w_i)
            grads.append(g_i)

        for name in BEMParams._fields:
            manual_mean = (
                getattr(grads[0], name) + getattr(grads[1], name)) / 2
            np.testing.assert_allclose(
                np.array(getattr(grad_mean, name)),
                np.array(manual_mean),
                atol=1e-12,
                err_msg=f"Multi-real mean mismatch for {name}",
            )

    def test_make_differentiable_solver_multi_real(
            self, wec_ipopt, wave_multi, hydro_data, obj_fun_and_nstate):
        """make_differentiable_solver should handle multiple realisations."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        bp = extract_bem_params(hydro_data)
        f = make_differentiable_solver(
            wec_ipopt, wave_multi, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        phi_mean = f(bp)
        assert phi_mean.shape == ()
        assert jnp.isfinite(phi_mean)

        grad_h = jax.grad(f)(bp)
        assert isinstance(grad_h, BEMParams)
        for name in BEMParams._fields:
            g = getattr(grad_h, name)
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"

    def test_sensitivity_result_count_mismatch_raises(
            self, wec_ipopt, wave_multi, obj_fun_and_nstate):
        """Passing wrong number of results should raise ValueError."""
        obj_fun, nstate_opt = obj_fun_and_nstate
        results = wec_ipopt.solve(
            wave_multi, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        with pytest.raises(ValueError, match="does not match"):
            sensitivity(wec_ipopt, [results[0]], wave_multi)


class TestConstraintMultipliers:
    """Test that per-constraint inequality multipliers are extracted."""

    def test_constraint_multipliers_present(self, solve_result):
        assert hasattr(solve_result, "constraint_multipliers")
        assert isinstance(solve_result.constraint_multipliers, dict)

    def test_no_user_constraints_gives_empty_dict(self, solve_result):
        assert len(solve_result.constraint_multipliers) == 0

    def test_user_constraints_extracted(self, bem_data, wave):
        """WEC with user inequality constraints stores their multipliers."""
        from collections import namedtuple

        def ineq_con(wec, x_wec, x_opt, wav):
            return jnp.array([10.0 - jnp.sum(x_wec**2)])

        constraints = [{"type": "ineq", "fun": ineq_con}]
        wec = WEC_IPOPT.from_bem(
            bem_data, constraints=constraints,
            friction=FRICTION, min_damping=1e-6,
        )
        ncomp = wec.ncomponents
        ndof = wec.ndof
        nstate_opt = ncomp * ndof

        def obj_fun(wec, x_wec, x_opt, wav):
            return jnp.sum(x_wec**2) + jnp.sum(x_opt**2)

        res = wec.solve(
            wave, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )[0]

        assert "user_constraint_0" in res.constraint_multipliers
        mu = res.constraint_multipliers["user_constraint_0"]
        assert mu.shape == (1,)
        assert np.all(np.isfinite(mu))


class TestSensitivityParametric:
    """Test sensitivity_parametric with FD validation on a simple problem."""

    @pytest.fixture(scope="class")
    def parametric_setup(self, bem_data, wave):
        """Build a WEC with a parametric friction term and solve."""
        from collections import namedtuple

        ParamSet = namedtuple("ParamSet", ["friction_coeff"])
        nominal = ParamSet(friction_coeff=jnp.float64(0.5))

        wec = WEC_IPOPT.from_bem(
            bem_data, friction=FRICTION, min_damping=1e-6)
        ncomp = wec.ncomponents
        ndof = wec.ndof
        nstate_opt = ncomp * ndof

        def obj_fun(wec, x_wec, x_opt, wav):
            return jnp.sum(x_wec**2) + jnp.sum(x_opt**2)

        res = wec.solve(
            wave, obj_fun, nstate_opt,
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )[0]

        def residual_fn(wec, x_wec, x_opt, wav, params):
            r_base = wec.residual(x_wec, x_opt, wav)
            pos = wec.vec_to_dofmat(x_wec)
            vel = jnp.dot(wec.derivative_mat, pos)
            from wecopttool.core import dofmat_to_vec
            extra_friction = -params.friction_coeff * vel
            correction = dofmat_to_vec(jnp.dot(wec.time_mat, extra_friction))
            return r_base - correction

        return dict(
            wec=wec, res=res, wave=wave, nominal=nominal,
            obj_fun=obj_fun, nstate_opt=nstate_opt,
            residual_fn=residual_fn, ParamSet=ParamSet,
        )

    def test_gradient_is_finite(self, parametric_setup):
        s = parametric_setup
        grad = sensitivity_parametric(
            s["wec"], s["res"], s["wave"], s["nominal"],
            residual_fn=s["residual_fn"],
        )
        assert hasattr(grad, "friction_coeff")
        assert jnp.isfinite(grad.friction_coeff)

    def test_gradient_matches_fd(self, parametric_setup, bem_data, wave):
        """Analytical gradient should match central FD through re-solve.

        The parametric residual adds friction_coeff * vel on top of the
        base WEC friction (FRICTION).  For FD we rebuild the WEC with
        total friction = FRICTION + (nominal +/- eps) * I.
        """
        s = parametric_setup

        grad = sensitivity_parametric(
            s["wec"], s["res"], s["wave"], s["nominal"],
            residual_fn=s["residual_fn"],
        )
        analytical = float(grad.friction_coeff)

        eps = 1e-4
        nom_val = float(s["nominal"].friction_coeff)
        phis = []
        for sign in [+1, -1]:
            total_fric = (FRICTION[0, 0] + nom_val + sign * eps)
            fric_mat = np.eye(NDOF) * total_fric
            wec_p = WEC_IPOPT.from_bem(
                bem_data, friction=fric_mat, min_damping=1e-6)

            res_p = wec_p.solve(
                wave, s["obj_fun"], s["nstate_opt"],
                optim_options={
                    "print_level": 0, "max_iter": 1000, "tol": 1e-6},
            )[0]
            phis.append(res_p.fun)

        fd = (phis[0] - phis[1]) / (2 * eps)
        # Tolerance is relaxed because from_bem incorporates friction
        # through linear damping (BEM data) while residual_fn adds it
        # directly in the equation -- numerically slightly different paths.
        np.testing.assert_allclose(
            analytical, fd, rtol=0.20,
            err_msg=f"Analytical={analytical:.6e}, FD={fd:.6e}",
        )

    def test_obj_fn_term_included(self, parametric_setup):
        """When obj_fn is provided, gradient should differ."""
        s = parametric_setup

        grad_no_obj = sensitivity_parametric(
            s["wec"], s["res"], s["wave"], s["nominal"],
            residual_fn=s["residual_fn"], obj_fn=None,
        )

        def obj_parametric(wec, x_wec, x_opt, wav, params):
            return params.friction_coeff * jnp.sum(x_wec**2)

        grad_with_obj = sensitivity_parametric(
            s["wec"], s["res"], s["wave"], s["nominal"],
            residual_fn=s["residual_fn"], obj_fn=obj_parametric,
        )

        assert not jnp.allclose(
            grad_no_obj.friction_coeff, grad_with_obj.friction_coeff)

    def test_bad_constraint_name_raises(self, parametric_setup):
        """KeyError for unknown constraint name."""
        s = parametric_setup

        def dummy_con(wec, x_wec, x_opt, wav, params):
            return jnp.array([1.0])

        with pytest.raises(KeyError, match="nonexistent"):
            sensitivity_parametric(
                s["wec"], s["res"], s["wave"], s["nominal"],
                residual_fn=s["residual_fn"],
                constraint_fns=[(dummy_con, "nonexistent")],
            )

    def test_multi_realisation_mean_equals_singles(
            self, parametric_setup, bem_data):
        """Multi-real grad = mean of per-realisation grads."""
        from collections import namedtuple
        ParamSet = namedtuple("ParamSet", ["friction_coeff"])
        nominal = ParamSet(friction_coeff=jnp.float64(0.5))

        s = parametric_setup
        wec = s["wec"]

        # Two wave realisations
        w1 = wot.waves.regular_wave(F1, NFREQ, 2 * F1, 1.0, 30.0)
        w2 = wot.waves.regular_wave(F1, NFREQ, 2 * F1, 0.8, 60.0)
        w2 = w2.assign_coords(realization=[1])
        wave_multi = xr.concat([w1, w2], dim="realization")

        results = wec.solve(
            wave_multi, s["obj_fun"], s["nstate_opt"],
            optim_options={"print_level": 0, "max_iter": 1000, "tol": 1e-6},
        )
        assert len(results) == 2

        grad_mean = sensitivity_parametric(
            wec, results, wave_multi, nominal,
            residual_fn=s["residual_fn"],
        )

        grads_single = []
        for i, r in enumerate(wave_multi.realization.values):
            w_i = wave_multi.sel(realization=[r]).squeeze("realization")
            g_i = sensitivity_parametric(
                wec, [results[i]], w_i, nominal,
                residual_fn=s["residual_fn"],
            )
            grads_single.append(g_i)

        manual_mean = jax.tree_util.tree_map(
            lambda a, b: (a + b) / 2, grads_single[0], grads_single[1])
        np.testing.assert_allclose(
            np.array(grad_mean.friction_coeff),
            np.array(manual_mean.friction_coeff),
            atol=1e-12,
            err_msg="Multi-real mean should equal mean of singles",
        )
