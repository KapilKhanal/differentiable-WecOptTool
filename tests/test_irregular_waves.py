"""Tests for multi-realization (irregular wave) sensitivity averaging.

Validates:
1. sensitivity() with multiple realizations returns the same structure
   as a single-realization call.
2. The averaged gradient equals the mean of per-realization gradients.
3. Each per-realization gradient is finite.

Uses the WaveBot geometry with Pierson-Moskowitz irregular waves.

Run with::

    pytest tests/test_irregular_waves.py -v
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import wecopttool as wot
from wecopttool_differentiable import (
    WEC_IPOPT,
    sensitivity,
    BEMParams,
    extract_bem_params,
)

jax.config.update("jax_enable_x64", True)

NREALIZATIONS = 2


@pytest.fixture(scope="module")
def irregular_setup():
    """Build WaveBot with irregular waves, solve all realizations."""
    import capytaine as cpy
    from capytaine.io.meshio import load_from_meshio

    wavefreq = 0.3
    f1 = wavefreq
    nfreq = 10
    freq = wot.frequency(f1, nfreq, False)

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

    Tp = 1.0 / wavefreq
    Hs = 0.125
    spectrum = wot.waves.pierson_moskowitz_spectrum(freq, Tp, Hs)

    waves = wot.waves.long_crested_wave(
        spectrum, nrealizations=NREALIZATIONS, direction=0, seed=42)

    obj_fun = pto.mechanical_average_power
    nstate_opt = 2 * nfreq

    solve_kw = dict(
        scale_x_wec=1e1, scale_x_opt=1e-3, scale_obj=1e-2,
        optim_options={"max_iter": 1000, "tol": 1e-8, "print_level": 0},
    )

    results = wec.solve(waves, obj_fun, nstate_opt, **solve_kw)
    assert len(results) == NREALIZATIONS
    for i, r in enumerate(results):
        assert r.success, f"IPOPT failed for realization {i}: {r.message}"

    hydro_data = wot.add_linear_friction(bem_data, friction=None)
    hydro_data = wot.check_radiation_damping(hydro_data)
    bp = extract_bem_params(hydro_data)

    return dict(
        wec=wec, waves=waves, results=results, bp=bp,
        obj_fun=obj_fun, nstate_opt=nstate_opt, solve_kw=solve_kw,
    )


class TestIrregularSensitivityStructure:
    """Multi-realization sensitivity returns correct structure."""

    def test_returns_bem_params(self, irregular_setup):
        s = irregular_setup
        grad = sensitivity(s["wec"], s["results"], s["waves"])
        for name in BEMParams._fields:
            g = getattr(grad, name)
            assert g is not None, f"Missing field {name}"
            assert jnp.all(jnp.isfinite(g)), f"Non-finite in grad_{name}"


class TestIrregularSensitivityAveraging:
    """Averaged gradient equals mean of per-realization gradients."""

    def test_average_matches_per_realization(self, irregular_setup):
        s = irregular_setup
        wec, waves, results = s["wec"], s["waves"], s["results"]

        grad_avg = sensitivity(wec, results, waves)

        per_real_grads = []
        for i in range(NREALIZATIONS):
            wave_i = waves.sel(realization=i)
            grad_i = sensitivity(wec, results[i], wave_i)
            per_real_grads.append(grad_i)

        for name in BEMParams._fields:
            g_avg = np.array(getattr(grad_avg, name))
            g_manual = np.mean(
                [np.array(getattr(g, name)) for g in per_real_grads],
                axis=0,
            )
            np.testing.assert_allclose(
                g_avg, g_manual, atol=1e-12,
                err_msg=f"Averaged gradient mismatch for {name}",
            )


class TestIrregularPerRealizationFinite:
    """Each per-realization gradient is finite and non-trivial."""

    def test_each_realization_finite(self, irregular_setup):
        s = irregular_setup
        wec, waves, results = s["wec"], s["waves"], s["results"]

        for i in range(NREALIZATIONS):
            wave_i = waves.sel(realization=i)
            grad_i = sensitivity(wec, results[i], wave_i)
            for name in BEMParams._fields:
                g = getattr(grad_i, name)
                assert jnp.all(jnp.isfinite(g)), \
                    f"Non-finite in realization {i}, field {name}"
