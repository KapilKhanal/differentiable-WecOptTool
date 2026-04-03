# wecopttool-differentiable

Extension package for [WecOptTool](https://github.com/sandialabs/WecOptTool) that adds **post-optimality sensitivity analysis** via the Fiacco envelope theorem and JAX automatic differentiation.

Given an optimal WEC control solution from IPOPT, this package computes exact gradients of the optimal objective (e.g., power) with respect to all BEM hydrodynamic parameters — in a single backward pass, **without re-solving** the optimization problem.

## What it does

1. **Solve** the WEC optimal control problem with IPOPT (returns Lagrange multipliers)
2. **Differentiate** through the dynamics residual using JAX
3. **Compute** $d\varphi^*/dh = \lambda^\top \partial r / \partial h$ via the Fiacco formula

The gradient is exact (up to solver tolerance) and covers all ~63 real-valued BEM entries simultaneously.

## Installation

```bash
# Install cyipopt (pre-built binaries via conda recommended)
conda install -c conda-forge cyipopt

# Install wecopttool and this package
pip install wecopttool
pip install -e .
```

If `cyipopt` fails to build from PyPI, install from git:

```bash
pip install "cyipopt @ git+https://github.com/mechmotum/cyipopt.git"
```

## Quick start

```python
import wecopttool as wot
from wecopttool_differentiable import (
    WEC_IPOPT,
    extract_bem_params,
    extract_wave_data,
    residual_parametric,
)
import jax

# Build WEC with IPOPT solver (returns Lagrange multipliers)
wec = WEC_IPOPT.from_bem(bem_data, constraints=constraints, f_add=f_add)
results = wec.solve(waves, obj_fun, nstate_opt)
res = results[0]

# Extract BEM parameters and wave data as JAX arrays
hydro_data = wot.add_linear_friction(bem_data, friction=None)
hydro_data = wot.check_radiation_damping(hydro_data)
bp = extract_bem_params(hydro_data)
wd = extract_wave_data(wave, hydro_data["Froude_Krylov_force"])

# Fiacco sensitivity: lambda^T * dr/dh via jax.vjp
x_wec, x_opt = wec.decompose_state(res.x)
lam = res.dynamics_mult_g

def r_of_h(h):
    return residual_parametric(jnp.array(x_wec), jnp.array(x_opt), wd, h, wec)

_, vjp_fn = jax.vjp(r_of_h, bp)
(grad_h,) = vjp_fn(jnp.array(lam))
```

`grad_h` is a `BEMParams` namedtuple containing the gradient of optimal power with respect to every BEM parameter (added mass, radiation damping, excitation forces, hydrostatic stiffness, inertia, friction).

## Tutorial

See [`examples/tutorial_1_WaveBot_sensitivity.ipynb`](examples/tutorial_1_WaveBot_sensitivity.ipynb) for a complete walkthrough including:

- WaveBot geometry setup and BEM
- IPOPT solve with force constraints
- Step-by-step Fiacco sensitivity computation
- Sensitivity visualization (bar charts, per-frequency plots)
- Finite-difference validation (VJP-level and full NLP re-solve)

## Running tests

```bash
# Unit tests (fast, no IPOPT re-solves)
pytest tests/ -v -m "not validation"

# Full validation suite (includes IPOPT re-solves, slower)
pytest tests/ -v
```

## Package contents

| Module | Purpose |
|--------|---------|
| `solver_ipopt.py` | `WEC_IPOPT` solver, `sensitivity()`, `make_differentiable_solver()` |
| `parametric.py` | JAX-differentiable dynamics residual, BEM parameter extraction |
| `qp_kkt.py` | KKT-based backward differentiation (adjoint KKT system) |
| `sensitivity_plots.py` | Visualization utilities for sensitivity results |

## Dependencies

- [wecopttool](https://github.com/sandialabs/WecOptTool)
- [JAX](https://github.com/google/jax)
- [cyipopt](https://github.com/mechmotum/cyipopt)
- NumPy, SciPy, xarray

## License

GPL-3.0 (same as WecOptTool)
