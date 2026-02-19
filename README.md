# wecopttool-differentiable

Extension package for [WecOptTool](https://github.com/sandialabs/WecOptTool) that adds Fiacco post-optimality sensitivity analysis and JAX differentiability.

## Features

- **WEC_IPOPT**: IPOPT-backed solver (via cyipopt) that returns Lagrange multipliers
- **sensitivity()**: Fiacco sensitivity for BEM, PTO, or joint parameters
- **make_differentiable_solver()**: JAX-differentiable solve via `custom_vjp`
- **residual_parametric**: Pure-JAX dynamics residual for `jax.vjp` / `jax.grad`

## Installation

```bash
pip install wecopttool
pip install wecopttool-differentiable
```

If `cyipopt` fails to build from PyPI (deprecated setuptools in older releases), install it from git first:

```bash
pip install "cyipopt @ git+https://github.com/mechmotum/cyipopt.git"
pip install wecopttool wecopttool-differentiable
```

Or use conda for cyipopt (pre-built binaries):

```bash
conda install -c conda-forge cyipopt
pip install wecopttool wecopttool-differentiable
```

To install from source:

```bash
pip install -e .
```

## Quick start

### BEM sensitivity (simplest case)

```python
import wecopttool as wot
from wecopttool_differentiable import WEC_IPOPT, sensitivity

wec = WEC_IPOPT.from_bem(hydro_data, friction=friction, f_add=f_add)
results = wec.solve(waves, obj_fun, nstate_opt)
res = results[0]

# Fiacco sensitivity (BEM parameters) — one-liner
grad_h = sensitivity(wec, res, waves)

# Or use the convenience method on the WEC object directly:
grad_h = wec.compute_sensitivity(res, waves)
```

### PTO sensitivity (parametric forces)

```python
from wecopttool_differentiable import (
    sensitivity, make_pto_passive_parametric,
    make_electrical_power_obj_parametric,
)

f_pto_param = make_pto_passive_parametric(gear_ratios, friction_dict)
obj_param = make_electrical_power_obj_parametric(pto)

# additional_forces auto-computed from wec.forces when omitted
grad_pto = sensitivity(
    wec, res, waves,
    params=pto_params,
    parametric_forces={"PTO_passive": f_pto_param},
    obj_fn=obj_param,
)
```

### Joint BEM + PTO sensitivity

```python
from wecopttool_differentiable import make_joint_params, extract_bem_params

joint = make_joint_params(extract_bem_params(hydro_data), pto=pto_params)
grad_joint = sensitivity(
    wec, res, waves,
    params=joint,
    parametric_forces={"PTO_passive": f_pto_param},
    obj_fn=obj_param,
)
# grad_joint.bem  -> BEMParams gradients
# grad_joint.pto  -> PTOParams gradients
```

## Validating sensitivity results

The package provides three levels of finite-difference validation to give
you confidence the analytical gradients are correct before using them in
gradient-based optimization.

### Validation levels

| Function | What it checks | Expected tolerance | Speed |
|---|---|---|---|
| `fd_check_residual` | lambda^T dr/dp | < 1e-4 relative | Fast (no re-solve) |
| `fd_check_objective` | df/dp | < 1e-4 relative | Fast (no re-solve) |
| `fd_validate` | Full d(phi\*)/dp | < 10% relative | Slow (IPOPT re-solve per param) |

The looser tolerance for `fd_validate` is expected because:

- IPOPT solver tolerance (~1e-6) introduces noise in phi\*
- Active-set changes at perturbed parameters
- Central differences have O(eps^2) truncation error

### Quick validation

```python
from wecopttool_differentiable import validate_sensitivity

report = validate_sensitivity(
    wec, res, waves, grad_pto,
    params=pto_params,
    parametric_forces={"PTO_passive": f_pto_param},
    obj_fn=obj_param,
)
# Runs residual + objective checks automatically and prints a summary.
```

### Full NLP re-solve validation

```python
from wecopttool_differentiable import fd_validate, make_re_solve_fn

re_solve = make_re_solve_fn(wec_factory, waves, obj_fun, nstate_opt, **solve_kw)
results = fd_validate(grad_pto, pto_params, re_solve)
```

## Running tests

```bash
# Fast unit tests (skip expensive re-solve validation)
pytest tests/ -v -m "not validation"

# Full validation suite (includes IPOPT re-solves)
pytest tests/ -v

# Only the expensive validation tests
pytest tests/ -v -m "validation"
```

## API tiers

The package organises exports into four tiers:

- **Tier 1 (core):** `WEC_IPOPT`, `sensitivity`, `make_differentiable_solver`
- **Tier 2 (parametric factories):** `make_joint_params`, `make_pto_passive_parametric`, `make_linear_mooring_parametric`, `make_electrical_power_obj_parametric`
- **Tier 3 (validation):** `fd_validate`, `fd_check_residual`, `fd_check_objective`, `validate_sensitivity`, `make_re_solve_fn`, `FDResult`
- **Tier 4 (advanced):** `BEMParams`, `WaveData`, `extract_bem_params`, `extract_wave_data`, `residual_parametric`, plotting utilities

Most users only need Tier 1. Import Tier 2 when working with PTO or custom parameters. Use Tier 3 to verify gradients. Tier 4 is for advanced use (building custom residual functions).

## Dependencies

- [wecopttool](https://pypi.org/project/wecopttool/) (Sandia)
- [jax](https://pypi.org/project/jax/)
- [cyipopt](https://pypi.org/project/cyipopt/)

## License

GPL-3.0 (same as WecOptTool)
