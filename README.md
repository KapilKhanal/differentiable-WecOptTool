# wecopttool-differentiable

Extension package for [WecOptTool](https://github.com/sandialabs/WecOptTool) that adds Fiacco post-optimality sensitivity analysis and JAX differentiability.

## Features

- **WEC_IPOPT**: IPOPT-backed solver (via cyipopt) that returns Lagrange multipliers
- **sensitivity()**: Fiacco sensitivity for BEM, PTO, or joint parameters
- **make_differentiable_solver()**: JAX-differentiable solve via `custom_vjp`
- **residual_parametric**: Pure-JAX dynamics residual for `jax.vjp` / `jax.grad`

## Installation

```bash
pip install wecopttool wecopttool[ipopt]
pip install wecopttool-differentiable
```

Or install from source:

```bash
pip install -e .
```

## Quick start

```python
import wecopttool as wot
from wecopttool_differentiable import WEC_IPOPT, sensitivity

wec = WEC_IPOPT.from_bem(hydro_data, friction=friction)
results = wec.solve(waves, obj_fun, nstate_opt)
res = results[0]

# Fiacco sensitivity (BEM parameters)
grad_h = sensitivity(wec, res, waves)
```

## Dependencies

- [wecopttool](https://pypi.org/project/wecopttool/) (Sandia)
- [jax](https://pypi.org/project/jax/)
- [cyipopt](https://pypi.org/project/cyipopt/)

## License

GPL-3.0 (same as WecOptTool)
