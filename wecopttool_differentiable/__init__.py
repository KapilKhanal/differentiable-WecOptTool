"""wecopttool-differentiable: Fiacco sensitivity and JAX differentiability for WecOptTool.

Extension package for Sandia's WecOptTool. Adds:
- WEC_IPOPT: IPOPT-backed solver that returns Lagrange multipliers
- sensitivity(): Fiacco post-optimality sensitivity (BEM, PTO, joint)
- make_differentiable_solver(): JAX-differentiable solve via custom_vjp

Requires wecopttool and cyipopt.
"""

from .parametric import (
    BEMParams,
    WaveData,
    extract_bem_params,
    extract_wave_data,
    residual_parametric,
)
from .solver_ipopt import (
    WEC_IPOPT,
    make_differentiable_solver,
    sensitivity,
    sensitivity_parametric,
)
from .sensitivity_plots import (
    plot_sensitivity_bars,
    plot_frequency_sensitivity,
    plot_fd_comparison,
)

__all__ = [
    "BEMParams",
    "WaveData",
    "extract_bem_params",
    "extract_wave_data",
    "residual_parametric",
    "WEC_IPOPT",
    "make_differentiable_solver",
    "sensitivity",
    "sensitivity_parametric",
    "plot_sensitivity_bars",
    "plot_frequency_sensitivity",
    "plot_fd_comparison",
]
