"""wecopttool-differentiable: Fiacco & KKT sensitivity for WecOptTool.

Extension package for Sandia's WecOptTool. Adds:
- WEC_IPOPT: IPOPT-backed solver that returns Lagrange multipliers
- sensitivity(): post-optimality Fiacco envelope-theorem gradient
- make_differentiable_solver(): JAX-differentiable solve via custom_vjp
  (Fiacco for scalar objective, KKT for full state)

Requires wecopttool and cyipopt.
"""

from .solver_ipopt import (
    WEC_IPOPT,
    sensitivity,
    make_differentiable_solver,
)

from .parametric import (
    BEMParams,
    WaveData,
    extract_bem_params,
    extract_wave_data,
    residual_parametric,
)

from .qp_kkt import kkt_vjp

from .sensitivity_plots import (
    plot_sensitivity_bars,
    plot_frequency_sensitivity,
    plot_fd_comparison,
)

__all__ = [
    "WEC_IPOPT",
    "sensitivity",
    "make_differentiable_solver",
    "BEMParams",
    "WaveData",
    "extract_bem_params",
    "extract_wave_data",
    "residual_parametric",
    "kkt_vjp",
    "plot_sensitivity_bars",
    "plot_frequency_sensitivity",
    "plot_fd_comparison",
]
