"""wecopttool-differentiable: Fiacco sensitivity and JAX differentiability for WecOptTool.

Extension package for Sandia's WecOptTool. Adds:
- WEC_IPOPT: IPOPT-backed solver that returns Lagrange multipliers
- sensitivity(): Fiacco post-optimality sensitivity (BEM, PTO, joint)
- make_differentiable_solver(): JAX-differentiable solve via custom_vjp

Requires wecopttool and cyipopt.
"""

# -- Tier 1: core solver and sensitivity ------------------------------------
from .solver_ipopt import (
    WEC_IPOPT,
    sensitivity,
    make_differentiable_solver,
    make_differentiable_state_solver,
    ffo_sensitivity,
    sensitivity_parametric,
)

# -- Tier 2: parametric force / objective factories -------------------------
from .parametric_utils import (
    make_joint_params,
    make_linear_mooring_parametric,
    make_pto_passive_parametric,
    make_electrical_power_obj_parametric,
)

# -- Tier 3: validation utilities -------------------------------------------
from .validation import (
    fd_validate,
    fd_check_residual,
    fd_check_objective,
    make_re_solve_fn,
    validate_sensitivity,
    check_regularity,
    FDResult,
    RegularityResult,
)

# -- Tier 4: advanced / internals ------------------------------------------
from .parametric import (
    BEMParams,
    WaveData,
    extract_bem_params,
    extract_wave_data,
    residual_parametric,
)
from .sensitivity_plots import (
    plot_sensitivity_bars,
    plot_frequency_sensitivity,
    plot_fd_comparison,
)

__all__ = [
    # Tier 1 — core
    "WEC_IPOPT",
    "sensitivity",
    "make_differentiable_solver",
    "make_differentiable_state_solver",
    "ffo_sensitivity",
    "sensitivity_parametric",
    # Tier 2 — parametric factories
    "make_joint_params",
    "make_linear_mooring_parametric",
    "make_pto_passive_parametric",
    "make_electrical_power_obj_parametric",
    # Tier 3 — validation
    "fd_validate",
    "fd_check_residual",
    "fd_check_objective",
    "make_re_solve_fn",
    "validate_sensitivity",
    "check_regularity",
    "FDResult",
    "RegularityResult",
    # Tier 4 — advanced / internals
    "BEMParams",
    "WaveData",
    "extract_bem_params",
    "extract_wave_data",
    "residual_parametric",
    "plot_sensitivity_bars",
    "plot_frequency_sensitivity",
    "plot_fd_comparison",
]
