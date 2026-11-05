from __future__ import annotations

"""Compatibility shim for legacy imports.

Prefer importing directly from:
- sr2hand.optimization.phase1_common
- sr2hand.optimization.phase1_strategy_cem
- sr2hand.optimization.phase1_strategy_mjx_autodiff
- sr2hand.optimization.phase1_strategy_diffmjx
"""

from .phase1_common import Phase1EvalConfig, Phase1GraspEvaluator
from .phase1_strategy_cem import Phase1OptimizationConfig, optimize_finger_controls
from .phase1_strategy_diffmjx import (
    Phase1DiffMJXMVPConfig,
    optimize_finger_controls_diffmjx_mvp,
)
from .phase1_strategy_mjx_autodiff import (
    Phase1AutodiffConfig,
    optimize_finger_controls_autodiff,
)

__all__ = [
    "Phase1EvalConfig",
    "Phase1GraspEvaluator",
    "Phase1OptimizationConfig",
    "Phase1AutodiffConfig",
    "Phase1DiffMJXMVPConfig",
    "optimize_finger_controls",
    "optimize_finger_controls_autodiff",
    "optimize_finger_controls_diffmjx_mvp",
]
