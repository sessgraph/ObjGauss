"""Deprecated compatibility surface for assignment solver checkpoint and evaluation."""

from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA,
    assignment_solver_v2_checkpoint,
    assignment_solver_v2_state_from_checkpoint,
    validate_assignment_solver_v2_checkpoint,
)
from objgauss.evaluation.assignment_solver_v2_eval import (
    ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA,
    AssignmentSolverV2StabilityEvalReport,
    evaluate_assignment_solver_v2_stability,
    validate_assignment_solver_v2_stability_eval_summary,
)

__all__ = (
    "ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA",
    "ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA",
    "AssignmentSolverV2StabilityEvalReport",
    "assignment_solver_v2_checkpoint",
    "assignment_solver_v2_state_from_checkpoint",
    "evaluate_assignment_solver_v2_stability",
    "validate_assignment_solver_v2_checkpoint",
    "validate_assignment_solver_v2_stability_eval_summary",
)
