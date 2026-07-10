"""Deprecated compatibility import for assignment/renderer joint validation."""

from objgauss.pipelines.assignment_v2_renderer_validation import (
    ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA,
    AssignmentV2RendererJointValidationReport,
    evaluate_assignment_v2_renderer_joint,
    validate_assignment_v2_renderer_joint_summary,
)

__all__ = (
    "ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA",
    "AssignmentV2RendererJointValidationReport",
    "evaluate_assignment_v2_renderer_joint",
    "validate_assignment_v2_renderer_joint_summary",
)
