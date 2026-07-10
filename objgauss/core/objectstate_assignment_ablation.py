"""Deprecated compatibility import for assignment ablation experiments."""

from objgauss.pipelines.objectstate_assignment_ablation import (
    DEFAULT_ASSIGNMENT_ABLATION_POLICIES,
    OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA,
    objectstate_assignment_ablation_summary,
    validate_objectstate_assignment_ablation_summary,
)

__all__ = (
    "OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA",
    "DEFAULT_ASSIGNMENT_ABLATION_POLICIES",
    "objectstate_assignment_ablation_summary",
    "validate_objectstate_assignment_ablation_summary",
)
