"""Deprecated compatibility import for the temporal assignment runner."""

from objgauss.pipelines.objectstate_temporal_assignment import (
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA,
    objectstate_temporal_assignment_summary,
    validate_objectstate_temporal_assignment_summary,
)

__all__ = (
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA",
    "objectstate_temporal_assignment_summary",
    "validate_objectstate_temporal_assignment_summary",
)
