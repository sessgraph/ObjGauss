"""Deprecated compatibility import for the controlled identity handoff."""

from objgauss.pipelines.objectstate_controlled_identity_handoff import (
    OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA,
    objectstate_controlled_identity_handoff,
    validate_objectstate_controlled_identity_handoff_summary,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA",
    "OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA",
    "OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA",
    "objectstate_controlled_identity_handoff",
    "validate_objectstate_controlled_identity_handoff_summary",
)
