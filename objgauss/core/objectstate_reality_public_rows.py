"""Deprecated compatibility import for public-artifact reality rows."""

from objgauss.evaluation.objectstate_reality_public_rows import (
    OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA,
    ObjectStateRealityPublicArtifact,
    default_objectstate_reality_public_artifacts,
    evaluate_public_artifact_reality_gate,
    objectstate_reality_public_rows_summary,
    objectstate_reality_rows_from_public_artifacts,
    validate_objectstate_reality_public_artifact,
    validate_objectstate_reality_public_rows_summary,
)

__all__ = (
    "OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA",
    "ObjectStateRealityPublicArtifact",
    "default_objectstate_reality_public_artifacts",
    "objectstate_reality_rows_from_public_artifacts",
    "evaluate_public_artifact_reality_gate",
    "objectstate_reality_public_rows_summary",
    "validate_objectstate_reality_public_rows_summary",
    "validate_objectstate_reality_public_artifact",
)
