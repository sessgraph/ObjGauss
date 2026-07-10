"""Deprecated compatibility import for public-interaction reality rows."""

from objgauss.evaluation.objectstate_public_interaction_reality_rows import (
    OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA,
    objectstate_public_interaction_reality_rows_from_handoff,
    objectstate_public_interaction_reality_rows_summary,
    read_objectstate_public_interaction_handoff_summary,
    validate_objectstate_public_interaction_reality_rows_summary,
)

__all__ = (
    "OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA",
    "read_objectstate_public_interaction_handoff_summary",
    "objectstate_public_interaction_reality_rows_from_handoff",
    "objectstate_public_interaction_reality_rows_summary",
    "validate_objectstate_public_interaction_reality_rows_summary",
)
