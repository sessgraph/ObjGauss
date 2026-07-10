"""Deprecated compatibility import for the BOP identity handoff."""

from objgauss.pipelines.objectstate_bop_identity_handoff import (
    OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA,
    objectstate_bop_identity_handoff,
    validate_objectstate_bop_identity_handoff_summary,
)

__all__ = (
    "OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA",
    "objectstate_bop_identity_handoff",
    "validate_objectstate_bop_identity_handoff_summary",
)
