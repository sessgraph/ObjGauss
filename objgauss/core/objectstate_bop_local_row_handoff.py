"""Deprecated compatibility import for the BOP local-row handoff."""

from objgauss.pipelines.objectstate_bop_local_row_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
    objectstate_bop_local_row_handoff,
    validate_objectstate_bop_local_row_handoff_summary,
)

__all__ = (
    "OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA",
    "objectstate_bop_local_row_handoff",
    "validate_objectstate_bop_local_row_handoff_summary",
)
