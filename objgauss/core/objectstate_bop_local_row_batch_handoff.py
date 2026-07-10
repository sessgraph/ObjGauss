"""Deprecated compatibility imports for the BOP local-row batch contract."""

from objgauss.datasets.objectstate_bop_local_row_batch_spec import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
    read_objectstate_bop_local_row_batch_spec,
    validate_objectstate_bop_local_row_batch_spec,
)
from objgauss.pipelines.objectstate_bop_local_row_batch_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA,
    objectstate_bop_local_row_batch_handoff,
    validate_objectstate_bop_local_row_batch_handoff_summary,
)

__all__ = (
    "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA",
    "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA",
    "read_objectstate_bop_local_row_batch_spec",
    "objectstate_bop_local_row_batch_handoff",
    "validate_objectstate_bop_local_row_batch_spec",
    "validate_objectstate_bop_local_row_batch_handoff_summary",
)
