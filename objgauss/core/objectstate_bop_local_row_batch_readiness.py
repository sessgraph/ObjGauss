"""Deprecated compatibility import for BOP local-row batch readiness."""

from objgauss.pipelines.objectstate_bop_local_row_batch_readiness import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA,
    objectstate_bop_local_row_batch_readiness,
    validate_objectstate_bop_local_row_batch_readiness_summary,
)

__all__ = (
    "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA",
    "objectstate_bop_local_row_batch_readiness",
    "validate_objectstate_bop_local_row_batch_readiness_summary",
)
