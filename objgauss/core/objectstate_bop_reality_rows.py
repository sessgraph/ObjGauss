"""Deprecated compatibility import for BOP reality-row evaluation."""

from objgauss.evaluation.objectstate_bop_reality_rows import (
    OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
    objectstate_bop_reality_rows_from_summary,
    objectstate_bop_reality_rows_summary,
    read_objectstate_bop_local_row_summary,
    validate_objectstate_bop_reality_rows_summary,
)

__all__ = (
    "OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA",
    "read_objectstate_bop_local_row_summary",
    "objectstate_bop_reality_rows_from_summary",
    "objectstate_bop_reality_rows_summary",
    "validate_objectstate_bop_reality_rows_summary",
)
