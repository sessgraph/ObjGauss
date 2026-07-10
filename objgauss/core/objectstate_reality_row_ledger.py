"""Deprecated compatibility import for the aggregate reality-row ledger."""

from objgauss.evaluation.objectstate_reality_row_ledger import (
    OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA,
    objectstate_reality_row_ledger,
    objectstate_reality_rows_from_summary,
    read_objectstate_reality_row_summary,
    validate_objectstate_reality_row_ledger_summary,
)

__all__ = (
    "OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA",
    "read_objectstate_reality_row_summary",
    "objectstate_reality_rows_from_summary",
    "objectstate_reality_row_ledger",
    "validate_objectstate_reality_row_ledger_summary",
)
