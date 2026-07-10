"""Deprecated compatibility import for real identity-row evaluation."""

from objgauss.evaluation.objectstate_real_identity_rows import (
    OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA,
    objectstate_real_identity_rows_from_bundle,
    objectstate_real_identity_rows_summary,
    read_objectstate_real_identity_rows_summary,
    validate_objectstate_real_identity_rows_summary,
)

__all__ = (
    "OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA",
    "read_objectstate_real_identity_rows_summary",
    "objectstate_real_identity_rows_summary",
    "objectstate_real_identity_rows_from_bundle",
    "validate_objectstate_real_identity_rows_summary",
)
