"""Deprecated compatibility import for real prediction-row evaluation."""

from objgauss.evaluation.objectstate_real_prediction_rows import (
    OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA,
    objectstate_real_prediction_rows_from_bundle,
    objectstate_real_prediction_rows_summary,
    read_objectstate_real_prediction_rows_summary,
    validate_objectstate_real_prediction_rows_summary,
)

__all__ = (
    "OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA",
    "read_objectstate_real_prediction_rows_summary",
    "objectstate_real_prediction_rows_summary",
    "objectstate_real_prediction_rows_from_bundle",
    "validate_objectstate_real_prediction_rows_summary",
)
