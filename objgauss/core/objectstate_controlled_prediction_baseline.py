"""Deprecated compatibility import for the controlled prediction baseline."""

from objgauss.pipelines.objectstate_controlled_prediction_baseline import (
    OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA,
    validate_objectstate_controlled_prediction_baseline_summary,
    write_objectstate_controlled_prediction_baseline_candidates,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA",
    "write_objectstate_controlled_prediction_baseline_candidates",
    "validate_objectstate_controlled_prediction_baseline_summary",
)
