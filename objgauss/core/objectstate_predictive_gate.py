"""Deprecated compatibility import for the ObjectState predictive gate."""

from objgauss.evaluation.objectstate_predictive_gate import (
    OBJECTSTATE_PREDICTIVE_GATE_SCHEMA,
    ObjectStatePredictiveGateReport,
    ObjectStatePredictiveGateThresholds,
    ObjectStatePredictiveRow,
    evaluate_objectstate_predictive_gate,
    validate_objectstate_predictive_gate_summary,
    validate_objectstate_predictive_gate_thresholds,
    validate_objectstate_predictive_row,
)

__all__ = (
    "OBJECTSTATE_PREDICTIVE_GATE_SCHEMA",
    "ObjectStatePredictiveGateThresholds",
    "ObjectStatePredictiveRow",
    "ObjectStatePredictiveGateReport",
    "evaluate_objectstate_predictive_gate",
    "validate_objectstate_predictive_gate_summary",
    "validate_objectstate_predictive_gate_thresholds",
    "validate_objectstate_predictive_row",
)
