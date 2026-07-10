"""Deprecated compatibility import for the ObjectState causal gate."""

from objgauss.evaluation.objectstate_causal_gate import (
    OBJECTSTATE_ACTION_SCHEMA,
    OBJECTSTATE_CAUSAL_ACTIONS,
    OBJECTSTATE_CAUSAL_GATE_SCHEMA,
    ObjectStateAction,
    ObjectStateCausalGateReport,
    ObjectStateCausalGateThresholds,
    ObjectStateCausalRow,
    evaluate_objectstate_causal_gate,
    validate_objectstate_action,
    validate_objectstate_causal_gate_summary,
    validate_objectstate_causal_gate_thresholds,
    validate_objectstate_causal_row,
)

__all__ = (
    "OBJECTSTATE_CAUSAL_GATE_SCHEMA",
    "OBJECTSTATE_ACTION_SCHEMA",
    "OBJECTSTATE_CAUSAL_ACTIONS",
    "ObjectStateAction",
    "ObjectStateCausalGateThresholds",
    "ObjectStateCausalRow",
    "ObjectStateCausalGateReport",
    "evaluate_objectstate_causal_gate",
    "validate_objectstate_causal_gate_summary",
    "validate_objectstate_causal_gate_thresholds",
    "validate_objectstate_action",
    "validate_objectstate_causal_row",
)
