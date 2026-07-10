"""Deprecated compatibility import for the ObjectState identity gate."""

from objgauss.evaluation.objectstate_identity_gate import (
    OBJECTSTATE_IDENTITY_DATASET_SCHEMA,
    OBJECTSTATE_IDENTITY_GATE_SCHEMA,
    ObjectStateIdentityGateReport,
    ObjectStateIdentityGateThresholds,
    ObjectStateIdentityRow,
    evaluate_objectstate_identity_gate,
    validate_objectstate_identity_gate_summary,
    validate_objectstate_identity_gate_thresholds,
    validate_objectstate_identity_row,
)

__all__ = (
    "OBJECTSTATE_IDENTITY_GATE_SCHEMA",
    "OBJECTSTATE_IDENTITY_DATASET_SCHEMA",
    "ObjectStateIdentityGateThresholds",
    "ObjectStateIdentityRow",
    "ObjectStateIdentityGateReport",
    "evaluate_objectstate_identity_gate",
    "validate_objectstate_identity_gate_summary",
    "validate_objectstate_identity_gate_thresholds",
    "validate_objectstate_identity_row",
)
