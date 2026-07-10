"""Deprecated compatibility import for the model identity gate."""

from objgauss.evaluation.objectstate_model_identity_gate import (
    OBJECTSTATE_MODEL_IDENTITY_BASELINES,
    OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA,
    ObjectStateModelIdentityGateThresholds,
    objectstate_model_identity_gate_summary,
    validate_objectstate_model_identity_gate_summary,
)

__all__ = (
    "OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA",
    "OBJECTSTATE_MODEL_IDENTITY_BASELINES",
    "ObjectStateModelIdentityGateThresholds",
    "objectstate_model_identity_gate_summary",
    "validate_objectstate_model_identity_gate_summary",
)
