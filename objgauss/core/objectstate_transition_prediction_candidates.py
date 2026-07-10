"""Deprecated compatibility import for transition prediction candidates."""

from objgauss.pipelines.objectstate_transition_prediction_candidates import (
    OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_TRANSITION_PREDICTION_POLICIES,
    objectstate_transition_prediction_candidates,
    objectstate_transition_prediction_candidates_summary,
    validate_objectstate_transition_prediction_candidates_summary,
    write_objectstate_transition_prediction_candidates,
)

__all__ = (
    "OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA",
    "OBJECTSTATE_TRANSITION_PREDICTION_POLICIES",
    "objectstate_transition_prediction_candidates",
    "objectstate_transition_prediction_candidates_summary",
    "write_objectstate_transition_prediction_candidates",
    "validate_objectstate_transition_prediction_candidates_summary",
)
