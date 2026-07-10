"""Deprecated compatibility import for the canonical identity evaluator.

New code must import :mod:`objgauss.evaluation.objectstate_controlled_identity_eval`.
The explicit aliases below preserve object identity for legacy callers.
"""

from objgauss.evaluation.objectstate_controlled_identity_eval import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
    ObjectStateControlledIdentityThresholds,
    evaluate_objectstate_controlled_identity_predictions,
    read_objectstate_controlled_identity_predictions,
    validate_objectstate_controlled_identity_eval_summary,
    validate_objectstate_controlled_identity_predictions,
    validate_objectstate_controlled_identity_thresholds,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA",
    "OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA",
    "ObjectStateControlledIdentityThresholds",
    "evaluate_objectstate_controlled_identity_predictions",
    "read_objectstate_controlled_identity_predictions",
    "validate_objectstate_controlled_identity_eval_summary",
    "validate_objectstate_controlled_identity_predictions",
    "validate_objectstate_controlled_identity_thresholds",
)
