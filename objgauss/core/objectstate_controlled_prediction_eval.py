"""Deprecated compatibility import for the canonical prediction evaluator.

New code must import :mod:`objgauss.evaluation.objectstate_controlled_prediction_eval`.
The explicit aliases below preserve object identity for legacy callers.
"""

from objgauss.evaluation.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
    ObjectStateControlledPredictionThresholds,
    evaluate_objectstate_controlled_prediction_candidates,
    read_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_eval_summary,
    validate_objectstate_controlled_prediction_thresholds,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA",
    "OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA",
    "ObjectStateControlledPredictionThresholds",
    "evaluate_objectstate_controlled_prediction_candidates",
    "read_objectstate_controlled_prediction_candidates",
    "validate_objectstate_controlled_prediction_candidates",
    "validate_objectstate_controlled_prediction_eval_summary",
    "validate_objectstate_controlled_prediction_thresholds",
)
