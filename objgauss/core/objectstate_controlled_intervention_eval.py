"""Deprecated compatibility import for the canonical intervention evaluator.

New code must import :mod:`objgauss.evaluation.objectstate_controlled_intervention_eval`.
The explicit aliases below preserve object identity for legacy callers.
"""

from objgauss.evaluation.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
    ObjectStateControlledInterventionThresholds,
    evaluate_objectstate_controlled_intervention_candidates,
    read_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_eval_summary,
    validate_objectstate_controlled_intervention_thresholds,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA",
    "OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA",
    "ObjectStateControlledInterventionThresholds",
    "evaluate_objectstate_controlled_intervention_candidates",
    "read_objectstate_controlled_intervention_candidates",
    "validate_objectstate_controlled_intervention_candidates",
    "validate_objectstate_controlled_intervention_eval_summary",
    "validate_objectstate_controlled_intervention_thresholds",
)
