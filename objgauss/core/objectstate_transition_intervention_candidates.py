"""Deprecated compatibility import for transition intervention candidates."""

from objgauss.pipelines.objectstate_transition_intervention_candidates import (
    OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_TRANSITION_INTERVENTION_POLICIES,
    objectstate_transition_intervention_candidates,
    objectstate_transition_intervention_candidates_summary,
    validate_objectstate_transition_intervention_candidates_summary,
    write_objectstate_transition_intervention_candidates,
)

__all__ = (
    "OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA",
    "OBJECTSTATE_TRANSITION_INTERVENTION_POLICIES",
    "objectstate_transition_intervention_candidates",
    "objectstate_transition_intervention_candidates_summary",
    "write_objectstate_transition_intervention_candidates",
    "validate_objectstate_transition_intervention_candidates_summary",
)
