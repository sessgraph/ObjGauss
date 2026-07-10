"""Deprecated compatibility import for the transition reality handoff."""

from objgauss.pipelines.objectstate_transition_reality_handoff import (
    OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA,
    objectstate_transition_reality_handoff,
    validate_objectstate_transition_reality_handoff_summary,
    write_objectstate_transition_reality_handoff,
)

__all__ = (
    "OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA",
    "objectstate_transition_reality_handoff",
    "write_objectstate_transition_reality_handoff",
    "validate_objectstate_transition_reality_handoff_summary",
)
