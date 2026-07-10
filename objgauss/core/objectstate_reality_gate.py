"""Deprecated compatibility import for :mod:`objgauss.evaluation.objectstate_reality_gate`.

New code must import the canonical evaluation module directly.  These explicit
aliases preserve object identity for callers migrating from the historical
``objgauss.core`` path.
"""

from objgauss.evaluation.objectstate_reality_gate import (
    OBJECTSTATE_REALITY_EVIDENCE_KINDS,
    OBJECTSTATE_REALITY_GATE_SCHEMA,
    OBJECTSTATE_REALITY_ROW_SCHEMA,
    OBJECTSTATE_REALITY_ROW_STATUSES,
    OBJECTSTATE_REALITY_SOURCE_KINDS,
    ObjectStateRealityGateReport,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_gate_thresholds,
    validate_objectstate_reality_row,
)

__all__ = (
    "OBJECTSTATE_REALITY_EVIDENCE_KINDS",
    "OBJECTSTATE_REALITY_GATE_SCHEMA",
    "OBJECTSTATE_REALITY_ROW_SCHEMA",
    "OBJECTSTATE_REALITY_ROW_STATUSES",
    "OBJECTSTATE_REALITY_SOURCE_KINDS",
    "ObjectStateRealityGateReport",
    "ObjectStateRealityGateThresholds",
    "ObjectStateRealityRow",
    "evaluate_objectstate_reality_gate",
    "objectstate_reality_blocked_rows_markdown",
    "validate_objectstate_reality_gate_summary",
    "validate_objectstate_reality_gate_thresholds",
    "validate_objectstate_reality_row",
)
