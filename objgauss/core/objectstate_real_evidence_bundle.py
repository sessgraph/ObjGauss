"""Deprecated compatibility import for the real-evidence dataset contract."""

from objgauss.datasets.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA,
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA,
    OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA,
    OBJECTSTATE_REAL_GATE_ACCOUNTING_STATUSES,
    OBJECTSTATE_REAL_GATE_EVIDENCE_KINDS,
    OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA,
    OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA,
    OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA,
    OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA,
    objectstate_real_evidence_bundle_summary,
    read_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle_summary,
)

__all__ = (
    "OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA",
    "OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA",
    "OBJECTSTATE_REAL_OBSERVATION_ROW_SCHEMA",
    "OBJECTSTATE_REAL_OBJECT_POSE_ROW_SCHEMA",
    "OBJECTSTATE_REAL_IDENTITY_LINK_ROW_SCHEMA",
    "OBJECTSTATE_REAL_ACTION_INTERVAL_ROW_SCHEMA",
    "OBJECTSTATE_REAL_STATE_TRANSITION_ROW_SCHEMA",
    "OBJECTSTATE_REAL_GATE_ACCOUNTING_ROW_SCHEMA",
    "OBJECTSTATE_REAL_GATE_EVIDENCE_KINDS",
    "OBJECTSTATE_REAL_GATE_ACCOUNTING_STATUSES",
    "read_objectstate_real_evidence_bundle",
    "objectstate_real_evidence_bundle_summary",
    "validate_objectstate_real_evidence_bundle",
    "validate_objectstate_real_evidence_bundle_summary",
)
