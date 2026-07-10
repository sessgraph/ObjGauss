"""Deprecated compatibility import for controlled-real readiness evaluation."""

from objgauss.evaluation.objectstate_controlled_real_readiness_audit import (
    CONTROLLED_REAL_READINESS_BLOCK_REASONS,
    OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA,
    objectstate_controlled_real_readiness_audit,
    objectstate_controlled_real_readiness_audit_from_file,
    objectstate_controlled_real_readiness_breakdown_csv,
    objectstate_controlled_real_readiness_markdown,
    validate_objectstate_controlled_real_readiness_audit,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA",
    "CONTROLLED_REAL_READINESS_BLOCK_REASONS",
    "objectstate_controlled_real_readiness_audit_from_file",
    "objectstate_controlled_real_readiness_audit",
    "objectstate_controlled_real_readiness_markdown",
    "objectstate_controlled_real_readiness_breakdown_csv",
    "validate_objectstate_controlled_real_readiness_audit",
)
