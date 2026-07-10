"""Deprecated compatibility import for teacher-evidence leakage audit."""

from objgauss.evaluation.objectstate_teacher_evidence_leakage_audit import (
    OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA,
    TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS,
    TeacherEvidenceLeakageAuditThresholds,
    objectstate_teacher_evidence_leakage_audit_summary,
    teacher_evidence_scenarios_from_audit,
    validate_objectstate_teacher_evidence_leakage_audit_summary,
)

__all__ = (
    "OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA",
    "TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS",
    "TeacherEvidenceLeakageAuditThresholds",
    "objectstate_teacher_evidence_leakage_audit_summary",
    "teacher_evidence_scenarios_from_audit",
    "validate_objectstate_teacher_evidence_leakage_audit_summary",
)
