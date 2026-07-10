"""Deprecated compatibility import for the teacher-evidence dataset contract."""

from objgauss.datasets.objectstate_teacher_evidence import (
    OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA,
    OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA,
    OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA,
    TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS,
    TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES,
    TEACHER_EVIDENCE_LEAKAGE_RISK_LEVELS,
    TEACHER_EVIDENCE_REQUIRED_PROVENANCE_KEYS,
    TEACHER_EVIDENCE_SOURCES,
    TEACHER_EVIDENCE_TRAINING_RISK_LEVELS,
    TeacherEvidenceBatch,
    objectstate_teacher_evidence_contract_summary,
    teacher_evidence_batch_summary,
    validate_objectstate_teacher_evidence_contract_summary,
    validate_teacher_evidence_batch,
    validate_teacher_evidence_batch_summary,
)

__all__ = (
    "OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA",
    "OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA",
    "OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA",
    "TEACHER_EVIDENCE_SOURCES",
    "TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES",
    "TEACHER_EVIDENCE_LEAKAGE_RISK_LEVELS",
    "TEACHER_EVIDENCE_TRAINING_RISK_LEVELS",
    "TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS",
    "TEACHER_EVIDENCE_REQUIRED_PROVENANCE_KEYS",
    "TeacherEvidenceBatch",
    "validate_teacher_evidence_batch",
    "teacher_evidence_batch_summary",
    "validate_teacher_evidence_batch_summary",
    "objectstate_teacher_evidence_contract_summary",
    "validate_objectstate_teacher_evidence_contract_summary",
)
