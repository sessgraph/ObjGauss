"""Deprecated compatibility import for controlled-real identity evaluation."""

from objgauss.evaluation.objectstate_controlled_real_identity_eval import (
    OBJECTSTATE_CONTROLLED_REAL_IDENTITY_BASELINES,
    OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA,
    objectstate_controlled_real_identity_accounting_csv,
    objectstate_controlled_real_identity_artifact_manifest,
    objectstate_controlled_real_identity_eval,
    objectstate_controlled_real_identity_eval_from_files,
    objectstate_controlled_real_identity_pairwise_csv,
    objectstate_controlled_real_identity_report,
    read_objectstate_controlled_real_identity_teacher_evidence,
    validate_objectstate_controlled_real_identity_eval,
    validate_objectstate_controlled_real_identity_teacher_evidence,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA",
    "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA",
    "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_BASELINES",
    "objectstate_controlled_real_identity_eval_from_files",
    "read_objectstate_controlled_real_identity_teacher_evidence",
    "objectstate_controlled_real_identity_eval",
    "objectstate_controlled_real_identity_report",
    "objectstate_controlled_real_identity_accounting_csv",
    "objectstate_controlled_real_identity_pairwise_csv",
    "objectstate_controlled_real_identity_artifact_manifest",
    "validate_objectstate_controlled_real_identity_teacher_evidence",
    "validate_objectstate_controlled_real_identity_eval",
)
