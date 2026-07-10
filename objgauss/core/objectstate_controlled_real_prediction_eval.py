"""Deprecated compatibility import for controlled-real prediction evaluation."""

from objgauss.evaluation.objectstate_controlled_real_prediction_eval import (
    OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA,
    objectstate_controlled_real_prediction_accounting_csv,
    objectstate_controlled_real_prediction_artifact_manifest,
    objectstate_controlled_real_prediction_errors_csv,
    objectstate_controlled_real_prediction_eval,
    objectstate_controlled_real_prediction_eval_from_files,
    objectstate_controlled_real_prediction_report,
    read_objectstate_controlled_real_identity_eval,
    validate_objectstate_controlled_real_prediction_eval,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA",
    "objectstate_controlled_real_prediction_eval_from_files",
    "read_objectstate_controlled_real_identity_eval",
    "objectstate_controlled_real_prediction_eval",
    "objectstate_controlled_real_prediction_report",
    "objectstate_controlled_real_prediction_accounting_csv",
    "objectstate_controlled_real_prediction_errors_csv",
    "objectstate_controlled_real_prediction_artifact_manifest",
    "validate_objectstate_controlled_real_prediction_eval",
)
