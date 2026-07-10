"""Deprecated compatibility import for core-model validation orchestration."""

from objgauss.pipelines.core_model_validation import (
    CORE_MODEL_TRAIN_VALIDATE_SCHEMA,
    CoreModelTrainValidateReport,
    core_model_train_validate_report,
    validate_core_model_train_validate_summary,
)

__all__ = (
    "CORE_MODEL_TRAIN_VALIDATE_SCHEMA",
    "CoreModelTrainValidateReport",
    "core_model_train_validate_report",
    "validate_core_model_train_validate_summary",
)
