"""Deprecated compatibility import for assignment training orchestration."""

from objgauss.pipelines.objectstate_assignment_train import (
    OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA,
    OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA,
    objectstate_assignment_train_dataset_summary,
    objectstate_assignment_train_smoke,
    validate_objectstate_assignment_train_dataset_summary,
    validate_objectstate_assignment_train_run_summary,
)

__all__ = (
    "OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA",
    "OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA",
    "objectstate_assignment_train_dataset_summary",
    "objectstate_assignment_train_smoke",
    "validate_objectstate_assignment_train_dataset_summary",
    "validate_objectstate_assignment_train_run_summary",
)
