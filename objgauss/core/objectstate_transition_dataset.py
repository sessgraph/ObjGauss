"""Deprecated compatibility import for the ObjectState transition dataset."""

from objgauss.datasets.objectstate_transition_dataset import (
    OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA,
    OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
    OBJECTSTATE_TRANSITION_ROW_SCHEMA,
    objectstate_transition_dataset_audit,
    objectstate_transition_dataset_audit_from_path,
    objectstate_transition_dataset_from_capture_manifest,
    read_objectstate_transition_dataset,
    validate_objectstate_transition_dataset,
    validate_objectstate_transition_dataset_audit,
    write_objectstate_transition_dataset,
)

__all__ = (
    "OBJECTSTATE_TRANSITION_DATASET_SCHEMA",
    "OBJECTSTATE_TRANSITION_ROW_SCHEMA",
    "OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA",
    "objectstate_transition_dataset_from_capture_manifest",
    "write_objectstate_transition_dataset",
    "read_objectstate_transition_dataset",
    "objectstate_transition_dataset_audit_from_path",
    "objectstate_transition_dataset_audit",
    "validate_objectstate_transition_dataset",
    "validate_objectstate_transition_dataset_audit",
)
