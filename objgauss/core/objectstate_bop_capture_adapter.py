"""Deprecated compatibility import for the BOP capture dataset adapter."""

from objgauss.datasets.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICIES,
    BOP_IDENTITY_POLICY_POSE_TRACK_PER_OBJ_ID,
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA,
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    objectstate_bop_capture_adapter_summary,
    objectstate_bop_capture_condition_sidecar_summary,
    objectstate_bop_capture_manifest_from_scene,
    validate_objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_adapter_summary,
    validate_objectstate_bop_capture_condition_sidecar,
    validate_objectstate_bop_capture_condition_sidecar_summary,
)

__all__ = (
    "BOP_IDENTITY_POLICIES",
    "BOP_IDENTITY_POLICY_POSE_TRACK_PER_OBJ_ID",
    "BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID",
    "DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M",
    "OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA",
    "OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA",
    "OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA",
    "OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA",
    "objectstate_bop_capture_acceptance_summary",
    "objectstate_bop_capture_adapter_summary",
    "objectstate_bop_capture_condition_sidecar_summary",
    "objectstate_bop_capture_manifest_from_scene",
    "validate_objectstate_bop_capture_acceptance_summary",
    "validate_objectstate_bop_capture_adapter_summary",
    "validate_objectstate_bop_capture_condition_sidecar",
    "validate_objectstate_bop_capture_condition_sidecar_summary",
)
