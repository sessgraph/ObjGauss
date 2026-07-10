"""Deprecated compatibility import for the controlled real-evidence adapter."""

from objgauss.datasets.objectstate_controlled_real_evidence_bundle import (
    OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA,
    objectstate_controlled_real_evidence_bundle_adapter_summary,
    objectstate_controlled_real_evidence_bundle_adapter_summary_from_file,
    objectstate_controlled_real_evidence_bundle_from_capture_manifest,
    read_objectstate_controlled_real_evidence_bundle_adapter_summary,
    validate_objectstate_controlled_real_evidence_bundle_adapter_summary,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA",
    "read_objectstate_controlled_real_evidence_bundle_adapter_summary",
    "objectstate_controlled_real_evidence_bundle_adapter_summary_from_file",
    "objectstate_controlled_real_evidence_bundle_adapter_summary",
    "objectstate_controlled_real_evidence_bundle_from_capture_manifest",
    "validate_objectstate_controlled_real_evidence_bundle_adapter_summary",
)
