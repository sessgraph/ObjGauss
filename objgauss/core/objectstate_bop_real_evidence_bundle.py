"""Deprecated compatibility import for BOP real-evidence bundle orchestration."""

from objgauss.pipelines.objectstate_bop_real_evidence_bundle import (
    OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA,
    objectstate_bop_real_evidence_bundle_adapter_summary,
    objectstate_bop_real_evidence_bundle_adapter_summary_from_files,
    objectstate_bop_real_evidence_bundle_from_summaries,
    read_objectstate_bop_real_evidence_bundle_adapter_summary,
    validate_objectstate_bop_real_evidence_bundle_adapter_summary,
)

__all__ = (
    "OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA",
    "read_objectstate_bop_real_evidence_bundle_adapter_summary",
    "objectstate_bop_real_evidence_bundle_adapter_summary_from_files",
    "objectstate_bop_real_evidence_bundle_adapter_summary",
    "objectstate_bop_real_evidence_bundle_from_summaries",
    "validate_objectstate_bop_real_evidence_bundle_adapter_summary",
)
