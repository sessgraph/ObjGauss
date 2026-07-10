"""Deprecated compatibility import for the temporal assignment contract."""

from objgauss.pipelines.objectstate_temporal_assignment_contract import (
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY,
    ObjectStateTemporalAssignmentContractThresholds,
    objectstate_temporal_assignment_contract_summary,
    validate_objectstate_temporal_assignment_contract_summary,
)

__all__ = (
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS",
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS",
    "ObjectStateTemporalAssignmentContractThresholds",
    "objectstate_temporal_assignment_contract_summary",
    "validate_objectstate_temporal_assignment_contract_summary",
)
