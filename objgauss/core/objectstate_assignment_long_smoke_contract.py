"""Deprecated compatibility import for the assignment smoke contract."""

from objgauss.pipelines.objectstate_assignment_long_smoke_contract import (
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SCHEMA,
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA,
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_REQUIRED_POLICY,
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SUCCESS_CRITERIA,
    ObjectStateAssignmentLongSmokeContractThresholds,
    objectstate_assignment_long_smoke_contract_summary,
    validate_objectstate_assignment_long_smoke_contract_summary,
)

__all__ = (
    "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SCHEMA",
    "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA",
    "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_REQUIRED_POLICY",
    "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SUCCESS_CRITERIA",
    "ObjectStateAssignmentLongSmokeContractThresholds",
    "objectstate_assignment_long_smoke_contract_summary",
    "validate_objectstate_assignment_long_smoke_contract_summary",
)
