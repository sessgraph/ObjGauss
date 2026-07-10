"""Deprecated compatibility import for v2 stability diagnostics."""

from objgauss.evaluation.v2_stability_diagnostics import (
    V2_STABILITY_DIAGNOSTICS_SCHEMA,
    V2_STABILITY_FAILURE_MODES,
    FailureModeClassifier,
    FailureModeEvent,
    IdentitySlotObservation,
    SyntheticStabilityDiagnosticsReport,
    diagnose_synthetic_stability_fixture,
    expected_slots_for_synthetic_fixture,
    validate_synthetic_stability_diagnostics_summary,
)

__all__ = (
    "V2_STABILITY_DIAGNOSTICS_SCHEMA",
    "V2_STABILITY_FAILURE_MODES",
    "IdentitySlotObservation",
    "FailureModeEvent",
    "FailureModeClassifier",
    "SyntheticStabilityDiagnosticsReport",
    "diagnose_synthetic_stability_fixture",
    "expected_slots_for_synthetic_fixture",
    "validate_synthetic_stability_diagnostics_summary",
)
