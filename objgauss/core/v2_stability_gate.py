"""Deprecated compatibility import for the v2 synthetic stability gate."""

from objgauss.evaluation.v2_stability_gate import (
    V2_STABILITY_GATE_HARD_CHECKS,
    V2_STABILITY_GATE_SCHEMA,
    V2_STABILITY_GATE_SUITE_SCHEMA,
    SyntheticStabilityGateReport,
    SyntheticStabilitySuiteGateReport,
    evaluate_synthetic_stability_gate,
    evaluate_synthetic_stability_suite_gate,
    validate_synthetic_stability_gate_summary,
    validate_synthetic_stability_suite_gate_summary,
)

__all__ = (
    "V2_STABILITY_GATE_SCHEMA",
    "V2_STABILITY_GATE_SUITE_SCHEMA",
    "V2_STABILITY_GATE_HARD_CHECKS",
    "SyntheticStabilityGateReport",
    "SyntheticStabilitySuiteGateReport",
    "evaluate_synthetic_stability_gate",
    "evaluate_synthetic_stability_suite_gate",
    "validate_synthetic_stability_gate_summary",
    "validate_synthetic_stability_suite_gate_summary",
)
