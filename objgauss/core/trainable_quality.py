"""Deprecated compatibility import for trainable artifact quality reporting."""

from objgauss.pipelines.trainable_quality import (
    TRAINABLE_QUALITY_REPORT_SCHEMA,
    trainable_quality_report,
    validate_trainable_quality_report,
    write_trainable_quality_report,
)

__all__ = (
    "TRAINABLE_QUALITY_REPORT_SCHEMA",
    "trainable_quality_report",
    "write_trainable_quality_report",
    "validate_trainable_quality_report",
)
