from __future__ import annotations

import json
from pathlib import Path

import pytest

from objgauss.pipelines.trainable_quality import (
    TRAINABLE_QUALITY_REPORT_SCHEMA,
    trainable_quality_report,
    validate_trainable_quality_report,
    write_trainable_quality_report,
)


def test_trainable_quality_report_matches_debug_fixture_metrics(tmp_path):
    artifact = json.loads(
        Path("public/models/trainable-mvp-debug/model-artifact.json").read_text(encoding="utf-8")
    )

    report = trainable_quality_report(
        artifact,
        report_id="trainable-debug-fixture-quality",
        source={"type": "fixture"},
    )

    assert report["schema"] == TRAINABLE_QUALITY_REPORT_SCHEMA
    assert report["status"] == "warn"
    assert report["metrics"]["assignment_entropy"] == pytest.approx(0.68875)
    assert report["metrics"]["temporal_drift"] == pytest.approx(0.017205)
    assert report["metrics"]["assignment_jitter"] == pytest.approx(0.0225)
    assert report["metrics"]["slot_utilization"] == 1.0
    assert report["metrics"]["object_purity"] == pytest.approx(0.79625)
    assert {gate["name"]: gate["status"] for gate in report["gates"]} == {
        "slot_utilization": "pass",
        "assignment_entropy": "warn",
        "temporal_drift": "pass",
    }
    assert validate_trainable_quality_report(report) is True

    output = tmp_path / "quality-report.json"
    written = write_trainable_quality_report(output, artifact)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert written["schema"] == TRAINABLE_QUALITY_REPORT_SCHEMA
    assert loaded["schema"] == TRAINABLE_QUALITY_REPORT_SCHEMA
    assert loaded["metrics"]["bbox_stability"] == pytest.approx(0.960402)
