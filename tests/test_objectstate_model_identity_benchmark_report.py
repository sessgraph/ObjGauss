from __future__ import annotations

import csv
import json

from objgauss.core.objectstate_model_identity_benchmark_report import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES,
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA,
    validate_objectstate_model_identity_benchmark_report_summary,
    write_objectstate_model_identity_benchmark_report,
)
from objgauss.core.objectstate_model_identity_gate import OBJECTSTATE_MODEL_IDENTITY_BASELINES


def test_model_identity_benchmark_report_writes_auditable_outputs(tmp_path):
    artifact_dir = tmp_path / "identity-benchmark-artifacts"
    summary = write_objectstate_model_identity_benchmark_report(
        tmp_path,
        artifact_dir=artifact_dir,
        sample_id="identity-benchmark-report-test",
        seed=3,
    )

    assert summary["schema"] == OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA
    assert summary["status"] == "objectstate_model_identity_benchmark_report_candidate_ready"
    assert summary["scenario_count"] == 15
    assert summary["difficulty_levels"] == list(
        OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES
    )
    assert {row["baseline"] for row in summary["overall_ranking"]} == set(
        OBJECTSTATE_MODEL_IDENTITY_BASELINES
    )
    assert summary["long_training_gate"]["status"] == "candidate_ready"
    assert validate_objectstate_model_identity_benchmark_report_summary(summary) == summary

    refs = summary["artifact_refs"]
    report_path = tmp_path / "identity-benchmark-report.md"
    csv_path = tmp_path / "identity-benchmark-breakdown.csv"
    summary_path = tmp_path / "identity-benchmark-summary.json"
    assert refs["identity_benchmark_report"] == str(report_path)
    assert refs["identity_benchmark_breakdown"] == str(csv_path)
    assert refs["identity_benchmark_summary"] == str(summary_path)
    assert refs["identity_benchmark_artifacts"] == str(artifact_dir)
    assert report_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()
    assert (artifact_dir / "identity-benchmark-summary.json").exists()

    markdown = report_path.read_text(encoding="utf-8")
    assert "# ObjectState Identity Benchmark Report" in markdown
    assert "## Overall Ranking" in markdown
    assert "## Perturbation Breakdown" in markdown
    assert "## Difficulty Ladder" in markdown
    assert "Decision: `candidate_ready`" in markdown

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 15 * len(OBJECTSTATE_MODEL_IDENTITY_BASELINES)
    assert {row["difficulty"] for row in rows} == set(
        OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES
    )
    assert {row["perturbation_kind"] for row in rows} == {
        "viewpoint",
        "dropout",
        "occlusion",
        "appearance",
        "spatial",
    }
    assert json.loads(summary_path.read_text(encoding="utf-8"))["schema"] == (
        OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA
    )
