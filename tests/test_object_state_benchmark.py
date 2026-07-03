from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core import (
    OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA,
    object_state_stability_benchmark as core_object_state_stability_benchmark,
    validate_object_state_stability_benchmark as core_validate_object_state_stability_benchmark,
    write_object_state_stability_benchmark as core_write_object_state_stability_benchmark,
)
from objgauss.core.object_state_benchmark import (
    object_state_stability_benchmark,
    validate_object_state_stability_benchmark,
    write_object_state_stability_benchmark,
)


def test_object_state_stability_benchmark_covers_expected_failure_modes():
    report = object_state_stability_benchmark()

    assert report["schema"] == "objgauss-object-state-stability-benchmark-v1"
    assert report["status"] == "pass"
    assert report["aggregate"]["case_count"] == 8
    assert report["aggregate"]["warn_count"] == 0
    assert report["aggregate"]["observed_warn_count"] == 6
    assert {
        "uniform_assignment",
        "slot_collapse",
        "soft_assignment_noise",
        "slot_permutation",
        "temporal_jitter",
        "birth_unmatched",
        "duplicate_fragment",
    }.issubset(set(report["aggregate"]["failure_mode_coverage"]))
    assert validate_object_state_stability_benchmark(report, strict=True) is True

    cases = {case["name"]: case for case in report["cases"]}
    assert set(cases) == {
        "clean_sparse",
        "uniform_mixed",
        "single_slot_collapse",
        "soft_noise",
        "slot_permutation",
        "temporal_jitter",
        "birth_unmatched",
        "duplicate_fragment",
    }

    assert cases["clean_sparse"]["observed_status"] == "pass"
    assert cases["uniform_mixed"]["observed_status"] == "warn"
    assert cases["uniform_mixed"]["stability"]["diagnostics"] == [
        "low_assignment_confidence",
        "mixed_slots",
        "low_object_purity",
    ]
    assert cases["single_slot_collapse"]["stability"]["diagnostics"] == [
        "low_confidence_slots",
        "slot_collapse:0",
    ]
    assert cases["soft_noise"]["metrics"]["label_fragmentation"] > 0.4

    permutation = cases["slot_permutation"]
    assert permutation["metrics"]["raw_assignment_jitter"] == pytest.approx(1.0)
    assert permutation["metrics"]["max_temporal_drift"] == pytest.approx(0.0)
    assert permutation["metrics"]["slot_permutation_resolved"] is True
    assert {(match["previous_id"], match["current_id"]) for match in permutation["temporal"]["matches"]} == {
        (0, 1),
        (1, 0),
    }

    assert cases["temporal_jitter"]["temporal"]["diagnostics"] == ["high_temporal_drift"]
    assert cases["temporal_jitter"]["metrics"]["max_temporal_drift"] >= 0.03
    assert "birth_unmatched" in cases["birth_unmatched"]["dynamic_k"]["proposal_kinds"]
    assert "merge_duplicate" in cases["duplicate_fragment"]["dynamic_k"]["proposal_kinds"]


def test_core_namespace_exports_object_state_benchmark(tmp_path):
    report = core_object_state_stability_benchmark()

    assert OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA == report["schema"]
    assert core_validate_object_state_stability_benchmark(report, strict=True) is True

    output = tmp_path / "core-object-state-benchmark.json"
    written = core_write_object_state_stability_benchmark(output, strict=True)

    assert written["status"] == "pass"
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == report["schema"]


def test_object_state_stability_benchmark_strict_rejects_failed_suite():
    report = object_state_stability_benchmark()
    failed = {**report, "status": "warn"}

    assert validate_object_state_stability_benchmark(failed) is True
    with pytest.raises(ValueError, match="strict gate failed"):
        validate_object_state_stability_benchmark(failed, strict=True)


def test_object_state_stability_benchmark_cli_writes_report(tmp_path, capsys):
    output = tmp_path / "object-state-benchmark.json"

    assert (
        main(
            [
                "object-state",
                "stability-benchmark",
                "--output",
                str(output),
                "--strict",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert "schema=objgauss-object-state-stability-benchmark-v1" in stdout
    assert "status=pass" in stdout
    assert "cases=8" in stdout
    assert payload["status"] == "pass"
    assert payload["aggregate"]["observed_warn_count"] == 6


def test_write_object_state_stability_benchmark_respects_custom_report_id(tmp_path):
    output = tmp_path / "named-benchmark.json"

    report = write_object_state_stability_benchmark(
        output,
        report_id="pretrain-objectstate-gate",
        strict=True,
    )

    assert report["report_id"] == "pretrain-objectstate-gate"
    assert json.loads(output.read_text(encoding="utf-8"))["report_id"] == "pretrain-objectstate-gate"
