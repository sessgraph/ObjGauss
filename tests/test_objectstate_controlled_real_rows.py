from __future__ import annotations

import json

import pytest

from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    read_objectstate_controlled_real_manifest,
)
from objgauss.evaluation.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    evaluate_controlled_real_manifest_reality_gate,
    objectstate_controlled_real_rows_summary,
    objectstate_reality_rows_from_controlled_real_manifest,
    validate_objectstate_controlled_real_rows_summary,
)
from objgauss.evaluation.objectstate_reality_gate import ObjectStateRealityGateThresholds


def test_controlled_real_manifest_imports_identity_pass_and_blocked_future_rows():
    manifest = _identity_only_manifest()

    rows = objectstate_reality_rows_from_controlled_real_manifest(manifest)
    summary = objectstate_controlled_real_rows_summary(manifest)

    assert len(rows) == 3
    assert rows[0].source_kind == "controlled_real"
    assert rows[0].evidence_kind == "identity"
    assert rows[0].status == "pass"
    assert rows[0].has_identity_gt is True
    assert rows[0].has_timestamp is True
    assert rows[0].metrics["idf1"] == 1.0
    assert {row.status for row in rows[1:]} == {"blocked"}
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA
    assert summary["pass_row_count"] == 1
    assert summary["blocked_row_count"] == 2
    assert summary["gate"]["status"] == "objectstate_reality_gate_fail"
    assert summary["gate"]["hard_gates"]["identity_pass_rows_present"] is True
    assert summary["gate"]["hard_gates"]["prediction_pass_rows_present"] is False
    assert summary["claim_policy"] == {
        "controlled_real_manifest_required": True,
        "importer_does_not_create_ground_truth": True,
        "non_blocked_rows_require_manifest_ground_truth": True,
        "does_not_claim_open_world_generalization": True,
    }
    assert "missing 6DoF pose tracks" in summary["blocked_rows_markdown"]
    assert validate_objectstate_controlled_real_rows_summary(summary) is summary


def test_controlled_real_manifest_can_pass_identity_only_thresholds():
    report = evaluate_controlled_real_manifest_reality_gate(
        _identity_only_manifest(),
        thresholds=ObjectStateRealityGateThresholds(
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )
    payload = report.as_dict()

    assert payload["status"] == "objectstate_reality_gate_pass"
    assert payload["pass_row_count"] == 1
    assert payload["blocked_row_count"] == 2
    assert payload["metrics"]["controlled_real_fragmentation_rate"] == 0.0


def test_controlled_real_manifest_reads_json_file(tmp_path):
    manifest_path = tmp_path / "controlled-real-manifest.json"
    manifest_path.write_text(
        json.dumps(_identity_only_manifest()),
        encoding="utf-8",
    )

    manifest = read_objectstate_controlled_real_manifest(manifest_path)

    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert manifest["sample"]["sample_id"] == "controlled-tabletop-cup-identity-001"


def test_controlled_real_manifest_rejects_non_blocked_identity_without_gt():
    manifest = _identity_only_manifest()
    manifest["ground_truth"]["identity"] = False

    with pytest.raises(ValueError, match="identity reality rows require identity ground truth"):
        objectstate_reality_rows_from_controlled_real_manifest(manifest)


def test_controlled_real_manifest_imports_identity_fail_row_as_fail_not_blocked():
    manifest = _identity_only_manifest()
    manifest["evidence_rows"][0] = {
        "evidence_kind": "identity",
        "status": "fail",
        "metrics": {
            "idf1": 0.45,
            "fragmentation_rate": 0.5,
            "swap_rate": 0.25,
            "identity_collapse": True,
        },
        "failure_reason": "reappeared cup assigned a new identity",
    }

    summary = objectstate_controlled_real_rows_summary(manifest)

    assert summary["fail_row_count"] == 1
    assert summary["pass_row_count"] == 0
    assert summary["gate"]["status"] == "objectstate_reality_gate_fail"
    assert summary["gate"]["hard_gates"]["controlled_real_identity_collapse_absent"] is False
    assert summary["gate"]["hard_gates"]["failed_rows_absent"] is False


def test_controlled_real_summary_does_not_readvertise_forged_identity_pass():
    manifest = _identity_only_manifest()
    manifest["evidence_rows"][0]["metrics"].update(
        {
            "idf1": 0.2,
            "fragmentation_rate": 0.8,
            "swap_rate": 0.1,
            "identity_collapse": True,
        }
    )

    summary = objectstate_controlled_real_rows_summary(manifest)

    assert summary["pass_row_count"] == 0
    assert summary["fail_row_count"] == 1
    assert summary["rows"] == summary["gate"]["rows"]
    assert summary["rows"][0]["status"] == "fail"
    diagnostics = summary["gate"]["declaration_diagnostics"]
    assert diagnostics["caller_status_mismatch_count"] == 1
    assert diagnostics["caller_status_mismatches"][0]["caller_status"] == "pass"


def test_controlled_real_manifest_requires_failed_reason():
    manifest = _identity_only_manifest()
    manifest["evidence_rows"][0] = {
        "evidence_kind": "identity",
        "status": "fail",
        "metrics": {
            "idf1": 0.45,
            "fragmentation_rate": 0.5,
            "swap_rate": 0.25,
            "identity_collapse": True,
        },
    }

    with pytest.raises(ValueError, match="failed controlled real evidence rows require failure_reason"):
        objectstate_reality_rows_from_controlled_real_manifest(manifest)


def _identity_only_manifest():
    return {
        "schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "controlled-tabletop-cup-identity-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "cross_view_occlusion_reappearance",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/controlled-real/cup-identity-001/manifest.json",
                "outputs/controlled-real/cup-identity-001/objectstates.json",
            ],
            "license": "local controlled capture; not public release",
        },
        "ground_truth": {
            "identity": True,
            "pose": False,
            "action": False,
            "timestamp": True,
        },
        "evidence_rows": [
            {
                "evidence_kind": "identity",
                "status": "pass",
                "metrics": {
                    "idf1": 1.0,
                    "fragmentation_rate": 0.0,
                    "swap_rate": 0.0,
                    "identity_collapse": False,
                    "raw_prediction_observations": True,
                },
            },
            {
                "evidence_kind": "prediction",
                "status": "blocked",
                "metrics": {},
                "block_reason": "missing 6DoF pose tracks for future-pose targets",
            },
            {
                "evidence_kind": "intervention",
                "status": "blocked",
                "metrics": {},
                "block_reason": "missing action events and counterfactual outcomes",
            },
        ],
    }
