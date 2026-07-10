from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
)


def test_object_state_controlled_real_gate_cli_writes_fail_summary_and_blocked_rows(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "controlled-real-manifest.json"
    summary_path = tmp_path / "summary.json"
    blocked_rows_path = tmp_path / "reports" / "blocked-rows.md"
    manifest_path.write_text(json.dumps(_identity_only_manifest()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "controlled-real-gate",
                str(manifest_path),
                "--summary-output",
                str(summary_path),
                "--blocked-rows-output",
                str(blocked_rows_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    blocked_rows = blocked_rows_path.read_text(encoding="utf-8")

    assert f"schema={OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA}" in stdout
    assert "gate_status=objectstate_reality_gate_fail" in stdout
    assert "pass_rows=1" in stdout
    assert "blocked_rows=2" in stdout
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA
    assert summary["gate"]["status"] == "objectstate_reality_gate_fail"
    assert summary["gate"]["hard_gates"]["identity_pass_rows_present"] is True
    assert summary["gate"]["hard_gates"]["prediction_pass_rows_present"] is False
    assert summary["gate"]["hard_gates"]["intervention_pass_rows_present"] is False
    assert "missing 6DoF pose tracks" in blocked_rows
    assert "missing action events and counterfactual outcomes" in blocked_rows


def test_object_state_controlled_real_gate_cli_can_require_identity_only_pass(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "controlled-real-manifest.json"
    summary_path = tmp_path / "identity-summary.json"
    manifest_path.write_text(json.dumps(_identity_only_manifest()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "controlled-real-gate",
                str(manifest_path),
                "--identity-only",
                "--require-pass",
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "gate_status=objectstate_reality_gate_pass" in stdout
    assert "hard_blockers=none" in stdout
    assert summary["gate"]["status"] == "objectstate_reality_gate_pass"
    assert summary["gate"]["thresholds"]["require_prediction_pass_row"] is False
    assert summary["gate"]["thresholds"]["require_intervention_pass_row"] is False
    assert summary["blocked_row_count"] == 2


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
