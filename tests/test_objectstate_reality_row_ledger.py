from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_real_rows import (
    objectstate_controlled_real_rows_summary,
)
from objgauss.core.objectstate_reality_row_ledger import (
    OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA,
    objectstate_reality_row_ledger,
    validate_objectstate_reality_row_ledger_summary,
)


def test_reality_row_ledger_aggregates_existing_row_summaries(tmp_path):
    summary_a = objectstate_controlled_real_rows_summary(
        _controlled_manifest("controlled-a", prediction_status="pass")
    )
    summary_b = objectstate_controlled_real_rows_summary(
        _controlled_manifest("controlled-b", prediction_status="fail")
    )
    path_a = tmp_path / "controlled-a-rows.json"
    path_b = tmp_path / "controlled-b-rows.json"
    _write_json(path_a, summary_a)
    _write_json(path_b, summary_b)

    ledger = objectstate_reality_row_ledger((path_a, path_b))

    assert ledger["schema"] == OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA
    assert validate_objectstate_reality_row_ledger_summary(ledger) == ledger
    assert ledger["status"] == "objectstate_reality_row_ledger_reviewable"
    assert ledger["summary_count"] == 2
    assert ledger["row_count"] == 6
    assert ledger["pass_row_count"] == 1
    assert ledger["fail_row_count"] == 3
    assert ledger["blocked_row_count"] == 2
    assert ledger["sample_scope"]["sample_ids"] == ["controlled-a", "controlled-b"]
    assert ledger["gate"]["status"] == "objectstate_reality_gate_fail"
    assert ledger["gap_summary"]["present_pass_evidence_kinds"] == ["prediction"]
    assert ledger["gap_summary"]["missing_pass_evidence_kinds"] == [
        "identity",
        "intervention",
    ]
    experiment_status = {
        row["experiment"]: row["status"]
        for row in ledger["state_variable_evidence_matrix"]
    }
    assert experiment_status == {
        "identity_persistence": "objectstate_state_variable_experiment_fail",
        "occlusion_recovery": "objectstate_state_variable_experiment_missing_metric",
        "view_invariance": "objectstate_state_variable_experiment_missing_metric",
        "predictive_sufficiency": "objectstate_state_variable_experiment_pass",
        "counterfactual_action_interface": (
            "objectstate_state_variable_experiment_blocked"
        ),
    }
    occlusion_row = next(
        row
        for row in ledger["state_variable_evidence_matrix"]
        if row["experiment"] == "occlusion_recovery"
    )
    assert occlusion_row["missing_metrics"] == ["occlusion_recovery_rate"]
    assert (
        occlusion_row["challenge_status"]
        == "objectstate_state_variable_challenge_unknown"
    )
    view_row = next(
        row
        for row in ledger["state_variable_evidence_matrix"]
        if row["experiment"] == "view_invariance"
    )
    assert (
        view_row["challenge_status"]
        == "objectstate_state_variable_challenge_unknown"
    )
    intervention_row = next(
        row
        for row in ledger["state_variable_evidence_matrix"]
        if row["experiment"] == "counterfactual_action_interface"
    )
    assert (
        intervention_row["challenge_status"]
        == "objectstate_state_variable_challenge_unknown"
    )
    assert ledger["state_variable_evidence_matrix_markdown"].startswith(
        "# ObjectState State-Variable Evidence Matrix"
    )
    assert [action["evidence_kind"] for action in ledger["next_actions"]] == [
        "identity",
        "intervention",
    ]
    assert [action["priority"] for action in ledger["next_actions"]] == ["p0", "p0"]
    assert (
        ledger["next_actions"][0]["recommended_route"]
        == "controlled_real_identity_handoff"
    )
    assert (
        "controlled-identity-bundle-handoff"
        in ledger["next_actions"][0]["commands"][-1]
    )
    assert (
        ledger["next_actions"][1]["recommended_route"]
        == "controlled_reality_bundle_handoff"
    )
    assert (
        "controlled-reality-bundle-handoff"
        in ledger["next_actions"][1]["commands"][-1]
    )
    assert ledger["next_actions_markdown"].startswith(
        "# ObjectState Reality Row Ledger Next Actions"
    )
    assert ledger["claim_policy"]["does_not_claim_world_model"] is True
    assert "full ObjectState reality gate did not pass" in ledger["issues"][-1]


def test_reality_row_ledger_cli_writes_summary_and_blocked_rows(tmp_path, capsys):
    summary_a = objectstate_controlled_real_rows_summary(
        _controlled_manifest("controlled-a", prediction_status="pass")
    )
    summary_b = objectstate_controlled_real_rows_summary(
        _controlled_manifest("controlled-b", prediction_status="fail")
    )
    path_a = tmp_path / "controlled-a-rows.json"
    path_b = tmp_path / "controlled-b-rows.json"
    summary_path = tmp_path / "reality-row-ledger.json"
    blocked_path = tmp_path / "reality-row-ledger-blocked.md"
    matrix_path = tmp_path / "reality-row-ledger-experiment-matrix.md"
    actions_path = tmp_path / "reality-row-ledger-next-actions.md"
    _write_json(path_a, summary_a)
    _write_json(path_b, summary_b)

    assert (
        main(
            [
                "object-state",
                "audit-reality-row-ledger",
                str(path_a),
                str(path_b),
                "--summary-output",
                str(summary_path),
                "--blocked-rows-output",
                str(blocked_path),
                "--experiment-matrix-output",
                str(matrix_path),
                "--next-actions-output",
                str(actions_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    written_summary = _read_json(summary_path)

    assert f"schema={OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA}" in stdout
    assert "gate_status=objectstate_reality_gate_fail" in stdout
    assert "missing_pass_evidence_kinds=identity,intervention" in stdout
    assert (
        "experiment=identity_persistence:"
        "objectstate_state_variable_experiment_fail:"
        "objectstate_state_variable_challenge_not_required"
    ) in stdout
    assert (
        "experiment=predictive_sufficiency:"
        "objectstate_state_variable_experiment_pass:"
        "objectstate_state_variable_challenge_not_required"
    ) in stdout
    assert (
        "next_action=identity:pass_evidence_missing:"
        "controlled_real_identity_handoff"
    ) in stdout
    assert (
        "next_action=intervention:pass_evidence_missing:"
        "controlled_reality_bundle_handoff"
    ) in stdout
    assert validate_objectstate_reality_row_ledger_summary(written_summary) == (
        written_summary
    )
    assert blocked_path.read_text(encoding="utf-8").startswith("| row_id |")
    assert matrix_path.read_text(encoding="utf-8").startswith(
        "# ObjectState State-Variable Evidence Matrix"
    )
    assert actions_path.read_text(encoding="utf-8").startswith(
        "# ObjectState Reality Row Ledger Next Actions"
    )


def _controlled_manifest(sample_id: str, *, prediction_status: str):
    prediction_metrics = {
        "state_ade": 0.01 if prediction_status == "pass" else 0.30,
        "history_ade": 0.02,
        "prediction_gap_vs_history_model": -0.01
        if prediction_status == "pass"
        else 0.28,
    }
    prediction_row = {
        "evidence_kind": "prediction",
        "status": prediction_status,
        "metrics": prediction_metrics,
        "artifact_refs": [f"outputs/controlled-real/{sample_id}/prediction.json"],
    }
    if prediction_status == "fail":
        prediction_row["failure_reason"] = "prediction baseline missed future pose"
    return {
        "schema": "objgauss-objectstate-controlled-real-manifest-v1",
        "sample": {
            "sample_id": sample_id,
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "tabletop_push",
            "observation_modalities": ["rgb", "gaussian", "pose"],
            "artifact_refs": [f"outputs/controlled-real/{sample_id}/capture.json"],
            "license": "local controlled capture; not for redistribution",
        },
        "ground_truth": {
            "identity": True,
            "pose": True,
            "action": False,
            "timestamp": True,
        },
        "evidence_rows": [
            {
                "evidence_kind": "identity",
                "status": "fail",
                "metrics": {
                    "idf1": 1.0,
                    "fragmentation_rate": 0.0,
                    "swap_rate": 0.0,
                    "identity_collapse": True,
                },
                "failure_reason": "baseline collapsed physical identities",
                "artifact_refs": [f"outputs/controlled-real/{sample_id}/identity.json"],
            },
            prediction_row,
            {
                "evidence_kind": "intervention",
                "status": "blocked",
                "metrics": {},
                "block_reason": "missing action-conditioned intervention candidates",
                "artifact_refs": [
                    f"outputs/controlled-real/{sample_id}/intervention.json"
                ],
            },
        ],
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
