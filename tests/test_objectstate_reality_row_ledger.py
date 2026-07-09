from __future__ import annotations

import json
from pathlib import Path

from objgauss.cli import main
from objgauss.core.objectstate_controlled_real_rows import (
    objectstate_controlled_real_rows_summary,
)
from objgauss.core.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
)
from objgauss.core.objectstate_real_evidence_bundle_ledger import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA,
    validate_objectstate_real_evidence_bundle_ledger_summary,
    write_objectstate_real_evidence_bundle_ledger,
)
from objgauss.core.objectstate_real_evidence_bundle_ledger_audit import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_PACKAGE_AUDIT_SCHEMA,
    objectstate_real_evidence_bundle_ledger_package_audit,
    validate_objectstate_real_evidence_bundle_ledger_package_audit,
)
from objgauss.core.objectstate_real_identity_rows import (
    objectstate_real_identity_rows_summary,
)
from objgauss.core.objectstate_real_intervention_rows import (
    objectstate_real_intervention_rows_summary,
)
from objgauss.core.objectstate_real_prediction_rows import (
    objectstate_real_prediction_rows_summary,
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


def test_reality_row_ledger_accepts_real_evidence_accounting_summaries(tmp_path):
    bundle = _real_evidence_bundle()
    identity = objectstate_real_identity_rows_summary(bundle)
    prediction = objectstate_real_prediction_rows_summary(bundle)
    intervention = objectstate_real_intervention_rows_summary(bundle)
    identity_path = tmp_path / "real-identity-summary.json"
    prediction_path = tmp_path / "real-prediction-summary.json"
    intervention_path = tmp_path / "real-intervention-summary.json"
    _write_json(identity_path, identity)
    _write_json(prediction_path, prediction)
    _write_json(intervention_path, intervention)

    ledger = objectstate_reality_row_ledger(
        (identity_path, prediction_path, intervention_path)
    )

    assert ledger["status"] == "objectstate_reality_row_ledger_reviewable"
    assert ledger["summary_count"] == 3
    assert ledger["row_count"] == 3
    assert ledger["pass_row_count"] == 3
    assert ledger["fail_row_count"] == 0
    assert ledger["blocked_row_count"] == 0
    assert ledger["gate"]["status"] == "objectstate_reality_gate_pass"
    assert ledger["gap_summary"]["missing_pass_evidence_kinds"] == []
    assert ledger["row_counts"]["by_evidence_kind"] == {
        "identity": 1,
        "prediction": 1,
        "intervention": 1,
    }
    experiment_status = {
        row["experiment"]: row["status"]
        for row in ledger["state_variable_evidence_matrix"]
    }
    assert experiment_status["identity_persistence"] == (
        "objectstate_state_variable_experiment_pass"
    )
    assert experiment_status["predictive_sufficiency"] == (
        "objectstate_state_variable_experiment_pass"
    )
    assert experiment_status["counterfactual_action_interface"] == (
        "objectstate_state_variable_experiment_pass"
    )
    assert ledger["next_actions"] == []
    assert validate_objectstate_reality_row_ledger_summary(ledger) == ledger


def test_real_evidence_bundle_ledger_writes_phase1_audit_package(tmp_path):
    bundle_path = tmp_path / "real-evidence-bundle.json"
    output_root = tmp_path / "phase1-ledger"
    _write_json(bundle_path, _real_evidence_bundle())

    summary = write_objectstate_real_evidence_bundle_ledger(
        (bundle_path,),
        output_root=output_root,
    )

    assert summary["schema"] == OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA
    assert summary["status"] == "objectstate_real_evidence_bundle_ledger_reviewable"
    assert summary["bundle_count"] == 1
    assert summary["row_summary_count"] == 3
    assert summary["ledger"]["schema"] == OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA
    assert summary["ledger"]["gate"]["status"] == "objectstate_reality_gate_pass"
    assert summary["row_counts"] == {
        "row_count": 3,
        "pass_row_count": 3,
        "fail_row_count": 0,
        "blocked_row_count": 0,
        "evidence_incomplete_row_count": 0,
        "unsupported_row_count": 0,
    }
    assert summary["accounting_status_counts"]["all"] == {
        "pass": 3,
        "fail": 0,
        "evidence_incomplete": 0,
        "unsupported": 0,
    }
    assert summary["evidence_accounts"]["static_scene_evidence"] == {
        "available_bundle_count": 1,
        "usable_for_state_variable_gate": False,
    }
    assert summary["evidence_accounts"]["state_variable_evidence"] == {
        "ready_bundle_count": 1,
        "intervention_ready_bundle_count": 1,
        "requires_full_reality_row_ledger": True,
    }
    record = summary["records"][0]
    for key in (
        "bundle_summary_path",
        "identity_summary_path",
        "prediction_summary_path",
        "intervention_summary_path",
    ):
        assert _read_json(record[key])["schema"].startswith("objgauss-objectstate-")
    assert _read_json(summary["ledger_summary_path"]) == summary["ledger"]
    assert summary["blocked_rows_path"].endswith("reality-row-ledger-blocked.md")
    assert summary["state_variable_evidence_matrix_path"].endswith(
        "state-variable-evidence-matrix.md"
    )
    assert validate_objectstate_real_evidence_bundle_ledger_summary(summary) == summary


def test_real_evidence_bundle_ledger_keeps_incomplete_and_unsupported_counts(
    tmp_path,
):
    bundle_path = tmp_path / "real-evidence-bundle.json"
    output_root = tmp_path / "phase1-ledger"
    _write_json(
        bundle_path,
        _real_evidence_bundle_with_accounting_statuses(
            identity_status="evidence_incomplete",
            prediction_status="unsupported",
            intervention_status="evidence_incomplete",
        ),
    )

    summary = write_objectstate_real_evidence_bundle_ledger(
        (bundle_path,),
        output_root=output_root,
    )
    package = objectstate_real_evidence_bundle_ledger_package_audit(output_root)

    assert summary["status"] == "objectstate_real_evidence_bundle_ledger_reviewable"
    assert summary["ledger"]["gate"]["status"] == "objectstate_reality_gate_fail"
    assert summary["row_counts"] == {
        "row_count": 3,
        "pass_row_count": 0,
        "fail_row_count": 0,
        "blocked_row_count": 3,
        "evidence_incomplete_row_count": 2,
        "unsupported_row_count": 1,
    }
    assert summary["accounting_status_counts"]["all"] == {
        "pass": 0,
        "fail": 0,
        "evidence_incomplete": 2,
        "unsupported": 1,
    }
    assert summary["accounting_status_counts"]["identity"]["evidence_incomplete"] == 1
    assert summary["accounting_status_counts"]["prediction"]["unsupported"] == 1
    assert (
        summary["accounting_status_counts"]["intervention"]["evidence_incomplete"]
        == 1
    )
    assert package["accounting_status_counts"] == summary["accounting_status_counts"]
    assert package["row_counts"]["evidence_incomplete_row_count"] == 2
    assert package["row_counts"]["unsupported_row_count"] == 1
    assert package["phase1_acceptance_status"] == (
        "objectstate_phase1_evidence_system_acceptance_incomplete"
    )
    assert package["phase1_acceptance_gates"][
        "controlled_or_public_real_bundle_loaded"
    ] is True
    assert package["phase1_acceptance_gates"]["identity_rows_enter_accounting"] is True
    assert package["phase1_acceptance_gates"]["prediction_rows_enter_accounting"] is True
    assert package["phase1_acceptance_gates"][
        "intervention_rows_enter_accounting"
    ] is True
    assert package["phase1_acceptance_gates"][
        "missing_gt_accounting_is_separate_from_fail"
    ] is True
    assert package["phase1_acceptance_gates"][
        "evaluable_real_accounting_rows_present"
    ] is False
    assert "phase1 acceptance gate failed: evaluable_real_accounting_rows_present" in (
        package["phase1_acceptance_issues"]
    )
    assert validate_objectstate_real_evidence_bundle_ledger_summary(summary) == summary


def test_real_evidence_bundle_ledger_cli_writes_full_ledger_outputs(tmp_path, capsys):
    bundle_path = tmp_path / "real-evidence-bundle.json"
    output_root = tmp_path / "phase1-ledger"
    wrapper_path = tmp_path / "wrapper.json"
    _write_json(bundle_path, _real_evidence_bundle())

    assert (
        main(
            [
                "object-state",
                "audit-real-evidence-bundle-ledger",
                str(bundle_path),
                "--output-root",
                str(output_root),
                "--summary-output",
                str(wrapper_path),
                "--require-reviewable",
                "--require-gate-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(wrapper_path)
    ledger = _read_json(output_root / "reality-row-ledger.json")

    assert f"schema={OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA}" in stdout
    assert "ledger_status=objectstate_reality_row_ledger_reviewable" in stdout
    assert "gate_status=objectstate_reality_gate_pass" in stdout
    assert "evidence_incomplete_row_count=0" in stdout
    assert "unsupported_row_count=0" in stdout
    assert "bundle=controlled-tabletop-cup-001:" in stdout
    assert summary["ledger"] == ledger
    assert ledger["row_count"] == 3
    assert (output_root / "real-evidence-bundle-ledger.json").exists()
    assert (output_root / "reality-row-ledger-blocked.md").exists()
    assert (output_root / "state-variable-evidence-matrix.md").exists()
    assert (output_root / "reality-row-ledger-next-actions.md").exists()
    assert validate_objectstate_real_evidence_bundle_ledger_summary(summary) == summary


def test_real_evidence_bundle_ledger_package_audit_is_reviewable(tmp_path):
    bundle_path = tmp_path / "real-evidence-bundle.json"
    output_root = tmp_path / "phase1-ledger"
    _write_json(bundle_path, _real_evidence_bundle())
    write_objectstate_real_evidence_bundle_ledger(
        (bundle_path,),
        output_root=output_root,
    )

    summary = objectstate_real_evidence_bundle_ledger_package_audit(output_root)

    assert (
        summary["schema"]
        == OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_PACKAGE_AUDIT_SCHEMA
    )
    assert summary["status"] == (
        "objectstate_real_evidence_bundle_ledger_package_audit_reviewable"
    )
    assert all(summary["reviewability_gates"].values())
    assert summary["sample_ids"] == ["controlled-tabletop-cup-001"]
    assert summary["row_counts"] == {
        "row_count": 3,
        "pass_row_count": 3,
        "fail_row_count": 0,
        "blocked_row_count": 0,
        "evidence_incomplete_row_count": 0,
        "unsupported_row_count": 0,
    }
    assert summary["accounting_status_counts"]["all"] == {
        "pass": 3,
        "fail": 0,
        "evidence_incomplete": 0,
        "unsupported": 0,
    }
    assert summary["evidence_accounts"]["static_scene_evidence"] == {
        "available_bundle_count": 1,
        "usable_for_state_variable_gate": False,
    }
    assert summary["phase1_acceptance_status"] == (
        "objectstate_phase1_evidence_system_acceptance_pass"
    )
    assert all(summary["phase1_acceptance_gates"].values())
    assert summary["phase1_acceptance_counts"] == {
        "bundle_count": 1,
        "controlled_public_bundle_count": 1,
        "identity_accounting_row_count": 1,
        "prediction_accounting_row_count": 1,
        "intervention_accounting_row_count": 1,
        "pass_row_count": 3,
        "fail_row_count": 0,
        "pass_fail_row_count": 3,
        "evidence_incomplete_row_count": 0,
        "unsupported_row_count": 0,
        "static_scene_available_bundle_count": 1,
        "state_variable_ready_bundle_count": 1,
        "state_variable_intervention_ready_bundle_count": 1,
    }
    assert summary["claim_policy"]["full_reality_row_ledger_is_authoritative"] is True
    assert validate_objectstate_real_evidence_bundle_ledger_package_audit(summary) == (
        summary
    )


def test_real_evidence_bundle_ledger_package_audit_reports_missing_output(tmp_path):
    bundle_path = tmp_path / "real-evidence-bundle.json"
    output_root = tmp_path / "phase1-ledger"
    _write_json(bundle_path, _real_evidence_bundle())
    write_objectstate_real_evidence_bundle_ledger(
        (bundle_path,),
        output_root=output_root,
    )
    (output_root / "state-variable-evidence-matrix.md").unlink()

    summary = objectstate_real_evidence_bundle_ledger_package_audit(output_root)

    assert summary["status"] == (
        "objectstate_real_evidence_bundle_ledger_package_audit_incomplete"
    )
    assert summary["reviewability_gates"]["required_files_present"] is False
    assert summary["reviewability_gates"]["markdown_outputs_present"] is False
    assert any(
        "state_variable_evidence_matrix_markdown" in issue
        for issue in summary["issues"]
    )


def test_real_evidence_bundle_ledger_package_audit_cli(tmp_path, capsys):
    bundle_path = tmp_path / "real-evidence-bundle.json"
    output_root = tmp_path / "phase1-ledger"
    audit_path = tmp_path / "package-audit.json"
    _write_json(bundle_path, _real_evidence_bundle())
    write_objectstate_real_evidence_bundle_ledger(
        (bundle_path,),
        output_root=output_root,
    )

    assert (
        main(
            [
                "object-state",
                "audit-real-evidence-bundle-ledger-package",
                str(output_root),
                "--summary-output",
                str(audit_path),
                "--require-reviewable",
                "--require-phase1-acceptance",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(audit_path)

    assert (
        f"schema={OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_PACKAGE_AUDIT_SCHEMA}"
        in stdout
    )
    assert "status=objectstate_real_evidence_bundle_ledger_package_audit_reviewable" in stdout
    assert (
        "phase1_acceptance_status="
        "objectstate_phase1_evidence_system_acceptance_pass"
    ) in stdout
    assert "evidence_incomplete_row_count=0" in stdout
    assert "unsupported_row_count=0" in stdout
    assert "reviewability.wrapper_embedded_ledger_matches_file=true" in stdout
    assert "phase1.controlled_or_public_real_bundle_loaded=true" in stdout
    assert "phase1.static_scene_and_state_variable_evidence_split=true" in stdout
    assert summary["status"] == (
        "objectstate_real_evidence_bundle_ledger_package_audit_reviewable"
    )
    assert validate_objectstate_real_evidence_bundle_ledger_package_audit(summary) == (
        summary
    )


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


def _real_evidence_bundle():
    return {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "kind": "objectstate_real_evidence_bundle",
        "sample": {
            "sample_id": "controlled-tabletop-cup-001",
            "scene_id": "tabletop-cup-box",
            "sequence_id": "push-left-001",
            "source_dataset": "local-controlled-tabletop",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "push_left_counterfactual",
            "gt_provenance": "external-motion-capture",
            "license": "local-research",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["outputs/captures/cup/capture-manifest.json"],
        },
        "row_schemas": {
            "observation": "objgauss-objectstate-real-observation-row-v1",
            "object_pose": "objgauss-objectstate-real-object-pose-row-v1",
            "identity_link": "objgauss-objectstate-real-identity-link-row-v1",
            "action_interval": "objgauss-objectstate-real-action-interval-row-v1",
            "state_transition": "objgauss-objectstate-real-state-transition-row-v1",
            "gate_accounting": "objgauss-objectstate-real-gate-accounting-row-v1",
        },
        "observation_rows": [
            _observation("obs-000", "f000", 0.0),
            _observation("obs-001", "f001", 0.1),
        ],
        "object_pose_rows": [
            _pose("pose-000", "f000", 0.0, [0.0, 0.0, 0.0]),
            _pose("pose-001", "f001", 0.1, [-0.1, 0.0, 0.0]),
        ],
        "identity_link_rows": [
            _identity_link("identity-000", "f000", 0.0),
            _identity_link("identity-001", "f001", 0.1),
        ],
        "action_interval_rows": [
            {
                "schema": "objgauss-objectstate-real-action-interval-row-v1",
                "row_id": "action-row-001",
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "cup-001",
                "action_start_ts": 0.02,
                "action_end_ts": 0.08,
                "action_vector": [-0.1, 0.0, 0.0],
                "actor": "hand-001",
                "gt_provenance": "external-motion-capture",
            }
        ],
        "state_transition_rows": [
            {
                "schema": "objgauss-objectstate-real-state-transition-row-v1",
                "row_id": "transition-row-001",
                "transition_id": "transition-cup-000-001",
                "object_id": "cup-001",
                "source_frame_id": "f000",
                "target_frame_id": "f001",
                "source_timestamp": 0.0,
                "target_timestamp": 0.1,
                "source_pose_row_id": "pose-000",
                "target_pose_row_id": "pose-001",
                "gt_provenance": "external-motion-capture",
            }
        ],
        "gate_accounting_rows": [
            {
                "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
                "row_id": "cup-identity-pass-001",
                "evidence_kind": "identity",
                "accounting_status": "pass",
                "object_id": "cup-001",
                "metrics": {
                    "idf1": 1.0,
                    "fragmentation_rate": 0.0,
                    "swap_rate": 0.0,
                    "identity_collapse": False,
                },
                "artifact_refs": ["outputs/captures/cup/identity-eval.json"],
                "gt_requirements": {
                    "identity": True,
                    "pose": False,
                    "action": False,
                    "timestamp": True,
                },
            },
            {
                "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
                "row_id": "cup-prediction-pass-001",
                "evidence_kind": "prediction",
                "accounting_status": "pass",
                "object_id": "cup-001",
                "transition_id": "transition-cup-000-001",
                "metrics": {
                    "state_ade": 0.02,
                    "history_ade": 0.04,
                    "prediction_gap_vs_history_model": -0.02,
                },
                "artifact_refs": ["outputs/captures/cup/prediction-eval.json"],
                "gt_requirements": {
                    "identity": True,
                    "pose": True,
                    "action": False,
                    "timestamp": True,
                },
            },
            {
                "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
                "row_id": "cup-intervention-pass-001",
                "evidence_kind": "intervention",
                "accounting_status": "pass",
                "object_id": "cup-001",
                "action_id": "push-left-001",
                "transition_id": "transition-cup-000-001",
                "metrics": {
                    "action_conditioned_ade": 0.01,
                    "no_action_ade": 0.12,
                    "counterfactual_outcome_accuracy": 1.0,
                    "wrong_direction_rate": 0.0,
                    "identity_consistency_rate": 1.0,
                },
                "artifact_refs": ["outputs/captures/cup/intervention-eval.json"],
                "gt_requirements": {
                    "identity": True,
                    "pose": True,
                    "action": True,
                    "timestamp": True,
                },
            },
        ],
    }


def _real_evidence_bundle_with_accounting_statuses(
    *,
    identity_status: str,
    prediction_status: str,
    intervention_status: str,
):
    bundle = json.loads(json.dumps(_real_evidence_bundle()))
    statuses = {
        "identity": identity_status,
        "prediction": prediction_status,
        "intervention": intervention_status,
    }
    for row in bundle["gate_accounting_rows"]:
        status = statuses[row["evidence_kind"]]
        row["accounting_status"] = status
        if status in {"evidence_incomplete", "unsupported"}:
            row["metrics"] = {}
            row["reason"] = f"{row['evidence_kind']} evidence is {status}"
        else:
            row.pop("reason", None)
    return bundle


def _observation(row_id, frame_id, timestamp):
    return {
        "schema": "objgauss-objectstate-real-observation-row-v1",
        "row_id": row_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_id": "cam-001",
        "observation": {
            "rgb": f"rgb/{frame_id}.png",
            "gaussian": f"gaussians/{frame_id}.ply",
        },
    }


def _pose(row_id, frame_id, timestamp, position):
    return {
        "schema": "objgauss-objectstate-real-object-pose-row-v1",
        "row_id": row_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_id": "cam-001",
        "object_id": "cup-001",
        "object_pose_6dof": {
            "position": position,
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "object_visibility": 1.0,
    }


def _identity_link(row_id, frame_id, timestamp):
    return {
        "schema": "objgauss-objectstate-real-identity-link-row-v1",
        "row_id": row_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "object_id": "cup-001",
        "physical_identity_id": "cup-physical-001",
        "gt_provenance": "external-motion-capture",
        "confidence": 1.0,
    }


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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
