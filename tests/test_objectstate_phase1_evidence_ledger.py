from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_phase1_evidence_ledger import (
    OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
    objectstate_phase1_evidence_ledger,
    validate_objectstate_phase1_evidence_ledger_summary,
)


def test_phase1_evidence_ledger_tracks_identity_and_prediction(tmp_path):
    identity_path = tmp_path / "identity-summary.json"
    prediction_path = tmp_path / "prediction-summary.json"
    _write_json(identity_path, _identity_summary(row_status="fail"))
    _write_json(prediction_path, _prediction_summary(row_status="pass"))

    summary = objectstate_phase1_evidence_ledger(
        identity_summaries=(identity_path,),
        prediction_summaries=(prediction_path,),
    )

    assert summary["schema"] == OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA
    assert summary["status"] == "objectstate_phase1_evidence_ledger_reviewable"
    assert summary["maturity"] == "identity_prediction_reviewable"
    assert summary["stage_summary"]["identity"]["reviewable_count"] == 1
    assert summary["stage_summary"]["identity"]["fail_row_count"] == 1
    assert summary["stage_summary"]["prediction"]["pass_row_count"] == 1
    assert summary["phase1_evidence_gates"]["identity_evidence_reviewable"] is True
    assert summary["phase1_evidence_gates"]["prediction_evidence_reviewable"] is True
    assert summary["phase1_evidence_gates"]["full_reality_evidence_reviewable"] is False
    assert summary["phase1_evidence_gates"]["does_not_claim_world_model"] is True
    assert summary["issues"] == []
    assert validate_objectstate_phase1_evidence_ledger_summary(summary) == summary


def test_phase1_evidence_ledger_tracks_full_reality_package(tmp_path):
    reality_path = tmp_path / "reality-summary.json"
    _write_json(reality_path, _reality_summary(pass_rows=2, fail_rows=1, blocked_rows=0))

    summary = objectstate_phase1_evidence_ledger(reality_summaries=(reality_path,))

    assert summary["maturity"] == "full_reality_reviewable"
    assert summary["phase1_evidence_gates"]["identity_evidence_reviewable"] is True
    assert summary["phase1_evidence_gates"]["prediction_evidence_reviewable"] is True
    assert summary["phase1_evidence_gates"]["intervention_evidence_reviewable"] is True
    assert summary["stage_summary"]["full_reality"]["pass_row_count"] == 2
    assert summary["stage_summary"]["full_reality"]["fail_row_count"] == 1
    assert summary["stage_summary"]["full_reality"]["blocked_row_count"] == 0


def test_phase1_evidence_ledger_reports_missing_summary(tmp_path):
    missing_path = tmp_path / "missing-identity-summary.json"

    summary = objectstate_phase1_evidence_ledger(
        identity_summaries=(missing_path,),
    )

    assert summary["status"] == "objectstate_phase1_evidence_ledger_incomplete"
    assert summary["ledger_gates"]["all_files_present"] is False
    assert any("summary file is missing" in issue for issue in summary["issues"])


def test_object_state_audit_phase1_evidence_ledger_cli(tmp_path, capsys):
    identity_path = tmp_path / "identity-summary.json"
    prediction_path = tmp_path / "prediction-summary.json"
    reality_path = tmp_path / "reality-summary.json"
    summary_path = tmp_path / "phase1-ledger-summary.json"
    _write_json(identity_path, _identity_summary(row_status="pass"))
    _write_json(prediction_path, _prediction_summary(row_status="pass"))
    _write_json(reality_path, _reality_summary(pass_rows=3, fail_rows=0, blocked_rows=0))

    assert (
        main(
            [
                "object-state",
                "audit-phase1-evidence-ledger",
                "--identity-summary",
                str(identity_path),
                "--prediction-summary",
                str(prediction_path),
                "--reality-summary",
                str(reality_path),
                "--summary-output",
                str(summary_path),
                "--require-reviewable",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA}" in stdout
    assert "ledger_status=objectstate_phase1_evidence_ledger_reviewable" in stdout
    assert "maturity=full_reality_reviewable" in stdout
    assert "identity_reviewable=1" in stdout
    assert "prediction_reviewable=1" in stdout
    assert "full_reality_reviewable=1" in stdout
    assert "phase1_gate.does_not_claim_world_model=true" in stdout
    assert "issue_count=0" in stdout
    assert summary["maturity"] == "full_reality_reviewable"


def _identity_summary(*, row_status: str):
    gates = {
        "required_files_present": True,
        "required_json_schemas_valid": True,
        "sample_ids_consistent": True,
        "capture_file_audit_pass": True,
        "candidate_artifact_file_audit_pass": True,
        "candidate_artifact_ref_match": True,
        "identity_scenario_audit_pass": True,
        "identity_predictions_present": True,
        "identity_eval_present": True,
        "identity_row_present": True,
        "identity_row_is_pass_or_fail": True,
        "controlled_real_output_matches_eval": True,
        "standalone_outputs_match_handoff": True,
        "identity_only_gate_does_not_require_prediction_or_intervention": True,
    }
    return {
        "schema": "objgauss-objectstate-controlled-identity-evidence-package-v1",
        "kind": "objectstate_controlled_identity_evidence_package",
        "status": "objectstate_controlled_identity_evidence_package_reviewable",
        "package_root": "/tmp/identity-package",
        "sample_id": "controlled-tabletop-cup-box-001",
        "files": [_file_record("handoff_summary")],
        "sample_consistency": {
            "consistent": True,
            "sample_id": "controlled-tabletop-cup-box-001",
            "unique_sample_ids": ["controlled-tabletop-cup-box-001"],
            "values": {"handoff_summary": "controlled-tabletop-cup-box-001"},
        },
        "identity": {
            "identity_eval_present": True,
            "identity_eval_status": (
                "objectstate_controlled_identity_eval_pass"
                if row_status == "pass"
                else "objectstate_controlled_identity_eval_fail"
            ),
            "identity_prediction_count": 6,
            "identity_row_present": True,
            "identity_row_status": row_status,
        },
        "evidence": {},
        "handoff_consistency": {
            "controlled_real_matches_eval": True,
            "standalone_outputs_match_handoff": True,
            "issues": [],
        },
        "reviewability_gates": gates,
        "issues": [],
        "claim_policy": _identity_claim_policy(),
        "non_goals": _non_goals(identity=True),
    }


def _prediction_summary(*, row_status: str):
    gates = {
        "required_files_present": True,
        "required_json_schemas_valid": True,
        "sample_ids_consistent": True,
        "bop_acceptance_pass": True,
        "phase1_gaussian_evidence_ready": True,
        "capture_file_audit_pass": True,
        "prediction_candidates_present": True,
        "prediction_eval_present": True,
        "prediction_row_present": True,
        "prediction_row_is_pass_or_fail": True,
        "controlled_real_output_matches_eval": True,
    }
    return {
        "schema": "objgauss-objectstate-controlled-prediction-evidence-package-v1",
        "kind": "objectstate_controlled_prediction_evidence_package",
        "status": "objectstate_controlled_prediction_evidence_package_reviewable",
        "package_root": "/tmp/prediction-package",
        "candidate_dir": "/tmp/prediction-package/reality-candidates",
        "sample_id": "bop-ycbv-scene-000001",
        "files": [_file_record("prediction_eval_summary")],
        "sample_consistency": {
            "consistent": True,
            "sample_id": "bop-ycbv-scene-000001",
            "unique_sample_ids": ["bop-ycbv-scene-000001"],
            "values": {"prediction_eval_summary": "bop-ycbv-scene-000001"},
        },
        "acceptance": {
            "status": "objectstate_bop_capture_acceptance_pass",
            "bop_acceptance_pass": True,
            "phase1_gaussian_evidence_ready": True,
            "capture_file_audit_pass": True,
        },
        "prediction": {
            "prediction_eval_present": True,
            "prediction_eval_status": (
                "objectstate_controlled_prediction_eval_pass"
                if row_status == "pass"
                else "objectstate_controlled_prediction_eval_fail"
            ),
            "prediction_candidate_count": 4,
            "prediction_row_present": True,
            "prediction_row_status": row_status,
        },
        "output_consistency": {"matches": True, "issues": []},
        "reviewability_gates": gates,
        "issues": [],
        "claim_policy": _prediction_claim_policy(),
        "non_goals": _non_goals(),
    }


def _reality_summary(*, pass_rows: int, fail_rows: int, blocked_rows: int):
    gates = {
        "required_files_present": True,
        "required_json_schemas_valid": True,
        "sample_ids_consistent": True,
        "full_reality_handoff_ready_recorded": True,
        "handoff_summary_present": True,
        "standalone_outputs_match_handoff_summary": True,
        "controlled_real_summary_present": True,
        "row_accounting_present": True,
        "identity_prediction_intervention_rows_present": True,
        "blocked_rows_markdown_present": True,
    }
    return {
        "schema": "objgauss-objectstate-controlled-reality-evidence-package-v1",
        "kind": "objectstate_controlled_reality_evidence_package",
        "status": "objectstate_controlled_reality_evidence_package_reviewable",
        "package_root": "/tmp/reality-package",
        "candidate_dir": "/tmp/reality-package/reality-candidates",
        "handoff_dir": "/tmp/reality-package/reality-handoff",
        "sample_id": "controlled-tabletop-cup-box-001",
        "files": [_file_record("reality_bundle_handoff_summary")],
        "sample_consistency": {
            "consistent": True,
            "sample_id": "controlled-tabletop-cup-box-001",
            "unique_sample_ids": ["controlled-tabletop-cup-box-001"],
            "values": {
                "reality_bundle_handoff_summary": "controlled-tabletop-cup-box-001"
            },
        },
        "readiness": {
            "status": "objectstate_controlled_reality_bundle_readiness_ready",
            "full_reality_handoff_ready": True,
        },
        "handoff": {
            "status": "objectstate_controlled_reality_bundle_handoff_pass",
            "full_reality_gate_status": "objectstate_reality_gate_pass",
            "prediction_eval_status": "objectstate_controlled_prediction_eval_pass",
            "intervention_eval_status": "objectstate_controlled_intervention_eval_pass",
        },
        "row_accounting": {
            "present": True,
            "row_count": pass_rows + fail_rows + blocked_rows,
            "pass_row_count": pass_rows,
            "fail_row_count": fail_rows,
            "blocked_row_count": blocked_rows,
            "evidence_kinds": ["identity", "intervention", "prediction"],
            "identity_prediction_intervention_rows_present": True,
        },
        "output_consistency": {"matches": True, "issues": []},
        "reviewability_gates": gates,
        "issues": [],
        "claim_policy": _reality_claim_policy(),
        "non_goals": _non_goals(),
    }


def _file_record(key: str):
    return {
        "key": key,
        "path": f"/tmp/{key}.json",
        "required": True,
        "kind": "json",
        "exists": True,
        "is_file": True,
        "size_bytes": 128,
        "schema": "fixture-schema",
        "expected_schema": "fixture-schema",
        "schema_ok": True,
        "validator_ok": True,
        "sample_id": "controlled-tabletop-cup-box-001",
        "status": "fixture-status",
        "issues": [],
    }


def _identity_claim_policy():
    policy = _base_claim_policy()
    policy.update(
        {
            "checks_local_identity_evidence_package": True,
            "requires_real_gaussian_file_acceptance": True,
            "requires_candidate_artifact_audit": True,
            "requires_identity_scenario_audit": True,
            "reviewable_allows_identity_pass_or_fail": True,
            "does_not_run_identity_eval": True,
            "does_not_claim_prediction_or_intervention_gate": True,
        }
    )
    return policy


def _prediction_claim_policy():
    policy = _base_claim_policy()
    policy.update(
        {
            "checks_local_prediction_evidence_package": True,
            "requires_real_gaussian_file_acceptance": True,
        }
    )
    return policy


def _reality_claim_policy():
    policy = _base_claim_policy()
    policy.update(
        {
            "checks_local_evidence_package": True,
            "does_not_run_identity_handoff": True,
        }
    )
    return policy


def _base_claim_policy():
    return {
        "read_only_audit": True,
        "does_not_create_ground_truth": True,
        "does_not_run_prediction_eval": True,
        "does_not_run_intervention_eval": True,
        "does_not_claim_metric_pass": True,
        "does_not_claim_intervention_gate": True,
        "does_not_claim_world_model": True,
    }


def _non_goals(*, identity: bool = False):
    non_goals = {
        "captures_video": False,
        "creates_ground_truth": False,
        "reconstructs_gaussians": False,
        "runs_tracking_model": False,
        "runs_prediction_model": False,
        "runs_intervention_model": False,
        "trains_gaussian_model": False,
        "trains_dynamics_model": False,
        "writes_public_samples": False,
        "uses_replay_buffer": False,
        "uses_diffusion": False,
        "mutates_viewer_defaults": False,
    }
    if identity:
        non_goals["runs_identity_model"] = False
    return non_goals


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
