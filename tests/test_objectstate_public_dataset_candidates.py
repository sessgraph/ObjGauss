from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_public_dataset_candidates import (
    OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA,
    OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA,
    ObjectStatePublicDatasetCandidate,
    default_objectstate_public_dataset_candidates,
    objectstate_public_dataset_candidates_audit,
    objectstate_public_dataset_candidates_markdown,
    objectstate_public_interaction_route_audit,
    objectstate_public_interaction_route_markdown,
    validate_objectstate_public_dataset_candidates_audit,
    validate_objectstate_public_interaction_route_audit,
)


def test_public_dataset_candidate_audit_keeps_dataset_choice_blocked():
    summary = objectstate_public_dataset_candidates_audit()

    assert summary["schema"] == OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA
    assert summary["candidate_count"] == 5
    assert summary["recommended_first"] == "bop-ycbv-keyframes"
    assert summary["recommended_action_candidate"] == "hot3d-clips"
    assert summary["readiness"]["has_identity_pose_dataset"] is True
    assert summary["readiness"]["has_action_like_dataset"] is True
    assert summary["readiness"]["has_direct_gaussian_evidence"] is False
    assert summary["readiness"]["has_direct_phase1_ready_dataset"] is False
    assert summary["claim_policy"]["does_not_claim_reality_gate_pass"] is True
    assert summary["non_goals"]["downloads_datasets"] is False
    assert summary["coverage_counts"]["counterfactual_interface"]["blocked"] == 4
    assert "no public candidate directly supplies ObjGauss per-frame Gaussian evidence" in (
        summary["hard_blockers"]
    )
    assert validate_objectstate_public_dataset_candidates_audit(summary) == summary


def test_public_dataset_candidate_markdown_reports_recommendations():
    summary = objectstate_public_dataset_candidates_audit()
    markdown = objectstate_public_dataset_candidates_markdown(summary)

    assert "# ObjectState Public Dataset Candidate Audit" in markdown
    assert "bop-ycbv-keyframes" in markdown
    assert "hot3d-clips" in markdown
    assert "counterfactual" in markdown


def test_public_dataset_candidate_validation_rejects_fake_ready_dataset():
    summary = objectstate_public_dataset_candidates_audit()
    summary["readiness"]["has_direct_phase1_ready_dataset"] = True

    with pytest.raises(ValueError, match="direct Phase 1 ready"):
        validate_objectstate_public_dataset_candidates_audit(summary)


def test_public_dataset_candidate_requires_complete_gate_coverage():
    candidate = default_objectstate_public_dataset_candidates()[0]
    bad_candidate = ObjectStatePublicDatasetCandidate(
        **{
            **candidate.__dict__,
            "gate_coverage": {"identity_persistence": "ready_with_adapter"},
        }
    )

    with pytest.raises(ValueError, match="unsupported coverage"):
        objectstate_public_dataset_candidates_audit((bad_candidate,))


def test_object_state_audit_public_dataset_candidates_cli(tmp_path, capsys):
    summary_path = tmp_path / "public-dataset-candidates.json"
    markdown_path = tmp_path / "public-dataset-candidates.md"

    assert (
        main(
            [
                "object-state",
                "audit-public-dataset-candidates",
                "--summary-output",
                str(summary_path),
                "--markdown-output",
                str(markdown_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert f"schema={OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA}" in stdout
    assert "recommended_first=bop-ycbv-keyframes" in stdout
    assert "direct_phase1_ready=false" in stdout
    assert summary["candidate_count"] == 5
    assert "BOP YCB-V" in markdown


def test_public_interaction_route_audit_keeps_missing_local_clip_blocked():
    summary = objectstate_public_interaction_route_audit(candidate_id="hot3d-clips")

    assert summary["schema"] == OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA
    assert (
        summary["status"]
        == "objectstate_public_interaction_route_blocked_no_local_dataset"
    )
    assert summary["candidate"]["candidate_id"] == "hot3d-clips"
    assert summary["readiness"]["candidate_has_action_gt"] is True
    assert summary["readiness"]["dataset_root_present"] is False
    assert summary["readiness"]["controlled_reality_handoff_ready"] is False
    assert summary["accounting_route_status"] == "evidence_incomplete"
    assert "local public interaction dataset root is missing" in summary["hard_blockers"]
    assert summary["claim_policy"]["does_not_claim_intervention_pass"] is True
    assert validate_objectstate_public_interaction_route_audit(summary) == summary


def test_public_interaction_route_audit_reports_handoff_ready(tmp_path):
    bundle_root = tmp_path / "hot3d-clip"
    _write_interaction_route_bundle(bundle_root)

    summary = objectstate_public_interaction_route_audit(dataset_root=bundle_root)

    assert (
        summary["status"]
        == "objectstate_public_interaction_route_handoff_ready"
    )
    assert summary["accounting_route_status"] == "handoff_ready"
    assert summary["readiness"]["capture_intervention_ready"] is True
    assert summary["readiness"]["capture_intervention_action_gt_ready"] is True
    assert summary["readiness"]["gaussian_evidence_declared"] is True
    assert summary["readiness"]["candidate_artifact_present"] is True
    assert summary["readiness"]["prediction_candidates_valid"] is True
    assert summary["readiness"]["intervention_candidates_valid"] is True
    assert summary["readiness"]["identity_accounting_ready"] is True
    assert summary["readiness"]["prediction_accounting_ready"] is True
    assert summary["readiness"]["intervention_accounting_ready"] is True
    assert summary["readiness"]["sample_ids_match"] is True
    assert summary["hard_blockers"] == []
    markdown = objectstate_public_interaction_route_markdown(summary)
    assert "accounting route status: `handoff_ready`" in markdown
    assert "handoff ready: `true`" in markdown
    assert "| capture_intervention_action_gt_ready | true |" in markdown
    assert "| prediction_accounting_ready | true |" in markdown
    assert "observed interactions are action-like" in markdown


def test_public_interaction_route_keeps_weak_action_gt_prediction_ready(tmp_path):
    bundle_root = tmp_path / "hot3d-clip"
    _write_interaction_route_bundle(bundle_root, action_vector=[0.0, 0.0, 0.0])

    summary = objectstate_public_interaction_route_audit(dataset_root=bundle_root)

    assert (
        summary["status"]
        == "objectstate_public_interaction_route_intervention_gt_required"
    )
    assert summary["readiness"]["capture_intervention_ready"] is True
    assert summary["readiness"]["capture_intervention_action_gt_ready"] is False
    assert summary["readiness"]["controlled_reality_handoff_ready"] is False
    assert summary["accounting_route_status"] == "prediction_ready"
    assert summary["readiness"]["identity_accounting_ready"] is True
    assert summary["readiness"]["prediction_accounting_ready"] is True
    assert summary["readiness"]["intervention_accounting_ready"] is False
    assert "capture manifest lacks usable intervention action GT" in (
        summary["hard_blockers"]
    )
    assert summary["next_actions"] == [
        (
            "fill action rows with non-zero vectors whose intervals cover "
            "object pose transitions until intervention_action_gt_ready=true"
        ),
        "run controlled-reality-bundle-handoff only after this audit reports handoff_ready",
        (
            "keep counterfactual proof blocked unless observed actions have an explicit "
            "counterfactual evaluation design"
        ),
    ]


def test_public_interaction_route_can_be_identity_ready_without_prediction_candidates(tmp_path):
    bundle_root = tmp_path / "hot3d-clip"
    _write_interaction_route_bundle(bundle_root, write_prediction_candidates=False)

    summary = objectstate_public_interaction_route_audit(dataset_root=bundle_root)

    assert (
        summary["status"]
        == "objectstate_public_interaction_route_candidate_artifacts_required"
    )
    assert summary["accounting_route_status"] == "identity_ready"
    assert summary["readiness"]["capture_identity_ready"] is True
    assert summary["readiness"]["capture_prediction_ready"] is True
    assert summary["readiness"]["identity_accounting_ready"] is True
    assert summary["readiness"]["prediction_accounting_ready"] is False
    assert summary["readiness"]["intervention_accounting_ready"] is False
    assert "prediction candidates JSON is missing or invalid" in summary["hard_blockers"]


def test_public_interaction_route_rejects_pose_only_candidate(tmp_path):
    bundle_root = tmp_path / "bop-pose-only"
    bundle_root.mkdir()

    summary = objectstate_public_interaction_route_audit(
        candidate_id="bop-ycbv-keyframes",
        dataset_root=bundle_root,
    )

    assert (
        summary["status"]
        == "objectstate_public_interaction_route_unsupported_candidate"
    )
    assert summary["accounting_route_status"] == "evidence_incomplete"
    assert summary["readiness"]["candidate_has_action_gt"] is False
    assert "selected candidate does not advertise action ground truth" in (
        summary["hard_blockers"]
    )


def test_object_state_audit_public_interaction_route_cli(tmp_path, capsys):
    bundle_root = tmp_path / "hot3d-clip"
    summary_path = tmp_path / "public-interaction-route.json"
    markdown_path = tmp_path / "public-interaction-route.md"
    _write_interaction_route_bundle(bundle_root)

    assert (
        main(
            [
                "object-state",
                "audit-public-interaction-route",
                str(bundle_root),
                "--summary-output",
                str(summary_path),
                "--markdown-output",
                str(markdown_path),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert f"schema={OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA}" in stdout
    assert "accounting_route_status=handoff_ready" in stdout
    assert "handoff_ready=true" in stdout
    assert "identity_accounting_ready=true" in stdout
    assert "prediction_accounting_ready=true" in stdout
    assert "intervention_accounting_ready=true" in stdout
    assert "capture_intervention_action_gt_ready=true" in stdout
    assert summary["readiness"]["controlled_reality_handoff_ready"] is True
    assert "ObjectState Public Interaction Route Audit" in markdown


def _write_interaction_route_bundle(
    bundle_root,
    *,
    action_vector=None,
    write_prediction_candidates=True,
    write_intervention_candidates=True,
) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    sample_id = "hot3d-clip-unit-001"
    vector = [1.0, 0.0, 0.0] if action_vector is None else action_vector
    (bundle_root / "reality-candidates").mkdir(parents=True, exist_ok=True)
    (bundle_root / "objectstates.json").write_text("{}", encoding="utf-8")
    capture = {
        "schema": "objgauss-objectstate-controlled-capture-manifest-v1",
        "sample": {
            "sample_id": sample_id,
            "source_kind": "controlled_real",
            "object_category": "hand_object_interaction",
            "scenario": "public_interaction_action_like_clip",
            "fps": 30.0,
            "capture_device": "public-interaction-dataset",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["capture-manifest.json"],
            "license": "public interaction dataset license; verify before redistribution",
        },
        "objects": [
            {
                "object_id": "object-001",
                "category": "cup",
                "instance_label": "fixture cup",
            }
        ],
        "actions": [
            {
                "action_id": "action-001",
                "action_type": "move_right",
                "object_id": "object-001",
                "start_timestamp": 0.0333333333,
                "end_timestamp": 0.0666666667,
                "actor": "hand",
                "vector": vector,
            }
        ],
        "frames": [
            _interaction_frame("f000", 0.0, [0.0, 0.0, 0.0]),
            _interaction_frame(
                "f001",
                0.0333333333,
                [0.05, 0.0, 0.0],
                action_id="action-001",
            ),
            _interaction_frame("f002", 0.0666666667, [0.1, 0.0, 0.0]),
        ],
    }
    prediction_candidates = {
        "schema": "objgauss-objectstate-controlled-prediction-candidates-v1",
        "sample_id": sample_id,
        "candidate": {
            "candidate_id": "fixture-predictor",
            "source": "unit-test",
            "artifact_refs": ["reality-candidates/prediction-candidates.json"],
        },
        "predictions": [
            {
                "source_frame_id": "f000",
                "target_frame_id": "f002",
                "object_id": "object-001",
                "predicted_position": [0.1, 0.0, 0.0],
                "history_baseline_position": [0.0, 0.0, 0.0],
            }
        ],
    }
    intervention_candidates = {
        "schema": "objgauss-objectstate-controlled-intervention-candidates-v1",
        "sample_id": sample_id,
        "candidate": {
            "candidate_id": "fixture-intervention",
            "source": "unit-test",
            "artifact_refs": ["reality-candidates/intervention-candidates.json"],
        },
        "interventions": [
            {
                "source_frame_id": "f000",
                "target_frame_id": "f002",
                "object_id": "object-001",
                "action_id": "action-001",
                "action_conditioned_position": [0.1, 0.0, 0.0],
                "no_action_baseline_position": [0.0, 0.0, 0.0],
            }
        ],
    }
    (bundle_root / "capture-manifest.json").write_text(
        json.dumps(capture),
        encoding="utf-8",
    )
    if write_prediction_candidates:
        (bundle_root / "reality-candidates" / "prediction-candidates.json").write_text(
            json.dumps(prediction_candidates),
            encoding="utf-8",
        )
    if write_intervention_candidates:
        (bundle_root / "reality-candidates" / "intervention-candidates.json").write_text(
            json.dumps(intervention_candidates),
            encoding="utf-8",
        )


def _interaction_frame(frame_id, timestamp, position, *, action_id=None):
    frame = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "observation": {
            "rgb": f"rgb/{frame_id}.png",
            "gaussian": f"gaussians/{frame_id}.ply",
        },
        "objects": [
            {
                "object_id": "object-001",
                "visible": True,
                "pose": {
                    "position": position,
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    }
    if action_id is not None:
        frame["action_id"] = action_id
    return frame
