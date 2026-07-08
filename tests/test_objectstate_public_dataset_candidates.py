from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_public_dataset_candidates import (
    OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA,
    ObjectStatePublicDatasetCandidate,
    default_objectstate_public_dataset_candidates,
    objectstate_public_dataset_candidates_audit,
    objectstate_public_dataset_candidates_markdown,
    validate_objectstate_public_dataset_candidates_audit,
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
