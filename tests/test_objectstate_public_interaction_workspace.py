from __future__ import annotations

import json
from pathlib import Path

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_public_interaction_workspace import (
    OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA,
    OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_PROGRESS_SCHEMA,
    objectstate_public_interaction_workspace_progress,
    validate_objectstate_public_interaction_workspace_progress_summary,
    validate_objectstate_public_interaction_workspace_summary,
    write_objectstate_public_interaction_workspace,
)


def test_public_interaction_workspace_writes_authoring_skeleton(tmp_path):
    workspace = tmp_path / "hot3d-clip"

    summary = write_objectstate_public_interaction_workspace(
        workspace,
        sample_id="hot3d-clip-unit-001",
        source_sequence_id="hot3d-sequence-unit-001",
        objects=[
            {
                "object_id": "cup-001",
                "category": "cup",
                "instance_label": "unit cup",
            }
        ],
    )

    assert summary["schema"] == OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA
    assert summary["status"] == "objectstate_public_interaction_workspace_ready"
    assert summary["candidate"]["candidate_id"] == "hot3d-clips"
    assert summary["candidate"]["source_kind"] == "public_interaction_dataset"
    assert summary["source_sequence_id"] == "hot3d-sequence-unit-001"
    assert summary["sample"]["source_kind"] == "controlled_real"
    assert (
        summary["controlled_capture_template"]["sample"]["source_kind"]
        == "controlled_real"
    )
    assert summary["authoring_policy"] == {
        "uses_controlled_capture_contract_for_local_authoring": True,
        "final_rows_must_be_converted_to_public_replay": True,
        "public_dataset_candidate_required": True,
        "source_sequence_id_required_before_review": True,
        "route_audit_required_before_handoff": True,
        "full_handoff_required_before_reality_rows": True,
    }
    assert summary["claim_policy"]["workspace_only"] is True
    assert summary["claim_policy"]["does_not_create_ground_truth"] is True
    assert summary["claim_policy"]["does_not_claim_counterfactual_proof"] is True
    assert all(value is False for value in summary["non_goals"].values())
    assert validate_objectstate_public_interaction_workspace_summary(summary) == summary

    for key in (
        "sample_json",
        "objects_csv",
        "frames_csv",
        "annotations_csv",
        "actions_csv",
        "readme",
        "public_interaction_readme",
    ):
        assert Path(summary["files"][key]).is_file()
    assert (workspace / "rgb").is_dir()
    assert (workspace / "gaussians").is_dir()
    assert "cup-001,cup,unit cup" in (workspace / "objects.csv").read_text(
        encoding="utf-8"
    )

    route_readme = (workspace / "PUBLIC_INTERACTION_ROUTE.md").read_text(
        encoding="utf-8"
    )
    assert "audit-public-interaction-route" in route_readme
    assert "audit-public-interaction-reality-rows" in route_readme
    assert "source_kind=public_replay" in route_readme
    assert "Observed public interactions are not randomized counterfactual proof" in (
        route_readme
    )
    assert "does not contain captured RGB frames" in (
        workspace / "README.md"
    ).read_text(encoding="utf-8")


def test_public_interaction_workspace_rejects_pose_only_candidate(tmp_path):
    with pytest.raises(ValueError, match="not a public interaction dataset"):
        write_objectstate_public_interaction_workspace(
            tmp_path / "bop-pose-only",
            sample_id="bop-pose-only-001",
            candidate_id="bop-ycbv-keyframes",
        )


def test_public_interaction_workspace_progress_reports_authoring_gap(tmp_path):
    workspace = tmp_path / "hot3d-clip"
    workspace_summary = workspace / "public-interaction-workspace.json"
    scaffold = write_objectstate_public_interaction_workspace(
        workspace,
        sample_id="hot3d-clip-unit-001",
        source_sequence_id="hot3d-sequence-unit-001",
        objects=[{"object_id": "cup-001", "category": "cup"}],
    )
    workspace_summary.write_text(
        json.dumps(scaffold, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = objectstate_public_interaction_workspace_progress(workspace)

    assert summary["schema"] == OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_PROGRESS_SCHEMA
    assert summary["status"] == (
        "objectstate_public_interaction_workspace_progress_blocked"
    )
    assert summary["source_sequence_id"] == "hot3d-sequence-unit-001"
    assert summary["readiness"]["workspace_layout_ready"] is True
    assert summary["readiness"]["workspace_summary_valid"] is True
    assert summary["readiness"]["source_sequence_bound"] is True
    assert summary["readiness"]["controlled_bundle_import_ready"] is False
    assert summary["readiness"]["route_handoff_ready"] is False
    assert summary["readiness"]["public_replay_rows_valid"] is False
    assert summary["readiness"]["evidence_chain_reviewable"] is False
    assert "controlled capture CSV rows are not import-ready" in (
        summary["hard_blockers"]
    )
    assert summary["next_actions"] == [
        (
            "fill objects.csv, frames.csv, annotations.csv and actions.csv with "
            "timestamped identity, 6DoF pose and action rows"
        )
    ]
    assert (
        summary["claim_policy"]["final_rows_must_be_public_replay"] is True
    )
    assert validate_objectstate_public_interaction_workspace_progress_summary(
        summary
    ) == summary


def test_public_interaction_workspace_progress_keeps_todo_sequence_blocked(tmp_path):
    workspace = tmp_path / "hot3d-clip"
    write_objectstate_public_interaction_workspace(
        workspace,
        sample_id="hot3d-clip-unit-001",
    )

    summary = objectstate_public_interaction_workspace_progress(workspace)

    assert summary["readiness"]["source_sequence_bound"] is False
    assert "source_sequence_id is missing or still TODO" in summary["hard_blockers"]
    assert summary["next_actions"] == [
        "replace TODO source_sequence_id with the real public dataset clip id"
    ]


def test_object_state_init_public_interaction_workspace_cli(tmp_path, capsys):
    workspace = tmp_path / "hot3d-clip"
    summary_path = tmp_path / "workspace-summary.json"

    assert (
        main(
            [
                "object-state",
                "init-public-interaction-route-workspace",
                str(workspace),
                "--sample-id",
                "hot3d-clip-unit-001",
                "--source-sequence-id",
                "hot3d-sequence-unit-001",
                "--object",
                "cup-001:cup:unit cup",
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA}" in stdout
    assert "candidate=hot3d-clips" in stdout
    assert "sample_id=hot3d-clip-unit-001" in stdout
    assert "object_row_count=1" in stdout
    assert "public_replay_rows_command=" in stdout
    assert summary["schema"] == OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA
    assert summary["files"]["public_interaction_readme"].endswith(
        "PUBLIC_INTERACTION_ROUTE.md"
    )


def test_object_state_audit_public_interaction_workspace_progress_cli(
    tmp_path,
    capsys,
):
    workspace = tmp_path / "hot3d-clip"
    scaffold_summary_path = workspace / "public-interaction-workspace.json"
    progress_summary_path = tmp_path / "progress-summary.json"
    scaffold = write_objectstate_public_interaction_workspace(
        workspace,
        sample_id="hot3d-clip-unit-001",
        source_sequence_id="hot3d-sequence-unit-001",
    )
    scaffold_summary_path.write_text(
        json.dumps(scaffold, indent=2) + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "object-state",
                "audit-public-interaction-workspace-progress",
                str(workspace),
                "--summary-output",
                str(progress_summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(progress_summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_PROGRESS_SCHEMA}" in stdout
    assert "progress_status=objectstate_public_interaction_workspace_progress_blocked" in stdout
    assert "source_sequence_bound=true" in stdout
    assert "evidence_chain_reviewable=false" in stdout
    assert summary["readiness"]["source_sequence_bound"] is True
