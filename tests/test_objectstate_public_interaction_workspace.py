from __future__ import annotations

import json
from pathlib import Path

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_public_interaction_workspace import (
    OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA,
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
