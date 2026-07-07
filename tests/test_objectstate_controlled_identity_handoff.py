from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.core.objectstate_controlled_identity_handoff import (
    OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
    objectstate_controlled_identity_handoff,
    validate_objectstate_controlled_identity_handoff_summary,
)
from objgauss.core.trainable_artifact import TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA


def test_controlled_identity_handoff_runs_identity_only_reality_gate():
    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(),
        _trainable_artifact(),
        candidate_id="stable-objectstate-slots",
        artifact_refs=("outputs/controlled-real/cup-box/objectstates.json",),
        max_centroid_distance=0.05,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA
    assert summary["status"] == "objectstate_controlled_identity_handoff_pass"
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_manifest"]["evidence_rows"][0]["status"] == "pass"
    assert summary["controlled_real_manifest"]["evidence_rows"][1]["status"] == "blocked"
    assert summary["controlled_real_manifest"]["evidence_rows"][2]["status"] == "blocked"
    gate = summary["controlled_real_summary"]["gate"]
    assert gate["status"] == "objectstate_reality_gate_pass"
    assert gate["hard_blockers"] == []
    assert summary["controlled_real_summary"]["blocked_row_count"] == 2
    assert validate_objectstate_controlled_identity_handoff_summary(summary) is summary


def test_controlled_identity_handoff_surfaces_failed_identity_gate():
    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(),
        _trainable_artifact(slot_ids_by_frame=((0, 1), (1, 0), (1, 0))),
        candidate_id="fragmented-objectstate-slots",
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["controlled_real_manifest"]["evidence_rows"][0]["status"] == "fail"
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_fail"
    assert "failed_rows_absent" in summary["controlled_real_summary"]["gate"]["hard_blockers"]


def test_object_state_controlled_identity_handoff_cli_writes_artifacts(tmp_path, capsys):
    capture_path = tmp_path / "capture.json"
    artifact_path = tmp_path / "objectstates.json"
    output_dir = tmp_path / "handoff"
    capture_path.write_text(json.dumps(_capture_manifest()), encoding="utf-8")
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "controlled-identity-handoff",
                str(capture_path),
                str(artifact_path),
                "--output-dir",
                str(output_dir),
                "--candidate-id",
                "cli-objectstate-slots",
                "--max-centroid-distance",
                "0.05",
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    handoff = json.loads((output_dir / "handoff-summary.json").read_text(encoding="utf-8"))
    predictions = json.loads((output_dir / "identity-predictions.json").read_text(encoding="utf-8"))
    identity_eval = json.loads((output_dir / "identity-eval-summary.json").read_text(encoding="utf-8"))
    controlled_real = json.loads((output_dir / "controlled-real.json").read_text(encoding="utf-8"))
    controlled_real_summary = json.loads((output_dir / "controlled-real-summary.json").read_text(encoding="utf-8"))
    blocked_rows = (output_dir / "blocked-rows.md").read_text(encoding="utf-8")

    assert f"schema={OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA}" in stdout
    assert "handoff_status=objectstate_controlled_identity_handoff_pass" in stdout
    assert "identity_gate_status=objectstate_reality_gate_pass" in stdout
    assert predictions["candidate"]["candidate_id"] == "cli-objectstate-slots"
    assert identity_eval["status"] == "objectstate_controlled_identity_eval_pass"
    assert controlled_real["evidence_rows"][0]["status"] == "pass"
    assert controlled_real_summary["blocked_row_count"] == 2
    assert "prediction" in blocked_rows
    assert handoff["status"] == "objectstate_controlled_identity_handoff_pass"


def _capture_manifest():
    frames = []
    for frame_index, timestamp in enumerate((0.0, 0.033333, 0.066667)):
        frame_objects = []
        for object_id, x in (("cup-001", 0.1), ("box-001", 0.4)):
            frame_objects.append(
                {
                    "object_id": object_id,
                    "visible": True,
                    "occlusion_fraction": 0.0,
                    "pose": {
                        "position": [x + 0.01 * frame_index, 0.2, 0.3],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            )
        frames.append(
            {
                "frame_id": f"frame-{frame_index:06d}",
                "timestamp": timestamp,
                "observation": {
                    "rgb": f"rgb/{frame_index:06d}.png",
                    "gaussian": f"gaussians/{frame_index:06d}.ply",
                },
                "objects": frame_objects,
            }
        )
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "controlled-tabletop-cup-box-identity-001",
            "source_kind": "controlled_real",
            "object_category": "cup_box",
            "scenario": "cross_view_occlusion_reappearance",
            "fps": 30.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/controlled-real/cup-box-identity-001/capture.json",
                "outputs/controlled-real/cup-box-identity-001/rgb/",
                "outputs/controlled-real/cup-box-identity-001/gaussians/",
            ],
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {"object_id": "cup-001", "category": "cup", "instance_label": "blue cup"},
            {"object_id": "box-001", "category": "box", "instance_label": "red box"},
        ],
        "actions": [],
        "frames": frames,
    }


def _trainable_artifact(
    *,
    slot_ids_by_frame: tuple[tuple[int, int], ...] = ((0, 1), (0, 1), (0, 1)),
):
    object_states = []
    assignments = []
    for frame_index, slot_ids in enumerate(slot_ids_by_frame):
        cup_slot, box_slot = slot_ids
        cup_x = 0.1 + 0.01 * frame_index
        box_x = 0.4 + 0.01 * frame_index
        object_states.append(
            {
                "frame_index": frame_index,
                "states": [
                    _state(cup_slot, [cup_x, 0.2, 0.3]),
                    _state(box_slot, [box_x, 0.2, 0.3]),
                ],
                "derived_object_ids": [cup_slot, box_slot],
            }
        )
        assignments.append(
            {
                "frame_index": frame_index,
                "shape": [2, 2],
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
    return {
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "label": "fixture-trainable-objectstates",
        "source": {
            "input": "outputs/controlled-real/cup-box-identity-001/objectstates.json",
            "sample": None,
        },
        "training": {
            "schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": len(object_states),
        },
        "renderer_api": {},
        "learned_parameters": {"decoder_colors": []},
        "assignments": assignments,
        "object_states": object_states,
        "artifact_policy": {
            "git_policy": "do_not_commit_training_outputs_by_default",
        },
    }


def _state(state_id: int, centroid: list[float]):
    return {
        "id": state_id,
        "slot_mass": 1.0,
        "confidence": 0.92,
        "mass_fraction": 0.5,
        "assignment_entropy": 0.0,
        "normalized_assignment_entropy": 0.0,
        "centroid": centroid,
        "bbox": [
            [centroid[0] - 0.01, centroid[1] - 0.01, centroid[2] - 0.01],
            [centroid[0] + 0.01, centroid[1] + 0.01, centroid[2] + 0.01],
        ],
        "feature": [centroid[0], centroid[1], centroid[2]],
        "status": "active",
        "diagnostics": [],
    }
