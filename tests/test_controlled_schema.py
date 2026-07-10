from __future__ import annotations

import objgauss.core.controlled_schema as legacy_controlled_schema
import objgauss.core.objectstate_controlled_capture_actions as legacy_capture_actions
import objgauss.core.objectstate_controlled_capture_annotations as legacy_capture_annotations
import objgauss.core.objectstate_controlled_capture_files as legacy_capture_files
import objgauss.core.objectstate_controlled_capture_environment as legacy_capture_environment
import objgauss.core.objectstate_controlled_capture as legacy_capture
import objgauss.core.objectstate_controlled_capture_bundle_readiness as legacy_bundle_readiness
import objgauss.core.objectstate_controlled_capture_intervention_action_gt as legacy_action_gt
import objgauss.core.objectstate_controlled_capture_import as legacy_capture_import
import objgauss.core.objectstate_controlled_capture_frames as legacy_capture_frames
import objgauss.core.objectstate_controlled_capture_template as legacy_capture_template
import objgauss.datasets.controlled_schema as canonical_controlled_schema
import objgauss.datasets.objectstate_controlled_capture_actions as canonical_capture_actions
import objgauss.datasets.objectstate_controlled_capture_annotations as canonical_capture_annotations
import objgauss.datasets.objectstate_controlled_capture_files as canonical_capture_files
import objgauss.datasets.objectstate_controlled_capture_environment as canonical_capture_environment
import objgauss.datasets.objectstate_controlled_capture as canonical_capture
import objgauss.datasets.objectstate_controlled_capture_bundle_readiness as canonical_bundle_readiness
import objgauss.datasets.objectstate_controlled_capture_intervention_action_gt as canonical_action_gt
import objgauss.datasets.objectstate_controlled_capture_import as canonical_capture_import
import objgauss.datasets.objectstate_controlled_capture_frames as canonical_capture_frames
import objgauss.datasets.objectstate_controlled_capture_template as canonical_capture_template
from objgauss.datasets.controlled_schema import (
    OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA,
    OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA,
    objectstate_controlled_dataset_contract_summary,
    validate_objectstate_controlled_dataset_contract_summary,
)
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)


def test_core_compatibility_module_reexports_canonical_objects():
    exported_names = (
        "OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA",
        "OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA",
        "objectstate_controlled_dataset_contract_summary",
        "validate_objectstate_controlled_dataset_contract_summary",
    )

    for name in exported_names:
        assert getattr(legacy_controlled_schema, name) is getattr(
            canonical_controlled_schema, name
        )


def test_core_action_gt_compatibility_preserves_canonical_object_identity():
    assert legacy_action_gt.__all__
    for name in legacy_action_gt.__all__:
        assert getattr(legacy_action_gt, name) is getattr(canonical_action_gt, name)


def test_core_capture_compatibility_preserves_canonical_object_identity():
    assert legacy_capture.__all__
    for name in legacy_capture.__all__:
        assert getattr(legacy_capture, name) is getattr(canonical_capture, name)


def test_core_capture_template_compatibility_preserves_object_identity():
    assert legacy_capture_template.__all__
    for name in legacy_capture_template.__all__:
        assert getattr(legacy_capture_template, name) is getattr(
            canonical_capture_template, name
        )


def test_core_capture_frames_compatibility_preserves_object_identity():
    assert legacy_capture_frames.__all__
    for name in legacy_capture_frames.__all__:
        assert getattr(legacy_capture_frames, name) is getattr(
            canonical_capture_frames, name
        )


def test_core_capture_annotations_compatibility_preserves_object_identity():
    assert legacy_capture_annotations.__all__
    for name in legacy_capture_annotations.__all__:
        assert getattr(legacy_capture_annotations, name) is getattr(
            canonical_capture_annotations, name
        )


def test_core_capture_actions_compatibility_preserves_object_identity():
    assert legacy_capture_actions.__all__
    for name in legacy_capture_actions.__all__:
        assert getattr(legacy_capture_actions, name) is getattr(
            canonical_capture_actions, name
        )


def test_core_capture_files_compatibility_preserves_object_identity():
    assert legacy_capture_files.__all__
    for name in legacy_capture_files.__all__:
        assert getattr(legacy_capture_files, name) is getattr(
            canonical_capture_files, name
        )


def test_core_capture_environment_compatibility_preserves_object_identity():
    assert legacy_capture_environment.__all__
    for name in legacy_capture_environment.__all__:
        assert getattr(legacy_capture_environment, name) is getattr(
            canonical_capture_environment, name
        )


def test_core_capture_import_compatibility_preserves_object_identity():
    assert legacy_capture_import.__all__
    for name in legacy_capture_import.__all__:
        assert getattr(legacy_capture_import, name) is getattr(
            canonical_capture_import, name
        )


def test_core_bundle_readiness_compatibility_preserves_object_identity():
    assert legacy_bundle_readiness.__all__
    for name in legacy_bundle_readiness.__all__:
        assert getattr(legacy_bundle_readiness, name) is getattr(
            canonical_bundle_readiness, name
        )


def test_controlled_dataset_contract_summary_accepts_full_invariants():
    summary = objectstate_controlled_dataset_contract_summary(
        _manifest(include_pose=True, include_action=True)
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA
    assert summary["contract_schema"] == OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA
    assert summary["source_schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert summary["dataset_language"]["episode"] == {
        "sample_id": "controlled-tabletop-contract-001",
        "scene_id": "push_left_occlusion",
        "camera": "fixed-camera",
        "object_count": 1,
        "action_count": 1,
        "frame_count": 3,
    }
    assert summary["invariants"]["identity"]["ready"] is True
    assert summary["invariants"]["prediction"]["ready"] is True
    assert summary["invariants"]["causal"]["ready"] is True
    assert summary["readiness"]["controlled_dataset_contract_ready"] is True
    assert summary["metrics"]["usable_action_transition_count"] == 1
    assert summary["hard_blockers"] == []
    assert validate_objectstate_controlled_dataset_contract_summary(summary) == summary


def test_controlled_dataset_contract_keeps_missing_pose_and_action_blocked():
    summary = objectstate_controlled_dataset_contract_summary(
        _manifest(include_pose=False, include_action=False)
    )

    assert summary["invariants"]["identity"]["ready"] is True
    assert summary["invariants"]["prediction"]["ready"] is False
    assert summary["invariants"]["causal"]["ready"] is False
    assert summary["readiness"]["controlled_dataset_contract_ready"] is False
    assert "prediction invariant requires timestamped 6DoF pose tracks" in summary[
        "hard_blockers"
    ]
    assert "intervention action GT requires at least one action row" in summary[
        "hard_blockers"
    ]


def test_controlled_dataset_contract_rejects_zero_vector_as_causal_ready():
    manifest = _manifest(include_pose=True, include_action=True)
    manifest["actions"][0]["vector"] = [0.0, 0.0, 0.0]

    summary = objectstate_controlled_dataset_contract_summary(manifest)

    assert summary["invariants"]["identity"]["ready"] is True
    assert summary["invariants"]["prediction"]["ready"] is True
    assert summary["invariants"]["causal"]["ready"] is False
    assert summary["readiness"]["controlled_dataset_contract_ready"] is False
    assert "action push-left-001 requires a non-zero vector" in summary[
        "hard_blockers"
    ]


def _manifest(*, include_pose: bool, include_action: bool):
    frames = []
    for index, timestamp in enumerate((0.0, 1.0, 2.0)):
        frame_object = {
            "object_id": "cube-001",
            "visible": True,
            "occlusion_fraction": 0.5 if index == 1 else 0.0,
        }
        if include_pose:
            frame_object["pose"] = {
                "position": [float(index) * 0.1, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        frame = {
            "frame_id": f"f{index:03d}",
            "timestamp": timestamp,
            "observation": {
                "rgb": f"rgb/f{index:03d}.png",
                "gaussian": f"gaussians/f{index:03d}.ply",
            },
            "objects": [frame_object],
            "condition": {
                "view_id": "front" if index < 2 else "side",
                "lighting_id": "lighting-a",
                "camera_pose": {
                    "position": [0.05 * index, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        }
        if include_action and index == 1:
            frame["action_id"] = "push-left-001"
        frames.append(frame)
    actions = []
    if include_action:
        actions.append(
            {
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "cube-001",
                "start_timestamp": 1.0,
                "end_timestamp": 2.0,
                "actor": "scripted-hand",
                "vector": [-0.1, 0.0, 0.0],
            }
        )
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "controlled-tabletop-contract-001",
            "source_kind": "controlled_real",
            "object_category": "cube",
            "scenario": "push_left_occlusion",
            "fps": 1.0,
            "capture_device": "fixed-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/captures/controlled-tabletop-contract-001/capture.json"
            ],
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {
                "object_id": "cube-001",
                "category": "cube",
                "instance_label": "test cube",
                "dimensions_m": [0.05, 0.05, 0.05],
            }
        ],
        "actions": actions,
        "frames": frames,
    }
