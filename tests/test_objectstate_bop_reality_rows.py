from __future__ import annotations

import binascii
import json
import struct
import zlib

import numpy as np

from objgauss.cli import main
from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
)
from objgauss.core.objectstate_bop_reality_rows import (
    OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
    objectstate_bop_reality_rows_from_summary,
    objectstate_bop_reality_rows_summary,
    validate_objectstate_bop_reality_rows_summary,
)
from objgauss.core.objectstate_bop_real_evidence_bundle import (
    OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA,
    objectstate_bop_real_evidence_bundle_adapter_summary,
    validate_objectstate_bop_real_evidence_bundle_adapter_summary,
)
from objgauss.core.objectstate_bop_rgbd_baseline_local_row_handoff import (
    objectstate_bop_rgbd_baseline_local_row_handoff,
)
from objgauss.core.objectstate_real_evidence_bundle_ledger import (
    write_objectstate_real_evidence_bundle_ledger,
)
from objgauss.core.objectstate_real_evidence_bundle_ledger_audit import (
    objectstate_real_evidence_bundle_ledger_package_audit,
)
from objgauss.core.objectstate_reality_row_ledger import (
    objectstate_reality_row_ledger,
)


def test_bop_reality_rows_convert_existing_rgbd_local_row_summary(tmp_path):
    source_summary = _rgbd_local_row_summary(tmp_path)

    rows = objectstate_bop_reality_rows_from_summary(
        source_summary,
        source_summary_ref="bop-rgbd-baseline-local-row-summary.json",
    )
    summary = objectstate_bop_reality_rows_summary(
        source_summary,
        source_summary_ref="bop-rgbd-baseline-local-row-summary.json",
    )

    assert summary["schema"] == OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA
    assert validate_objectstate_bop_reality_rows_summary(summary) == summary
    assert [(row.evidence_kind, row.status) for row in rows] == [
        ("identity", "fail"),
        ("prediction", "pass"),
        ("intervention", "blocked"),
    ]
    assert {row.source_kind for row in rows} == {"public_replay"}
    assert summary["row_count"] == 3
    assert summary["pass_row_count"] == 1
    assert summary["fail_row_count"] == 1
    assert summary["blocked_row_count"] == 1
    assert summary["gate"]["status"] == "objectstate_reality_gate_fail"
    assert summary["gate"]["metrics"]["controlled_real_identity_collapse"] is True
    assert summary["claim_policy"]["does_not_claim_world_model"] is True
    assert summary["claim_policy"]["does_not_claim_intervention_gate"] is True
    assert "prediction:pass" in {
        f"{row['evidence_kind']}:{row['status']}" for row in summary["rows"]
    }
    identity_row = next(row for row in summary["rows"] if row["evidence_kind"] == "identity")
    assert identity_row["metrics"]["identity_scenario_audit_present"] is True
    assert identity_row["metrics"]["identity_scenario_metadata_ready"] is True
    assert identity_row["metrics"]["occlusion_challenge_present"] is True
    assert identity_row["metrics"]["occlusion_reappearance_track_count"] == 1.0
    assert identity_row["metrics"]["view_challenge_present"] is True
    assert identity_row["metrics"]["view_condition_count"] == 2.0
    assert identity_row["metrics"]["lighting_challenge_present"] is True
    assert identity_row["metrics"]["camera_motion_challenge_present"] is True
    intervention_row = next(
        row for row in summary["rows"] if row["evidence_kind"] == "intervention"
    )
    assert intervention_row["metrics"]["action_challenge_present"] is False
    assert summary["identity_scenario_metrics"]["occlusion_challenge_present"] is True
    assert summary["identity_scenario_metrics"]["view_challenge_present"] is True
    summary_path = tmp_path / "bop-reality-rows-summary.json"
    _write_json(summary_path, summary)
    ledger = objectstate_reality_row_ledger((summary_path,))
    experiment_challenges = {
        row["experiment"]: row["challenge_status"]
        for row in ledger["state_variable_evidence_matrix"]
    }
    assert experiment_challenges["occlusion_recovery"] == (
        "objectstate_state_variable_challenge_present"
    )
    assert experiment_challenges["view_invariance"] == (
        "objectstate_state_variable_challenge_present"
    )
    assert experiment_challenges["counterfactual_action_interface"] == (
        "objectstate_state_variable_challenge_absent"
    )
    assert "intervention" in summary["blocked_rows_markdown"]
    assert any("full ObjectState reality gate did not pass" in issue for issue in summary["issues"])


def test_bop_reality_rows_cli_writes_summary_and_blocked_rows(tmp_path, capsys):
    source_summary = _rgbd_local_row_summary(tmp_path)
    source_path = tmp_path / "bop-rgbd-baseline-local-row-summary.json"
    summary_path = tmp_path / "bop-reality-rows-summary.json"
    blocked_path = tmp_path / "bop-reality-blocked-rows.md"
    _write_json(source_path, source_summary)

    assert (
        main(
            [
                "object-state",
                "audit-bop-reality-rows",
                str(source_path),
                "--summary-output",
                str(summary_path),
                "--blocked-rows-output",
                str(blocked_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    written_summary = _read_json(summary_path)

    assert f"schema={OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA}" in stdout
    assert "gate_status=objectstate_reality_gate_fail" in stdout
    assert "row=identity:fail:" in stdout
    assert "row=prediction:pass:" in stdout
    assert "row=intervention:blocked:" in stdout
    assert validate_objectstate_bop_reality_rows_summary(written_summary) == written_summary
    assert blocked_path.read_text(encoding="utf-8").startswith("| row_id |")


def test_bop_reality_rows_enter_real_evidence_bundle_ledger(tmp_path):
    source_summary = _rgbd_local_row_summary(tmp_path)
    acceptance = source_summary["baseline_local_row_handoff"]["local_row_handoff"][
        "identity_handoff"
    ]["acceptance"]
    reality = objectstate_bop_reality_rows_summary(
        source_summary,
        source_summary_ref="bop-rgbd-baseline-local-row-summary.json",
    )

    adapter = objectstate_bop_real_evidence_bundle_adapter_summary(
        acceptance,
        reality,
        source_summary_ref="bop-reality-rows-summary.json",
    )
    bundle = adapter["bundle"]

    assert adapter["schema"] == OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA
    assert validate_objectstate_bop_real_evidence_bundle_adapter_summary(adapter) == adapter
    assert adapter["status"] == "objectstate_bop_real_evidence_bundle_adapter_ready"
    assert bundle["sample"]["source_kind"] == "public_replay"
    assert adapter["accounting_status_counts"] == {
        "pass": 1,
        "fail": 1,
        "evidence_incomplete": 1,
        "unsupported": 0,
    }
    assert [(row["evidence_kind"], row["accounting_status"]) for row in bundle["gate_accounting_rows"]] == [
        ("identity", "fail"),
        ("prediction", "pass"),
        ("intervention", "evidence_incomplete"),
    ]
    assert bundle["state_transition_rows"]
    assert not bundle["action_interval_rows"]
    prediction_row = next(
        row for row in bundle["gate_accounting_rows"] if row["evidence_kind"] == "prediction"
    )
    assert prediction_row["transition_id"] == bundle["state_transition_rows"][0]["transition_id"]
    assert "state_vs_history_error_ratio" in prediction_row["metrics"]
    intervention_row = next(
        row for row in bundle["gate_accounting_rows"] if row["evidence_kind"] == "intervention"
    )
    assert intervention_row["accounting_status"] == "evidence_incomplete"
    assert "action_id" not in intervention_row
    assert adapter["readiness"]["intervention_pass_not_created_without_action_gt"] is True

    bundle_path = tmp_path / "bop-real-evidence-bundle.json"
    ledger_root = tmp_path / "bop-real-bundle-ledger"
    _write_json(bundle_path, bundle)
    ledger = write_objectstate_real_evidence_bundle_ledger(
        (bundle_path,),
        output_root=ledger_root,
    )
    package = objectstate_real_evidence_bundle_ledger_package_audit(ledger_root)

    assert ledger["status"] == "objectstate_real_evidence_bundle_ledger_reviewable"
    assert ledger["row_counts"] == {
        "row_count": 3,
        "pass_row_count": 1,
        "fail_row_count": 1,
        "blocked_row_count": 1,
        "evidence_incomplete_row_count": 1,
        "unsupported_row_count": 0,
    }
    assert ledger["accounting_status_counts"]["all"] == adapter["accounting_status_counts"]
    assert package["phase1_acceptance_status"] == (
        "objectstate_phase1_evidence_system_acceptance_pass"
    )
    assert package["phase1_acceptance_gates"]["identity_rows_enter_accounting"] is True
    assert package["phase1_acceptance_gates"]["prediction_rows_enter_accounting"] is True
    assert package["phase1_acceptance_gates"]["intervention_rows_enter_accounting"] is True
    assert package["phase1_acceptance_gates"][
        "missing_gt_accounting_is_separate_from_fail"
    ] is True


def test_bop_real_evidence_bundle_cli_writes_bundle(tmp_path, capsys):
    source_summary = _rgbd_local_row_summary(tmp_path)
    acceptance = source_summary["baseline_local_row_handoff"]["local_row_handoff"][
        "identity_handoff"
    ]["acceptance"]
    reality = objectstate_bop_reality_rows_summary(source_summary)
    acceptance_path = tmp_path / "bop-acceptance-summary.json"
    reality_path = tmp_path / "bop-reality-rows-summary.json"
    bundle_path = tmp_path / "bop-real-evidence-bundle.json"
    summary_path = tmp_path / "bop-real-evidence-bundle-adapter.json"
    _write_json(acceptance_path, acceptance)
    _write_json(reality_path, reality)

    assert (
        main(
            [
                "object-state",
                "bop-real-evidence-bundle",
                str(acceptance_path),
                str(reality_path),
                "--bundle-output",
                str(bundle_path),
                "--summary-output",
                str(summary_path),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    adapter = _read_json(summary_path)
    bundle = _read_json(bundle_path)

    assert f"schema={OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA}" in stdout
    assert "status=objectstate_bop_real_evidence_bundle_adapter_ready" in stdout
    assert "bundle_status=objectstate_real_evidence_bundle_ready" in stdout
    assert "accounting.evidence_incomplete=1" in stdout
    assert adapter["bundle"] == bundle
    assert validate_objectstate_bop_real_evidence_bundle_adapter_summary(adapter) == adapter


def _rgbd_local_row_summary(tmp_path):
    scene_root = tmp_path / "bop-rgbd-scene"
    output_root = tmp_path / "rgbd-baseline-local-row"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    _write_bop_rgbd_scene(scene_root)
    _write_json(sidecar_path, _condition_sidecar_payload())
    return objectstate_bop_rgbd_baseline_local_row_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-rgbd-scene-000001",
        condition_sidecar=sidecar_path,
        ply_format="ascii",
        max_points_per_frame=None,
    )


def _write_bop_rgbd_scene(root) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[1000, 2000], [0, 3000]], dtype=np.uint16)
    for frame_id in range(3):
        (root / "rgb" / f"{frame_id:06d}.png").write_bytes(_png_bytes(rgb))
        (root / "depth" / f"{frame_id:06d}.png").write_bytes(_png_bytes(depth))
    scene_camera = {
        str(frame_id): {
            "cam_K": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        for frame_id in range(3)
    }
    identity_rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    scene_gt = {}
    scene_gt_info = {}
    visibility_by_frame = (1.0, 0.2, 1.0)
    for frame_id in range(3):
        scene_gt[str(frame_id)] = [
            {
                "obj_id": 1,
                "cam_R_m2c": identity_rotation,
                "cam_t_m2c": [10.0 + frame_id, 20.0, 30.0],
            },
            {
                "obj_id": 2,
                "cam_R_m2c": identity_rotation,
                "cam_t_m2c": [40.0 + frame_id, 50.0, 60.0],
            },
        ]
        scene_gt_info[str(frame_id)] = [
            {
                "bbox_obj": [10, 20, 30, 40],
                "bbox_visib": [10, 20, 30, 40],
                "px_count_all": 1000,
                "px_count_valid": 1000,
                "px_count_visib": int(1000 * visibility_by_frame[frame_id]),
                "visib_fract": visibility_by_frame[frame_id],
            },
            {
                "bbox_obj": [50, 60, 30, 40],
                "bbox_visib": [50, 60, 30, 40],
                "px_count_all": 900,
                "px_count_valid": 900,
                "px_count_visib": 900,
                "visib_fract": 1.0,
            },
        ]
    _write_json(root / "scene_camera.json", scene_camera)
    _write_json(root / "scene_gt.json", scene_gt)
    _write_json(root / "scene_gt_info.json", scene_gt_info)


def _condition_sidecar_payload():
    return {
        "schema": OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
        "kind": "objectstate_bop_capture_condition_sidecar",
        "frames": {
            "0": {
                "view_id": "front",
                "lighting_id": "bright",
                "camera_pose": {
                    "position": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "1": {
                "view_id": "front",
                "lighting_id": "dim",
                "camera_pose": {
                    "position": [0.02, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "000002": {
                "view_id": "right",
                "lighting_id": "dim",
                "camera_pose": {
                    "position": [0.04, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
        "condition_policy": {
            "sidecar_only": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_from_pixels": True,
        },
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _png_bytes(array: np.ndarray) -> bytes:
    if array.dtype == np.uint8:
        bit_depth = 8
        row_bytes = array
    elif array.dtype == np.uint16:
        bit_depth = 16
        row_bytes = array.astype(">u2", copy=False)
    else:
        raise TypeError("test PNG helper supports uint8 and uint16 only")
    if array.ndim == 2:
        height, width = array.shape
        color_type = 0
    elif array.ndim == 3 and array.shape[2] == 3:
        height, width, _channels = array.shape
        color_type = 2
    else:
        raise ValueError("test PNG helper supports grayscale or RGB arrays")
    raw = b"".join(
        b"\x00" + row_bytes[row_index].tobytes()
        for row_index in range(height)
    )
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        color_type,
        0,
        0,
        0,
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)
