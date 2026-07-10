from __future__ import annotations

import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping

OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA = (
    "objgauss-objectstate-controlled-capture-environment-v1"
)

CommandResolver = Callable[[str], str | None]
Importer = Callable[[str], Any]


def objectstate_controlled_capture_environment(
    *,
    dev_root: str | Path = "/dev",
    command_resolver: CommandResolver = shutil.which,
    importer: Importer = import_module,
) -> dict[str, Any]:
    root = Path(dev_root)
    devices = _device_summary(root)
    commands = _command_summary(command_resolver)
    python_modules = _python_module_summary(importer)
    readiness = {
        "video_device_visible": bool(devices["video_devices"]),
        "rgb_capture_tool_available": bool(
            commands["ffmpeg"]["available"]
            or python_modules["cv2"]["available"]
        ),
        "camera_inspection_tool_available": bool(commands["v4l2_ctl"]["available"]),
        "colmap_available": bool(commands["colmap"]["available"]),
        "nerfstudio_process_data_available": bool(
            commands["ns_process_data"]["available"]
        ),
        "nerfstudio_splatfacto_available": bool(
            commands["ns_train"]["available"] and commands["ns_export"]["available"]
        ),
    }
    readiness["rgb_capture_ready"] = bool(
        readiness["video_device_visible"]
        and readiness["rgb_capture_tool_available"]
    )
    readiness["gaussian_reconstruction_ready"] = bool(
        readiness["nerfstudio_splatfacto_available"]
        and (
            readiness["colmap_available"]
            or readiness["nerfstudio_process_data_available"]
        )
    )
    readiness["controlled_capture_environment_ready"] = bool(
        readiness["rgb_capture_ready"]
        and readiness["gaussian_reconstruction_ready"]
    )
    hard_blockers = _hard_blockers(readiness)
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA,
        "kind": "objectstate_controlled_capture_environment",
        "status": (
            "objectstate_controlled_capture_environment_ready"
            if readiness["controlled_capture_environment_ready"]
            else "objectstate_controlled_capture_environment_blocked"
        ),
        "dev_root": str(root),
        "devices": devices,
        "commands": commands,
        "python_modules": python_modules,
        "readiness": readiness,
        "hard_blockers": hard_blockers,
        "next_actions": _next_actions(readiness, hard_blockers),
        "claim_policy": {
            "environment_preflight_only": True,
            "may_be_sandbox_limited": True,
            "does_not_capture_video": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_training": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_frame_rows": False,
            "creates_annotation_rows": False,
            "creates_action_rows": False,
            "reconstructs_gaussians": False,
            "runs_identity_handoff": False,
            "runs_prediction_eval": False,
            "runs_intervention_eval": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_capture_environment_summary(payload)


def validate_objectstate_controlled_capture_environment_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture environment summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA:
        raise ValueError(
            "unsupported controlled capture environment schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_environment":
        raise ValueError("controlled capture environment kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_capture_environment_ready",
        "objectstate_controlled_capture_environment_blocked",
    }:
        raise ValueError("controlled capture environment status is unsupported")
    for key in ("dev_root", "devices", "commands", "python_modules", "readiness"):
        if key not in payload:
            raise ValueError(f"controlled capture environment requires {key}")
    if not isinstance(payload.get("dev_root"), str) or not payload["dev_root"]:
        raise ValueError("controlled capture environment requires dev_root")
    devices = payload["devices"]
    if not isinstance(devices, Mapping):
        raise ValueError("controlled capture environment devices must be mapping")
    for key in ("video_devices", "media_devices"):
        if not isinstance(devices.get(key), list):
            raise ValueError(f"controlled capture environment device list invalid: {key}")
    for group_name in ("commands", "python_modules"):
        group = payload.get(group_name)
        if not isinstance(group, Mapping) or not group:
            raise ValueError(f"controlled capture environment requires {group_name}")
        for name, record in group.items():
            if not isinstance(name, str) or not isinstance(record, Mapping):
                raise ValueError(
                    f"controlled capture environment invalid {group_name} record"
                )
            if not isinstance(record.get("available"), bool):
                raise ValueError(
                    f"controlled capture environment {group_name} availability invalid"
                )
    readiness = payload["readiness"]
    if not isinstance(readiness, Mapping):
        raise ValueError("controlled capture environment readiness must be mapping")
    for key in (
        "video_device_visible",
        "rgb_capture_tool_available",
        "camera_inspection_tool_available",
        "colmap_available",
        "nerfstudio_process_data_available",
        "nerfstudio_splatfacto_available",
        "rgb_capture_ready",
        "gaussian_reconstruction_ready",
        "controlled_capture_environment_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"controlled capture environment missing bool {key}")
    expected_status = (
        "objectstate_controlled_capture_environment_ready"
        if readiness["controlled_capture_environment_ready"]
        else "objectstate_controlled_capture_environment_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled capture environment status mismatch")
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("controlled capture environment hard_blockers must be list")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("controlled capture environment next_actions must be list")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("environment_preflight_only")
        or not claim_policy.get("may_be_sandbox_limited")
        or not claim_policy.get("does_not_capture_video")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_training")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled capture environment must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("creates_frame_rows")
        or non_goals.get("creates_annotation_rows")
        or non_goals.get("creates_action_rows")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("runs_identity_handoff")
        or non_goals.get("runs_prediction_eval")
        or non_goals.get("runs_intervention_eval")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled capture environment cannot claim capture, GT, rows, "
            "reconstruction, eval, training, public samples, replay, diffusion, "
            "or viewer mutation"
        )
    return dict(payload)


def _device_summary(dev_root: Path) -> dict[str, Any]:
    video_devices = _glob_names(dev_root, "video*")
    media_devices = _glob_names(dev_root, "media*")
    return {
        "dev_root_exists": bool(dev_root.exists()),
        "video_devices": video_devices,
        "media_devices": media_devices,
        "note": (
            "Sandboxed sessions may not expose host camera devices; rerun on "
            "the capture host if this list is empty."
        ),
    }


def _glob_names(root: Path, pattern: str) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path) for path in root.glob(pattern))


def _command_summary(command_resolver: CommandResolver) -> dict[str, dict[str, Any]]:
    command_map = {
        "ffmpeg": "ffmpeg",
        "v4l2_ctl": "v4l2-ctl",
        "colmap": "colmap",
        "ns_process_data": "ns-process-data",
        "ns_train": "ns-train",
        "ns_export": "ns-export",
    }
    return {
        key: _command_record(command, command_resolver)
        for key, command in command_map.items()
    }


def _command_record(
    command: str,
    command_resolver: CommandResolver,
) -> dict[str, Any]:
    path = command_resolver(command)
    return {
        "command": command,
        "available": path is not None,
        "path": path,
    }


def _python_module_summary(importer: Importer) -> dict[str, dict[str, Any]]:
    return {
        "cv2": _python_module_record("cv2", importer),
    }


def _python_module_record(module: str, importer: Importer) -> dict[str, Any]:
    try:
        imported = importer(module)
    except Exception as exc:  # noqa: BLE001 - preflight reports optional import errors.
        return {
            "module": module,
            "available": False,
            "version": None,
            "error": str(exc),
        }
    return {
        "module": module,
        "available": True,
        "version": getattr(imported, "__version__", None),
        "error": None,
    }


def _hard_blockers(readiness: Mapping[str, bool]) -> list[str]:
    blockers = []
    if not readiness["video_device_visible"]:
        blockers.append("no video device visible under dev_root")
    if not readiness["rgb_capture_tool_available"]:
        blockers.append("no RGB capture tool available: install ffmpeg or cv2")
    if not readiness["gaussian_reconstruction_ready"]:
        blockers.append(
            "Gaussian reconstruction tools not ready: need ns-train/ns-export "
            "and either colmap or ns-process-data"
        )
    return blockers


def _next_actions(
    readiness: Mapping[str, bool],
    hard_blockers: list[str],
) -> list[str]:
    if not hard_blockers:
        return [
            "create or fill outputs/captures/<sample_id> bundle",
            "capture RGB frames and reconstruct per-frame Gaussian evidence",
            "record timestamped pose/action GT rows",
        ]
    actions = []
    if not readiness["video_device_visible"]:
        actions.append("rerun preflight on the physical capture host with camera access")
    if not readiness["rgb_capture_tool_available"]:
        actions.append("install or expose ffmpeg, or provide an OpenCV cv2 environment")
    if not readiness["gaussian_reconstruction_ready"]:
        actions.append(
            "install or expose COLMAP/Nerfstudio tools for Gaussian reconstruction"
        )
    actions.append("keep any captured RGB/Gaussian files in ignored outputs/captures/")
    return actions
