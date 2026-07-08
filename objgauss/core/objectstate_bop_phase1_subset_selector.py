from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_bop_capture_adapter import (
    BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA,
    objectstate_bop_capture_adapter_summary,
    validate_objectstate_bop_capture_adapter_summary,
)

OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA = (
    "objgauss-objectstate-bop-phase1-subset-selector-v1"
)


def objectstate_bop_phase1_subset_selector(
    dataset_root: str | Path,
    *,
    dataset_id: str = "bop-ycbv",
    output_root: str | Path | None = None,
    object_category: str = "bop_objects",
    scenario: str = "bop_pose_sequence",
    fps: float = 30.0,
    license_text: str = "BOP dataset terms; verify source dataset license before redistribution",
    rgb_dir: str = "rgb",
    max_frames: int | None = None,
    frame_step: int = 1,
    identity_policy: str = BOP_IDENTITY_POLICY_SINGLE_INSTANCE_PER_OBJ_ID,
    pose_track_max_distance_m: float = DEFAULT_BOP_POSE_TRACK_MAX_DISTANCE_M,
    max_depth: int = 3,
    max_scene_candidates: int = 20,
    min_frames: int = 3,
    min_objects: int = 1,
    min_persistent_objects: int = 1,
) -> dict[str, Any]:
    root = Path(dataset_root)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if max_scene_candidates < 1:
        raise ValueError("max_scene_candidates must be >= 1")
    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    if min_objects < 1:
        raise ValueError("min_objects must be >= 1")
    if min_persistent_objects < 1:
        raise ValueError("min_persistent_objects must be >= 1")

    scene_roots = _discover_scene_roots(
        root,
        max_depth=max_depth,
        max_scene_candidates=max_scene_candidates,
    )
    candidates = [
        _candidate_summary(
            scene_root,
            dataset_root=root,
            dataset_id=dataset_id,
            output_root=output_root,
            object_category=object_category,
            scenario=scenario,
            fps=fps,
            license_text=license_text,
            rgb_dir=rgb_dir,
            max_frames=max_frames,
            frame_step=frame_step,
            identity_policy=identity_policy,
            pose_track_max_distance_m=pose_track_max_distance_m,
            min_frames=min_frames,
            min_objects=min_objects,
            min_persistent_objects=min_persistent_objects,
        )
        for scene_root in scene_roots
    ]
    recommended = next(
        (candidate for candidate in candidates if candidate["readiness"]["phase1_seed_ready"]),
        None,
    )
    readiness = {
        "dataset_root_exists": root.exists(),
        "scene_candidates_found": bool(scene_roots),
        "recommended_scene_ready": recommended is not None,
    }
    payload = {
        "schema": OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA,
        "kind": "objectstate_bop_phase1_subset_selector",
        "status": (
            "objectstate_bop_phase1_subset_selector_ready"
            if recommended
            else "objectstate_bop_phase1_subset_selector_blocked"
        ),
        "dataset_root": str(root),
        "dataset_id": dataset_id,
        "requirements": {
            "min_frames": int(min_frames),
            "min_objects": int(min_objects),
            "min_persistent_objects": int(min_persistent_objects),
            "max_depth": int(max_depth),
            "max_scene_candidates": int(max_scene_candidates),
            "max_frames": int(max_frames) if max_frames is not None else None,
            "frame_step": int(frame_step),
        },
        "row_counts": {
            "scene_candidates": len(candidates),
            "ready_candidates": sum(
                1 for candidate in candidates if candidate["readiness"]["phase1_seed_ready"]
            ),
        },
        "adapter_schema": OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA,
        "readiness": readiness,
        "recommended": _recommended_record(recommended),
        "candidates": candidates,
        "hard_blockers": _hard_blockers(root, candidates, recommended),
        "next_actions": _next_actions(root, candidates, recommended),
        "next_commands": _next_commands(recommended),
        "claim_policy": {
            "read_only_dataset_scan": True,
            "uses_bop_adapter_for_scene_validation": True,
            "does_not_download_dataset": True,
            "does_not_copy_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_condition_metadata": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_run_handoff": True,
            "does_not_train_model": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "copies_dataset": False,
            "creates_ground_truth": False,
            "infers_condition_metadata": False,
            "reconstructs_gaussians": False,
            "runs_handoff": False,
            "trains_model": False,
            "writes_public_samples": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_phase1_subset_selector_summary(payload)


def validate_objectstate_bop_phase1_subset_selector_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP Phase 1 subset selector summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA:
        raise ValueError(
            "unsupported BOP Phase 1 subset selector schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_phase1_subset_selector":
        raise ValueError("BOP Phase 1 subset selector kind is unsupported")
    if payload.get("status") not in {
        "objectstate_bop_phase1_subset_selector_ready",
        "objectstate_bop_phase1_subset_selector_blocked",
    }:
        raise ValueError("BOP Phase 1 subset selector status is unsupported")
    for key in ("dataset_root", "dataset_id"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"BOP Phase 1 subset selector requires {key}")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("BOP Phase 1 subset selector requires requirements")
    for key in (
        "min_frames",
        "min_objects",
        "min_persistent_objects",
        "max_depth",
        "max_scene_candidates",
        "frame_step",
    ):
        if not isinstance(requirements.get(key), int):
            raise ValueError(f"BOP Phase 1 subset selector requirement {key} must be int")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP Phase 1 subset selector requires row_counts")
    for key in ("scene_candidates", "ready_candidates"):
        if not isinstance(row_counts.get(key), int):
            raise ValueError(f"BOP Phase 1 subset selector row count {key} must be int")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP Phase 1 subset selector requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP Phase 1 subset selector readiness values must be bool")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("BOP Phase 1 subset selector requires candidates")
    for candidate in candidates:
        _validate_candidate(candidate)
    recommended = payload.get("recommended")
    if recommended is not None:
        _validate_recommended(recommended)
    expected_ready = recommended is not None
    if payload["status"] != (
        "objectstate_bop_phase1_subset_selector_ready"
        if expected_ready
        else "objectstate_bop_phase1_subset_selector_blocked"
    ):
        raise ValueError("BOP Phase 1 subset selector status mismatch")
    if row_counts["ready_candidates"] != sum(
        1 for candidate in candidates if candidate["readiness"]["phase1_seed_ready"]
    ):
        raise ValueError("BOP Phase 1 subset selector ready count mismatch")
    for key in ("hard_blockers", "next_actions", "next_commands"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 subset selector {key} must be string list")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("read_only_dataset_scan")
        or not claim_policy.get("uses_bop_adapter_for_scene_validation")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_copy_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_condition_metadata")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP Phase 1 subset selector must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP Phase 1 subset selector cannot claim downloads, copies, GT, "
            "condition inference, reconstruction, handoff, training, public "
            "samples, or viewer mutation"
        )
    return dict(payload)


def _discover_scene_roots(
    root: Path,
    *,
    max_depth: int,
    max_scene_candidates: int,
) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    queue: list[tuple[Path, int]] = [(root, 0)]
    scenes: list[Path] = []
    while queue and len(scenes) < max_scene_candidates:
        current, depth = queue.pop(0)
        if _is_bop_scene_root(current):
            scenes.append(current)
            continue
        if depth >= max_depth:
            continue
        try:
            children = sorted(path for path in current.iterdir() if path.is_dir())
        except OSError:
            continue
        queue.extend((child, depth + 1) for child in children)
    return scenes


def _is_bop_scene_root(path: Path) -> bool:
    return (path / "scene_gt.json").is_file() and (path / "scene_camera.json").is_file()


def _candidate_summary(
    scene_root: Path,
    *,
    dataset_root: Path,
    dataset_id: str,
    output_root: str | Path | None,
    object_category: str,
    scenario: str,
    fps: float,
    license_text: str,
    rgb_dir: str,
    max_frames: int | None,
    frame_step: int,
    identity_policy: str,
    pose_track_max_distance_m: float,
    min_frames: int,
    min_objects: int,
    min_persistent_objects: int,
) -> dict[str, Any]:
    sample_id = _sample_id(dataset_id, dataset_root, scene_root)
    try:
        adapter = objectstate_bop_capture_adapter_summary(
            scene_root,
            sample_id=sample_id,
            dataset_id=dataset_id,
            object_category=object_category,
            scenario=scenario,
            fps=fps,
            license_text=license_text,
            rgb_dir=rgb_dir,
            max_frames=max_frames,
            frame_step=frame_step,
            identity_policy=identity_policy,
            pose_track_max_distance_m=pose_track_max_distance_m,
        )
        persistent_objects = _persistent_object_count(adapter["manifest"])
        readiness = {
            "adapter_ready": adapter["readiness"]["bop_scene_adapter_ready"],
            "min_frames_met": adapter["row_counts"]["frames"] >= min_frames,
            "min_objects_met": adapter["row_counts"]["objects"] >= min_objects,
            "min_persistent_objects_met": persistent_objects >= min_persistent_objects,
            "identity_stage_ready": adapter["readiness"]["identity_stage_ready"],
            "prediction_stage_ready": adapter["readiness"]["prediction_stage_ready"],
        }
        readiness["phase1_seed_ready"] = all(readiness.values())
        issues = _candidate_issues(
            readiness,
            frames=adapter["row_counts"]["frames"],
            objects=adapter["row_counts"]["objects"],
            persistent_objects=persistent_objects,
            min_frames=min_frames,
            min_objects=min_objects,
            min_persistent_objects=min_persistent_objects,
        )
        return {
            "scene_root": str(scene_root),
            "relative_scene_root": _relative(scene_root, dataset_root),
            "sample_id": sample_id,
            "status": (
                "bop_phase1_subset_candidate_ready"
                if readiness["phase1_seed_ready"]
                else "bop_phase1_subset_candidate_blocked"
            ),
            "readiness": readiness,
            "metrics": {
                "frames": adapter["row_counts"]["frames"],
                "objects": adapter["row_counts"]["objects"],
                "annotations": adapter["row_counts"]["annotations"],
                "persistent_objects": persistent_objects,
            },
            "selected_frame_ids": adapter["selected_frame_ids"],
            "issues": issues,
            "next_commands": _candidate_next_commands(
                scene_root,
                sample_id=sample_id,
                dataset_id=dataset_id,
                output_root=output_root,
            ),
            "adapter": adapter,
        }
    except Exception as exc:  # noqa: BLE001 - selector reports blocked scenes.
        return {
            "scene_root": str(scene_root),
            "relative_scene_root": _relative(scene_root, dataset_root),
            "sample_id": sample_id,
            "status": "bop_phase1_subset_candidate_blocked",
            "readiness": {
                "adapter_ready": False,
                "min_frames_met": False,
                "min_objects_met": False,
                "min_persistent_objects_met": False,
                "identity_stage_ready": False,
                "prediction_stage_ready": False,
                "phase1_seed_ready": False,
            },
            "metrics": {
                "frames": 0,
                "objects": 0,
                "annotations": 0,
                "persistent_objects": 0,
            },
            "selected_frame_ids": [],
            "issues": [f"BOP adapter rejected scene: {exc}"],
            "next_commands": [],
            "adapter": None,
        }


def _persistent_object_count(manifest: Mapping[str, Any]) -> int:
    counts: dict[str, int] = {}
    frames = manifest.get("frames", [])
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return 0
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        objects = frame.get("objects", [])
        if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
            continue
        for item in objects:
            if isinstance(item, Mapping) and isinstance(item.get("object_id"), str):
                counts[item["object_id"]] = counts.get(item["object_id"], 0) + 1
    return sum(1 for count in counts.values() if count >= 2)


def _candidate_issues(
    readiness: Mapping[str, bool],
    *,
    frames: int,
    objects: int,
    persistent_objects: int,
    min_frames: int,
    min_objects: int,
    min_persistent_objects: int,
) -> list[str]:
    issues = []
    if not readiness["min_frames_met"]:
        issues.append(f"selected frame count {frames} is below required {min_frames}")
    if not readiness["min_objects_met"]:
        issues.append(f"object count {objects} is below required {min_objects}")
    if not readiness["min_persistent_objects_met"]:
        issues.append(
            "persistent object count "
            f"{persistent_objects} is below required {min_persistent_objects}"
        )
    if not readiness["identity_stage_ready"]:
        issues.append("adapter did not mark identity_stage_ready")
    if not readiness["prediction_stage_ready"]:
        issues.append("adapter did not mark prediction_stage_ready")
    return issues


def _recommended_record(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "scene_root": candidate["scene_root"],
        "sample_id": candidate["sample_id"],
        "metrics": candidate["metrics"],
        "selected_frame_ids": candidate["selected_frame_ids"],
        "next_commands": candidate["next_commands"],
    }


def _hard_blockers(
    root: Path,
    candidates: list[Mapping[str, Any]],
    recommended: Mapping[str, Any] | None,
) -> list[str]:
    blockers = []
    if not root.exists():
        blockers.append(f"BOP dataset root does not exist: {root}")
    elif not candidates:
        blockers.append(
            "no BOP scene roots with scene_gt.json and scene_camera.json were found"
        )
    if candidates and recommended is None:
        blockers.append("no scanned BOP scene satisfies the Phase 1 seed requirements")
    return blockers


def _next_actions(
    root: Path,
    candidates: list[Mapping[str, Any]],
    recommended: Mapping[str, Any] | None,
) -> list[str]:
    if not root.exists():
        return [
            "place or mount a local BOP dataset root before running the selector again",
        ]
    if not candidates:
        return [
            "point the selector at a BOP split or scene folder containing scene_gt.json and scene_camera.json",
        ]
    if recommended is None:
        return [
            "inspect blocked candidate issues and choose a scene with enough frames, objects, RGB files, and repeated identities",
        ]
    return [
        "export a condition CSV template for the recommended scene",
        "fill view, lighting, and camera pose metadata from the capture setup",
        "run accept-bop-capture-scene after per-frame Gaussian evidence exists",
    ]


def _next_commands(candidate: Mapping[str, Any] | None) -> list[str]:
    if candidate is None:
        return []
    commands = candidate.get("next_commands")
    if not isinstance(commands, list):
        return []
    return [str(command) for command in commands]


def _candidate_next_commands(
    scene_root: Path,
    *,
    sample_id: str,
    dataset_id: str,
    output_root: str | Path | None,
) -> list[str]:
    out = Path(output_root) if output_root else Path("outputs/captures") / sample_id
    return [
        (
            "uv run objgauss object-state init-bop-condition-sidecar "
            f"{scene_root} "
            f"--condition-csv-template-output {out / 'bop-conditions.template.csv'} "
            f"--output {out / 'bop-condition-sidecar.json'} "
            f"--summary-output {out / 'bop-condition-sidecar-summary.json'}"
        ),
        (
            "uv run objgauss object-state accept-bop-capture-scene "
            f"{scene_root} "
            f"--output {out / 'capture-manifest.json'} "
            f"--summary-output {out / 'bop-acceptance-summary.json'} "
            f"--file-audit-output {out / 'bop-file-audit.json'} "
            f"--controlled-real-output {out / 'controlled-real-seed.json'} "
            f"--sample-id {sample_id} --dataset-id {dataset_id} "
            f"--condition-sidecar {out / 'bop-condition-sidecar.json'} "
            "--require-gaussian-files"
        ),
        (
            "uv run objgauss object-state audit-bop-phase1-local-row "
            f"{scene_root} "
            f"--output-root {out} "
            f"--sample-id {sample_id} --dataset-id {dataset_id} "
            f"--condition-sidecar {out / 'bop-condition-sidecar.json'}"
        ),
    ]


def _validate_candidate(candidate: Any) -> None:
    if not isinstance(candidate, Mapping):
        raise ValueError("BOP Phase 1 subset selector candidates must map")
    for key in ("scene_root", "relative_scene_root", "sample_id", "status"):
        if not isinstance(candidate.get(key), str) or not candidate[key]:
            raise ValueError(f"BOP Phase 1 subset candidate requires {key}")
    if candidate["status"] not in {
        "bop_phase1_subset_candidate_ready",
        "bop_phase1_subset_candidate_blocked",
    }:
        raise ValueError("BOP Phase 1 subset candidate status is unsupported")
    readiness = candidate.get("readiness")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP Phase 1 subset candidate requires readiness")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP Phase 1 subset candidate readiness values must be bool")
    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("BOP Phase 1 subset candidate requires metrics")
    for key in ("frames", "objects", "annotations", "persistent_objects"):
        if not isinstance(metrics.get(key), int):
            raise ValueError(f"BOP Phase 1 subset candidate metric {key} must be int")
    if not isinstance(candidate.get("selected_frame_ids"), list):
        raise ValueError("BOP Phase 1 subset candidate requires selected_frame_ids")
    for key in ("issues", "next_commands"):
        values = candidate.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP Phase 1 subset candidate {key} must be string list")
    adapter = candidate.get("adapter")
    if adapter is not None:
        validate_objectstate_bop_capture_adapter_summary(adapter)
    expected_status = (
        "bop_phase1_subset_candidate_ready"
        if readiness["phase1_seed_ready"]
        else "bop_phase1_subset_candidate_blocked"
    )
    if candidate["status"] != expected_status:
        raise ValueError("BOP Phase 1 subset candidate status mismatch")


def _validate_recommended(recommended: Any) -> None:
    if not isinstance(recommended, Mapping):
        raise ValueError("BOP Phase 1 subset selector recommended must map")
    for key in ("scene_root", "sample_id"):
        if not isinstance(recommended.get(key), str) or not recommended[key]:
            raise ValueError(f"BOP Phase 1 subset recommended requires {key}")
    if not isinstance(recommended.get("metrics"), Mapping):
        raise ValueError("BOP Phase 1 subset recommended requires metrics")
    if not isinstance(recommended.get("selected_frame_ids"), list):
        raise ValueError("BOP Phase 1 subset recommended requires selected frames")
    commands = recommended.get("next_commands")
    if not isinstance(commands, list) or any(
        not isinstance(command, str) for command in commands
    ):
        raise ValueError("BOP Phase 1 subset recommended commands must be strings")


def _sample_id(dataset_id: str, dataset_root: Path, scene_root: Path) -> str:
    return "-".join(
        part
        for part in (
            _slug(dataset_id),
            _slug(_relative(scene_root, dataset_root)),
        )
        if part
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _slug(value: str) -> str:
    safe = "".join(char if char.isalnum() else "-" for char in str(value))
    safe = "-".join(part for part in safe.split("-") if part)
    return safe.lower() or "scene"
