from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    objectstate_controlled_capture_summary,
    read_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
)

OBJECTSTATE_TRANSITION_DATASET_SCHEMA = (
    "objgauss-objectstate-transition-dataset-v1"
)
OBJECTSTATE_TRANSITION_ROW_SCHEMA = "objgauss-objectstate-transition-row-v1"


def objectstate_transition_dataset_from_capture_manifest(
    manifest: Mapping[str, Any],
    *,
    require_pose: bool = True,
    require_action_transition: bool = False,
) -> dict[str, Any]:
    checked_manifest = validate_objectstate_controlled_capture_manifest(manifest)
    capture_summary = objectstate_controlled_capture_summary(checked_manifest)
    actions_by_id = {
        action["action_id"]: dict(action) for action in checked_manifest["actions"]
    }
    objects_by_id = {
        item["object_id"]: dict(item) for item in checked_manifest["objects"]
    }
    object_frame_tracks = _object_frame_tracks(
        checked_manifest["frames"],
        require_pose=require_pose,
    )
    transitions: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for object_id, observations in sorted(object_frame_tracks.items()):
        if len(observations) < 2:
            continue
        transition_ids = []
        for source, target in zip(observations[:-1], observations[1:], strict=False):
            transition = _transition_row(
                sample_id=checked_manifest["sample"]["sample_id"],
                object_record=objects_by_id[object_id],
                source=source,
                target=target,
                actions_by_id=actions_by_id,
                all_actions=checked_manifest["actions"],
            )
            transitions.append(transition)
            transition_ids.append(transition["transition_id"])
        episodes.append(
            {
                "object_id": object_id,
                "category": objects_by_id[object_id]["category"],
                "observation_count": len(observations),
                "transition_count": len(transition_ids),
                "transition_ids": transition_ids,
            }
        )
    if not transitions:
        raise ValueError(
            "ObjectState transition dataset requires at least one object transition"
        )
    action_transition_count = sum(1 for item in transitions if item["has_action"])
    if require_action_transition and action_transition_count < 1:
        raise ValueError(
            "ObjectState transition dataset requires at least one action transition"
        )
    readiness = {
        "object_episode_ready": bool(episodes),
        "pose_transition_ready": all(
            "pose" in item["state_t"] and "pose" in item["state_t1"]
            for item in transitions
        ),
        "action_conditioned_transition_ready": action_transition_count > 0,
        "real_gaussian_refs_present": bool(
            capture_summary["readiness"]["real_gaussian_reconstruction_present"]
        ),
    }
    payload = {
        "schema": OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
        "kind": "objectstate_transition_dataset",
        "row_schema": OBJECTSTATE_TRANSITION_ROW_SCHEMA,
        "status": "objectstate_transition_dataset_ready",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": dict(checked_manifest["sample"]),
        "requirements": {
            "pose_required": bool(require_pose),
            "action_transition_required": bool(require_action_transition),
        },
        "readiness": readiness,
        "row_counts": {
            "objects": len(checked_manifest["objects"]),
            "object_episodes": len(episodes),
            "frames": len(checked_manifest["frames"]),
            "source_actions": len(checked_manifest["actions"]),
            "transitions": len(transitions),
            "action_conditioned_transitions": action_transition_count,
            "no_action_transitions": len(transitions) - action_transition_count,
        },
        "episodes": episodes,
        "transitions": transitions,
        "capture_summary": capture_summary,
        "claim_policy": {
            "compiles_existing_capture_ground_truth": True,
            "object_level_transition_dataset": True,
            "uses_validated_controlled_capture_manifest": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_identity": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_dynamics_model": True,
            "does_not_run_prediction_eval": True,
            "does_not_run_intervention_eval": True,
            "does_not_create_reality_rows": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "downloads_dataset": False,
            "creates_ground_truth": False,
            "infers_identity": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
            "runs_prediction_model": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "creates_replay_buffer": False,
            "uses_diffusion": False,
            "creates_reality_rows": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_transition_dataset(payload)


def write_objectstate_transition_dataset(
    capture_manifest: str | Path,
    output: str | Path,
    *,
    require_pose: bool = True,
    require_action_transition: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(capture_manifest)
    output_path = Path(output)
    manifest = read_objectstate_controlled_capture_manifest(manifest_path)
    dataset = objectstate_transition_dataset_from_capture_manifest(
        manifest,
        require_pose=require_pose,
        require_action_transition=require_action_transition,
    )
    payload = {
        **dataset,
        "source_capture_manifest": str(manifest_path),
        "output": str(output_path),
        "next_commands": {
            "review_dataset": (
                "uv run objgauss object-state compile-objectstate-transitions "
                f"{manifest_path} --output {output_path}"
            ),
        },
    }
    checked = validate_objectstate_transition_dataset(payload)
    _ensure_can_write(output_path, force=force)
    _write_json(output_path, checked)
    return checked


def validate_objectstate_transition_dataset(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("ObjectState transition dataset must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TRANSITION_DATASET_SCHEMA:
        raise ValueError(
            f"unsupported ObjectState transition dataset schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_transition_dataset":
        raise ValueError("ObjectState transition dataset kind is unsupported")
    if payload.get("row_schema") != OBJECTSTATE_TRANSITION_ROW_SCHEMA:
        raise ValueError("ObjectState transition dataset row schema is unsupported")
    if payload.get("status") != "objectstate_transition_dataset_ready":
        raise ValueError("ObjectState transition dataset status is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("ObjectState transition dataset capture schema mismatch")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping) or not sample.get("sample_id"):
        raise ValueError("ObjectState transition dataset requires sample")
    requirements = payload.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("ObjectState transition dataset requires requirements")
    for key in ("pose_required", "action_transition_required"):
        if not isinstance(requirements.get(key), bool):
            raise ValueError(f"ObjectState transition dataset missing bool {key}")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("ObjectState transition dataset requires readiness")
    for key in (
        "object_episode_ready",
        "pose_transition_ready",
        "action_conditioned_transition_ready",
        "real_gaussian_refs_present",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"ObjectState transition dataset missing readiness {key}")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("ObjectState transition dataset requires row_counts")
    transitions = _sequence(payload.get("transitions"), "transitions")
    episodes = _sequence(payload.get("episodes"), "episodes")
    if not transitions:
        raise ValueError("ObjectState transition dataset requires transition rows")
    expected_transition_count = len(transitions)
    if row_counts.get("transitions") != expected_transition_count:
        raise ValueError("ObjectState transition dataset transition count mismatch")
    action_count = sum(1 for item in transitions if bool(item.get("has_action")))
    if row_counts.get("action_conditioned_transitions") != action_count:
        raise ValueError("ObjectState transition dataset action count mismatch")
    if row_counts.get("no_action_transitions") != expected_transition_count - action_count:
        raise ValueError("ObjectState transition dataset no-action count mismatch")
    if row_counts.get("object_episodes") != len(episodes):
        raise ValueError("ObjectState transition dataset episode count mismatch")
    for transition in transitions:
        _validate_transition_row(transition)
    for episode in episodes:
        _validate_episode(episode)
    if requirements["pose_required"] and not readiness["pose_transition_ready"]:
        raise ValueError("ObjectState transition dataset required pose transitions")
    if (
        requirements["action_transition_required"]
        and not readiness["action_conditioned_transition_ready"]
    ):
        raise ValueError("ObjectState transition dataset required action transitions")
    capture_summary = payload.get("capture_summary")
    if not isinstance(capture_summary, Mapping):
        raise ValueError("ObjectState transition dataset requires capture_summary")
    if capture_summary.get("sample", {}).get("sample_id") != sample["sample_id"]:
        raise ValueError("ObjectState transition dataset sample mismatch")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("compiles_existing_capture_ground_truth")
        or not claim_policy.get("object_level_transition_dataset")
        or not claim_policy.get("uses_validated_controlled_capture_manifest")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_infer_identity")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_dynamics_model")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_run_intervention_eval")
        or not claim_policy.get("does_not_create_reality_rows")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("ObjectState transition dataset must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "ObjectState transition dataset cannot claim capture, download, GT "
            "creation, inference, reconstruction, model runs, training, replay, "
            "diffusion, reality rows, or viewer mutation"
        )
    return dict(payload)


def _object_frame_tracks(
    frames: Sequence[Mapping[str, Any]],
    *,
    require_pose: bool,
) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        for frame_object in frame["objects"]:
            if require_pose and "pose" not in frame_object:
                raise ValueError(
                    "ObjectState transition dataset requires pose for object "
                    f"{frame_object['object_id']} in frame {frame['frame_id']}"
                )
            tracks.setdefault(frame_object["object_id"], []).append(
                {
                    "frame": frame,
                    "object": frame_object,
                }
            )
    return tracks


def _transition_row(
    *,
    sample_id: str,
    object_record: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    all_actions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_frame = source["frame"]
    target_frame = target["frame"]
    object_id = object_record["object_id"]
    action_context = _action_context_for_transition(
        object_id=object_id,
        source_frame=source_frame,
        target_frame=target_frame,
        actions_by_id=actions_by_id,
        all_actions=all_actions,
    )
    transition_id = (
        f"{sample_id}:{object_id}:"
        f"{source_frame['frame_id']}->{target_frame['frame_id']}"
    )
    delta_t = float(target_frame["timestamp"]) - float(source_frame["timestamp"])
    if delta_t <= 0.0:
        raise ValueError("ObjectState transition delta_t must be positive")
    return {
        "schema": OBJECTSTATE_TRANSITION_ROW_SCHEMA,
        "transition_id": transition_id,
        "sample_id": sample_id,
        "object_id": object_id,
        "category": object_record["category"],
        "source_frame_id": source_frame["frame_id"],
        "target_frame_id": target_frame["frame_id"],
        "source_timestamp": float(source_frame["timestamp"]),
        "target_timestamp": float(target_frame["timestamp"]),
        "delta_t": delta_t,
        "state_t": _state_record(source_frame, source["object"]),
        "state_t1": _state_record(target_frame, target["object"]),
        "action_context": action_context,
        "action_ids": [item["action_id"] for item in action_context],
        "has_action": bool(action_context),
    }


def _state_record(
    frame: Mapping[str, Any],
    frame_object: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "frame_id": frame["frame_id"],
        "timestamp": float(frame["timestamp"]),
        "visible": bool(frame_object.get("visible", True)),
        "occlusion_fraction": float(frame_object.get("occlusion_fraction", 0.0)),
        "observation": dict(frame["observation"]),
    }
    if "condition" in frame:
        record["condition"] = _plain(frame["condition"])
    if "pose" in frame_object:
        record["pose"] = _plain(frame_object["pose"])
    return record


def _action_context_for_transition(
    *,
    object_id: str,
    source_frame: Mapping[str, Any],
    target_frame: Mapping[str, Any],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    all_actions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    actions: dict[str, dict[str, Any]] = {}
    frame_action_id = source_frame.get("action_id")
    if isinstance(frame_action_id, str) and frame_action_id:
        action = actions_by_id.get(frame_action_id)
        if action is not None and _action_refs_object(action, object_id):
            actions[action["action_id"]] = dict(action)
    start = float(source_frame["timestamp"])
    end = float(target_frame["timestamp"])
    for action in all_actions:
        if not _action_refs_object(action, object_id):
            continue
        if float(action["start_timestamp"]) <= end and float(action["end_timestamp"]) >= start:
            actions[action["action_id"]] = dict(action)
    return [_plain(item) for item in sorted(actions.values(), key=lambda item: item["action_id"])]


def _action_refs_object(action: Mapping[str, Any], object_id: str) -> bool:
    return action.get("object_id") == object_id or action.get("target_object_id") == object_id


def _validate_transition_row(row: Any) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("ObjectState transition row must be a mapping")
    if row.get("schema") != OBJECTSTATE_TRANSITION_ROW_SCHEMA:
        raise ValueError("ObjectState transition row schema mismatch")
    for key in (
        "transition_id",
        "sample_id",
        "object_id",
        "category",
        "source_frame_id",
        "target_frame_id",
    ):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"ObjectState transition row missing {key}")
    for key in ("source_timestamp", "target_timestamp", "delta_t"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"ObjectState transition row {key} must be numeric")
    if float(row["delta_t"]) <= 0.0:
        raise ValueError("ObjectState transition row delta_t must be positive")
    for key in ("state_t", "state_t1"):
        if not isinstance(row.get(key), Mapping):
            raise ValueError(f"ObjectState transition row requires {key}")
    if not isinstance(row.get("action_context"), list):
        raise ValueError("ObjectState transition row action_context must be a list")
    if not isinstance(row.get("action_ids"), list):
        raise ValueError("ObjectState transition row action_ids must be a list")
    if bool(row.get("has_action")) != bool(row["action_context"]):
        raise ValueError("ObjectState transition row has_action mismatch")


def _validate_episode(episode: Any) -> None:
    if not isinstance(episode, Mapping):
        raise ValueError("ObjectState transition episode must be a mapping")
    for key in ("object_id", "category"):
        if not isinstance(episode.get(key), str) or not episode[key]:
            raise ValueError(f"ObjectState transition episode missing {key}")
    for key in ("observation_count", "transition_count"):
        value = episode.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"ObjectState transition episode {key} must be positive")
    ids = episode.get("transition_ids")
    if not isinstance(ids, list) or len(ids) != episode["transition_count"]:
        raise ValueError("ObjectState transition episode transition_ids mismatch")


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"ObjectState transition dataset {name} must be a sequence")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"ObjectState transition dataset refuses to overwrite existing file: {path}"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
