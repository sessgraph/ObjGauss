from __future__ import annotations

from typing import Any, Mapping, Sequence


def objectstate_controlled_capture_intervention_action_gt_readiness(
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if manifest is None:
        return {
            "ready": False,
            "readiness": {
                "actions_present": False,
                "nonzero_action_vectors_present": False,
                "usable_action_transition_present": False,
            },
            "metrics": {
                "action_count": 0,
                "nonzero_vector_action_count": 0,
                "usable_action_transition_count": 0,
            },
            "usable_action_ids": [],
            "issues": ["capture manifest is not import-ready"],
        }
    actions = list(manifest.get("actions", ()))
    frames = list(manifest.get("frames", ()))
    issues: list[str] = []
    nonzero_action_ids: set[str] = set()
    usable_action_ids: set[str] = set()
    if not actions:
        issues.append("intervention action GT requires at least one action row")
    object_tracks = _object_pose_tracks(frames)
    for action in actions:
        action_id = str(action.get("action_id", ""))
        vector = action.get("vector")
        if not _is_nonzero_vector(vector):
            issues.append(f"action {action_id or '-'} requires a non-zero vector")
            continue
        nonzero_action_ids.add(action_id)
        refs = [str(action.get("object_id", ""))]
        target = action.get("target_object_id")
        if isinstance(target, str) and target:
            refs.append(target)
        if any(
            _action_fits_object_transition(action, object_tracks.get(object_id, ()))
            for object_id in refs
            if object_id
        ):
            usable_action_ids.add(action_id)
        else:
            issues.append(
                f"action {action_id or '-'} does not fit any referenced object transition"
            )
    readiness = {
        "actions_present": bool(actions),
        "nonzero_action_vectors_present": bool(actions)
        and len(nonzero_action_ids) == len(actions),
        "usable_action_transition_present": bool(actions)
        and len(usable_action_ids) == len(actions),
    }
    if actions and not usable_action_ids:
        issues.append("intervention action GT requires at least one usable action transition")
    payload = {
        "ready": all(readiness.values()),
        "readiness": readiness,
        "metrics": {
            "action_count": len(actions),
            "nonzero_vector_action_count": len(nonzero_action_ids),
            "usable_action_transition_count": len(usable_action_ids),
        },
        "usable_action_ids": sorted(usable_action_ids),
        "issues": _dedupe(issues),
    }
    return validate_objectstate_controlled_capture_intervention_action_gt_readiness(
        payload
    )


def validate_objectstate_controlled_capture_intervention_action_gt_readiness(
    payload: Any,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("intervention_action_gt requires mapping")
    if not isinstance(payload.get("ready"), bool):
        raise ValueError("intervention_action_gt requires ready bool")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("intervention_action_gt requires readiness")
    for key in (
        "actions_present",
        "nonzero_action_vectors_present",
        "usable_action_transition_present",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"intervention_action_gt readiness requires bool {key}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("intervention_action_gt requires metrics")
    for key in (
        "action_count",
        "nonzero_vector_action_count",
        "usable_action_transition_count",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"intervention_action_gt metrics requires int {key}")
    usable = payload.get("usable_action_ids")
    if not isinstance(usable, list) or any(not isinstance(item, str) for item in usable):
        raise ValueError("intervention_action_gt requires usable_action_ids list")
    issues = payload.get("issues")
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise ValueError("intervention_action_gt requires issues list")
    if payload["ready"] != all(bool(readiness[key]) for key in readiness):
        raise ValueError("intervention_action_gt ready must match readiness gates")
    return dict(payload)


def _object_pose_tracks(
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        timestamp = float(frame["timestamp"])
        for item in frame.get("objects", ()):
            object_id = str(item.get("object_id", ""))
            if not object_id or "pose" not in item:
                continue
            tracks.setdefault(object_id, []).append(
                {
                    "frame_id": str(frame.get("frame_id", "")),
                    "timestamp": timestamp,
                }
            )
    for observations in tracks.values():
        observations.sort(key=lambda item: item["timestamp"])
    return tracks


def _action_fits_object_transition(
    action: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> bool:
    if len(observations) < 2:
        return False
    start = float(action["start_timestamp"])
    end = float(action["end_timestamp"])
    for source, target in zip(observations[:-1], observations[1:], strict=False):
        if start >= float(source["timestamp"]) and end <= float(target["timestamp"]):
            return True
    return False


def _is_nonzero_vector(value: Any) -> bool:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        return False
    try:
        vector = [float(component) for component in value]
    except (TypeError, ValueError):
        return False
    return sum(component * component for component in vector) > 0.0


def _dedupe(items: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
