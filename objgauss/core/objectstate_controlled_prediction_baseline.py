from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    read_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    validate_objectstate_controlled_prediction_candidates,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA,
    finalize_objectstate_controlled_prediction_candidate_template,
    validate_objectstate_controlled_prediction_candidate_finalize_summary,
    validate_objectstate_controlled_prediction_candidates_template,
)

OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA = (
    "objgauss-objectstate-controlled-prediction-baseline-candidates-v1"
)

_FILLED_TEMPLATE_FILE = "prediction-candidates.baseline-filled.template.json"
_PREDICTION_CANDIDATES_FILE = "prediction-candidates.json"
_PREDICTION_FINALIZE_SUMMARY_FILE = "prediction-finalize-summary.json"


def write_objectstate_controlled_prediction_baseline_candidates(
    capture_manifest: str | Path,
    prediction_template: str | Path,
    *,
    output_dir: str | Path,
    policy: str = "constant_velocity",
    candidate_id: str = "controlled-prediction-baseline-constant-velocity",
    candidate_source: str | None = None,
    artifact_ref: str = "generated-controlled-prediction-baseline",
    confidence: float = 0.5,
    force: bool = False,
) -> dict[str, Any]:
    capture_path = Path(capture_manifest)
    template_path = Path(prediction_template)
    out = Path(output_dir)
    checked_policy = _validate_policy(policy)
    checked_confidence = _confidence(confidence)
    manifest = read_objectstate_controlled_capture_manifest(capture_path)
    checked_manifest = validate_objectstate_controlled_capture_manifest(manifest)
    template = _read_json(template_path)
    checked_template = validate_objectstate_controlled_prediction_candidates_template(
        template
    )
    if checked_template["sample_id"] != checked_manifest["sample"]["sample_id"]:
        raise ValueError("prediction template sample_id must match capture manifest")
    candidate = {
        "candidate_id": _required_string(candidate_id, "candidate_id"),
        "source": _required_string(
            str(candidate_source)
            if candidate_source is not None
            else f"controlled_prediction_baseline:{checked_policy}",
            "candidate_source",
        ),
        "artifact_refs": [_required_string(artifact_ref, "artifact_ref")],
    }
    filled_template, row_records = _filled_template(
        checked_manifest,
        checked_template,
        candidate=candidate,
        policy=checked_policy,
        confidence=checked_confidence,
    )
    files = {
        "filled_prediction_template": out / _FILLED_TEMPLATE_FILE,
        "prediction_candidates": out / _PREDICTION_CANDIDATES_FILE,
        "prediction_finalize_summary": out / _PREDICTION_FINALIZE_SUMMARY_FILE,
    }
    _ensure_can_write(files.values(), force=force)
    _write_json(files["filled_prediction_template"], filled_template)
    finalize_summary = finalize_objectstate_controlled_prediction_candidate_template(
        files["filled_prediction_template"],
        output_dir=out,
        capture_manifest=capture_path,
        force=force,
    )
    _write_json(files["prediction_finalize_summary"], finalize_summary)
    prediction_candidates = _read_json(files["prediction_candidates"])
    validate_objectstate_controlled_prediction_candidates(prediction_candidates)
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA,
        "kind": "objectstate_controlled_prediction_baseline_candidates",
        "status": "objectstate_controlled_prediction_baseline_candidates_ready",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "template_schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA,
        "finalize_schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA,
        "target_eval_schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "capture_manifest": str(capture_path),
        "source_template": str(template_path),
        "output_dir": str(out),
        "sample_id": checked_template["sample_id"],
        "candidate": candidate,
        "policy": {
            "name": checked_policy,
            "history_baseline": "hold_source_pose",
            "prediction": (
                "source_pose_plus_previous_velocity"
                if checked_policy == "constant_velocity"
                else "hold_source_pose"
            ),
            "fallback_without_previous_pose": "hold_source_pose",
            "uses_target_timestamp": True,
            "uses_target_pose_values": False,
        },
        "row_counts": {
            "prediction_candidates": len(row_records),
            "constant_velocity_rows": sum(
                1 for item in row_records if item["prediction_mode"] == "constant_velocity"
            ),
            "hold_rows": sum(1 for item in row_records if item["prediction_mode"] == "hold"),
        },
        "files": {key: str(value) for key, value in files.items()},
        "prediction_finalize_summary": finalize_summary,
        "row_records": row_records,
        "next_commands": {
            "eval_prediction": (
                "uv run objgauss object-state eval-controlled-prediction "
                f"{capture_path} {files['prediction_candidates']} "
                f"--summary-output {out / 'prediction-eval-summary.json'} "
                f"--controlled-real-output {out / 'controlled-real-prediction.json'}"
            ),
            "audit_prediction_evidence_package": (
                "uv run objgauss object-state "
                f"audit-controlled-prediction-evidence-package {capture_path.parent} "
                f"--candidate-dir {out}"
            ),
        },
        "claim_policy": {
            "baseline_candidate_generator": True,
            "uses_source_and_prior_pose_only": True,
            "uses_target_timestamp_only": True,
            "does_not_read_target_pose_values": True,
            "finalizer_validates_eval_schema": True,
            "does_not_create_ground_truth": True,
            "does_not_run_prediction_eval": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_learned_model": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
            "runs_prediction_model": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_prediction_baseline_summary(payload)


def validate_objectstate_controlled_prediction_baseline_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled prediction baseline summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA:
        raise ValueError(
            "unsupported controlled prediction baseline schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_prediction_baseline_candidates":
        raise ValueError("controlled prediction baseline kind is unsupported")
    if payload.get("status") != "objectstate_controlled_prediction_baseline_candidates_ready":
        raise ValueError("controlled prediction baseline status is unsupported")
    for key in (
        "capture_schema",
        "template_schema",
        "finalize_schema",
        "target_eval_schema",
        "capture_manifest",
        "source_template",
        "output_dir",
        "sample_id",
    ):
        if key not in payload:
            raise ValueError(f"controlled prediction baseline requires {key}")
    if payload["capture_schema"] != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled prediction baseline capture schema mismatch")
    if payload["template_schema"] != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA:
        raise ValueError("controlled prediction baseline template schema mismatch")
    if payload["finalize_schema"] != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA:
        raise ValueError("controlled prediction baseline finalize schema mismatch")
    if payload["target_eval_schema"] != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA:
        raise ValueError("controlled prediction baseline eval schema mismatch")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("controlled prediction baseline requires candidate")
    _required_string(candidate.get("candidate_id"), "candidate_id")
    _required_string(candidate.get("source"), "source")
    refs = candidate.get("artifact_refs")
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence) or not refs:
        raise ValueError("controlled prediction baseline artifact_refs must be non-empty")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("controlled prediction baseline requires policy")
    if _validate_policy(policy.get("name")) != policy.get("name"):
        raise ValueError("controlled prediction baseline policy name mismatch")
    if policy.get("uses_target_pose_values") is not False:
        raise ValueError("controlled prediction baseline cannot use target pose values")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("controlled prediction baseline requires row_counts")
    count = row_counts.get("prediction_candidates")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("controlled prediction baseline requires prediction rows")
    constant_velocity_rows = row_counts.get("constant_velocity_rows")
    hold_rows = row_counts.get("hold_rows")
    if (
        isinstance(constant_velocity_rows, bool)
        or not isinstance(constant_velocity_rows, int)
        or constant_velocity_rows < 0
        or isinstance(hold_rows, bool)
        or not isinstance(hold_rows, int)
        or hold_rows < 0
        or constant_velocity_rows + hold_rows != count
    ):
        raise ValueError("controlled prediction baseline row counts are inconsistent")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("controlled prediction baseline requires files")
    for key in (
        "filled_prediction_template",
        "prediction_candidates",
        "prediction_finalize_summary",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"controlled prediction baseline missing file {key}")
    validate_objectstate_controlled_prediction_candidate_finalize_summary(
        payload.get("prediction_finalize_summary")
    )
    rows = payload.get("row_records")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) != count:
        raise ValueError("controlled prediction baseline row_records mismatch")
    for row in rows:
        _validate_row_record(row)
    next_commands = payload.get("next_commands")
    if (
        not isinstance(next_commands, Mapping)
        or not isinstance(next_commands.get("eval_prediction"), str)
        or not next_commands["eval_prediction"]
        or not isinstance(next_commands.get("audit_prediction_evidence_package"), str)
        or not next_commands["audit_prediction_evidence_package"]
    ):
        raise ValueError("controlled prediction baseline requires next commands")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("baseline_candidate_generator")
        or not claim_policy.get("uses_source_and_prior_pose_only")
        or not claim_policy.get("uses_target_timestamp_only")
        or not claim_policy.get("does_not_read_target_pose_values")
        or not claim_policy.get("finalizer_validates_eval_schema")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_learned_model")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled prediction baseline must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(item) for item in non_goals.values()):
        raise ValueError(
            "controlled prediction baseline cannot claim capture, GT, reconstruction, "
            "models, training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _filled_template(
    capture_manifest: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    policy: str,
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = _frame_metadata(capture_manifest)
    histories = _object_pose_histories(capture_manifest)
    rows = []
    records = []
    for item in template["predictions"]:
        source_frame_id = _required_string(item.get("source_frame_id"), "source_frame_id")
        target_frame_id = _required_string(item.get("target_frame_id"), "target_frame_id")
        object_id = _required_string(item.get("object_id"), "object_id")
        source = _pose_at(histories, object_id, source_frame_id)
        previous = _previous_pose(histories, object_id, source["timestamp"])
        if source_frame_id not in frames:
            raise ValueError(f"unknown source frame: {source_frame_id}")
        if target_frame_id not in frames:
            raise ValueError(f"unknown target frame: {target_frame_id}")
        target_timestamp = frames[target_frame_id]["timestamp"]
        horizon = target_timestamp - source["timestamp"]
        if horizon <= 0:
            raise ValueError("prediction target timestamp must be after source timestamp")
        predicted, mode = _predict_position(
            source,
            previous,
            horizon_seconds=horizon,
            policy=policy,
        )
        row = {
            "source_frame_id": source_frame_id,
            "target_frame_id": target_frame_id,
            "object_id": object_id,
            "predicted_position": predicted,
            "history_baseline_position": list(source["position"]),
            "confidence": confidence,
            "authoring_reference": {
                "generated_by": "controlled_prediction_baseline",
                "policy": policy,
                "uses_target_timestamp": True,
                "target_pose_values_not_read": True,
            },
        }
        rows.append(row)
        records.append(
            {
                "source_frame_id": source_frame_id,
                "target_frame_id": target_frame_id,
                "object_id": object_id,
                "horizon_seconds": horizon,
                "prediction_mode": mode,
                "previous_frame_id": previous["frame_id"] if previous else None,
            }
        )
    filled = dict(template)
    filled["candidate"] = dict(candidate)
    filled["predictions"] = rows
    return filled, records


def _predict_position(
    source: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    horizon_seconds: float,
    policy: str,
) -> tuple[list[float], str]:
    source_position = list(source["position"])
    if policy == "hold" or previous is None:
        return source_position, "hold"
    dt = source["timestamp"] - previous["timestamp"]
    if dt <= 0:
        return source_position, "hold"
    previous_position = previous["position"]
    return [
        float(source_position[index])
        + (
            (float(source_position[index]) - float(previous_position[index]))
            / dt
            * horizon_seconds
        )
        for index in range(3)
    ], "constant_velocity"


def _frame_metadata(capture_manifest: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        frame["frame_id"]: {"timestamp": float(frame["timestamp"])}
        for frame in capture_manifest["frames"]
    }


def _object_pose_histories(
    capture_manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = {}
    for frame in capture_manifest["frames"]:
        timestamp = float(frame["timestamp"])
        for item in frame["objects"]:
            if "pose" not in item:
                continue
            position = item["pose"].get("position")
            if (
                isinstance(position, (str, bytes))
                or not isinstance(position, Sequence)
                or len(position) != 3
            ):
                raise ValueError("object pose position must be a length-3 sequence")
            histories.setdefault(item["object_id"], []).append(
                {
                    "frame_id": frame["frame_id"],
                    "timestamp": timestamp,
                    "position": [float(value) for value in position],
                }
            )
    return {
        object_id: sorted(rows, key=lambda row: row["timestamp"])
        for object_id, rows in histories.items()
    }


def _pose_at(
    histories: Mapping[str, list[dict[str, Any]]],
    object_id: str,
    frame_id: str,
) -> dict[str, Any]:
    for item in histories.get(object_id, []):
        if item["frame_id"] == frame_id:
            return item
    raise ValueError(f"missing source pose for {object_id} at {frame_id}")


def _previous_pose(
    histories: Mapping[str, list[dict[str, Any]]],
    object_id: str,
    source_timestamp: float,
) -> dict[str, Any] | None:
    previous = [
        item for item in histories.get(object_id, []) if item["timestamp"] < source_timestamp
    ]
    return previous[-1] if previous else None


def _validate_row_record(row: Any) -> None:
    if not isinstance(row, Mapping):
        raise TypeError("controlled prediction baseline row records must be mappings")
    for key in ("source_frame_id", "target_frame_id", "object_id", "prediction_mode"):
        _required_string(row.get(key), key)
    if row["prediction_mode"] not in {"hold", "constant_velocity"}:
        raise ValueError("controlled prediction baseline row mode is unsupported")
    horizon = row.get("horizon_seconds")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, float)) or horizon <= 0:
        raise ValueError("controlled prediction baseline row horizon must be positive")


def _validate_policy(policy: Any) -> str:
    if policy not in {"hold", "constant_velocity"}:
        raise ValueError("prediction baseline policy must be hold or constant_velocity")
    return str(policy)


def _confidence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence must be numeric")
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return result


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if value.strip().upper().startswith("TODO"):
        raise ValueError(f"{name} must not be TODO")
    return value


def _ensure_can_write(paths: Sequence[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "controlled prediction baseline refuses to overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
