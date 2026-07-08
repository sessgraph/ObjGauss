from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    objectstate_controlled_capture_summary,
)
from objgauss.core.objectstate_controlled_capture_import import (
    objectstate_controlled_capture_manifest_from_bundle,
)
from objgauss.core.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
    validate_objectstate_controlled_intervention_candidates,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    validate_objectstate_controlled_prediction_candidates,
)

OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA = (
    "objgauss-objectstate-controlled-reality-candidate-template-v1"
)
OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA = (
    "objgauss-objectstate-controlled-reality-candidate-finalize-v1"
)
OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA = (
    "objgauss-objectstate-controlled-prediction-candidates-template-v1"
)
OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA = (
    "objgauss-objectstate-controlled-intervention-candidates-template-v1"
)

_PREDICTION_TEMPLATE_FILE = "prediction-candidates.template.json"
_INTERVENTION_TEMPLATE_FILE = "intervention-candidates.template.json"
_PREDICTION_CANDIDATES_FILE = "prediction-candidates.json"
_INTERVENTION_CANDIDATES_FILE = "intervention-candidates.json"
_README_FILE = "README.md"
_TODO_POSITION = "TODO_FILL_WITH_CANDIDATE_POSITION_XYZ"
_TODO_BASELINE = "TODO_FILL_WITH_BASELINE_POSITION_XYZ"
_TODO_PREFIX = "TODO"
_FORBIDDEN_ROW_FIELDS = {
    "target_position",
    "target_pose",
    "target_rotation",
    "target_rotation_xyzw",
    "ground_truth_position",
    "ground_truth_pose",
}


def write_objectstate_controlled_reality_candidate_templates(
    bundle_root: str | Path,
    *,
    output_dir: str | Path,
    sample_json: str | Path = "sample.json",
    objects_csv: str | Path = "objects.csv",
    frames_csv: str | Path = "frames.csv",
    annotations_csv: str | Path = "annotations.csv",
    actions_csv: str | Path | None = "actions.csv",
    candidate_id: str = "TODO_CANDIDATE_ID",
    candidate_source: str = "TODO_CANDIDATE_SOURCE",
    artifact_ref: str = "TODO_CANDIDATE_ARTIFACT_REF",
    force: bool = False,
) -> dict[str, Any]:
    root = Path(bundle_root)
    out = Path(output_dir)
    manifest = objectstate_controlled_capture_manifest_from_bundle(
        root,
        sample_json=sample_json,
        objects_csv=objects_csv,
        frames_csv=frames_csv,
        annotations_csv=annotations_csv,
        actions_csv=actions_csv,
    )
    capture_summary = objectstate_controlled_capture_summary(manifest)
    sample = manifest["sample"]
    candidate = {
        "candidate_id": str(candidate_id),
        "source": str(candidate_source),
        "artifact_refs": [str(artifact_ref)],
    }
    prediction_template = _prediction_template(
        manifest,
        candidate=candidate,
    )
    intervention_template = _intervention_template(
        manifest,
        candidate=candidate,
    )
    files = {
        "prediction_template": out / _PREDICTION_TEMPLATE_FILE,
        "intervention_template": out / _INTERVENTION_TEMPLATE_FILE,
        "readme": out / _README_FILE,
    }
    _ensure_can_write(files.values(), force=force)
    _write_json(files["prediction_template"], prediction_template)
    _write_json(files["intervention_template"], intervention_template)
    files["readme"].write_text(
        _readme_text(bundle_root=root, output_dir=out),
        encoding="utf-8",
    )
    prediction_count = len(prediction_template["predictions"])
    intervention_count = len(intervention_template["interventions"])
    readiness = {
        "prediction_template_has_rows": prediction_count > 0,
        "intervention_template_has_rows": intervention_count > 0,
        "capture_prediction_stage_ready": bool(
            capture_summary["readiness"]["prediction_stage_ready"]
        ),
        "capture_intervention_stage_ready": bool(
            capture_summary["readiness"]["intervention_stage_ready"]
        ),
    }
    issues = _template_issues(
        readiness,
        prediction_count=prediction_count,
        intervention_count=intervention_count,
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
        "kind": "objectstate_controlled_reality_candidate_template",
        "status": "objectstate_controlled_reality_candidate_template_ready",
        "capture_schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "bundle_root": str(root),
        "output_dir": str(out),
        "sample": {
            "sample_id": sample["sample_id"],
            "source_kind": sample["source_kind"],
            "scenario": sample["scenario"],
        },
        "candidate": candidate,
        "files": {key: str(value) for key, value in files.items()},
        "template_schemas": {
            "prediction": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA,
            "intervention": (
                OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA
            ),
        },
        "target_eval_schemas": {
            "prediction": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
            "intervention": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        },
        "row_counts": {
            "prediction_drafts": prediction_count,
            "intervention_drafts": intervention_count,
        },
        "readiness": readiness,
        "issues": issues,
        "next_commands": {
            "audit_full_readiness": (
                "uv run objgauss object-state "
                "audit-controlled-reality-bundle-readiness "
                f"{root} <objectstates.json> "
                f"{out / 'prediction-candidates.json'} "
                f"{out / 'intervention-candidates.json'}"
            ),
            "full_handoff": (
                "uv run objgauss object-state controlled-reality-bundle-handoff "
                f"{root} <objectstates.json> "
                f"{out / 'prediction-candidates.json'} "
                f"{out / 'intervention-candidates.json'} "
                f"--output-dir {out / 'reality-handoff'}"
            ),
        },
        "claim_policy": _claim_policy(),
        "non_goals": _non_goals(),
    }
    return validate_objectstate_controlled_reality_candidate_template_summary(payload)


def finalize_objectstate_controlled_reality_candidate_templates(
    prediction_template: str | Path,
    intervention_template: str | Path,
    *,
    output_dir: str | Path,
    bundle_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    prediction_template_path = Path(prediction_template)
    intervention_template_path = Path(intervention_template)
    out = Path(output_dir)
    prediction_payload = _read_json(prediction_template_path)
    intervention_payload = _read_json(intervention_template_path)
    prediction_candidates = _prediction_candidates_from_filled_template(
        prediction_payload
    )
    intervention_candidates = _intervention_candidates_from_filled_template(
        intervention_payload
    )
    if prediction_candidates["sample_id"] != intervention_candidates["sample_id"]:
        raise ValueError("prediction and intervention templates must use same sample_id")
    files = {
        "prediction_candidates": out / _PREDICTION_CANDIDATES_FILE,
        "intervention_candidates": out / _INTERVENTION_CANDIDATES_FILE,
    }
    _ensure_can_write(files.values(), force=force)
    _write_json(files["prediction_candidates"], prediction_candidates)
    _write_json(files["intervention_candidates"], intervention_candidates)
    bundle_ref = str(bundle_root) if bundle_root is not None else "<bundle-root>"
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA,
        "kind": "objectstate_controlled_reality_candidate_finalize",
        "status": "objectstate_controlled_reality_candidate_finalize_ready",
        "sample_id": prediction_candidates["sample_id"],
        "source_templates": {
            "prediction_template": str(prediction_template_path),
            "intervention_template": str(intervention_template_path),
        },
        "files": {key: str(value) for key, value in files.items()},
        "target_eval_schemas": {
            "prediction": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
            "intervention": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        },
        "row_counts": {
            "prediction_candidates": len(prediction_candidates["predictions"]),
            "intervention_candidates": len(
                intervention_candidates["interventions"]
            ),
        },
        "candidate_ids": {
            "prediction": prediction_candidates["candidate"]["candidate_id"],
            "intervention": intervention_candidates["candidate"]["candidate_id"],
        },
        "next_commands": {
            "audit_full_readiness": (
                "uv run objgauss object-state "
                "audit-controlled-reality-bundle-readiness "
                f"{bundle_ref} <objectstates.json> "
                f"{files['prediction_candidates']} "
                f"{files['intervention_candidates']}"
            ),
            "full_handoff": (
                "uv run objgauss object-state controlled-reality-bundle-handoff "
                f"{bundle_ref} <objectstates.json> "
                f"{files['prediction_candidates']} "
                f"{files['intervention_candidates']} "
                f"--output-dir {out / 'reality-handoff'}"
            ),
        },
        "claim_policy": _finalize_claim_policy(),
        "non_goals": _non_goals(),
    }
    return validate_objectstate_controlled_reality_candidate_finalize_summary(payload)


def validate_objectstate_controlled_reality_candidate_template_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled reality candidate template summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA:
        raise ValueError(
            "unsupported controlled reality candidate template schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_reality_candidate_template":
        raise ValueError("controlled reality candidate template kind is unsupported")
    if payload.get("status") != "objectstate_controlled_reality_candidate_template_ready":
        raise ValueError("controlled reality candidate template status is unsupported")
    if payload.get("capture_schema") != OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA:
        raise ValueError("controlled reality candidate template capture_schema is unsupported")
    for key in ("bundle_root", "output_dir"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"controlled reality candidate template requires {key}")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping) or not sample.get("sample_id"):
        raise ValueError("controlled reality candidate template requires sample")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("controlled reality candidate template requires candidate")
    for key in ("candidate_id", "source", "artifact_refs"):
        if key not in candidate:
            raise ValueError(f"controlled reality candidate missing {key}")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("controlled reality candidate template requires files")
    for key in ("prediction_template", "intervention_template", "readme"):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"controlled reality candidate template missing file {key}")
    template_schemas = payload.get("template_schemas")
    target_schemas = payload.get("target_eval_schemas")
    if not isinstance(template_schemas, Mapping) or not isinstance(target_schemas, Mapping):
        raise ValueError(
            "controlled reality candidate template requires schema maps"
        )
    if template_schemas.get("prediction") != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA:
        raise ValueError("controlled prediction template schema mismatch")
    if template_schemas.get("intervention") != OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA:
        raise ValueError("controlled intervention template schema mismatch")
    if template_schemas.get("prediction") == target_schemas.get("prediction"):
        raise ValueError("prediction template schema must differ from eval schema")
    if template_schemas.get("intervention") == target_schemas.get("intervention"):
        raise ValueError("intervention template schema must differ from eval schema")
    row_counts = payload.get("row_counts")
    readiness = payload.get("readiness")
    if not isinstance(row_counts, Mapping) or not isinstance(readiness, Mapping):
        raise ValueError(
            "controlled reality candidate template requires row_counts and readiness"
        )
    for key in ("prediction_drafts", "intervention_drafts"):
        value = row_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"controlled reality candidate row count invalid: {key}")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled reality candidate template issues must be a list")
    next_commands = payload.get("next_commands")
    if not isinstance(next_commands, Mapping):
        raise ValueError("controlled reality candidate template requires next_commands")
    for key in ("audit_full_readiness", "full_handoff"):
        if not isinstance(next_commands.get(key), str) or not next_commands[key]:
            raise ValueError(f"controlled reality candidate template missing command {key}")
    _validate_claim_policy(payload.get("claim_policy", {}))
    _validate_non_goals(payload.get("non_goals", {}))
    return dict(payload)


def validate_objectstate_controlled_reality_candidate_finalize_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled reality candidate finalize summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA:
        raise ValueError(
            "unsupported controlled reality candidate finalize schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_reality_candidate_finalize":
        raise ValueError("controlled reality candidate finalize kind is unsupported")
    if payload.get("status") != "objectstate_controlled_reality_candidate_finalize_ready":
        raise ValueError("controlled reality candidate finalize status is unsupported")
    if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"]:
        raise ValueError("controlled reality candidate finalize requires sample_id")
    for key in ("source_templates", "files", "target_eval_schemas", "row_counts"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError(f"controlled reality candidate finalize requires {key}")
    files = payload["files"]
    for key in ("prediction_candidates", "intervention_candidates"):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"controlled reality candidate finalize missing {key}")
    target_schemas = payload["target_eval_schemas"]
    if target_schemas.get("prediction") != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA:
        raise ValueError("controlled reality candidate finalize prediction schema mismatch")
    if target_schemas.get("intervention") != OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA:
        raise ValueError("controlled reality candidate finalize intervention schema mismatch")
    for key in ("prediction_candidates", "intervention_candidates"):
        value = payload["row_counts"].get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"controlled reality candidate finalize row count invalid: {key}")
    next_commands = payload.get("next_commands")
    if not isinstance(next_commands, Mapping):
        raise ValueError("controlled reality candidate finalize requires next_commands")
    for key in ("audit_full_readiness", "full_handoff"):
        if not isinstance(next_commands.get(key), str) or not next_commands[key]:
            raise ValueError(f"controlled reality candidate finalize missing command {key}")
    _validate_finalize_claim_policy(payload.get("claim_policy", {}))
    _validate_non_goals(payload.get("non_goals", {}))
    return dict(payload)


def validate_objectstate_controlled_prediction_candidates_template(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled prediction candidates template must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA:
        raise ValueError(
            "unsupported controlled prediction candidates template schema: "
            f"{payload.get('schema')}"
        )
    _validate_common_template(payload, expected_kind="prediction")
    if payload.get("target_eval_schema") != OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA:
        raise ValueError("controlled prediction template target eval schema mismatch")
    predictions = _sequence(payload.get("predictions"), "predictions")
    for item in predictions:
        if not isinstance(item, Mapping):
            raise TypeError("controlled prediction draft rows must be mappings")
        _required_string(item, "source_frame_id")
        _required_string(item, "target_frame_id")
        _required_string(item, "object_id")
        if item.get("predicted_position") != _TODO_POSITION:
            raise ValueError("controlled prediction template must keep predicted_position TODO")
        if item.get("history_baseline_position") != _TODO_BASELINE:
            raise ValueError(
                "controlled prediction template must keep history_baseline_position TODO"
            )
    return dict(payload)


def validate_objectstate_controlled_intervention_candidates_template(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled intervention candidates template must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA:
        raise ValueError(
            "unsupported controlled intervention candidates template schema: "
            f"{payload.get('schema')}"
        )
    _validate_common_template(payload, expected_kind="intervention")
    if payload.get("target_eval_schema") != OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA:
        raise ValueError("controlled intervention template target eval schema mismatch")
    interventions = _sequence(payload.get("interventions"), "interventions")
    for item in interventions:
        if not isinstance(item, Mapping):
            raise TypeError("controlled intervention draft rows must be mappings")
        _required_string(item, "source_frame_id")
        _required_string(item, "target_frame_id")
        _required_string(item, "object_id")
        _required_string(item, "action_id")
        if item.get("action_conditioned_position") != _TODO_POSITION:
            raise ValueError(
                "controlled intervention template must keep action_conditioned_position TODO"
            )
        if item.get("no_action_baseline_position") != _TODO_BASELINE:
            raise ValueError(
                "controlled intervention template must keep no_action_baseline_position TODO"
            )
    return dict(payload)


def _prediction_template(
    manifest: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for object_id, pose_frames in _pose_frames_by_object(manifest).items():
        for source, target in zip(pose_frames, pose_frames[1:]):
            rows.append(
                {
                    "source_frame_id": source["frame_id"],
                    "target_frame_id": target["frame_id"],
                    "object_id": object_id,
                    "predicted_position": _TODO_POSITION,
                    "history_baseline_position": _TODO_BASELINE,
                    "confidence": "TODO_OPTIONAL_CONFIDENCE_0_1",
                    "authoring_reference": {
                        "horizon_seconds": float(target["timestamp"])
                        - float(source["timestamp"]),
                        "source_pose_gt_available": True,
                        "target_pose_gt_available": True,
                        "target_pose_values_not_included_to_avoid_leakage": True,
                    },
                }
            )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA,
        "kind": "prediction",
        "template_status": "draft_not_valid_for_eval",
        "target_eval_schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": manifest["sample"]["sample_id"],
        "candidate": dict(candidate),
        "predictions": rows,
        "authoring_contract": {
            "fill_positions_with_model_outputs": True,
            "write_eval_schema_only_after_todos_are_replaced": True,
            "do_not_copy_target_ground_truth_into_predictions": True,
        },
        "claim_policy": _claim_policy(),
        "non_goals": _non_goals(),
    }
    return validate_objectstate_controlled_prediction_candidates_template(payload)


def _intervention_template(
    manifest: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    pose_frames = _pose_frames_by_object(manifest)
    rows = []
    skipped_actions = []
    for action in manifest["actions"]:
        object_id = action["object_id"]
        source, target = _action_bracketing_frames(
            pose_frames.get(object_id, ()),
            action=action,
        )
        if source is None or target is None:
            skipped_actions.append(
                {
                    "action_id": action["action_id"],
                    "object_id": object_id,
                    "reason": "no pose-annotated source/target frames bracket action interval",
                }
            )
            continue
        rows.append(
            {
                "source_frame_id": source["frame_id"],
                "target_frame_id": target["frame_id"],
                "object_id": object_id,
                "action_id": action["action_id"],
                "action_type": action["action_type"],
                "action_vector": list(action.get("vector", ())),
                "action_conditioned_position": _TODO_POSITION,
                "no_action_baseline_position": _TODO_BASELINE,
                "confidence": "TODO_OPTIONAL_CONFIDENCE_0_1",
                "authoring_reference": {
                    "horizon_seconds": float(target["timestamp"])
                    - float(source["timestamp"]),
                    "action_interval_seconds": [
                        float(action["start_timestamp"]),
                        float(action["end_timestamp"]),
                    ],
                    "target_pose_gt_available": True,
                    "target_pose_values_not_included_to_avoid_leakage": True,
                },
            }
        )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA,
        "kind": "intervention",
        "template_status": "draft_not_valid_for_eval",
        "target_eval_schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "sample_id": manifest["sample"]["sample_id"],
        "candidate": dict(candidate),
        "interventions": rows,
        "skipped_actions": skipped_actions,
        "authoring_contract": {
            "fill_action_conditioned_positions_with_model_outputs": True,
            "fill_no_action_baseline_positions_with_baseline_outputs": True,
            "write_eval_schema_only_after_todos_are_replaced": True,
            "do_not_copy_target_ground_truth_into_predictions": True,
        },
        "claim_policy": _claim_policy(),
        "non_goals": _non_goals(),
    }
    return validate_objectstate_controlled_intervention_candidates_template(payload)


def _prediction_candidates_from_filled_template(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_filled_template_common(
        payload,
        expected_schema=OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA,
        expected_kind="prediction",
        target_schema=OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    )
    rows = []
    for item in _sequence(payload.get("predictions"), "predictions"):
        if not isinstance(item, Mapping):
            raise TypeError("controlled prediction candidate rows must be mappings")
        _reject_forbidden_row_fields(item, "prediction")
        row: dict[str, Any] = {
            "source_frame_id": _required_string(item, "source_frame_id"),
            "target_frame_id": _required_string(item, "target_frame_id"),
            "object_id": _required_string(item, "object_id"),
            "predicted_position": _filled_vector(
                item.get("predicted_position"),
                "predicted_position",
            ),
            "history_baseline_position": _filled_vector(
                item.get("history_baseline_position"),
                "history_baseline_position",
            ),
        }
        confidence = _optional_confidence(item)
        if confidence is not None:
            row["confidence"] = confidence
        rows.append(row)
    result = {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": _required_string(payload, "sample_id"),
        "candidate": _filled_candidate(payload.get("candidate")),
        "predictions": rows,
    }
    return validate_objectstate_controlled_prediction_candidates(result)


def _intervention_candidates_from_filled_template(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_filled_template_common(
        payload,
        expected_schema=OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA,
        expected_kind="intervention",
        target_schema=OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
    )
    rows = []
    for item in _sequence(payload.get("interventions"), "interventions"):
        if not isinstance(item, Mapping):
            raise TypeError("controlled intervention candidate rows must be mappings")
        _reject_forbidden_row_fields(item, "intervention")
        row = {
            "source_frame_id": _required_string(item, "source_frame_id"),
            "target_frame_id": _required_string(item, "target_frame_id"),
            "object_id": _required_string(item, "object_id"),
            "action_id": _required_string(item, "action_id"),
            "action_conditioned_position": _filled_vector(
                item.get("action_conditioned_position"),
                "action_conditioned_position",
            ),
            "no_action_baseline_position": _filled_vector(
                item.get("no_action_baseline_position"),
                "no_action_baseline_position",
            ),
        }
        confidence = _optional_confidence(item)
        if confidence is not None:
            row["confidence"] = confidence
        rows.append(row)
    result = {
        "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "sample_id": _required_string(payload, "sample_id"),
        "candidate": _filled_candidate(payload.get("candidate")),
        "interventions": rows,
    }
    return validate_objectstate_controlled_intervention_candidates(result)


def _pose_frames_by_object(
    manifest: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], ...]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for frame in manifest["frames"]:
        for item in frame["objects"]:
            if "pose" not in item:
                continue
            result.setdefault(item["object_id"], []).append(
                {
                    "frame_id": frame["frame_id"],
                    "timestamp": float(frame["timestamp"]),
                }
            )
    return {
        object_id: tuple(sorted(rows, key=lambda item: float(item["timestamp"])))
        for object_id, rows in result.items()
    }


def _action_bracketing_frames(
    pose_frames: Sequence[Mapping[str, Any]],
    *,
    action: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    start = float(action["start_timestamp"])
    end = float(action["end_timestamp"])
    before = [
        frame for frame in pose_frames if float(frame["timestamp"]) <= start
    ]
    after = [
        frame for frame in pose_frames if float(frame["timestamp"]) >= end
    ]
    source = max(before, key=lambda item: float(item["timestamp"])) if before else None
    target = min(after, key=lambda item: float(item["timestamp"])) if after else None
    if source is not None and target is not None and source["frame_id"] == target["frame_id"]:
        return None, None
    return source, target


def _template_issues(
    readiness: Mapping[str, bool],
    *,
    prediction_count: int,
    intervention_count: int,
) -> list[str]:
    issues = []
    if not readiness["capture_prediction_stage_ready"]:
        issues.append("capture manifest is not prediction-stage ready")
    if not readiness["capture_intervention_stage_ready"]:
        issues.append("capture manifest is not intervention-stage ready")
    if prediction_count == 0:
        issues.append("no prediction draft rows were generated")
    if intervention_count == 0:
        issues.append("no intervention draft rows were generated")
    return issues


def _validate_common_template(
    payload: Mapping[str, Any],
    *,
    expected_kind: str,
) -> None:
    if payload.get("kind") != expected_kind:
        raise ValueError("controlled candidate template kind is unsupported")
    if payload.get("template_status") != "draft_not_valid_for_eval":
        raise ValueError("controlled candidate template must stay draft-only")
    if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"]:
        raise ValueError("controlled candidate template requires sample_id")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("controlled candidate template requires candidate")
    _required_string(candidate, "candidate_id")
    _required_string(candidate, "source")
    refs = candidate.get("artifact_refs")
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence) or not refs:
        raise ValueError("controlled candidate template artifact_refs must be non-empty")
    contract = payload.get("authoring_contract")
    if not isinstance(contract, Mapping) or not contract.get(
        "write_eval_schema_only_after_todos_are_replaced"
    ):
        raise ValueError("controlled candidate template must preserve authoring contract")
    _validate_claim_policy(payload.get("claim_policy", {}))
    _validate_non_goals(payload.get("non_goals", {}))


def _claim_policy() -> dict[str, bool]:
    return {
        "template_only": True,
        "draft_not_valid_for_eval": True,
        "evaluator_rejects_template_schema": True,
        "requires_real_capture_manifest": True,
        "requires_external_candidate_outputs": True,
        "does_not_create_ground_truth": True,
        "does_not_claim_prediction_pass": True,
        "does_not_claim_intervention_pass": True,
        "does_not_claim_world_model": True,
    }


def _finalize_claim_policy() -> dict[str, bool]:
    return {
        "filled_templates_required": True,
        "todo_values_rejected": True,
        "eval_schema_outputs_validated": True,
        "obvious_target_gt_leakage_rejected": True,
        "requires_external_candidate_outputs": True,
        "does_not_create_ground_truth": True,
        "does_not_run_prediction_model": True,
        "does_not_run_intervention_model": True,
        "does_not_claim_prediction_pass": True,
        "does_not_claim_intervention_pass": True,
        "does_not_claim_world_model": True,
    }


def _non_goals() -> dict[str, bool]:
    return {
        "captures_video": False,
        "creates_ground_truth": False,
        "runs_prediction_model": False,
        "runs_intervention_model": False,
        "trains_gaussian_model": False,
        "trains_dynamics_model": False,
        "writes_public_samples": False,
        "uses_replay_buffer": False,
        "uses_diffusion": False,
        "mutates_viewer_defaults": False,
    }


def _validate_claim_policy(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("controlled candidate template requires claim_policy")
    expected = _claim_policy()
    if any(value.get(key) is not expected_value for key, expected_value in expected.items()):
        raise ValueError("controlled candidate template must preserve claim policy")


def _validate_finalize_claim_policy(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("controlled candidate finalize requires claim_policy")
    expected = _finalize_claim_policy()
    if any(value.get(key) is not expected_value for key, expected_value in expected.items()):
        raise ValueError("controlled candidate finalize must preserve claim policy")


def _validate_non_goals(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("controlled candidate template requires non_goals")
    if any(bool(item) for item in value.values()):
        raise ValueError(
            "controlled candidate template cannot claim capture, GT, model runs, "
            "training, public samples, replay, diffusion, or viewer mutation"
        )
    expected = set(_non_goals())
    if set(value) != expected:
        raise ValueError("controlled candidate template non_goals keys changed")


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _validate_filled_template_common(
    payload: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_kind: str,
    target_schema: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("filled controlled candidate template must be a mapping")
    if payload.get("schema") != expected_schema:
        raise ValueError(
            "filled controlled candidate template has unsupported schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != expected_kind:
        raise ValueError("filled controlled candidate template kind is unsupported")
    if payload.get("template_status") != "draft_not_valid_for_eval":
        raise ValueError("filled controlled candidate template must originate from draft template")
    if payload.get("target_eval_schema") != target_schema:
        raise ValueError("filled controlled candidate template target eval schema mismatch")
    contract = payload.get("authoring_contract")
    if not isinstance(contract, Mapping) or not contract.get(
        "write_eval_schema_only_after_todos_are_replaced"
    ):
        raise ValueError("filled controlled candidate template must preserve authoring contract")
    _validate_claim_policy(payload.get("claim_policy", {}))
    _validate_non_goals(payload.get("non_goals", {}))


def _filled_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("filled controlled candidate metadata must be a mapping")
    candidate_id = _non_todo_string(value, "candidate_id")
    source = _non_todo_string(value, "source")
    refs = value.get("artifact_refs")
    if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence) or not refs:
        raise ValueError("filled controlled candidate artifact_refs must be non-empty")
    artifact_refs = []
    for item in refs:
        if not isinstance(item, str) or not item or _is_todo_string(item):
            raise ValueError("filled controlled candidate artifact_refs cannot contain TODO")
        artifact_refs.append(item)
    return {
        "candidate_id": candidate_id,
        "source": source,
        "artifact_refs": artifact_refs,
    }


def _non_todo_string(value: Mapping[str, Any], key: str) -> str:
    result = _required_string(value, key)
    if _is_todo_string(result):
        raise ValueError(f"{key} must be replaced before finalizing candidates")
    return result


def _filled_vector(value: Any, name: str) -> list[float]:
    if _is_todo_string(value):
        raise ValueError(f"{name} must be replaced before finalizing candidates")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a numeric length-3 sequence")
    if len(value) != 3:
        raise ValueError(f"{name} must have length 3")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain numeric values")
        result.append(float(item))
    return result


def _optional_confidence(value: Mapping[str, Any]) -> float | None:
    if "confidence" not in value:
        return None
    confidence = value["confidence"]
    if _is_todo_string(confidence):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    result = float(confidence)
    if result < 0.0 or result > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return result


def _reject_forbidden_row_fields(value: Mapping[str, Any], kind: str) -> None:
    forbidden = sorted(_FORBIDDEN_ROW_FIELDS.intersection(value))
    if forbidden:
        raise ValueError(
            f"controlled {kind} candidate row contains forbidden GT leakage fields: "
            + ", ".join(forbidden)
        )


def _is_todo_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper().startswith(_TODO_PREFIX)


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _ensure_can_write(paths: Sequence[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "controlled reality candidate template refuses to overwrite existing "
            "files: "
            + ", ".join(str(path) for path in existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _readme_text(*, bundle_root: Path, output_dir: Path) -> str:
    return f"""# ObjGauss Controlled Reality Candidate Templates

This directory contains draft JSON templates for the controlled real prediction
and intervention gates. The templates are intentionally not valid evaluator
inputs. Fill model outputs into separate files named:

- prediction-candidates.json
- intervention-candidates.json

Do not rename these `.template.json` files into evaluator inputs without
replacing every TODO value, changing the schema to the target eval schema, and
removing authoring-only fields.

Validation commands:

```bash
uv run objgauss object-state audit-controlled-reality-bundle-readiness {bundle_root} <objectstates.json> {output_dir / 'prediction-candidates.json'} {output_dir / 'intervention-candidates.json'}
uv run objgauss object-state controlled-reality-bundle-handoff {bundle_root} <objectstates.json> {output_dir / 'prediction-candidates.json'} {output_dir / 'intervention-candidates.json'} --output-dir {output_dir / 'reality-handoff'}
```

These templates do not create ground truth, run prediction or intervention
models, train Gaussian or dynamics models, write public samples, use a replay
buffer, use diffusion, or mutate viewer defaults.
"""
