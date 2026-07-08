from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture_import import (
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA,
    objectstate_controlled_capture_bundle_acceptance_summary,
    validate_objectstate_controlled_capture_bundle_acceptance_summary,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    ObjectStateControlledIdentityThresholds,
)
from objgauss.core.objectstate_controlled_identity_handoff import (
    OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
    objectstate_controlled_identity_handoff,
    validate_objectstate_controlled_identity_handoff_summary,
)

OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA = (
    "objgauss-objectstate-controlled-identity-bundle-handoff-v1"
)


def objectstate_controlled_identity_bundle_handoff(
    root: str | Path,
    model_artifact: Mapping[str, Any],
    *,
    sample_json: str | Path = "sample.json",
    objects_csv: str | Path = "objects.csv",
    frames_csv: str | Path = "frames.csv",
    annotations_csv: str | Path = "annotations.csv",
    actions_csv: str | Path | None = "actions.csv",
    require_prediction_ready: bool = False,
    require_intervention_ready: bool = False,
    candidate_id: str | None = None,
    source: str = "trainable_kernel_objectstate_nearest_pose_adapter",
    artifact_refs: Sequence[str] | None = None,
    max_centroid_distance: float | None = None,
    identity_thresholds: ObjectStateControlledIdentityThresholds | None = None,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
    check_artifact_refs: bool = False,
    min_rgb_bytes: int = 1,
    min_gaussian_bytes: int = 1,
    require_frame_formats: bool = True,
    hash_files: bool = False,
    candidate_artifact_path: str | Path | None = None,
    min_candidate_artifact_bytes: int = 1,
    hash_candidate_artifact: bool = False,
    min_identity_scenario_frames: int = 3,
    min_occlusion_fraction: float = 0.5,
    min_view_conditions: int = 2,
    min_lighting_conditions: int = 2,
    min_camera_motion_m: float = 0.01,
) -> dict[str, Any]:
    bundle_root = Path(root)
    capture_bundle_acceptance = objectstate_controlled_capture_bundle_acceptance_summary(
        bundle_root,
        sample_json=sample_json,
        objects_csv=objects_csv,
        frames_csv=frames_csv,
        annotations_csv=annotations_csv,
        actions_csv=actions_csv,
        require_identity_ready=True,
        require_prediction_ready=require_prediction_ready,
        require_intervention_ready=require_intervention_ready,
        require_gaussian_files=True,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
    )
    resolved_artifact_refs = _identity_handoff_artifact_refs(
        artifact_refs,
        candidate_artifact_path,
    )
    identity_handoff = objectstate_controlled_identity_handoff(
        capture_bundle_acceptance["import_summary"]["manifest"],
        model_artifact,
        candidate_id=candidate_id,
        source=source,
        artifact_refs=resolved_artifact_refs,
        max_centroid_distance=max_centroid_distance,
        identity_thresholds=identity_thresholds,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        min_real_or_public_rows=int(min_real_or_public_rows),
        capture_root=bundle_root,
        check_artifact_refs=check_artifact_refs,
        min_rgb_bytes=min_rgb_bytes,
        min_gaussian_bytes=min_gaussian_bytes,
        require_frame_formats=require_frame_formats,
        hash_files=hash_files,
        candidate_artifact_path=candidate_artifact_path,
        min_candidate_artifact_bytes=min_candidate_artifact_bytes,
        hash_candidate_artifact=hash_candidate_artifact,
        min_identity_scenario_frames=min_identity_scenario_frames,
        min_occlusion_fraction=min_occlusion_fraction,
        min_view_conditions=min_view_conditions,
        min_lighting_conditions=min_lighting_conditions,
        min_camera_motion_m=min_camera_motion_m,
    )
    gates = {
        "capture_bundle_acceptance_pass": (
            capture_bundle_acceptance["status"]
            == "objectstate_controlled_capture_bundle_acceptance_pass"
        ),
        "identity_handoff_pass": (
            identity_handoff["status"]
            == "objectstate_controlled_identity_handoff_pass"
        ),
    }
    issues = []
    if not gates["capture_bundle_acceptance_pass"]:
        issues.append("controlled capture bundle acceptance did not pass")
        issues.extend(capture_bundle_acceptance["issues"])
    if not gates["identity_handoff_pass"]:
        issues.append("controlled identity handoff did not pass")
        issues.extend(identity_handoff["capture_file_audit"]["issues"])
        issues.extend(identity_handoff["identity_scenario_audit"]["issues"])
        issues.extend(identity_handoff["controlled_real_summary"]["gate"]["hard_blockers"])

    payload = {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA,
        "kind": "objectstate_controlled_identity_bundle_handoff",
        "status": (
            "objectstate_controlled_identity_bundle_handoff_pass"
            if all(gates.values())
            else "objectstate_controlled_identity_bundle_handoff_fail"
        ),
        "capture_bundle_acceptance_schema": (
            OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA
        ),
        "identity_handoff_schema": OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
        "root": str(bundle_root),
        "sample": dict(identity_handoff["sample"]),
        "candidate": dict(identity_handoff["candidate"]),
        "handoff_gates": gates,
        "issues": issues,
        "capture_bundle_acceptance": capture_bundle_acceptance,
        "identity_handoff": identity_handoff,
        "identity_predictions": identity_handoff["identity_predictions"],
        "identity_eval": identity_handoff["identity_eval"],
        "controlled_real_manifest": identity_handoff["controlled_real_manifest"],
        "controlled_real_summary": identity_handoff["controlled_real_summary"],
        "handoff_contract": {
            "imports_bundle_csv": True,
            "runs_bundle_acceptance": True,
            "runs_identity_handoff": True,
            "uses_imported_manifest_for_handoff": True,
            "requires_bundle_acceptance_pass": True,
            "requires_identity_handoff_pass": True,
            "writes_capture_manifest": True,
            "writes_identity_predictions": True,
            "writes_identity_eval": True,
            "writes_controlled_real_manifest": True,
            "prediction_and_intervention_rows_remain_visible": True,
        },
        "claim_policy": {
            "controlled_capture_bundle_required": True,
            "identity_stage_ready_required": True,
            "real_gaussian_file_audit_required": True,
            "candidate_artifact_required": True,
            "candidate_artifact_file_audit_required": True,
            "identity_handoff_required": True,
            "identity_only_stage1_gate": True,
            "does_not_claim_prediction_or_intervention": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_identity_bundle_handoff_summary(payload)


def validate_objectstate_controlled_identity_bundle_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled identity bundle handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported controlled identity bundle handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_identity_bundle_handoff":
        raise ValueError("controlled identity bundle handoff kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_identity_bundle_handoff_pass",
        "objectstate_controlled_identity_bundle_handoff_fail",
    }:
        raise ValueError("controlled identity bundle handoff status is unsupported")
    if (
        payload.get("capture_bundle_acceptance_schema")
        != OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA
    ):
        raise ValueError(
            "controlled identity bundle handoff has unsupported acceptance schema"
        )
    if payload.get("identity_handoff_schema") != OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA:
        raise ValueError(
            "controlled identity bundle handoff has unsupported identity handoff schema"
        )
    if not isinstance(payload.get("root"), str) or not payload["root"]:
        raise ValueError("controlled identity bundle handoff requires root")

    capture_bundle_acceptance = (
        validate_objectstate_controlled_capture_bundle_acceptance_summary(
            payload.get("capture_bundle_acceptance")
        )
    )
    identity_handoff = validate_objectstate_controlled_identity_handoff_summary(
        payload.get("identity_handoff")
    )
    sample_id = identity_handoff["sample"]["sample_id"]
    if (
        capture_bundle_acceptance["import_summary"]["manifest"]["sample"]["sample_id"]
        != sample_id
    ):
        raise ValueError("controlled identity bundle handoff sample mismatch")
    if not isinstance(payload.get("sample"), Mapping):
        raise ValueError("controlled identity bundle handoff requires sample")
    if payload["sample"].get("sample_id") != sample_id:
        raise ValueError("controlled identity bundle handoff sample field mismatch")
    if payload.get("candidate") != identity_handoff["candidate"]:
        raise ValueError("controlled identity bundle handoff candidate mismatch")
    gates = payload.get("handoff_gates")
    if not isinstance(gates, Mapping) or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise ValueError("controlled identity bundle handoff gates must be bools")
    expected_gates = {
        "capture_bundle_acceptance_pass": (
            capture_bundle_acceptance["status"]
            == "objectstate_controlled_capture_bundle_acceptance_pass"
        ),
        "identity_handoff_pass": (
            identity_handoff["status"]
            == "objectstate_controlled_identity_handoff_pass"
        ),
    }
    if dict(gates) != expected_gates:
        raise ValueError("controlled identity bundle handoff gates must match children")
    expected_status = (
        "objectstate_controlled_identity_bundle_handoff_pass"
        if all(expected_gates.values())
        else "objectstate_controlled_identity_bundle_handoff_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled identity bundle handoff status must match gates")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled identity bundle handoff requires issues")
    if not _json_equivalent(
        payload.get("identity_predictions"),
        identity_handoff["identity_predictions"],
    ):
        raise ValueError("controlled identity bundle handoff predictions mismatch")
    if not _json_equivalent(payload.get("identity_eval"), identity_handoff["identity_eval"]):
        raise ValueError("controlled identity bundle handoff identity eval mismatch")
    if not _json_equivalent(
        payload.get("controlled_real_manifest"),
        identity_handoff["controlled_real_manifest"],
    ):
        raise ValueError(
            "controlled identity bundle handoff controlled-real manifest mismatch"
        )
    if not _json_equivalent(
        payload.get("controlled_real_summary"),
        identity_handoff["controlled_real_summary"],
    ):
        raise ValueError(
            "controlled identity bundle handoff controlled-real summary mismatch"
        )

    handoff_contract = payload.get("handoff_contract", {})
    if (
        not handoff_contract.get("imports_bundle_csv")
        or not handoff_contract.get("runs_bundle_acceptance")
        or not handoff_contract.get("runs_identity_handoff")
        or not handoff_contract.get("uses_imported_manifest_for_handoff")
        or not handoff_contract.get("requires_bundle_acceptance_pass")
        or not handoff_contract.get("requires_identity_handoff_pass")
        or not handoff_contract.get("writes_capture_manifest")
        or not handoff_contract.get("writes_identity_predictions")
        or not handoff_contract.get("writes_identity_eval")
        or not handoff_contract.get("writes_controlled_real_manifest")
        or not handoff_contract.get("prediction_and_intervention_rows_remain_visible")
    ):
        raise ValueError("controlled identity bundle handoff contract is incomplete")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("controlled_capture_bundle_required")
        or not claim_policy.get("identity_stage_ready_required")
        or not claim_policy.get("real_gaussian_file_audit_required")
        or not claim_policy.get("candidate_artifact_required")
        or not claim_policy.get("candidate_artifact_file_audit_required")
        or not claim_policy.get("identity_handoff_required")
        or not claim_policy.get("identity_only_stage1_gate")
        or not claim_policy.get("does_not_claim_prediction_or_intervention")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled identity bundle handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("runs_tracking_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled identity bundle handoff cannot claim capture, GT, "
            "reconstruction, tracking, training, replay, diffusion, public sample, "
            "or viewer mutation"
        )
    return dict(payload)


def _json_equivalent(left: Any, right: Any) -> bool:
    return _json_normalize(left) == _json_normalize(right)


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def _identity_handoff_artifact_refs(
    artifact_refs: Sequence[str] | None,
    candidate_artifact_path: str | Path | None,
) -> Sequence[str] | None:
    if artifact_refs is not None:
        if isinstance(artifact_refs, (str, bytes)) or not isinstance(
            artifact_refs,
            Sequence,
        ):
            raise TypeError("artifact_refs must be a sequence of strings")
        return tuple(str(item) for item in artifact_refs)
    if candidate_artifact_path is not None:
        return (str(candidate_artifact_path),)
    return None
