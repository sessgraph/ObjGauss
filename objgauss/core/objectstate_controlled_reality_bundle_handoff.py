from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_identity_bundle_handoff import (
    OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA,
    objectstate_controlled_identity_bundle_handoff,
    validate_objectstate_controlled_identity_bundle_handoff_summary,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    ObjectStateControlledIdentityThresholds,
)
from objgauss.core.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
    ObjectStateControlledInterventionThresholds,
    evaluate_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_eval_summary,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
    ObjectStateControlledPredictionThresholds,
    evaluate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_eval_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    objectstate_controlled_real_rows_summary,
    validate_objectstate_controlled_real_manifest,
    validate_objectstate_controlled_real_rows_summary,
)
from objgauss.core.objectstate_reality_gate import ObjectStateRealityGateThresholds

OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA = (
    "objgauss-objectstate-controlled-reality-bundle-handoff-v1"
)


def objectstate_controlled_reality_bundle_handoff(
    root: str | Path,
    model_artifact: Mapping[str, Any],
    prediction_candidates: Mapping[str, Any],
    intervention_candidates: Mapping[str, Any],
    *,
    sample_json: str | Path = "sample.json",
    objects_csv: str | Path = "objects.csv",
    frames_csv: str | Path = "frames.csv",
    annotations_csv: str | Path = "annotations.csv",
    actions_csv: str | Path | None = "actions.csv",
    candidate_id: str | None = None,
    source: str = "trainable_kernel_objectstate_nearest_pose_adapter",
    artifact_refs: Sequence[str] | None = None,
    max_centroid_distance: float | None = None,
    identity_thresholds: ObjectStateControlledIdentityThresholds | None = None,
    prediction_thresholds: ObjectStateControlledPredictionThresholds | None = None,
    intervention_thresholds: ObjectStateControlledInterventionThresholds | None = None,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 3,
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
    identity_bundle_handoff = objectstate_controlled_identity_bundle_handoff(
        root,
        model_artifact,
        sample_json=sample_json,
        objects_csv=objects_csv,
        frames_csv=frames_csv,
        annotations_csv=annotations_csv,
        actions_csv=actions_csv,
        require_prediction_ready=True,
        require_intervention_ready=True,
        candidate_id=candidate_id,
        source=source,
        artifact_refs=artifact_refs,
        max_centroid_distance=max_centroid_distance,
        identity_thresholds=identity_thresholds,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        min_real_or_public_rows=1,
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
    capture_manifest = identity_bundle_handoff["capture_bundle_acceptance"][
        "import_summary"
    ]["manifest"]
    prediction_eval = evaluate_objectstate_controlled_prediction_candidates(
        capture_manifest,
        prediction_candidates,
        thresholds=prediction_thresholds,
    )
    intervention_eval = evaluate_objectstate_controlled_intervention_candidates(
        capture_manifest,
        intervention_candidates,
        thresholds=intervention_thresholds,
    )
    controlled_real_manifest = _merged_controlled_real_manifest(
        identity_bundle_handoff["controlled_real_manifest"],
        prediction_eval["controlled_real_manifest"],
        intervention_eval["controlled_real_manifest"],
    )
    controlled_real_summary = objectstate_controlled_real_rows_summary(
        controlled_real_manifest,
        synthetic_smoke_passed=bool(synthetic_smoke_passed),
        thresholds=ObjectStateRealityGateThresholds(
            min_real_or_public_rows=int(min_real_or_public_rows),
            require_identity_pass_row=True,
            require_prediction_pass_row=True,
            require_intervention_pass_row=True,
            fail_on_failed_rows=True,
        ),
    )
    gates = {
        "capture_bundle_acceptance_pass": (
            identity_bundle_handoff["capture_bundle_acceptance"]["status"]
            == "objectstate_controlled_capture_bundle_acceptance_pass"
        ),
        "identity_handoff_pass": (
            identity_bundle_handoff["identity_handoff"]["status"]
            == "objectstate_controlled_identity_handoff_pass"
        ),
        "prediction_eval_pass": (
            prediction_eval["status"] == "objectstate_controlled_prediction_eval_pass"
        ),
        "intervention_eval_pass": (
            intervention_eval["status"]
            == "objectstate_controlled_intervention_eval_pass"
        ),
        "full_reality_gate_pass": (
            controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
        ),
    }
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA,
        "kind": "objectstate_controlled_reality_bundle_handoff",
        "status": (
            "objectstate_controlled_reality_bundle_handoff_pass"
            if all(gates.values())
            else "objectstate_controlled_reality_bundle_handoff_fail"
        ),
        "identity_bundle_handoff_schema": (
            OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA
        ),
        "prediction_eval_schema": OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
        "intervention_eval_schema": OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
        "controlled_real_manifest_schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "controlled_real_rows_schema": OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
        "root": str(Path(root)),
        "sample": dict(controlled_real_summary["sample"]),
        "candidate": {
            "identity_candidate_id": identity_bundle_handoff["candidate"][
                "candidate_id"
            ],
            "prediction_candidate_id": prediction_eval["candidate"]["candidate_id"],
            "intervention_candidate_id": intervention_eval["candidate"][
                "candidate_id"
            ],
            "identity_artifact_refs": list(
                identity_bundle_handoff["candidate"]["artifact_refs"]
            ),
            "prediction_artifact_refs": list(
                prediction_eval["candidate"]["artifact_refs"]
            ),
            "intervention_artifact_refs": list(
                intervention_eval["candidate"]["artifact_refs"]
            ),
        },
        "handoff_gates": gates,
        "issues": _handoff_issues(
            identity_bundle_handoff,
            prediction_eval,
            intervention_eval,
            controlled_real_summary,
        ),
        "identity_bundle_handoff": identity_bundle_handoff,
        "identity_handoff": identity_bundle_handoff["identity_handoff"],
        "identity_predictions": identity_bundle_handoff["identity_predictions"],
        "identity_eval": identity_bundle_handoff["identity_eval"],
        "prediction_eval": prediction_eval,
        "intervention_eval": intervention_eval,
        "controlled_real_manifest": controlled_real_manifest,
        "controlled_real_summary": controlled_real_summary,
        "handoff_contract": {
            "imports_bundle_csv": True,
            "runs_bundle_acceptance": True,
            "runs_identity_handoff": True,
            "runs_prediction_eval": True,
            "runs_intervention_eval": True,
            "uses_imported_manifest_for_all_evals": True,
            "merges_identity_prediction_intervention_rows": True,
            "writes_full_reality_gate_summary": True,
            "requires_full_reality_gate_pass": True,
        },
        "claim_policy": {
            "controlled_capture_bundle_required": True,
            "identity_prediction_intervention_candidates_required": True,
            "real_gaussian_file_audit_required": True,
            "candidate_artifact_required_for_identity": True,
            "candidate_future_predictions_required": True,
            "candidate_action_predictions_required": True,
            "does_not_create_predictions": True,
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
    return validate_objectstate_controlled_reality_bundle_handoff_summary(payload)


def validate_objectstate_controlled_reality_bundle_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled reality bundle handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported controlled reality bundle handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_reality_bundle_handoff":
        raise ValueError("controlled reality bundle handoff kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_reality_bundle_handoff_pass",
        "objectstate_controlled_reality_bundle_handoff_fail",
    }:
        raise ValueError("controlled reality bundle handoff status is unsupported")
    if (
        payload.get("identity_bundle_handoff_schema")
        != OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA
    ):
        raise ValueError(
            "controlled reality bundle handoff has unsupported identity schema"
        )
    if payload.get("prediction_eval_schema") != OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA:
        raise ValueError(
            "controlled reality bundle handoff has unsupported prediction schema"
        )
    if (
        payload.get("intervention_eval_schema")
        != OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA
    ):
        raise ValueError(
            "controlled reality bundle handoff has unsupported intervention schema"
        )
    if payload.get("controlled_real_manifest_schema") != OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA:
        raise ValueError(
            "controlled reality bundle handoff has unsupported manifest schema"
        )
    if payload.get("controlled_real_rows_schema") != OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA:
        raise ValueError(
            "controlled reality bundle handoff has unsupported rows schema"
        )
    if not isinstance(payload.get("root"), str) or not payload["root"]:
        raise ValueError("controlled reality bundle handoff requires root")

    identity_bundle_handoff = (
        validate_objectstate_controlled_identity_bundle_handoff_summary(
            payload.get("identity_bundle_handoff")
        )
    )
    prediction_eval = validate_objectstate_controlled_prediction_eval_summary(
        payload.get("prediction_eval")
    )
    intervention_eval = validate_objectstate_controlled_intervention_eval_summary(
        payload.get("intervention_eval")
    )
    controlled_real_manifest = validate_objectstate_controlled_real_manifest(
        payload.get("controlled_real_manifest")
    )
    controlled_real_summary = validate_objectstate_controlled_real_rows_summary(
        payload.get("controlled_real_summary")
    )
    sample_id = controlled_real_summary["sample"]["sample_id"]
    for child_name, child_sample_id in (
        ("identity bundle", identity_bundle_handoff["sample"]["sample_id"]),
        ("prediction eval", prediction_eval["sample"]["sample_id"]),
        ("intervention eval", intervention_eval["sample"]["sample_id"]),
        ("controlled manifest", controlled_real_manifest["sample"]["sample_id"]),
    ):
        if child_sample_id != sample_id:
            raise ValueError(
                f"controlled reality bundle handoff {child_name} sample mismatch"
            )
    if not isinstance(payload.get("sample"), Mapping):
        raise ValueError("controlled reality bundle handoff requires sample")
    if payload["sample"].get("sample_id") != sample_id:
        raise ValueError("controlled reality bundle handoff sample field mismatch")
    if not _json_equivalent(
        payload.get("identity_handoff"),
        identity_bundle_handoff["identity_handoff"],
    ):
        raise ValueError("controlled reality bundle handoff identity handoff mismatch")
    if not _json_equivalent(
        payload.get("identity_predictions"),
        identity_bundle_handoff["identity_predictions"],
    ):
        raise ValueError(
            "controlled reality bundle handoff identity predictions mismatch"
        )
    if not _json_equivalent(
        payload.get("identity_eval"),
        identity_bundle_handoff["identity_eval"],
    ):
        raise ValueError("controlled reality bundle handoff identity eval mismatch")
    expected_manifest = _merged_controlled_real_manifest(
        identity_bundle_handoff["controlled_real_manifest"],
        prediction_eval["controlled_real_manifest"],
        intervention_eval["controlled_real_manifest"],
    )
    if not _json_equivalent(controlled_real_manifest, expected_manifest):
        raise ValueError(
            "controlled reality bundle handoff manifest must merge child rows"
        )
    if controlled_real_summary["sample"]["sample_id"] != sample_id:
        raise ValueError("controlled reality bundle handoff summary sample mismatch")
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("controlled reality bundle handoff requires candidate")
    if (
        candidate.get("identity_candidate_id")
        != identity_bundle_handoff["candidate"]["candidate_id"]
        or candidate.get("prediction_candidate_id")
        != prediction_eval["candidate"]["candidate_id"]
        or candidate.get("intervention_candidate_id")
        != intervention_eval["candidate"]["candidate_id"]
    ):
        raise ValueError("controlled reality bundle handoff candidate id mismatch")
    gates = payload.get("handoff_gates")
    if not isinstance(gates, Mapping) or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise ValueError("controlled reality bundle handoff gates must be bools")
    expected_gates = {
        "capture_bundle_acceptance_pass": (
            identity_bundle_handoff["capture_bundle_acceptance"]["status"]
            == "objectstate_controlled_capture_bundle_acceptance_pass"
        ),
        "identity_handoff_pass": (
            identity_bundle_handoff["identity_handoff"]["status"]
            == "objectstate_controlled_identity_handoff_pass"
        ),
        "prediction_eval_pass": (
            prediction_eval["status"] == "objectstate_controlled_prediction_eval_pass"
        ),
        "intervention_eval_pass": (
            intervention_eval["status"]
            == "objectstate_controlled_intervention_eval_pass"
        ),
        "full_reality_gate_pass": (
            controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
        ),
    }
    if dict(gates) != expected_gates:
        raise ValueError("controlled reality bundle handoff gates must match children")
    expected_status = (
        "objectstate_controlled_reality_bundle_handoff_pass"
        if all(expected_gates.values())
        else "objectstate_controlled_reality_bundle_handoff_fail"
    )
    if payload["status"] != expected_status:
        raise ValueError("controlled reality bundle handoff status must match gates")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("controlled reality bundle handoff requires issues")
    handoff_contract = payload.get("handoff_contract", {})
    if (
        not handoff_contract.get("imports_bundle_csv")
        or not handoff_contract.get("runs_bundle_acceptance")
        or not handoff_contract.get("runs_identity_handoff")
        or not handoff_contract.get("runs_prediction_eval")
        or not handoff_contract.get("runs_intervention_eval")
        or not handoff_contract.get("uses_imported_manifest_for_all_evals")
        or not handoff_contract.get("merges_identity_prediction_intervention_rows")
        or not handoff_contract.get("writes_full_reality_gate_summary")
        or not handoff_contract.get("requires_full_reality_gate_pass")
    ):
        raise ValueError("controlled reality bundle handoff contract is incomplete")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("controlled_capture_bundle_required")
        or not claim_policy.get("identity_prediction_intervention_candidates_required")
        or not claim_policy.get("real_gaussian_file_audit_required")
        or not claim_policy.get("candidate_artifact_required_for_identity")
        or not claim_policy.get("candidate_future_predictions_required")
        or not claim_policy.get("candidate_action_predictions_required")
        or not claim_policy.get("does_not_create_predictions")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled reality bundle handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("runs_tracking_model")
        or non_goals.get("runs_prediction_model")
        or non_goals.get("runs_intervention_model")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled reality bundle handoff cannot claim capture, GT, "
            "reconstruction, model run, training, replay, diffusion, public sample, "
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


def _merged_controlled_real_manifest(
    identity_manifest: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    intervention_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checked_identity = validate_objectstate_controlled_real_manifest(identity_manifest)
    checked_prediction = validate_objectstate_controlled_real_manifest(
        prediction_manifest
    )
    checked_intervention = validate_objectstate_controlled_real_manifest(
        intervention_manifest
    )
    sample_id = checked_identity["sample"]["sample_id"]
    if (
        checked_prediction["sample"]["sample_id"] != sample_id
        or checked_intervention["sample"]["sample_id"] != sample_id
    ):
        raise ValueError("controlled reality bundle child manifests sample mismatch")
    for child in (checked_prediction, checked_intervention):
        if child["sample"] != checked_identity["sample"]:
            raise ValueError("controlled reality bundle child manifests sample mismatch")
        if child["ground_truth"] != checked_identity["ground_truth"]:
            raise ValueError(
                "controlled reality bundle child manifests ground truth mismatch"
            )
    rows_by_kind = {
        "identity": _single_evidence_row(checked_identity, "identity"),
        "prediction": _single_evidence_row(checked_prediction, "prediction"),
        "intervention": _single_evidence_row(checked_intervention, "intervention"),
    }
    return validate_objectstate_controlled_real_manifest(
        {
            "schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
            "sample": checked_identity["sample"],
            "ground_truth": checked_identity["ground_truth"],
            "evidence_rows": [
                rows_by_kind["identity"],
                rows_by_kind["prediction"],
                rows_by_kind["intervention"],
            ],
        }
    )


def _single_evidence_row(
    manifest: Mapping[str, Any],
    evidence_kind: str,
) -> dict[str, Any]:
    matches = [
        row for row in manifest["evidence_rows"] if row["evidence_kind"] == evidence_kind
    ]
    if len(matches) != 1:
        raise ValueError(
            "controlled reality bundle manifest must contain exactly one "
            f"{evidence_kind} row"
        )
    return dict(matches[0])


def _handoff_issues(
    identity_bundle_handoff: Mapping[str, Any],
    prediction_eval: Mapping[str, Any],
    intervention_eval: Mapping[str, Any],
    controlled_real_summary: Mapping[str, Any],
) -> list[str]:
    issues = []
    if (
        identity_bundle_handoff["status"]
        != "objectstate_controlled_identity_bundle_handoff_pass"
    ):
        issues.append("controlled identity bundle handoff did not pass")
        issues.extend(str(item) for item in identity_bundle_handoff["issues"])
    if prediction_eval["status"] != "objectstate_controlled_prediction_eval_pass":
        issues.append("controlled prediction eval did not pass")
        issues.extend(
            f"prediction gate failed: {key}"
            for key, value in prediction_eval["pass_gates"].items()
            if not value
        )
    if (
        intervention_eval["status"]
        != "objectstate_controlled_intervention_eval_pass"
    ):
        issues.append("controlled intervention eval did not pass")
        issues.extend(
            f"intervention gate failed: {key}"
            for key, value in intervention_eval["pass_gates"].items()
            if not value
        )
    if controlled_real_summary["gate"]["status"] != "objectstate_reality_gate_pass":
        issues.append("full controlled reality gate did not pass")
        issues.extend(
            f"full reality gate failed: {key}"
            for key in controlled_real_summary["gate"]["hard_blockers"]
        )
    return issues
