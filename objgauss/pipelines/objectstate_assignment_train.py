from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from objgauss.core.assignment_evidence import (
    ASSIGNMENT_EVIDENCE_BATCH_SCHEMA,
    AssignmentEvidenceBatch,
    validate_assignment_evidence_batch,
)
from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
    ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA,
    AssignmentSolverV2State,
    assignment_solver_v2_state_from_dict,
    train_assignment_solver_v2,
    validate_assignment_solver_v2_state,
)
from objgauss.core.features import extract_features, positions
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io import append_or_replace_property, write_ply
from objgauss.core.object_state import validate_assignment_matrix
from objgauss.pipelines.objectstate_assignment_mvp import (
    OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA,
    objectstate_assignment_mvp_summary,
    validate_objectstate_assignment_mvp_summary,
)

OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA = (
    "objgauss-objectstate-assignment-train-dataset-v1"
)
OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA = (
    "objgauss-objectstate-assignment-train-run-v1"
)

__all__ = (
    "OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA",
    "OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA",
    "objectstate_assignment_train_dataset_summary",
    "objectstate_assignment_train_smoke",
    "validate_objectstate_assignment_train_dataset_summary",
    "validate_objectstate_assignment_train_run_summary",
)


def objectstate_assignment_train_dataset_summary(
    cloud: GaussianCloud,
    target_assignment: np.ndarray,
    *,
    sample_id: str,
    source_kind: str = "synthetic",
    split: str = "train",
    license: str = "local synthetic fixture; not public release",
    gaussian_ref: str | None = None,
    target_source: str | None = None,
    evidence_features: np.ndarray | None = None,
) -> dict[str, Any]:
    target = validate_assignment_matrix(target_assignment, evidence_count=cloud.count)
    evidence = _assignment_evidence_from_cloud(
        cloud,
        target_assignment=target,
        source=f"assignment-train:{sample_id}",
        evidence_features=evidence_features,
    )
    payload = {
        "schema": OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA,
        "kind": "objectstate_assignment_train_dataset",
        "sample_id": str(sample_id),
        "source_kind": _source_kind(source_kind),
        "split": _split(split),
        "gaussian_count": cloud.count,
        "gaussian_fields": list(cloud.fields),
        "feature_dim": evidence.feature_dim,
        "position_dim": int(evidence.positions.shape[1]),
        "slots": int(target.shape[1]),
        "evidence_schema": ASSIGNMENT_EVIDENCE_BATCH_SCHEMA,
        "target_assignment_shape": list(target.shape),
        "target_object_labels": np.argmax(target, axis=1).astype(int).tolist(),
        "license": str(license),
        "claim_policy": {
            "target_assignment_is_supervision_only": True,
            "hard_object_id_is_not_primary_state": True,
            "synthetic_and_real_rows_must_remain_separate": True,
        },
    }
    if gaussian_ref is not None:
        payload["gaussian_ref"] = str(gaussian_ref)
    if target_source is not None:
        payload["target_source"] = str(target_source)
    return validate_objectstate_assignment_train_dataset_summary(payload)


def objectstate_assignment_train_smoke(
    cloud: GaussianCloud,
    target_assignment: np.ndarray,
    *,
    output_dir: str | Path,
    sample_id: str = "assignment-smoke-001",
    source_kind: str = "synthetic",
    split: str = "train",
    license: str = "local synthetic fixture; not public release",
    gaussian_ref: str | None = None,
    target_source: str | None = None,
    initial_state: AssignmentSolverV2State | None = None,
    iterations: int = 50,
    learning_rate: float = 1e-1,
    assignment_weight: float = 1.0,
    compactness_weight: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    target = validate_assignment_matrix(target_assignment, evidence_count=cloud.count)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if iterations > 600:
        raise ValueError("assignment smoke training iterations must stay <= 600")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if assignment_weight <= 0.0:
        raise ValueError("assignment_weight must be > 0")
    if compactness_weight < 0.0:
        raise ValueError("compactness_weight must be >= 0")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = objectstate_assignment_train_dataset_summary(
        cloud,
        target,
        sample_id=sample_id,
        source_kind=source_kind,
        split=split,
        license=license,
        gaussian_ref=gaussian_ref,
        target_source=target_source,
    )
    evidence = _assignment_evidence_from_cloud(
        cloud,
        target_assignment=target,
        source=f"assignment-train:{sample_id}",
    )
    if initial_state is not None:
        initial_state = validate_assignment_solver_v2_state(initial_state)
    result = train_assignment_solver_v2(
        [evidence],
        slots=target.shape[1],
        initial_state=initial_state,
        iterations=iterations,
        learning_rate=learning_rate,
        cluster_weight=compactness_weight,
        entropy_weight=0.0,
        balance_weight=0.0,
        supervised_weight=assignment_weight,
        seed=seed,
        record_every=max(1, iterations // 5),
    )
    before = objectstate_assignment_mvp_summary(
        cloud,
        result.initial_state,
        target_assignment=target,
        source=f"{sample_id}:before",
        include_assignment=True,
    )
    after = objectstate_assignment_mvp_summary(
        cloud,
        result.final_state,
        target_assignment=target,
        source=f"{sample_id}:after",
        include_assignment=True,
    )
    before_ply = output_root / "assignment-before.ply"
    after_ply = output_root / "assignment-after.ply"
    _write_assignment_visualization(
        before_ply,
        cloud,
        np.asarray(before["assignment"]["derived_object_ids"], dtype=np.int32),
        comments=("ObjGauss assignment smoke before training",),
    )
    _write_assignment_visualization(
        after_ply,
        cloud,
        np.asarray(after["assignment"]["derived_object_ids"], dtype=np.int32),
        comments=("ObjGauss assignment smoke after training",),
    )
    checkpoint_path = output_root / "assignment-solver-v2-final-state.json"
    checkpoint = result.final_state.as_dict(include_arrays=True)
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    restored = assignment_solver_v2_state_from_dict(checkpoint)
    restored_after = objectstate_assignment_mvp_summary(
        cloud,
        restored,
        target_assignment=target,
        source=f"{sample_id}:restored",
    )
    summary_path = output_root / "summary.json"
    metrics = _before_after_metrics(before, after)
    payload = {
        "schema": OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA,
        "kind": "objectstate_assignment_train_smoke",
        "status": (
            "objectstate_assignment_train_smoke_pass"
            if _smoke_pass(result, metrics, after)
            else "objectstate_assignment_train_smoke_reviewable"
        ),
        "dataset_schema": OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA,
        "mvp_schema": OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA,
        "training_schema": ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA,
        "solver_state_schema": ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
        "sample_id": str(sample_id),
        "dataset": dataset,
        "training_config": {
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "loss_weights": {
                "assignment": float(assignment_weight),
                "compactness": float(compactness_weight),
                "entropy": 0.0,
                "balance": 0.0,
                "temporal": 0.0,
                "matching": 0.0,
            },
            "seed": int(seed),
            "max_duration_policy": "short_smoke_under_10_minutes",
        },
        "loss_curve": [record.as_dict() for record in result.history],
        "loss": {
            "initial_total": float(result.initial_loss.total_loss),
            "final_total": float(result.final_loss.total_loss),
            "loss_decreased": bool(result.final_loss.total_loss < result.initial_loss.total_loss),
            "initial_assignment": float(result.initial_loss.supervised_loss),
            "final_assignment": float(result.final_loss.supervised_loss),
            "assignment_loss_decreased": bool(
                result.final_loss.supervised_loss < result.initial_loss.supervised_loss
            ),
        },
        "before_metrics": before["target_metrics"],
        "after_metrics": after["target_metrics"],
        "metric_delta": metrics,
        "before_assignment": before["assignment"],
        "after_assignment": after["assignment"],
        "before_projection": before["projection"],
        "after_projection": after["projection"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "schema": checkpoint["schema"],
            "roundtrip_ok": _roundtrip_ok(after, restored_after),
        },
        "visualization_refs": {
            "before_ply": str(before_ply),
            "after_ply": str(after_ply),
            "coloring": "argmax(A[N,K])",
        },
        "gate_handoff": {
            "objectstate_projection_ready": True,
            "identity_gate_status": "not_run",
            "prediction_gate_status": "blocked_missing_future_state_candidates",
            "causal_gate_status": "blocked_missing_action_conditioned_candidates",
        },
        "long_training_gate": {
            "allowed": bool(_smoke_pass(result, metrics, after)),
            "requires_loss_decrease": bool(result.final_loss.total_loss < result.initial_loss.total_loss),
            "requires_assignment_metric_improvement": bool(_metrics_improved(metrics, after)),
            "requires_visualization_refs": True,
            "requires_checkpoint_roundtrip": _roundtrip_ok(after, restored_after),
            "requires_gate_handoff_not_collapsed": True,
        },
        "claim_policy": {
            "short_smoke_only": True,
            "target_assignment_is_supervision_only": True,
            "assignment_matrix_is_single_source_of_truth": True,
            "does_not_claim_identity_gate_pass": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "uses_gpu": False,
            "uses_renderer_loss": False,
            "uses_transformer": False,
            "uses_slot_attention": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "uses_dynamics_model": False,
            "runs_long_training": False,
            "mutates_viewer_defaults": False,
        },
    }
    checked = validate_objectstate_assignment_train_run_summary(payload)
    summary_path.write_text(
        json.dumps(checked, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_assignment_train_run_summary(checked)


def validate_objectstate_assignment_train_dataset_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("assignment train dataset summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA:
        raise ValueError(f"unsupported assignment train dataset schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_assignment_train_dataset":
        raise ValueError("assignment train dataset kind is unsupported")
    for key in ("sample_id", "source_kind", "split", "license"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"assignment train dataset requires {key}")
    for key in ("gaussian_ref", "target_source"):
        if key in payload and (not isinstance(payload[key], str) or not payload[key]):
            raise ValueError(f"assignment train dataset {key} must be a non-empty string")
    for key in ("gaussian_count", "feature_dim", "position_dim", "slots"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"assignment train dataset requires positive int {key}")
    shape = payload.get("target_assignment_shape")
    if shape != [payload["gaussian_count"], payload["slots"]]:
        raise ValueError("target_assignment_shape must match gaussian_count and slots")
    labels = payload.get("target_object_labels")
    if not isinstance(labels, list) or len(labels) != payload["gaussian_count"]:
        raise ValueError("target_object_labels must match gaussian_count")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("target_assignment_is_supervision_only")
        or not claim_policy.get("hard_object_id_is_not_primary_state")
        or not claim_policy.get("synthetic_and_real_rows_must_remain_separate")
    ):
        raise ValueError("assignment train dataset must preserve claim policy")
    return dict(payload)


def validate_objectstate_assignment_train_run_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("assignment train run summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA:
        raise ValueError(f"unsupported assignment train run schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_assignment_train_smoke":
        raise ValueError("assignment train run kind is unsupported")
    if payload.get("status") not in {
        "objectstate_assignment_train_smoke_pass",
        "objectstate_assignment_train_smoke_reviewable",
    }:
        raise ValueError("assignment train run status is unsupported")
    validate_objectstate_assignment_train_dataset_summary(payload.get("dataset"))
    if payload.get("training_schema") != ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA:
        raise ValueError("assignment train run has unsupported training_schema")
    if payload.get("solver_state_schema") != ASSIGNMENT_SOLVER_V2_STATE_SCHEMA:
        raise ValueError("assignment train run has unsupported solver_state_schema")
    loss = _mapping(payload, "loss")
    for key in ("initial_total", "final_total", "initial_assignment", "final_assignment"):
        _finite(loss.get(key), f"loss.{key}")
    if not isinstance(loss.get("loss_decreased"), bool):
        raise ValueError("assignment train run loss_decreased must be bool")
    if not isinstance(loss.get("assignment_loss_decreased"), bool):
        raise ValueError("assignment train run assignment_loss_decreased must be bool")
    if not isinstance(payload.get("loss_curve"), list) or not payload["loss_curve"]:
        raise ValueError("assignment train run requires loss_curve")
    for key in ("before_metrics", "after_metrics", "metric_delta"):
        metrics = _mapping(payload, key)
        for metric_key in ("mean_best_iou", "ari", "purity"):
            _finite(metrics.get(metric_key), f"{key}.{metric_key}")
    for key in ("before_assignment", "after_assignment", "before_projection", "after_projection"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError(f"assignment train run requires {key}")
    checkpoint = _mapping(payload, "checkpoint")
    if checkpoint.get("schema") != ASSIGNMENT_SOLVER_V2_STATE_SCHEMA:
        raise ValueError("assignment train checkpoint schema is unsupported")
    if checkpoint.get("roundtrip_ok") is not True:
        raise ValueError("assignment train checkpoint roundtrip must pass")
    refs = _mapping(payload, "visualization_refs")
    for key in ("before_ply", "after_ply"):
        if not isinstance(refs.get(key), str) or not refs[key]:
            raise ValueError(f"assignment train visualization requires {key}")
    long_gate = _mapping(payload, "long_training_gate")
    for key in (
        "allowed",
        "requires_loss_decrease",
        "requires_assignment_metric_improvement",
        "requires_visualization_refs",
        "requires_checkpoint_roundtrip",
        "requires_gate_handoff_not_collapsed",
    ):
        if not isinstance(long_gate.get(key), bool):
            raise ValueError(f"long_training_gate requires bool {key}")
    claim_policy = _mapping(payload, "claim_policy")
    if (
        not claim_policy.get("short_smoke_only")
        or not claim_policy.get("target_assignment_is_supervision_only")
        or not claim_policy.get("assignment_matrix_is_single_source_of_truth")
        or not claim_policy.get("does_not_claim_identity_gate_pass")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("assignment train run must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("assignment train run cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("assignment train run summary_path must be a string")
    return dict(payload)


def _assignment_evidence_from_cloud(
    cloud: GaussianCloud,
    *,
    target_assignment: np.ndarray,
    source: str,
    evidence_features: np.ndarray | None = None,
) -> AssignmentEvidenceBatch:
    return validate_assignment_evidence_batch(
        AssignmentEvidenceBatch(
            positions=positions(cloud),
            features=extract_features(cloud)
            if evidence_features is None
            else np.asarray(evidence_features, dtype=np.float32),
            target_assignment=target_assignment,
            source=source,
        )
    )


def _before_after_metrics(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, float]:
    before_metrics = _mapping(before, "target_metrics")
    after_metrics = _mapping(after, "target_metrics")
    return {
        "mean_best_iou": float(after_metrics["mean_best_iou"])
        - float(before_metrics["mean_best_iou"]),
        "ari": float(after_metrics["ari"]) - float(before_metrics["ari"]),
        "purity": float(after_metrics["purity"]) - float(before_metrics["purity"]),
    }


def _metrics_improved(metrics: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    if any(float(metrics[key]) > 1e-6 for key in ("mean_best_iou", "ari", "purity")):
        return True
    after_metrics = _mapping(after, "target_metrics")
    return all(
        float(after_metrics[key]) >= 0.99
        for key in ("mean_best_iou", "ari", "purity")
    )


def _smoke_pass(
    result: Any,
    metrics: Mapping[str, Any],
    after: Mapping[str, Any],
) -> bool:
    return bool(
        result.final_loss.total_loss < result.initial_loss.total_loss
        and _metrics_improved(metrics, after)
    )


def _roundtrip_ok(after: Mapping[str, Any], restored_after: Mapping[str, Any]) -> bool:
    return bool(
        after["assignment"]["shape"] == restored_after["assignment"]["shape"]
        and after["target_metrics"] == restored_after["target_metrics"]
        and after["projection"]["active_state_count"]
        == restored_after["projection"]["active_state_count"]
    )


def _write_assignment_visualization(
    path: Path,
    cloud: GaussianCloud,
    labels: np.ndarray,
    *,
    comments: tuple[str, ...],
) -> None:
    palette = np.asarray(
        [
            [230, 57, 70],
            [42, 157, 143],
            [69, 123, 157],
            [244, 162, 97],
            [131, 56, 236],
            [38, 70, 83],
            [255, 183, 3],
            [0, 109, 119],
        ],
        dtype=np.uint8,
    )
    colors = palette[np.asarray(labels, dtype=np.int64) % palette.shape[0]]
    vertices = append_or_replace_property(cloud.vertices, "red", colors[:, 0], "u1")
    vertices = append_or_replace_property(vertices, "green", colors[:, 1], "u1")
    vertices = append_or_replace_property(vertices, "blue", colors[:, 2], "u1")
    vertices = append_or_replace_property(
        vertices,
        "object_id",
        np.asarray(labels, dtype=np.int32),
        "i4",
    )
    vertices = append_or_replace_property(
        vertices,
        "predicted_object_id",
        np.asarray(labels, dtype=np.int32),
        "i4",
    )
    write_ply(
        path,
        GaussianCloud(
            vertices=vertices,
            comments=tuple(comments),
            source_format="ascii",
        ),
        fmt="ascii",
    )


def _source_kind(value: str) -> str:
    if value not in {"synthetic", "public_replay", "controlled_real"}:
        raise ValueError("source_kind must be synthetic, public_replay, or controlled_real")
    return value


def _split(value: str) -> str:
    if value not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    return value


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"assignment train run requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
