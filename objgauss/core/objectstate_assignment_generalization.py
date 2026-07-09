from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from objgauss.core.assignment_evidence import (
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
from objgauss.core.object_state import validate_assignment_matrix
from objgauss.core.objectstate_assignment_mvp import (
    OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA,
    objectstate_assignment_mvp_summary,
)
from objgauss.core.objectstate_assignment_train import (
    OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA,
    objectstate_assignment_train_dataset_summary,
    validate_objectstate_assignment_train_dataset_summary,
)

OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA = (
    "objgauss-objectstate-assignment-generalization-v1"
)


def objectstate_assignment_generalization_summary(
    train_cloud: GaussianCloud,
    train_target_assignment: np.ndarray,
    test_cloud: GaussianCloud,
    test_target_assignment: np.ndarray,
    *,
    output_dir: str | Path,
    sample_id: str = "assignment-generalization-001",
    train_sample_id: str = "train-scene",
    test_sample_id: str = "test-scene",
    source_kind: str = "synthetic",
    initial_state: AssignmentSolverV2State | None = None,
    iterations: int = 80,
    learning_rate: float = 0.2,
    assignment_weight: float = 1.0,
    compactness_weight: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    train_target = validate_assignment_matrix(
        train_target_assignment,
        evidence_count=train_cloud.count,
    )
    test_target = validate_assignment_matrix(
        test_target_assignment,
        evidence_count=test_cloud.count,
    )
    if train_target.shape[1] != test_target.shape[1]:
        raise ValueError("train and test target assignments must have the same slot count")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if iterations > 600:
        raise ValueError("assignment generalization iterations must stay <= 600")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if assignment_weight <= 0.0:
        raise ValueError("assignment_weight must be > 0")
    if compactness_weight < 0.0:
        raise ValueError("compactness_weight must be >= 0")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_dataset = objectstate_assignment_train_dataset_summary(
        train_cloud,
        train_target,
        sample_id=train_sample_id,
        source_kind=source_kind,
        split="train",
    )
    test_dataset = objectstate_assignment_train_dataset_summary(
        test_cloud,
        test_target,
        sample_id=test_sample_id,
        source_kind=source_kind,
        split="test",
    )
    train_evidence = _evidence_from_cloud(
        train_cloud,
        train_target,
        source=f"assignment-generalization:{train_sample_id}",
    )
    if initial_state is not None:
        initial_state = validate_assignment_solver_v2_state(initial_state)
    result = train_assignment_solver_v2(
        [train_evidence],
        slots=train_target.shape[1],
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
    train_before = objectstate_assignment_mvp_summary(
        train_cloud,
        result.initial_state,
        target_assignment=train_target,
        source=f"{sample_id}:train:before",
    )
    train_after = objectstate_assignment_mvp_summary(
        train_cloud,
        result.final_state,
        target_assignment=train_target,
        source=f"{sample_id}:train:after",
    )
    test_before = objectstate_assignment_mvp_summary(
        test_cloud,
        result.initial_state,
        target_assignment=test_target,
        source=f"{sample_id}:test:before",
    )
    test_after = objectstate_assignment_mvp_summary(
        test_cloud,
        result.final_state,
        target_assignment=test_target,
        source=f"{sample_id}:test:after",
    )
    checkpoint_path = output_root / "assignment-generalization-final-state.json"
    checkpoint = result.final_state.as_dict(include_arrays=True)
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    restored = assignment_solver_v2_state_from_dict(checkpoint)
    restored_test_after = objectstate_assignment_mvp_summary(
        test_cloud,
        restored,
        target_assignment=test_target,
        source=f"{sample_id}:test:restored",
    )
    train_delta = _metric_delta(train_before["target_metrics"], train_after["target_metrics"])
    test_delta = _metric_delta(test_before["target_metrics"], test_after["target_metrics"])
    gap = _generalization_gap(train_after["target_metrics"], test_after["target_metrics"])
    pass_status = _generalization_pass(
        result,
        train_delta,
        train_after["target_metrics"],
        test_delta,
        test_after["target_metrics"],
    )
    summary_path = output_root / "generalization-summary.json"
    payload = {
        "schema": OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA,
        "kind": "objectstate_assignment_generalization",
        "status": (
            "objectstate_assignment_generalization_pass"
            if pass_status
            else "objectstate_assignment_generalization_reviewable"
        ),
        "sample_id": str(sample_id),
        "dataset_schema": OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA,
        "mvp_schema": OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA,
        "training_schema": ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA,
        "solver_state_schema": ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
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
            "max_duration_policy": "short_generalization_smoke_under_10_minutes",
        },
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
        "loss_curve": [record.as_dict() for record in result.history],
        "train_before_metrics": train_before["target_metrics"],
        "train_after_metrics": train_after["target_metrics"],
        "train_metric_delta": train_delta,
        "test_before_metrics": test_before["target_metrics"],
        "test_after_metrics": test_after["target_metrics"],
        "test_metric_delta": test_delta,
        "generalization_gap": gap,
        "train_assignment": train_after["assignment"],
        "test_assignment": test_after["assignment"],
        "train_projection": train_after["projection"],
        "test_projection": test_after["projection"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "schema": checkpoint["schema"],
            "roundtrip_ok": bool(
                restored_test_after["target_metrics"] == test_after["target_metrics"]
            ),
        },
        "decision": _decision(
            result,
            train_delta,
            train_after["target_metrics"],
            test_delta,
            gap,
            test_after["target_metrics"],
        ),
        "long_training_gate": {
            "allowed": bool(pass_status),
            "requires_loss_decrease": bool(result.final_loss.total_loss < result.initial_loss.total_loss),
            "requires_train_metric_improvement": _metrics_improved(
                train_delta,
                train_after["target_metrics"],
            ),
            "requires_test_metric_improvement": _metrics_improved(
                test_delta,
                test_after["target_metrics"],
            ),
            "requires_checkpoint_roundtrip": bool(
                restored_test_after["target_metrics"] == test_after["target_metrics"]
            ),
            "requires_gate_handoff_not_collapsed": True,
        },
        "claim_policy": {
            "tests_held_out_assignment_sample": True,
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
    checked = validate_objectstate_assignment_generalization_summary(payload)
    summary_path.write_text(
        json.dumps(checked, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_assignment_generalization_summary(checked)


def validate_objectstate_assignment_generalization_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("assignment generalization summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA:
        raise ValueError(
            f"unsupported assignment generalization schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_assignment_generalization":
        raise ValueError("assignment generalization kind is unsupported")
    if payload.get("status") not in {
        "objectstate_assignment_generalization_pass",
        "objectstate_assignment_generalization_reviewable",
    }:
        raise ValueError("assignment generalization status is unsupported")
    validate_objectstate_assignment_train_dataset_summary(payload.get("train_dataset"))
    validate_objectstate_assignment_train_dataset_summary(payload.get("test_dataset"))
    if payload["train_dataset"]["split"] != "train":
        raise ValueError("assignment generalization train_dataset split must be train")
    if payload["test_dataset"]["split"] != "test":
        raise ValueError("assignment generalization test_dataset split must be test")
    if payload["train_dataset"]["slots"] != payload["test_dataset"]["slots"]:
        raise ValueError("assignment generalization train/test slots must match")
    if payload.get("training_schema") != ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA:
        raise ValueError("assignment generalization has unsupported training_schema")
    if payload.get("solver_state_schema") != ASSIGNMENT_SOLVER_V2_STATE_SCHEMA:
        raise ValueError("assignment generalization has unsupported solver_state_schema")
    loss = _mapping(payload, "loss")
    for key in ("initial_total", "final_total", "initial_assignment", "final_assignment"):
        _finite(loss.get(key), f"loss.{key}")
    if not isinstance(loss.get("loss_decreased"), bool):
        raise ValueError("assignment generalization loss_decreased must be bool")
    if not isinstance(payload.get("loss_curve"), list) or not payload["loss_curve"]:
        raise ValueError("assignment generalization requires loss_curve")
    for key in (
        "train_before_metrics",
        "train_after_metrics",
        "train_metric_delta",
        "test_before_metrics",
        "test_after_metrics",
        "test_metric_delta",
        "generalization_gap",
    ):
        metrics = _mapping(payload, key)
        for metric_key in ("mean_best_iou", "ari", "purity"):
            _finite(metrics.get(metric_key), f"{key}.{metric_key}")
    checkpoint = _mapping(payload, "checkpoint")
    if checkpoint.get("schema") != ASSIGNMENT_SOLVER_V2_STATE_SCHEMA:
        raise ValueError("assignment generalization checkpoint schema is unsupported")
    if checkpoint.get("roundtrip_ok") is not True:
        raise ValueError("assignment generalization checkpoint roundtrip must pass")
    decision = _mapping(payload, "decision")
    if not isinstance(decision.get("label"), str) or not decision["label"]:
        raise ValueError("assignment generalization decision requires label")
    if not isinstance(decision.get("issues"), list):
        raise ValueError("assignment generalization decision requires issues")
    long_gate = _mapping(payload, "long_training_gate")
    for key in (
        "allowed",
        "requires_loss_decrease",
        "requires_train_metric_improvement",
        "requires_test_metric_improvement",
        "requires_checkpoint_roundtrip",
        "requires_gate_handoff_not_collapsed",
    ):
        if not isinstance(long_gate.get(key), bool):
            raise ValueError(f"long_training_gate requires bool {key}")
    claim_policy = _mapping(payload, "claim_policy")
    if (
        not claim_policy.get("tests_held_out_assignment_sample")
        or not claim_policy.get("target_assignment_is_supervision_only")
        or not claim_policy.get("assignment_matrix_is_single_source_of_truth")
        or not claim_policy.get("does_not_claim_identity_gate_pass")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("assignment generalization must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("assignment generalization cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("assignment generalization summary_path must be a string")
    return dict(payload)


def _evidence_from_cloud(
    cloud: GaussianCloud,
    target_assignment: np.ndarray,
    *,
    source: str,
) -> AssignmentEvidenceBatch:
    return validate_assignment_evidence_batch(
        AssignmentEvidenceBatch(
            positions=positions(cloud),
            features=extract_features(cloud),
            target_assignment=target_assignment,
            source=source,
        )
    )


def _metric_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "mean_best_iou": float(after["mean_best_iou"]) - float(before["mean_best_iou"]),
        "ari": float(after["ari"]) - float(before["ari"]),
        "purity": float(after["purity"]) - float(before["purity"]),
    }


def _generalization_gap(
    train_after: Mapping[str, Any],
    test_after: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "mean_best_iou": float(train_after["mean_best_iou"]) - float(test_after["mean_best_iou"]),
        "ari": float(train_after["ari"]) - float(test_after["ari"]),
        "purity": float(train_after["purity"]) - float(test_after["purity"]),
    }


def _metrics_improved(delta: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    if any(float(delta[key]) > 1e-6 for key in ("mean_best_iou", "ari", "purity")):
        return True
    return all(float(after[key]) >= 0.99 for key in ("mean_best_iou", "ari", "purity"))


def _generalization_pass(
    result: Any,
    train_delta: Mapping[str, Any],
    train_after: Mapping[str, Any],
    test_delta: Mapping[str, Any],
    test_after: Mapping[str, Any],
) -> bool:
    return bool(
        result.final_loss.total_loss < result.initial_loss.total_loss
        and _metrics_improved(train_delta, train_after)
        and _metrics_improved(test_delta, test_after)
        and float(test_after["ari"]) >= 0.5
        and float(test_after["purity"]) >= 0.5
    )


def _decision(
    result: Any,
    train_delta: Mapping[str, Any],
    train_after: Mapping[str, Any],
    test_delta: Mapping[str, Any],
    gap: Mapping[str, Any],
    test_after: Mapping[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    if result.final_loss.total_loss >= result.initial_loss.total_loss:
        issues.append("loss_did_not_decrease")
    if not _metrics_improved(train_delta, train_after):
        issues.append("train_assignment_metrics_did_not_improve")
    if not _metrics_improved(test_delta, test_after):
        issues.append("held_out_assignment_metrics_did_not_improve")
    if any(float(gap[key]) > 0.5 for key in ("mean_best_iou", "ari", "purity")):
        issues.append("large_train_test_assignment_gap")
    if float(test_after["ari"]) < 0.5 or float(test_after["purity"]) < 0.5:
        issues.append("held_out_assignment_quality_below_floor")
    return {
        "label": "generalization_reviewable" if not issues else "generalization_needs_diagnosis",
        "issues": issues,
        "next_action": (
            "run_identity_gate_handoff"
            if not issues
            else "inspect_data_or_model_shortcut_before_long_training"
        ),
    }


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"assignment generalization requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
