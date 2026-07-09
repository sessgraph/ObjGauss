from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
)
from objgauss.core.features import colors, opacity
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_state import validate_assignment_matrix
from objgauss.core.objectstate_assignment_generalization import (
    OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA,
    objectstate_assignment_generalization_summary,
    validate_objectstate_assignment_generalization_summary,
)
from objgauss.core.objectstate_assignment_train import (
    OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA,
    validate_objectstate_assignment_train_dataset_summary,
)

OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA = (
    "objgauss-objectstate-assignment-ablation-v1"
)
DEFAULT_ASSIGNMENT_ABLATION_POLICIES = (
    "xyz",
    "xyz_rgb",
    "xyz_rgb_opacity",
)


@dataclass(frozen=True)
class _PolicySpec:
    name: str
    feature_blocks: tuple[str, ...]
    feature_weight: float
    position_weight: float
    complexity_rank: int


_POLICY_SPECS = {
    "xyz": _PolicySpec("xyz", ("constant",), 0.0, 1.0, 1),
    "rgb": _PolicySpec("rgb", ("rgb",), 1.0, 0.0, 2),
    "xyz_rgb": _PolicySpec("xyz_rgb", ("rgb",), 1.0, 1.0, 3),
    "xyz_rgb_opacity": _PolicySpec("xyz_rgb_opacity", ("rgb", "opacity"), 1.0, 1.0, 4),
    "xyz_rgb_opacity_semantic": _PolicySpec(
        "xyz_rgb_opacity_semantic",
        ("rgb", "opacity", "semantic"),
        1.0,
        1.0,
        5,
    ),
}


def objectstate_assignment_ablation_summary(
    train_cloud: GaussianCloud,
    train_target_assignment: np.ndarray,
    test_cloud: GaussianCloud,
    test_target_assignment: np.ndarray,
    *,
    output_dir: str | Path,
    sample_id: str = "assignment-ablation-001",
    train_sample_id: str = "train-scene",
    test_sample_id: str = "heldout-scene",
    source_kind: str = "synthetic",
    policies: Sequence[str] = DEFAULT_ASSIGNMENT_ABLATION_POLICIES,
    semantic_train_features: np.ndarray | None = None,
    semantic_test_features: np.ndarray | None = None,
    iterations: int = 80,
    learning_rate: float = 0.2,
    assignment_weight: float = 1.0,
    compactness_weight: float = 0.05,
    seed: int = 0,
    test_ari_floor: float = 0.5,
    test_purity_floor: float = 0.5,
    max_generalization_gap: float = 0.5,
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
        raise ValueError("assignment ablation iterations must stay <= 600")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if assignment_weight <= 0.0:
        raise ValueError("assignment_weight must be > 0")
    if compactness_weight < 0.0:
        raise ValueError("compactness_weight must be >= 0")
    semantic_pair = _validate_semantic_pair(
        semantic_train_features,
        semantic_test_features,
        train_count=train_cloud.count,
        test_count=test_cloud.count,
    )
    policy_specs = _resolve_policies(policies, semantic_pair=semantic_pair is not None)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    variants = []
    for index, spec in enumerate(policy_specs):
        train_features = _features_for_policy(
            spec,
            train_cloud,
            semantic_features=None if semantic_pair is None else semantic_pair[0],
        )
        test_features = _features_for_policy(
            spec,
            test_cloud,
            semantic_features=None if semantic_pair is None else semantic_pair[1],
        )
        initial_state = _initial_state_for_policy(
            spec,
            slots=train_target.shape[1],
            feature_dim=train_features.shape[1],
            seed=int(seed) + index * 17,
        )
        variant = objectstate_assignment_generalization_summary(
            train_cloud,
            train_target,
            test_cloud,
            test_target,
            output_dir=output_root / _safe_policy_name(spec.name),
            sample_id=f"{sample_id}:{spec.name}",
            train_sample_id=train_sample_id,
            test_sample_id=test_sample_id,
            source_kind=source_kind,
            train_features=train_features,
            test_features=test_features,
            initial_state=initial_state,
            iterations=iterations,
            learning_rate=learning_rate,
            assignment_weight=assignment_weight,
            compactness_weight=compactness_weight,
            seed=int(seed) + index * 17,
        )
        variants.append(_variant_summary(spec, variant))

    train_dataset = variants[0]["train_dataset"]
    test_dataset = variants[0]["test_dataset"]
    ranking = _ranking(
        variants,
        test_ari_floor=test_ari_floor,
        test_purity_floor=test_purity_floor,
        max_generalization_gap=max_generalization_gap,
    )
    summary_path = output_root / "ablation-summary.json"
    payload = {
        "schema": OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA,
        "kind": "objectstate_assignment_ablation",
        "status": (
            "objectstate_assignment_ablation_pass"
            if ranking["minimum_sufficient_evidence"]["found"]
            else "objectstate_assignment_ablation_reviewable"
        ),
        "sample_id": str(sample_id),
        "dataset_schema": OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA,
        "generalization_schema": OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA,
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
            "max_duration_policy": "short_ablation_smoke_under_10_minutes",
        },
        "success_thresholds": {
            "test_ari_floor": float(test_ari_floor),
            "test_purity_floor": float(test_purity_floor),
            "max_generalization_gap": float(max_generalization_gap),
        },
        "policies": [_policy_summary(spec) for spec in policy_specs],
        "variants": variants,
        "ranking": ranking,
        "shortcut_diagnostics": _shortcut_diagnostics(variants, ranking),
        "next_stage_gate": {
            "identity_gate_handoff_allowed": bool(
                ranking["minimum_sufficient_evidence"]["found"]
            ),
            "long_training_allowed": False,
            "long_training_blockers": [
                "identity_retrieval_not_evaluated",
                "temporal_consistency_not_evaluated",
                "real_evidence_not_evaluated",
            ],
        },
        "claim_policy": {
            "compares_minimum_sufficient_evidence": True,
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
    checked = validate_objectstate_assignment_ablation_summary(payload)
    summary_path.write_text(json.dumps(checked, indent=2, sort_keys=True), encoding="utf-8")
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_assignment_ablation_summary(checked)


def validate_objectstate_assignment_ablation_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("assignment ablation summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA:
        raise ValueError(f"unsupported assignment ablation schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_assignment_ablation":
        raise ValueError("assignment ablation kind is unsupported")
    if payload.get("status") not in {
        "objectstate_assignment_ablation_pass",
        "objectstate_assignment_ablation_reviewable",
    }:
        raise ValueError("assignment ablation status is unsupported")
    validate_objectstate_assignment_train_dataset_summary(payload.get("train_dataset"))
    validate_objectstate_assignment_train_dataset_summary(payload.get("test_dataset"))
    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("assignment ablation requires variants")
    for variant in variants:
        validate_objectstate_assignment_generalization_summary(variant["generalization"])
        _mapping(variant, "feature_policy")
        _mapping(variant, "test_after_metrics")
        _mapping(variant, "generalization_gap")
    ranking = _mapping(payload, "ranking")
    _mapping(ranking, "minimum_sufficient_evidence")
    next_stage = _mapping(payload, "next_stage_gate")
    if next_stage.get("long_training_allowed") is not False:
        raise ValueError("assignment ablation must keep long training blocked")
    claim_policy = _mapping(payload, "claim_policy")
    if (
        not claim_policy.get("compares_minimum_sufficient_evidence")
        or not claim_policy.get("target_assignment_is_supervision_only")
        or not claim_policy.get("assignment_matrix_is_single_source_of_truth")
        or not claim_policy.get("does_not_claim_identity_gate_pass")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("assignment ablation must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("assignment ablation cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("assignment ablation summary_path must be a string")
    return dict(payload)


def _variant_summary(spec: _PolicySpec, generalization: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy": spec.name,
        "status": (
            "objectstate_assignment_ablation_variant_pass"
            if generalization["status"] == "objectstate_assignment_generalization_pass"
            else "objectstate_assignment_ablation_variant_reviewable"
        ),
        "feature_policy": _policy_summary(spec),
        "train_dataset": generalization["train_dataset"],
        "test_dataset": generalization["test_dataset"],
        "loss": generalization["loss"],
        "loss_curve": generalization["loss_curve"],
        "train_before_metrics": generalization["train_before_metrics"],
        "train_after_metrics": generalization["train_after_metrics"],
        "train_metric_delta": generalization["train_metric_delta"],
        "test_before_metrics": generalization["test_before_metrics"],
        "test_after_metrics": generalization["test_after_metrics"],
        "test_metric_delta": generalization["test_metric_delta"],
        "generalization_gap": generalization["generalization_gap"],
        "checkpoint": generalization["checkpoint"],
        "decision": generalization["decision"],
        "generalization": dict(generalization),
    }


def _features_for_policy(
    spec: _PolicySpec,
    cloud: GaussianCloud,
    *,
    semantic_features: np.ndarray | None,
) -> np.ndarray:
    blocks = []
    for block in spec.feature_blocks:
        if block == "constant":
            blocks.append(np.ones((cloud.count, 1), dtype=np.float32))
        elif block == "rgb":
            blocks.append(_normalize(colors(cloud)))
        elif block == "opacity":
            blocks.append(_normalize(opacity(cloud)[:, None]))
        elif block == "semantic":
            if semantic_features is None:
                raise ValueError("semantic ablation policy requires semantic features")
            blocks.append(_normalize(semantic_features))
        else:
            raise ValueError(f"unsupported ablation feature block: {block}")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def _initial_state_for_policy(
    spec: _PolicySpec,
    *,
    slots: int,
    feature_dim: int,
    seed: int,
) -> AssignmentSolverV2State:
    rng = np.random.default_rng(seed)
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=int(slots),
            feature_dim=int(feature_dim),
            temperature=0.5,
            feature_weight=float(spec.feature_weight),
            position_weight=float(spec.position_weight),
        ),
        feature_centers=rng.normal(0.0, 0.05, size=(slots, feature_dim)).astype(np.float32),
        position_centers=rng.normal(0.0, 0.05, size=(slots, 3)).astype(np.float32),
        slot_bias=np.zeros(slots, dtype=np.float32),
        source=f"assignment_ablation_{spec.name}_init",
    )


def _ranking(
    variants: Sequence[Mapping[str, Any]],
    *,
    test_ari_floor: float,
    test_purity_floor: float,
    max_generalization_gap: float,
) -> dict[str, Any]:
    sorted_by_ari = sorted(
        variants,
        key=lambda item: (
            float(item["test_after_metrics"]["ari"]),
            float(item["test_after_metrics"]["purity"]),
        ),
        reverse=True,
    )
    passing = [
        variant
        for variant in variants
        if (
            float(variant["test_after_metrics"]["ari"]) >= test_ari_floor
            and float(variant["test_after_metrics"]["purity"]) >= test_purity_floor
            and abs(float(variant["generalization_gap"]["ari"])) <= max_generalization_gap
            and abs(float(variant["generalization_gap"]["purity"])) <= max_generalization_gap
        )
    ]
    passing = sorted(passing, key=lambda item: int(item["feature_policy"]["complexity_rank"]))
    minimum = passing[0] if passing else None
    best = sorted_by_ari[0]
    return {
        "best_by_test_ari": {
            "policy": best["policy"],
            "test_ari": float(best["test_after_metrics"]["ari"]),
            "test_purity": float(best["test_after_metrics"]["purity"]),
        },
        "minimum_sufficient_evidence": {
            "found": minimum is not None,
            "policy": None if minimum is None else minimum["policy"],
            "feature_blocks": [] if minimum is None else minimum["feature_policy"]["feature_blocks"],
            "test_ari": None if minimum is None else float(minimum["test_after_metrics"]["ari"]),
            "test_purity": None if minimum is None else float(minimum["test_after_metrics"]["purity"]),
        },
        "policies_by_test_ari": [
            {
                "policy": variant["policy"],
                "test_ari": float(variant["test_after_metrics"]["ari"]),
                "test_purity": float(variant["test_after_metrics"]["purity"]),
                "status": variant["status"],
            }
            for variant in sorted_by_ari
        ],
    }


def _shortcut_diagnostics(
    variants: Sequence[Mapping[str, Any]],
    ranking: Mapping[str, Any],
) -> dict[str, Any]:
    by_policy = {str(variant["policy"]): variant for variant in variants}
    xyz = by_policy.get("xyz")
    rgb = by_policy.get("rgb")
    combined = by_policy.get("xyz_rgb") or by_policy.get("xyz_rgb_opacity")
    notes: list[str] = []
    if xyz is not None and float(xyz["test_after_metrics"]["ari"]) >= 0.5:
        notes.append("xyz_alone_can_explain_held_out_assignment")
    if rgb is not None and float(rgb["test_after_metrics"]["ari"]) >= 0.5:
        notes.append("rgb_alone_can_explain_held_out_assignment")
    if (
        combined is not None
        and xyz is not None
        and float(combined["test_after_metrics"]["ari"])
        > float(xyz["test_after_metrics"]["ari"]) + 0.1
    ):
        notes.append("non_spatial_features_improve_over_xyz")
    if not ranking["minimum_sufficient_evidence"]["found"]:
        notes.append("no_policy_met_minimum_sufficient_evidence_threshold")
    return {
        "notes": notes,
        "position_shortcut_possible": "xyz_alone_can_explain_held_out_assignment" in notes,
        "color_shortcut_possible": "rgb_alone_can_explain_held_out_assignment" in notes,
        "requires_non_spatial_evidence": "non_spatial_features_improve_over_xyz" in notes,
    }


def _validate_semantic_pair(
    train_features: np.ndarray | None,
    test_features: np.ndarray | None,
    *,
    train_count: int,
    test_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if train_features is None and test_features is None:
        return None
    if train_features is None or test_features is None:
        raise ValueError("semantic ablation requires both train and test semantic features")
    train = _feature_matrix(train_features, "semantic_train_features", rows=train_count)
    test = _feature_matrix(test_features, "semantic_test_features", rows=test_count)
    if train.shape[1] != test.shape[1]:
        raise ValueError("semantic train and test feature_dim must match")
    return train, test


def _feature_matrix(value: np.ndarray, label: str, *, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if array.shape[0] != rows:
        raise ValueError(f"{label} rows must match Gaussian count")
    if array.shape[1] < 1:
        raise ValueError(f"{label} must contain at least one feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array.astype(np.float32, copy=False)


def _resolve_policies(
    policies: Sequence[str],
    *,
    semantic_pair: bool,
) -> tuple[_PolicySpec, ...]:
    if not policies:
        raise ValueError("assignment ablation requires at least one policy")
    resolved = []
    seen = set()
    for policy in policies:
        name = str(policy)
        if name in seen:
            raise ValueError(f"duplicate ablation policy: {name}")
        if name not in _POLICY_SPECS:
            raise ValueError(f"unsupported ablation policy: {name}")
        spec = _POLICY_SPECS[name]
        if "semantic" in spec.feature_blocks and not semantic_pair:
            raise ValueError("semantic ablation policy requires train and test semantic features")
        resolved.append(spec)
        seen.add(name)
    return tuple(resolved)


def _policy_summary(spec: _PolicySpec) -> dict[str, Any]:
    return {
        "policy": spec.name,
        "feature_blocks": list(spec.feature_blocks),
        "uses_position_cost": bool(spec.position_weight > 0.0),
        "uses_feature_cost": bool(spec.feature_weight > 0.0),
        "uses_color": "rgb" in spec.feature_blocks,
        "uses_opacity": "opacity" in spec.feature_blocks,
        "uses_semantic": "semantic" in spec.feature_blocks,
        "feature_weight": float(spec.feature_weight),
        "position_weight": float(spec.position_weight),
        "complexity_rank": int(spec.complexity_rank),
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("feature block must be 2D")
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((array - mean) / std).astype(np.float32, copy=False)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"assignment ablation requires {key}")
    return value


def _safe_policy_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
