from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_metrics import instance_segmentation_metrics
from objgauss.core.clustering import cluster_features
from objgauss.core.features import colors, positions
from objgauss.datasets.objectstate_model_diagnostic_synthetic import (
    OBJECTSTATE_MODEL_DIAGNOSTIC_CASES,
    ObjectStateModelDiagnosticDataset,
    diagnostic_semantic_proxy,
)
from objgauss.datasets.objectstate_multi_object_synthetic import (
    MultiObjectSyntheticDataset,
)
from objgauss.evaluation.objectstate_instance_segmentation import (
    connected_component_labels_3d,
    evaluate_multi_object_instance_benchmark,
    normalized_features,
)

OBJECTSTATE_MODEL_DIAGNOSTIC_SCHEMA = "objgauss-objectstate-model-diagnostic-v1"
OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES = (
    "xyz_only",
    "rgb_only",
    "xyz_rgb",
    "xyz_rgb_semantic",
)
OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES = (
    "xyz_kmeans",
    "rgb_kmeans",
    "connected_components_3d",
)

_METRICS = (
    "ari",
    "hungarian_mean_iou",
    "object_recall_iou_0_5",
    "object_recall_iou_0_75",
    "object_count_error",
    "merge_rate",
    "split_rate",
)
_IDENTITY_METRICS = ("slot_swap_rate", "unmapped_identity_rate")

__all__ = (
    "OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES",
    "OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES",
    "OBJECTSTATE_MODEL_DIAGNOSTIC_SCHEMA",
    "evaluate_objectstate_model_diagnostic",
    "identity_swap_metrics",
    "validate_objectstate_model_diagnostic",
)


def evaluate_objectstate_model_diagnostic(
    dataset: ObjectStateModelDiagnosticDataset,
    variant_assignments: Mapping[str, Mapping[int, np.ndarray]],
    *,
    m2_dataset: MultiObjectSyntheticDataset,
    m2_variant_assignments: Mapping[str, Mapping[int, np.ndarray]],
    m2_native_assignments: Mapping[int, np.ndarray],
    feature_orders: Mapping[str, Sequence[str]],
    expected_seeds: Sequence[int],
    connected_component_radius: float = 0.18,
    baseline_seed: int = 0,
) -> dict[str, Any]:
    if connected_component_radius <= 0.0:
        raise ValueError("connected_component_radius must be > 0")
    seeds = tuple(int(seed) for seed in expected_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must contain at least three unique seeds")
    if set(variant_assignments) != set(OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES):
        raise ValueError("diagnostic requires the canonical ordered feature policies")

    baseline_labels = _baseline_labels(
        dataset,
        radius=connected_component_radius,
        seed=baseline_seed,
    )
    baselines = {
        name: _evaluate_candidate(dataset, labels, candidate=name)
        for name, labels in baseline_labels.items()
    }

    variants: dict[str, Any] = {}
    ablation_rows: list[dict[str, Any]] = []
    for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES:
        runs = variant_assignments[policy]
        if set(runs) != set(seeds):
            raise ValueError(f"{policy} assignments do not cover expected_seeds")
        seed_rows = []
        for seed in seeds:
            assignment = np.asarray(runs[seed], dtype=np.float32)
            if assignment.ndim != 2 or assignment.shape[0] != dataset.cloud.count:
                raise ValueError(f"{policy} seed {seed} assignment shape is invalid")
            if not np.all(np.isfinite(assignment)):
                raise ValueError(f"{policy} seed {seed} assignment must be finite")
            result = _evaluate_candidate(
                dataset,
                _labels_by_scene(dataset, np.argmax(assignment, axis=1)),
                candidate=policy,
            )
            seed_row = {"seed": seed, **result}
            seed_rows.append(seed_row)
            ablation_rows.append(
                {
                    "cohort": "hard_case",
                    "policy": policy,
                    "seed": seed,
                    **{name: float(result["aggregate"][name]) for name in _METRICS},
                    **{
                        name: float(result["identity"][name])
                        for name in _IDENTITY_METRICS
                    },
                }
            )
        variants[policy] = _aggregate_seed_runs(policy, seed_rows)

    m2_reproduction = _evaluate_m2_reproduction(
        m2_dataset,
        m2_variant_assignments,
        m2_native_assignments=m2_native_assignments,
        seeds=seeds,
        connected_component_radius=connected_component_radius,
        baseline_seed=baseline_seed,
    )
    ablation_rows.extend(m2_reproduction["ablation_matrix"])

    leakage_gate = _leakage_gate(
        dataset,
        m2_dataset=m2_dataset,
        feature_orders=feature_orders,
        seeds=seeds,
        variant_assignments=variant_assignments,
        m2_variant_assignments=m2_variant_assignments,
        m2_native_assignments=m2_native_assignments,
    )
    hard_case_comparison = _comparison(baselines, variants)
    comparison = _combined_comparison(m2_reproduction, variants, hard_case_comparison)
    diagnosis = _diagnosis(
        m2_reproduction,
        baselines,
        variants,
        hard_case_comparison,
        comparison,
    )
    payload = {
        "schema": OBJECTSTATE_MODEL_DIAGNOSTIC_SCHEMA,
        "kind": "objectstate_model_diagnostic",
        "status": "reviewable" if leakage_gate["passed"] else "invalid",
        "question": "why_model_v0_loses_to_connected_components",
        "dataset_schema": dataset.schema,
        "split": dataset.as_dict()["split"],
        "feature_policies": {
            policy: {"feature_order": list(feature_orders[policy])}
            for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES
        },
        "training_seed_policy": {
            "seeds": list(seeds),
            "aggregation": "mean_and_population_std_over_all_configured_seeds",
            "best_seed_selection": False,
        },
        "baseline_contract": {
            "names": list(OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES),
            "kmeans_uses_target_object_count": True,
            "kmeans_seed": int(baseline_seed),
            "connected_component_radius": float(connected_component_radius),
        },
        "metric_contract": {
            "instance_matching": "maximum_iou_bipartite_hungarian",
            "identity_swap_matching": "anchor_view_mapping_frozen_for_target_view",
            "material_fraction": 0.1,
            "aggregate_grain": "equal_weight_per_heldout_observation",
            "metrics": list(_METRICS) + list(_IDENTITY_METRICS),
        },
        "leakage_gate": leakage_gate,
        "m2_reproduction": m2_reproduction,
        "hard_case": {
            "baselines": baselines,
            "variants": variants,
            "comparison": hard_case_comparison,
        },
        "ablation_matrix": ablation_rows,
        "comparison": comparison,
        "diagnosis": diagnosis,
        "claim_policy": {
            "diagnoses_existing_model_family": True,
            "semantic_proxy_is_oracle_style_class_evidence": True,
            "negative_result_is_valid_evidence": True,
            "does_not_claim_real_identity": True,
            "does_not_claim_prediction_or_intervention": True,
            "does_not_claim_model_superiority_unless_comparison_passes": True,
        },
    }
    return validate_objectstate_model_diagnostic(payload)


def identity_swap_metrics(
    anchor_predicted: np.ndarray,
    anchor_target: np.ndarray,
    target_predicted: np.ndarray,
    target_target: np.ndarray,
) -> dict[str, Any]:
    anchor = instance_segmentation_metrics(anchor_predicted, anchor_target)
    predicted_to_target = {
        int(row["predicted_id"]): int(row["target_id"])
        for row in anchor["matching"]
    }
    common = tuple(
        int(item)
        for item in sorted(set(np.unique(anchor_target)) & set(np.unique(target_target)))
    )
    rows = []
    swaps = 0
    unmapped = 0
    for target_id in common:
        predicted_values, counts = np.unique(
            np.asarray(target_predicted)[np.asarray(target_target) == target_id],
            return_counts=True,
        )
        dominant = int(predicted_values[int(np.argmax(counts))])
        mapped_target = predicted_to_target.get(dominant)
        is_unmapped = mapped_target is None
        is_swap = mapped_target is not None and mapped_target != target_id
        unmapped += int(is_unmapped)
        swaps += int(is_swap)
        rows.append(
            {
                "target_id": target_id,
                "dominant_predicted_id": dominant,
                "anchor_mapped_target_id": mapped_target,
                "swapped": is_swap,
                "unmapped": is_unmapped,
            }
        )
    denominator = len(common)
    return {
        "identity_count": denominator,
        "slot_swap_count": swaps,
        "slot_swap_rate": float(swaps / denominator) if denominator else 0.0,
        "unmapped_identity_count": unmapped,
        "unmapped_identity_rate": float(unmapped / denominator) if denominator else 0.0,
        "rows": rows,
    }


def validate_objectstate_model_diagnostic(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("model diagnostic must be a mapping")
    if payload.get("schema") != OBJECTSTATE_MODEL_DIAGNOSTIC_SCHEMA:
        raise ValueError(f"unsupported model diagnostic schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_model_diagnostic":
        raise ValueError("model diagnostic kind is unsupported")
    if payload.get("status") not in {"reviewable", "invalid"}:
        raise ValueError("model diagnostic status is unsupported")
    leakage = _mapping(payload, "leakage_gate")
    if (payload["status"] == "reviewable") != bool(leakage.get("passed")):
        raise ValueError("model diagnostic status must match leakage gate")
    hard_case = _mapping(payload, "hard_case")
    variants = _mapping(hard_case, "variants")
    if set(variants) != set(OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES):
        raise ValueError("model diagnostic variants are incomplete")
    baselines = _mapping(hard_case, "baselines")
    if set(baselines) != set(OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES):
        raise ValueError("model diagnostic baselines are incomplete")
    rows = payload.get("ablation_matrix")
    seed_policy = _mapping(payload, "training_seed_policy")
    expected_rows = (len(OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES) * 2 + 1) * len(
        seed_policy["seeds"]
    )
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError("model diagnostic ablation matrix has invalid coverage")
    if seed_policy.get("best_seed_selection") is not False:
        raise ValueError("model diagnostic cannot select a favorable seed")
    for section in (*baselines.values(), *variants.values()):
        aggregate = _mapping(section, "aggregate")
        for metric in _METRICS:
            _finite(aggregate.get(metric), f"aggregate.{metric}")
        identity = _mapping(section, "identity")
        for metric in _IDENTITY_METRICS:
            _finite(identity.get(metric), f"identity.{metric}")
    m2 = _mapping(payload, "m2_reproduction")
    m2_variants = _mapping(m2, "variants")
    if set(m2_variants) != set(OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES):
        raise ValueError("M2 diagnostic variants are incomplete")
    _mapping(m2, "native_anchor")
    claim = _mapping(payload, "claim_policy")
    if not all(bool(value) for value in claim.values()):
        raise ValueError("model diagnostic must preserve all claim boundaries")
    return dict(payload)


def _baseline_labels(
    dataset: ObjectStateModelDiagnosticDataset,
    *,
    radius: float,
    seed: int,
) -> dict[str, dict[int, np.ndarray]]:
    labels = {name: {} for name in OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES}
    scene_values = dataset.cloud.vertices["scene_id"].astype(np.int64, copy=False)
    xyz = positions(dataset.cloud)
    rgb = colors(dataset.cloud)
    target = dataset.cloud.vertices["gt_instance_id"].astype(np.int64, copy=False)
    for scene_id in dataset.heldout_scene_ids:
        mask = scene_values == scene_id
        slots = int(np.unique(target[mask]).shape[0])
        labels["xyz_kmeans"][scene_id] = cluster_features(
            normalized_features(xyz[mask]),
            clusters=slots,
            seed=seed + scene_id,
        ).labels
        labels["rgb_kmeans"][scene_id] = cluster_features(
            normalized_features(rgb[mask]),
            clusters=slots,
            seed=seed + scene_id,
        ).labels
        labels["connected_components_3d"][scene_id] = connected_component_labels_3d(
            xyz[mask],
            radius=radius,
        )
    return labels


def _labels_by_scene(
    dataset: ObjectStateModelDiagnosticDataset,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    values = np.asarray(labels)
    if values.shape != (dataset.cloud.count,):
        raise ValueError("hard labels must contain one value per Gaussian")
    scene = dataset.cloud.vertices["scene_id"].astype(np.int64, copy=False)
    return {scene_id: values[scene == scene_id] for scene_id in dataset.heldout_scene_ids}


def _evaluate_candidate(
    dataset: ObjectStateModelDiagnosticDataset,
    labels_by_scene: Mapping[int, np.ndarray],
    *,
    candidate: str,
) -> dict[str, Any]:
    cloud = dataset.cloud
    scenes = cloud.vertices["scene_id"].astype(np.int64, copy=False)
    targets = cloud.vertices["gt_instance_id"].astype(np.int64, copy=False)
    observation_by_id = {int(row["scene_id"]): row for row in dataset.observations}
    observations = []
    for scene_id in dataset.heldout_scene_ids:
        mask = scenes == scene_id
        metrics = instance_segmentation_metrics(labels_by_scene[scene_id], targets[mask])
        observations.append(
            {
                "scene_id": scene_id,
                "case": str(observation_by_id[scene_id]["case"]),
                "gaussian_count": int(mask.sum()),
                **metrics,
            }
        )
    case_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        case_rows[str(row["case"])].append(row)
    cases = {
        case: _mean_metrics(case_rows[case])
        for case in OBJECTSTATE_MODEL_DIAGNOSTIC_CASES
    }
    identity_rows = []
    for pair in dataset.cross_view_pairs:
        anchor_id = int(pair["anchor_scene_id"])
        target_id = int(pair["target_scene_id"])
        anchor_mask = scenes == anchor_id
        target_mask = scenes == target_id
        identity_rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                **identity_swap_metrics(
                    labels_by_scene[anchor_id],
                    targets[anchor_mask],
                    labels_by_scene[target_id],
                    targets[target_mask],
                ),
            }
        )
    identity = {
        "pair_count": len(identity_rows),
        **{
            metric: float(np.mean([row[metric] for row in identity_rows]))
            if identity_rows
            else 0.0
            for metric in _IDENTITY_METRICS
        },
        "pairs": identity_rows,
    }
    return {
        "candidate": candidate,
        "aggregate": _mean_metrics(observations),
        "cases": cases,
        "identity": identity,
        "observations": observations,
    }


def _aggregate_seed_runs(policy: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {
        name: float(np.mean([run["aggregate"][name] for run in runs]))
        for name in _METRICS
    }
    aggregate_std = {
        name: float(np.std([run["aggregate"][name] for run in runs]))
        for name in _METRICS
    }
    cases = {
        case: {
            name: float(np.mean([run["cases"][case][name] for run in runs]))
            for name in _METRICS
        }
        for case in OBJECTSTATE_MODEL_DIAGNOSTIC_CASES
    }
    case_std = {
        case: {
            name: float(np.std([run["cases"][case][name] for run in runs]))
            for name in _METRICS
        }
        for case in OBJECTSTATE_MODEL_DIAGNOSTIC_CASES
    }
    identity = {
        name: float(np.mean([run["identity"][name] for run in runs]))
        for name in _IDENTITY_METRICS
    }
    identity_std = {
        name: float(np.std([run["identity"][name] for run in runs]))
        for name in _IDENTITY_METRICS
    }
    return {
        "candidate": policy,
        "seed_count": len(runs),
        "aggregate": aggregate,
        "aggregate_std": aggregate_std,
        "cases": cases,
        "cases_std": case_std,
        "identity": identity,
        "identity_std": identity_std,
        "seeds": runs,
    }


def _evaluate_m2_reproduction(
    dataset: MultiObjectSyntheticDataset,
    variant_assignments: Mapping[str, Mapping[int, np.ndarray]],
    *,
    m2_native_assignments: Mapping[int, np.ndarray],
    seeds: tuple[int, ...],
    connected_component_radius: float,
    baseline_seed: int,
) -> dict[str, Any]:
    if set(variant_assignments) != set(OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES):
        raise ValueError("M2 reproduction requires all diagnostic feature policies")
    if set(m2_native_assignments) != set(seeds):
        raise ValueError("M2 native anchor must cover all configured seeds")
    baselines: dict[str, Any] | None = None
    variants: dict[str, Any] = {}
    matrix = []
    for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES:
        if set(variant_assignments[policy]) != set(seeds):
            raise ValueError(f"M2 {policy} does not cover all configured seeds")
        runs = []
        for seed in seeds:
            benchmark = evaluate_multi_object_instance_benchmark(
                dataset,
                variant_assignments[policy][seed],
                connected_component_radius=connected_component_radius,
                seed=baseline_seed,
            )
            if baselines is None:
                baselines = {
                    name: {"candidate": name, "aggregate": benchmark["aggregate"][name]}
                    for name in OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES
                }
            metrics = benchmark["aggregate"]["objectstate_model_v0"]
            runs.append({"seed": seed, "aggregate": metrics})
            matrix.append(
                {
                    "cohort": "m2_original",
                    "policy": policy,
                    "seed": seed,
                    **{name: float(metrics[name]) for name in _METRICS},
                    "slot_swap_rate": None,
                    "unmapped_identity_rate": None,
                }
            )
        variants[policy] = _aggregate_m2_seed_runs(policy, runs)

    native_runs = []
    for seed in seeds:
        benchmark = evaluate_multi_object_instance_benchmark(
            dataset,
            m2_native_assignments[seed],
            connected_component_radius=connected_component_radius,
            seed=baseline_seed,
        )
        metrics = benchmark["aggregate"]["objectstate_model_v0"]
        native_runs.append({"seed": seed, "aggregate": metrics})
        matrix.append(
            {
                "cohort": "m2_original",
                "policy": "native_xyz_rgb_opacity",
                "seed": seed,
                **{name: float(metrics[name]) for name in _METRICS},
                "slot_swap_rate": None,
                "unmapped_identity_rate": None,
            }
        )
    if baselines is None:
        raise ValueError("M2 reproduction did not evaluate baselines")
    native = _aggregate_m2_seed_runs("native_xyz_rgb_opacity", native_runs)
    comparison = _m2_comparison(baselines, variants, native=native)
    return {
        "dataset_schema": dataset.schema,
        "split": dataset.as_dict()["split"],
        "baselines": baselines,
        "variants": variants,
        "native_anchor": native,
        "comparison": comparison,
        "ablation_matrix": matrix,
        "claim_policy": {
            "reproduces_original_m2_distribution": True,
            "native_anchor_includes_opacity": True,
            "fixed_split_across_initialization_seeds": True,
            "does_not_test_cross_view_identity": True,
        },
    }


def _aggregate_m2_seed_runs(policy: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate": policy,
        "seed_count": len(runs),
        "aggregate": {
            name: float(np.mean([run["aggregate"][name] for run in runs]))
            for name in _METRICS
        },
        "aggregate_std": {
            name: float(np.std([run["aggregate"][name] for run in runs]))
            for name in _METRICS
        },
        "seeds": runs,
    }


def _m2_comparison(
    baselines: Mapping[str, Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
    *,
    native: Mapping[str, Any],
) -> dict[str, Any]:
    best_baseline = max(
        baselines,
        key=lambda name: float(baselines[name]["aggregate"]["hungarian_mean_iou"]),
    )
    baseline_score = float(baselines[best_baseline]["aggregate"]["hungarian_mean_iou"])
    best_variant = max(
        variants,
        key=lambda name: float(variants[name]["aggregate"]["hungarian_mean_iou"]),
    )
    per_variant = {}
    for policy, result in variants.items():
        seed_scores = [
            float(run["aggregate"]["hungarian_mean_iou"])
            for run in result["seeds"]
        ]
        score = float(result["aggregate"]["hungarian_mean_iou"])
        per_variant[policy] = {
            "hungarian_mean_iou": score,
            "delta_vs_best_baseline": score - baseline_score,
            "beats_best_baseline_all_seeds": all(
                seed_score > baseline_score + 1e-6 for seed_score in seed_scores
            ),
        }
    native_seed0 = next(
        run for run in native["seeds"] if int(run["seed"]) == 0
    )
    return {
        "best_baseline": best_baseline,
        "best_baseline_hungarian_mean_iou": baseline_score,
        "best_variant": best_variant,
        "best_variant_hungarian_mean_iou": float(
            variants[best_variant]["aggregate"]["hungarian_mean_iou"]
        ),
        "per_variant": per_variant,
        "native_anchor_mean_hungarian_mean_iou": float(
            native["aggregate"]["hungarian_mean_iou"]
        ),
        "native_anchor_seed0_hungarian_mean_iou": float(
            native_seed0["aggregate"]["hungarian_mean_iou"]
        ),
        "native_anchor_seed0_delta": float(
            native_seed0["aggregate"]["hungarian_mean_iou"] - baseline_score
        ),
    }


def _comparison(
    baselines: Mapping[str, Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    best_baseline = max(
        baselines,
        key=lambda name: float(baselines[name]["aggregate"]["hungarian_mean_iou"]),
    )
    best_baseline_score = float(
        baselines[best_baseline]["aggregate"]["hungarian_mean_iou"]
    )
    best_variant = max(
        variants,
        key=lambda name: float(variants[name]["aggregate"]["hungarian_mean_iou"]),
    )
    best_variant_score = float(
        variants[best_variant]["aggregate"]["hungarian_mean_iou"]
    )
    per_variant = {}
    for policy, result in variants.items():
        score = float(result["aggregate"]["hungarian_mean_iou"])
        seed_scores = [
            float(row["aggregate"]["hungarian_mean_iou"])
            for row in result["seeds"]
        ]
        per_variant[policy] = {
            "hungarian_mean_iou": score,
            "delta_vs_best_baseline": score - best_baseline_score,
            "beats_best_baseline_all_seeds": all(
                value > best_baseline_score + 1e-6 for value in seed_scores
            ),
        }
    superiority = bool(
        best_variant_score > best_baseline_score + 1e-6
        and per_variant[best_variant]["beats_best_baseline_all_seeds"]
    )
    return {
        "best_baseline": best_baseline,
        "best_baseline_hungarian_mean_iou": best_baseline_score,
        "best_variant": best_variant,
        "best_variant_hungarian_mean_iou": best_variant_score,
        "best_variant_delta": best_variant_score - best_baseline_score,
        "per_variant": per_variant,
        "model_superiority_established": superiority,
        "verdict": (
            "model_variant_beats_recorded_baselines_all_seeds"
            if superiority
            else "no_model_variant_beats_recorded_baselines_all_seeds"
        ),
    }


def _combined_comparison(
    m2: Mapping[str, Any],
    hard_variants: Mapping[str, Mapping[str, Any]],
    hard_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    m2_comparison = m2["comparison"]
    policies = {}
    for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES:
        m2_pass = bool(
            m2_comparison["per_variant"][policy]["beats_best_baseline_all_seeds"]
        )
        hard_pass = bool(
            hard_comparison["per_variant"][policy]["beats_best_baseline_all_seeds"]
        )
        identity_pass = all(
            float(run["identity"]["slot_swap_rate"]) == 0.0
            and float(run["identity"]["unmapped_identity_rate"]) == 0.0
            for run in hard_variants[policy]["seeds"]
        )
        policies[policy] = {
            "m2_segmentation_beats_baseline_all_seeds": m2_pass,
            "hard_case_segmentation_beats_baseline_all_seeds": hard_pass,
            "cross_view_identity_passes_all_seeds": identity_pass,
            "objectstate_superiority_pass": m2_pass and hard_pass and identity_pass,
        }
    passing = [
        policy for policy, result in policies.items()
        if result["objectstate_superiority_pass"]
    ]
    segmentation_winners = [
        policy for policy, result in policies.items()
        if result["m2_segmentation_beats_baseline_all_seeds"]
        and result["hard_case_segmentation_beats_baseline_all_seeds"]
    ]
    if passing:
        verdict = "model_variant_beats_baselines_and_preserves_cross_view_identity"
    elif segmentation_winners:
        verdict = "per_frame_segmentation_gain_but_cross_view_identity_failed"
    else:
        verdict = "no_model_variant_beats_recorded_segmentation_baselines"
    return {
        "m2": m2_comparison,
        "hard_case": dict(hard_comparison),
        "per_policy": policies,
        "segmentation_winner_policies": segmentation_winners,
        "objectstate_superiority_policies": passing,
        "per_frame_segmentation_superiority_established": bool(segmentation_winners),
        "cross_view_identity_persistence_established": bool(passing),
        "model_superiority_established": bool(passing),
        "verdict": verdict,
    }


def _diagnosis(
    m2: Mapping[str, Any],
    hard_baselines: Mapping[str, Mapping[str, Any]],
    hard_variants: Mapping[str, Mapping[str, Any]],
    hard_comparison: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    xyz = m2["variants"]["xyz_only"]
    combined = m2["variants"]["xyz_rgb"]
    semantic = m2["variants"]["xyz_rgb_semantic"]
    color_effect = float(
        combined["aggregate"]["hungarian_mean_iou"]
        - xyz["aggregate"]["hungarian_mean_iou"]
    )
    semantic_effect = float(
        semantic["aggregate"]["hungarian_mean_iou"]
        - combined["aggregate"]["hungarian_mean_iou"]
    )
    connected = hard_baselines["connected_components_3d"]
    best_variant = hard_variants[str(hard_comparison["best_variant"])]
    case_gaps = {
        case: float(
            connected["cases"][case]["hungarian_mean_iou"]
            - best_variant["cases"][case]["hungarian_mean_iou"]
        )
        for case in OBJECTSTATE_MODEL_DIAGNOSTIC_CASES
    }
    findings = []
    if m2["comparison"]["native_anchor_seed0_delta"] < 0.0:
        findings.append("native_m2_result_remains_below_connected_components")
    if abs(color_effect) <= 0.02:
        findings.append("rgb_addition_has_no_material_aggregate_effect")
    elif color_effect > 0.02:
        findings.append("rgb_addition_improves_aggregate_assignment")
    else:
        findings.append("rgb_addition_hurts_aggregate_assignment")
    if semantic_effect > 0.02:
        findings.append("class_semantic_proxy_improves_aggregate_assignment")
    elif semantic_effect < -0.02:
        findings.append("class_semantic_proxy_hurts_aggregate_assignment")
    else:
        findings.append("class_semantic_proxy_has_no_material_aggregate_effect")
    if connected["identity"]["slot_swap_rate"] > best_variant["identity"]["slot_swap_rate"]:
        findings.append("connected_components_has_higher_cross_view_slot_swap_rate")
    if any(value < -0.02 for case, value in case_gaps.items() if case != "cross_view"):
        findings.append("connected_components_advantage_reverses_on_contact_hard_cases")
    if not comparison["per_frame_segmentation_superiority_established"]:
        findings.append("m2_segmentation_blocks_model_superiority")
    if not comparison["cross_view_identity_persistence_established"]:
        findings.append("cross_view_identity_remains_unproven")
    return {
        "material_effect_threshold": 0.02,
        "xyz_rgb_minus_xyz_only_hungarian_mean_iou": color_effect,
        "semantic_minus_xyz_rgb_hungarian_mean_iou": semantic_effect,
        "m2_native_seed0_hungarian_mean_iou": m2["comparison"][
            "native_anchor_seed0_hungarian_mean_iou"
        ],
        "m2_native_seed0_delta_vs_connected_components": m2["comparison"][
            "native_anchor_seed0_delta"
        ],
        "connected_components_minus_best_variant_by_case": case_gaps,
        "findings": findings,
        "next_decision": (
            "test_non_oracle_semantic_and_temporal_evidence_without_scaling_model"
            if semantic_effect > 0.02
            else "class_semantics_not_supported_as_sufficient_fix; diagnose_relations_and_temporal_identity"
        ),
    }


def _leakage_gate(
    dataset: ObjectStateModelDiagnosticDataset,
    *,
    m2_dataset: MultiObjectSyntheticDataset,
    feature_orders: Mapping[str, Sequence[str]],
    seeds: tuple[int, ...],
    variant_assignments: Mapping[str, Mapping[int, np.ndarray]],
    m2_variant_assignments: Mapping[str, Mapping[int, np.ndarray]],
    m2_native_assignments: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    manifest = dataset.as_dict()
    observations = manifest["observations"]
    heldout_cases = {row["case"] for row in observations if row["split"] == "heldout"}
    semantic = diagnostic_semantic_proxy(dataset.cloud)
    target = dataset.cloud.vertices["gt_instance_id"].astype(np.int64, copy=False)
    cube0 = np.unique(semantic[target == 0], axis=0)
    cube1 = np.unique(semantic[target == 1], axis=0)
    forbidden = {"gt_instance_id", "instance_id", "object_id"}
    pair = dataset.cross_view_pairs[0] if dataset.cross_view_pairs else {}
    observation_by_id = {int(row["scene_id"]): row for row in observations}
    checks = {
        "m2_complete_scene_split": not bool(
            set(m2_dataset.train_scene_ids) & set(m2_dataset.heldout_scene_ids)
        ),
        "m2_independent_instance_target": (
            m2_dataset.as_dict()["source"]["target_derived_from_rgb"] is False
        ),
        "complete_scene_split": manifest["split"]["scene_overlap_count"] == 0,
        "complete_layout_split": manifest["split"]["layout_overlap_count"] == 0,
        "independent_instance_target": (
            manifest["source"]["type"] == "procedural_instance_authorship"
            and manifest["source"]["target_derived_from_rgb"] is False
        ),
        "target_excluded_from_feature_orders": all(
            forbidden.isdisjoint(order) for order in feature_orders.values()
        ),
        "semantic_proxy_shared_by_same_class_cubes": (
            cube0.shape == (1, 3)
            and cube1.shape == (1, 3)
            and np.array_equal(cube0, cube1)
        ),
        "all_required_hard_cases_present": heldout_cases
        == set(OBJECTSTATE_MODEL_DIAGNOSTIC_CASES),
        "cross_view_pair_is_heldout_only": bool(
            pair
            and pair["layout_id"] in dataset.heldout_layout_ids
            and pair["layout_id"] not in dataset.train_layout_ids
            and observation_by_id[int(pair["anchor_scene_id"])]["split"] == "heldout"
            and observation_by_id[int(pair["target_scene_id"])]["split"] == "heldout"
        ),
        "all_instances_observable": all(
            set(int(value) for value in row["instance_ids"]) == {0, 1, 2, 3}
            and all(int(count) > 0 for count in row["observed_point_counts"].values())
            for row in observations
        ),
        "all_configured_seeds_present": all(
            set(variant_assignments[policy]) == set(seeds)
            for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES
        ),
        "m2_all_configured_seeds_present": (
            set(m2_native_assignments) == set(seeds)
            and all(
                set(m2_variant_assignments[policy]) == set(seeds)
                for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES
            )
        ),
        "no_best_seed_selection": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": [
            {"name": name, "passed": bool(passed)}
            for name, passed in checks.items()
        ],
    }


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("metric aggregation requires at least one row")
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in _METRICS
    }


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"model diagnostic requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
