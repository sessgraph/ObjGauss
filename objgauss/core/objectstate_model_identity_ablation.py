from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
)
from objgauss.core.features import colors, opacity, positions
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.objectstate_model_identity_benchmark import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
    ObjectStateModelIdentityBenchmarkScenario,
    objectstate_model_identity_benchmark_summary,
    validate_objectstate_model_identity_benchmark_summary,
)
from objgauss.core.objectstate_model_identity_benchmark_report import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES,
    objectstate_model_identity_benchmark_report_difficulty_by_scenario,
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.core.objectstate_model_identity_gate import OBJECTSTATE_MODEL_IDENTITY_BASELINES

OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA = (
    "objgauss-objectstate-model-identity-ablation-v1"
)
DEFAULT_OBJECTSTATE_MODEL_IDENTITY_ABLATION_POLICIES = (
    "xyz",
    "rgb",
    "xyz_rgb",
    "xyz_rgb_opacity",
    "semantic",
)
_NATIVE_POLICIES = {"xyz", "rgb", "xyz_rgb", "xyz_rgb_opacity"}
_IDENTITY_METRIC_KEYS = (
    "identity_retrieval_at_1",
    "identity_margin",
    "slot_swap_rate",
    "objectstate_drift",
    "assignment_consistency",
    "occlusion_recovery",
)
_CLAIM_POLICY_KEYS = (
    "reuses_identity_benchmark_report_scenarios",
    "compares_feature_evidence_policies",
    "physical_identity_labels_are_evaluation_only",
    "assignment_matrix_is_single_source_of_truth",
    "semantic_policy_uses_synthetic_report_features_only",
    "does_not_claim_real_data_identity_pass",
    "does_not_claim_temporal_assignment",
    "does_not_claim_prediction_or_causal_gate",
    "does_not_claim_world_model",
)
_NON_GOAL_KEYS = (
    "trains_model",
    "runs_long_training",
    "uses_renderer_loss",
    "uses_temporal_loss",
    "uses_hungarian_dependency",
    "uses_transformer",
    "uses_slot_attention",
    "uses_replay_buffer",
    "uses_diffusion",
    "uses_dynamics_model",
    "ingests_real_capture",
    "mutates_viewer_defaults",
)


@dataclass(frozen=True)
class _IdentityPolicySpec:
    name: str
    feature_blocks: tuple[str, ...]
    feature_weight: float
    position_weight: float
    complexity_rank: int


_POLICY_SPECS = {
    "xyz": _IdentityPolicySpec("xyz", ("constant",), 0.0, 1.0, 1),
    "rgb": _IdentityPolicySpec("rgb", ("rgb",), 1.0, 0.0, 2),
    "xyz_rgb": _IdentityPolicySpec("xyz_rgb", ("rgb",), 1.0, 1.0, 3),
    "xyz_rgb_opacity": _IdentityPolicySpec("xyz_rgb_opacity", ("rgb", "opacity"), 1.0, 1.0, 4),
    "semantic": _IdentityPolicySpec("semantic", ("semantic",), 1.0, 0.0, 5),
    "xyz_rgb_opacity_semantic": _IdentityPolicySpec(
        "xyz_rgb_opacity_semantic",
        ("rgb", "opacity", "semantic"),
        1.0,
        1.0,
        6,
    ),
}


def objectstate_model_identity_ablation_summary(
    output_dir: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    sample_id: str = "objectstate-model-identity-ablation-001",
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario] | None = None,
    policies: Sequence[str] = DEFAULT_OBJECTSTATE_MODEL_IDENTITY_ABLATION_POLICIES,
    seed: int = 0,
) -> dict[str, Any]:
    scenario_list = tuple(scenarios or objectstate_model_identity_benchmark_report_scenarios())
    _validate_full_ladder(scenario_list)
    policy_specs = _resolve_policies(policies, scenarios=scenario_list)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = (
        Path(artifact_dir)
        if artifact_dir is not None
        else output_root / "identity-ablation-artifacts"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    variants = []
    for index, spec in enumerate(policy_specs):
        policy_scenarios = tuple(_scenario_with_policy(spec, scenario) for scenario in scenario_list)
        solver_state = _solver_state_for_policy(spec, policy_scenarios)
        benchmark = objectstate_model_identity_benchmark_summary(
            policy_scenarios,
            solver_state,
            output_dir=artifact_root / _safe_policy_name(spec.name),
            sample_id=f"{sample_id}:{spec.name}",
            evidence_policy=spec.name,
            evidence_policy_source="identity_ablation_feature_policy",
            native_gaussian_evidence_only=spec.name in _NATIVE_POLICIES,
            uses_semantic_evidence="semantic" in spec.feature_blocks,
            seed=int(seed) + index * 37,
        )
        variants.append(_variant_summary(spec, benchmark))

    ranking = _policy_ranking(variants)
    summary_path = output_root / "identity-ablation-summary.json"
    payload = {
        "schema": OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA,
        "kind": "objectstate_model_identity_ablation",
        "status": _ablation_status(ranking),
        "sample_id": str(sample_id),
        "benchmark_schema": OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
        "scenario_count": int(len(scenario_list)),
        "identity_pair_count": int(max(variant["identity_pair_count"] for variant in variants)),
        "difficulty_levels": list(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES),
        "perturbation_kinds": list(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS),
        "policies": [_policy_summary(spec) for spec in policy_specs],
        "variants": variants,
        "policy_ranking": ranking,
        "shortcut_diagnostics": _shortcut_diagnostics(variants, ranking),
        "next_stage_gate": _next_stage_gate(ranking),
        "artifact_refs": {
            "identity_ablation_summary": str(summary_path),
            "identity_ablation_artifacts": str(artifact_root),
            "policy_benchmark_summaries": {
                variant["policy"]: variant["artifact_refs"]["raw_benchmark_summary"]
                for variant in variants
            },
        },
        "claim_policy": {key: True for key in _CLAIM_POLICY_KEYS},
        "non_goals": {key: False for key in _NON_GOAL_KEYS},
    }
    checked = validate_objectstate_model_identity_ablation_summary(payload)
    summary_path.write_text(
        json.dumps(checked, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_model_identity_ablation_summary(checked)


def validate_objectstate_model_identity_ablation_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("model identity ablation summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA:
        raise ValueError(f"unsupported model identity ablation schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_model_identity_ablation":
        raise ValueError("model identity ablation kind is unsupported")
    if payload.get("status") not in {
        "objectstate_model_identity_ablation_native_candidate_ready",
        "objectstate_model_identity_ablation_teacher_evidence_indicated",
        "objectstate_model_identity_ablation_blocked",
    }:
        raise ValueError("model identity ablation status is unsupported")
    if payload.get("benchmark_schema") != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA:
        raise ValueError("model identity ablation must reference benchmark schema")
    if int(payload.get("scenario_count", 0)) != 15:
        raise ValueError("model identity ablation requires the 15-scenario report ladder")
    if int(payload.get("identity_pair_count", 0)) < 1:
        raise ValueError("model identity ablation requires identity pairs")
    if tuple(payload.get("difficulty_levels", ())) != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES:
        raise ValueError("model identity ablation difficulty levels are incomplete")
    if tuple(payload.get("perturbation_kinds", ())) != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        raise ValueError("model identity ablation perturbation kinds are incomplete")

    policies = payload.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("model identity ablation requires policies")
    policy_names = [str(item.get("policy")) for item in policies if isinstance(item, Mapping)]
    if len(policy_names) != len(set(policy_names)):
        raise ValueError("model identity ablation policies must be unique")

    variants = payload.get("variants")
    if not isinstance(variants, list) or {item.get("policy") for item in variants} != set(policy_names):
        raise ValueError("model identity ablation variants must match policies")
    for variant in variants:
        _validate_variant(variant)

    ranking = _mapping(payload, "policy_ranking")
    _mapping(ranking, "minimum_sufficient_evidence")
    if not isinstance(ranking.get("policies_by_identity_retrieval"), list) or not ranking["policies_by_identity_retrieval"]:
        raise ValueError("model identity ablation requires policy ranking rows")

    diagnostics = _mapping(payload, "shortcut_diagnostics")
    if not isinstance(diagnostics.get("notes"), list):
        raise ValueError("model identity ablation shortcut diagnostics require notes")

    next_stage = _mapping(payload, "next_stage_gate")
    if next_stage.get("long_training_allowed") is not False:
        raise ValueError("model identity ablation must not directly allow long training")

    artifact_refs = _mapping(payload, "artifact_refs")
    for key in ("identity_ablation_summary", "identity_ablation_artifacts", "policy_benchmark_summaries"):
        if key not in artifact_refs:
            raise ValueError(f"model identity ablation missing artifact ref {key}")

    claim_policy = _mapping(payload, "claim_policy")
    if any(not claim_policy.get(key) for key in _CLAIM_POLICY_KEYS):
        raise ValueError("model identity ablation must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(non_goals.get(key)) for key in _NON_GOAL_KEYS):
        raise ValueError("model identity ablation cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("model identity ablation summary_path must be a string")
    return dict(payload)


def _validate_full_ladder(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> None:
    if len(scenarios) != 15:
        raise ValueError("model identity ablation must reuse the 15 report scenarios")
    expected_ids = set(objectstate_model_identity_benchmark_report_difficulty_by_scenario())
    scenario_ids = {str(scenario.scenario_id) for scenario in scenarios}
    if scenario_ids != expected_ids:
        raise ValueError("model identity ablation scenarios must match the report ladder")
    perturbations = {str(scenario.perturbation_kind) for scenario in scenarios}
    if perturbations != set(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS):
        raise ValueError("model identity ablation scenarios must cover all perturbations")


def _resolve_policies(
    policies: Sequence[str],
    *,
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> tuple[_IdentityPolicySpec, ...]:
    if not policies:
        raise ValueError("model identity ablation requires at least one policy")
    resolved = []
    seen = set()
    semantic_available = all(
        scenario.frame0_features is not None and scenario.frame1_features is not None
        for scenario in scenarios
    )
    for item in policies:
        name = str(item)
        if name in seen:
            raise ValueError(f"duplicate model identity ablation policy: {name}")
        if name not in _POLICY_SPECS:
            raise ValueError(f"unsupported model identity ablation policy: {name}")
        spec = _POLICY_SPECS[name]
        if "semantic" in spec.feature_blocks and not semantic_available:
            raise ValueError("semantic identity ablation policy requires scenario feature evidence")
        resolved.append(spec)
        seen.add(name)
    return tuple(resolved)


def _scenario_with_policy(
    spec: _IdentityPolicySpec,
    scenario: ObjectStateModelIdentityBenchmarkScenario,
) -> ObjectStateModelIdentityBenchmarkScenario:
    return replace(
        scenario,
        frame0_features=_features_for_policy(
            spec,
            scenario.frame0_cloud,
            semantic_features=scenario.frame0_features,
        ),
        frame1_features=_features_for_policy(
            spec,
            scenario.frame1_cloud,
            semantic_features=scenario.frame1_features,
        ),
        description=f"{scenario.description}; evidence_policy={spec.name}",
    )


def _features_for_policy(
    spec: _IdentityPolicySpec,
    cloud: GaussianCloud,
    *,
    semantic_features: np.ndarray | None,
) -> np.ndarray:
    blocks = []
    for block in spec.feature_blocks:
        if block == "constant":
            blocks.append(np.ones((cloud.count, 1), dtype=np.float32))
        elif block == "rgb":
            blocks.append(colors(cloud).astype(np.float32, copy=False))
        elif block == "opacity":
            blocks.append(opacity(cloud)[:, None].astype(np.float32, copy=False))
        elif block == "semantic":
            if semantic_features is None:
                raise ValueError("semantic identity ablation policy requires semantic features")
            blocks.append(_feature_matrix(semantic_features, "semantic_features", rows=cloud.count))
        else:
            raise ValueError(f"unsupported identity ablation feature block: {block}")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def _solver_state_for_policy(
    spec: _IdentityPolicySpec,
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> AssignmentSolverV2State:
    reference = scenarios[0]
    slots = _slot_count(reference)
    reference_features = _feature_matrix(
        reference.frame0_features,
        "reference.frame0_features",
        rows=reference.frame0_cloud.count,
    )
    feature_dim = int(reference_features.shape[1])
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=int(slots),
            feature_dim=feature_dim,
            temperature=0.15,
            feature_weight=float(spec.feature_weight),
            position_weight=float(spec.position_weight),
        ),
        feature_centers=(
            _feature_centers_from_matrix(reference_features, slots=slots)
            if spec.feature_weight > 0.0
            else np.zeros((slots, feature_dim), dtype=np.float32)
        ),
        position_centers=(
            _position_centers_from_cloud(reference.frame0_cloud, slots=slots)
            if spec.position_weight > 0.0
            else np.zeros((slots, 3), dtype=np.float32)
        ),
        slot_bias=np.zeros(slots, dtype=np.float32),
        source=f"identity_ablation_{spec.name}_reference",
    )


def _slot_count(scenario: ObjectStateModelIdentityBenchmarkScenario) -> int:
    labels0 = np.asarray(scenario.frame0_identity_labels)
    labels1 = np.asarray(scenario.frame1_identity_labels)
    shared = sorted(set(int(item) for item in labels0) & set(int(item) for item in labels1))
    if len(shared) < 2:
        raise ValueError("model identity ablation requires at least two shared identities")
    return int(len(shared))


def _feature_centers_from_matrix(features: np.ndarray, *, slots: int) -> np.ndarray:
    matrix = _feature_matrix(features, "reference_features", rows=features.shape[0])
    rounded = np.round(matrix.astype(np.float64), 6)
    unique = np.unique(rounded, axis=0).astype(np.float32, copy=False)
    if unique.shape[0] >= slots:
        return unique[:slots].astype(np.float32, copy=False)
    if unique.shape[0] == 0:
        unique = np.zeros((1, matrix.shape[1]), dtype=np.float32)
    padding = np.repeat(unique[-1:, :], slots - unique.shape[0], axis=0)
    return np.concatenate([unique, padding], axis=0).astype(np.float32, copy=False)


def _position_centers_from_cloud(cloud: GaussianCloud, *, slots: int) -> np.ndarray:
    xyz = positions(cloud).astype(np.float32, copy=False)
    if xyz.shape[0] < slots:
        raise ValueError("not enough Gaussians to derive position centers")
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    picks = np.linspace(0, xyz.shape[0] - 1, slots).round().astype(np.int64)
    return xyz[order[picks]].astype(np.float32, copy=False)


def _variant_summary(
    spec: _IdentityPolicySpec,
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_objectstate_model_identity_benchmark_summary(benchmark)
    solver = checked["baselines"]["assignment_solver_v2"]["metrics"]
    xyz = checked["baselines"]["xyz_centroid"]["metrics"]
    random = checked["baselines"]["random_assignment"]["metrics"]
    oracle = checked["baselines"]["oracle_target_assignment"]["metrics"]
    return {
        "policy": spec.name,
        "status": (
            "objectstate_model_identity_ablation_variant_candidate_ready"
            if checked["long_training_gate"]["status"] == "candidate_ready"
            else "objectstate_model_identity_ablation_variant_blocked"
        ),
        "feature_policy": _policy_summary(spec),
        "benchmark_status": checked["status"],
        "scenario_count": int(checked["num_scenarios"]),
        "identity_pair_count": int(checked["num_pairs"]),
        "assignment_solver_metrics": _metric_digest(solver),
        "baseline_metrics": {
            name: _metric_digest(checked["baselines"][name]["metrics"])
            for name in OBJECTSTATE_MODEL_IDENTITY_BASELINES
        },
        "baseline_comparison": {
            "retrieval_lift_vs_xyz_centroid": float(
                solver["identity_retrieval_at_1"] - xyz["identity_retrieval_at_1"]
            ),
            "retrieval_lift_vs_random": float(
                solver["identity_retrieval_at_1"] - random["identity_retrieval_at_1"]
            ),
            "retrieval_gap_to_oracle": float(
                oracle["identity_retrieval_at_1"] - solver["identity_retrieval_at_1"]
            ),
            "occlusion_recovery_lift_vs_random": float(
                solver["occlusion_recovery"] - random["occlusion_recovery"]
            ),
        },
        "perturbation_breakdown": _perturbation_digest(checked),
        "long_training_gate": checked["long_training_gate"],
        "benchmark_digest": _benchmark_digest(checked),
        "artifact_refs": {
            "raw_benchmark_summary": checked["summary_path"],
            "scenario_summaries": checked["artifact_refs"]["scenario_summaries"],
        },
    }


def _policy_ranking(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(
        variants,
        key=lambda item: (
            float(item["assignment_solver_metrics"]["identity_retrieval_at_1"]),
            float(item["assignment_solver_metrics"]["identity_margin"]),
            float(item["assignment_solver_metrics"]["occlusion_recovery"]),
            -float(item["assignment_solver_metrics"]["slot_swap_rate"]),
            -float(item["assignment_solver_metrics"]["objectstate_drift"]),
        ),
        reverse=True,
    )
    candidate_rows = [
        item
        for item in variants
        if item["long_training_gate"]["status"] == "candidate_ready"
    ]
    native_candidates = [
        item for item in candidate_rows if str(item["policy"]) in _NATIVE_POLICIES
    ]
    semantic_candidates = [
        item for item in candidate_rows if str(item["policy"]) not in _NATIVE_POLICIES
    ]
    minimum = sorted(
        candidate_rows,
        key=lambda item: int(item["feature_policy"]["complexity_rank"]),
    )[0] if candidate_rows else None
    minimum_native = sorted(
        native_candidates,
        key=lambda item: int(item["feature_policy"]["complexity_rank"]),
    )[0] if native_candidates else None
    best = sorted_rows[0]
    return {
        "best_policy": {
            "policy": best["policy"],
            "identity_retrieval_at_1": float(
                best["assignment_solver_metrics"]["identity_retrieval_at_1"]
            ),
            "identity_margin": float(best["assignment_solver_metrics"]["identity_margin"]),
            "uses_semantic": bool(best["feature_policy"]["uses_semantic"]),
        },
        "minimum_sufficient_evidence": {
            "found": minimum is not None,
            "policy": None if minimum is None else minimum["policy"],
            "feature_blocks": [] if minimum is None else minimum["feature_policy"]["feature_blocks"],
            "identity_retrieval_at_1": None if minimum is None else float(
                minimum["assignment_solver_metrics"]["identity_retrieval_at_1"]
            ),
            "identity_margin": None if minimum is None else float(
                minimum["assignment_solver_metrics"]["identity_margin"]
            ),
            "uses_semantic": False if minimum is None else bool(
                minimum["feature_policy"]["uses_semantic"]
            ),
        },
        "minimum_native_candidate": {
            "found": minimum_native is not None,
            "policy": None if minimum_native is None else minimum_native["policy"],
            "feature_blocks": [] if minimum_native is None else minimum_native["feature_policy"]["feature_blocks"],
        },
        "candidate_policy_count": int(len(candidate_rows)),
        "native_candidate_policy_count": int(len(native_candidates)),
        "semantic_candidate_policy_count": int(len(semantic_candidates)),
        "policies_by_identity_retrieval": [
            {
                "policy": item["policy"],
                "status": item["status"],
                "identity_retrieval_at_1": float(
                    item["assignment_solver_metrics"]["identity_retrieval_at_1"]
                ),
                "identity_margin": float(item["assignment_solver_metrics"]["identity_margin"]),
                "slot_swap_rate": float(item["assignment_solver_metrics"]["slot_swap_rate"]),
                "objectstate_drift": float(item["assignment_solver_metrics"]["objectstate_drift"]),
                "assignment_consistency": float(
                    item["assignment_solver_metrics"]["assignment_consistency"]
                ),
                "occlusion_recovery": float(item["assignment_solver_metrics"]["occlusion_recovery"]),
                "uses_semantic": bool(item["feature_policy"]["uses_semantic"]),
            }
            for item in sorted_rows
        ],
    }


def _ablation_status(ranking: Mapping[str, Any]) -> str:
    if int(ranking["native_candidate_policy_count"]) > 0:
        return "objectstate_model_identity_ablation_native_candidate_ready"
    if int(ranking["semantic_candidate_policy_count"]) > 0:
        return "objectstate_model_identity_ablation_teacher_evidence_indicated"
    return "objectstate_model_identity_ablation_blocked"


def _shortcut_diagnostics(
    variants: Sequence[Mapping[str, Any]],
    ranking: Mapping[str, Any],
) -> dict[str, Any]:
    by_policy = {str(variant["policy"]): variant for variant in variants}
    xyz = by_policy.get("xyz")
    rgb = by_policy.get("rgb")
    native = by_policy.get("xyz_rgb_opacity") or by_policy.get("xyz_rgb")
    semantic = by_policy.get("semantic") or by_policy.get("xyz_rgb_opacity_semantic")
    notes: list[str] = []
    geometry_shortcut = False
    appearance_shortcut = False
    teacher_indicated = False
    native_candidate = ranking["minimum_native_candidate"]["found"]

    if xyz is not None and _retrieval(xyz) >= 0.75:
        geometry_shortcut = True
        notes.append("xyz_alone_strong_synthetic_geometry_may_be_too_separable")
    if rgb is not None and xyz is not None and _retrieval(rgb) > _retrieval(xyz) + 0.10:
        appearance_shortcut = True
        notes.append("rgb_improves_identity_appearance_shortcut_possible")
    if native is not None and native["long_training_gate"]["status"] == "candidate_ready":
        notes.append("xyz_rgb_opacity_native_evidence_is_candidate")
    if semantic is not None and native is not None and _retrieval(semantic) > _retrieval(native) + 0.10:
        teacher_indicated = True
        notes.append("semantic_evidence_improves_identity_teacher_layer_indicated")
    if semantic is not None and not native_candidate and semantic["long_training_gate"]["status"] == "candidate_ready":
        teacher_indicated = True
        notes.append("native_gaussian_attributes_not_sufficient_without_semantic_evidence")
    if not ranking["minimum_sufficient_evidence"]["found"]:
        notes.append("no_feature_policy_met_identity_candidate_gate")

    return {
        "notes": notes,
        "geometry_shortcut_possible": geometry_shortcut,
        "appearance_shortcut_possible": appearance_shortcut,
        "native_gaussian_evidence_candidate": bool(native_candidate),
        "teacher_evidence_layer_indicated": bool(teacher_indicated),
        "recommended_stressors_if_xyz_strong": [
            "same_space_contact",
            "overlapping_objects",
            "same_geometry_different_identity",
            "object_translation_across_regions",
        ],
        "recommended_stressors_if_rgb_strong": [
            "color_swap",
            "lighting_shift",
            "same_color_different_objects",
            "different_color_same_object",
        ],
    }


def _next_stage_gate(ranking: Mapping[str, Any]) -> dict[str, Any]:
    minimum = ranking["minimum_sufficient_evidence"]
    native = ranking["minimum_native_candidate"]
    return {
        "bounded_long_smoke_contract_recommended": bool(native["found"]),
        "bounded_long_smoke_candidate_policy": native["policy"],
        "native_long_training_gate": "candidate_ready" if native["found"] else "blocked",
        "semantic_long_training_gate": (
            "candidate_ready"
            if minimum["found"] and minimum["uses_semantic"]
            else "blocked"
        ),
        "teacher_evidence_layer_contract_recommended": bool(
            minimum["found"] and minimum["uses_semantic"] and not native["found"]
        ),
        "long_training_allowed": False,
        "long_training_blockers": [
            "requires_separate_bounded_long_smoke_contract",
            "temporal_assignment_not_evaluated",
            "real_controlled_identity_not_evaluated",
        ],
        "recommended_next_prs": [
            "OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-CONTRACT-001",
            "OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-001",
            "OBJECTSTATE-TEMPORAL-ASSIGNMENT-CONTRACT-001",
        ],
    }


def _benchmark_digest(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": benchmark["schema"],
        "kind": benchmark["kind"],
        "status": benchmark["status"],
        "sample_id": benchmark["sample_id"],
        "num_scenarios": benchmark["num_scenarios"],
        "num_pairs": benchmark["num_pairs"],
        "required_perturbations": benchmark["required_perturbations"],
        "perturbation_coverage": benchmark["perturbation_coverage"],
        "baselines": benchmark["baselines"],
        "perturbation_breakdown": benchmark["perturbation_breakdown"],
        "long_training_gate": benchmark["long_training_gate"],
        "artifact_refs": benchmark["artifact_refs"],
        "claim_policy": benchmark["claim_policy"],
        "non_goals": benchmark["non_goals"],
    }


def _perturbation_digest(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for kind, item in benchmark["perturbation_breakdown"].items():
        rows[kind] = {
            "num_scenarios": int(item["num_scenarios"]),
            "num_pairs": int(item["num_pairs"]),
            "assignment_solver_metrics": _metric_digest(
                item["baselines"]["assignment_solver_v2"]["metrics"]
            ),
            "xyz_centroid_metrics": _metric_digest(
                item["baselines"]["xyz_centroid"]["metrics"]
            ),
        }
    return rows


def _metric_digest(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in _IDENTITY_METRIC_KEYS}


def _validate_variant(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("model identity ablation variant must be a mapping")
    if payload.get("status") not in {
        "objectstate_model_identity_ablation_variant_candidate_ready",
        "objectstate_model_identity_ablation_variant_blocked",
    }:
        raise ValueError("model identity ablation variant status is unsupported")
    _mapping(payload, "feature_policy")
    if int(payload.get("scenario_count", 0)) != 15:
        raise ValueError("model identity ablation variant must include 15 scenarios")
    if int(payload.get("identity_pair_count", 0)) < 1:
        raise ValueError("model identity ablation variant requires identity pairs")
    metrics = _mapping(payload, "assignment_solver_metrics")
    for key in _IDENTITY_METRIC_KEYS:
        _finite(metrics.get(key), f"variant.assignment_solver_metrics.{key}")
    baselines = _mapping(payload, "baseline_metrics")
    for name in OBJECTSTATE_MODEL_IDENTITY_BASELINES:
        if name not in baselines:
            raise ValueError(f"model identity ablation variant missing baseline {name}")
    _mapping(payload, "baseline_comparison")
    breakdown = _mapping(payload, "perturbation_breakdown")
    for kind in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        if kind not in breakdown:
            raise ValueError(f"model identity ablation variant missing {kind} breakdown")
    _mapping(payload, "long_training_gate")
    digest = _mapping(payload, "benchmark_digest")
    if digest.get("schema") != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA:
        raise ValueError("model identity ablation variant digest schema is unsupported")
    _mapping(payload, "artifact_refs")


def _policy_summary(spec: _IdentityPolicySpec) -> dict[str, Any]:
    return {
        "policy": spec.name,
        "feature_blocks": list(spec.feature_blocks),
        "uses_position_cost": bool(spec.position_weight > 0.0),
        "uses_feature_cost": bool(spec.feature_weight > 0.0),
        "uses_xyz": bool(spec.position_weight > 0.0),
        "uses_color": "rgb" in spec.feature_blocks,
        "uses_opacity": "opacity" in spec.feature_blocks,
        "uses_semantic": "semantic" in spec.feature_blocks,
        "feature_weight": float(spec.feature_weight),
        "position_weight": float(spec.position_weight),
        "complexity_rank": int(spec.complexity_rank),
    }


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


def _retrieval(variant: Mapping[str, Any]) -> float:
    return float(variant["assignment_solver_metrics"]["identity_retrieval_at_1"])


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"model identity ablation requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _safe_policy_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
