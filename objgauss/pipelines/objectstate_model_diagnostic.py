from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.datasets.objectstate_model_diagnostic_synthetic import (
    OBJECTSTATE_MODEL_DIAGNOSTIC_CASES,
    build_objectstate_model_diagnostic_dataset,
    diagnostic_semantic_proxy,
)
from objgauss.datasets.objectstate_multi_object_synthetic import (
    build_multi_object_synthetic_dataset,
)
from objgauss.evaluation.objectstate_model_diagnostic import (
    OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES,
    OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES,
    evaluate_objectstate_model_diagnostic,
)
from objgauss.pipelines.objectstate_model_v0 import (
    OBJECTSTATE_MODEL_V0_FEATURE_ORDER,
    objectstate_model_v0_features,
    train_objectstate_model_v0,
)

OBJECTSTATE_MODEL_DIAGNOSTIC_RUN_SCHEMA = (
    "objgauss-objectstate-model-diagnostic-run-v1"
)
OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_BUNDLE_SCHEMA = (
    "objgauss-objectstate-model-diagnostic-datasets-v1"
)

__all__ = (
    "OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_BUNDLE_SCHEMA",
    "OBJECTSTATE_MODEL_DIAGNOSTIC_RUN_SCHEMA",
    "ObjectStateModelDiagnosticRun",
    "run_objectstate_model_diagnostic",
)


@dataclass(frozen=True)
class ObjectStateModelDiagnosticRun:
    output_dir: Path
    dataset_manifest_path: Path
    diagnostic_summary_path: Path
    ablation_matrix_path: Path
    hard_case_matrix_path: Path
    error_taxonomy_path: Path
    report_artifact_path: Path
    summary: dict[str, Any]


def run_objectstate_model_diagnostic(
    output_dir: str | Path,
    *,
    run_id: str = "objectstate-model-diagnostic-001",
    points_per_instance: int = 128,
    iterations: int = 240,
    learning_rate: float = 0.08,
    hidden_dim: int = 24,
    seeds: Sequence[int] = (0, 1, 2),
    split_seed: int = 0,
    heldout_stride: int = 3,
    connected_component_radius: float = 0.18,
    dataset_seed: int = 20260713,
) -> ObjectStateModelDiagnosticRun:
    if not run_id:
        raise ValueError("diagnostic run_id is required")
    resolved_seeds = tuple(int(seed) for seed in seeds)
    if len(resolved_seeds) < 3 or len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("diagnostic seeds must contain at least three unique values")
    if heldout_stride != 3 or split_seed % heldout_stride != 0:
        raise ValueError(
            "diagnostic contract requires heldout_stride=3 and split_seed aligned to offset 0"
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = build_objectstate_model_diagnostic_dataset(
        points_per_instance=points_per_instance,
        seed=dataset_seed,
    )
    m2_dataset = build_multi_object_synthetic_dataset(
        scene_count=12,
        points_per_instance=points_per_instance,
        heldout_stride=heldout_stride,
        split_seed=split_seed,
        seed=dataset_seed,
    )
    feature_matrices, feature_orders = _feature_policies(dataset.cloud)
    m2_feature_matrices, m2_feature_orders = _feature_policies(m2_dataset.cloud)
    if feature_orders != m2_feature_orders:
        raise ValueError("M2 and hard-case feature policy contracts drifted")
    assignments, training_records = _train_policy_runs(
        dataset.cloud,
        expected_heldout=dataset.heldout_scene_ids,
        feature_matrices=feature_matrices,
        feature_orders=feature_orders,
        output_root=output_root,
        cohort="hard_case",
        seeds=resolved_seeds,
        split_seed=split_seed,
        heldout_stride=heldout_stride,
        iterations=iterations,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
    )
    m2_assignments, m2_training_records = _train_policy_runs(
        m2_dataset.cloud,
        expected_heldout=m2_dataset.heldout_scene_ids,
        feature_matrices=m2_feature_matrices,
        feature_orders=m2_feature_orders,
        output_root=output_root,
        cohort="m2_original",
        seeds=resolved_seeds,
        split_seed=split_seed,
        heldout_stride=heldout_stride,
        iterations=iterations,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
    )
    m2_native_assignments, m2_native_records = _train_native_m2_runs(
        m2_dataset.cloud,
        expected_heldout=m2_dataset.heldout_scene_ids,
        output_root=output_root,
        seeds=resolved_seeds,
        split_seed=split_seed,
        heldout_stride=heldout_stride,
        iterations=iterations,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
    )

    diagnostic = evaluate_objectstate_model_diagnostic(
        dataset,
        assignments,
        m2_dataset=m2_dataset,
        m2_variant_assignments=m2_assignments,
        m2_native_assignments=m2_native_assignments,
        feature_orders=feature_orders,
        expected_seeds=resolved_seeds,
        connected_component_radius=connected_component_radius,
        baseline_seed=split_seed,
    )
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    datasets = {
        "hard_case": dataset.as_dict(),
        "m2_original": m2_dataset.as_dict(),
    }
    summary = {
        "schema": OBJECTSTATE_MODEL_DIAGNOSTIC_RUN_SCHEMA,
        "kind": "objectstate_model_diagnostic_run",
        "run_id": run_id,
        "generated_at": generated_at,
        "status": diagnostic["status"],
        "datasets": datasets,
        "training": {
            "model_family": "gaussian-object-encoder-assignment-head-v0",
            "iterations": iterations,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "seeds": list(resolved_seeds),
            "split_seed": split_seed,
            "runs": {
                "hard_case": training_records,
                "m2_original": m2_training_records,
                "m2_native_anchor": m2_native_records,
            },
        },
        "diagnostic": diagnostic,
        "claim_policy": {
            "assignment_prototype_not_objectstate_model_claim": True,
            "diagnosis_precedes_model_scaling": True,
            "semantic_proxy_not_deployable_teacher": True,
            "synthetic_result_not_reality_gate": True,
        },
        "report_notes": {
            "chart_map": {
                "ablation_chart": (
                    "original M2 mean held-out Hungarian mIoU by input policy"
                ),
                "hard_case_chart": (
                    "hard-case held-out Hungarian mIoU by case and candidate"
                ),
                "error_table": (
                    "aggregate merge, split, frozen-map swap, and unmapped rates"
                ),
            },
            "omitted_viewer_error_view": (
                "outside this diagnostic slice; machine-readable evidence and the "
                "portable technical report are the canonical consumers"
            ),
        },
    }

    dataset_manifest_path = output_root / "dataset-manifest.json"
    diagnostic_summary_path = output_root / "diagnostic-summary.json"
    ablation_matrix_path = output_root / "ablation-matrix.csv"
    hard_case_matrix_path = output_root / "hard-case-matrix.csv"
    error_taxonomy_path = output_root / "error-taxonomy.csv"
    report_artifact_path = output_root / "report-artifact.json"
    _write_json(
        dataset_manifest_path,
        {
            "schema": OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_BUNDLE_SCHEMA,
            "kind": "objectstate_model_diagnostic_dataset_bundle",
            "datasets": datasets,
            "relation": (
                "original M2 reproduction plus independent hard-case diagnosis"
            ),
        },
    )
    _write_json(diagnostic_summary_path, summary)
    _write_csv(ablation_matrix_path, diagnostic["ablation_matrix"])
    hard_case_rows = _hard_case_rows(diagnostic)
    error_rows = _error_taxonomy_rows(diagnostic)
    _write_csv(hard_case_matrix_path, hard_case_rows)
    _write_csv(error_taxonomy_path, error_rows)
    _write_json(
        report_artifact_path,
        _report_artifact(
            summary,
            hard_case_rows=hard_case_rows,
            error_rows=error_rows,
        ),
    )
    return ObjectStateModelDiagnosticRun(
        output_dir=output_root,
        dataset_manifest_path=dataset_manifest_path,
        diagnostic_summary_path=diagnostic_summary_path,
        ablation_matrix_path=ablation_matrix_path,
        hard_case_matrix_path=hard_case_matrix_path,
        error_taxonomy_path=error_taxonomy_path,
        report_artifact_path=report_artifact_path,
        summary=summary,
    )


def _train_policy_runs(
    cloud: GaussianCloud,
    *,
    expected_heldout: tuple[int, ...],
    feature_matrices: Mapping[str, np.ndarray],
    feature_orders: Mapping[str, tuple[str, ...]],
    output_root: Path,
    cohort: str,
    seeds: tuple[int, ...],
    split_seed: int,
    heldout_stride: int,
    iterations: int,
    learning_rate: float,
    hidden_dim: int,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, list[dict[str, Any]]]]:
    assignments = {
        policy: {} for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES
    }
    records = {
        policy: [] for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES
    }
    for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES:
        for seed in seeds:
            training = train_objectstate_model_v0(
                cloud,
                object_id_field="gt_instance_id",
                frame_field="scene_id",
                hidden_dim=hidden_dim,
                heldout_stride=heldout_stride,
                iterations=iterations,
                learning_rate=learning_rate,
                seed=seed,
                split_seed=split_seed,
                feature_matrix=feature_matrices[policy],
                feature_order=feature_orders[policy],
            )
            if tuple(training.heldout_frame_ids) != expected_heldout:
                raise ValueError(f"{cohort} training split drifted from dataset contract")
            record = _write_training_record(
                training,
                output_root=output_root,
                run_dir=output_root / "models" / cohort / policy / f"seed-{seed}",
                seed=seed,
            )
            assignments[policy][seed] = training.final_assignment
            records[policy].append(record)
    return assignments, records


def _train_native_m2_runs(
    cloud: GaussianCloud,
    *,
    expected_heldout: tuple[int, ...],
    output_root: Path,
    seeds: tuple[int, ...],
    split_seed: int,
    heldout_stride: int,
    iterations: int,
    learning_rate: float,
    hidden_dim: int,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    assignments = {}
    records = []
    for seed in seeds:
        training = train_objectstate_model_v0(
            cloud,
            object_id_field="gt_instance_id",
            frame_field="scene_id",
            hidden_dim=hidden_dim,
            heldout_stride=heldout_stride,
            iterations=iterations,
            learning_rate=learning_rate,
            seed=seed,
            split_seed=split_seed,
        )
        if tuple(training.heldout_frame_ids) != expected_heldout:
            raise ValueError("native M2 split drifted from dataset contract")
        assignments[seed] = training.final_assignment
        records.append(
            _write_training_record(
                training,
                output_root=output_root,
                run_dir=(
                    output_root
                    / "models"
                    / "m2_original"
                    / "native_xyz_rgb_opacity"
                    / f"seed-{seed}"
                ),
                seed=seed,
            )
        )
    return assignments, records


def _write_training_record(
    training,
    *,
    output_root: Path,
    run_dir: Path,
    seed: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    training_path = run_dir / "training-summary.json"
    checkpoint = training.final_state.as_dict(include_arrays=True)
    training_summary = training.as_dict()
    _write_json(checkpoint_path, checkpoint)
    _write_json(training_path, training_summary)
    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path.relative_to(output_root)),
        "training_summary": str(training_path.relative_to(output_root)),
        "training_status": training_summary["status"],
        "feature_order": list(training.final_state.feature_order),
        "split": training_summary["split"],
        "loss": training_summary["loss"],
    }


def _feature_policies(cloud) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    canonical = objectstate_model_v0_features(cloud)
    xyz = canonical[:, :3]
    rgb = canonical[:, 3:6]
    constant_opacity = np.zeros((cloud.count, 1), dtype=np.float32)
    zero_xyz = np.zeros_like(xyz)
    zero_rgb = np.zeros_like(rgb)
    semantic = diagnostic_semantic_proxy(cloud)
    canonical_order = OBJECTSTATE_MODEL_V0_FEATURE_ORDER
    semantic_order = canonical_order + (
        "semantic_cube",
        "semantic_cup",
        "semantic_tool",
    )
    matrices = {
        "xyz_only": np.column_stack([xyz, zero_rgb, constant_opacity]).astype(np.float32),
        "rgb_only": np.column_stack([zero_xyz, rgb, constant_opacity]).astype(np.float32),
        "xyz_rgb": np.column_stack([xyz, rgb, constant_opacity]).astype(np.float32),
        "xyz_rgb_semantic": np.column_stack(
            [xyz, rgb, constant_opacity, semantic]
        ).astype(np.float32),
    }
    orders = {
        "xyz_only": canonical_order,
        "rgb_only": canonical_order,
        "xyz_rgb": canonical_order,
        "xyz_rgb_semantic": semantic_order,
    }
    return matrices, orders


def _hard_case_rows(diagnostic: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    hard_case = diagnostic["hard_case"]
    sources = [
        *( (name, "baseline", hard_case["baselines"][name])
           for name in OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES ),
        *( (name, "model_variant", hard_case["variants"][name])
           for name in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES ),
    ]
    for candidate, candidate_type, result in sources:
        for case in OBJECTSTATE_MODEL_DIAGNOSTIC_CASES:
            metrics = result["cases"][case]
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": candidate_type,
                    "case": case,
                    "hungarian_mean_iou": metrics["hungarian_mean_iou"],
                    "ari": metrics["ari"],
                    "object_recall_iou_0_5": metrics["object_recall_iou_0_5"],
                    "object_count_error": metrics["object_count_error"],
                    "merge_rate": metrics["merge_rate"],
                    "split_rate": metrics["split_rate"],
                }
            )
    return rows


def _error_taxonomy_rows(diagnostic: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    hard_case = diagnostic["hard_case"]
    sources = [
        *( (name, "baseline", hard_case["baselines"][name])
           for name in OBJECTSTATE_MODEL_DIAGNOSTIC_BASELINES ),
        *( (name, "model_variant", hard_case["variants"][name])
           for name in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES ),
    ]
    for candidate, candidate_type, result in sources:
        error_values = {
            "merge": float(result["aggregate"]["merge_rate"]),
            "split": float(result["aggregate"]["split_rate"]),
            "identity_swap": float(result["identity"]["slot_swap_rate"]),
            "unmapped_identity": float(result["identity"]["unmapped_identity_rate"]),
        }
        dominant = max(error_values, key=error_values.get)
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": candidate_type,
                "merge_rate": error_values["merge"],
                "split_rate": error_values["split"],
                "slot_swap_rate": error_values["identity_swap"],
                "unmapped_identity_rate": error_values["unmapped_identity"],
                "dominant_error": dominant,
                "dominant_error_rate": error_values[dominant],
            }
        )
    return rows


def _report_artifact(
    summary: Mapping[str, Any],
    *,
    hard_case_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostic = summary["diagnostic"]
    comparison = diagnostic["comparison"]
    diagnosis = diagnostic["diagnosis"]
    m2 = diagnostic["m2_reproduction"]
    m2_comparison = m2["comparison"]
    hard_case = diagnostic["hard_case"]
    hard_comparison = hard_case["comparison"]
    best_variant = str(hard_comparison["best_variant"])
    best_baseline = str(hard_comparison["best_baseline"])
    ablation_rows = []
    for policy in OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES:
        result = m2["variants"][policy]
        ablation_rows.append(
            {
                "policy": policy,
                "hungarian_mean_iou": result["aggregate"]["hungarian_mean_iou"],
                "hungarian_mean_iou_std": result["aggregate_std"]["hungarian_mean_iou"],
                "ari": result["aggregate"]["ari"],
                "merge_rate": result["aggregate"]["merge_rate"],
                "split_rate": result["aggregate"]["split_rate"],
                "seed_count": result["seed_count"],
                "best_baseline": m2_comparison["best_baseline"],
                "best_baseline_miou": m2_comparison[
                    "best_baseline_hungarian_mean_iou"
                ],
                "delta_vs_best_baseline": m2_comparison["per_variant"][policy][
                    "delta_vs_best_baseline"
                ],
            }
        )
    native = m2["native_anchor"]
    ablation_rows.append(
        {
            "policy": "native_xyz_rgb_opacity",
            "hungarian_mean_iou": native["aggregate"]["hungarian_mean_iou"],
            "hungarian_mean_iou_std": native["aggregate_std"]["hungarian_mean_iou"],
            "ari": native["aggregate"]["ari"],
            "merge_rate": native["aggregate"]["merge_rate"],
            "split_rate": native["aggregate"]["split_rate"],
            "seed_count": native["seed_count"],
            "best_baseline": m2_comparison["best_baseline"],
            "best_baseline_miou": m2_comparison["best_baseline_hungarian_mean_iou"],
            "delta_vs_best_baseline": (
                native["aggregate"]["hungarian_mean_iou"]
                - m2_comparison["best_baseline_hungarian_mean_iou"]
            ),
        }
    )
    ablation_rows.append(
        {
            "policy": f"baseline_{m2_comparison['best_baseline']}",
            "hungarian_mean_iou": m2_comparison[
                "best_baseline_hungarian_mean_iou"
            ],
            "hungarian_mean_iou_std": 0.0,
            "ari": m2["baselines"][m2_comparison["best_baseline"]]["aggregate"][
                "ari"
            ],
            "merge_rate": m2["baselines"][m2_comparison["best_baseline"]][
                "aggregate"
            ]["merge_rate"],
            "split_rate": m2["baselines"][m2_comparison["best_baseline"]][
                "aggregate"
            ]["split_rate"],
            "seed_count": 0,
            "best_baseline": m2_comparison["best_baseline"],
            "best_baseline_miou": m2_comparison[
                "best_baseline_hungarian_mean_iou"
            ],
            "delta_vs_best_baseline": 0.0,
        }
    )
    chart_cases = [
        row
        for row in hard_case_rows
        if row["candidate"] in {
            best_variant,
            "xyz_kmeans",
            "rgb_kmeans",
            "connected_components_3d",
        }
    ]
    headline = [
        {
            "m2_native_seed0_miou": m2_comparison[
                "native_anchor_seed0_hungarian_mean_iou"
            ],
            "m2_best_baseline_miou": m2_comparison[
                "best_baseline_hungarian_mean_iou"
            ],
            "m2_native_seed0_delta": m2_comparison["native_anchor_seed0_delta"],
            "hard_best_variant_miou": hard_comparison[
                "best_variant_hungarian_mean_iou"
            ],
            "hard_best_baseline_miou": hard_comparison[
                "best_baseline_hungarian_mean_iou"
            ],
            "hard_model_delta": hard_comparison["best_variant_delta"],
            "hard_identity_swap_rate": hard_case["variants"][best_variant]["identity"][
                "slot_swap_rate"
            ],
            "semantic_effect": diagnosis["semantic_minus_xyz_rgb_hungarian_mean_iou"],
            "superiority": 1 if comparison["model_superiority_established"] else 0,
        }
    ]
    report_datasets = {
        "headline": headline,
        "ablation": ablation_rows,
        "hard_cases": chart_cases,
        "errors": error_rows,
    }
    source = {
        "id": "objectstate_diagnostic_summary",
        "label": "ObjGauss canonical model diagnostic outputs",
        "path": "diagnostic-summary.json",
        "query": {
            "engine": "portable_values_sql",
            "language": "sql",
            "sql": _portable_report_source_sql(report_datasets),
            "description": (
                "Runnable VALUES query reconstructing every bounded report row reviewed "
                "from the canonical Python-produced JSON/CSV outputs. Producer command: "
                "uv run --locked objgauss training objectstate-model-diagnostic "
                "--output-dir outputs/model-diagnostics/objectstate-model-diagnostic-001 "
                "--run-id objectstate-model-diagnostic-001 --points-per-instance 128 "
                "--iterations 240 --learning-rate 0.08 --hidden-dim 24 --seeds 0 1 2 "
                "--split-seed 0 --heldout-stride 3 --connected-component-radius 0.18 "
                "--dataset-seed 20260713."
            ),
            "tables_used": [
                "diagnostic-summary.json",
                "ablation-matrix.csv",
                "hard-case-matrix.csv",
                "error-taxonomy.csv",
            ],
            "filters": [
                "held-out scene ids 0,3,6,9,12",
                "three deterministic training seeds",
                "four instances per observation",
            ],
            "metric_definitions": {
                "hungarian_mean_iou": (
                    "Mean target-instance IoU after maximum-IoU bipartite matching; "
                    "unmatched target instances receive zero."
                ),
                "slot_swap_rate": (
                    "Fraction of identities whose dominant target-view slot maps to a "
                    "different identity under the frozen anchor-view mapping."
                ),
            },
            "executed_at": summary["generated_at"],
        },
    }
    title = "Why ObjectState Model v0 loses: diagnostic 001"
    verdict = str(comparison["verdict"])
    summary_text = (
        f"## Technical summary\n\n"
        f"**Result:** `{verdict}`. On the original M2 distribution, native seed 0 "
        f"reproduces **{m2_comparison['native_anchor_seed0_hungarian_mean_iou']:.3f}** "
        f"Hungarian mIoU versus **{m2_comparison['best_baseline_hungarian_mean_iou']:.3f}** "
        f"for `{m2_comparison['best_baseline']}`. On hard cases, `{best_variant}` reaches "
        f"**{hard_comparison['best_variant_hungarian_mean_iou']:.3f}**, but its frozen-map "
        f"cross-view swap rate is **{hard_case['variants'][best_variant]['identity']['slot_swap_rate']:.3f}**.\n\n"
        "The experiment diagnoses the existing tanh encoder/assignment head across all "
        "configured seeds. Per-frame segmentation can beat simple baselines on contact cases, "
        "but persistent same-class identity remains unproven."
    )
    findings_text = (
        "## Feature evidence does not justify scaling the model\n\n"
        f"Adding RGB to XYZ changes aggregate mIoU by "
        f"**{diagnosis['xyz_rgb_minus_xyz_only_hungarian_mean_iou']:+.3f}**; adding the "
        f"oracle-style class semantic proxy changes it by "
        f"**{diagnosis['semantic_minus_xyz_rgb_hungarian_mean_iou']:+.3f}**. "
        "The semantic proxy is shared by both red cubes, so it cannot encode instance identity."
    )
    hard_text = (
        "## The baseline advantage is case-dependent\n\n"
        "The grouped hard-case comparison attacks the spatial-separation shortcut with "
        "same-class proximity, cube–cup contact, partial occlusion, and a two-view layout. "
        "Connected components remains perfect on separated cross-view frames but merges contact "
        "cases; Model v0 reverses some per-frame gaps without preserving cube identity."
    )
    error_text = (
        "## Merge, split, and identity swap expose different failures\n\n"
        "Per-frame Hungarian matching removes label permutation for segmentation metrics. "
        "Identity swaps are therefore evaluated separately by freezing the anchor-view slot "
        "mapping before scoring the target view."
    )
    scope_text = (
        "## Scope, definitions, and experimental design\n\n"
        "The analysis has two cohorts: the original 12-scene M2 distribution and a 13-observation "
        "hard-case distribution with five held-out observations. Both contain four independently "
        "authored instances and complete-scene holdout. Hard-case train/test layout overlap is "
        "zero. Four feature policies plus the native opacity anchor use the same model family, "
        "fixed split and three initialization seeds."
    )
    methodology_text = (
        "## Metric and validation method\n\n"
        "Per-frame segmentation uses maximum-IoU Hungarian matching, with unmatched targets "
        "scored as zero. Merge and split use a 10% material-overlap threshold. Cross-view identity "
        "freezes the anchor mapping and never rematches the target view. Reported model values are "
        "means and population standard deviations over every configured seed; no best seed is selected."
    )
    limitation_text = (
        "## Limitations and robustness boundary\n\n"
        "This is a small procedural benchmark. The semantic vector is an authored class proxy, "
        "not output from a deployable teacher. Camera/occlusion behavior is geometric and does "
        "not reproduce sensor noise. Mean and population standard deviation cover initialization "
        "seeds only; they do not estimate real-scene uncertainty."
    )
    next_text = (
        "## Recommended next step\n\n"
        f"**{diagnosis['next_decision']}**. Do not add a Transformer or begin long training until "
        "a non-oracle semantic/temporal/relation slice demonstrates a repeatable gain over the "
        "same baselines without increasing identity swaps."
    )
    questions_text = (
        "## Further questions\n\n"
        "- Can learned semantic evidence reproduce any proxy gain without instance leakage?\n"
        "- Does temporal correspondence reduce same-class cube swaps across camera changes?\n"
        "- Which relation feature separates touching instances while preserving one object under occlusion?"
    )
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical diagnosis of the M2 Model v0 negative result.",
            "generatedAt": summary["generated_at"],
            "cards": [
                {
                    "id": "m2_native_card",
                    "description": "Native XYZ+RGB+opacity seed 0 on the original M2 split.",
                    "dataset": "headline",
                    "sourceId": "objectstate_diagnostic_summary",
                    "metrics": [
                        {
                            "label": "Native M2 mIoU",
                            "field": "m2_native_seed0_miou",
                            "format": "number",
                        },
                        {
                            "label": "vs baseline",
                            "field": "m2_native_seed0_delta",
                            "format": "number",
                            "signed": True,
                        },
                    ],
                },
                {
                    "id": "hard_case_card",
                    "description": f"Hard-case mean for {best_variant}.",
                    "dataset": "headline",
                    "sourceId": "objectstate_diagnostic_summary",
                    "metrics": [
                        {
                            "label": "Hard-case Model mIoU",
                            "field": "hard_best_variant_miou",
                            "format": "number",
                        },
                        {
                            "label": "vs baseline",
                            "field": "hard_model_delta",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "identity_card",
                    "description": "Frozen anchor-map identity swap rate for the best hard-case variant.",
                    "dataset": "headline",
                    "sourceId": "objectstate_diagnostic_summary",
                    "metrics": [
                        {
                            "label": "Cross-view swap rate",
                            "field": "hard_identity_swap_rate",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "semantic_effect_card",
                    "description": "XYZ+RGB+semantic minus XYZ+RGB mIoU.",
                    "dataset": "headline",
                    "sourceId": "objectstate_diagnostic_summary",
                    "metrics": [
                        {
                            "label": "Semantic proxy effect",
                            "field": "semantic_effect",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "ablation_chart",
                    "title": "Original M2 Model v0 input ablation",
                    "subtitle": "Three-seed model means, native opacity anchor, and the fixed best baseline.",
                    "type": "bar",
                    "dataset": "ablation",
                    "sourceId": "objectstate_diagnostic_summary",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "policy", "type": "nominal", "label": "Input policy"},
                        "y": {
                            "field": "hungarian_mean_iou",
                            "type": "quantitative",
                            "label": "Hungarian mIoU",
                        },
                        "tooltip": [
                            {
                                "field": "hungarian_mean_iou_std",
                                "type": "quantitative",
                                "label": "Seed std",
                            },
                            {
                                "field": "delta_vs_best_baseline",
                                "type": "quantitative",
                                "label": "Delta vs baseline",
                            },
                        ],
                    },
                },
                {
                    "id": "hard_case_chart",
                    "title": "Hungarian mIoU by hard case and candidate",
                    "subtitle": "Best model variant compared with all recorded simple baselines.",
                    "type": "bar",
                    "dataset": "hard_cases",
                    "sourceId": "objectstate_diagnostic_summary",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "case", "type": "nominal", "label": "Hard case"},
                        "y": {
                            "field": "hungarian_mean_iou",
                            "type": "quantitative",
                            "label": "Hungarian mIoU",
                        },
                        "color": {
                            "field": "candidate",
                            "type": "nominal",
                            "label": "Candidate",
                        },
                        "tooltip": [
                            {"field": "merge_rate", "type": "quantitative", "label": "Merge rate"},
                            {"field": "split_rate", "type": "quantitative", "label": "Split rate"},
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "error_table",
                    "title": "Aggregate error taxonomy",
                    "subtitle": "Equal-weight held-out observation means plus frozen-map cross-view swaps.",
                    "dataset": "errors",
                    "sourceId": "objectstate_diagnostic_summary",
                    "defaultSort": {"field": "dominant_error_rate", "direction": "desc"},
                    "columns": [
                        {"field": "candidate", "label": "Candidate", "type": "text"},
                        {"field": "candidate_type", "label": "Type", "type": "text"},
                        {"field": "merge_rate", "label": "Merge rate", "format": "number"},
                        {"field": "split_rate", "label": "Split rate", "format": "number"},
                        {"field": "slot_swap_rate", "label": "Swap rate", "format": "number"},
                        {"field": "unmapped_identity_rate", "label": "Unmapped", "format": "number"},
                        {"field": "dominant_error", "label": "Dominant error", "type": "text"},
                        {
                            "field": "dominant_error_rate",
                            "label": "Dominant rate",
                            "format": "number",
                        },
                    ],
                }
            ],
            "sources": [source],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": summary_text,
                    "sourceId": "objectstate_diagnostic_summary",
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "m2_native_card",
                        "hard_case_card",
                        "identity_card",
                        "semantic_effect_card",
                    ],
                },
                {
                    "id": "feature_findings",
                    "type": "markdown",
                    "body": findings_text,
                    "sourceId": "objectstate_diagnostic_summary",
                },
                {"id": "ablation_visual", "type": "chart", "chartId": "ablation_chart"},
                {
                    "id": "hard_case_findings",
                    "type": "markdown",
                    "body": hard_text,
                },
                {"id": "hard_case_visual", "type": "chart", "chartId": "hard_case_chart"},
                {
                    "id": "error_findings",
                    "type": "markdown",
                    "body": error_text,
                },
                {"id": "error_detail", "type": "table", "tableId": "error_table"},
                {"id": "scope", "type": "markdown", "body": scope_text},
                {"id": "methodology", "type": "markdown", "body": methodology_text},
                {"id": "limitations", "type": "markdown", "body": limitation_text},
                {"id": "next_step", "type": "markdown", "body": next_text},
                {"id": "questions", "type": "markdown", "body": questions_text},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": summary["generated_at"],
            "status": "ready",
            "datasets": report_datasets,
            "accessIssues": [],
        },
        "sources": [source],
    }


def _portable_report_source_sql(
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    values = []
    for dataset, rows in datasets.items():
        for row in rows:
            encoded = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            values.append(f"({_sql_literal(dataset)}, {_sql_literal(encoded)})")
    return (
        "WITH report_rows(dataset, row_json) AS (\n  VALUES\n    "
        + ",\n    ".join(values)
        + "\n)\nSELECT dataset, row_json FROM report_rows ORDER BY dataset, row_json"
    )


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty diagnostic CSV: {path.name}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
