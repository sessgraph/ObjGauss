from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_solver_v2 import AssignmentSolverV2Config, AssignmentSolverV2State
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.objectstate_model_identity_benchmark import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
    ObjectStateModelIdentityBenchmarkScenario,
    objectstate_model_identity_benchmark_summary,
    validate_objectstate_model_identity_benchmark_summary,
)
from objgauss.core.objectstate_model_identity_gate import OBJECTSTATE_MODEL_IDENTITY_BASELINES

OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA = (
    "objgauss-objectstate-model-identity-benchmark-report-v1"
)
OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES = ("easy", "medium", "hard")
_METRIC_COLUMNS = (
    "identity_retrieval_at_1",
    "identity_margin",
    "slot_swap_rate",
    "objectstate_drift",
    "assignment_consistency",
    "occlusion_recovery",
)


@dataclass(frozen=True)
class _ScenarioSpec:
    perturbation_kind: str
    difficulty: str
    parameter: float
    description: str


def write_objectstate_model_identity_benchmark_report(
    output_dir: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    sample_id: str = "objectstate-model-identity-benchmark-report-001",
    seed: int = 0,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(artifact_dir) if artifact_dir is not None else output_root / "identity-benchmark-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    difficulty_by_scenario = objectstate_model_identity_benchmark_report_difficulty_by_scenario()
    benchmark = objectstate_model_identity_benchmark_summary(
        scenarios,
        _feature_backed_solver_state(),
        output_dir=artifact_root,
        sample_id=sample_id,
        seed=seed,
    )
    validate_objectstate_model_identity_benchmark_summary(benchmark)

    rows = _breakdown_rows(benchmark, difficulty_by_scenario)
    ranking = _overall_ranking(benchmark)
    report_path = output_root / "identity-benchmark-report.md"
    csv_path = output_root / "identity-benchmark-breakdown.csv"
    summary_path = output_root / "identity-benchmark-summary.json"
    payload = {
        "schema": OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA,
        "kind": "objectstate_model_identity_benchmark_report",
        "status": _report_status(benchmark),
        "sample_id": str(sample_id),
        "benchmark_schema": OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
        "benchmark_status": benchmark["status"],
        "scenario_count": int(benchmark["num_scenarios"]),
        "identity_pair_count": int(benchmark["num_pairs"]),
        "difficulty_levels": list(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES),
        "perturbation_kinds": list(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS),
        "overall_ranking": ranking,
        "long_training_gate": benchmark["long_training_gate"],
        "benchmark_digest": _benchmark_digest(benchmark),
        "artifact_refs": {
            "identity_benchmark_summary": str(summary_path),
            "identity_benchmark_report": str(report_path),
            "identity_benchmark_breakdown": str(csv_path),
            "identity_benchmark_artifacts": str(artifact_root),
            "raw_benchmark_summary": benchmark["summary_path"],
        },
        "claim_policy": {
            "synthetic_controlled_evidence_only": True,
            "physical_identity_labels_are_evaluation_only": True,
            "does_not_claim_real_data_identity_pass": True,
            "does_not_claim_temporal_assignment": True,
            "does_not_claim_world_model": True,
        },
    }
    checked = validate_objectstate_model_identity_benchmark_report_summary(payload)
    _write_breakdown_csv(csv_path, rows)
    report_path.write_text(_render_markdown_report(checked, rows), encoding="utf-8")
    summary_path.write_text(
        json.dumps(checked, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return validate_objectstate_model_identity_benchmark_report_summary(checked)


def objectstate_model_identity_benchmark_report_scenarios() -> tuple[
    ObjectStateModelIdentityBenchmarkScenario, ...
]:
    return tuple(_scenario_from_spec(spec) for spec in _default_scenario_specs())


def objectstate_model_identity_benchmark_report_difficulty_by_scenario() -> dict[str, str]:
    return {
        f"{spec.perturbation_kind}-{spec.difficulty}": spec.difficulty
        for spec in _default_scenario_specs()
    }


def validate_objectstate_model_identity_benchmark_report_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("model identity benchmark report summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA:
        raise ValueError(f"unsupported identity benchmark report schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_model_identity_benchmark_report":
        raise ValueError("model identity benchmark report kind is unsupported")
    if payload.get("status") not in {
        "objectstate_model_identity_benchmark_report_candidate_ready",
        "objectstate_model_identity_benchmark_report_blocked",
    }:
        raise ValueError("model identity benchmark report status is unsupported")
    if payload.get("benchmark_schema") != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA:
        raise ValueError("model identity benchmark report must reference benchmark schema")
    if int(payload.get("scenario_count", 0)) < 15:
        raise ValueError("model identity benchmark report requires the full difficulty ladder")
    if int(payload.get("identity_pair_count", 0)) < 1:
        raise ValueError("model identity benchmark report requires evaluated identity pairs")
    if tuple(payload.get("difficulty_levels", ())) != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES:
        raise ValueError("model identity benchmark report difficulty levels are incomplete")
    if tuple(payload.get("perturbation_kinds", ())) != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        raise ValueError("model identity benchmark report perturbation kinds are incomplete")
    digest = payload.get("benchmark_digest")
    if not isinstance(digest, Mapping):
        raise ValueError("model identity benchmark report requires benchmark_digest")
    _validate_benchmark_digest(digest, expected_scenarios=int(payload["scenario_count"]))
    ranking = payload.get("overall_ranking")
    if not isinstance(ranking, list) or {row.get("baseline") for row in ranking} != set(OBJECTSTATE_MODEL_IDENTITY_BASELINES):
        raise ValueError("model identity benchmark report requires all baseline rankings")
    gate = payload.get("long_training_gate")
    if not isinstance(gate, Mapping) or gate.get("status") not in {"blocked", "candidate_ready"}:
        raise ValueError("model identity benchmark report requires long_training_gate")
    claim_policy = payload.get("claim_policy")
    if not isinstance(claim_policy, Mapping) or any(not bool(value) for value in claim_policy.values()):
        raise ValueError("model identity benchmark report must preserve claim policy")
    artifact_refs = payload.get("artifact_refs")
    if not isinstance(artifact_refs, Mapping):
        raise ValueError("model identity benchmark report requires artifact_refs")
    for key in (
        "identity_benchmark_summary",
        "identity_benchmark_report",
        "identity_benchmark_breakdown",
        "identity_benchmark_artifacts",
    ):
        if not isinstance(artifact_refs.get(key), str) or not artifact_refs[key]:
            raise ValueError(f"model identity benchmark report missing artifact ref {key}")
    return dict(payload)


def _default_scenario_specs() -> tuple[_ScenarioSpec, ...]:
    return (
        _ScenarioSpec("viewpoint", "easy", 15.0, "15 degree camera-like rotation"),
        _ScenarioSpec("viewpoint", "medium", 40.0, "40 degree oblique rotation"),
        _ScenarioSpec("viewpoint", "hard", 70.0, "70 degree oblique rotation"),
        _ScenarioSpec("dropout", "easy", 0.10, "10 percent deterministic Gaussian dropout"),
        _ScenarioSpec("dropout", "medium", 0.30, "30 percent deterministic Gaussian dropout"),
        _ScenarioSpec("dropout", "hard", 0.50, "50 percent deterministic Gaussian dropout"),
        _ScenarioSpec("occlusion", "easy", 0.25, "25 percent deterministic per-object occlusion"),
        _ScenarioSpec("occlusion", "medium", 0.50, "50 percent deterministic per-object occlusion"),
        _ScenarioSpec("occlusion", "hard", 0.75, "75 percent deterministic per-object occlusion"),
        _ScenarioSpec("appearance", "easy", 0.15, "light color and opacity perturbation"),
        _ScenarioSpec("appearance", "medium", 0.35, "medium color and opacity perturbation"),
        _ScenarioSpec("appearance", "hard", 0.65, "heavy color and opacity perturbation"),
        _ScenarioSpec("spatial", "easy", 0.12, "small translation perturbation"),
        _ScenarioSpec("spatial", "medium", 0.35, "medium translation perturbation"),
        _ScenarioSpec("spatial", "hard", 0.70, "rotation plus translation perturbation"),
    )


def _scenario_from_spec(spec: _ScenarioSpec) -> ObjectStateModelIdentityBenchmarkScenario:
    base = _frame_arrays("base", 0.0)
    perturbed = _frame_arrays(spec.perturbation_kind, spec.parameter)
    scenario_id = f"{spec.perturbation_kind}-{spec.difficulty}"
    return ObjectStateModelIdentityBenchmarkScenario(
        scenario_id=scenario_id,
        perturbation_kind=spec.perturbation_kind,
        frame0_cloud=_cloud(base["positions"], base["colors"], base["opacity"]),
        frame0_identity_labels=base["labels"],
        frame1_cloud=_cloud(perturbed["positions"], perturbed["colors"], perturbed["opacity"]),
        frame1_identity_labels=perturbed["labels"],
        frame0_id=f"{scenario_id}:t0",
        frame1_id=f"{scenario_id}:t1",
        frame0_features=base["features"],
        frame1_features=perturbed["features"],
        description=f"{spec.difficulty}: {spec.description}",
    )


def _feature_backed_solver_state() -> AssignmentSolverV2State:
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=4,
            feature_dim=4,
            temperature=0.15,
            feature_weight=1.0,
            position_weight=0.0,
        ),
        feature_centers=np.eye(4, dtype=np.float32),
        position_centers=np.zeros((4, 3), dtype=np.float32),
        slot_bias=np.zeros(4, dtype=np.float32),
        source="identity_benchmark_report_feature_backed_reference",
    )


def _frame_arrays(kind: str, parameter: float) -> dict[str, np.ndarray]:
    labels, positions, part_indices = _base_labels_positions()
    colors = _base_colors(labels)
    opacity = np.ones(labels.shape[0], dtype=np.float32)
    keep = np.ones(labels.shape[0], dtype=bool)
    if kind == "viewpoint":
        positions = _rotate_y(positions, math.radians(float(parameter)))
    elif kind == "dropout":
        keep = _keep_by_fraction(part_indices, labels, keep_fraction=1.0 - float(parameter))
    elif kind == "occlusion":
        keep = _keep_by_fraction(part_indices, labels, keep_fraction=max(0.25, 1.0 - float(parameter)))
    elif kind == "appearance":
        colors = np.clip(colors * (1.0 - parameter) + (1.0 - colors) * parameter, 0.0, 1.0)
        opacity = np.clip(opacity * (1.0 - parameter * 0.35), 0.05, 1.0)
    elif kind == "spatial":
        positions = _rotate_z(positions, math.radians(parameter * 65.0))
        positions = positions + np.asarray([parameter, -parameter * 0.4, parameter * 0.15], dtype=np.float32)
    elif kind != "base":
        raise ValueError(f"unsupported report perturbation kind: {kind}")
    labels = labels[keep]
    return {
        "labels": labels,
        "positions": positions[keep].astype(np.float32, copy=False),
        "features": np.eye(4, dtype=np.float32)[labels],
        "colors": colors[keep].astype(np.float32, copy=False),
        "opacity": opacity[keep].astype(np.float32, copy=False),
    }


def _base_labels_positions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    template = np.asarray(
        [
            [-0.36, -0.22, 0.00],
            [-0.16, 0.24, 0.04],
            [0.15, -0.26, 0.08],
            [0.37, 0.21, 0.12],
            [-0.30, 0.02, 0.18],
            [-0.02, 0.36, 0.22],
            [0.27, -0.02, 0.26],
            [0.03, -0.38, 0.30],
        ],
        dtype=np.float32,
    )
    labels, positions, part_indices = [], [], []
    for identity in range(4):
        offset = np.asarray([0.025 * identity, 0.012 * (identity % 2), 0.015 * identity], dtype=np.float32)
        for part_index, point in enumerate(template):
            labels.append(identity)
            positions.append(point + offset)
            part_indices.append(part_index)
    return np.asarray(labels, dtype=np.int64), np.asarray(positions, dtype=np.float32), np.asarray(part_indices)


def _base_colors(labels: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [[0.90, 0.12, 0.10], [0.10, 0.60, 0.95], [0.20, 0.85, 0.28], [0.86, 0.62, 0.12]],
        dtype=np.float32,
    )
    return palette[labels]


def _keep_by_fraction(part_indices: np.ndarray, labels: np.ndarray, *, keep_fraction: float) -> np.ndarray:
    keep = np.zeros(part_indices.shape[0], dtype=bool)
    for identity in sorted(set(int(label) for label in labels)):
        mask = labels == identity
        parts = part_indices[mask]
        keep_count = max(2, int(math.ceil(parts.shape[0] * keep_fraction)))
        selected = np.sort(parts)[:keep_count]
        keep[mask] = np.isin(parts, selected)
    return keep


def _rotate_y(values: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return values @ matrix.T


def _rotate_z(values: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return values @ matrix.T


def _cloud(positions: np.ndarray, colors: np.ndarray, opacity: np.ndarray) -> GaussianCloud:
    dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "f4"), ("green", "f4"), ("blue", "f4"), ("opacity", "f4")])
    vertices = np.zeros(positions.shape[0], dtype=dtype)
    for index, field in enumerate(("x", "y", "z")):
        vertices[field] = positions[:, index]
    for index, field in enumerate(("red", "green", "blue")):
        vertices[field] = colors[:, index]
    vertices["opacity"] = opacity
    return GaussianCloud(vertices)


def _breakdown_rows(
    benchmark: Mapping[str, Any],
    difficulty_by_scenario: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in benchmark["scenario_results"]:
        scenario_id = str(scenario["scenario_id"])
        for baseline in OBJECTSTATE_MODEL_IDENTITY_BASELINES:
            metrics = scenario["baselines"][baseline]["metrics"]
            rows.append({
                "scenario_id": scenario_id,
                "difficulty": difficulty_by_scenario[scenario_id],
                "perturbation_kind": scenario["perturbation_kind"],
                "baseline": baseline,
                **{key: metrics[key] for key in _METRIC_COLUMNS},
            })
    return rows


def _overall_ranking(benchmark: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for baseline in OBJECTSTATE_MODEL_IDENTITY_BASELINES:
        metrics = benchmark["baselines"][baseline]["metrics"]
        rows.append({
            "baseline": baseline,
            **{key: float(metrics[key]) for key in _METRIC_COLUMNS},
        })
    return sorted(rows, key=lambda row: row["identity_retrieval_at_1"], reverse=True)


def _validate_benchmark_digest(digest: Mapping[str, Any], *, expected_scenarios: int) -> None:
    if digest.get("schema") != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA:
        raise ValueError("model identity benchmark report digest schema is unsupported")
    if int(digest.get("num_scenarios", 0)) != expected_scenarios:
        raise ValueError("model identity benchmark report digest scenario count mismatch")
    if tuple(digest.get("required_perturbations", ())) != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        raise ValueError("model identity benchmark report digest perturbation coverage is incomplete")
    baselines = digest.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != set(OBJECTSTATE_MODEL_IDENTITY_BASELINES):
        raise ValueError("model identity benchmark report digest requires all baselines")
    breakdown = digest.get("perturbation_breakdown")
    if not isinstance(breakdown, Mapping):
        raise ValueError("model identity benchmark report digest requires perturbation breakdown")
    for kind in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        if kind not in breakdown:
            raise ValueError(f"model identity benchmark report digest missing {kind}")
    gate = digest.get("long_training_gate")
    if not isinstance(gate, Mapping) or gate.get("status") not in {"blocked", "candidate_ready"}:
        raise ValueError("model identity benchmark report digest requires long_training_gate")


def _benchmark_digest(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": benchmark["schema"],
        "kind": benchmark["kind"],
        "status": benchmark["status"],
        "sample_id": benchmark["sample_id"],
        "identity_gate_schema": benchmark["identity_gate_schema"],
        "num_scenarios": benchmark["num_scenarios"],
        "num_pairs": benchmark["num_pairs"],
        "required_perturbations": benchmark["required_perturbations"],
        "perturbation_coverage": benchmark["perturbation_coverage"],
        "thresholds": benchmark["thresholds"],
        "baselines": benchmark["baselines"],
        "perturbation_breakdown": benchmark["perturbation_breakdown"],
        "long_training_gate": benchmark["long_training_gate"],
        "artifact_refs": benchmark["artifact_refs"],
        "claim_policy": benchmark["claim_policy"],
        "non_goals": benchmark["non_goals"],
    }


def _report_status(benchmark: Mapping[str, Any]) -> str:
    return (
        "objectstate_model_identity_benchmark_report_candidate_ready"
        if benchmark["long_training_gate"]["status"] == "candidate_ready"
        else "objectstate_model_identity_benchmark_report_blocked"
    )


def _write_breakdown_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ("scenario_id", "difficulty", "perturbation_kind", "baseline", *_METRIC_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown_report(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    benchmark = summary["benchmark_digest"]
    lines = [
        "# ObjectState Identity Benchmark Report",
        "",
        f"- Schema: `{summary['schema']}`",
        f"- Sample: `{summary['sample_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Scenarios: `{summary['scenario_count']}`",
        f"- Identity pairs: `{summary['identity_pair_count']}`",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Baseline | Retrieval@1 | Margin | Drift | Occlusion | Slot Swap |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(summary["overall_ranking"], start=1):
        lines.append(
            f"| {index} | `{row['baseline']}` | {_fmt(row['identity_retrieval_at_1'])} | "
            f"{_fmt(row['identity_margin'])} | {_fmt(row['objectstate_drift'])} | "
            f"{_fmt(row['occlusion_recovery'])} | {_fmt(row['slot_swap_rate'])} |"
        )
    lines.extend([
        "",
        "## Perturbation Breakdown",
        "",
        "| Perturbation | Scenarios | Solver Retrieval@1 | XYZ Retrieval@1 | Solver - XYZ | Solver Occlusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for kind in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        item = benchmark["perturbation_breakdown"][kind]
        solver = item["baselines"]["assignment_solver_v2"]["metrics"]
        xyz = item["baselines"]["xyz_centroid"]["metrics"]
        lines.append(
            f"| `{kind}` | {item['num_scenarios']} | {_fmt(solver['identity_retrieval_at_1'])} | "
            f"{_fmt(xyz['identity_retrieval_at_1'])} | "
            f"{_fmt(solver['identity_retrieval_at_1'] - xyz['identity_retrieval_at_1'])} | "
            f"{_fmt(solver['occlusion_recovery'])} |"
        )
    lines.extend([
        "",
        "## Difficulty Ladder",
        "",
        "| Difficulty | Scenarios | Solver Retrieval@1 | Solver Margin | Solver Drift |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for difficulty in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES:
        matching = [row for row in rows if row["baseline"] == "assignment_solver_v2" and row["difficulty"] == difficulty]
        lines.append(
            f"| `{difficulty}` | {len(matching)} | {_fmt(_mean(matching, 'identity_retrieval_at_1'))} | "
            f"{_fmt(_mean(matching, 'identity_margin'))} | {_fmt(_mean(matching, 'objectstate_drift'))} |"
        )
    gate = benchmark["long_training_gate"]
    reasons = ", ".join(f"`{item}`" for item in gate["reasons"]) if gate["reasons"] else "none"
    lines.extend([
        "",
        "## Long Training Gate",
        "",
        f"- Decision: `{gate['status']}`",
        f"- Reasons: {reasons}",
        "",
        "This only gates a longer identity robustness smoke. It does not unlock world-model training.",
        "",
        "## Interpretation Boundary",
        "",
        "This report is deterministic controlled synthetic evidence. It does not use real controlled capture,",
        "does not run identity ablation, does not add temporal loss, and does not claim a real-data identity pass.",
        "",
    ])
    return "\n".join(lines)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"
