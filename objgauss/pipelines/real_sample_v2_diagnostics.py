from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from objgauss.core.gaussian import GaussianCloud
from objgauss.pipelines.real_sample_v2_smoke import (
    RealSampleV2SmokeReport,
    evaluate_real_sample_v2_smoke,
    real_sample_v2_smoke_from_cloud,
    validate_real_sample_v2_smoke_summary,
)
from objgauss.pipelines.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
    TrainableKernelSample,
    trainable_kernel_sample_from_cloud,
)

REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA = "objgauss-real-sample-v2-diagnostics-v1"

__all__ = (
    "REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA",
    "RealSampleV2DiagnosticsReport",
    "real_sample_v2_diagnostics_from_cloud",
    "evaluate_real_sample_v2_diagnostics",
    "validate_real_sample_v2_diagnostics_summary",
)
_STATUS_PASS = "real_sample_v2_diagnostics_pass"
_STATUS_FAIL = "real_sample_v2_diagnostics_fail"
_DEFAULT_TEMPERATURE_CANDIDATES = (1.0, 0.75, 0.5, 0.35, 0.25)


@dataclass(frozen=True)
class RealSampleV2DiagnosticsReport:
    candidates: tuple[RealSampleV2SmokeReport, ...]
    baseline_temperature: float
    sample_source: str
    object_id_field: str
    schema: str = REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA

    @property
    def baseline(self) -> RealSampleV2SmokeReport:
        for candidate in self.candidates:
            if _candidate_temperature(candidate) == float(self.baseline_temperature):
                return candidate
        return self.candidates[0]

    @property
    def best_candidate(self) -> RealSampleV2SmokeReport:
        passing = [
            candidate
            for candidate in self.candidates
            if candidate.as_dict()["gates"]["renderer_joint_passed"]
        ]
        if passing:
            return max(passing, key=_candidate_temperature)
        return max(self.candidates, key=_ranking_tuple)

    @property
    def passed(self) -> bool:
        return self.best_candidate.as_dict()["gates"]["renderer_joint_passed"]

    def as_dict(self) -> dict[str, Any]:
        baseline_summary = self.baseline.as_dict()
        best_summary = self.best_candidate.as_dict()
        candidate_records = tuple(_candidate_record(candidate) for candidate in self.candidates)
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_diagnostics",
            "status": _STATUS_PASS if self.passed else _STATUS_FAIL,
            "sample": baseline_summary["sample"],
            "baseline_temperature": float(self.baseline_temperature),
            "best_temperature": _candidate_temperature(self.best_candidate),
            "candidate_count": len(self.candidates),
            "temperature_sweep": [dict(record) for record in candidate_records],
            "baseline": _candidate_record(self.baseline),
            "best_candidate": _candidate_record(self.best_candidate),
            "failure_breakdown": _failure_breakdown(
                baseline_summary,
                best_summary,
            ),
            "recommendation": _recommendation(
                baseline_summary,
                best_summary,
            ),
            "best_checkpoint": {
                "schema": self.best_candidate.checkpoint["schema"],
                "source": self.best_candidate.checkpoint["source"],
                "solver_step": int(self.best_candidate.training_result.final_state.step),
                "solver_temperature": _candidate_temperature(self.best_candidate),
                "solver_state": self.best_candidate.training_result.final_state.as_dict(
                    include_arrays=True
                ),
            },
            "truth_contract": {
                "target_source": "object_id_one_hot_targets",
                "object_id_labels_are_training_targets": True,
                "semantic_ground_truth_claimed": False,
                "fixture_oracle_claimed": False,
            },
            "non_goals": {
                "uses_gpu": False,
                "unfreezes_gaussian_geometry": False,
                "mutates_dynamic_k": False,
                "implements_evidence_normalization": False,
                "uses_rollout_model": False,
                "uses_replay_buffer": False,
                "uses_diffusion": False,
            },
        }
        return validate_real_sample_v2_diagnostics_summary(payload)


def real_sample_v2_diagnostics_from_cloud(
    cloud: GaussianCloud,
    *,
    sample_source: str = "memory://gaussian-cloud",
    object_id_field: str = "object_id",
    slots: int | None = None,
    frame_count: int = 2,
    max_points: int | None = 24,
    temporal_offset: float = 0.01,
    image_width: int = 12,
    image_height: int = 12,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 4,
    iterations: int = 100,
    learning_rate: float = 0.4,
    cluster_weight: float = 0.0,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
    supervised_weight: float = 1.0,
    temperature_candidates: Sequence[float] = _DEFAULT_TEMPERATURE_CANDIDATES,
    baseline_temperature: float = 1.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
) -> RealSampleV2DiagnosticsReport:
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        slots=slots,
        frame_count=frame_count,
        max_points=max_points,
        object_id_field=object_id_field,
        temporal_offset=temporal_offset,
        bind_image_targets=True,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
    )
    return evaluate_real_sample_v2_diagnostics(
        sample,
        sample_source=sample_source,
        object_id_field=object_id_field,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
        iterations=iterations,
        learning_rate=learning_rate,
        cluster_weight=cluster_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        supervised_weight=supervised_weight,
        temperature_candidates=temperature_candidates,
        baseline_temperature=baseline_temperature,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
    )


def evaluate_real_sample_v2_diagnostics(
    sample: TrainableKernelSample,
    *,
    sample_source: str = "memory://trainable-kernel-sample",
    object_id_field: str = "object_id",
    image_width: int = 12,
    image_height: int = 12,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 4,
    iterations: int = 100,
    learning_rate: float = 0.4,
    cluster_weight: float = 0.0,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
    supervised_weight: float = 1.0,
    temperature_candidates: Sequence[float] = _DEFAULT_TEMPERATURE_CANDIDATES,
    baseline_temperature: float = 1.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
) -> RealSampleV2DiagnosticsReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    candidates = _temperature_candidates(
        temperature_candidates,
        baseline_temperature=baseline_temperature,
    )
    reports = tuple(
        evaluate_real_sample_v2_smoke(
            sample,
            sample_source=sample_source,
            object_id_field=object_id_field,
            image_width=image_width,
            image_height=image_height,
            point_radius=point_radius,
            visibility_policy=visibility_policy,
            seed=seed,
            iterations=iterations,
            learning_rate=learning_rate,
            cluster_weight=cluster_weight,
            entropy_weight=entropy_weight,
            balance_weight=balance_weight,
            supervised_weight=supervised_weight,
            solver_temperature=temperature,
            image_renderer=image_renderer,
            vram_reserve_gb=vram_reserve_gb,
        )
        for temperature in candidates
    )
    return RealSampleV2DiagnosticsReport(
        candidates=reports,
        baseline_temperature=float(baseline_temperature),
        sample_source=str(sample_source),
        object_id_field=str(object_id_field),
    )


def validate_real_sample_v2_diagnostics_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 diagnostics summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA:
        raise ValueError(f"unsupported real sample v2 diagnostics schema: {payload.get('schema')}")
    if payload.get("kind") != "real_sample_v2_diagnostics":
        raise ValueError("real sample v2 diagnostics kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_FAIL}:
        raise ValueError("real sample v2 diagnostics status is unsupported")
    for key in (
        "sample",
        "baseline_temperature",
        "best_temperature",
        "candidate_count",
        "temperature_sweep",
        "baseline",
        "best_candidate",
        "failure_breakdown",
        "recommendation",
        "best_checkpoint",
        "truth_contract",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"real sample v2 diagnostics summary missing {key}")
    candidates = payload["temperature_sweep"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("real sample v2 diagnostics requires non-empty temperature_sweep")
    if int(payload["candidate_count"]) != len(candidates):
        raise ValueError("real sample v2 diagnostics candidate_count must match sweep")
    temperatures = {float(candidate["solver_temperature"]) for candidate in candidates}
    if float(payload["baseline_temperature"]) not in temperatures:
        raise ValueError("real sample v2 diagnostics baseline temperature missing from sweep")
    if float(payload["best_temperature"]) not in temperatures:
        raise ValueError("real sample v2 diagnostics best temperature missing from sweep")
    for candidate in candidates:
        _validate_candidate_record(candidate)
    best = payload["best_candidate"]
    _validate_candidate_record(best)
    best_passes = best["renderer_joint_status"] == "assignment_v2_renderer_joint_validation_pass"
    expected_status = _STATUS_PASS if best_passes else _STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("real sample v2 diagnostics status must match best candidate")
    recommendation = payload["recommendation"]
    if not isinstance(recommendation, dict):
        raise ValueError("real sample v2 diagnostics recommendation must be a dict")
    checkpoint = payload["best_checkpoint"]
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("solver_state"), dict):
        raise ValueError("real sample v2 diagnostics best_checkpoint must include solver_state")
    if float(checkpoint["solver_temperature"]) != float(payload["best_temperature"]):
        raise ValueError("real sample v2 diagnostics checkpoint temperature must match best")
    if recommendation.get("requires_geometry_unfreeze") is not False:
        raise ValueError("real sample v2 diagnostics must not require geometry unfreeze")
    if recommendation.get("requires_diffusion_replay_or_rollout") is not False:
        raise ValueError("real sample v2 diagnostics must not require diffusion/replay/rollout")
    truth_contract = payload["truth_contract"]
    if truth_contract.get("semantic_ground_truth_claimed") is not False:
        raise ValueError("real sample v2 diagnostics must not claim semantic ground truth")
    if truth_contract.get("fixture_oracle_claimed") is not False:
        raise ValueError("real sample v2 diagnostics must not claim fixture oracle truth")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("uses_gpu")
        or non_goals.get("unfreezes_gaussian_geometry")
        or non_goals.get("mutates_dynamic_k")
        or non_goals.get("implements_evidence_normalization")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
    ):
        raise ValueError("real sample v2 diagnostics summary violates non-goals")
    return payload


def _temperature_candidates(
    values: Sequence[float],
    *,
    baseline_temperature: float,
) -> tuple[float, ...]:
    candidates = [float(baseline_temperature)]
    candidates.extend(float(value) for value in values)
    deduped = []
    for value in candidates:
        if value <= 0:
            raise ValueError("temperature candidates must be > 0")
        if not any(abs(value - existing) <= 1e-9 for existing in deduped):
            deduped.append(value)
    return tuple(deduped)


def _candidate_temperature(report: RealSampleV2SmokeReport) -> float:
    return float(report.training_result.final_state.config.temperature)


def _ranking_tuple(report: RealSampleV2SmokeReport) -> tuple[float, float, float, float]:
    record = _candidate_record(report)
    metrics = record["object_state_metrics"]
    purity = metrics["object_purity"]
    return (
        -1.0 if purity is None else float(purity),
        float(metrics["assignment_confidence"]),
        -float(metrics["mean_normalized_entropy"]),
        -float(record["training_loss"]["final_supervised_loss"]),
    )


def _candidate_record(report: RealSampleV2SmokeReport) -> dict[str, Any]:
    summary = validate_real_sample_v2_smoke_summary(report.as_dict())
    object_state = summary["renderer_joint"]["object_state_eval"]
    diagnostics = sorted(
        {
            diagnostic
            for frame in object_state["frames"]
            for diagnostic in frame["diagnostics"]
        }
    )
    return {
        "solver_temperature": _candidate_temperature(report),
        "status": summary["status"],
        "renderer_joint_status": summary["renderer_joint"]["status"],
        "renderer_boundary_status": summary["renderer_boundary"]["status"],
        "training_loss": {
            "initial_total_loss": float(summary["training_loss"]["initial_total_loss"]),
            "final_total_loss": float(summary["training_loss"]["final_total_loss"]),
            "loss_decreased": bool(summary["training_loss"]["loss_decreased"]),
            "initial_supervised_loss": float(
                summary["training_loss"]["initial_supervised_loss"]
            ),
            "final_supervised_loss": float(summary["training_loss"]["final_supervised_loss"]),
            "supervised_loss_decreased": bool(
                summary["training_loss"]["supervised_loss_decreased"]
            ),
        },
        "renderer_loss": {
            "initial_image_render_loss": float(
                summary["renderer_joint"]["initial_loss"]["image_render_loss"]
            ),
            "final_image_render_loss": float(
                summary["renderer_joint"]["final_loss"]["image_render_loss"]
            ),
            "image_render_loss_decreased": bool(
                summary["renderer_joint"]["image_render_loss_decreased"]
            ),
        },
        "object_state_metrics": {
            "status": object_state["status"],
            "mean_normalized_entropy": float(object_state["mean_normalized_entropy"]),
            "assignment_confidence": float(object_state["assignment_confidence"]),
            "object_purity": None
            if object_state["object_purity"] is None
            else float(object_state["object_purity"]),
            "max_dominant_slot_mass_fraction": float(
                object_state["max_dominant_slot_mass_fraction"]
            ),
            "slot_collapse": bool(object_state["slot_collapse"]),
        },
        "diagnostics": diagnostics,
        "checkpoint_roundtrip": summary["renderer_joint"]["checkpoint_roundtrip"],
    }


def _failure_breakdown(
    baseline_summary: dict[str, Any],
    best_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_object = baseline_summary["renderer_joint"]["object_state_eval"]
    best_object = best_summary["renderer_joint"]["object_state_eval"]
    baseline_diagnostics = sorted(
        {
            diagnostic
            for frame in baseline_object["frames"]
            for diagnostic in frame["diagnostics"]
        }
    )
    return {
        "baseline_status": baseline_summary["status"],
        "baseline_renderer_joint_status": baseline_summary["renderer_joint"]["status"],
        "baseline_object_state_status": baseline_object["status"],
        "baseline_diagnostics": baseline_diagnostics,
        "best_status": best_summary["status"],
        "best_renderer_joint_status": best_summary["renderer_joint"]["status"],
        "best_object_state_status": best_object["status"],
        "entropy_delta": float(best_object["mean_normalized_entropy"])
        - float(baseline_object["mean_normalized_entropy"]),
        "confidence_delta": float(best_object["assignment_confidence"])
        - float(baseline_object["assignment_confidence"]),
        "purity_delta": _nullable_delta(
            best_object["object_purity"],
            baseline_object["object_purity"],
        ),
    }


def _recommendation(
    baseline_summary: dict[str, Any],
    best_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_passes = baseline_summary["gates"]["renderer_joint_passed"]
    best_passes = best_summary["gates"]["renderer_joint_passed"]
    best_temperature = float(best_summary["training"]["final_state"]["config"]["temperature"])
    if baseline_passes:
        return {
            "decision": "baseline_already_passes",
            "action": "no_change_required",
            "solver_temperature": float(
                baseline_summary["training"]["final_state"]["config"]["temperature"]
            ),
            "evidence_normalization": "not_required_for_current_smoke",
            "requires_geometry_unfreeze": False,
            "requires_diffusion_replay_or_rollout": False,
        }
    if best_passes:
        return {
            "decision": "temperature_sharpening_sufficient",
            "action": "set_solver_temperature",
            "solver_temperature": best_temperature,
            "selection_policy": "highest_temperature_candidate_that_passes_objectstate_gate",
            "evidence_normalization": "not_required_for_current_smoke",
            "requires_geometry_unfreeze": False,
            "requires_diffusion_replay_or_rollout": False,
        }
    return {
        "decision": "temperature_sharpening_insufficient",
        "action": "evaluate_evidence_normalization_next",
        "solver_temperature": None,
        "evidence_normalization": "recommended_next_diagnostic",
        "requires_geometry_unfreeze": False,
        "requires_diffusion_replay_or_rollout": False,
    }


def _nullable_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _validate_candidate_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("real sample v2 diagnostics candidate must be a dict")
    if float(record["solver_temperature"]) <= 0:
        raise ValueError("real sample v2 diagnostics candidate temperature must be > 0")
    if record["renderer_joint_status"] not in {
        "assignment_v2_renderer_joint_validation_pass",
        "assignment_v2_renderer_joint_validation_fail",
    }:
        raise ValueError("real sample v2 diagnostics candidate renderer status unsupported")
    if not isinstance(record.get("diagnostics"), list):
        raise ValueError("real sample v2 diagnostics candidate diagnostics must be a list")
    metrics = record.get("object_state_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("real sample v2 diagnostics candidate missing object_state_metrics")
    for key in ("mean_normalized_entropy", "assignment_confidence", "max_dominant_slot_mass_fraction"):
        float(metrics[key])
