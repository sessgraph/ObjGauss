from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_solver_v2_eval import (
    ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA,
    assignment_solver_v2_state_from_checkpoint,
)
from objgauss.core.assignment_v2_renderer_validation import (
    AssignmentV2RendererJointValidationReport,
    evaluate_assignment_v2_renderer_joint,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.real_sample_v2_diagnostics import (
    RealSampleV2DiagnosticsReport,
    real_sample_v2_diagnostics_from_cloud,
    validate_real_sample_v2_diagnostics_summary,
)
from objgauss.core.renderer_loss import (
    RendererLossBoundaryReport,
    renderer_loss_boundary_report,
)
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
    TrainableKernelFrame,
    TrainableKernelSample,
    trainable_kernel_sample_from_cloud,
)

REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA = "objgauss-real-sample-v2-model-handoff-v1"
REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA = "objgauss-real-sample-v2-effect-preview-v1"
_STATUS_PASS = "real_sample_v2_model_handoff_pass"
_STATUS_FAIL = "real_sample_v2_model_handoff_fail"
_SLOT_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#f59e0b",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4b5563",
)


@dataclass(frozen=True)
class RealSampleV2ModelHandoffReport:
    diagnostics: RealSampleV2DiagnosticsReport
    restored_renderer_joint: AssignmentV2RendererJointValidationReport
    restored_renderer_boundary: RendererLossBoundaryReport
    checkpoint: dict[str, Any]
    sample_source: str
    schema: str = REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA

    @property
    def passed(self) -> bool:
        restored = self.restored_renderer_joint.as_dict()
        return (
            self.diagnostics.as_dict()["status"] == "real_sample_v2_diagnostics_pass"
            and restored["status"] == "assignment_v2_renderer_joint_validation_pass"
            and bool(restored["checkpoint_roundtrip"]["pass"])
        )

    def as_dict(self) -> dict[str, Any]:
        diagnostics_summary = self.diagnostics.as_dict()
        restored_summary = self.restored_renderer_joint.as_dict()
        boundary_summary = self.restored_renderer_boundary.as_dict()
        checkpoint_state = assignment_solver_v2_state_from_checkpoint(self.checkpoint)
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_model_handoff",
            "status": _STATUS_PASS if self.passed else _STATUS_FAIL,
            "sample": diagnostics_summary["sample"],
            "recommended_solver_temperature": float(
                diagnostics_summary["recommendation"]["solver_temperature"]
            ),
            "diagnostics_schema": diagnostics_summary["schema"],
            "checkpoint_schema": self.checkpoint["schema"],
            "model_checkpoint": {
                "schema": self.checkpoint["schema"],
                "source": self.checkpoint["source"],
                "solver_step": int(checkpoint_state.step),
                "solver_temperature": float(checkpoint_state.config.temperature),
                "slots": int(checkpoint_state.config.slots),
                "feature_dim": int(checkpoint_state.config.feature_dim),
                "state_schema": checkpoint_state.schema,
                "arrays_included": "arrays" in self.checkpoint["solver_state"],
            },
            "restore_validation": {
                "json_roundtrip_restored": True,
                "renderer_joint_status": restored_summary["status"],
                "object_state_status": restored_summary["object_state_eval"]["status"],
                "checkpoint_roundtrip": restored_summary["checkpoint_roundtrip"],
                "renderer_boundary_status": boundary_summary["status"],
            },
            "training_effect": {
                "baseline": diagnostics_summary["baseline"],
                "best_candidate": diagnostics_summary["best_candidate"],
                "failure_breakdown": diagnostics_summary["failure_breakdown"],
                "recommendation": diagnostics_summary["recommendation"],
            },
            "effect_preview": _effect_preview_payload(self.diagnostics),
            "diagnostics": diagnostics_summary,
            "restored_renderer_joint": restored_summary,
            "restored_renderer_boundary": boundary_summary,
            "output_policy": {
                "summary": "write to /tmp or ignored outputs for handoff",
                "checkpoint": "write to /tmp or ignored outputs; do not commit training checkpoints",
                "preview": "write HTML/SVG preview to /tmp or ignored outputs",
            },
            "non_goals": {
                "uses_gpu": False,
                "unfreezes_gaussian_geometry": False,
                "mutates_dynamic_k": False,
                "uses_rollout_model": False,
                "uses_replay_buffer": False,
                "uses_diffusion": False,
                "claims_public_demo_release": False,
            },
        }
        return validate_real_sample_v2_model_handoff_summary(payload)


def real_sample_v2_model_handoff_from_cloud(
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
    temperature_candidates: Sequence[float] = (1.0, 0.75, 0.5, 0.35, 0.25),
    baseline_temperature: float = 1.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
) -> RealSampleV2ModelHandoffReport:
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
    diagnostics = real_sample_v2_diagnostics_from_cloud(
        cloud,
        sample_source=sample_source,
        object_id_field=object_id_field,
        slots=slots,
        frame_count=frame_count,
        max_points=max_points,
        temporal_offset=temporal_offset,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
        iterations=iterations,
        learning_rate=learning_rate,
        temperature_candidates=temperature_candidates,
        baseline_temperature=baseline_temperature,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
    )
    return evaluate_real_sample_v2_model_handoff(
        sample,
        diagnostics=diagnostics,
        sample_source=sample_source,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
    )


def evaluate_real_sample_v2_model_handoff(
    sample: TrainableKernelSample,
    *,
    diagnostics: RealSampleV2DiagnosticsReport,
    sample_source: str = "memory://trainable-kernel-sample",
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
) -> RealSampleV2ModelHandoffReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    diagnostics_summary = validate_real_sample_v2_diagnostics_summary(diagnostics.as_dict())
    if diagnostics_summary["status"] != "real_sample_v2_diagnostics_pass":
        raise ValueError("model handoff requires passing real sample v2 diagnostics")
    checkpoint = _json_checkpoint_roundtrip(diagnostics.best_candidate.checkpoint)
    restored_renderer_joint = evaluate_assignment_v2_renderer_joint(
        sample.frames,
        checkpoint,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
    )
    restored_renderer_boundary = renderer_loss_boundary_report(
        restored_renderer_joint.as_dict()
    )
    return RealSampleV2ModelHandoffReport(
        diagnostics=diagnostics,
        restored_renderer_joint=restored_renderer_joint,
        restored_renderer_boundary=restored_renderer_boundary,
        checkpoint=checkpoint,
        sample_source=str(sample_source),
    )


def validate_real_sample_v2_model_handoff_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 model handoff summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA:
        raise ValueError(f"unsupported real sample v2 handoff schema: {payload.get('schema')}")
    if payload.get("kind") != "real_sample_v2_model_handoff":
        raise ValueError("real sample v2 handoff kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_FAIL}:
        raise ValueError("real sample v2 handoff status is unsupported")
    for key in (
        "sample",
        "model_checkpoint",
        "restore_validation",
        "training_effect",
        "effect_preview",
        "diagnostics",
        "restored_renderer_joint",
        "output_policy",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"real sample v2 handoff summary missing {key}")
    if payload.get("checkpoint_schema") != ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA:
        raise ValueError("real sample v2 handoff must reference assignment v2 checkpoint")
    restore = payload["restore_validation"]
    if restore.get("json_roundtrip_restored") is not True:
        raise ValueError("real sample v2 handoff must restore checkpoint JSON")
    if restore.get("renderer_joint_status") not in {
        "assignment_v2_renderer_joint_validation_pass",
        "assignment_v2_renderer_joint_validation_fail",
    }:
        raise ValueError("real sample v2 handoff restore status unsupported")
    if restore.get("checkpoint_roundtrip", {}).get("pass") is not True:
        raise ValueError("real sample v2 handoff checkpoint roundtrip must pass")
    preview = payload["effect_preview"]
    validate_real_sample_v2_effect_preview(preview)
    expected_status = (
        _STATUS_PASS
        if restore["renderer_joint_status"] == "assignment_v2_renderer_joint_validation_pass"
        else _STATUS_FAIL
    )
    if payload["status"] != expected_status:
        raise ValueError("real sample v2 handoff status must match restore gate")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("uses_gpu")
        or non_goals.get("unfreezes_gaussian_geometry")
        or non_goals.get("mutates_dynamic_k")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("claims_public_demo_release")
    ):
        raise ValueError("real sample v2 handoff summary violates non-goals")
    return payload


def validate_real_sample_v2_effect_preview(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 effect preview must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA:
        raise ValueError(f"unsupported real sample v2 preview schema: {payload.get('schema')}")
    if payload.get("kind") != "assignment_effect_preview":
        raise ValueError("real sample v2 preview kind is unsupported")
    panels = payload.get("panels")
    if not isinstance(panels, list) or len(panels) != 2:
        raise ValueError("real sample v2 preview requires baseline and best panels")
    for panel in panels:
        if not isinstance(panel.get("points"), list) or not panel["points"]:
            raise ValueError("real sample v2 preview panel requires points")
    return payload


def render_real_sample_v2_model_handoff_html(summary: dict[str, Any]) -> str:
    checked = validate_real_sample_v2_model_handoff_summary(summary)
    preview = checked["effect_preview"]
    baseline, best = preview["panels"]
    metrics = checked["training_effect"]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>ObjGauss Real Sample V2 Handoff</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}",
            ".wrap{max-width:1120px;margin:0 auto}",
            ".metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0}",
            ".metric{background:white;border:1px solid #e2e8f0;border-radius:8px;padding:12px}",
            ".metric b{display:block;font-size:13px;color:#475569}.metric span{font-size:20px;font-weight:700;line-height:1.15;overflow-wrap:anywhere}",
            ".panels{display:grid;grid-template-columns:1fr 1fr;gap:16px}",
            ".panel{background:white;border:1px solid #e2e8f0;border-radius:8px;padding:16px}",
            ".panel h2{font-size:18px;margin:0 0 6px}.caption{color:#475569;font-size:13px;margin:0 0 12px}",
            "svg{width:100%;height:auto;border:1px solid #e2e8f0;background:#fff}",
            ".legend{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px;font-size:13px;color:#334155}",
            ".swatch{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-1px}",
            ".note{font-size:13px;color:#475569;margin-top:14px}",
            "</style>",
            "</head>",
            "<body>",
            '<main class="wrap">',
            "<h1>ObjGauss Real Sample V2 Model Handoff</h1>",
            f"<p>{html.escape(preview['display_claim'])}</p>",
            '<section class="metrics">',
            _metric("Status", checked["status"].replace("real_sample_v2_model_handoff_", "")),
            _metric("Best temperature", checked["recommended_solver_temperature"]),
            _metric("Baseline purity", metrics["baseline"]["object_state_metrics"]["object_purity"]),
            _metric("Best purity", metrics["best_candidate"]["object_state_metrics"]["object_purity"]),
            "</section>",
            '<section class="panels">',
            _panel_svg(baseline),
            _panel_svg(best),
            "</section>",
            _legend_html(preview["legend"]),
            '<p class="note">Circle color is predicted object slot. Thin outer ring is the object_id target slot; larger opacity indicates higher assignment confidence.</p>',
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _effect_preview_payload(diagnostics: RealSampleV2DiagnosticsReport) -> dict[str, Any]:
    baseline = diagnostics.baseline
    best = diagnostics.best_candidate
    frames = baseline.sample.frames
    frame = frames[0]
    slot_count = int(baseline.sample.slots)
    panels = [
        _preview_panel(
            label="baseline",
            report=baseline,
            frame=frame,
        ),
        _preview_panel(
            label="trained_temperature_sharpened",
            report=best,
            frame=frame,
        ),
    ]
    return validate_real_sample_v2_effect_preview(
        {
            "schema": REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA,
            "kind": "assignment_effect_preview",
            "frame_index": 0,
            "point_count": int(frame.positions.shape[0]),
            "slot_count": slot_count,
            "display_claim": (
                "Sampled point assignment preview for model debugging; this is not a full 3DGS render."
            ),
            "legend": [
                {"slot": slot, "color": _slot_color(slot)}
                for slot in range(slot_count)
            ],
            "panels": panels,
        }
    )


def _preview_panel(
    *,
    label: str,
    report,
    frame: TrainableKernelFrame,
) -> dict[str, Any]:
    summary = report.as_dict()
    object_state = summary["renderer_joint"]["object_state_eval"]
    assignment = report.renderer_joint.final_assignments[0]
    target = np.asarray(frame.target_assignment, dtype=np.float32)
    positions = np.asarray(frame.positions, dtype=np.float32)
    normalized = _normalize_xy(positions[:, :2])
    points = []
    for index, (xy, row, target_row) in enumerate(zip(normalized, assignment, target, strict=True)):
        slot = int(np.argmax(row))
        target_slot = int(np.argmax(target_row))
        points.append(
            {
                "index": int(index),
                "x": float(round(float(xy[0]), 6)),
                "y": float(round(float(xy[1]), 6)),
                "predicted_slot": slot,
                "target_slot": target_slot,
                "confidence": float(round(float(np.max(row)), 6)),
                "matches_target": bool(slot == target_slot),
                "color": _slot_color(slot),
                "target_color": _slot_color(target_slot),
            }
        )
    return {
        "label": label,
        "solver_temperature": float(report.training_result.final_state.config.temperature),
        "status": summary["status"],
        "renderer_joint_status": summary["renderer_joint"]["status"],
        "object_state_metrics": {
            "status": object_state["status"],
            "mean_normalized_entropy": float(object_state["mean_normalized_entropy"]),
            "assignment_confidence": float(object_state["assignment_confidence"]),
            "object_purity": None
            if object_state["object_purity"] is None
            else float(object_state["object_purity"]),
        },
        "points": points,
    }


def _normalize_xy(xy: np.ndarray) -> np.ndarray:
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    normalized = (xy - mins) / span
    normalized[:, 1] = 1.0 - normalized[:, 1]
    return normalized


def _json_checkpoint_roundtrip(checkpoint: dict[str, Any]) -> dict[str, Any]:
    restored = json.loads(json.dumps(checkpoint))
    assignment_solver_v2_state_from_checkpoint(restored)
    return restored


def _slot_color(slot: int) -> str:
    return _SLOT_COLORS[int(slot) % len(_SLOT_COLORS)]


def _metric(label: str, value: Any) -> str:
    if isinstance(value, float):
        display = f"{value:.6f}"
    else:
        display = str(value)
    return (
        '<div class="metric">'
        f"<b>{html.escape(label)}</b>"
        f"<span>{html.escape(display)}</span>"
        "</div>"
    )


def _panel_svg(panel: dict[str, Any]) -> str:
    title = (
        f"{panel['label']} temp={panel['solver_temperature']} "
        f"{panel['object_state_metrics']['status']}"
    )
    circles = []
    for point in panel["points"]:
        x = 32 + float(point["x"]) * 336
        y = 32 + float(point["y"]) * 256
        opacity = 0.35 + 0.65 * float(point["confidence"])
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7.5" '
            f'fill="{html.escape(point["color"])}" fill-opacity="{opacity:.3f}" '
            f'stroke="{html.escape(point["target_color"])}" stroke-width="2" />'
        )
    return "\n".join(
        [
            '<article class="panel">',
            f"<h2>{html.escape(str(title))}</h2>",
            (
                '<p class="caption">'
                f"confidence={panel['object_state_metrics']['assignment_confidence']:.6f}, "
                f"purity={panel['object_state_metrics']['object_purity']:.6f}"
                "</p>"
            ),
            '<svg viewBox="0 0 400 320" role="img" aria-label="assignment preview">',
            '<rect x="0" y="0" width="400" height="320" fill="#ffffff" />',
            "\n".join(circles),
            "</svg>",
            "</article>",
        ]
    )


def _legend_html(legend: list[dict[str, Any]]) -> str:
    items = []
    for item in legend:
        items.append(
            '<span>'
            f'<i class="swatch" style="background:{html.escape(item["color"])}"></i>'
            f"slot {int(item['slot'])}"
            "</span>"
        )
    return '<div class="legend">' + "".join(items) + "</div>"
