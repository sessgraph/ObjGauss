from __future__ import annotations

from pathlib import Path
from typing import Any

TRAINING_SCALE_PLAN_SCHEMA = "objgauss-training-scale-plan-v1"


def solver_decoder_training_scale_plan(
    *,
    total_iterations: int,
    checkpoint_every: int | None,
    loss_log_every: int | None,
    output_dir: str | Path,
    image_renderer: str,
    vram_reserve_gb: int = 1,
) -> dict[str, Any]:
    if total_iterations < 1:
        raise ValueError("total_iterations must be >= 1")
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if loss_log_every is not None and loss_log_every < 1:
        raise ValueError("loss_log_every must be >= 1")
    if vram_reserve_gb < 0:
        raise ValueError("vram_reserve_gb must be >= 0")
    root = Path(output_dir)
    interval = int(checkpoint_every or total_iterations)
    segments: list[dict[str, Any]] = []
    start = 0
    index = 1
    while start < total_iterations:
        iterations = min(interval, total_iterations - start)
        segment_id = f"segment-{index:04d}"
        segments.append(
            {
                "index": index,
                "segment_id": segment_id,
                "start_iteration": start,
                "end_iteration": start + iterations,
                "iterations": iterations,
                "summary_path": str(root / "segments" / f"{segment_id}-summary.json"),
                "checkpoint_path": str(root / "segments" / f"{segment_id}-checkpoint.json"),
            }
        )
        start += iterations
        index += 1
    plan = {
        "schema": TRAINING_SCALE_PLAN_SCHEMA,
        "kind": "solver_decoder_training_scale_plan",
        "total_iterations": int(total_iterations),
        "checkpoint_every": interval,
        "loss_log_every": None if loss_log_every is None else int(loss_log_every),
        "segment_count": len(segments),
        "segments": segments,
        "outputs": {
            "root": str(root),
            "plan": str(root / "training-scale-plan.json"),
            "final_summary": str(root / "final-summary.json"),
            "final_checkpoint": str(root / "final-checkpoint.json"),
            "renderer_loss_boundary": str(root / "renderer-loss-boundary.json"),
        },
        "gpu_policy": {
            "image_renderer": image_renderer,
            "preflight_required": image_renderer == "gsplat",
            "vram_reserve_gb": int(vram_reserve_gb),
        },
        "artifact_policy": {
            "repository_write": "do_not_commit_training_runs",
            "intended_locations": ["/tmp", "ignored outputs/"],
            "large_artifacts": "keep_out_of_git",
        },
    }
    return validate_solver_decoder_training_scale_plan(plan)


def validate_solver_decoder_training_scale_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("training scale plan payload must be a dict")
    if payload.get("schema") != TRAINING_SCALE_PLAN_SCHEMA:
        raise ValueError(f"unsupported training scale plan schema: {payload.get('schema')}")
    if payload.get("kind") != "solver_decoder_training_scale_plan":
        raise ValueError("training scale plan kind must be solver_decoder_training_scale_plan")
    total_iterations = int(payload.get("total_iterations"))
    checkpoint_every = int(payload.get("checkpoint_every"))
    if total_iterations < 1:
        raise ValueError("total_iterations must be >= 1")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    loss_log_every = payload.get("loss_log_every")
    if loss_log_every is not None and int(loss_log_every) < 1:
        raise ValueError("loss_log_every must be >= 1")
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("training scale plan must contain segments")
    if int(payload.get("segment_count")) != len(segments):
        raise ValueError("segment_count must match segments length")
    expected_start = 0
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError("training scale plan segment must be an object")
        if int(segment.get("index")) != index:
            raise ValueError("training scale plan segment index mismatch")
        start = int(segment.get("start_iteration"))
        end = int(segment.get("end_iteration"))
        iterations = int(segment.get("iterations"))
        if start != expected_start:
            raise ValueError("training scale plan segments must be contiguous")
        if iterations < 1 or end - start != iterations:
            raise ValueError("training scale plan segment iterations are invalid")
        if not segment.get("summary_path") or not segment.get("checkpoint_path"):
            raise ValueError("training scale plan segment missing output paths")
        expected_start = end
    if expected_start != total_iterations:
        raise ValueError("training scale plan segments must cover total_iterations")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("training scale plan missing outputs")
    for key in ("root", "plan", "final_summary", "final_checkpoint", "renderer_loss_boundary"):
        if not outputs.get(key):
            raise ValueError(f"training scale plan missing outputs.{key}")
    gpu_policy = payload.get("gpu_policy")
    if not isinstance(gpu_policy, dict):
        raise ValueError("training scale plan missing gpu_policy")
    if int(gpu_policy.get("vram_reserve_gb")) < 0:
        raise ValueError("gpu_policy.vram_reserve_gb must be >= 0")
    if gpu_policy.get("image_renderer") not in {"point", "gsplat"}:
        raise ValueError("gpu_policy.image_renderer must be point or gsplat")
    return payload
