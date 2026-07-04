from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

TENSORBOARD_SCALAR_EXPORT_SCHEMA = "objgauss-tensorboard-scalar-export-v1"


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


def write_solver_decoder_tensorboard_events(
    summary: dict[str, Any],
    logdir: str | Path,
    *,
    writer_factory: Callable[[str | Path], ScalarWriter] | None = None,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise TypeError("solver-decoder summary must be a dict")
    training_scale = summary.get("training_scale")
    if not isinstance(training_scale, dict):
        raise ValueError("solver-decoder summary missing training_scale")
    segments = training_scale.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("training_scale.segments must contain segment records")
    logdir_path = Path(logdir)
    logdir_path.mkdir(parents=True, exist_ok=True)
    writer = writer_factory(logdir_path) if writer_factory is not None else _default_summary_writer(logdir_path)
    scalar_count = 0
    try:
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("training_scale segment record must be an object")
            start = int(segment["start_iteration"])
            end = int(segment["end_iteration"])
            scalar_count += _write_loss_pair(
                writer,
                "loss/total",
                start=start,
                end=end,
                initial=segment["initial_total_loss"],
                final=segment["final_total_loss"],
            )
            scalar_count += _write_loss_pair(
                writer,
                "loss/image_render",
                start=start,
                end=end,
                initial=segment["initial_image_render_loss"],
                final=segment["final_image_render_loss"],
            )
            if "initial_object_loss" in segment and "final_object_loss" in segment:
                scalar_count += _write_loss_pair(
                    writer,
                    "loss/object",
                    start=start,
                    end=end,
                    initial=segment["initial_object_loss"],
                    final=segment["final_object_loss"],
                )
            if "final_entropy_loss" in segment:
                writer.add_scalar("loss/entropy", float(segment["final_entropy_loss"]), end)
                scalar_count += 1
            if "final_balance_loss" in segment:
                writer.add_scalar("loss/balance", float(segment["final_balance_loss"]), end)
                scalar_count += 1
            if "final_decoder_opacity_scale_mean" in segment:
                writer.add_scalar(
                    "decoder/opacity_scale_min",
                    float(segment["final_decoder_opacity_scale_min"]),
                    end,
                )
                writer.add_scalar(
                    "decoder/opacity_scale_mean",
                    float(segment["final_decoder_opacity_scale_mean"]),
                    end,
                )
                writer.add_scalar(
                    "decoder/opacity_scale_max",
                    float(segment["final_decoder_opacity_scale_max"]),
                    end,
                )
                scalar_count += 3
            if "final_decoder_scale_multiplier_mean" in segment:
                writer.add_scalar(
                    "decoder/scale_multiplier_min",
                    float(segment["final_decoder_scale_multiplier_min"]),
                    end,
                )
                writer.add_scalar(
                    "decoder/scale_multiplier_mean",
                    float(segment["final_decoder_scale_multiplier_mean"]),
                    end,
                )
                writer.add_scalar(
                    "decoder/scale_multiplier_max",
                    float(segment["final_decoder_scale_multiplier_max"]),
                    end,
                )
                scalar_count += 3
        run_loss = summary.get("run_loss") if isinstance(summary.get("run_loss"), dict) else {}
        if "final_total_loss" in run_loss:
            writer.add_scalar(
                "run/final_total_loss",
                float(run_loss["final_total_loss"]),
                int(training_scale["total_iterations"]),
            )
            scalar_count += 1
        writer.flush()
    finally:
        writer.close()
    return {
        "schema": TENSORBOARD_SCALAR_EXPORT_SCHEMA,
        "kind": "solver_decoder_tensorboard_scalars",
        "logdir": str(logdir_path),
        "segment_count": len(segments),
        "scalar_count": scalar_count,
        "tags": [
            "loss/total",
            "loss/image_render",
            "loss/object",
            "loss/entropy",
            "loss/balance",
            "decoder/opacity_scale_min",
            "decoder/opacity_scale_mean",
            "decoder/opacity_scale_max",
            "decoder/scale_multiplier_min",
            "decoder/scale_multiplier_mean",
            "decoder/scale_multiplier_max",
            "run/final_total_loss",
        ],
    }


def _write_loss_pair(
    writer: ScalarWriter,
    tag: str,
    *,
    start: int,
    end: int,
    initial: Any,
    final: Any,
) -> int:
    writer.add_scalar(tag, float(initial), start)
    writer.add_scalar(tag, float(final), end)
    return 2


def _default_summary_writer(logdir: Path) -> ScalarWriter:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as error:  # pragma: no cover - exercised by CLI integration
        raise RuntimeError(
            "TensorBoard writer is unavailable; rerun with torch and tensorboard installed, "
            "for example: uv run --with torch --with tensorboard ..."
        ) from error
    return SummaryWriter(log_dir=str(logdir))
