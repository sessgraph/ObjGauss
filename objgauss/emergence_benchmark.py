from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from objgauss.emergence import object_emergence_curve, write_emergence_curve_csv
from objgauss.emergence_report import EmergenceCurveInput, write_emergence_curve_report
from objgauss.mask_voting import vote_masks_to_gaussians
from objgauss.object_field import cloud_positions_for_metrics, load_object_field
from objgauss.ply import read_ply
from objgauss.render_probe import load_render_probe_frames


def run_emergence_benchmark(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    report_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "object_emergence_benchmark":
        raise ValueError("benchmark manifest kind must be object_emergence_benchmark")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path) if report_path else output_dir / "report.html"
    summary_path = Path(summary_path) if summary_path else output_dir / "summary.json"

    root = _benchmark_root(manifest_path, manifest)
    defaults = _dict(manifest.get("defaults"))
    global_thresholds = _dict(manifest.get("thresholds"))
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("benchmark manifest must contain a non-empty scenes list")

    curve_inputs: list[EmergenceCurveInput] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene in scenes:
        scene_summary, curve = _run_scene(
            scene,
            root=root,
            output_dir=output_dir,
            defaults=defaults,
            global_thresholds=global_thresholds,
        )
        scene_summaries.append(scene_summary)
        curve_inputs.append(EmergenceCurveInput(label=scene_summary["label"], curve=curve))

    report = write_emergence_curve_report(
        report_path,
        curve_inputs,
        title=str(manifest.get("title") or "Object Emergence Benchmark"),
    )
    passed = all(scene["passed"] for scene in scene_summaries)
    summary = {
        "kind": "object_emergence_benchmark_summary",
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "report": report,
        "scenes": scene_summaries,
        "passed": passed,
    }
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)

    if strict and not passed:
        failed = ", ".join(scene["id"] for scene in scene_summaries if not scene["passed"])
        raise ValueError(f"emergence benchmark failed: {failed}")
    return summary


def _run_scene(
    scene: object,
    *,
    root: Path,
    output_dir: Path,
    defaults: dict[str, Any],
    global_thresholds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(scene, dict):
        raise ValueError("benchmark scene entries must be objects")
    scene_id = _required_id(scene)
    label = str(scene.get("label") or scene_id)
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    input_path = _resolve(root, _required_str(scene, "input"))
    field_path = _resolve(root, _required_str(scene, "field"))
    masks_path = _resolve(root, _required_str(scene, "masks"))
    cloud = read_ply(input_path)
    field = load_object_field(field_path)
    if cloud.count != field.gaussian_count:
        raise ValueError(f"scene {scene_id}: field has {field.gaussian_count} gaussians for cloud with {cloud.count}")

    max_frames = _optional_int(scene.get("max_frames", defaults.get("max_frames")))
    render_size = _int_setting(scene, defaults, "render_size", 128)
    iterations = _int_setting(scene, defaults, "iterations", 100)
    learning_rate = _float_setting(scene, defaults, "learning_rate", 0.5)
    eval_every = _int_setting(scene, defaults, "eval_every", 10)

    votes = vote_masks_to_gaussians(
        cloud,
        masks_path,
        slots=field.slots,
        max_frames=max_frames,
    )
    render_frames = load_render_probe_frames(
        masks_path,
        max_frames=max_frames,
        max_size=render_size,
    )
    curve = object_emergence_curve(
        field,
        votes,
        positions_xyz=cloud_positions_for_metrics(cloud),
        cloud=cloud,
        render_frames=render_frames,
        iterations=iterations,
        learning_rate=learning_rate,
        eval_every=eval_every,
    )

    curve_path = scene_dir / "curve.json"
    csv_path = scene_dir / "curve.csv"
    _write_json(curve_path, curve)
    write_emergence_curve_csv(csv_path, curve)

    first = curve["points"][0]
    final = curve["points"][-1]
    thresholds = {**global_thresholds, **_dict(scene.get("thresholds"))}
    checks = _evaluate_thresholds(first, final, points=len(curve["points"]), thresholds=thresholds)
    scene_summary = {
        "id": scene_id,
        "label": label,
        "passed": all(check["passed"] for check in checks),
        "curve": str(curve_path),
        "csv": str(csv_path),
        "gaussians": int(curve["gaussians"]),
        "slots": int(curve["slots"]),
        "points": len(curve["points"]),
        "initial_projection_loss": float(first["projection_loss"]),
        "final_projection_loss": float(final["projection_loss"]),
        "final_assignment_confidence": float(final["assignment_confidence"]),
        "final_ari_to_initial": float(final["ari_to_initial"]),
        "final_spatial_compactness_score": _optional_float(final.get("spatial_compactness_score")),
        "final_render_occlusion_effect_score": _render_effect(final),
        "final_object_emergence_score": _score(final),
        "checks": checks,
    }
    return scene_summary, curve


def _evaluate_thresholds(
    first: dict[str, Any],
    final: dict[str, Any],
    *,
    points: int,
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    initial_loss = float(first["projection_loss"])
    final_loss = float(final["projection_loss"])
    if bool(thresholds.get("require_projection_loss_decrease", True)):
        checks.append(
            _check(
                "projection_loss_decreased",
                final_loss < initial_loss,
                actual=final_loss,
                expected=f"< {initial_loss:.6f}",
            )
        )
    if "min_points" in thresholds:
        minimum = int(thresholds["min_points"])
        checks.append(_check("min_points", points >= minimum, actual=points, expected=f">= {minimum}"))
    if "max_final_projection_loss" in thresholds:
        maximum = float(thresholds["max_final_projection_loss"])
        checks.append(
            _check(
                "max_final_projection_loss",
                final_loss <= maximum,
                actual=final_loss,
                expected=f"<= {maximum:.6f}",
            )
        )
    if "min_projection_loss_drop" in thresholds:
        minimum = float(thresholds["min_projection_loss_drop"])
        drop = initial_loss - final_loss
        checks.append(
            _check(
                "min_projection_loss_drop",
                drop >= minimum,
                actual=drop,
                expected=f">= {minimum:.6f}",
            )
        )
    if "min_render_occlusion_effect_score" in thresholds:
        minimum = float(thresholds["min_render_occlusion_effect_score"])
        effect = _render_effect(final)
        checks.append(
            _check(
                "min_render_occlusion_effect_score",
                effect is not None and effect >= minimum,
                actual=effect,
                expected=f">= {minimum:.6f}",
            )
        )
    if "min_final_object_emergence_score" in thresholds:
        minimum = float(thresholds["min_final_object_emergence_score"])
        score = _score(final)
        checks.append(
            _check(
                "min_final_object_emergence_score",
                score is not None and score >= minimum,
                actual=score,
                expected=f">= {minimum:.6f}",
            )
        )
    return checks


def _check(name: str, passed: bool, *, actual: object, expected: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _benchmark_root(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    root_value = str(manifest.get("root") or ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = manifest_path.parent / root
    return root.resolve()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _required_id(scene: dict[str, Any]) -> str:
    value = _required_str(scene, "id")
    if "/" in value or "\\" in value or value in {"", ".", ".."}:
        raise ValueError(f"invalid benchmark scene id: {value!r}")
    return value


def _required_str(scene: dict[str, Any], key: str) -> str:
    value = scene.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark scene requires non-empty {key!r}")
    return value


def _int_setting(scene: dict[str, Any], defaults: dict[str, Any], key: str, fallback: int) -> int:
    return int(scene.get(key, defaults.get(key, fallback)))


def _float_setting(scene: dict[str, Any], defaults: dict[str, Any], key: str, fallback: float) -> float:
    return float(scene.get(key, defaults.get(key, fallback)))


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _render_effect(point: dict[str, Any]) -> float | None:
    render = point.get("render_occlusion_delta")
    if not isinstance(render, dict):
        return None
    return _optional_float(render.get("occlusion_effect_score"))


def _score(point: dict[str, Any]) -> float | None:
    score = point.get("object_emergence_score")
    if not isinstance(score, dict):
        return None
    return _optional_float(score.get("score"))


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("benchmark manifest settings must be objects")
    return dict(value)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
