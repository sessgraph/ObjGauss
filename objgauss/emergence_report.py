from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class EmergenceCurveInput:
    label: str
    curve: dict[str, Any]


@dataclass(frozen=True)
class _MetricSpec:
    key: str
    label: str
    unit: str


_METRICS = (
    _MetricSpec("projection_loss", "Projection loss", "loss"),
    _MetricSpec("assignment_confidence", "Assignment confidence", "score"),
    _MetricSpec("mean_normalized_entropy", "Mean normalized entropy", "score"),
    _MetricSpec("ari_to_initial", "ARI to initial", "score"),
    _MetricSpec("spatial_compactness_score", "Spatial compactness", "score"),
    _MetricSpec("render_occlusion_effect_score", "Render occlusion effect", "score"),
    _MetricSpec("object_emergence_score", "Object emergence score", "score"),
)

_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be123c",
)


def load_emergence_curve(path: str | Path, *, label: str | None = None) -> EmergenceCurveInput:
    path = Path(path)
    curve = json.loads(path.read_text(encoding="utf-8"))
    if curve.get("kind") != "object_emergence_curve":
        raise ValueError(f"{path} is not an object_emergence_curve JSON")
    return EmergenceCurveInput(label=label or path.stem, curve=curve)


def write_emergence_curve_report(
    path: str | Path,
    curves: Sequence[EmergenceCurveInput],
    *,
    title: str = "Object Emergence Benchmark",
) -> dict[str, Any]:
    if not curves:
        raise ValueError("at least one emergence curve is required")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    charts = [
        _metric_chart(metric, curves)
        for metric in _METRICS
        if _metric_has_values(metric, curves)
    ]
    document = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_escape(title)}</title>",
            "<style>",
            _stylesheet(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{_escape(title)}</h1>",
            _summary_table(curves),
            *charts,
            _method_note(curves),
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    output.write_text(document, encoding="utf-8")
    return {
        "output": str(output),
        "curves": len(curves),
        "charts": len(charts),
        "metrics": [metric.key for metric in _METRICS if _metric_has_values(metric, curves)],
    }


def _summary_table(curves: Sequence[EmergenceCurveInput]) -> str:
    rows = []
    for item in curves:
        points = _points(item.curve)
        first = points[0]
        final = points[-1]
        render = final.get("render_occlusion_delta") or {}
        score = final.get("object_emergence_score") or {}
        rows.append(
            "<tr>"
            f"<td>{_escape(item.label)}</td>"
            f"<td>{len(points)}</td>"
            f"<td>{int(item.curve.get('gaussians', 0) or 0)}</td>"
            f"<td>{int(item.curve.get('slots', 0) or 0)}</td>"
            f"<td>{_escape(str(item.curve.get('occlusion_delta_kind', '-')))}</td>"
            f"<td>{_fmt(first.get('projection_loss'))} -> {_fmt(final.get('projection_loss'))}</td>"
            f"<td>{_fmt(final.get('assignment_confidence'))}</td>"
            f"<td>{_fmt(final.get('ari_to_initial'))}</td>"
            f"<td>{_fmt(render.get('occlusion_effect_score'))}</td>"
            f"<td>{_fmt(score.get('score'))}</td>"
            "</tr>"
        )
    return (
        "<section>"
        "<h2>Summary</h2>"
        "<table>"
        "<thead><tr>"
        "<th>Scene</th><th>Points</th><th>Gaussians</th><th>Slots</th>"
        "<th>Occlusion kind</th><th>Projection loss</th><th>Final confidence</th>"
        "<th>Final ARI</th><th>Render effect</th><th>Final OES</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</section>"
    )


def _metric_chart(metric: _MetricSpec, curves: Sequence[EmergenceCurveInput]) -> str:
    width = 860
    height = 260
    left = 54
    right = 18
    top = 18
    bottom = 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [
        value
        for item in curves
        for _, value in _series(metric, item.curve)
        if value is not None
    ]
    steps = [
        step
        for item in curves
        for step, value in _series(metric, item.curve)
        if value is not None
    ]
    y_min = min(values)
    y_max = max(values)
    if abs(y_max - y_min) < 1e-9:
        pad = max(abs(y_max) * 0.05, 0.05)
        y_min -= pad
        y_max += pad
    x_min = min(steps)
    x_max = max(steps)
    if x_max == x_min:
        x_max = x_min + 1

    def map_x(step: float) -> float:
        return left + (step - x_min) / (x_max - x_min) * plot_width

    def map_y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" />',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" />',
        (
            f'<text class="tick" x="{left - 8}" y="{top + 4}" text-anchor="end">'
            f"{_escape(_fmt(y_max))}</text>"
        ),
        (
            f'<text class="tick" x="{left - 8}" y="{top + plot_height + 4}" text-anchor="end">'
            f"{_escape(_fmt(y_min))}</text>"
        ),
        (
            f'<text class="tick" x="{left}" y="{height - 8}" text-anchor="middle">'
            f"{_escape(_fmt(x_min))}</text>"
        ),
        (
            f'<text class="tick" x="{left + plot_width}" y="{height - 8}" text-anchor="middle">'
            f"{_escape(_fmt(x_max))}</text>"
        ),
    ]
    legend = []
    for index, item in enumerate(curves):
        color = _COLORS[index % len(_COLORS)]
        series = _series(metric, item.curve)
        points = [
            f"{map_x(step):.2f},{map_y(value):.2f}"
            for step, value in series
            if value is not None
        ]
        if len(points) >= 2:
            lines.append(
                f'<polyline class="series" points="{" ".join(points)}" stroke="{color}" />'
            )
        elif len(points) == 1:
            x, y = points[0].split(",")
            lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}" />')
        legend.append(
            f'<span><i style="background:{color}"></i>{_escape(item.label)}</span>'
        )

    return (
        "<section>"
        f"<h2>{_escape(metric.label)}</h2>"
        f'<p class="metric-unit">{_escape(metric.unit)}</p>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(metric.label)}">'
        f"{''.join(lines)}"
        "</svg>"
        f'<div class="legend">{"".join(legend)}</div>'
        "</section>"
    )


def _method_note(curves: Sequence[EmergenceCurveInput]) -> str:
    kinds = sorted({str(item.curve.get("occlusion_delta_kind", "-")) for item in curves})
    return (
        "<section>"
        "<h2>Method</h2>"
        "<p>"
        "Curves are sampled from ObjGauss mask-vote training checkpoints. "
        "Render occlusion is measured by removing each hard slot and comparing "
        "point-splat/depth probe renders in image space. "
        f"Occlusion kinds: {_escape(', '.join(kinds))}."
        "</p>"
        "</section>"
    )


def _metric_has_values(metric: _MetricSpec, curves: Sequence[EmergenceCurveInput]) -> bool:
    return any(value is not None for item in curves for _, value in _series(metric, item.curve))


def _series(metric: _MetricSpec, curve: dict[str, Any]) -> list[tuple[float, float | None]]:
    series = []
    for point in _points(curve):
        step = float(point.get("step", 0.0) or 0.0)
        series.append((step, _point_metric(metric.key, point)))
    return series


def _points(curve: dict[str, Any]) -> list[dict[str, Any]]:
    points = curve.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("emergence curve must contain non-empty points")
    if not all(isinstance(point, dict) for point in points):
        raise ValueError("emergence curve points must be objects")
    return points


def _point_metric(key: str, point: dict[str, Any]) -> float | None:
    if key == "render_occlusion_effect_score":
        render = point.get("render_occlusion_delta")
        value = render.get("occlusion_effect_score") if isinstance(render, dict) else None
        return _optional_float(value)
    if key == "object_emergence_score":
        score = point.get("object_emergence_score")
        value = score.get("score") if isinstance(score, dict) else None
        return _optional_float(value)
    return _optional_float(point.get(key))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if abs(numeric) < 1e-9:
        numeric = 0.0
    if abs(numeric) >= 100:
        return f"{numeric:.2f}"
    if abs(numeric) >= 10:
        return f"{numeric:.3f}"
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _stylesheet() -> str:
    return """
body {
  margin: 0;
  background: #f8fafc;
  color: #111827;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 980px;
  margin: 0 auto;
  padding: 32px 20px 48px;
}
h1 {
  margin: 0 0 24px;
  font-size: 30px;
  line-height: 1.15;
}
h2 {
  margin: 0 0 6px;
  font-size: 17px;
}
section {
  margin: 18px 0;
  padding: 16px;
  background: #ffffff;
  border: 1px solid #d1d5db;
  border-radius: 8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th,
td {
  padding: 8px 9px;
  border-bottom: 1px solid #e5e7eb;
  text-align: left;
  vertical-align: top;
}
th {
  color: #374151;
  font-weight: 650;
}
svg {
  display: block;
  width: 100%;
  height: auto;
  background: #ffffff;
}
.axis {
  stroke: #9ca3af;
  stroke-width: 1;
}
.tick {
  fill: #6b7280;
  font-size: 11px;
}
.series {
  fill: none;
  stroke-width: 2.5;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 8px;
  font-size: 13px;
  color: #374151;
}
.legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.legend i {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  display: inline-block;
}
.metric-unit {
  margin: 0 0 8px;
  color: #6b7280;
  font-size: 12px;
}
p {
  margin: 0;
  color: #374151;
  line-height: 1.55;
}
""".strip()
