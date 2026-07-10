from __future__ import annotations

import gc
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from objgauss.datasets.masks import read_image_rgba

DEFAULT_PROMPT_TEMPLATES = ("{label}",)
DEFAULT_BACKGROUND_LABELS = (
    "background",
    "white background",
    "table surface",
    "floor",
    "wall",
    "shadow",
    "cast shadow",
    "object shadow",
)
CLIP_LABEL_PRESETS: dict[str, tuple[str, ...]] = {
    "nerf-lego-v1": (
        "yellow lego vehicle body",
        "red lego cab roof",
        "black rubber tire",
        "gray wheel hub",
        "lego wheel tread",
        "yellow front scoop",
        "white background",
        "table surface",
        "cast shadow",
    ),
    "lego-parts-v1": (
        "yellow lego vehicle body",
        "red lego cab roof",
        "black rubber tire",
        "gray wheel hub",
        "lego wheel tread",
        "white background",
        "table surface",
        "cast shadow",
    ),
}

__all__ = (
    "CLIP_LABEL_PRESETS",
    "DEFAULT_BACKGROUND_LABELS",
    "DEFAULT_PROMPT_TEMPLATES",
    "ClipMaskScorer",
    "ClipScoringResult",
    "HashClipMaskScorer",
    "TransformersClipMaskScorer",
    "read_clip_labels",
    "score_mask_manifest_with_clip",
)


@dataclass(frozen=True)
class ClipScoringResult:
    source_manifest: Path
    output_manifest: Path
    backend: str
    model: str
    labels: tuple[str, ...]
    frames: int
    masks: int
    scored_masks: int
    cached_masks: int
    prompt_templates: tuple[str, ...]
    background_fill: str
    naming_quality: dict[str, Any]

    @property
    def named_masks(self) -> int:
        return self.scored_masks + self.cached_masks

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "objgauss-clip-mask-score-v1",
            "source_manifest": str(self.source_manifest),
            "output_manifest": str(self.output_manifest),
            "backend": self.backend,
            "model": self.model,
            "labels": list(self.labels),
            "frames": int(self.frames),
            "masks": int(self.masks),
            "scored_masks": int(self.scored_masks),
            "cached_masks": int(self.cached_masks),
            "named_masks": int(self.named_masks),
            "prompt_templates": list(self.prompt_templates),
            "background_fill": self.background_fill,
            "naming_quality": self.naming_quality,
        }


class ClipMaskScorer(Protocol):
    backend: str
    model: str

    def score(self, images: list[np.ndarray], labels: tuple[str, ...]) -> np.ndarray:
        ...


class HashClipMaskScorer:
    backend = "hash-diagnostic"
    model = "sha256-image-label-v1"

    def score(self, images: list[np.ndarray], labels: tuple[str, ...]) -> np.ndarray:
        rows: list[list[float]] = []
        for image in images:
            image_digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()
            raw_scores = []
            for label in labels:
                digest = hashlib.sha256(image_digest + label.encode("utf-8")).digest()
                raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
                raw_scores.append(float(raw) / float(2**64 - 1))
            rows.append(_normalize_scores(raw_scores))
        return np.asarray(rows, dtype=np.float32)


class TransformersClipMaskScorer:
    backend = "transformers"

    def __init__(self, *, model: str, device: str | None = None) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ValueError(
                "CLIP scoring backend 'transformers' requires optional dependencies. "
                "Run with, for example: uv run --with torch --with transformers "
                "objgauss masks score-clip ..."
            ) from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = CLIPProcessor.from_pretrained(model)
        self._model = CLIPModel.from_pretrained(model).to(self._device)
        self._model.eval()
        self.model = model

    def score(self, images: list[np.ndarray], labels: tuple[str, ...]) -> np.ndarray:
        if not images:
            return np.empty((0, len(labels)), dtype=np.float32)
        inputs = self._processor(
            text=list(labels),
            images=[np.asarray(image, dtype=np.uint8) for image in images],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            probabilities = outputs.logits_per_image.softmax(dim=1)
        return probabilities.detach().cpu().numpy().astype(np.float32, copy=False)

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        self._model = None
        self._processor = None
        if self._device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        gc.collect()


def score_mask_manifest_with_clip(
    manifest_path: str | Path,
    *,
    output: str | Path,
    labels: list[str] | tuple[str, ...],
    dataset: str | Path | None = None,
    backend: str = "transformers",
    model: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
    max_frames: int | None = None,
    max_masks: int | None = None,
    crop_padding: float = 0.05,
    background_fill: str = "white",
    prompt_templates: list[str] | tuple[str, ...] | None = None,
    background_labels: list[str] | tuple[str, ...] | None = None,
    min_unique_top_labels: int = 2,
    max_top_label_fraction: float = 0.75,
    max_background_label_fraction: float = 0.5,
    overwrite_scores: bool = False,
    scorer: ClipMaskScorer | None = None,
) -> ClipScoringResult:
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if max_masks is not None and max_masks < 1:
        raise ValueError("max_masks must be >= 1")
    if crop_padding < 0.0:
        raise ValueError("crop_padding must be >= 0")
    if background_fill not in {"white", "black", "gray", "mean", "image"}:
        raise ValueError("background_fill must be one of: white, black, gray, mean, image")
    if min_unique_top_labels < 1:
        raise ValueError("min_unique_top_labels must be >= 1")
    if not 0.0 < max_top_label_fraction <= 1.0:
        raise ValueError("max_top_label_fraction must be in (0, 1]")
    if not 0.0 <= max_background_label_fraction <= 1.0:
        raise ValueError("max_background_label_fraction must be in [0, 1]")
    clean_labels = _clean_labels(labels)
    clean_prompt_templates = _clean_prompt_templates(prompt_templates)
    scoring_labels = _prompted_labels(clean_labels, clean_prompt_templates)
    clean_background_labels = _clean_background_labels(background_labels)
    manifest_path = Path(manifest_path)
    output = Path(output)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("mask manifest must contain a non-empty frames list")

    source_root = _source_root(payload, dataset=dataset, manifest_path=manifest_path)
    scorer = scorer or _create_scorer(backend=backend, model=model, device=device)
    scored_payload = json.loads(json.dumps(payload))
    scored_frames = scored_payload["frames"]

    pending: list[tuple[dict[str, Any], np.ndarray]] = []
    visited_frames = 0
    visited_masks = 0
    cached_masks = 0
    visited_output_masks: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(frames[:max_frames]):
        if max_masks is not None and visited_masks >= max_masks:
            break
        if not isinstance(frame, dict):
            raise ValueError("each mask frame must be an object")
        output_frame = scored_frames[frame_index]
        width = _required_int(
            frame.get("width", payload.get("width") or payload.get("image_width")),
            "frame width",
        )
        height = _required_int(
            frame.get("height", payload.get("height") or payload.get("image_height")),
            "frame height",
        )
        image_path = _resolve_image_path(
            frame,
            source_root=source_root,
            manifest_path=manifest_path,
        )
        rgba = read_image_rgba(image_path)
        if rgba.shape[:2] != (height, width):
            raise ValueError(
                f"frame {frame_index} image shape {rgba.shape[:2]} "
                f"does not match {(height, width)}"
            )
        masks = frame.get("masks")
        output_masks = output_frame.get("masks") if isinstance(output_frame, dict) else None
        if not isinstance(masks, list) or not isinstance(output_masks, list):
            raise ValueError("each mask frame must contain a masks list")
        visited_frames += 1
        for mask_index, mask in enumerate(masks):
            if max_masks is not None and visited_masks >= max_masks:
                break
            if not isinstance(mask, dict) or not isinstance(output_masks[mask_index], dict):
                raise ValueError("mask entries must be objects")
            output_mask = output_masks[mask_index]
            visited_masks += 1
            visited_output_masks.append(output_mask)
            if _has_clip_scores(
                mask,
                labels=clean_labels,
                backend=scorer.backend,
                model=scorer.model,
                prompt_templates=clean_prompt_templates,
            ) and not overwrite_scores:
                cached_masks += 1
                continue
            mask_array = _mask_array(mask, root=manifest_path.parent, width=width, height=height)
            crop = _masked_crop_rgb(
                rgba[:, :, :3],
                mask_array,
                padding=crop_padding,
                background_fill=background_fill,
            )
            pending.append((output_mask, crop))

    if pending:
        probabilities = scorer.score([crop for _, crop in pending], scoring_labels)
        if probabilities.shape != (len(pending), len(scoring_labels)):
            raise ValueError(
                f"CLIP scorer returned shape {probabilities.shape}, "
                f"expected {(len(pending), len(scoring_labels))}"
            )
        for (mask, _crop), scores in zip(pending, probabilities, strict=True):
            label_scores = _aggregate_prompt_scores(
                np.asarray([scores], dtype=np.float32),
                labels=clean_labels,
                prompt_templates=clean_prompt_templates,
            )[0]
            _attach_scores(
                mask,
                scores=label_scores,
                labels=clean_labels,
                backend=scorer.backend,
                model=scorer.model,
                prompt_templates=clean_prompt_templates,
            )

    naming_quality = _clip_naming_quality(
        visited_output_masks,
        background_labels=clean_background_labels,
        min_unique_top_labels=min_unique_top_labels,
        max_top_label_fraction=max_top_label_fraction,
        max_background_label_fraction=max_background_label_fraction,
    )

    _rewrite_manifest_paths_for_output(
        scored_payload,
        source_root=manifest_path.parent,
        output_root=output.parent,
    )
    summary = {
        "kind": "objgauss-clip-mask-score-v1",
        "backend": scorer.backend,
        "model": scorer.model,
        "labels": list(clean_labels),
        "frames": int(visited_frames),
        "masks": int(visited_masks),
        "scored_masks": int(len(pending)),
        "cached_masks": int(cached_masks),
        "named_masks": int(len(pending) + cached_masks),
        "prompt_templates": list(clean_prompt_templates),
        "scoring_labels": list(scoring_labels),
        "background_fill": background_fill,
        "background_labels": list(clean_background_labels),
        "max_frames": max_frames,
        "max_masks": max_masks,
        "crop_padding": float(crop_padding),
        "overwrite_scores": bool(overwrite_scores),
        "naming_quality": naming_quality,
    }
    scored_payload["clip_scoring"] = summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(scored_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = ClipScoringResult(
        source_manifest=manifest_path,
        output_manifest=output,
        backend=scorer.backend,
        model=scorer.model,
        labels=clean_labels,
        frames=visited_frames,
        masks=visited_masks,
        scored_masks=len(pending),
        cached_masks=cached_masks,
        prompt_templates=clean_prompt_templates,
        background_fill=background_fill,
        naming_quality=naming_quality,
    )
    _close_scorer(scorer)
    return result


def read_clip_labels(
    values: list[str] | tuple[str, ...],
    labels_file: str | Path | None = None,
    presets: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    labels = list(values)
    for preset in presets or ():
        if preset not in CLIP_LABEL_PRESETS:
            known = ", ".join(sorted(CLIP_LABEL_PRESETS))
            raise ValueError(f"unknown CLIP label preset {preset!r}; known presets: {known}")
        labels.extend(CLIP_LABEL_PRESETS[preset])
    if labels_file is not None:
        path = Path(labels_file)
        text = path.read_text(encoding="utf-8")
        stripped = text.strip()
        if stripped.startswith("["):
            loaded = json.loads(stripped)
            if not isinstance(loaded, list):
                raise ValueError("labels JSON must be a list")
            labels.extend(str(item) for item in loaded)
        else:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    labels.append(stripped)
    return labels


def _create_scorer(*, backend: str, model: str, device: str | None) -> ClipMaskScorer:
    if backend == "transformers":
        return TransformersClipMaskScorer(model=model, device=device)
    if backend in {"hash", "hash-diagnostic"}:
        return HashClipMaskScorer()
    raise ValueError("backend must be transformers or hash")


def _close_scorer(scorer: ClipMaskScorer) -> None:
    close = getattr(scorer, "close", None)
    if callable(close):
        close()


def _clean_labels(labels: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    if not cleaned:
        raise ValueError("at least one CLIP label is required")
    return tuple(cleaned)


def _clean_prompt_templates(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    templates = list(values or DEFAULT_PROMPT_TEMPLATES)
    cleaned: list[str] = []
    seen: set[str] = set()
    for template in templates:
        text = str(template).strip()
        if not text:
            continue
        if "{label}" not in text:
            raise ValueError("each prompt template must contain {label}")
        if text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    if not cleaned:
        raise ValueError("at least one prompt template is required")
    return tuple(cleaned)


def _prompted_labels(labels: tuple[str, ...], prompt_templates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        template.format(label=label)
        for label in labels
        for template in prompt_templates
    )


def _clean_background_labels(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    labels = values or DEFAULT_BACKGROUND_LABELS
    cleaned: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        cleaned.append(text)
        seen.add(key)
    return tuple(cleaned)


def _source_root(payload: dict[str, Any], *, dataset: str | Path | None, manifest_path: Path) -> Path:
    if dataset is not None:
        return Path(dataset)
    source = payload.get("source")
    if isinstance(source, str) and source:
        return Path(source)
    return manifest_path.parent


def _resolve_image_path(frame: dict[str, Any], *, source_root: Path, manifest_path: Path) -> Path:
    value = frame.get("image_path")
    if not isinstance(value, str) or not value:
        raise ValueError("mask frame is missing image_path")
    path = Path(value)
    if path.is_absolute():
        return path
    if source_root == manifest_path.parent and not (source_root / path).exists():
        return manifest_path.parent / path
    return source_root / path


def _mask_array(mask: dict[str, Any], *, root: Path, width: int, height: int) -> np.ndarray:
    if "rect" in mask:
        x0, y0, x1, y1 = _required_rect(mask["rect"])
        array = np.zeros((height, width), dtype=bool)
        array[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
        return array
    mask_path = mask.get("mask_path")
    if not isinstance(mask_path, str) or not mask_path:
        raise ValueError("mask entry must include mask_path or rect")
    array = np.load(root / mask_path)
    if array.shape != (height, width):
        raise ValueError(
            f"mask {mask_path} shape {array.shape} does not match {height}x{width}"
        )
    return array.astype(bool, copy=False)


def _required_rect(value: object) -> tuple[int, int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("mask rect must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (int(round(float(part))) for part in value)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("mask rect must have x1 > x0 and y1 > y0")
    return x0, y0, x1, y1


def _masked_crop_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    padding: float,
    background_fill: str,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("cannot score an empty mask")
    height, width = mask.shape
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    pad = int(round(max(x1 - x0, y1 - y0) * padding))
    x0 = max(0, x0 - pad)
    x1 = min(width, x1 + pad)
    y0 = max(0, y0 - pad)
    y1 = min(height, y1 + pad)
    crop = np.asarray(rgb[y0:y1, x0:x1], dtype=np.uint8).copy()
    crop_mask = mask[y0:y1, x0:x1]
    if background_fill != "image":
        crop[~crop_mask] = _background_fill_rgb(crop, crop_mask, background_fill)
    return crop


def _background_fill_rgb(crop: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "white":
        return np.array([255, 255, 255], dtype=np.uint8)
    if mode == "black":
        return np.array([0, 0, 0], dtype=np.uint8)
    if mode == "gray":
        return np.array([127, 127, 127], dtype=np.uint8)
    if mode == "mean":
        selected = crop[mask]
        if selected.size == 0:
            return np.array([127, 127, 127], dtype=np.uint8)
        return np.rint(selected.mean(axis=0)).astype(np.uint8)
    raise ValueError("background_fill must be one of: white, black, gray, mean, image")


def _aggregate_prompt_scores(
    probabilities: np.ndarray,
    *,
    labels: tuple[str, ...],
    prompt_templates: tuple[str, ...],
) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float32)
    expected = len(labels) * len(prompt_templates)
    if array.ndim != 2 or array.shape[1] != expected:
        raise ValueError(f"prompt score shape {array.shape} does not match {expected} prompts")
    if len(prompt_templates) == 1:
        return array
    grouped = array.reshape((array.shape[0], len(labels), len(prompt_templates))).sum(axis=2)
    return np.asarray([_normalize_scores(row) for row in grouped], dtype=np.float32)


def _attach_scores(
    mask: dict[str, Any],
    *,
    scores: np.ndarray,
    labels: tuple[str, ...],
    backend: str,
    model: str,
    prompt_templates: tuple[str, ...],
) -> None:
    normalized = _normalize_scores(float(score) for score in scores)
    pairs = {label: float(score) for label, score in zip(labels, normalized, strict=True)}
    ranked = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
    mask["clip_scores"] = pairs
    mask["clip"] = {
        "kind": "objgauss-mask-clip-score-v1",
        "backend": backend,
        "model": model,
        "labels": list(labels),
        "prompt_templates": list(prompt_templates),
        "scores": pairs,
        "top_label": ranked[0][0],
        "top_score": float(ranked[0][1]),
    }


def _has_clip_scores(
    mask: dict[str, Any],
    *,
    labels: tuple[str, ...],
    backend: str,
    model: str,
    prompt_templates: tuple[str, ...],
) -> bool:
    value = mask.get("clip_scores")
    if not isinstance(value, dict) or not value:
        clip = mask.get("clip")
        if isinstance(clip, dict) and isinstance(clip.get("scores"), dict):
            value = clip["scores"]
        else:
            return False
    if set(value) != set(labels):
        return False
    clip = mask.get("clip")
    if not isinstance(clip, dict):
        return True
    if clip.get("backend") not in {None, backend}:
        return False
    if clip.get("model") not in {None, model}:
        return False
    cached_templates = clip.get("prompt_templates")
    if cached_templates is not None and tuple(cached_templates) != prompt_templates:
        return False
    if prompt_templates != DEFAULT_PROMPT_TEMPLATES and cached_templates is None:
        return False
    return True


def _clip_naming_quality(
    masks: list[dict[str, Any]],
    *,
    background_labels: tuple[str, ...],
    min_unique_top_labels: int,
    max_top_label_fraction: float,
    max_background_label_fraction: float,
) -> dict[str, Any]:
    top_counts: Counter[str] = Counter()
    top_scores: list[float] = []
    top_margins: list[float] = []
    for mask in masks:
        scores = _clip_scores_for_mask(mask)
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_label, top_score = ranked[0]
        top_counts[top_label] += 1
        top_scores.append(float(top_score))
        second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        top_margins.append(float(top_score) - second_score)

    named_masks = int(sum(top_counts.values()))
    background_keys = {label.lower() for label in background_labels}
    background_count = int(
        sum(count for label, count in top_counts.items() if label.lower() in background_keys)
    )
    max_label = ""
    max_count = 0
    if top_counts:
        max_label, max_count = sorted(top_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    max_fraction = float(max_count / named_masks) if named_masks else 0.0
    background_fraction = float(background_count / named_masks) if named_masks else 0.0
    unique_top_labels = int(len(top_counts))

    blockers: list[str] = []
    if named_masks == 0:
        blockers.append("no-named-masks")
    if unique_top_labels < min_unique_top_labels:
        blockers.append("not-enough-unique-top-labels")
    if max_fraction > max_top_label_fraction:
        blockers.append(f"top-label-dominant:{max_label}")
    if background_fraction > max_background_label_fraction:
        blockers.append("background-label-dominant")

    return {
        "kind": "objgauss-clip-naming-quality-v1",
        "passed": not blockers,
        "blockers": blockers,
        "named_masks": named_masks,
        "unique_top_labels": unique_top_labels,
        "top_label_counts": dict(sorted(top_counts.items(), key=lambda item: (-item[1], item[0]))),
        "max_top_label": max_label,
        "max_top_label_count": int(max_count),
        "max_top_label_fraction": max_fraction,
        "background_labels": list(background_labels),
        "background_top_label_count": background_count,
        "background_top_label_fraction": background_fraction,
        "mean_top_score": float(np.mean(top_scores)) if top_scores else 0.0,
        "mean_top_margin": float(np.mean(top_margins)) if top_margins else 0.0,
        "thresholds": {
            "min_unique_top_labels": int(min_unique_top_labels),
            "max_top_label_fraction": float(max_top_label_fraction),
            "max_background_label_fraction": float(max_background_label_fraction),
        },
    }


def _clip_scores_for_mask(mask: dict[str, Any]) -> dict[str, float]:
    value = mask.get("clip_scores")
    if isinstance(value, dict):
        return _numeric_scores(value)
    clip = mask.get("clip")
    if isinstance(clip, dict) and isinstance(clip.get("scores"), dict):
        return _numeric_scores(clip["scores"])
    return {}


def _numeric_scores(value: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, raw in value.items():
        try:
            scores[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return scores


def _normalize_scores(values: Any) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return []
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return [float(1.0 / array.size)] * int(array.size)
    return [float(value / total) for value in array]


def _rewrite_manifest_paths_for_output(
    payload: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
) -> None:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        _rewrite_relative_path(
            frame,
            "ignore_mask_path",
            source_root=source_root,
            output_root=output_root,
        )
        masks = frame.get("masks")
        if not isinstance(masks, list):
            continue
        for mask in masks:
            if not isinstance(mask, dict):
                continue
            _rewrite_relative_path(
                mask,
                "mask_path",
                source_root=source_root,
                output_root=output_root,
            )
            _rewrite_relative_path(
                mask,
                "ignore_mask_path",
                source_root=source_root,
                output_root=output_root,
            )


def _rewrite_relative_path(
    entry: dict[str, Any],
    key: str,
    *,
    source_root: Path,
    output_root: Path,
) -> None:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        return
    path = Path(value)
    if path.is_absolute():
        return
    absolute = (source_root / path).resolve()
    entry[key] = os.path.relpath(absolute, output_root.resolve())


def _required_int(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    return int(value)
