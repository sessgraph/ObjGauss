from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from objgauss.masks import read_image_rgba


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
    overwrite_scores: bool = False,
    scorer: ClipMaskScorer | None = None,
) -> ClipScoringResult:
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if max_masks is not None and max_masks < 1:
        raise ValueError("max_masks must be >= 1")
    if crop_padding < 0.0:
        raise ValueError("crop_padding must be >= 0")
    clean_labels = _clean_labels(labels)
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
            if _has_clip_scores(mask) and not overwrite_scores:
                cached_masks += 1
                continue
            mask_array = _mask_array(mask, root=manifest_path.parent, width=width, height=height)
            crop = _masked_crop_rgb(rgba[:, :, :3], mask_array, padding=crop_padding)
            pending.append((output_mask, crop))

    if pending:
        probabilities = scorer.score([crop for _, crop in pending], clean_labels)
        if probabilities.shape != (len(pending), len(clean_labels)):
            raise ValueError(
                f"CLIP scorer returned shape {probabilities.shape}, "
                f"expected {(len(pending), len(clean_labels))}"
            )
        for (mask, _crop), scores in zip(pending, probabilities, strict=True):
            _attach_scores(mask, scores=scores, labels=clean_labels, backend=scorer.backend, model=scorer.model)

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
        "max_frames": max_frames,
        "max_masks": max_masks,
        "crop_padding": float(crop_padding),
        "overwrite_scores": bool(overwrite_scores),
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
    )
    _close_scorer(scorer)
    return result


def read_clip_labels(
    values: list[str] | tuple[str, ...],
    labels_file: str | Path | None = None,
) -> list[str]:
    labels = list(values)
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
            labels.extend(
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.startswith("#")
            )
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


def _masked_crop_rgb(rgb: np.ndarray, mask: np.ndarray, *, padding: float) -> np.ndarray:
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
    crop[~crop_mask] = np.array([255, 255, 255], dtype=np.uint8)
    return crop


def _attach_scores(
    mask: dict[str, Any],
    *,
    scores: np.ndarray,
    labels: tuple[str, ...],
    backend: str,
    model: str,
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
        "scores": pairs,
        "top_label": ranked[0][0],
        "top_score": float(ranked[0][1]),
    }


def _has_clip_scores(mask: dict[str, Any]) -> bool:
    value = mask.get("clip_scores")
    if isinstance(value, dict) and value:
        return True
    clip = mask.get("clip")
    return isinstance(clip, dict) and isinstance(clip.get("scores"), dict) and bool(clip["scores"])


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
