from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.ply import read_ply

DEVELOPMENT_STAGE_NOTICE = (
    "Development-stage research comparison. Assets, metrics, file layout, "
    "thresholds, and promotion decisions may change before a stable release."
)


def compare_baseline_candidates(
    candidate_specs: list[str] | tuple[str, ...],
    *,
    min_supervised_fraction: float = 0.2,
    max_vote_conflict_fraction: float = 0.05,
    min_slot_balance_score: float = 0.01,
    min_object_active_slots: int = 2,
    object_id_field: str = "object_id",
) -> dict[str, Any]:
    if not candidate_specs:
        raise ValueError("at least one --candidate name=path is required")
    _validate_thresholds(
        min_supervised_fraction=min_supervised_fraction,
        max_vote_conflict_fraction=max_vote_conflict_fraction,
        min_slot_balance_score=min_slot_balance_score,
        min_object_active_slots=min_object_active_slots,
    )
    candidates = [
        _build_candidate(
            name,
            paths,
            min_supervised_fraction=min_supervised_fraction,
            max_vote_conflict_fraction=max_vote_conflict_fraction,
            min_slot_balance_score=min_slot_balance_score,
            min_object_active_slots=min_object_active_slots,
            object_id_field=object_id_field,
        )
        for name, paths in _parse_candidate_specs(candidate_specs).items()
    ]
    policy = _promotion_policy(
        candidates,
        thresholds={
            "min_supervised_fraction": float(min_supervised_fraction),
            "max_vote_conflict_fraction": float(max_vote_conflict_fraction),
            "min_slot_balance_score": float(min_slot_balance_score),
            "min_object_active_slots": int(min_object_active_slots),
            "object_id_field": object_id_field,
        },
    )
    return {
        "kind": "objgauss-clip-baseline-comparison-v1",
        "development_stage_notice": DEVELOPMENT_STAGE_NOTICE,
        "candidate_count": len(candidates),
        "thresholds": policy["thresholds"],
        "promotion_policy": policy,
        "candidates": candidates,
    }


def render_comparison_markdown(summary: dict[str, Any]) -> str:
    policy = summary["promotion_policy"]
    lines = [
        "# ObjGauss CLIP Baseline Comparison",
        "",
        f"> {summary.get('development_stage_notice', DEVELOPMENT_STAGE_NOTICE)}",
        "",
        "## Promotion Policy",
        "",
        f"- Status: `{policy['status']}`",
        f"- Recommended candidate: `{policy.get('recommended_candidate') or '-'}`",
        f"- Blockers: `{', '.join(policy.get('blockers') or []) or '-'}`",
        "",
        "Promote a CLIP semantic route only when slot naming passes, downstream vote/training "
        "evidence is present, and all threshold checks pass. Baseline-only candidates remain "
        "reference rows unless they also include semantic naming evidence.",
        "",
        "## Comparison Summary",
        "",
        (
            "| Candidate | Evidence | Naming | Vote supervised | Vote conflict | "
            "Slot balance | Training loss | Object slots | Promotion |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in summary.get("candidates", []):
        vote = _effective_vote_metrics(candidate)
        training = candidate.get("training") or {}
        object_stats = candidate.get("object_id_stats") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(candidate["name"]),
                    _md(", ".join(candidate.get("evidence_kinds", [])) or "-"),
                    _md(_naming_summary(candidate)),
                    _md(_fmt_float(vote.get("supervised_fraction"))),
                    _md(_fmt_float(vote.get("vote_conflict_fraction"))),
                    _md(_fmt_float(vote.get("slot_balance_score"))),
                    _md(_loss_summary(training)),
                    _md(str(object_stats.get("active_slots", "-"))),
                    _md(candidate.get("promotion", {}).get("status", "-")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Candidate Blockers", ""])
    for candidate in summary.get("candidates", []):
        blockers = candidate.get("promotion", {}).get("blockers") or []
        lines.append(f"- `{candidate['name']}`: {', '.join(blockers) if blockers else '-'}")
    return "\n".join(lines) + "\n"


def write_comparison_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_comparison_markdown(summary), encoding="utf-8")


def _parse_candidate_specs(candidate_specs: list[str] | tuple[str, ...]) -> dict[str, list[Path]]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for spec in candidate_specs:
        if "=" not in spec:
            raise ValueError(f"candidate spec must be name=path: {spec}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name:
            raise ValueError(f"candidate name is empty in spec: {spec}")
        if not raw_path:
            raise ValueError(f"candidate path is empty in spec: {spec}")
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"candidate path does not exist: {path}")
        candidates[name].append(path)
    return dict(candidates)


def _build_candidate(
    name: str,
    paths: list[Path],
    *,
    min_supervised_fraction: float,
    max_vote_conflict_fraction: float,
    min_slot_balance_score: float,
    min_object_active_slots: int,
    object_id_field: str,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "name": name,
        "source_paths": [str(path) for path in paths],
        "evidence_kinds": [],
    }
    for path in paths:
        _merge_path(candidate, path, object_id_field=object_id_field)
    candidate["evidence_kinds"] = sorted(set(candidate["evidence_kinds"]))
    candidate["promotion"] = _candidate_promotion(
        candidate,
        min_supervised_fraction=min_supervised_fraction,
        max_vote_conflict_fraction=max_vote_conflict_fraction,
        min_slot_balance_score=min_slot_balance_score,
        min_object_active_slots=min_object_active_slots,
    )
    return candidate


def _merge_path(candidate: dict[str, Any], path: Path, *, object_id_field: str) -> None:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON evidence must be an object: {path}")
        _merge_json_payload(candidate, payload, path=path)
        return
    if path.suffix.lower() == ".ply":
        candidate["object_id_stats"] = _object_id_stats(path, object_id_field=object_id_field)
        candidate["evidence_kinds"].append("objgauss-object-id-ply-stats-v1")
        return
    raise ValueError(f"unsupported candidate evidence type: {path}")


def _merge_json_payload(candidate: dict[str, Any], payload: dict[str, Any], *, path: Path) -> None:
    kind = _infer_json_kind(payload)
    candidate["evidence_kinds"].append(kind)
    candidate.setdefault("json_sources", []).append({"path": str(path), "kind": kind})

    clip_summary = payload.get("clip_scoring")
    if isinstance(clip_summary, dict):
        _merge_mask_naming(candidate, clip_summary.get("naming_quality"))
    _merge_mask_naming(candidate, payload.get("naming_quality"))

    slot_alignment = payload.get("slot_alignment")
    if isinstance(slot_alignment, dict):
        _merge_slot_naming(candidate, slot_alignment.get("slot_naming_quality"))
        record_filters = slot_alignment.get("record_filters")
        if isinstance(record_filters, dict):
            candidate["record_filters"] = dict(record_filters)
        slot_rebalance = slot_alignment.get("slot_rebalance")
        if isinstance(slot_rebalance, dict):
            candidate["slot_rebalance"] = _json_safe_dict(slot_rebalance)

    slots = payload.get("slots")
    if isinstance(slots, list):
        candidate["slot_manifest"] = _slot_manifest_summary(payload)

    training = payload.get("training")
    if isinstance(training, dict):
        _merge_training(candidate, training)
    if _looks_like_training_summary(payload):
        _merge_training(candidate, payload)

    object_delta = payload.get("object_field_delta")
    if isinstance(object_delta, dict):
        candidate["object_field_delta"] = _json_safe_dict(object_delta)
    acceptance = payload.get("acceptance")
    if isinstance(acceptance, dict):
        candidate["acceptance"] = _json_safe_dict(acceptance)

    if _looks_like_depth_diagnostic(payload):
        candidate["depth_diagnostic"] = _depth_diagnostic_summary(payload)


def _merge_mask_naming(candidate: dict[str, Any], quality: Any) -> None:
    if isinstance(quality, dict) and quality.get("kind") == "objgauss-clip-naming-quality-v1":
        candidate["mask_naming_quality"] = _json_safe_dict(quality)


def _merge_slot_naming(candidate: dict[str, Any], quality: Any) -> None:
    if isinstance(quality, dict) and quality.get("kind") == "objgauss-slot-naming-quality-v1":
        candidate["slot_naming_quality"] = _json_safe_dict(quality)


def _merge_training(candidate: dict[str, Any], training: dict[str, Any]) -> None:
    summary = {
        "initial_loss": _optional_float(training.get("initial_loss")),
        "final_loss": _optional_float(training.get("final_loss")),
        "iterations": _optional_int(training.get("iterations")),
        "supervised_gaussians": _optional_int(training.get("supervised_gaussians")),
        "frames": _optional_int(training.get("frames")),
        "projected": _optional_int(training.get("projected")),
        "matched": _optional_int(training.get("matched")),
    }
    initial = summary["initial_loss"]
    final = summary["final_loss"]
    summary["loss_reduced"] = bool(
        initial is not None and final is not None and final < initial
    )
    if initial is not None and final is not None:
        summary["loss_delta"] = float(initial - final)
        summary["loss_ratio"] = None if initial == 0 else float(final / initial)
    candidate["training"] = summary

    metrics = training.get("metrics")
    if isinstance(metrics, dict):
        candidate["object_field_metrics"] = _json_safe_dict(metrics)
    vote_quality = training.get("vote_quality")
    if isinstance(vote_quality, dict):
        candidate["vote_quality"] = _vote_quality_summary(vote_quality)


def _vote_quality_summary(vote_quality: dict[str, Any]) -> dict[str, Any]:
    conflict = vote_quality.get("vote_conflict")
    if not isinstance(conflict, dict):
        conflict = {}
    confidence = vote_quality.get("target_confidence")
    if not isinstance(confidence, dict):
        confidence = {}
    slot_balance = _slot_balance_from_per_slot(vote_quality.get("per_slot"))
    return {
        "gaussian_count": _optional_int(vote_quality.get("gaussian_count")),
        "slots": _optional_int(vote_quality.get("slots")),
        "supervised_gaussians": _optional_int(vote_quality.get("supervised_gaussians")),
        "supervised_fraction": _optional_float(vote_quality.get("supervised_fraction")),
        "matched_projected_fraction": _optional_float(
            vote_quality.get("matched_projected_fraction")
        ),
        "vote_conflict_gaussians": _optional_int(conflict.get("gaussians")),
        "vote_conflict_fraction": _optional_float(conflict.get("fraction")),
        "normalized_target_entropy": _optional_float(conflict.get("normalized_target_entropy")),
        "target_confidence_mean": _optional_float(confidence.get("mean")),
        "slot_balance": slot_balance,
        "slot_balance_score": slot_balance["score"],
        "per_slot": _json_safe(vote_quality.get("per_slot")),
    }


def _depth_diagnostic_summary(payload: dict[str, Any]) -> dict[str, Any]:
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    depth = payload.get("depth_aware") if isinstance(payload.get("depth_aware"), dict) else {}
    deltas = payload.get("deltas") if isinstance(payload.get("deltas"), dict) else {}
    return {
        "kind": payload.get("kind", "objgauss-depth-visibility-vote-diagnostic-v1"),
        "recommendation": payload.get("recommendation"),
        "baseline": _diagnostic_vote_summary(baseline),
        "depth_aware": _diagnostic_vote_summary(depth),
        "deltas": _json_safe_dict(deltas),
    }


def _diagnostic_vote_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "supervised_fraction": _optional_float(value.get("supervised_fraction")),
        "vote_conflict_fraction": _optional_float(value.get("vote_conflict_fraction")),
        "normalized_target_entropy": _optional_float(value.get("normalized_target_entropy")),
        "slot_balance_score": _optional_float(value.get("slot_balance_score")),
        "slot_balance": _json_safe(value.get("slot_balance")),
    }


def _slot_manifest_summary(payload: dict[str, Any]) -> dict[str, Any]:
    slots = payload.get("slots")
    labels = []
    if isinstance(slots, list):
        for slot in slots:
            if isinstance(slot, dict):
                labels.append(
                    str(slot.get("label") or slot.get("name") or slot.get("slot") or "")
                )
    return {
        "slot_count": _optional_int(payload.get("slot_count")) or len(labels),
        "labels": labels,
        "frames": len(payload.get("frames", [])) if isinstance(payload.get("frames"), list) else None,
    }


def _object_id_stats(path: Path, *, object_id_field: str) -> dict[str, Any]:
    cloud = read_ply(path)
    stats: dict[str, Any] = {
        "path": str(path),
        "gaussian_count": int(cloud.count),
        "object_id_field": object_id_field,
        "has_object_id": object_id_field in cloud.fields,
    }
    if object_id_field not in cloud.fields:
        return stats
    labels, counts = np.unique(cloud.vertices[object_id_field], return_counts=True)
    object_counts = [
        {"object_id": int(label), "count": int(count)}
        for label, count in zip(labels, counts, strict=True)
    ]
    balance = _slot_balance_from_counts([item["count"] for item in object_counts])
    stats.update(
        {
            "active_slots": int(len(object_counts)),
            "object_counts": object_counts,
            "slot_balance": balance,
            "slot_balance_score": balance["score"],
        }
    )
    return stats


def _candidate_promotion(
    candidate: dict[str, Any],
    *,
    min_supervised_fraction: float,
    max_vote_conflict_fraction: float,
    min_slot_balance_score: float,
    min_object_active_slots: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    checks: dict[str, Any] = {}
    has_semantic = bool(candidate.get("slot_naming_quality") or candidate.get("mask_naming_quality"))
    if not has_semantic:
        return {
            "ready": False,
            "status": "reference-only",
            "blockers": ["reference-only:no-semantic-naming-evidence"],
            "checks": checks,
        }

    slot_quality = candidate.get("slot_naming_quality")
    if not isinstance(slot_quality, dict):
        blockers.append("missing-slot-naming-evidence")
        checks["slot_naming_passed"] = False
    else:
        checks["slot_naming_passed"] = bool(slot_quality.get("passed"))
        if not slot_quality.get("passed"):
            blockers.extend(f"slot-naming:{item}" for item in slot_quality.get("blockers", []))

    mask_quality = candidate.get("mask_naming_quality")
    if isinstance(mask_quality, dict):
        checks["mask_naming_passed"] = bool(mask_quality.get("passed"))
        if not mask_quality.get("passed"):
            blockers.extend(f"mask-naming:{item}" for item in mask_quality.get("blockers", []))

    vote = _effective_vote_metrics(candidate)
    if not vote:
        blockers.append("missing-vote-quality-evidence")
        checks["vote_quality_present"] = False
    else:
        checks["vote_quality_present"] = True
        _check_float_floor(
            blockers,
            checks,
            "supervised_fraction",
            vote.get("supervised_fraction"),
            min_supervised_fraction,
        )
        _check_float_ceiling(
            blockers,
            checks,
            "vote_conflict_fraction",
            vote.get("vote_conflict_fraction"),
            max_vote_conflict_fraction,
        )
        _check_float_floor(
            blockers,
            checks,
            "slot_balance_score",
            vote.get("slot_balance_score"),
            min_slot_balance_score,
        )

    training = candidate.get("training")
    if not isinstance(training, dict):
        blockers.append("missing-training-summary")
        checks["training_loss_reduced"] = False
    else:
        checks["training_loss_reduced"] = bool(training.get("loss_reduced"))
        if not training.get("loss_reduced"):
            blockers.append("training-loss-not-reduced")

    object_stats = candidate.get("object_id_stats")
    if isinstance(object_stats, dict) and object_stats.get("has_object_id"):
        active_slots = int(object_stats.get("active_slots", 0) or 0)
        checks["object_active_slots"] = active_slots
        if active_slots < min_object_active_slots:
            blockers.append(
                f"object-active-slots-below-threshold:{active_slots}<{min_object_active_slots}"
            )

    return {
        "ready": not blockers,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "checks": checks,
    }


def _promotion_policy(
    candidates: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    ready = [candidate for candidate in candidates if candidate["promotion"]["ready"]]
    blockers: list[str] = []
    if not ready:
        blockers.append("no-semantic-candidate-passed-promotion-checks")
        for candidate in candidates:
            promotion = candidate.get("promotion", {})
            if promotion.get("status") == "blocked":
                blockers.extend(
                    f"{candidate['name']}:{blocker}"
                    for blocker in promotion.get("blockers", [])
                )
    return {
        "kind": "objgauss-semantic-promotion-policy-v1",
        "status": "promote" if ready else "do-not-promote",
        "recommended_candidate": ready[0]["name"] if ready else None,
        "thresholds": thresholds,
        "requirements": [
            "slot_naming_quality.passed must be true for semantic CLIP candidates",
            "mask_naming_quality must pass when present",
            "vote quality evidence must meet supervised/conflict/balance thresholds",
            "training evidence must reduce projection loss",
            "baseline-only candidates are reference rows without semantic promotion",
        ],
        "blockers": blockers,
    }


def _effective_vote_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    vote = candidate.get("vote_quality")
    if isinstance(vote, dict):
        return vote
    diagnostic = candidate.get("depth_diagnostic")
    if isinstance(diagnostic, dict):
        depth = diagnostic.get("depth_aware")
        if isinstance(depth, dict):
            return depth
    return {}


def _slot_balance_from_per_slot(per_slot: Any) -> dict[str, Any]:
    if not isinstance(per_slot, list):
        return {"score": 0.0, "active_slots": 0, "min_winners": 0, "max_winners": 0}
    winners = []
    for slot in per_slot:
        if isinstance(slot, dict):
            winners.append(int(slot.get("winner_gaussians", 0) or 0))
    return _slot_balance_from_counts(winners)


def _slot_balance_from_counts(counts: list[int]) -> dict[str, Any]:
    active = [int(count) for count in counts if int(count) > 0]
    if not active:
        return {
            "score": 0.0,
            "active_slots": 0,
            "min_winners": 0,
            "max_winners": 0,
            "winner_gaussians": [int(count) for count in counts],
        }
    min_winners = min(active)
    max_winners = max(active)
    return {
        "score": 0.0 if max_winners <= 0 else float(min_winners / max_winners),
        "active_slots": int(len(active)),
        "min_winners": int(min_winners),
        "max_winners": int(max_winners),
        "winner_gaussians": [int(count) for count in counts],
    }


def _check_float_floor(
    blockers: list[str],
    checks: dict[str, Any],
    name: str,
    value: Any,
    threshold: float,
) -> None:
    checks[name] = value
    if value is None or float(value) < threshold:
        blockers.append(f"{name}-below-threshold:{_fmt_float(value)}<{threshold:.6f}")


def _check_float_ceiling(
    blockers: list[str],
    checks: dict[str, Any],
    name: str,
    value: Any,
    threshold: float,
) -> None:
    checks[name] = value
    if value is None or float(value) > threshold:
        blockers.append(f"{name}-above-threshold:{_fmt_float(value)}>{threshold:.6f}")


def _infer_json_kind(payload: dict[str, Any]) -> str:
    kind = payload.get("kind")
    if isinstance(kind, str) and kind:
        return kind
    if _looks_like_depth_diagnostic(payload):
        return "objgauss-depth-visibility-vote-diagnostic-v1"
    if isinstance(payload.get("slot_alignment"), dict):
        return "objgauss-cross-view-slot-alignment-manifest-v1"
    if isinstance(payload.get("clip_scoring"), dict):
        return "objgauss-clip-scored-mask-manifest-v1"
    if isinstance(payload.get("training"), dict) and "asset_id" in payload:
        return "objgauss-training-output-manifest-v1"
    if _looks_like_training_summary(payload):
        return "objgauss-mask-training-summary-v1"
    if isinstance(payload.get("frames"), list):
        return "objgauss-mask-manifest-v1"
    return "objgauss-json-evidence-v1"


def _looks_like_training_summary(payload: dict[str, Any]) -> bool:
    return "initial_loss" in payload and "final_loss" in payload and "iterations" in payload


def _looks_like_depth_diagnostic(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("baseline"), dict) and isinstance(payload.get("depth_aware"), dict)


def _validate_thresholds(
    *,
    min_supervised_fraction: float,
    max_vote_conflict_fraction: float,
    min_slot_balance_score: float,
    min_object_active_slots: int,
) -> None:
    for name, value in {
        "min_supervised_fraction": min_supervised_fraction,
        "max_vote_conflict_fraction": max_vote_conflict_fraction,
        "min_slot_balance_score": min_slot_balance_score,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if min_object_active_slots < 0:
        raise ValueError("min_object_active_slots must be >= 0")


def _json_safe_dict(value: dict[str, Any]) -> dict[str, Any]:
    return _json_safe(value) if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _naming_summary(candidate: dict[str, Any]) -> str:
    slot = candidate.get("slot_naming_quality")
    if isinstance(slot, dict):
        status = "pass" if slot.get("passed") else "fail"
        counts = slot.get("slot_label_counts")
        return f"slot {status} {counts if counts else ''}".strip()
    mask = candidate.get("mask_naming_quality")
    if isinstance(mask, dict):
        status = "pass" if mask.get("passed") else "fail"
        counts = mask.get("top_label_counts")
        return f"mask {status} {counts if counts else ''}".strip()
    return "-"


def _loss_summary(training: dict[str, Any]) -> str:
    if not training:
        return "-"
    initial = training.get("initial_loss")
    final = training.get("final_loss")
    if initial is None or final is None:
        return "-"
    marker = "down" if training.get("loss_reduced") else "flat"
    return f"{_fmt_float(initial)}->{_fmt_float(final)} {marker}"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except Exception:
        return "-"
    if not np.isfinite(numeric):
        return "-"
    return f"{numeric:.6f}"


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
