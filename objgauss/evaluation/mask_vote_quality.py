from __future__ import annotations

from typing import Any

__all__ = ("mask_vote_quality_check",)


def mask_vote_quality_check(
    training: dict[str, Any],
    *,
    expected_slots: int | None = None,
) -> tuple[bool, str]:
    quality = training.get("vote_quality") if isinstance(training, dict) else None
    if not isinstance(quality, dict):
        return False, "missing vote_quality"
    per_slot = quality.get("per_slot")
    if not isinstance(per_slot, list) or not per_slot:
        return False, "missing per_slot coverage"
    if expected_slots is not None and len(per_slot) != expected_slots:
        return False, f"per_slot={len(per_slot)} expected={expected_slots}"
    supervised = int(quality.get("supervised_gaussians", 0) or 0)
    supervised_fraction = float(quality.get("supervised_fraction", 0.0) or 0.0)
    conflict = (
        quality.get("vote_conflict")
        if isinstance(quality.get("vote_conflict"), dict)
        else {}
    )
    conflict_fraction = float(conflict.get("fraction", 0.0) or 0.0)
    entropy = float(conflict.get("normalized_target_entropy", 0.0) or 0.0)
    ok = supervised > 0 and supervised_fraction > 0.0
    detail = (
        f"supervised_gaussians={supervised} "
        f"supervised_fraction={supervised_fraction:.6f} "
        f"conflict_fraction={conflict_fraction:.6f} "
        f"normalized_target_entropy={entropy:.6f} "
        f"slots={len(per_slot)}"
    )
    return ok, detail
