"""Dataset contract for explicit BOP local-row batch specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA = (
    "objgauss-objectstate-bop-local-row-batch-spec-v1"
)


def read_objectstate_bop_local_row_batch_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("BOP local row batch spec must contain a JSON object")
    return validate_objectstate_bop_local_row_batch_spec(payload)


def validate_objectstate_bop_local_row_batch_spec(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP local row batch spec must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA:
        raise ValueError(
            "unsupported BOP local row batch spec schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_local_row_batch_spec":
        raise ValueError("BOP local row batch spec kind is unsupported")
    batch = payload.get("batch")
    if not isinstance(batch, Mapping):
        raise ValueError("BOP local row batch spec requires batch")
    batch_id = batch.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("BOP local row batch spec requires batch_id")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("BOP local row batch defaults must be a mapping")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("BOP local row batch spec requires non-empty samples")
    sample_ids: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"BOP local row batch sample {index} must be mapping")
        for key in ("sample_id", "scene_root", "candidate_artifact"):
            if not isinstance(sample.get(key), str) or not sample[key]:
                raise ValueError(
                    f"BOP local row batch sample {index} requires {key}"
                )
        sample_id = sample["sample_id"]
        if sample_id in sample_ids:
            raise ValueError(f"BOP local row batch duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        for optional_key in ("output_root", "condition_sidecar"):
            value = sample.get(optional_key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    f"BOP local row batch sample {index} invalid {optional_key}"
                )
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("local_only")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP local row batch spec must preserve claim policy")
    return dict(payload)


__all__ = (
    "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA",
    "read_objectstate_bop_local_row_batch_spec",
    "validate_objectstate_bop_local_row_batch_spec",
)
