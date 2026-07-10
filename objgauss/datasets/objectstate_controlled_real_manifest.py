from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA = (
    "objgauss-objectstate-controlled-real-manifest-v1"
)


def read_objectstate_controlled_real_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("controlled real manifest JSON must be an object")
    return validate_objectstate_controlled_real_manifest(payload)


def validate_objectstate_controlled_real_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("controlled real manifest must be a mapping")
    if manifest.get("schema") != OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported controlled real manifest schema: {manifest.get('schema')}"
        )
    sample = _validate_sample(manifest.get("sample"))
    ground_truth = _validate_ground_truth(manifest.get("ground_truth"))
    evidence_rows = manifest.get("evidence_rows")
    if not isinstance(evidence_rows, Sequence) or isinstance(
        evidence_rows, (str, bytes)
    ):
        raise TypeError("controlled real manifest evidence_rows must be a sequence")
    checked_rows = tuple(_validate_evidence_row(row) for row in evidence_rows)
    if not checked_rows:
        raise ValueError("controlled real manifest requires at least one evidence row")
    return {
        "schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "sample": sample,
        "ground_truth": ground_truth,
        "evidence_rows": checked_rows,
    }


def _validate_sample(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled real sample must be a mapping")
    sample_id = _required_string(value, "sample_id")
    object_category = _required_string(value, "object_category")
    scenario = _required_string(value, "scenario")
    license_text = _required_string(value, "license")
    source_kind = str(value.get("source_kind", "controlled_real"))
    if source_kind != "controlled_real":
        raise ValueError(
            "controlled real manifest sample.source_kind must be controlled_real"
        )
    return {
        "sample_id": sample_id,
        "source_kind": source_kind,
        "object_category": object_category,
        "scenario": scenario,
        "observation_modalities": _string_tuple(
            value.get("observation_modalities"),
            "observation_modalities",
        ),
        "artifact_refs": _string_tuple(value.get("artifact_refs"), "artifact_refs"),
        "license": license_text,
    }


def _validate_ground_truth(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled real ground_truth must be a mapping")
    return {
        "identity": bool(value.get("identity", False)),
        "pose": bool(value.get("pose", False)),
        "action": bool(value.get("action", False)),
        "timestamp": bool(value.get("timestamp", False)),
    }


def _validate_evidence_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("controlled real evidence rows must be mappings")
    evidence_kind = _required_string(value, "evidence_kind")
    status = _required_string(value, "status")
    if evidence_kind not in {"identity", "prediction", "intervention"}:
        raise ValueError(f"unsupported controlled real evidence_kind: {evidence_kind}")
    if status not in {"pass", "fail", "blocked"}:
        raise ValueError(f"unsupported controlled real row status: {status}")
    metrics = value.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("controlled real evidence row metrics must be a mapping")
    block_reason = value.get("block_reason")
    failure_reason = value.get("failure_reason")
    if status == "blocked" and not block_reason:
        raise ValueError("blocked controlled real evidence rows require block_reason")
    if status == "fail" and not failure_reason:
        raise ValueError("failed controlled real evidence rows require failure_reason")
    if status == "pass" and (block_reason or failure_reason):
        raise ValueError(
            "pass controlled real evidence rows cannot carry block or failure reasons"
        )
    row = {
        "evidence_kind": evidence_kind,
        "status": status,
        "metrics": dict(metrics),
    }
    if "row_id" in value:
        row["row_id"] = _required_string(value, "row_id")
    if "artifact_refs" in value:
        row["artifact_refs"] = _string_tuple(
            value.get("artifact_refs"), "artifact_refs"
        )
    if block_reason:
        row["block_reason"] = str(block_reason)
    if failure_reason:
        row["failure_reason"] = str(failure_reason)
    return row


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(str(item) for item in value)
    if not normalized or any(not item for item in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized
