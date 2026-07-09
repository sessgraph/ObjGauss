from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA = (
    "objgauss-objectstate-teacher-evidence-batch-v1"
)
OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA = (
    "objgauss-objectstate-teacher-evidence-contract-v1"
)
OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA = (
    "objgauss-objectstate-teacher-evidence-contract-summary-v1"
)

TEACHER_EVIDENCE_SOURCES = (
    "dino_v2",
    "clip",
    "sam2",
    "grounding_dino",
    "tracking",
    "teacher_fusion",
    "synthetic_semantic",
    "manual_fixture",
)
TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES = (
    "dino_v2",
    "clip",
    "sam2",
    "grounding_dino",
    "tracking",
    "teacher_fusion",
)
TEACHER_EVIDENCE_LEAKAGE_RISK_LEVELS = ("none", "low", "medium", "high")
TEACHER_EVIDENCE_TRAINING_RISK_LEVELS = ("none", "low")
TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS = (
    "physical_identity",
    "physical_identity_label",
    "identity_label",
    "target_assignment",
    "oracle_object_id",
    "gt_object_id",
    "ground_truth_object_id",
    "test_label",
)
TEACHER_EVIDENCE_REQUIRED_PROVENANCE_KEYS = (
    "producer",
    "feature_space",
    "input_refs",
    "generation_method",
)


@dataclass(frozen=True)
class TeacherEvidenceBatch:
    sample_id: str
    gaussian_ids: Sequence[str]
    feature_matrix: np.ndarray
    evidence_policy: str = "semantic"
    source: str = "synthetic_semantic"
    confidence: np.ndarray | float | None = None
    uncertainty: np.ndarray | float | None = None
    provenance: Mapping[str, Any] | None = None
    allowed_for_training: bool = False
    allowed_for_evaluation: bool = True
    leakage_risk: str = "medium"
    schema: str = OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA

    @property
    def evidence_count(self) -> int:
        return int(self.feature_matrix.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.feature_matrix.shape[1])

    def as_dict(self) -> dict[str, Any]:
        return teacher_evidence_batch_summary(self)


def validate_teacher_evidence_batch(batch: TeacherEvidenceBatch) -> TeacherEvidenceBatch:
    if not isinstance(batch, TeacherEvidenceBatch):
        raise TypeError("teacher evidence batch must be TeacherEvidenceBatch")
    if batch.schema != OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA:
        raise ValueError(f"unsupported teacher evidence batch schema: {batch.schema}")
    sample_id = _non_empty_string(batch.sample_id, "sample_id")
    gaussian_ids = _gaussian_ids(batch.gaussian_ids)
    features = _feature_matrix(batch.feature_matrix, rows=len(gaussian_ids))
    policy = _non_empty_string(batch.evidence_policy, "evidence_policy")
    source = _source(batch.source)
    provenance = _provenance(batch.provenance)
    leakage_risk = _leakage_risk(batch.leakage_risk)
    confidence = _optional_metric_vector(
        batch.confidence,
        "confidence",
        rows=features.shape[0],
        default=1.0,
    )
    uncertainty = _optional_metric_vector(
        batch.uncertainty,
        "uncertainty",
        rows=features.shape[0],
        default=0.0,
    )
    allowed_for_training = bool(batch.allowed_for_training)
    allowed_for_evaluation = bool(batch.allowed_for_evaluation)
    forbidden = _forbidden_provenance_keys(provenance)
    if forbidden:
        raise ValueError(
            "teacher evidence provenance contains forbidden GT leakage keys: "
            + ", ".join(forbidden)
        )
    if allowed_for_training and leakage_risk not in TEACHER_EVIDENCE_TRAINING_RISK_LEVELS:
        raise ValueError("teacher evidence allowed_for_training requires leakage_risk none or low")
    if allowed_for_training and source not in TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES:
        raise ValueError("teacher evidence allowed_for_training requires inference-time source")
    if not allowed_for_training and not allowed_for_evaluation:
        raise ValueError("teacher evidence must be allowed for training or evaluation")
    return TeacherEvidenceBatch(
        sample_id=sample_id,
        gaussian_ids=gaussian_ids,
        feature_matrix=features,
        evidence_policy=policy,
        source=source,
        confidence=confidence,
        uncertainty=uncertainty,
        provenance=provenance,
        allowed_for_training=allowed_for_training,
        allowed_for_evaluation=allowed_for_evaluation,
        leakage_risk=leakage_risk,
        schema=batch.schema,
    )


def teacher_evidence_batch_summary(batch: TeacherEvidenceBatch) -> dict[str, Any]:
    checked = validate_teacher_evidence_batch(batch)
    provenance = dict(checked.provenance or {})
    payload = {
        "schema": OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA,
        "kind": "objectstate_teacher_evidence_batch",
        "sample_id": checked.sample_id,
        "evidence_policy": checked.evidence_policy,
        "source": checked.source,
        "evidence_count": int(checked.feature_matrix.shape[0]),
        "feature_dim": int(checked.feature_matrix.shape[1]),
        "gaussian_id_count": len(checked.gaussian_ids),
        "feature_matrix": {
            "shape": list(checked.feature_matrix.shape),
            "dtype": str(checked.feature_matrix.dtype),
            "inline_values_stored": False,
        },
        "confidence": _metric_summary(checked.confidence),
        "uncertainty": _metric_summary(checked.uncertainty),
        "provenance": provenance,
        "permissions": {
            "allowed_for_training": bool(checked.allowed_for_training),
            "allowed_for_evaluation": bool(checked.allowed_for_evaluation),
        },
        "leakage": {
            "risk": checked.leakage_risk,
            "forbidden_provenance_keys_present": False,
            "training_allowed_by_leakage_policy": bool(checked.allowed_for_training),
            "source_is_inference_time": checked.source in TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES,
            "requires_leakage_audit": checked.leakage_risk in {"medium", "high"}
            or checked.source not in TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES,
        },
        "claim_policy": {
            "teacher_evidence_is_perception_evidence": True,
            "teacher_evidence_is_not_ground_truth_identity": True,
            "target_assignment_is_forbidden_in_provenance": True,
            "physical_identity_label_is_forbidden_in_provenance": True,
            "does_not_claim_identity_gate_pass": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "runs_teacher_model": False,
            "downloads_teacher_weights": False,
            "creates_ground_truth": False,
            "runs_leakage_audit": False,
            "trains_model": False,
            "uses_renderer_loss": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_teacher_evidence_batch_summary(payload)


def validate_teacher_evidence_batch_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("teacher evidence batch summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA:
        raise ValueError(f"unsupported teacher evidence batch summary schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_teacher_evidence_batch":
        raise ValueError("teacher evidence batch summary kind is unsupported")
    _non_empty_string(payload.get("sample_id"), "sample_id")
    _non_empty_string(payload.get("evidence_policy"), "evidence_policy")
    _source(payload.get("source"))
    evidence_count = _positive_int(payload.get("evidence_count"), "evidence_count")
    feature_dim = _positive_int(payload.get("feature_dim"), "feature_dim")
    gaussian_id_count = _positive_int(payload.get("gaussian_id_count"), "gaussian_id_count")
    if evidence_count != gaussian_id_count:
        raise ValueError("teacher evidence summary gaussian_id_count must match evidence_count")
    matrix = _mapping(payload, "feature_matrix")
    if matrix.get("shape") != [evidence_count, feature_dim]:
        raise ValueError("teacher evidence summary feature_matrix shape mismatch")
    if matrix.get("inline_values_stored") is not False:
        raise ValueError("teacher evidence summary must not inline feature values")
    _metric_summary_mapping(_mapping(payload, "confidence"), "confidence", rows=evidence_count)
    _metric_summary_mapping(_mapping(payload, "uncertainty"), "uncertainty", rows=evidence_count)
    provenance = _mapping(payload, "provenance")
    missing = [key for key in TEACHER_EVIDENCE_REQUIRED_PROVENANCE_KEYS if key not in provenance]
    if missing:
        raise ValueError("teacher evidence summary provenance missing required keys")
    leakage = _mapping(payload, "leakage")
    _leakage_risk(leakage.get("risk"))
    if leakage.get("forbidden_provenance_keys_present") is not False:
        raise ValueError("teacher evidence summary cannot contain forbidden provenance keys")
    permissions = _mapping(payload, "permissions")
    if not isinstance(permissions.get("allowed_for_training"), bool):
        raise ValueError("teacher evidence permissions allowed_for_training must be bool")
    if not isinstance(permissions.get("allowed_for_evaluation"), bool):
        raise ValueError("teacher evidence permissions allowed_for_evaluation must be bool")
    if not permissions["allowed_for_training"] and not permissions["allowed_for_evaluation"]:
        raise ValueError("teacher evidence summary must be allowed for training or evaluation")
    if permissions["allowed_for_training"] and leakage["risk"] not in TEACHER_EVIDENCE_TRAINING_RISK_LEVELS:
        raise ValueError("teacher evidence training permission contradicts leakage risk")
    if permissions["allowed_for_training"] and payload["source"] not in TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES:
        raise ValueError("teacher evidence training permission requires inference-time source")
    claim_policy = _mapping(payload, "claim_policy")
    if any(not bool(value) for value in claim_policy.values()):
        raise ValueError("teacher evidence batch summary must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("teacher evidence batch summary cannot claim non-goals")
    return dict(payload)


def objectstate_teacher_evidence_contract_summary(
    batch: TeacherEvidenceBatch | None = None,
) -> dict[str, Any]:
    batch_summary = None if batch is None else teacher_evidence_batch_summary(batch)
    payload = {
        "schema": OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA,
        "kind": "objectstate_teacher_evidence_contract_summary",
        "contract_schema": OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA,
        "batch_schema": OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA,
        "required_batch_fields": [
            "sample_id",
            "gaussian_ids",
            "evidence_policy",
            "feature_matrix",
            "source",
            "confidence",
            "uncertainty",
            "provenance",
            "allowed_for_training",
            "allowed_for_evaluation",
            "leakage_risk",
        ],
        "allowed_sources": list(TEACHER_EVIDENCE_SOURCES),
        "inference_time_sources": list(TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES),
        "leakage_risk_levels": list(TEACHER_EVIDENCE_LEAKAGE_RISK_LEVELS),
        "training_allowed_risk_levels": list(TEACHER_EVIDENCE_TRAINING_RISK_LEVELS),
        "required_provenance_keys": list(TEACHER_EVIDENCE_REQUIRED_PROVENANCE_KEYS),
        "forbidden_provenance_keys": list(TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS),
        "contract_rules": {
            "teacher_evidence_is_perception_evidence": True,
            "teacher_evidence_is_not_ground_truth_identity": True,
            "training_requires_inference_time_source": True,
            "training_requires_leakage_risk_none_or_low": True,
            "evaluation_can_include_synthetic_fixture_with_declared_risk": True,
            "forbids_physical_identity_label_or_target_assignment_in_provenance": True,
        },
        "next_required_audits": [
            "semantic_feature_shuffle",
            "physical_label_ban",
            "random_semantic_baseline",
            "train_test_semantic_source_split",
        ],
        "example_batch_summary": batch_summary,
        "claim_policy": {
            "defines_teacher_evidence_contract": True,
            "does_not_run_teacher_model": True,
            "does_not_run_leakage_audit": True,
            "does_not_claim_identity_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_teacher_weights": False,
            "runs_dino": False,
            "runs_sam": False,
            "runs_tracking": False,
            "creates_ground_truth": False,
            "trains_model": False,
            "runs_long_smoke": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_teacher_evidence_contract_summary(payload)


def validate_objectstate_teacher_evidence_contract_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("teacher evidence contract summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA:
        raise ValueError(
            "unsupported teacher evidence contract summary schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_teacher_evidence_contract_summary":
        raise ValueError("teacher evidence contract summary kind is unsupported")
    if payload.get("contract_schema") != OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA:
        raise ValueError("teacher evidence contract summary has unsupported contract_schema")
    if payload.get("batch_schema") != OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA:
        raise ValueError("teacher evidence contract summary has unsupported batch_schema")
    _string_list(payload.get("required_batch_fields"), "required_batch_fields")
    if set(payload.get("allowed_sources", ())) != set(TEACHER_EVIDENCE_SOURCES):
        raise ValueError("teacher evidence contract summary allowed sources mismatch")
    if set(payload.get("inference_time_sources", ())) != set(TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES):
        raise ValueError("teacher evidence contract summary inference sources mismatch")
    if tuple(payload.get("leakage_risk_levels", ())) != TEACHER_EVIDENCE_LEAKAGE_RISK_LEVELS:
        raise ValueError("teacher evidence contract summary leakage risk levels mismatch")
    if tuple(payload.get("training_allowed_risk_levels", ())) != TEACHER_EVIDENCE_TRAINING_RISK_LEVELS:
        raise ValueError("teacher evidence contract summary training risk levels mismatch")
    if set(payload.get("required_provenance_keys", ())) != set(TEACHER_EVIDENCE_REQUIRED_PROVENANCE_KEYS):
        raise ValueError("teacher evidence contract summary provenance keys mismatch")
    if set(payload.get("forbidden_provenance_keys", ())) != set(TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS):
        raise ValueError("teacher evidence contract summary forbidden keys mismatch")
    rules = _mapping(payload, "contract_rules")
    if any(not bool(value) for value in rules.values()):
        raise ValueError("teacher evidence contract rules must be true")
    audits = _string_list(payload.get("next_required_audits"), "next_required_audits")
    for required in (
        "semantic_feature_shuffle",
        "physical_label_ban",
        "random_semantic_baseline",
        "train_test_semantic_source_split",
    ):
        if required not in audits:
            raise ValueError(f"teacher evidence contract missing audit {required}")
    batch_summary = payload.get("example_batch_summary")
    if batch_summary is not None:
        validate_teacher_evidence_batch_summary(batch_summary)
    claim_policy = _mapping(payload, "claim_policy")
    if any(not bool(value) for value in claim_policy.values()):
        raise ValueError("teacher evidence contract summary must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("teacher evidence contract summary cannot claim non-goals")
    return dict(payload)


def _feature_matrix(value: np.ndarray, *, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("teacher evidence feature_matrix must be 2D")
    if array.shape[0] != rows:
        raise ValueError("teacher evidence feature_matrix rows must match gaussian_ids")
    if array.shape[1] < 1:
        raise ValueError("teacher evidence feature_matrix must contain at least one feature")
    if not np.isfinite(array).all():
        raise ValueError("teacher evidence feature_matrix must contain only finite values")
    return array.astype(np.float32, copy=False)


def _gaussian_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("teacher evidence gaussian_ids must be a sequence")
    ids = tuple(_non_empty_string(item, "gaussian_id") for item in value)
    if not ids:
        raise ValueError("teacher evidence gaussian_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("teacher evidence gaussian_ids must be unique")
    return ids


def _optional_metric_vector(
    value: np.ndarray | float | None,
    label: str,
    *,
    rows: int,
    default: float,
) -> np.ndarray:
    if value is None:
        return np.full(rows, float(default), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full(rows, float(array), dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"teacher evidence {label} must be scalar or 1D")
    if array.shape[0] != rows:
        raise ValueError(f"teacher evidence {label} length must match gaussian_ids")
    if not np.isfinite(array).all():
        raise ValueError(f"teacher evidence {label} must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"teacher evidence {label} must be in [0,1]")
    return array.astype(np.float32, copy=False)


def _metric_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "mean": float(array.mean()) if array.size else 0.0,
    }


def _metric_summary_mapping(payload: Mapping[str, Any], label: str, *, rows: int) -> None:
    if payload.get("shape") != [rows]:
        raise ValueError(f"teacher evidence {label} summary shape mismatch")
    for key in ("min", "max", "mean"):
        number = _finite(payload.get(key), f"{label}.{key}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"teacher evidence {label}.{key} must be in [0,1]")


def _provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        raise ValueError("teacher evidence provenance is required")
    if not isinstance(value, Mapping):
        raise ValueError("teacher evidence provenance must be a mapping")
    provenance = dict(value)
    missing = [key for key in TEACHER_EVIDENCE_REQUIRED_PROVENANCE_KEYS if key not in provenance]
    if missing:
        raise ValueError("teacher evidence provenance missing required keys: " + ", ".join(missing))
    return provenance


def _forbidden_provenance_keys(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            normalized = key_text.lower()
            if normalized in TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS:
                found.append(f"{prefix}{key_text}")
            found.extend(_forbidden_provenance_keys(nested, prefix=f"{prefix}{key_text}."))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_provenance_keys(nested, prefix=f"{prefix}{index}."))
    return _dedupe(found)


def _source(value: Any) -> str:
    source = _non_empty_string(value, "source")
    if source not in TEACHER_EVIDENCE_SOURCES:
        raise ValueError(f"unsupported teacher evidence source: {source}")
    return source


def _leakage_risk(value: Any) -> str:
    risk = _non_empty_string(value, "leakage_risk")
    if risk not in TEACHER_EVIDENCE_LEAKAGE_RISK_LEVELS:
        raise ValueError(f"unsupported teacher evidence leakage_risk: {risk}")
    return risk


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"teacher evidence {label} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"teacher evidence {label} must be a positive integer")
    return int(value)


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"teacher evidence {label} must be a list of strings")
    return list(value)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"teacher evidence requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"teacher evidence {label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"teacher evidence {label} must be finite")
    return number


def _dedupe(items: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
