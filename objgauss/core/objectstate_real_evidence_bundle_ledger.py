from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_real_evidence_bundle import (
    objectstate_real_evidence_bundle_summary,
    read_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle_summary,
)
from objgauss.core.objectstate_real_identity_rows import (
    objectstate_real_identity_rows_summary,
    validate_objectstate_real_identity_rows_summary,
)
from objgauss.core.objectstate_real_intervention_rows import (
    objectstate_real_intervention_rows_summary,
    validate_objectstate_real_intervention_rows_summary,
)
from objgauss.core.objectstate_real_prediction_rows import (
    objectstate_real_prediction_rows_summary,
    validate_objectstate_real_prediction_rows_summary,
)
from objgauss.core.objectstate_reality_row_ledger import (
    objectstate_reality_row_ledger,
    validate_objectstate_reality_row_ledger_summary,
)

OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA = (
    "objgauss-objectstate-real-evidence-bundle-ledger-v1"
)
_ACCOUNTING_STATUSES = ("pass", "fail", "evidence_incomplete", "unsupported")


def write_objectstate_real_evidence_bundle_ledger(
    bundle_paths: Sequence[str | Path],
    *,
    output_root: str | Path,
    synthetic_smoke_passed: bool = True,
    min_real_or_public_rows: int = 1,
) -> dict[str, Any]:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    row_summary_paths: list[Path] = []
    for index, raw_bundle_path in enumerate(bundle_paths):
        bundle_path = Path(raw_bundle_path)
        bundle = read_objectstate_real_evidence_bundle(bundle_path)
        bundle_summary = objectstate_real_evidence_bundle_summary(bundle)
        sample_id = str(bundle_summary["sample"]["sample_id"])
        sample_root = output_root_path / f"{index:03d}-{_slug(sample_id)}"
        sample_root.mkdir(parents=True, exist_ok=True)

        identity_summary = objectstate_real_identity_rows_summary(
            bundle,
            synthetic_smoke_passed=synthetic_smoke_passed,
            min_real_or_public_rows=min_real_or_public_rows,
        )
        prediction_summary = objectstate_real_prediction_rows_summary(
            bundle,
            synthetic_smoke_passed=synthetic_smoke_passed,
            min_real_or_public_rows=min_real_or_public_rows,
        )
        intervention_summary = objectstate_real_intervention_rows_summary(
            bundle,
            synthetic_smoke_passed=synthetic_smoke_passed,
            min_real_or_public_rows=min_real_or_public_rows,
        )

        bundle_summary_path = sample_root / "real-evidence-bundle-summary.json"
        identity_summary_path = sample_root / "real-identity-rows-summary.json"
        prediction_summary_path = sample_root / "real-prediction-rows-summary.json"
        intervention_summary_path = sample_root / "real-intervention-rows-summary.json"
        _write_json(bundle_summary_path, bundle_summary)
        _write_json(identity_summary_path, identity_summary)
        _write_json(prediction_summary_path, prediction_summary)
        _write_json(intervention_summary_path, intervention_summary)

        row_summary_paths.extend(
            (identity_summary_path, prediction_summary_path, intervention_summary_path)
        )
        records.append(
            {
                "bundle_path": str(bundle_path),
                "sample_id": sample_id,
                "sample_output_root": str(sample_root),
                "bundle_summary_path": str(bundle_summary_path),
                "identity_summary_path": str(identity_summary_path),
                "prediction_summary_path": str(prediction_summary_path),
                "intervention_summary_path": str(intervention_summary_path),
                "bundle_status": bundle_summary["status"],
                "identity_status": identity_summary["status"],
                "prediction_status": prediction_summary["status"],
                "intervention_status": intervention_summary["status"],
                "bundle_readiness": dict(bundle_summary["readiness"]),
                "bundle_metrics": dict(bundle_summary["metrics"]),
                "evidence_accounts": dict(bundle_summary["evidence_accounts"]),
                "row_counts": {
                    "identity_rows": identity_summary["row_counts"]["identity_rows"],
                    "prediction_rows": prediction_summary["row_counts"]["prediction_rows"],
                    "intervention_rows": (
                        intervention_summary["row_counts"]["intervention_rows"]
                    ),
                },
                "accounting_status_counts": _record_accounting_status_counts(
                    bundle_summary,
                    identity_summary,
                    prediction_summary,
                    intervention_summary,
                ),
            }
        )

    ledger = objectstate_reality_row_ledger(
        row_summary_paths,
        synthetic_smoke_passed=synthetic_smoke_passed,
    )
    ledger_summary_path = output_root_path / "reality-row-ledger.json"
    blocked_rows_path = output_root_path / "reality-row-ledger-blocked.md"
    matrix_path = output_root_path / "state-variable-evidence-matrix.md"
    next_actions_path = output_root_path / "reality-row-ledger-next-actions.md"
    _write_json(ledger_summary_path, ledger)
    blocked_rows_path.write_text(ledger["blocked_rows_markdown"], encoding="utf-8")
    matrix_path.write_text(
        ledger["state_variable_evidence_matrix_markdown"],
        encoding="utf-8",
    )
    next_actions_path.write_text(ledger["next_actions_markdown"], encoding="utf-8")

    payload = {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA,
        "kind": "objectstate_real_evidence_bundle_ledger",
        "status": (
            "objectstate_real_evidence_bundle_ledger_reviewable"
            if records and ledger["status"] == "objectstate_reality_row_ledger_reviewable"
            else "objectstate_real_evidence_bundle_ledger_incomplete"
        ),
        "output_root": str(output_root_path),
        "bundle_count": len(records),
        "row_summary_count": len(row_summary_paths),
        "records": records,
        "ledger_summary_path": str(ledger_summary_path),
        "blocked_rows_path": str(blocked_rows_path),
        "state_variable_evidence_matrix_path": str(matrix_path),
        "next_actions_path": str(next_actions_path),
        "ledger": ledger,
        "row_counts": {
            "row_count": ledger["row_count"],
            "pass_row_count": ledger["pass_row_count"],
            "fail_row_count": ledger["fail_row_count"],
            "blocked_row_count": ledger["blocked_row_count"],
            "evidence_incomplete_row_count": _accounting_status_counts(records)["all"][
                "evidence_incomplete"
            ],
            "unsupported_row_count": _accounting_status_counts(records)["all"][
                "unsupported"
            ],
        },
        "accounting_status_counts": _accounting_status_counts(records),
        "evidence_accounts": _evidence_accounts(records),
        "claim_policy": {
            "bundle_ledger_is_auditable_handoff": True,
            "full_reality_row_ledger_is_authoritative": True,
            "static_scene_evidence_is_separate_from_state_variable_evidence": True,
            "evidence_incomplete_is_not_model_fail": True,
            "blocked_rows_are_not_pass_rows": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_identity_model": False,
            "runs_prediction_model": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    summary = validate_objectstate_real_evidence_bundle_ledger_summary(payload)
    _write_json(output_root_path / "real-evidence-bundle-ledger.json", summary)
    return summary


def validate_objectstate_real_evidence_bundle_ledger_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("real evidence bundle ledger must be a mapping")
    if payload.get("schema") != OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA:
        raise ValueError(
            "unsupported real evidence bundle ledger schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_real_evidence_bundle_ledger":
        raise ValueError("real evidence bundle ledger kind is unsupported")
    if payload.get("status") not in {
        "objectstate_real_evidence_bundle_ledger_reviewable",
        "objectstate_real_evidence_bundle_ledger_incomplete",
    }:
        raise ValueError("real evidence bundle ledger status is unsupported")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("real evidence bundle ledger requires records")
    if payload.get("bundle_count") != len(records):
        raise ValueError("real evidence bundle ledger bundle_count mismatch")
    if payload.get("row_summary_count") != len(records) * 3:
        raise ValueError("real evidence bundle ledger row_summary_count mismatch")
    for record in records:
        _validate_record(record)
    ledger = payload.get("ledger")
    if not isinstance(ledger, Mapping):
        raise ValueError("real evidence bundle ledger requires ledger")
    validate_objectstate_reality_row_ledger_summary(ledger)
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("real evidence bundle ledger requires row_counts")
    for key in ("row_count", "pass_row_count", "fail_row_count", "blocked_row_count"):
        if row_counts.get(key) != ledger.get(key):
            raise ValueError(f"real evidence bundle ledger row_counts.{key} mismatch")
    accounting_counts = payload.get("accounting_status_counts")
    if not isinstance(accounting_counts, Mapping):
        raise ValueError("real evidence bundle ledger requires accounting_status_counts")
    _validate_accounting_status_counts(accounting_counts)
    for key in ("evidence_incomplete", "unsupported"):
        row_key = f"{key}_row_count"
        if row_counts.get(row_key) != accounting_counts["all"][key]:
            raise ValueError(f"real evidence bundle ledger row_counts.{row_key} mismatch")
    accounts = payload.get("evidence_accounts")
    if not isinstance(accounts, Mapping):
        raise ValueError("real evidence bundle ledger requires evidence_accounts")
    for key in ("static_scene_evidence", "state_variable_evidence"):
        if not isinstance(accounts.get(key), Mapping):
            raise ValueError(f"real evidence bundle ledger missing account {key}")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("bundle_ledger_is_auditable_handoff")
        or not claim_policy.get("full_reality_row_ledger_is_authoritative")
        or not claim_policy.get(
            "static_scene_evidence_is_separate_from_state_variable_evidence"
        )
        or not claim_policy.get("evidence_incomplete_is_not_model_fail")
        or not claim_policy.get("blocked_rows_are_not_pass_rows")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("real evidence bundle ledger must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("real evidence bundle ledger cannot claim non-goal behavior")
    return dict(payload)


def _validate_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("real evidence bundle ledger record must be an object")
    for key in (
        "bundle_path",
        "sample_id",
        "sample_output_root",
        "bundle_summary_path",
        "identity_summary_path",
        "prediction_summary_path",
        "intervention_summary_path",
        "bundle_status",
        "identity_status",
        "prediction_status",
        "intervention_status",
    ):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"real evidence bundle ledger record missing {key}")
    bundle_summary = _read_json(record["bundle_summary_path"])
    validate_objectstate_real_evidence_bundle_summary(bundle_summary)
    identity_summary = _read_json(record["identity_summary_path"])
    validate_objectstate_real_identity_rows_summary(identity_summary)
    prediction_summary = _read_json(record["prediction_summary_path"])
    validate_objectstate_real_prediction_rows_summary(prediction_summary)
    intervention_summary = _read_json(record["intervention_summary_path"])
    validate_objectstate_real_intervention_rows_summary(intervention_summary)
    if record["sample_id"] != bundle_summary["sample"]["sample_id"]:
        raise ValueError("real evidence bundle ledger record sample_id mismatch")
    counts = record.get("accounting_status_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("real evidence bundle ledger record requires accounting_status_counts")
    _validate_accounting_status_counts(counts)


def _record_accounting_status_counts(
    bundle_summary: Mapping[str, Any],
    identity_summary: Mapping[str, Any],
    prediction_summary: Mapping[str, Any],
    intervention_summary: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    identity = _status_counts(
        identity_summary["metrics"].get("identity_accounting_status_counts", {})
    )
    prediction = _status_counts(
        prediction_summary["metrics"].get("prediction_accounting_status_counts", {})
    )
    intervention = _status_counts(
        intervention_summary["metrics"].get("intervention_accounting_status_counts", {})
    )
    bundle = _status_counts(
        bundle_summary["metrics"].get("gate_accounting_status_counts", {})
    )
    all_counts = _sum_status_counts(identity, prediction, intervention)
    return {
        "identity": identity,
        "prediction": prediction,
        "intervention": intervention,
        "bundle": bundle,
        "all": all_counts,
    }


def _accounting_status_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    totals = {
        "identity": _zero_status_counts(),
        "prediction": _zero_status_counts(),
        "intervention": _zero_status_counts(),
        "bundle": _zero_status_counts(),
        "all": _zero_status_counts(),
    }
    for record in records:
        counts = record.get("accounting_status_counts", {})
        if not isinstance(counts, Mapping):
            continue
        for section in totals:
            totals[section] = _sum_status_counts(
                totals[section],
                _status_counts(counts.get(section, {})),
            )
    return totals


def _status_counts(raw: Any) -> dict[str, int]:
    counts = _zero_status_counts()
    if isinstance(raw, Mapping):
        for status in _ACCOUNTING_STATUSES:
            counts[status] = int(raw.get(status, 0) or 0)
    return counts


def _zero_status_counts() -> dict[str, int]:
    return {status: 0 for status in _ACCOUNTING_STATUSES}


def _sum_status_counts(*items: Mapping[str, int]) -> dict[str, int]:
    result = _zero_status_counts()
    for item in items:
        for status in _ACCOUNTING_STATUSES:
            result[status] += int(item.get(status, 0) or 0)
    return result


def _validate_accounting_status_counts(counts: Mapping[str, Any]) -> None:
    for section in ("identity", "prediction", "intervention", "bundle", "all"):
        section_counts = counts.get(section)
        if not isinstance(section_counts, Mapping):
            raise ValueError(
                f"real evidence bundle ledger accounting_status_counts.{section} missing"
            )
        for status in _ACCOUNTING_STATUSES:
            value = section_counts.get(status)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    "real evidence bundle ledger accounting_status_counts "
                    f"{section}.{status} must be non-negative int"
                )


def _evidence_accounts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    static_available = 0
    state_ready = 0
    intervention_ready = 0
    for record in records:
        accounts = record["evidence_accounts"]
        static = accounts["static_scene_evidence"]
        state = accounts["state_variable_evidence"]
        if static["available"]:
            static_available += 1
        if state["available"]:
            state_ready += 1
        if record["bundle_readiness"]["intervention_accounting_ready"]:
            intervention_ready += 1
    return {
        "static_scene_evidence": {
            "available_bundle_count": static_available,
            "usable_for_state_variable_gate": False,
        },
        "state_variable_evidence": {
            "ready_bundle_count": state_ready,
            "intervention_ready_bundle_count": intervention_ready,
            "requires_full_reality_row_ledger": True,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "sample"
