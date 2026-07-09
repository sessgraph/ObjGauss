from __future__ import annotations

from typing import Any, Mapping

from objgauss.core.objectstate_controlled_capture_bundle_readiness import (
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA,
    validate_objectstate_controlled_capture_bundle_readiness_summary,
)
from objgauss.core.objectstate_controlled_capture_environment import (
    OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA,
    validate_objectstate_controlled_capture_environment_summary,
)
from objgauss.core.objectstate_temporal_assignment import (
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA,
    validate_objectstate_temporal_assignment_summary,
)

OBJECTSTATE_CONTROLLED_CAPTURE_HANDOFF_SCHEMA = (
    "objgauss-objectstate-controlled-capture-handoff-v1"
)
_CLAIM_POLICY_KEYS = (
    "requires_passed_temporal_assignment",
    "routes_to_existing_controlled_capture_toolchain",
    "keeps_real_capture_evidence_separate_from_synthetic_evidence",
    "does_not_claim_capture_completed",
    "does_not_claim_ground_truth_created",
    "does_not_claim_reality_gate_pass",
    "does_not_claim_world_model",
)
_NON_GOAL_KEYS = (
    "captures_video",
    "creates_ground_truth",
    "reconstructs_gaussians",
    "runs_identity_handoff",
    "runs_prediction_eval",
    "runs_intervention_eval",
    "trains_model",
    "uses_renderer_loss",
    "uses_dynamics",
    "uses_diffusion",
    "uses_replay_buffer",
    "mutates_viewer_defaults",
)


def objectstate_controlled_capture_handoff_summary(
    *,
    sample_id: str = "objectstate-controlled-capture-handoff-001",
    temporal_assignment_summary: Mapping[str, Any],
    capture_environment_summary: Mapping[str, Any] | None = None,
    capture_bundle_readiness_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    temporal = _temporal_digest(temporal_assignment_summary)
    environment = _environment_digest(capture_environment_summary)
    bundle = _bundle_digest(capture_bundle_readiness_summary)
    readiness = {
        "temporal_assignment_passed": temporal["status"] == "pass",
        "controlled_capture_environment_ready": environment["status"] == "ready",
        "controlled_capture_bundle_ready": bundle["status"] == "ready",
    }
    readiness["controlled_capture_collection_ready"] = bool(
        readiness["temporal_assignment_passed"]
        and readiness["controlled_capture_environment_ready"]
    )
    readiness["controlled_evidence_handoff_ready"] = bool(
        readiness["temporal_assignment_passed"]
        and readiness["controlled_capture_bundle_ready"]
    )
    blockers = _hard_blockers(readiness, environment=environment, bundle=bundle)
    status = _status(readiness, blockers)
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_HANDOFF_SCHEMA,
        "kind": "objectstate_controlled_capture_handoff",
        "status": status,
        "sample_id": str(sample_id),
        "temporal_assignment_schema": OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA,
        "capture_environment_schema": OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA,
        "capture_bundle_readiness_schema": (
            OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA
        ),
        "preconditions": {
            "temporal_assignment": temporal,
            "capture_environment": environment,
            "capture_bundle_readiness": bundle,
        },
        "readiness": readiness,
        "hard_blockers": blockers,
        "next_actions": _next_actions(readiness, environment=environment, bundle=bundle),
        "handoff_routes": {
            "environment_preflight_command": (
                "uv run objgauss object-state audit-controlled-capture-environment"
            ),
            "bundle_template_command": (
                "uv run objgauss object-state init-controlled-capture-bundle "
                "outputs/captures/controlled-tabletop-cup-box-001"
            ),
            "bundle_readiness_command": (
                "uv run objgauss object-state audit-controlled-capture-bundle-readiness "
                "outputs/captures/controlled-tabletop-cup-box-001 "
                "--summary-output "
                "outputs/captures/controlled-tabletop-cup-box-001/readiness-summary.json"
            ),
            "identity_handoff_command": (
                "uv run objgauss object-state controlled-identity-bundle-handoff "
                "<bundle-root> <objectstate-artifact.json>"
            ),
        },
        "claim_policy": {key: True for key in _CLAIM_POLICY_KEYS},
        "non_goals": {key: False for key in _NON_GOAL_KEYS},
    }
    return validate_objectstate_controlled_capture_handoff_summary(payload)


def validate_objectstate_controlled_capture_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported controlled capture handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_handoff":
        raise ValueError("controlled capture handoff kind is unsupported")
    if payload.get("status") not in {
        "objectstate_controlled_capture_handoff_ready",
        "objectstate_controlled_capture_collection_ready",
        "objectstate_controlled_capture_handoff_blocked",
    }:
        raise ValueError("controlled capture handoff status is unsupported")
    if payload.get("temporal_assignment_schema") != OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA:
        raise ValueError("controlled capture handoff temporal schema mismatch")
    if (
        payload.get("capture_environment_schema")
        != OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA
    ):
        raise ValueError("controlled capture handoff environment schema mismatch")
    if (
        payload.get("capture_bundle_readiness_schema")
        != OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA
    ):
        raise ValueError("controlled capture handoff bundle schema mismatch")
    preconditions = _mapping(payload, "preconditions")
    _validate_temporal_digest(_mapping(preconditions, "temporal_assignment"))
    _validate_environment_digest(_mapping(preconditions, "capture_environment"))
    _validate_bundle_digest(_mapping(preconditions, "capture_bundle_readiness"))
    readiness = _mapping(payload, "readiness")
    for key in (
        "temporal_assignment_passed",
        "controlled_capture_environment_ready",
        "controlled_capture_bundle_ready",
        "controlled_capture_collection_ready",
        "controlled_evidence_handoff_ready",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"controlled capture handoff readiness missing bool {key}")
    if readiness["controlled_capture_collection_ready"] is not bool(
        readiness["temporal_assignment_passed"]
        and readiness["controlled_capture_environment_ready"]
    ):
        raise ValueError("controlled capture collection readiness mismatch")
    if readiness["controlled_evidence_handoff_ready"] is not bool(
        readiness["temporal_assignment_passed"]
        and readiness["controlled_capture_bundle_ready"]
    ):
        raise ValueError("controlled evidence handoff readiness mismatch")
    expected = (
        "objectstate_controlled_capture_handoff_ready"
        if readiness["controlled_evidence_handoff_ready"]
        else (
            "objectstate_controlled_capture_collection_ready"
            if readiness["controlled_capture_collection_ready"]
            else "objectstate_controlled_capture_handoff_blocked"
        )
    )
    if payload["status"] != expected:
        raise ValueError("controlled capture handoff status mismatch")
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("controlled capture handoff hard_blockers must be list")
    if not isinstance(payload.get("next_actions"), list):
        raise ValueError("controlled capture handoff next_actions must be list")
    routes = _mapping(payload, "handoff_routes")
    for key in (
        "environment_preflight_command",
        "bundle_template_command",
        "bundle_readiness_command",
        "identity_handoff_command",
    ):
        if not isinstance(routes.get(key), str) or not routes[key]:
            raise ValueError(f"controlled capture handoff missing route {key}")
    claim_policy = _mapping(payload, "claim_policy")
    if any(not bool(claim_policy.get(key)) for key in _CLAIM_POLICY_KEYS):
        raise ValueError("controlled capture handoff must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(non_goals.get(key)) for key in _NON_GOAL_KEYS):
        raise ValueError("controlled capture handoff cannot claim non-goals")
    return dict(payload)


def _temporal_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_objectstate_temporal_assignment_summary(payload)
    gate = _mapping(checked, "next_stage_gate")
    return {
        "schema": checked["schema"],
        "status": (
            "pass" if checked["status"] == "objectstate_temporal_assignment_pass" else "reviewable"
        ),
        "sample_id": checked["sample_id"],
        "controlled_capture_allowed": bool(gate.get("controlled_capture_allowed")),
        "summary_path": checked.get("summary_path"),
        "blocked_reasons": [
            str(item)
            for item in gate.get("blocked_reasons", [])
            if isinstance(item, str)
        ],
    }


def _environment_digest(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "schema": OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA,
            "status": "missing",
            "controlled_capture_environment_ready": False,
            "hard_blockers": ["capture_environment_summary_missing"],
        }
    checked = validate_objectstate_controlled_capture_environment_summary(payload)
    readiness = _mapping(checked, "readiness")
    return {
        "schema": checked["schema"],
        "status": (
            "ready"
            if checked["status"] == "objectstate_controlled_capture_environment_ready"
            else "blocked"
        ),
        "controlled_capture_environment_ready": bool(
            readiness["controlled_capture_environment_ready"]
        ),
        "hard_blockers": [
            str(item) for item in checked.get("hard_blockers", []) if isinstance(item, str)
        ],
    }


def _bundle_digest(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "schema": OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA,
            "status": "missing",
            "identity_bundle_handoff_ready": False,
            "hard_blockers": ["capture_bundle_readiness_summary_missing"],
        }
    checked = validate_objectstate_controlled_capture_bundle_readiness_summary(payload)
    readiness = _mapping(checked, "readiness")
    return {
        "schema": checked["schema"],
        "status": (
            "ready"
            if checked["status"] == "objectstate_controlled_capture_bundle_readiness_ready"
            else "blocked"
        ),
        "identity_bundle_handoff_ready": bool(
            readiness["identity_bundle_handoff_ready"]
        ),
        "hard_blockers": [
            str(item) for item in checked.get("hard_blockers", []) if isinstance(item, str)
        ],
    }


def _hard_blockers(
    readiness: Mapping[str, bool],
    *,
    environment: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> list[str]:
    blockers = []
    if not readiness["temporal_assignment_passed"]:
        blockers.append("temporal_assignment_not_passed")
    if not readiness["controlled_capture_environment_ready"]:
        blockers.append(f"capture_environment_{environment['status']}")
    if not readiness["controlled_capture_bundle_ready"]:
        blockers.append(f"capture_bundle_{bundle['status']}")
    blockers.extend(str(item) for item in environment.get("hard_blockers", ()))
    blockers.extend(str(item) for item in bundle.get("hard_blockers", ()))
    return sorted(set(blockers))


def _status(readiness: Mapping[str, bool], blockers: list[str]) -> str:
    if readiness["controlled_evidence_handoff_ready"]:
        return "objectstate_controlled_capture_handoff_ready"
    if readiness["controlled_capture_collection_ready"]:
        return "objectstate_controlled_capture_collection_ready"
    if not blockers:
        raise ValueError("controlled capture handoff blocked status requires blockers")
    return "objectstate_controlled_capture_handoff_blocked"


def _next_actions(
    readiness: Mapping[str, bool],
    *,
    environment: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> list[str]:
    if readiness["controlled_evidence_handoff_ready"]:
        return [
            "run controlled-identity-bundle-handoff on the ready capture bundle",
            "then run identity-only reality gate and ledger package audit",
        ]
    actions = []
    if not readiness["controlled_capture_environment_ready"]:
        if environment["status"] == "missing":
            actions.append("run audit-controlled-capture-environment on the capture host")
        else:
            actions.append("clear controlled capture environment blockers")
    if readiness["controlled_capture_environment_ready"] and bundle["status"] == "missing":
        actions.extend(
            [
                "initialize a local controlled capture bundle skeleton",
                "capture RGB / Gaussian evidence and fill objects, frames and annotations CSVs",
                "run audit-controlled-capture-bundle-readiness",
            ]
        )
    elif bundle["status"] == "blocked":
        actions.append("clear controlled capture bundle readiness blockers")
    return actions


def _validate_temporal_digest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA:
        raise ValueError("controlled capture handoff temporal digest schema mismatch")
    if payload.get("status") not in {"pass", "reviewable"}:
        raise ValueError("controlled capture handoff temporal digest status unsupported")
    if payload.get("status") == "pass" and payload.get("controlled_capture_allowed") is not True:
        raise ValueError("passed temporal digest must allow controlled capture")
    if not isinstance(payload.get("blocked_reasons"), list):
        raise ValueError("controlled capture temporal blocked_reasons must be list")


def _validate_environment_digest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA:
        raise ValueError("controlled capture handoff environment digest schema mismatch")
    if payload.get("status") not in {"missing", "blocked", "ready"}:
        raise ValueError("controlled capture handoff environment status unsupported")
    if not isinstance(payload.get("controlled_capture_environment_ready"), bool):
        raise ValueError("controlled capture environment ready flag must be bool")
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("controlled capture environment hard_blockers must be list")


def _validate_bundle_digest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA:
        raise ValueError("controlled capture handoff bundle digest schema mismatch")
    if payload.get("status") not in {"missing", "blocked", "ready"}:
        raise ValueError("controlled capture handoff bundle status unsupported")
    if not isinstance(payload.get("identity_bundle_handoff_ready"), bool):
        raise ValueError("controlled capture bundle ready flag must be bool")
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("controlled capture bundle hard_blockers must be list")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"controlled capture handoff requires {key}")
    return value
