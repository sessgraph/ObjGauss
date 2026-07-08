from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture_bundle_readiness import (
    objectstate_controlled_capture_bundle_readiness,
    validate_objectstate_controlled_capture_bundle_readiness_summary,
)
from objgauss.core.objectstate_controlled_reality_bundle_handoff import (
    validate_objectstate_controlled_reality_bundle_handoff_summary,
)
from objgauss.core.objectstate_controlled_capture_template import (
    write_objectstate_controlled_capture_bundle_template,
    validate_objectstate_controlled_capture_bundle_template_summary,
)
from objgauss.core.objectstate_public_dataset_candidates import (
    default_objectstate_public_dataset_candidates,
    objectstate_public_interaction_route_audit,
    validate_objectstate_public_dataset_candidate,
    validate_objectstate_public_interaction_route_audit,
)
from objgauss.core.objectstate_public_interaction_reality_rows import (
    validate_objectstate_public_interaction_reality_rows_summary,
)
from objgauss.core.objectstate_reality_row_ledger import (
    validate_objectstate_reality_row_ledger_summary,
)

OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA = (
    "objgauss-objectstate-public-interaction-workspace-v1"
)
OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_PROGRESS_SCHEMA = (
    "objgauss-objectstate-public-interaction-workspace-progress-v1"
)
_TODO_SEQUENCE_ID = "TODO_PUBLIC_INTERACTION_SEQUENCE_ID"


def write_objectstate_public_interaction_workspace(
    root: str | Path,
    *,
    sample_id: str,
    candidate_id: str = "hot3d-clips",
    source_sequence_id: str = "TODO_PUBLIC_INTERACTION_SEQUENCE_ID",
    object_category: str = "public_interaction_objects",
    scenario: str = "public_interaction_action_like_clip",
    fps: float = 30.0,
    license_text: str | None = None,
    objects: Sequence[Mapping[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    workspace_root = Path(root)
    candidate = _public_interaction_candidate(candidate_id)
    effective_license = license_text or candidate.source_license
    route_readme = workspace_root / "PUBLIC_INTERACTION_ROUTE.md"
    if route_readme.exists() and not force:
        raise FileExistsError(
            "public interaction workspace refuses to overwrite existing file: "
            f"{route_readme}"
        )
    controlled_template = write_objectstate_controlled_capture_bundle_template(
        workspace_root,
        sample_id=sample_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        capture_device=f"{candidate.candidate_id}-public-interaction",
        license_text=effective_license,
        objects=objects,
        force=force,
    )
    next_commands = _next_commands(workspace_root)
    route_readme.write_text(
        _route_readme_text(
            candidate_id=candidate.candidate_id,
            sample_id=sample_id,
            source_sequence_id=source_sequence_id,
            next_commands=next_commands,
        ),
        encoding="utf-8",
    )
    files = dict(controlled_template["files"])
    files["public_interaction_readme"] = str(route_readme)
    payload = {
        "schema": OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA,
        "kind": "objectstate_public_interaction_workspace",
        "status": "objectstate_public_interaction_workspace_ready",
        "root": str(workspace_root),
        "candidate": candidate.as_dict(),
        "source_sequence_id": str(source_sequence_id),
        "controlled_capture_template": controlled_template,
        "sample": {
            "sample_id": sample_id,
            "source_kind": controlled_template["sample"]["source_kind"],
            "object_category": object_category,
            "scenario": scenario,
            "fps": float(fps),
            "license": effective_license,
        },
        "files": files,
        "directories": dict(controlled_template["directories"]),
        "next_commands": next_commands,
        "authoring_policy": {
            "uses_controlled_capture_contract_for_local_authoring": True,
            "final_rows_must_be_converted_to_public_replay": True,
            "public_dataset_candidate_required": True,
            "source_sequence_id_required_before_review": True,
            "route_audit_required_before_handoff": True,
            "full_handoff_required_before_reality_rows": True,
        },
        "claim_policy": {
            "workspace_only": True,
            "requires_external_public_dataset_clip": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_create_frame_rows": True,
            "does_not_create_annotation_rows": True,
            "does_not_create_action_rows": True,
            "does_not_create_candidates": True,
            "does_not_run_handoff": True,
            "does_not_create_reality_rows": True,
            "does_not_claim_intervention_pass": True,
            "does_not_claim_counterfactual_proof": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_frame_rows": False,
            "creates_annotation_rows": False,
            "creates_action_rows": False,
            "creates_prediction_candidates": False,
            "creates_intervention_candidates": False,
            "runs_handoff": False,
            "runs_eval": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_public_interaction_workspace_summary(payload)


def objectstate_public_interaction_workspace_progress(
    root: str | Path,
    *,
    workspace_summary: str | Path | None = None,
    candidate_id: str | None = None,
    source_sequence_id: str | None = None,
) -> dict[str, Any]:
    workspace_root = Path(root)
    paths = _progress_paths(workspace_root, workspace_summary=workspace_summary)
    workspace_record = _json_record(
        paths["workspace_summary"],
        validator=validate_objectstate_public_interaction_workspace_summary,
    )
    workspace_payload = workspace_record.get("payload")
    effective_candidate_id = (
        str(candidate_id).strip()
        if candidate_id
        else _workspace_candidate_id(workspace_payload)
    )
    effective_source_sequence_id = (
        str(source_sequence_id).strip()
        if source_sequence_id
        else _workspace_source_sequence_id(workspace_payload)
    )
    bundle_readiness = _bundle_progress_record(workspace_root)
    route_record = _route_progress_record(
        workspace_root,
        candidate_id=effective_candidate_id,
        paths=paths,
    )
    handoff_record = _json_record(
        paths["handoff_summary"],
        validator=validate_objectstate_controlled_reality_bundle_handoff_summary,
    )
    rows_record = _json_record(
        paths["public_replay_rows"],
        validator=validate_objectstate_public_interaction_reality_rows_summary,
    )
    ledger_record = _json_record(
        paths["ledger"],
        validator=validate_objectstate_reality_row_ledger_summary,
    )
    readiness = _progress_readiness(
        workspace_record=workspace_record,
        source_sequence_id=effective_source_sequence_id,
        bundle_readiness=bundle_readiness,
        route_record=route_record,
        handoff_record=handoff_record,
        rows_record=rows_record,
        ledger_record=ledger_record,
    )
    payload = {
        "schema": OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_PROGRESS_SCHEMA,
        "kind": "objectstate_public_interaction_workspace_progress",
        "status": _progress_status(readiness),
        "root": str(workspace_root),
        "candidate_id": effective_candidate_id,
        "source_sequence_id": effective_source_sequence_id,
        "paths": {key: str(path) for key, path in paths.items()},
        "workspace_summary": _record_public_info(workspace_record),
        "controlled_bundle_readiness": bundle_readiness,
        "route_audit": route_record,
        "handoff_summary": _record_public_info(handoff_record),
        "public_replay_rows": _record_public_info(rows_record),
        "ledger": _record_public_info(ledger_record),
        "readiness": readiness,
        "hard_blockers": _progress_blockers(readiness),
        "next_actions": _progress_next_actions(readiness, paths),
        "claim_policy": {
            "progress_audit_only": True,
            "checks_existing_workspace_files": True,
            "final_rows_must_be_public_replay": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_create_frame_rows": True,
            "does_not_create_annotation_rows": True,
            "does_not_create_action_rows": True,
            "does_not_create_candidates": True,
            "does_not_run_handoff": True,
            "does_not_run_eval": True,
            "does_not_create_reality_rows": True,
            "does_not_claim_intervention_pass": True,
            "does_not_claim_counterfactual_proof": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_frame_rows": False,
            "creates_annotation_rows": False,
            "creates_action_rows": False,
            "creates_prediction_candidates": False,
            "creates_intervention_candidates": False,
            "runs_handoff": False,
            "runs_eval": False,
            "reconstructs_gaussians": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_public_interaction_workspace_progress_summary(payload)


def validate_objectstate_public_interaction_workspace_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("public interaction workspace summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA:
        raise ValueError(
            "unsupported public interaction workspace schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_public_interaction_workspace":
        raise ValueError("public interaction workspace kind is unsupported")
    if payload.get("status") != "objectstate_public_interaction_workspace_ready":
        raise ValueError("public interaction workspace status is unsupported")
    if not isinstance(payload.get("root"), str) or not payload["root"]:
        raise ValueError("public interaction workspace requires root")
    _validate_candidate_payload(payload.get("candidate"))
    if not isinstance(payload.get("source_sequence_id"), str) or not payload["source_sequence_id"]:
        raise ValueError("public interaction workspace requires source_sequence_id")
    controlled_template = validate_objectstate_controlled_capture_bundle_template_summary(
        payload.get("controlled_capture_template")
    )
    sample = payload.get("sample")
    if not isinstance(sample, Mapping):
        raise ValueError("public interaction workspace requires sample")
    if sample.get("sample_id") != controlled_template["sample"]["sample_id"]:
        raise ValueError("public interaction workspace sample_id mismatch")
    if sample.get("source_kind") != "controlled_real":
        raise ValueError(
            "public interaction workspace must keep local authoring sample "
            "compatible with controlled_real"
        )
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("public interaction workspace requires files")
    for key in (
        "sample_json",
        "objects_csv",
        "frames_csv",
        "annotations_csv",
        "actions_csv",
        "readme",
        "public_interaction_readme",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"public interaction workspace missing file {key}")
    next_commands = payload.get("next_commands")
    if not isinstance(next_commands, Mapping):
        raise ValueError("public interaction workspace requires next_commands")
    for key in (
        "route_audit",
        "import_bundle",
        "accept_bundle",
        "init_candidate_templates",
        "finalize_candidates",
        "full_handoff",
        "public_replay_rows",
        "ledger",
    ):
        if not isinstance(next_commands.get(key), str) or not next_commands[key]:
            raise ValueError(f"public interaction workspace missing command {key}")
    authoring_policy = payload.get("authoring_policy", {})
    if (
        not authoring_policy.get("uses_controlled_capture_contract_for_local_authoring")
        or not authoring_policy.get("final_rows_must_be_converted_to_public_replay")
        or not authoring_policy.get("public_dataset_candidate_required")
        or not authoring_policy.get("source_sequence_id_required_before_review")
        or not authoring_policy.get("route_audit_required_before_handoff")
        or not authoring_policy.get("full_handoff_required_before_reality_rows")
    ):
        raise ValueError("public interaction workspace must preserve authoring policy")
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("workspace_only")
        or not claim_policy.get("requires_external_public_dataset_clip")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_create_frame_rows")
        or not claim_policy.get("does_not_create_annotation_rows")
        or not claim_policy.get("does_not_create_action_rows")
        or not claim_policy.get("does_not_create_candidates")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_create_reality_rows")
        or not claim_policy.get("does_not_claim_intervention_pass")
        or not claim_policy.get("does_not_claim_counterfactual_proof")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("public interaction workspace must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "public interaction workspace cannot download, capture, create GT/rows/"
            "candidates, run handoff/eval, reconstruct, train, write public "
            "samples, replay, diffuse, or mutate viewer defaults"
        )
    return dict(payload)


def validate_objectstate_public_interaction_workspace_progress_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("public interaction workspace progress must be a mapping")
    if payload.get("schema") != OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_PROGRESS_SCHEMA:
        raise ValueError(
            "unsupported public interaction workspace progress schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_public_interaction_workspace_progress":
        raise ValueError("public interaction workspace progress kind is unsupported")
    readiness = payload.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("public interaction workspace progress requires readiness")
    for key in (
        "workspace_layout_ready",
        "workspace_summary_valid",
        "source_sequence_bound",
        "controlled_bundle_import_ready",
        "controlled_bundle_files_ready",
        "controlled_bundle_intervention_ready",
        "candidate_artifact_present",
        "prediction_candidates_valid",
        "intervention_candidates_valid",
        "route_handoff_ready",
        "handoff_summary_valid",
        "public_replay_rows_valid",
        "ledger_valid",
        "evidence_chain_reviewable",
    ):
        if not isinstance(readiness.get(key), bool):
            raise ValueError(f"public interaction progress missing bool {key}")
    expected_status = _progress_status(readiness)
    if payload.get("status") != expected_status:
        raise ValueError("public interaction progress status must match readiness")
    if not isinstance(payload.get("root"), str) or not payload["root"]:
        raise ValueError("public interaction progress requires root")
    if not isinstance(payload.get("candidate_id"), str) or not payload["candidate_id"]:
        raise ValueError("public interaction progress requires candidate_id")
    if not isinstance(payload.get("source_sequence_id"), str):
        raise ValueError("public interaction progress requires source_sequence_id")
    paths = payload.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("public interaction progress requires paths")
    for key in (
        "workspace_summary",
        "capture_manifest",
        "candidate_artifact",
        "prediction_candidates",
        "intervention_candidates",
        "handoff_summary",
        "public_replay_rows",
        "ledger",
    ):
        if not isinstance(paths.get(key), str) or not paths[key]:
            raise ValueError(f"public interaction progress missing path {key}")
    validate_objectstate_controlled_capture_bundle_readiness_summary(
        payload["controlled_bundle_readiness"]
    )
    validate_objectstate_public_interaction_route_audit(payload["route_audit"])
    for key in (
        "workspace_summary",
        "handoff_summary",
        "public_replay_rows",
        "ledger",
    ):
        _validate_progress_record(payload.get(key), key)
    if not isinstance(payload.get("hard_blockers"), list):
        raise ValueError("public interaction progress hard_blockers must be a list")
    next_actions = payload.get("next_actions")
    if not isinstance(next_actions, list) or not next_actions:
        raise ValueError("public interaction progress requires next_actions")
    claim_policy = payload.get("claim_policy", {})
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("progress_audit_only")
        or not claim_policy.get("checks_existing_workspace_files")
        or not claim_policy.get("final_rows_must_be_public_replay")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_create_frame_rows")
        or not claim_policy.get("does_not_create_annotation_rows")
        or not claim_policy.get("does_not_create_action_rows")
        or not claim_policy.get("does_not_create_candidates")
        or not claim_policy.get("does_not_run_handoff")
        or not claim_policy.get("does_not_run_eval")
        or not claim_policy.get("does_not_create_reality_rows")
        or not claim_policy.get("does_not_claim_intervention_pass")
        or not claim_policy.get("does_not_claim_counterfactual_proof")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("public interaction progress must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "public interaction progress cannot download, capture, create GT/rows/"
            "candidates, run handoff/eval, reconstruct, train, write public "
            "samples, replay, diffuse, or mutate viewer defaults"
        )
    if readiness["evidence_chain_reviewable"] and payload["hard_blockers"]:
        raise ValueError("reviewable public interaction progress cannot have blockers")
    return dict(payload)


def _public_interaction_candidate(candidate_id: str):
    normalized = str(candidate_id).strip()
    for candidate in default_objectstate_public_dataset_candidates():
        checked = validate_objectstate_public_dataset_candidate(candidate)
        if checked.candidate_id == normalized:
            if checked.source_kind != "public_interaction_dataset":
                raise ValueError(
                    f"candidate is not a public interaction dataset: {candidate_id}"
                )
            if not checked.has_action_gt:
                raise ValueError(
                    f"candidate does not advertise action ground truth: {candidate_id}"
                )
            return checked
    raise ValueError(f"unknown public interaction candidate id: {candidate_id}")


def _next_commands(root: Path) -> dict[str, str]:
    reality_dir = root / "reality-candidates"
    handoff_dir = root / "reality-handoff"
    rows_json = root / "public-interaction-reality-rows.json"
    return {
        "route_audit": (
            "uv run objgauss object-state audit-public-interaction-route "
            f"{root} --summary-output {root / 'public-interaction-route-summary.json'} "
            f"--markdown-output {root / 'public-interaction-route.md'}"
        ),
        "import_bundle": (
            "uv run objgauss object-state import-controlled-capture-bundle "
            f"{root} --output {root / 'capture-manifest.json'} "
            f"--summary-output {root / 'bundle-import-summary.json'}"
        ),
        "accept_bundle": (
            "uv run objgauss object-state accept-controlled-capture-bundle "
            f"{root} --output {root / 'capture-manifest.json'} "
            f"--summary-output {root / 'bundle-acceptance-summary.json'} "
            f"--file-audit-output {root / 'bundle-file-audit.json'} "
            f"--missing-files-output {root / 'bundle-missing-files.md'} "
            "--require-gaussian-files"
        ),
        "init_candidate_templates": (
            "uv run objgauss object-state init-controlled-reality-candidates "
            f"{root} --output-dir {reality_dir} "
            f"--candidate-id <candidate-id> --candidate-source <candidate-source> "
            f"--artifact-ref {root / 'objectstates.json'}"
        ),
        "finalize_candidates": (
            "uv run objgauss object-state finalize-controlled-reality-candidates "
            f"{reality_dir / 'prediction-candidates.template.json'} "
            f"{reality_dir / 'intervention-candidates.template.json'} "
            f"--output-dir {reality_dir}"
        ),
        "full_handoff": (
            "uv run objgauss object-state controlled-reality-bundle-handoff "
            f"{root} {root / 'objectstates.json'} "
            f"{reality_dir / 'prediction-candidates.json'} "
            f"{reality_dir / 'intervention-candidates.json'} "
            f"--output-dir {handoff_dir} --require-pass"
        ),
        "public_replay_rows": (
            "uv run objgauss object-state audit-public-interaction-reality-rows "
            f"{handoff_dir / 'reality-bundle-handoff-summary.json'} "
            f"--summary-output {rows_json} "
            f"--blocked-rows-output {root / 'public-interaction-blocked-rows.md'}"
        ),
        "ledger": (
            "uv run objgauss object-state audit-reality-row-ledger "
            f"{rows_json} --summary-output {root / 'public-interaction-ledger.json'}"
        ),
    }


def _route_readme_text(
    *,
    candidate_id: str,
    sample_id: str,
    source_sequence_id: str,
    next_commands: Mapping[str, str],
) -> str:
    lines = [
        "# Public Interaction ObjectState Route Workspace",
        "",
        f"- candidate: `{candidate_id}`",
        f"- sample_id: `{sample_id}`",
        f"- source_sequence_id: `{source_sequence_id}`",
        "",
        "Fill this workspace with a license-reviewed public interaction clip.",
        "Do not commit raw public dataset frames, Gaussian evidence or model outputs.",
        "",
        "Required authoring facts:",
        "",
        "- frame rows with timestamp, RGB ref, Gaussian ref and optional action_id",
        "- object annotations with physical object ids and 6DoF poses",
        "- action rows with object id, time interval and non-zero action vector",
        "- ObjectState candidate artifact and prediction/intervention candidates",
        "",
        "Command chain:",
        "",
    ]
    for name, command in next_commands.items():
        lines.extend((f"## {name}", "", "```bash", command, "```", ""))
    lines.extend(
        (
            "Claim boundary:",
            "",
            "- This workspace is not evidence.",
            "- Final rows must be converted to `source_kind=public_replay`.",
            "- Observed public interactions are not randomized counterfactual proof.",
            "",
        )
    )
    return "\n".join(lines)


def _progress_paths(
    root: Path,
    *,
    workspace_summary: str | Path | None,
) -> dict[str, Path]:
    reality_dir = root / "reality-candidates"
    return {
        "workspace_summary": (
            root / "public-interaction-workspace.json"
            if workspace_summary is None
            else Path(workspace_summary)
        ),
        "capture_manifest": root / "capture-manifest.json",
        "candidate_artifact": root / "objectstates.json",
        "prediction_candidates": reality_dir / "prediction-candidates.json",
        "intervention_candidates": reality_dir / "intervention-candidates.json",
        "handoff_summary": (
            root
            / "reality-handoff"
            / "reality-bundle-handoff-summary.json"
        ),
        "public_replay_rows": root / "public-interaction-reality-rows.json",
        "ledger": root / "public-interaction-ledger.json",
    }


def _json_record(path: Path, *, validator: Any) -> dict[str, Any]:
    payload = None
    error = None
    schema = None
    present = path.is_file()
    valid = False
    if present:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, Mapping):
                schema = payload.get("schema")
            payload = validator(payload)
            valid = True
        except Exception as exc:  # noqa: BLE001 - progress audits report errors.
            error = str(exc)
            payload = None
    return {
        "path": str(path),
        "present": bool(present),
        "valid": bool(valid),
        "schema": schema,
        "error": error,
        "payload": payload,
    }


def _record_public_info(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": record.get("path"),
        "present": bool(record.get("present")),
        "valid": bool(record.get("valid")),
        "schema": record.get("schema"),
        "error": record.get("error"),
    }


def _bundle_progress_record(root: Path) -> dict[str, Any]:
    return objectstate_controlled_capture_bundle_readiness(
        root,
        require_prediction_ready=True,
        require_intervention_ready=True,
        candidate_artifact=root / "objectstates.json",
        require_candidate_artifact=False,
    )


def _route_progress_record(
    root: Path,
    *,
    candidate_id: str,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    try:
        return objectstate_public_interaction_route_audit(
            candidate_id=candidate_id,
            dataset_root=root,
            capture_manifest=paths["capture_manifest"],
            candidate_artifact=paths["candidate_artifact"],
            prediction_candidates=paths["prediction_candidates"],
            intervention_candidates=paths["intervention_candidates"],
        )
    except Exception as exc:  # noqa: BLE001 - progress audits report errors.
        fallback = objectstate_public_interaction_route_audit(
            candidate_id="hot3d-clips",
            dataset_root=None,
        )
        fallback["hard_blockers"] = [f"route audit failed: {exc}"]
        fallback["readiness"]["controlled_reality_handoff_ready"] = False
        return validate_objectstate_public_interaction_route_audit(fallback)


def _workspace_candidate_id(payload: Any) -> str:
    if isinstance(payload, Mapping):
        candidate = payload.get("candidate")
        if isinstance(candidate, Mapping) and isinstance(candidate.get("candidate_id"), str):
            return candidate["candidate_id"]
    return "hot3d-clips"


def _workspace_source_sequence_id(payload: Any) -> str:
    if isinstance(payload, Mapping) and isinstance(payload.get("source_sequence_id"), str):
        return payload["source_sequence_id"]
    return _TODO_SEQUENCE_ID


def _progress_readiness(
    *,
    workspace_record: Mapping[str, Any],
    source_sequence_id: str,
    bundle_readiness: Mapping[str, Any],
    route_record: Mapping[str, Any],
    handoff_record: Mapping[str, Any],
    rows_record: Mapping[str, Any],
    ledger_record: Mapping[str, Any],
) -> dict[str, bool]:
    bundle_gates = bundle_readiness["readiness"]
    route_gates = route_record["readiness"]
    source_sequence_bound = bool(
        source_sequence_id and source_sequence_id != _TODO_SEQUENCE_ID
    )
    readiness = {
        "workspace_layout_ready": bool(bundle_gates["layout_ready"]),
        "workspace_summary_valid": bool(workspace_record.get("valid")),
        "source_sequence_bound": source_sequence_bound,
        "controlled_bundle_import_ready": bool(bundle_gates["capture_import_ready"]),
        "controlled_bundle_files_ready": bool(bundle_gates["capture_files_ready"]),
        "controlled_bundle_intervention_ready": bool(
            bundle_gates["intervention_stage_ready"]
        ),
        "candidate_artifact_present": bool(route_gates["candidate_artifact_present"]),
        "prediction_candidates_valid": bool(route_gates["prediction_candidates_valid"]),
        "intervention_candidates_valid": bool(
            route_gates["intervention_candidates_valid"]
        ),
        "route_handoff_ready": bool(route_gates["controlled_reality_handoff_ready"]),
        "handoff_summary_valid": bool(handoff_record.get("valid")),
        "public_replay_rows_valid": bool(rows_record.get("valid")),
        "ledger_valid": bool(ledger_record.get("valid")),
        "evidence_chain_reviewable": False,
    }
    readiness["evidence_chain_reviewable"] = all(
        (
            readiness["source_sequence_bound"],
            readiness["route_handoff_ready"],
            readiness["handoff_summary_valid"],
            readiness["public_replay_rows_valid"],
            readiness["ledger_valid"],
        )
    )
    return readiness


def _progress_status(readiness: Mapping[str, bool]) -> str:
    if readiness["evidence_chain_reviewable"]:
        return "objectstate_public_interaction_workspace_progress_reviewable"
    if readiness["public_replay_rows_valid"]:
        return "objectstate_public_interaction_workspace_progress_rows_ready"
    if readiness["route_handoff_ready"]:
        return "objectstate_public_interaction_workspace_progress_route_ready"
    return "objectstate_public_interaction_workspace_progress_blocked"


def _progress_blockers(readiness: Mapping[str, bool]) -> list[str]:
    blockers = []
    checks = (
        ("workspace_layout_ready", "workspace skeleton layout is incomplete"),
        ("source_sequence_bound", "source_sequence_id is missing or still TODO"),
        (
            "controlled_bundle_import_ready",
            "controlled capture CSV rows are not import-ready",
        ),
        (
            "controlled_bundle_files_ready",
            "referenced RGB / Gaussian files are missing or invalid",
        ),
        (
            "controlled_bundle_intervention_ready",
            "pose/action/timestamp GT is not intervention-ready",
        ),
        ("candidate_artifact_present", "ObjectState candidate artifact is missing"),
        ("prediction_candidates_valid", "prediction candidates JSON is missing or invalid"),
        (
            "intervention_candidates_valid",
            "intervention candidates JSON is missing or invalid",
        ),
        ("route_handoff_ready", "public interaction route audit is not handoff-ready"),
        ("handoff_summary_valid", "full controlled reality handoff summary is missing"),
        ("public_replay_rows_valid", "public_replay reality rows summary is missing"),
        ("ledger_valid", "public interaction ledger summary is missing"),
    )
    for key, message in checks:
        if not readiness[key]:
            blockers.append(message)
    return blockers


def _progress_next_actions(
    readiness: Mapping[str, bool],
    paths: Mapping[str, Path],
) -> list[str]:
    root = paths["capture_manifest"].parent
    if not readiness["workspace_layout_ready"]:
        return [
            "run init-public-interaction-route-workspace before filling public "
            "interaction evidence"
        ]
    if not readiness["source_sequence_bound"]:
        return ["replace TODO source_sequence_id with the real public dataset clip id"]
    if not readiness["controlled_bundle_import_ready"]:
        return [
            "fill objects.csv, frames.csv, annotations.csv and actions.csv with "
            "timestamped identity, 6DoF pose and action rows"
        ]
    if not readiness["controlled_bundle_files_ready"]:
        return [
            "place referenced RGB frames and per-frame Gaussian files under the "
            "workspace and rerun accept-controlled-capture-bundle"
        ]
    if not readiness["route_handoff_ready"]:
        return [
            "write objectstates.json plus finalized prediction/intervention "
            "candidate JSON files, then rerun audit-public-interaction-route"
        ]
    if not readiness["handoff_summary_valid"]:
        return [
            "run controlled-reality-bundle-handoff against the public interaction "
            "workspace"
        ]
    if not readiness["public_replay_rows_valid"]:
        return [
            "run audit-public-interaction-reality-rows to convert the handoff into "
            "source_kind=public_replay rows"
        ]
    if not readiness["ledger_valid"]:
        return [
            "run audit-reality-row-ledger "
            f"{paths['public_replay_rows']} --summary-output {root / 'public-interaction-ledger.json'}"
        ]
    return [
        "review the ledger gate and state-variable matrix; do not claim world model "
        "unless identity, prediction and intervention evidence all pass"
    ]


def _validate_progress_record(payload: Any, key: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"public interaction progress requires {key}")
    for field in ("path", "present", "valid"):
        if field not in payload:
            raise ValueError(f"public interaction progress {key} missing {field}")
    if not isinstance(payload["path"], str) or not payload["path"]:
        raise ValueError(f"public interaction progress {key}.path is required")
    if not isinstance(payload["present"], bool) or not isinstance(payload["valid"], bool):
        raise ValueError(f"public interaction progress {key} present/valid must be bool")


def _validate_candidate_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("public interaction workspace requires candidate payload")
    if payload.get("source_kind") != "public_interaction_dataset":
        raise ValueError("candidate payload must be public_interaction_dataset")
    ground_truth = payload.get("ground_truth")
    if not isinstance(ground_truth, Mapping) or not ground_truth.get("action"):
        raise ValueError("candidate payload must advertise action ground truth")
