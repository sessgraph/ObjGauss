from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_controlled_capture_template import (
    write_objectstate_controlled_capture_bundle_template,
    validate_objectstate_controlled_capture_bundle_template_summary,
)
from objgauss.core.objectstate_public_dataset_candidates import (
    default_objectstate_public_dataset_candidates,
    validate_objectstate_public_dataset_candidate,
)

OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA = (
    "objgauss-objectstate-public-interaction-workspace-v1"
)


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


def _validate_candidate_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("public interaction workspace requires candidate payload")
    if payload.get("source_kind") != "public_interaction_dataset":
        raise ValueError("candidate payload must be public_interaction_dataset")
    ground_truth = payload.get("ground_truth")
    if not isinstance(ground_truth, Mapping) or not ground_truth.get("action"):
        raise ValueError("candidate payload must advertise action ground truth")
