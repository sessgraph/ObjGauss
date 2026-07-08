from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from objgauss.core.objectstate_bop_local_row_batch_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
    read_objectstate_bop_local_row_batch_spec,
    validate_objectstate_bop_local_row_batch_spec,
)
from objgauss.core.objectstate_bop_phase1_local_row_readiness import (
    OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA,
    objectstate_bop_phase1_local_row_readiness,
    validate_objectstate_bop_phase1_local_row_readiness_summary,
)

OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA = (
    "objgauss-objectstate-bop-local-row-batch-readiness-v1"
)


def objectstate_bop_local_row_batch_readiness(
    batch_spec: Mapping[str, Any] | str | Path,
    *,
    output_root: str | Path | None = None,
    min_reviewable_samples: int = 3,
    min_scene_or_category_coverage: int = 3,
) -> dict[str, Any]:
    spec_path: Path | None = None
    if isinstance(batch_spec, (str, Path)):
        spec_path = Path(batch_spec)
        spec = read_objectstate_bop_local_row_batch_spec(spec_path)
    else:
        spec = validate_objectstate_bop_local_row_batch_spec(batch_spec)
    spec_root = spec_path.parent if spec_path is not None else Path.cwd()
    batch = spec.get("batch", {})
    batch_id = batch.get("batch_id", "bop-local-row-batch")
    out = _resolve_output_root(
        explicit_output_root=output_root,
        spec_output_root=batch.get("output_root"),
        spec_root=spec_root,
    )
    thresholds = {
        "min_reviewable_samples": _positive_int(
            min_reviewable_samples,
            "min_reviewable_samples",
        ),
        "min_scene_or_category_coverage": _positive_int(
            min_scene_or_category_coverage,
            "min_scene_or_category_coverage",
        ),
    }

    defaults = spec.get("defaults", {})
    sample_summaries: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    for index, sample in enumerate(spec["samples"]):
        sample_id = sample["sample_id"]
        sample_output_root = _resolve_sample_output_root(
            sample.get("output_root"),
            batch_output_root=out,
            sample_id=sample_id,
        )
        merged = _merged_options(defaults, sample)
        summary = objectstate_bop_phase1_local_row_readiness(
            _resolve_input_path(sample["scene_root"], spec_root),
            output_root=sample_output_root,
            sample_id=sample_id,
            candidate_artifact=_resolve_input_path(
                sample["candidate_artifact"],
                spec_root,
            ),
            identity_dir=merged.get("identity_dir", "identity-handoff"),
            dataset_id=merged.get("dataset_id", "bop-ycbv"),
            object_category=merged.get("object_category", "bop_objects"),
            scenario=merged.get("scenario", "bop_pose_sequence"),
            fps=float(merged.get("fps", 30.0)),
            license_text=merged.get(
                "license_text",
                "BOP dataset terms; verify source dataset license before redistribution",
            ),
            rgb_dir=merged.get("rgb_dir", "rgb"),
            depth_dir=merged.get("depth_dir", "depth"),
            gaussian_dir=merged.get("gaussian_dir", "gaussians"),
            condition_sidecar=_optional_input_path(
                merged.get("condition_sidecar"),
                spec_root,
            ),
            max_frames=merged.get("max_frames"),
            frame_step=int(merged.get("frame_step", 1)),
            check_artifact_refs=bool(merged.get("check_artifact_refs", False)),
            min_rgb_bytes=int(merged.get("min_rgb_bytes", 1)),
            min_gaussian_bytes=int(merged.get("min_gaussian_bytes", 1)),
            require_frame_formats=bool(merged.get("require_frame_formats", True)),
            hash_files=bool(merged.get("hash_files", False)),
            min_identity_scenario_frames=int(
                merged.get("min_identity_scenario_frames", 3)
            ),
            min_occlusion_fraction=float(merged.get("min_occlusion_fraction", 0.5)),
            min_view_conditions=int(merged.get("min_view_conditions", 2)),
            min_lighting_conditions=int(merged.get("min_lighting_conditions", 2)),
            min_camera_motion_m=float(merged.get("min_camera_motion_m", 0.01)),
        )
        sample_summaries.append(summary)
        sample_records.append(_sample_record(index, sample, merged, sample_output_root, summary))

    coverage = _coverage(sample_records)
    sample_counts = _sample_counts(sample_records)
    readiness_gates = _readiness_gates(
        sample_records,
        sample_counts=sample_counts,
        coverage=coverage,
        thresholds=thresholds,
    )
    status = _status(readiness_gates)
    payload = {
        "schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA,
        "kind": "objectstate_bop_local_row_batch_readiness",
        "status": status,
        "batch_spec_schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
        "local_row_readiness_schema": OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA,
        "batch": {
            "batch_id": batch_id,
            "spec_path": str(spec_path) if spec_path is not None else None,
            "output_root": str(out),
        },
        "thresholds": thresholds,
        "sample_summary": sample_counts,
        "coverage": coverage,
        "readiness_gates": readiness_gates,
        "sample_records": sample_records,
        "local_row_readiness_summaries": sample_summaries,
        "sample_table_markdown": _sample_table_markdown(sample_records),
        "hard_blockers": _hard_blockers(sample_records, readiness_gates),
        "next_actions": _next_actions(
            spec_path=spec_path,
            output_root=out,
            readiness_gates=readiness_gates,
            sample_records=sample_records,
            thresholds=thresholds,
            sample_counts=sample_counts,
            coverage=coverage,
        ),
        "issues": _issues(sample_records, readiness_gates),
        "claim_policy": {
            "read_only_batch_audit": True,
            "uses_explicit_batch_spec": True,
            "runs_local_row_readiness_audits": True,
            "checks_identity_prediction_handoff_preconditions": True,
            "checks_scene_or_category_coverage": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_write_rgbd_gaussian_export": True,
            "does_not_run_local_row_handoff": True,
            "does_not_run_identity_handoff": True,
            "does_not_run_prediction_handoff": True,
            "does_not_run_identity_eval": True,
            "does_not_run_prediction_eval": True,
            "does_not_train_model": True,
            "does_not_claim_intervention_gate": True,
            "does_not_claim_metric_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "downloads_dataset": False,
            "captures_video": False,
            "creates_ground_truth": False,
            "reconstructs_gaussians": False,
            "runs_tracking_model": False,
            "runs_local_row_handoff": False,
            "runs_identity_eval": False,
            "runs_prediction_eval": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_bop_local_row_batch_readiness_summary(payload)


def validate_objectstate_bop_local_row_batch_readiness_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP local row batch readiness summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA:
        raise ValueError(
            "unsupported BOP local row batch readiness schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_local_row_batch_readiness":
        raise ValueError("BOP local row batch readiness kind is unsupported")
    if payload.get("batch_spec_schema") != OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA:
        raise ValueError("BOP local row batch readiness spec schema mismatch")
    if (
        payload.get("local_row_readiness_schema")
        != OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA
    ):
        raise ValueError("BOP local row batch readiness local schema mismatch")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("BOP local row batch readiness requires thresholds")
    _positive_int(thresholds.get("min_reviewable_samples"), "min_reviewable_samples")
    _positive_int(
        thresholds.get("min_scene_or_category_coverage"),
        "min_scene_or_category_coverage",
    )
    sample_records = payload.get("sample_records")
    if not isinstance(sample_records, list):
        raise ValueError("BOP local row batch readiness requires sample records")
    sample_summaries = payload.get("local_row_readiness_summaries")
    if not isinstance(sample_summaries, list):
        raise ValueError("BOP local row batch readiness requires local summaries")
    if len(sample_records) != len(sample_summaries):
        raise ValueError("BOP local row batch readiness sample count mismatch")
    checked_summaries = [
        validate_objectstate_bop_phase1_local_row_readiness_summary(summary)
        for summary in sample_summaries
    ]
    for record, summary in zip(sample_records, checked_summaries):
        _validate_sample_record(record, summary)
    sample_counts = _sample_counts(sample_records)
    if payload.get("sample_summary") != sample_counts:
        raise ValueError("BOP local row batch readiness sample summary mismatch")
    coverage = _coverage(sample_records)
    if payload.get("coverage") != coverage:
        raise ValueError("BOP local row batch readiness coverage mismatch")
    readiness = payload.get("readiness_gates")
    if not isinstance(readiness, Mapping) or not readiness:
        raise ValueError("BOP local row batch readiness requires readiness gates")
    if any(not isinstance(value, bool) for value in readiness.values()):
        raise ValueError("BOP local row batch readiness gates must be bool")
    expected_readiness = _readiness_gates(
        sample_records,
        sample_counts=sample_counts,
        coverage=coverage,
        thresholds=dict(thresholds),
    )
    if dict(readiness) != expected_readiness:
        raise ValueError("BOP local row batch readiness gates mismatch")
    if payload.get("status") != _status(readiness):
        raise ValueError("BOP local row batch readiness status mismatch")
    for key in ("hard_blockers", "next_actions", "issues"):
        values = payload.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP local row batch readiness {key} must be strings")
    table = payload.get("sample_table_markdown")
    if not isinstance(table, str) or not table:
        raise ValueError("BOP local row batch readiness requires sample table")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("read_only_batch_audit")
        or not claim_policy.get("uses_explicit_batch_spec")
        or not claim_policy.get("runs_local_row_readiness_audits")
        or not claim_policy.get("checks_identity_prediction_handoff_preconditions")
        or not claim_policy.get("checks_scene_or_category_coverage")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_write_rgbd_gaussian_export")
        or not claim_policy.get("does_not_run_local_row_handoff")
        or not claim_policy.get("does_not_run_identity_handoff")
        or not claim_policy.get("does_not_run_prediction_handoff")
        or not claim_policy.get("does_not_run_identity_eval")
        or not claim_policy.get("does_not_run_prediction_eval")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP local row batch readiness must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP local row batch readiness cannot claim downloads, capture, GT, "
            "Gaussian reconstruction, handoff, eval, training, public samples, "
            "replay, diffusion, or viewer mutation"
        )
    return dict(payload)


def _sample_record(
    index: int,
    sample: Mapping[str, Any],
    merged: Mapping[str, Any],
    sample_output_root: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = summary["readiness"]
    ready_for_handoff = bool(
        readiness["identity_route_ready_for_handoff"]
        and readiness["prediction_route_ready_for_handoff"]
    )
    reviewable = bool(readiness["phase1_identity_prediction_reviewable"])
    return {
        "index": index,
        "sample_id": summary["sample_id"],
        "scene_root": summary["scene_root"],
        "output_root": str(sample_output_root),
        "candidate_artifact": summary["files"]["candidate_artifact"],
        "object_category": str(merged.get("object_category", "bop_objects")),
        "scenario": str(merged.get("scenario", "bop_pose_sequence")),
        "status": summary["status"],
        "blocking_stage": summary["blocking_stage"],
        "ready_for_local_row_handoff": ready_for_handoff,
        "identity_prediction_reviewable": reviewable,
        "ready_or_reviewable": bool(ready_for_handoff or reviewable),
        "readiness": dict(readiness),
        "hard_blockers": list(summary["hard_blockers"]),
        "next_actions": list(summary["next_actions"]),
        "issues": list(summary["issues"]),
        "files": dict(summary["files"]),
        "spec": {
            "scene_root": sample["scene_root"],
            "candidate_artifact": sample["candidate_artifact"],
            "output_root": sample.get("output_root"),
            "condition_sidecar": sample.get("condition_sidecar"),
        },
    }


def _validate_sample_record(
    record: Any,
    summary: Mapping[str, Any],
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("BOP local row batch readiness sample record must be mapping")
    for key in (
        "index",
        "sample_id",
        "scene_root",
        "output_root",
        "candidate_artifact",
        "object_category",
        "scenario",
        "status",
        "blocking_stage",
    ):
        if key == "index":
            if isinstance(record.get(key), bool) or not isinstance(record.get(key), int):
                raise ValueError("BOP local row batch readiness sample index invalid")
            continue
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"BOP local row batch readiness sample requires {key}")
    if record["sample_id"] != summary["sample_id"]:
        raise ValueError("BOP local row batch readiness sample id mismatch")
    if record["status"] != summary["status"]:
        raise ValueError("BOP local row batch readiness sample status mismatch")
    readiness = record.get("readiness")
    if readiness != summary["readiness"]:
        raise ValueError("BOP local row batch readiness sample gates mismatch")
    for key in (
        "ready_for_local_row_handoff",
        "identity_prediction_reviewable",
        "ready_or_reviewable",
    ):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"BOP local row batch readiness sample {key} invalid")
    expected_ready = bool(
        summary["readiness"]["identity_route_ready_for_handoff"]
        and summary["readiness"]["prediction_route_ready_for_handoff"]
    )
    expected_reviewable = bool(summary["readiness"]["phase1_identity_prediction_reviewable"])
    if record["ready_for_local_row_handoff"] != expected_ready:
        raise ValueError("BOP local row batch readiness handoff flag mismatch")
    if record["identity_prediction_reviewable"] != expected_reviewable:
        raise ValueError("BOP local row batch readiness reviewable flag mismatch")
    if record["ready_or_reviewable"] != bool(expected_ready or expected_reviewable):
        raise ValueError("BOP local row batch readiness ready/reviewable mismatch")
    for key in ("hard_blockers", "next_actions", "issues"):
        values = record.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"BOP local row batch readiness sample {key} invalid")
    if not isinstance(record.get("files"), Mapping):
        raise ValueError("BOP local row batch readiness sample files invalid")
    if not isinstance(record.get("spec"), Mapping):
        raise ValueError("BOP local row batch readiness sample spec invalid")


def _sample_counts(sample_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "samples": len(sample_records),
        "ready_for_local_row_handoff_samples": sum(
            1 for record in sample_records if record["ready_for_local_row_handoff"]
        ),
        "identity_prediction_reviewable_samples": sum(
            1 for record in sample_records if record["identity_prediction_reviewable"]
        ),
        "ready_or_reviewable_samples": sum(
            1 for record in sample_records if record["ready_or_reviewable"]
        ),
        "gaussian_evidence_ready_samples": sum(
            1
            for record in sample_records
            if record["readiness"]["phase1_gaussian_evidence_ready"]
        ),
        "candidate_artifact_binding_ready_samples": sum(
            1
            for record in sample_records
            if record["readiness"]["candidate_artifact_binding_ready"]
        ),
        "identity_scenario_metadata_ready_samples": sum(
            1
            for record in sample_records
            if record["readiness"]["identity_scenario_metadata_ready"]
        ),
    }


def _coverage(sample_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    scene_roots = {str(record["scene_root"]) for record in sample_records}
    categories = {str(record["object_category"]) for record in sample_records}
    scenarios = {str(record["scenario"]) for record in sample_records}
    max_coverage = max(
        (len(sample_records), len(scene_roots), len(categories), len(scenarios)),
        default=0,
    )
    return {
        "sample_count": len(sample_records),
        "scene_root_count": len(scene_roots),
        "object_category_count": len(categories),
        "scenario_count": len(scenarios),
        "max_scene_or_category_coverage": max_coverage,
    }


def _readiness_gates(
    sample_records: Sequence[Mapping[str, Any]],
    *,
    sample_counts: Mapping[str, int],
    coverage: Mapping[str, int],
    thresholds: Mapping[str, int],
) -> dict[str, bool]:
    return {
        "sample_count_nonzero": bool(sample_records),
        "all_sample_readiness_audits_valid": True,
        "all_samples_gaussian_evidence_ready": bool(sample_records)
        and sample_counts["gaussian_evidence_ready_samples"] == len(sample_records),
        "all_samples_candidate_artifact_binding_ready": bool(sample_records)
        and sample_counts["candidate_artifact_binding_ready_samples"] == len(sample_records),
        "all_samples_identity_scenario_metadata_ready": bool(sample_records)
        and sample_counts["identity_scenario_metadata_ready_samples"] == len(sample_records),
        "all_samples_ready_or_reviewable": bool(sample_records)
        and sample_counts["ready_or_reviewable_samples"] == len(sample_records),
        "min_reviewable_samples_met": (
            sample_counts["ready_or_reviewable_samples"]
            >= thresholds["min_reviewable_samples"]
        ),
        "scene_or_category_coverage_met": (
            coverage["max_scene_or_category_coverage"]
            >= thresholds["min_scene_or_category_coverage"]
        ),
        "does_not_claim_metric_pass": True,
        "does_not_claim_intervention_gate": True,
        "does_not_claim_world_model": True,
    }


def _status(readiness: Mapping[str, bool]) -> str:
    prefix = "objectstate_bop_local_row_batch_readiness"
    if all(readiness.values()):
        return f"{prefix}_ready"
    if readiness.get("sample_count_nonzero") and readiness.get("min_reviewable_samples_met"):
        return f"{prefix}_partial"
    return f"{prefix}_blocked"


def _hard_blockers(
    sample_records: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, bool],
) -> list[str]:
    blockers: list[str] = []
    if not readiness["sample_count_nonzero"]:
        blockers.append("batch spec did not include any samples")
    if not readiness["all_samples_gaussian_evidence_ready"]:
        blockers.append("one or more samples lack per-frame Gaussian evidence")
    if not readiness["all_samples_candidate_artifact_binding_ready"]:
        blockers.append("one or more samples lack a valid bound ObjectState candidate artifact")
    if not readiness["all_samples_identity_scenario_metadata_ready"]:
        blockers.append("one or more samples lack identity scenario metadata")
    if not readiness["all_samples_ready_or_reviewable"]:
        blockers.append("one or more samples are not ready for local-row handoff")
    if not readiness["min_reviewable_samples_met"]:
        blockers.append("batch does not meet the minimum ready/reviewable sample count")
    if not readiness["scene_or_category_coverage_met"]:
        blockers.append("batch does not meet scene/category/scenario coverage threshold")
    for record in sample_records:
        if record["ready_or_reviewable"]:
            continue
        for blocker in record["hard_blockers"][:3]:
            blockers.append(f"{record['sample_id']}: {blocker}")
    return _dedupe(blockers)


def _next_actions(
    *,
    spec_path: Path | None,
    output_root: Path,
    readiness_gates: Mapping[str, bool],
    sample_records: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, int],
    sample_counts: Mapping[str, int],
    coverage: Mapping[str, int],
) -> list[str]:
    if all(readiness_gates.values()):
        spec = str(spec_path) if spec_path is not None else "<batch-spec.json>"
        return [
            "run BOP local-row batch handoff using the same explicit batch spec",
            (
                "uv run objgauss object-state bop-local-row-batch-handoff "
                f"{spec} --output-root {output_root} --require-reviewable"
            ),
        ]
    actions: list[str] = []
    if sample_counts["ready_or_reviewable_samples"] < thresholds["min_reviewable_samples"]:
        actions.append(
            "add or repair samples until ready/reviewable sample count reaches "
            f"{thresholds['min_reviewable_samples']}"
        )
    if (
        coverage["max_scene_or_category_coverage"]
        < thresholds["min_scene_or_category_coverage"]
    ):
        actions.append(
            "expand scene/category/scenario coverage before treating this as cross-sample evidence"
        )
    for record in sample_records:
        if record["ready_or_reviewable"]:
            continue
        if record["next_actions"]:
            actions.append(f"{record['sample_id']}: {record['next_actions'][0]}")
        else:
            actions.append(f"{record['sample_id']}: rerun local row readiness after fixing blockers")
    return _dedupe(actions)


def _issues(
    sample_records: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, bool],
) -> list[str]:
    issues = [
        f"readiness gate failed: {gate}"
        for gate, passed in readiness.items()
        if not passed
    ]
    for record in sample_records:
        for issue in record["issues"][:5]:
            issues.append(f"{record['sample_id']}: {issue}")
    return _dedupe(issues)


def _sample_table_markdown(sample_records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| sample_id | status | blocking_stage | ready_or_reviewable | object_category | scenario |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in sample_records:
        lines.append(
            "| {sample_id} | {status} | {blocking_stage} | {ready} | {category} | {scenario} |".format(
                sample_id=record["sample_id"],
                status=record["status"],
                blocking_stage=record["blocking_stage"],
                ready=str(record["ready_or_reviewable"]).lower(),
                category=record["object_category"],
                scenario=record["scenario"],
            )
        )
    return "\n".join(lines) + "\n"


def _merged_options(
    defaults: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(
        {
            key: value
            for key, value in sample.items()
            if key not in {"sample_id", "scene_root", "candidate_artifact", "output_root"}
        }
    )
    if "license" in merged and "license_text" not in merged:
        merged["license_text"] = merged["license"]
    return merged


def _resolve_output_root(
    *,
    explicit_output_root: str | Path | None,
    spec_output_root: Any,
    spec_root: Path,
) -> Path:
    if explicit_output_root is not None:
        return Path(explicit_output_root)
    if isinstance(spec_output_root, str) and spec_output_root:
        return _resolve_input_path(spec_output_root, spec_root)
    return spec_root / "bop-local-row-batch"


def _resolve_sample_output_root(
    value: Any,
    *,
    batch_output_root: Path,
    sample_id: str,
) -> Path:
    if isinstance(value, str) and value:
        path = Path(value)
        return path if path.is_absolute() else batch_output_root / path
    return batch_output_root / "samples" / sample_id


def _resolve_input_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _optional_input_path(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("BOP local row batch readiness optional path must be a string")
    return _resolve_input_path(value, base)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"BOP local row batch readiness requires positive int {name}")
    return value


def _as_strings(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [str(value) for value in values]


def _dedupe(values: Any) -> list[str]:
    result = []
    seen: set[str] = set()
    for value in _as_strings(values):
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
