from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from objgauss.datasets.objectstate_bop_local_row_batch_spec import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
    read_objectstate_bop_local_row_batch_spec,
    validate_objectstate_bop_local_row_batch_spec,
)

from objgauss.pipelines.objectstate_bop_cross_sample_ledger import (
    OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA,
    objectstate_bop_cross_sample_ledger,
    validate_objectstate_bop_cross_sample_ledger_summary,
)
from objgauss.pipelines.objectstate_bop_local_row_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
    objectstate_bop_local_row_handoff,
    validate_objectstate_bop_local_row_handoff_summary,
)
from objgauss.evaluation.objectstate_controlled_identity_eval import (
    ObjectStateControlledIdentityThresholds,
)
from objgauss.evaluation.objectstate_controlled_prediction_eval import (
    ObjectStateControlledPredictionThresholds,
)

OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA = (
    "objgauss-objectstate-bop-local-row-batch-handoff-v1"
)


def objectstate_bop_local_row_batch_handoff(
    batch_spec: Mapping[str, Any] | str | Path,
    *,
    output_root: str | Path | None = None,
    min_reviewable_samples: int = 3,
    min_scene_or_category_coverage: int = 3,
    force: bool = False,
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
    _ensure_batch_output_paths(out, force=force)

    defaults = spec.get("defaults", {})
    sample_summaries: list[dict[str, Any]] = []
    sample_summary_paths: list[Path] = []
    sample_records: list[dict[str, Any]] = []
    for index, sample in enumerate(spec["samples"]):
        sample_id = sample["sample_id"]
        sample_output_root = _resolve_sample_output_root(
            sample.get("output_root"),
            batch_output_root=out,
            sample_id=sample_id,
        )
        merged = _merged_options(defaults, sample)
        identity_thresholds = _identity_thresholds_from_options(merged)
        prediction_thresholds = _prediction_thresholds_from_options(merged)
        summary = objectstate_bop_local_row_handoff(
            _resolve_input_path(sample["scene_root"], spec_root),
            output_root=sample_output_root,
            sample_id=sample_id,
            candidate_artifact=_resolve_input_path(sample["candidate_artifact"], spec_root),
            dataset_id=merged.get("dataset_id", "bop-ycbv"),
            object_category=merged.get("object_category", "bop_objects"),
            scenario=merged.get("scenario", "bop_pose_sequence"),
            fps=float(merged.get("fps", 30.0)),
            license_text=merged.get(
                "license_text",
                "BOP dataset terms; verify source dataset license before redistribution",
            ),
            rgb_dir=merged.get("rgb_dir", "rgb"),
            gaussian_dir=merged.get("gaussian_dir", "gaussians"),
            condition_sidecar=_optional_input_path(
                merged.get("condition_sidecar"),
                spec_root,
            ),
            max_frames=merged.get("max_frames"),
            frame_step=int(merged.get("frame_step", 1)),
            identity_candidate_id=merged.get("identity_candidate_id"),
            identity_candidate_source=merged.get(
                "identity_candidate_source",
                "trainable_kernel_objectstate_raw_track_adapter",
            ),
            max_centroid_distance=merged.get("max_centroid_distance"),
            prediction_policy=merged.get("prediction_policy", "constant_velocity"),
            prediction_candidate_id=merged.get(
                "prediction_candidate_id",
                "bop-constant-velocity-baseline",
            ),
            prediction_candidate_source=merged.get(
                "prediction_candidate_source",
                "controlled-prediction-baseline",
            ),
            prediction_confidence=float(merged.get("prediction_confidence", 0.5)),
            check_artifact_refs=bool(merged.get("check_artifact_refs", False)),
            min_rgb_bytes=int(merged.get("min_rgb_bytes", 1)),
            min_gaussian_bytes=int(merged.get("min_gaussian_bytes", 1)),
            require_frame_formats=bool(merged.get("require_frame_formats", True)),
            hash_files=bool(merged.get("hash_files", False)),
            min_candidate_artifact_bytes=int(
                merged.get("min_candidate_artifact_bytes", 1)
            ),
            hash_candidate_artifact=bool(
                merged.get("hash_candidate_artifact", False)
            ),
            min_identity_scenario_frames=int(
                merged.get("min_identity_scenario_frames", 3)
            ),
            min_occlusion_fraction=float(merged.get("min_occlusion_fraction", 0.5)),
            min_view_conditions=int(merged.get("min_view_conditions", 2)),
            min_lighting_conditions=int(merged.get("min_lighting_conditions", 2)),
            min_camera_motion_m=float(merged.get("min_camera_motion_m", 0.01)),
            identity_thresholds=identity_thresholds,
            prediction_thresholds=prediction_thresholds,
            synthetic_smoke_passed=bool(merged.get("synthetic_smoke_passed", True)),
            min_real_or_public_rows=int(merged.get("min_real_or_public_rows", 1)),
            force=force,
        )
        sample_summary_path = sample_output_root / "bop-local-row-handoff-summary.json"
        _write_json(sample_summary_path, summary)
        sample_summaries.append(summary)
        sample_summary_paths.append(sample_summary_path)
        sample_records.append(_sample_record(index, sample, sample_output_root, summary))

    cross_sample_ledger = objectstate_bop_cross_sample_ledger(
        local_row_summaries=sample_summary_paths,
        min_reviewable_samples=min_reviewable_samples,
        min_scene_or_category_coverage=min_scene_or_category_coverage,
    )
    cross_sample_ledger_path = out / "bop-cross-sample-ledger.json"
    cross_sample_table_path = out / "bop-cross-sample-table.md"
    batch_summary_path = out / "bop-local-row-batch-handoff-summary.json"
    _write_json(cross_sample_ledger_path, cross_sample_ledger)
    cross_sample_table_path.write_text(
        cross_sample_ledger["sample_table_markdown"],
        encoding="utf-8",
    )

    reviewability = {
        "sample_count_nonzero": bool(sample_records),
        "all_local_row_handoffs_reviewable": all(
            record["status"] == "objectstate_bop_local_row_handoff_reviewable"
            for record in sample_records
        ),
        "all_local_row_summary_files_written": all(
            Path(record["files"]["local_row_summary"]).is_file()
            for record in sample_records
        ),
        "cross_sample_ledger_reviewable": (
            cross_sample_ledger["status"]
            == "objectstate_bop_cross_sample_ledger_reviewable"
        ),
    }
    pass_gates = {
        "all_identity_handoffs_pass": all(
            record["pass_gates"]["identity_handoff_pass"] for record in sample_records
        ),
        "all_prediction_evals_pass": all(
            record["pass_gates"]["prediction_eval_pass"] for record in sample_records
        ),
        "candidate_cross_sample_ready": bool(
            cross_sample_ledger["candidate_gate"]["candidate_cross_sample_ready"]
        ),
    }
    status = (
        "objectstate_bop_local_row_batch_handoff_reviewable"
        if all(reviewability.values())
        else "objectstate_bop_local_row_batch_handoff_incomplete"
    )
    payload = {
        "schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA,
        "kind": "objectstate_bop_local_row_batch_handoff",
        "status": status,
        "batch_spec_schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
        "local_row_handoff_schema": OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
        "cross_sample_ledger_schema": OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA,
        "batch": {
            "batch_id": batch_id,
            "spec_path": str(spec_path) if spec_path is not None else None,
            "output_root": str(out),
        },
        "thresholds": {
            "min_reviewable_samples": min_reviewable_samples,
            "min_scene_or_category_coverage": min_scene_or_category_coverage,
        },
        "reviewability_gates": reviewability,
        "pass_gates": pass_gates,
        "files": {
            "batch_summary": str(batch_summary_path),
            "cross_sample_ledger": str(cross_sample_ledger_path),
            "cross_sample_table": str(cross_sample_table_path),
        },
        "row_counts": {
            "samples": len(sample_records),
            "reviewable_samples": cross_sample_ledger["sample_summary"][
                "reviewable_sample_count"
            ],
            "identity_prediction_reviewable_samples": cross_sample_ledger[
                "sample_summary"
            ]["identity_prediction_reviewable_sample_count"],
            "identity_pass_samples": cross_sample_ledger["sample_summary"][
                "identity_pass_sample_count"
            ],
            "prediction_pass_samples": cross_sample_ledger["sample_summary"][
                "prediction_pass_sample_count"
            ],
        },
        "sample_records": sample_records,
        "local_row_summaries": sample_summaries,
        "cross_sample_ledger": cross_sample_ledger,
        "issues": _batch_issues(reviewability, cross_sample_ledger),
        "claim_policy": {
            "orchestrates_existing_bop_local_row_handoff": True,
            "writes_cross_sample_ledger": True,
            "uses_explicit_batch_spec": True,
            "requires_per_sample_candidate_artifacts": True,
            "requires_per_frame_gaussian_evidence_via_local_row_handoff": True,
            "reviewable_allows_metric_pass_or_fail": True,
            "does_not_download_dataset": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
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
            "runs_learned_prediction_model": False,
            "runs_intervention_model": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    checked = validate_objectstate_bop_local_row_batch_handoff_summary(payload)
    _write_json(batch_summary_path, checked)
    return checked


def validate_objectstate_bop_local_row_batch_handoff_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("BOP local row batch handoff summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA:
        raise ValueError(
            "unsupported BOP local row batch handoff schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_bop_local_row_batch_handoff":
        raise ValueError("BOP local row batch handoff kind is unsupported")
    reviewability = payload.get("reviewability_gates")
    if not isinstance(reviewability, Mapping) or not reviewability:
        raise ValueError("BOP local row batch handoff requires reviewability gates")
    if any(not isinstance(value, bool) for value in reviewability.values()):
        raise ValueError("BOP local row batch reviewability gates must be bool")
    expected_status = (
        "objectstate_bop_local_row_batch_handoff_reviewable"
        if all(reviewability.values())
        else "objectstate_bop_local_row_batch_handoff_incomplete"
    )
    if payload.get("status") != expected_status:
        raise ValueError("BOP local row batch handoff status mismatch")
    pass_gates = payload.get("pass_gates")
    if not isinstance(pass_gates, Mapping) or not pass_gates:
        raise ValueError("BOP local row batch handoff requires pass gates")
    if any(not isinstance(value, bool) for value in pass_gates.values()):
        raise ValueError("BOP local row batch pass gates must be bool")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("BOP local row batch handoff requires thresholds")
    _positive_int(thresholds.get("min_reviewable_samples"), "min_reviewable_samples")
    _positive_int(
        thresholds.get("min_scene_or_category_coverage"),
        "min_scene_or_category_coverage",
    )
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("BOP local row batch handoff requires files")
    for key in ("batch_summary", "cross_sample_ledger", "cross_sample_table"):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"BOP local row batch handoff missing file {key}")
    row_counts = payload.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("BOP local row batch handoff requires row_counts")
    for key in (
        "samples",
        "reviewable_samples",
        "identity_prediction_reviewable_samples",
        "identity_pass_samples",
        "prediction_pass_samples",
    ):
        _non_negative_int(row_counts.get(key), key)
    sample_records = payload.get("sample_records")
    if not isinstance(sample_records, list):
        raise ValueError("BOP local row batch handoff requires sample records")
    local_row_summaries = payload.get("local_row_summaries")
    if not isinstance(local_row_summaries, list):
        raise ValueError("BOP local row batch handoff requires local summaries")
    if len(local_row_summaries) != row_counts["samples"]:
        raise ValueError("BOP local row batch summary count mismatch")
    for summary in local_row_summaries:
        validate_objectstate_bop_local_row_handoff_summary(summary)
    validate_objectstate_bop_cross_sample_ledger_summary(
        payload.get("cross_sample_ledger")
    )
    if not isinstance(payload.get("issues"), list):
        raise ValueError("BOP local row batch handoff issues must be list")
    claim_policy = payload.get("claim_policy")
    if (
        not isinstance(claim_policy, Mapping)
        or not claim_policy.get("orchestrates_existing_bop_local_row_handoff")
        or not claim_policy.get("writes_cross_sample_ledger")
        or not claim_policy.get("uses_explicit_batch_spec")
        or not claim_policy.get("requires_per_sample_candidate_artifacts")
        or not claim_policy.get("requires_per_frame_gaussian_evidence_via_local_row_handoff")
        or not claim_policy.get("reviewable_allows_metric_pass_or_fail")
        or not claim_policy.get("does_not_download_dataset")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_reconstruct_gaussians")
        or not claim_policy.get("does_not_train_model")
        or not claim_policy.get("does_not_claim_intervention_gate")
        or not claim_policy.get("does_not_claim_metric_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("BOP local row batch handoff must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if not isinstance(non_goals, Mapping) or any(bool(value) for value in non_goals.values()):
        raise ValueError(
            "BOP local row batch handoff cannot claim downloads, capture, GT, "
            "Gaussian reconstruction, tracking, learned prediction, intervention, "
            "training, public samples, replay, diffusion, or viewer mutation"
        )
    return dict(payload)


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


def _identity_thresholds_from_options(
    options: Mapping[str, Any],
) -> ObjectStateControlledIdentityThresholds:
    return ObjectStateControlledIdentityThresholds(
        min_idf1=float(options.get("min_idf1", 0.95)),
        min_track_retrieval_recall_at_1=float(
            options.get("min_track_retrieval_recall_at_1", 0.95)
        ),
        max_fragmentation_rate=float(options.get("max_fragmentation_rate", 0.05)),
        max_long_term_drift_rate=float(options.get("max_long_term_drift_rate", 0.05)),
        max_swap_rate=float(options.get("max_swap_rate", 0.0)),
        min_reconstruction_noise_robustness=float(
            options.get("min_reconstruction_noise_robustness", 0.95)
        ),
        min_reconstruction_noise_variants=int(
            options.get("min_reconstruction_noise_variants", 2)
        ),
        require_no_identity_collapse=not bool(
            options.get("allow_identity_collapse", False)
        ),
    )


def _prediction_thresholds_from_options(
    options: Mapping[str, Any],
) -> ObjectStateControlledPredictionThresholds:
    return ObjectStateControlledPredictionThresholds(
        max_state_ade=float(options.get("max_state_ade", 0.05)),
        max_prediction_gap_vs_history_model=float(
            options.get("max_prediction_gap_vs_history_model", 0.02)
        ),
        max_error_ratio_vs_history_model=float(
            options.get("max_error_ratio_vs_history_model", 1.25)
        ),
        min_prediction_count=int(options.get("min_prediction_count", 1)),
    )


def _sample_record(
    index: int,
    sample: Mapping[str, Any],
    sample_output_root: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "index": index,
        "sample_id": summary["sample_id"],
        "scene_root": summary["scene_root"],
        "output_root": str(sample_output_root),
        "candidate_artifact": summary["candidate_artifact"],
        "status": summary["status"],
        "reviewability_gates": dict(summary["reviewability_gates"]),
        "pass_gates": dict(summary["pass_gates"]),
        "row_counts": dict(summary["row_counts"]),
        "files": {
            **dict(summary["files"]),
            "local_row_summary": str(
                sample_output_root / "bop-local-row-handoff-summary.json"
            ),
        },
        "issues": list(summary["issues"]),
        "spec": {
            "scene_root": sample["scene_root"],
            "candidate_artifact": sample["candidate_artifact"],
            "output_root": sample.get("output_root"),
            "condition_sidecar": sample.get("condition_sidecar"),
        },
    }


def _batch_issues(
    reviewability: Mapping[str, bool],
    cross_sample_ledger: Mapping[str, Any],
) -> list[str]:
    labels = {
        "sample_count_nonzero": "batch spec did not produce any sample rows",
        "all_local_row_handoffs_reviewable": (
            "one or more BOP local-row handoffs are not reviewable"
        ),
        "all_local_row_summary_files_written": (
            "one or more BOP local-row summary files were not written"
        ),
        "cross_sample_ledger_reviewable": "cross-sample ledger is not reviewable",
    }
    issues = [message for key, message in labels.items() if not reviewability.get(key)]
    issues.extend(
        f"cross-sample ledger: {issue}"
        for issue in cross_sample_ledger.get("issues", [])
    )
    return issues


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
        raise ValueError("BOP local row batch optional path must be a string")
    return _resolve_input_path(value, base)


def _ensure_batch_output_paths(output_root: Path, *, force: bool) -> None:
    files = (
        output_root / "bop-local-row-batch-handoff-summary.json",
        output_root / "bop-cross-sample-ledger.json",
        output_root / "bop-cross-sample-table.md",
    )
    existing = [path for path in files if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "BOP local row batch handoff refuses to overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"BOP local row batch requires positive int {name}")


def _non_negative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"BOP local row batch requires non-negative int {name}")


__all__ = (
    "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA",
    "objectstate_bop_local_row_batch_handoff",
    "validate_objectstate_bop_local_row_batch_handoff_summary",
)
