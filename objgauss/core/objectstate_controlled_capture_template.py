from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA = (
    "objgauss-objectstate-controlled-capture-bundle-template-v1"
)

OBJECTS_CSV_HEADER = (
    "object_id",
    "category",
    "instance_label",
    "dimension_x_m",
    "dimension_y_m",
    "dimension_z_m",
)
FRAMES_CSV_HEADER = (
    "frame_id",
    "timestamp",
    "rgb",
    "gaussian",
    "action_id",
    "view_id",
    "lighting_id",
    "camera_x",
    "camera_y",
    "camera_z",
    "camera_qx",
    "camera_qy",
    "camera_qz",
    "camera_qw",
)
ANNOTATIONS_CSV_HEADER = (
    "frame_id",
    "object_id",
    "visible",
    "occlusion_fraction",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
)
ACTIONS_CSV_HEADER = (
    "action_id",
    "action_type",
    "object_id",
    "start_timestamp",
    "end_timestamp",
    "actor",
    "target_object_id",
    "vector_x",
    "vector_y",
    "vector_z",
)


def write_objectstate_controlled_capture_bundle_template(
    root: str | Path,
    *,
    sample_id: str,
    object_category: str = "controlled_tabletop",
    scenario: str = "cross_view_occlusion_reappearance",
    fps: float = 30.0,
    capture_device: str = "controlled-camera",
    license_text: str = "local controlled capture; not public release",
    objects: Sequence[Mapping[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    bundle_root = Path(root)
    sample = _sample_payload(
        sample_id=sample_id,
        object_category=object_category,
        scenario=scenario,
        fps=fps,
        capture_device=capture_device,
        license_text=license_text,
    )
    object_rows = [_object_row(item) for item in (objects or ())]
    files = _template_files(bundle_root)
    _ensure_can_write(files.values(), force=force)
    (bundle_root / "rgb").mkdir(parents=True, exist_ok=True)
    (bundle_root / "gaussians").mkdir(parents=True, exist_ok=True)
    _write_json(files["sample_json"], sample)
    _write_csv(files["objects_csv"], OBJECTS_CSV_HEADER, object_rows)
    _write_csv(files["frames_csv"], FRAMES_CSV_HEADER, ())
    _write_csv(files["annotations_csv"], ANNOTATIONS_CSV_HEADER, ())
    _write_csv(files["actions_csv"], ACTIONS_CSV_HEADER, ())
    files["readme"].write_text(
        _readme_text(sample_id=sample_id),
        encoding="utf-8",
    )
    payload = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA,
        "kind": "objectstate_controlled_capture_bundle_template",
        "status": "objectstate_controlled_capture_bundle_template_ready",
        "root": str(bundle_root),
        "sample": sample,
        "object_row_count": len(object_rows),
        "csv_headers": {
            "objects_csv": list(OBJECTS_CSV_HEADER),
            "frames_csv": list(FRAMES_CSV_HEADER),
            "annotations_csv": list(ANNOTATIONS_CSV_HEADER),
            "actions_csv": list(ACTIONS_CSV_HEADER),
        },
        "files": {key: str(value) for key, value in files.items()},
        "directories": {
            "rgb": str(bundle_root / "rgb"),
            "gaussians": str(bundle_root / "gaussians"),
        },
        "next_commands": {
            "populate_frames": (
                "uv run objgauss object-state populate-controlled-capture-frames "
                f"{bundle_root} --summary-output "
                f"{bundle_root / 'frames-summary.json'} --require-ready"
            ),
            "init_annotations": (
                "uv run objgauss object-state init-controlled-capture-annotations "
                f"{bundle_root} --summary-output "
                f"{bundle_root / 'annotation-template-summary.json'} --require-ready"
            ),
            "finalize_annotations": (
                "uv run objgauss object-state finalize-controlled-capture-annotations "
                f"{bundle_root} --summary-output "
                f"{bundle_root / 'annotation-finalize-summary.json'} --require-ready"
            ),
            "import_bundle": (
                "uv run objgauss object-state import-controlled-capture-bundle "
                f"{bundle_root} --output {bundle_root / 'capture-manifest.json'}"
            ),
            "accept_bundle": (
                "uv run objgauss object-state accept-controlled-capture-bundle "
                f"{bundle_root} --output {bundle_root / 'capture-manifest.json'} "
                f"--summary-output {bundle_root / 'acceptance-summary.json'}"
            ),
            "identity_bundle_handoff": (
                "uv run objgauss object-state controlled-identity-bundle-handoff "
                f"{bundle_root} <objectstates.json> "
                f"--output-dir {bundle_root / 'identity-handoff'}"
            ),
        },
        "claim_policy": {
            "template_for_real_capture": True,
            "requires_human_or_external_capture": True,
            "requires_real_rgb_files": True,
            "requires_real_gaussian_files": True,
            "requires_pose_annotations": True,
            "does_not_create_ground_truth": True,
            "does_not_claim_identity_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "captures_video": False,
            "creates_ground_truth": False,
            "creates_frame_rows": False,
            "creates_annotation_rows": False,
            "creates_action_rows": False,
            "reconstructs_gaussians": False,
            "runs_identity_handoff": False,
            "trains_gaussian_model": False,
            "trains_dynamics_model": False,
            "writes_public_samples": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "mutates_viewer_defaults": False,
        },
    }
    return validate_objectstate_controlled_capture_bundle_template_summary(payload)


def validate_objectstate_controlled_capture_bundle_template_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("controlled capture bundle template summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA:
        raise ValueError(
            "unsupported controlled capture bundle template schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_controlled_capture_bundle_template":
        raise ValueError("controlled capture bundle template kind is unsupported")
    if payload.get("status") != "objectstate_controlled_capture_bundle_template_ready":
        raise ValueError("controlled capture bundle template status is unsupported")
    if not isinstance(payload.get("root"), str) or not payload["root"]:
        raise ValueError("controlled capture bundle template requires root")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping):
        raise ValueError("controlled capture bundle template requires sample")
    for key in (
        "sample_id",
        "source_kind",
        "object_category",
        "scenario",
        "fps",
        "capture_device",
        "observation_modalities",
        "artifact_refs",
        "license",
    ):
        if key not in sample:
            raise ValueError(f"controlled capture bundle template sample missing {key}")
    if sample["source_kind"] != "controlled_real":
        raise ValueError("controlled capture bundle template must use controlled_real")
    if sample["observation_modalities"] != ["rgb", "gaussian"]:
        raise ValueError(
            "controlled capture bundle template must require rgb and gaussian"
        )
    if not isinstance(payload.get("object_row_count"), int) or payload["object_row_count"] < 0:
        raise ValueError("controlled capture bundle template requires object_row_count")
    headers = payload.get("csv_headers")
    if not isinstance(headers, Mapping):
        raise ValueError("controlled capture bundle template requires csv_headers")
    expected_headers = {
        "objects_csv": list(OBJECTS_CSV_HEADER),
        "frames_csv": list(FRAMES_CSV_HEADER),
        "annotations_csv": list(ANNOTATIONS_CSV_HEADER),
        "actions_csv": list(ACTIONS_CSV_HEADER),
    }
    if dict(headers) != expected_headers:
        raise ValueError("controlled capture bundle template headers are invalid")
    files = payload.get("files")
    directories = payload.get("directories")
    next_commands = payload.get("next_commands")
    if (
        not isinstance(files, Mapping)
        or not isinstance(directories, Mapping)
        or not isinstance(next_commands, Mapping)
    ):
        raise ValueError(
            "controlled capture bundle template requires files, directories, "
            "and next_commands"
        )
    for key in (
        "sample_json",
        "objects_csv",
        "frames_csv",
        "annotations_csv",
        "actions_csv",
        "readme",
    ):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"controlled capture bundle template missing file {key}")
    for key in ("rgb", "gaussians"):
        if not isinstance(directories.get(key), str) or not directories[key]:
            raise ValueError(
                f"controlled capture bundle template missing directory {key}"
            )
    for key in (
        "populate_frames",
        "init_annotations",
        "finalize_annotations",
        "import_bundle",
        "accept_bundle",
        "identity_bundle_handoff",
    ):
        if not isinstance(next_commands.get(key), str) or not next_commands[key]:
            raise ValueError(
                f"controlled capture bundle template missing command {key}"
            )
    claim_policy = payload.get("claim_policy", {})
    if (
        not claim_policy.get("template_for_real_capture")
        or not claim_policy.get("requires_human_or_external_capture")
        or not claim_policy.get("requires_real_rgb_files")
        or not claim_policy.get("requires_real_gaussian_files")
        or not claim_policy.get("requires_pose_annotations")
        or not claim_policy.get("does_not_create_ground_truth")
        or not claim_policy.get("does_not_claim_identity_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("controlled capture bundle template must preserve claim policy")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("captures_video")
        or non_goals.get("creates_ground_truth")
        or non_goals.get("creates_frame_rows")
        or non_goals.get("creates_annotation_rows")
        or non_goals.get("creates_action_rows")
        or non_goals.get("reconstructs_gaussians")
        or non_goals.get("runs_identity_handoff")
        or non_goals.get("trains_gaussian_model")
        or non_goals.get("trains_dynamics_model")
        or non_goals.get("writes_public_samples")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError(
            "controlled capture bundle template cannot claim capture, GT, rows, "
            "reconstruction, handoff, training, public samples, replay, diffusion, "
            "or viewer mutation"
        )
    return dict(payload)


def _sample_payload(
    *,
    sample_id: str,
    object_category: str,
    scenario: str,
    fps: float,
    capture_device: str,
    license_text: str,
) -> dict[str, Any]:
    if not sample_id:
        raise ValueError("sample_id is required")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    return {
        "sample_id": str(sample_id),
        "source_kind": "controlled_real",
        "object_category": str(object_category),
        "scenario": str(scenario),
        "fps": float(fps),
        "capture_device": str(capture_device),
        "observation_modalities": ["rgb", "gaussian"],
        "artifact_refs": [
            "capture-manifest.json",
            "rgb/",
            "gaussians/",
        ],
        "license": str(license_text),
    }


def _object_row(item: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(item, Mapping):
        raise TypeError("controlled capture template objects must be mappings")
    object_id = _required_text(item, "object_id")
    category = _required_text(item, "category")
    row = {
        "object_id": object_id,
        "category": category,
        "instance_label": str(item.get("instance_label", "") or ""),
        "dimension_x_m": "",
        "dimension_y_m": "",
        "dimension_z_m": "",
    }
    dimensions = item.get("dimensions_m")
    if dimensions is not None:
        if (
            isinstance(dimensions, (str, bytes))
            or not isinstance(dimensions, Sequence)
            or len(dimensions) != 3
        ):
            raise ValueError("object dimensions_m must be a length-3 sequence")
        row["dimension_x_m"] = str(float(dimensions[0]))
        row["dimension_y_m"] = str(float(dimensions[1]))
        row["dimension_z_m"] = str(float(dimensions[2]))
    return row


def _required_text(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"object {key} must be a non-empty string")
    return value


def _template_files(root: Path) -> dict[str, Path]:
    return {
        "sample_json": root / "sample.json",
        "objects_csv": root / "objects.csv",
        "frames_csv": root / "frames.csv",
        "annotations_csv": root / "annotations.csv",
        "actions_csv": root / "actions.csv",
        "readme": root / "README.md",
    }


def _ensure_can_write(paths: Sequence[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "controlled capture template refuses to overwrite existing files: "
            + ", ".join(str(path) for path in existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def _readme_text(*, sample_id: str) -> str:
    return f"""# ObjGauss Controlled Capture Bundle

Sample: {sample_id}

This directory is a local-only skeleton for controlled real ObjectState
validation. It does not contain captured RGB frames, Gaussian reconstructions,
ground truth poses, actions, or candidate model outputs yet.

Follow the repository runbook before filling the CSV rows:

```text
docs/training/controlled-real-capture-runbook.md
```

Required files:

- sample.json: sample metadata and local-only license.
- objects.csv: physical object declarations.
- frames.csv: one row per timestamped RGB/Gaussian observation.
- annotations.csv: one row per frame/object pose and visibility annotation.
- actions.csv: optional action events.
- rgb/: place captured RGB frames referenced by frames.csv.
- gaussians/: place reconstructed per-frame Gaussian files referenced by frames.csv.

Annotation authoring:

- `annotations.template.csv` is a draft helper and is not valid import input.
- Fill measured visibility, occlusion, and 6DoF pose values externally.
- Run `finalize-controlled-capture-annotations` to write `annotations.csv`.

Minimum Stage 1 identity scenario:

- At least three frames.
- At least one object visible before occlusion, occluded, and visible again.
- At least two view_id values.
- At least two lighting_id values.
- At least two camera_pose positions with measurable camera motion.
- A trainable ObjectState artifact with explicit identity_evidence for
  reconstruction noise robustness.

Validation commands:

```bash
uv run objgauss object-state populate-controlled-capture-frames . --summary-output frames-summary.json --require-ready
uv run objgauss object-state init-controlled-capture-annotations . --summary-output annotation-template-summary.json --require-ready
uv run objgauss object-state finalize-controlled-capture-annotations . --summary-output annotation-finalize-summary.json --require-ready
uv run objgauss object-state audit-controlled-capture-bundle-readiness . --summary-output readiness-summary.json
uv run objgauss object-state import-controlled-capture-bundle . --output capture-manifest.json
uv run objgauss object-state accept-controlled-capture-bundle . --output capture-manifest.json --summary-output acceptance-summary.json
uv run objgauss object-state controlled-identity-bundle-handoff . <objectstates.json> --output-dir identity-handoff
```

Do not copy large captures, Gaussian reconstructions, training outputs, or
unlicensed public samples into git.
"""
