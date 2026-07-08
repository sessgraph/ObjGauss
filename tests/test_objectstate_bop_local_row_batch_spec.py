from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_bop_local_row_batch_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
    validate_objectstate_bop_local_row_batch_spec,
)
from objgauss.core.objectstate_bop_local_row_batch_spec import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA,
    objectstate_bop_local_row_batch_spec_authoring,
    validate_objectstate_bop_local_row_batch_spec_authoring_summary,
)


def test_bop_local_row_batch_spec_authoring_writes_native_spec(tmp_path):
    samples_csv = tmp_path / "inputs" / "samples.csv"
    output = tmp_path / "batch" / "bop-local-row-batch.json"
    rows = []
    for index in range(1, 3):
        sample_id = f"bop-ycbv-scene-{index:06d}"
        scene_root = tmp_path / "dataset" / f"scene-{index:06d}"
        artifact = tmp_path / "artifacts" / sample_id / "objectstates.json"
        sidecar = tmp_path / "conditions" / sample_id / "bop-condition-sidecar.json"
        scene_root.mkdir(parents=True)
        _write_json(artifact, {"schema": "unit-test-candidate"})
        _write_json(sidecar, {"schema": "unit-test-sidecar"})
        rows.append(
                {
                    "sample_id": sample_id,
                    "scene_root": os.path.relpath(scene_root, samples_csv.parent),
                    "candidate_artifact": os.path.relpath(
                        artifact,
                        samples_csv.parent,
                    ),
                    "condition_sidecar": os.path.relpath(
                        sidecar,
                        samples_csv.parent,
                    ),
                    "object_category": f"category-{index}",
                }
            )
    _write_csv(samples_csv, rows)

    summary = objectstate_bop_local_row_batch_spec_authoring(
        samples_csv,
        output=output,
        batch_id="fixture-batch",
        batch_output_root="batch-output",
        dataset_id="bop-ycbv",
    )

    assert summary["schema"] == OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA
    assert summary["status"] == "objectstate_bop_local_row_batch_spec_authoring_ready"
    assert validate_objectstate_bop_local_row_batch_spec_authoring_summary(summary) == summary
    assert summary["readiness"] == {
        "sample_count_nonzero": True,
        "all_scene_roots_present": True,
        "all_candidate_artifacts_present": True,
        "all_declared_condition_sidecars_present": True,
        "native_batch_spec_valid": True,
    }
    spec = _read_json(output)
    assert spec["schema"] == OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA
    assert validate_objectstate_bop_local_row_batch_spec(spec) == spec
    assert spec["batch"] == {
        "batch_id": "fixture-batch",
        "output_root": "batch-output",
    }
    assert spec["defaults"]["dataset_id"] == "bop-ycbv"
    assert spec["samples"][0]["sample_id"] == "bop-ycbv-scene-000001"
    assert spec["samples"][0]["object_category"] == "category-1"
    assert _resolved(output.parent, spec["samples"][0]["scene_root"]).is_dir()
    assert _resolved(output.parent, spec["samples"][0]["candidate_artifact"]).is_file()
    assert _resolved(output.parent, spec["samples"][0]["condition_sidecar"]).is_file()
    assert any(
        "audit-bop-local-row-batch-readiness" in command
        for command in summary["next_commands"]
    )


def test_bop_local_row_batch_spec_authoring_keeps_missing_candidate_visible(tmp_path):
    samples_csv = tmp_path / "samples.csv"
    output = tmp_path / "batch.json"
    scene_root = tmp_path / "scene-000001"
    scene_root.mkdir()
    _write_csv(
        samples_csv,
        [
            {
                "sample_id": "bop-ycbv-scene-000001",
                "scene_root": "scene-000001",
                "candidate_artifact": "missing/objectstates.json",
            }
        ],
    )

    summary = objectstate_bop_local_row_batch_spec_authoring(
        samples_csv,
        output=output,
    )

    assert summary["status"] == "objectstate_bop_local_row_batch_spec_authoring_blocked"
    assert summary["readiness"]["all_candidate_artifacts_present"] is False
    assert summary["row_counts"]["candidate_artifacts_present"] == 0
    assert _read_json(output)["samples"][0]["candidate_artifact"] == "missing/objectstates.json"
    assert any("candidate_artifact" in blocker for blocker in summary["hard_blockers"])
    assert any(
        "all_candidate_artifacts_present" in issue for issue in summary["issues"]
    )


def test_bop_local_row_batch_spec_authoring_cli(tmp_path, capsys):
    samples_csv = tmp_path / "samples.csv"
    output = tmp_path / "batch" / "batch.json"
    summary_output = tmp_path / "batch" / "summary.json"
    scene_root = tmp_path / "scene-000001"
    artifact = tmp_path / "objectstates.json"
    scene_root.mkdir()
    _write_json(artifact, {"schema": "unit-test-candidate"})
    _write_csv(
        samples_csv,
        [
            {
                "sample_id": "bop-ycbv-scene-000001",
                "scene_root": "scene-000001",
                "candidate_artifact": "objectstates.json",
                "frame_step": "2",
                "check_artifact_refs": "true",
            }
        ],
    )

    assert (
        main(
            [
                "object-state",
                "init-bop-local-row-batch-spec",
                "--samples-csv",
                str(samples_csv),
                "--output",
                str(output),
                "--summary-output",
                str(summary_output),
                "--batch-id",
                "cli-batch",
                "--batch-output-root",
                "cli-output",
                "--require-inputs",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_output)
    spec = _read_json(output)

    assert f"schema={OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA}" in stdout
    assert "bop_local_row_batch_spec_authoring_status=objectstate_bop_local_row_batch_spec_authoring_ready" in stdout
    assert "next_command=" in stdout
    assert summary["status"] == "objectstate_bop_local_row_batch_spec_authoring_ready"
    assert spec["batch"]["batch_id"] == "cli-batch"
    assert spec["samples"][0]["frame_step"] == 2
    assert spec["samples"][0]["check_artifact_refs"] is True


def test_bop_local_row_batch_spec_authoring_cli_require_inputs_fails(tmp_path):
    samples_csv = tmp_path / "samples.csv"
    output = tmp_path / "batch.json"
    _write_csv(
        samples_csv,
        [
            {
                "sample_id": "bop-ycbv-scene-000001",
                "scene_root": "missing-scene",
                "candidate_artifact": "missing-objectstates.json",
            }
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "object-state",
                "init-bop-local-row-batch-spec",
                "--samples-csv",
                str(samples_csv),
                "--output",
                str(output),
                "--require-inputs",
            ]
        )
    assert exc.value.code == 2


def _resolved(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
