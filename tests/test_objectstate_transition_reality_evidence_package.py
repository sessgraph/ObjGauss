from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.pipelines.objectstate_phase1_evidence_ledger import (
    objectstate_phase1_evidence_ledger,
)
from objgauss.datasets.objectstate_transition_dataset import (
    write_objectstate_transition_dataset,
)
from objgauss.pipelines.objectstate_transition_reality_evidence_package import (
    OBJECTSTATE_TRANSITION_REALITY_EVIDENCE_PACKAGE_SCHEMA,
    objectstate_transition_reality_evidence_package,
    validate_objectstate_transition_reality_evidence_package_summary,
)
from objgauss.pipelines.objectstate_transition_reality_handoff import (
    write_objectstate_transition_reality_handoff,
)


def test_transition_reality_evidence_package_is_reviewable(tmp_path):
    _write_reviewable_package(tmp_path)

    summary = objectstate_transition_reality_evidence_package(tmp_path)

    assert summary["schema"] == OBJECTSTATE_TRANSITION_REALITY_EVIDENCE_PACKAGE_SCHEMA
    assert (
        summary["status"]
        == "objectstate_transition_reality_evidence_package_reviewable"
    )
    assert summary["sample_id"] == "transition-package-cup-001"
    assert all(summary["reviewability_gates"].values())
    assert summary["transition"]["transition_dataset_ready"] is True
    assert summary["row_accounting"]["identity_row_status"] == "blocked"
    assert summary["row_accounting"]["prediction_row_status"] == "pass"
    assert summary["row_accounting"]["intervention_row_status"] == "pass"
    assert summary["row_accounting"]["pass_row_count"] == 2
    assert summary["row_accounting"]["blocked_row_count"] == 1
    assert summary["handoff"]["requires_identity_pass_row"] is False
    assert summary["output_consistency"]["matches"] is True
    assert summary["issues"] == []
    assert (
        validate_objectstate_transition_reality_evidence_package_summary(summary)
        == summary
    )


def test_transition_reality_evidence_package_reports_missing_eval(tmp_path):
    paths = _write_reviewable_package(tmp_path)
    (paths["handoff_dir"] / "intervention-eval-summary.json").unlink()

    summary = objectstate_transition_reality_evidence_package(tmp_path)

    assert (
        summary["status"]
        == "objectstate_transition_reality_evidence_package_incomplete"
    )
    assert summary["reviewability_gates"]["required_files_present"] is False
    assert (
        summary["reviewability_gates"]["standalone_outputs_match_handoff_summary"]
        is False
    )
    assert any("intervention_eval_summary" in issue for issue in summary["issues"])


def test_transition_reality_evidence_package_cli_and_ledger(tmp_path, capsys):
    _write_reviewable_package(tmp_path)
    package_summary = tmp_path / "transition-reality-evidence-package-summary.json"
    ledger_summary = tmp_path / "phase1-ledger-summary.json"

    assert (
        main(
            [
                "object-state",
                "audit-transition-reality-evidence-package",
                str(tmp_path),
                "--summary-output",
                str(package_summary),
                "--require-reviewable",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    package = json.loads(package_summary.read_text(encoding="utf-8"))
    assert f"schema={OBJECTSTATE_TRANSITION_REALITY_EVIDENCE_PACKAGE_SCHEMA}" in stdout
    assert "reviewable=true" in stdout
    assert "identity_row_status=blocked" in stdout
    assert "prediction_row_status=pass" in stdout
    assert "intervention_row_status=pass" in stdout
    assert "requires_identity_pass_row=false" in stdout
    assert package["status"] == (
        "objectstate_transition_reality_evidence_package_reviewable"
    )

    ledger = objectstate_phase1_evidence_ledger(
        transition_reality_summaries=(package_summary,)
    )
    assert ledger["maturity"] == "transition_reality_reviewable"
    assert ledger["phase1_evidence_gates"]["identity_evidence_reviewable"] is False
    assert ledger["phase1_evidence_gates"]["prediction_evidence_reviewable"] is True
    assert ledger["phase1_evidence_gates"]["intervention_evidence_reviewable"] is True
    assert ledger["phase1_evidence_gates"]["full_reality_evidence_reviewable"] is False
    assert ledger["stage_summary"]["transition_reality"]["reviewable_count"] == 1
    assert ledger["stage_summary"]["transition_reality"]["pass_row_count"] == 2
    assert ledger["stage_summary"]["transition_reality"]["blocked_row_count"] == 1

    assert (
        main(
            [
                "object-state",
                "audit-phase1-evidence-ledger",
                "--transition-reality-summary",
                str(package_summary),
                "--summary-output",
                str(ledger_summary),
                "--require-reviewable",
            ]
        )
        == 0
    )

    ledger_stdout = capsys.readouterr().out
    ledger_payload = json.loads(ledger_summary.read_text(encoding="utf-8"))
    assert "maturity=transition_reality_reviewable" in ledger_stdout
    assert "transition_reality_reviewable=1" in ledger_stdout
    assert "phase1_gate.identity_evidence_reviewable=false" in ledger_stdout
    assert ledger_payload["maturity"] == "transition_reality_reviewable"


def test_phase1_ledger_discovers_transition_reality_summary(tmp_path):
    sample_root = tmp_path / "sample"
    _write_reviewable_package(sample_root)
    package_summary = (
        sample_root / "transition-reality-evidence-package-summary.json"
    )
    summary = objectstate_transition_reality_evidence_package(sample_root)
    _write_json(package_summary, summary)

    ledger = objectstate_phase1_evidence_ledger(
        discover_roots=(tmp_path,),
        max_depth=2,
    )

    assert ledger["status"] == "objectstate_phase1_evidence_ledger_reviewable"
    assert ledger["maturity"] == "transition_reality_reviewable"
    assert ledger["discovery"]["transition_reality_summary_count"] == 1
    assert ledger["stage_summary"]["transition_reality"]["package_count"] == 1


def _write_reviewable_package(root):
    root.mkdir(parents=True, exist_ok=True)
    capture_path = root / "capture-manifest.json"
    transition_path = root / "objectstate-transitions.json"
    handoff_dir = root / "transition-reality-handoff"
    _write_json(capture_path, _capture_manifest())
    write_objectstate_transition_dataset(
        capture_path,
        transition_path,
        require_action_transition=True,
    )
    write_objectstate_transition_reality_handoff(
        capture_path,
        transition_path,
        handoff_dir,
        require_gaussian_refs=True,
    )
    return {"handoff_dir": handoff_dir}


def _capture_manifest() -> dict:
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "transition-package-cup-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "transition_reality_evidence_package",
            "fps": 2.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/captures/transition-package-cup-001/capture-manifest.json",
                "outputs/captures/transition-package-cup-001/rgb/",
                "outputs/captures/transition-package-cup-001/gaussians/",
            ],
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {"object_id": "cup-001", "category": "cup", "instance_label": "blue cup"},
        ],
        "actions": [
            {
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "cup-001",
                "start_timestamp": 0.0,
                "end_timestamp": 0.5,
                "actor": "fixture-human",
                "vector": [-0.1, 0.0, 0.0],
            }
        ],
        "frames": [
            _frame("frame-000000", 0.0, [0.1, 0.2, 0.3], action_id="push-left-001"),
            _frame("frame-000001", 0.5, [0.0, 0.2, 0.3]),
            _frame("frame-000002", 1.0, [-0.1, 0.2, 0.3]),
        ],
    }


def _frame(
    frame_id: str,
    timestamp: float,
    position: list[float],
    *,
    action_id: str | None = None,
) -> dict:
    frame = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "observation": {
            "rgb": f"rgb/{frame_id}.png",
            "gaussian": f"gaussians/{frame_id}.ply",
        },
        "objects": [
            {
                "object_id": "cup-001",
                "visible": True,
                "occlusion_fraction": 0.0,
                "pose": {
                    "position": position,
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    }
    if action_id is not None:
        frame["action_id"] = action_id
    return frame


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
