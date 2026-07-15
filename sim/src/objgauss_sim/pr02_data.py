"""Materialize only the frozen PR-02C train/validation sibling source."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

from .adapter import SOURCE_COMMIT_POLICY, producer_tree_sha256, run_group
from .cohort import CONTEXT_SPEC, directory_bytes
from .writer import sha256_bytes, strict_json_bytes


REPORT_VERSION = "0.1.0"
REPORT_KIND = "objgauss.pr02c-source-materialization"
PLAN_KIND = "objgauss.pr02c-source-plan"
ALLOWED_SPLITS = ("train", "validation")
BRANCH_IDS = (
    "hold",
    "push-neg-x-weak",
    "push-pos-x-strong",
    "push-pos-x-weak",
    "push-pos-y-weak",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def validate_requested_splits(values: list[str]) -> tuple[str, ...]:
    if not values:
        raise ValueError("at least one source split is required")
    if len(values) != len(set(values)):
        raise ValueError("source splits must be unique")
    forbidden = [value for value in values if value not in ALLOWED_SPLITS]
    if forbidden:
        raise ValueError(
            f"PR-02C source split is forbidden: {','.join(sorted(forbidden))}"
        )
    return tuple(split for split in ALLOWED_SPLITS if split in values)


def validate_output_root(repo_root: Path, output_root: Path) -> Path:
    root = repo_root.resolve()
    output = output_root.resolve()
    allowed = (root / "generated" / "pr02c").resolve()
    if allowed not in output.parents:
        raise ValueError("PR-02C source output must be below generated/pr02c/")
    if output.exists() and any(output.iterdir()):
        raise ValueError("PR-02C source output root must be empty")
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_frozen_inputs(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
    formal: dict[str, Any],
    formal_path: Path,
    dynamics: dict[str, Any],
    dynamics_path: Path,
    pilot: dict[str, Any],
    pilot_path: Path,
    source_experiment: dict[str, Any],
    source_experiment_path: Path,
    lock_path: Path,
) -> None:
    if manifest.get("manifest_kind") != "objgauss.pr02c-data-boundary":
        raise ValueError("PR-02C data boundary manifest kind drift")
    frozen = manifest.get("frozen_inputs", {})
    for name in ("pilot_spec", "source_experiment", "simulator_lock"):
        entry = frozen.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"missing frozen input: {name}")
        path = (repo_root / entry["path"]).resolve()
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"frozen input checksum drift: {name}")
    if file_sha256(source_experiment_path) != frozen["source_experiment"]["sha256"]:
        raise ValueError("source experiment path differs from frozen input")
    if file_sha256(lock_path) != frozen["simulator_lock"]["sha256"]:
        raise ValueError("simulator lock path differs from frozen input")
    formal_sha256 = file_sha256(formal_path)
    if formal_sha256 != frozen.get("formal_data_spec", {}).get("sha256"):
        raise ValueError("formal data spec checksum drift")
    pilot_sha256 = file_sha256(pilot_path)
    if pilot.get("verdict", {}).get("status") != "supported":
        raise ValueError("PR-02B pilot report is not supported")
    if pilot.get("freeze", {}).get("formal_data_spec_sha256") != formal_sha256:
        raise ValueError("pilot report does not bind the formal data spec")
    if dynamics.get("contract_kind") != "objgauss.dynamics_experiment":
        raise ValueError("dynamics experiment kind drift")
    if dynamics.get("source", {}).get("source_gate_report_sha256") != pilot_sha256:
        raise ValueError("dynamics experiment does not bind the current pilot report")
    if dynamics.get("source", {}).get("runtime_lock_sha256") != file_sha256(lock_path):
        raise ValueError("dynamics experiment simulator lock drift")
    expected_counts = {"train": 48, "validation": 12, "test": 12}
    if formal.get("group_counts") != expected_counts or formal.get("total_groups") != 72:
        raise ValueError("formal split counts drift")
    if formal.get("final_access") != {
        "evaluator_only_reads_gt_future": True,
        "inference_reads_gt_future": False,
        "max_formal_runs": 1,
        "trainer_loader_rejects_test": True,
    }:
        raise ValueError("formal final-access policy drift")
    materialization = manifest.get("materialization", {})
    if materialization.get("allowed_splits") != list(ALLOWED_SPLITS):
        raise ValueError("materialized split policy drift")
    if materialization.get("group_counts") != {"train": 48, "validation": 12}:
        raise ValueError("materialized group counts drift")
    if materialization.get("branch_ids") != list(BRANCH_IDS):
        raise ValueError("materialized branch set drift")
    if formal.get("actions") != source_experiment.get("actions"):
        raise ValueError("formal action schedule differs from audited source action schedule")
    for split in ("train", "validation", "test"):
        partition = formal["partitions"][split]
        projection = dynamics["data_policy"][split]
        expected = {
            "object_identity_ids": sorted(partition["objects"]),
            "layout_ids": sorted(partition["layouts"]),
            "group_ids": [group["group_id"] for group in partition["groups"]],
            "group_count": len(partition["groups"]),
        }
        if projection != expected:
            raise ValueError(f"dynamics experiment projection drift: {split}")
    del dynamics_path


def build_source_plan(
    *,
    formal: dict[str, Any],
    formal_sha256: str,
    pilot_sha256: str,
    source_experiment: dict[str, Any],
    lock_sha256: str,
    requested_splits: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "plan_version": REPORT_VERSION,
        "plan_kind": PLAN_KIND,
        "identity": {
            "experiment_id": formal["experiment_id"],
            "fixture_id": "pr02c-source-v0",
            "experiment_spec_sha256": formal_sha256,
        },
        "source_gate": {"status": "supported", "report_sha256": pilot_sha256},
        "runtime": {
            **source_experiment["runtime"],
            "lock_sha256": lock_sha256,
        },
        "thresholds": source_experiment["thresholds"],
        "budgets": {"branch_timeout_s": source_experiment["budgets"]["branch_timeout_s"]},
        "actions": formal["actions"],
        "materialization": {
            "splits": list(requested_splits),
            "test_materialized": False,
            "group_counts": {
                split: len(formal["partitions"][split]["groups"])
                for split in requested_splits
            },
            "branches_per_group": len(BRANCH_IDS),
        },
        "claim_boundary": "train/validation source only; no final GT, loader, trainer, or model claim",
    }


def group_config(
    *,
    formal: dict[str, Any],
    formal_sha256: str,
    plan: dict[str, Any],
    split: str,
    group: dict[str, Any],
    source_commit: str,
    asset_manifest_sha256: str,
) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"cannot build PR-02C group config for split={split}")
    partition = formal["partitions"][split]
    return {
        "config_version": "0.2.0",
        "experiment_id": plan["identity"]["experiment_id"],
        "fixture_id": plan["identity"]["fixture_id"],
        "experiment_spec_sha256": formal_sha256,
        "group_id": group["group_id"],
        "split": split,
        "object_spec_id": group["object_identity_id"],
        "layout_id": group["layout_id"],
        "start_pose_id": group["start_pose_id"],
        "reset_seed": group["reset_seed"],
        "target_object_id": "target",
        "snapshot_id": f"snapshot-{group['group_id']}",
        "source_commit_policy": SOURCE_COMMIT_POLICY,
        "source_commit": source_commit,
        "asset_manifest_sha256": asset_manifest_sha256,
        "scene_spec": {
            "target": partition["objects"][group["object_identity_id"]],
            "context": CONTEXT_SPEC,
            "context_xy_m": partition["layouts"][group["layout_id"]]["context_xy_m"],
            "target_base_xy_m": partition["starts"][group["start_pose_id"]]["target_base_xy_m"],
            "target_jitter_m": partition["starts"][group["start_pose_id"]]["target_jitter_m"],
        },
    }


def branch_index(output_root: Path, experiment_id: str, group_id: str) -> list[dict[str, str]]:
    result = []
    for branch_id in BRANCH_IDS:
        directory = output_root / "dataset" / experiment_id / group_id / branch_id
        publication = load_json(directory / "publication.json")
        result.append(
            {
                "group_id": group_id,
                "branch_id": branch_id,
                "episode_sha256": publication["episode_sha256"],
                "trajectory_sha256": publication["trajectory_sha256"],
                "semantic_sha256": publication["semantic_sha256"],
            }
        )
    return result


def run_materialization(
    *,
    repo_root: Path,
    output_root: Path,
    manifest_path: Path,
    formal_path: Path,
    dynamics_path: Path,
    pilot_path: Path,
    source_experiment_path: Path,
    pilot_spec_path: Path,
    lock_path: Path,
    validator: Path,
    node: str,
    source_commit: str,
    requested_splits: tuple[str, ...],
) -> dict[str, Any]:
    output_root = validate_output_root(repo_root, output_root)
    manifest = load_json(manifest_path)
    formal = load_json(formal_path)
    dynamics = load_json(dynamics_path)
    pilot = load_json(pilot_path)
    source_experiment = load_json(source_experiment_path)
    pilot_spec = load_json(pilot_spec_path)
    validate_frozen_inputs(
        repo_root=repo_root,
        manifest=manifest,
        formal=formal,
        formal_path=formal_path,
        dynamics=dynamics,
        dynamics_path=dynamics_path,
        pilot=pilot,
        pilot_path=pilot_path,
        source_experiment=source_experiment,
        source_experiment_path=source_experiment_path,
        lock_path=lock_path,
    )
    formal_sha256 = file_sha256(formal_path)
    plan = build_source_plan(
        formal=formal,
        formal_sha256=formal_sha256,
        pilot_sha256=file_sha256(pilot_path),
        source_experiment=source_experiment,
        lock_sha256=file_sha256(lock_path),
        requested_splits=requested_splits,
    )
    plan_path = output_root / "source-plan.json"
    plan_path.write_bytes(strict_json_bytes(plan))
    started = time.monotonic()
    index: list[dict[str, str]] = []
    group_reports = []
    config_root = output_root / ".configs"
    config_root.mkdir()
    try:
        ordinal = 0
        for split in requested_splits:
            for group in formal["partitions"][split]["groups"]:
                ordinal += 1
                config = group_config(
                    formal=formal,
                    formal_sha256=formal_sha256,
                    plan=plan,
                    split=split,
                    group=group,
                    source_commit=source_commit,
                    asset_manifest_sha256=pilot_spec["asset_manifest_sha256"],
                )
                config_path = config_root / f"{group['group_id']}.json"
                config_path.write_bytes(strict_json_bytes(config))
                result = run_group(
                    order="canonical",
                    output_root=output_root,
                    config_path=config_path,
                    experiment_path=plan_path,
                    lock_path=lock_path,
                    validator=validator,
                    node=node,
                    source_commit=source_commit,
                )
                if result["local_verdict"] != "supported":
                    raise RuntimeError(f"source group failed: {group['group_id']}")
                entries = branch_index(
                    output_root, plan["identity"]["experiment_id"], group["group_id"]
                )
                index.extend(entries)
                group_reports.append(
                    {
                        "ordinal": ordinal,
                        "split": split,
                        "group_id": group["group_id"],
                        "evidence_sha256": result["evidence_sha256"],
                        "source_tree_sha256": result["source_tree_sha256"],
                    }
                )
    finally:
        for path in config_root.glob("*.json"):
            path.unlink()
        config_root.rmdir()
    index.sort(key=lambda item: (item["group_id"], item["branch_id"]))
    wall_seconds = time.monotonic() - started
    report = {
        "report_version": REPORT_VERSION,
        "report_kind": REPORT_KIND,
        "verdict": {"status": "supported", "reason_code": "all_c1_source_gates_passed"},
        "producer": {
            "name": "objgauss.pr02c-source",
            "version": "0.1.0",
            "source_commit": source_commit,
            "source_tree_sha256": producer_tree_sha256(validator),
        },
        "inputs": {
            "data_boundary_manifest_sha256": file_sha256(manifest_path),
            "formal_data_spec_sha256": formal_sha256,
            "dynamics_experiment_sha256": file_sha256(dynamics_path),
            "pilot_report_sha256": file_sha256(pilot_path),
            "source_plan_sha256": file_sha256(plan_path),
            "simulator_lock_sha256": file_sha256(lock_path),
        },
        "counts": {
            "groups": len(group_reports),
            "branches": len(index),
            "episodes": len(index),
            "failed_attempts": 0,
            "split_groups": {
                split: sum(item["split"] == split for item in group_reports)
                for split in ALLOWED_SPLITS
            },
        },
        "isolation": {
            "materialized_splits": list(requested_splits),
            "test_materialized": False,
            "test_group_count": 0,
        },
        "data_index_sha256": sha256_bytes(strict_json_bytes(index)),
        "groups": group_reports,
        "telemetry": {
            "wall_seconds": wall_seconds,
            "artifact_bytes": directory_bytes(output_root / "dataset"),
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "claim_boundary": {
            "supported_claim": "train-validation-source-materialized",
            "excluded_claims": [
                "test-source-materialized",
                "loader-validated",
                "trainer-implemented",
                "model-performance",
            ],
        },
    }
    (output_root / "source-report.json").write_bytes(strict_json_bytes(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("learning/data-boundary-manifest.json"))
    parser.add_argument("--formal-spec", type=Path, required=True)
    parser.add_argument("--dynamics-experiment", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, default=Path("contracts/fixtures/pr02b/source-experiment.json"))
    parser.add_argument("--pilot-spec", type=Path, default=Path("contracts/fixtures/pr02b/pilot-spec.json"))
    parser.add_argument("--lock", type=Path, default=Path("sim/uv.lock"))
    parser.add_argument("--validator", type=Path, default=Path("scripts/validate-pr01-document.mjs"))
    parser.add_argument("--node", default="node")
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    resolve = lambda path: path if path.is_absolute() else repo_root / path
    try:
        splits = validate_requested_splits(args.splits)
        report = run_materialization(
            repo_root=repo_root,
            output_root=resolve(args.output_root),
            manifest_path=resolve(args.manifest),
            formal_path=resolve(args.formal_spec),
            dynamics_path=resolve(args.dynamics_experiment),
            pilot_path=resolve(args.pilot_report),
            source_experiment_path=resolve(args.source_experiment),
            pilot_spec_path=resolve(args.pilot_spec),
            lock_path=resolve(args.lock),
            validator=resolve(args.validator),
            node=args.node,
            source_commit=args.source_commit,
            requested_splits=splits,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "verdict": "invalid",
                    "reason_code": "source_or_lineage_invalid",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                sort_keys=True,
            )
        )
        return 4
    print(
        json.dumps(
            {
                "verdict": report["verdict"]["status"],
                "counts": report["counts"],
                "data_index_sha256": report["data_index_sha256"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
