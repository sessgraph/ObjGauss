"""Frozen PR-01E preflight and formal cohort orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import tempfile
import time
from pathlib import Path
from typing import Any


CONTEXT_SPEC = {"half_size_m": [0.02, 0.025, 0.015], "density_kg_m3": 700.0}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank_key(
    object_spec_id: str, layout_id: str, start_pose_id: str, seed: int
) -> str:
    payload = f"{object_spec_id}\0{layout_id}\0{start_pose_id}\0{seed}".encode()
    return hashlib.sha256(payload).hexdigest()


def assigned_split(
    *,
    object_spec_id: str,
    layout_id: str,
    start_pose_id: str,
    seed: int,
    seeds: list[int],
    allocation: dict[str, int],
) -> str:
    ranked = sorted(
        seeds,
        key=lambda item: stable_rank_key(
            object_spec_id, layout_id, start_pose_id, item
        ),
    )
    rank = ranked.index(seed)
    if rank < allocation["train"]:
        return "train"
    if rank < allocation["train"] + allocation["validation"]:
        return "validation"
    return "test"


def design_groups(
    spec: dict[str, Any], manifest: dict[str, Any], mode: str
) -> list[dict[str, Any]]:
    if mode == "preflight":
        starts = spec["preflight_start_pose_ids"]
        seeds = spec["preflight_reset_seeds"]
    elif mode == "formal":
        starts = list(spec["start_poses"])
        seeds = spec["formal_reset_seeds"]
    else:
        raise ValueError(f"unknown cohort mode: {mode}")
    groups = []
    for object_spec_id in spec["object_specs"]:
        for layout_id in spec["layouts"]:
            for start_pose_id in starts:
                for seed in seeds:
                    split = "preflight" if mode == "preflight" else assigned_split(
                        object_spec_id=object_spec_id,
                        layout_id=layout_id,
                        start_pose_id=start_pose_id,
                        seed=seed,
                        seeds=spec["formal_reset_seeds"],
                        allocation=spec["split_allocation"],
                    )
                    group_id = (
                        f"group-{object_spec_id}-{layout_id}-{start_pose_id}-seed-{seed}"
                    )
                    groups.append(
                        {
                            "group_id": group_id,
                            "split": split,
                            "object_spec_id": object_spec_id,
                            "layout_id": layout_id,
                            "start_pose_id": start_pose_id,
                            "reset_seed": seed,
                        }
                    )
    return groups


def validate_spec(
    spec: dict[str, Any], manifest: dict[str, Any], spec_sha256: str
) -> None:
    identity = manifest["identity"]
    design = manifest["design"]
    if identity["experiment_id"] != spec["experiment_id"]:
        raise ValueError("experiment_id differs between cohort spec and manifest")
    if identity["fixture_id"] != spec["fixture_id"]:
        raise ValueError("fixture_id differs between cohort spec and manifest")
    if identity["experiment_spec_sha256"] != spec_sha256:
        raise ValueError("cohort spec checksum differs from manifest")
    comparisons = (
        (design["object_spec_ids"], list(spec["object_specs"]), "object specs"),
        (design["layout_ids"], list(spec["layouts"]), "layouts"),
        (design["start_pose_ids"], list(spec["start_poses"]), "start poses"),
        (design["reset_seeds"], spec["formal_reset_seeds"], "formal seeds"),
        (manifest["preflight"]["start_pose_ids"], spec["preflight_start_pose_ids"], "preflight starts"),
        (manifest["preflight"]["reserved_reset_seeds"], spec["preflight_reset_seeds"], "preflight seeds"),
        (manifest["split_policy"]["seed_rank_allocation"], spec["split_allocation"], "split allocation"),
        (manifest["thresholds"], spec["thresholds"], "thresholds"),
    )
    for actual, expected, name in comparisons:
        if actual != expected:
            raise ValueError(f"{name} differ between cohort spec and manifest")


def percentile95(values: list[float | int]) -> float:
    if not values:
        raise ValueError("cannot compute p95 of an empty collection")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def branch_wall_seconds(path: Path) -> list[float]:
    values = []
    for attempt_path in path.rglob("attempt.json"):
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        values.append(float(attempt["timing"]["wall_seconds"]))
    return values


def group_config(
    spec: dict[str, Any],
    spec_sha256: str,
    group: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    object_spec = spec["object_specs"][group["object_spec_id"]]
    layout = spec["layouts"][group["layout_id"]]
    start = spec["start_poses"][group["start_pose_id"]]
    return {
        "config_version": "0.2.0",
        "experiment_id": spec["experiment_id"],
        "fixture_id": spec["fixture_id"],
        "experiment_spec_sha256": spec_sha256,
        **group,
        "target_object_id": "target",
        "snapshot_id": f"snapshot-{group['group_id']}",
        "source_commit_policy": spec["source_commit_policy"],
        "source_commit": source_commit,
        "asset_manifest_sha256": spec["asset_manifest_sha256"],
        "scene_spec": {
            "target": object_spec,
            "context": CONTEXT_SPEC,
            "context_xy_m": layout["context_xy_m"],
            "target_base_xy_m": start["target_base_xy_m"],
            "target_jitter_m": start["target_jitter_m"],
        },
    }


def run_cohort(
    *,
    mode: str,
    output_root: Path,
    report_path: Path,
    spec_path: Path,
    experiment_path: Path,
    lock_path: Path,
    validator: Path,
    node: str,
    source_commit: str,
) -> dict[str, Any]:
    from .adapter import (
        SOURCE_COMMIT_POLICY,
        run_group,
        validated_source_commit,
    )
    from .runtime import install_network_guard
    from .writer import strict_json_bytes

    install_network_guard()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(experiment_path.read_text(encoding="utf-8"))
    source_commit = validated_source_commit(source_commit)
    if spec.get("source_commit_policy") != SOURCE_COMMIT_POLICY:
        raise ValueError(
            f"cohort spec must use source_commit_policy={SOURCE_COMMIT_POLICY}"
        )
    spec_sha256 = file_sha256(spec_path)
    validate_spec(spec, manifest, spec_sha256)
    groups = design_groups(spec, manifest, mode)
    expected_groups = (
        manifest["preflight"]["expected_group_count"]
        if mode == "preflight"
        else manifest["design"]["expected_group_count"]
    )
    if len(groups) != expected_groups:
        raise RuntimeError(
            f"designed {len(groups)} groups but manifest requires {expected_groups}"
        )

    started = time.monotonic()
    group_reports = []
    group_runtime_s: list[float] = []
    group_artifact_bytes: list[int] = []
    branch_runtime_s: list[float] = []
    with tempfile.TemporaryDirectory(prefix="objgauss-pr01e-config-") as temporary:
        config_root = Path(temporary)
        for index, group in enumerate(groups, start=1):
            config_path = config_root / f"{group['group_id']}.json"
            config_path.write_bytes(
                strict_json_bytes(
                    group_config(spec, spec_sha256, group, source_commit)
                )
            )
            group_started = time.monotonic()
            result = run_group(
                order="canonical",
                output_root=output_root,
                config_path=config_path,
                experiment_path=experiment_path,
                lock_path=lock_path,
                validator=validator,
                node=node,
                source_commit=source_commit,
            )
            group_wall = time.monotonic() - group_started
            if result["local_verdict"] != "supported":
                raise RuntimeError(
                    f"group {group['group_id']} writer gate is {result['local_verdict']}"
                )
            branch_root = (
                output_root / "dataset" / spec["experiment_id"] / group["group_id"]
            )
            artifact_bytes = directory_bytes(branch_root)
            branch_walls = branch_wall_seconds(branch_root)
            if len(branch_walls) != 5:
                raise RuntimeError(
                    f"group {group['group_id']} does not have five success attempts"
                )
            if max(branch_walls) > manifest["budgets"]["branch_timeout_s"]:
                raise RuntimeError(
                    f"group {group['group_id']} exceeded branch timeout"
                )
            group_runtime_s.append(group_wall)
            group_artifact_bytes.append(artifact_bytes)
            branch_runtime_s.extend(branch_walls)
            group_reports.append(
                {
                    **group,
                    "ordinal": index,
                    "evidence_sha256": result["evidence_sha256"],
                    "source_tree_sha256": result["source_tree_sha256"],
                    "wall_seconds": group_wall,
                    "artifact_bytes": artifact_bytes,
                    "branch_count": len(result["branch_semantic_sha256"]),
                    "idempotent_replay_all_noop": result[
                        "idempotent_replay_all_noop"
                    ],
                }
            )

    wall_seconds = time.monotonic() - started
    total_bytes = directory_bytes(output_root / "dataset")
    max_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report = {
        "report_version": "0.1.0",
        "report_kind": "objgauss.pr01e-cohort-run",
        "mode": mode,
        "verdict": "supported",
        "experiment_id": spec["experiment_id"],
        "experiment_manifest_sha256": file_sha256(experiment_path),
        "experiment_spec_sha256": spec_sha256,
        "source_commit": source_commit,
        "counts": {
            "groups": len(groups),
            "episodes": len(groups) * 5,
            "attempts": len(groups) * 5,
            "failed_attempts": 0,
            "extra_attempts": 0,
        },
        "telemetry": {
            "wall_seconds": wall_seconds,
            "p95_group_runtime_s": percentile95(group_runtime_s),
            "p95_branch_runtime_s": percentile95(branch_runtime_s),
            "max_branch_runtime_s": max(branch_runtime_s),
            "total_artifact_bytes": total_bytes,
            "p95_group_artifact_bytes": int(percentile95(group_artifact_bytes)),
            "max_rss_bytes": max_rss_bytes,
        },
        "thresholds": manifest["thresholds"],
        "budgets": manifest["budgets"],
        "split_counts": {
            split: sum(group["split"] == split for group in groups)
            for split in ("train", "validation", "test", "preflight")
        },
        "groups": group_reports,
    }
    if mode == "preflight":
        report["freeze_candidate"] = {
            "branch_timeout_s": manifest["budgets"]["branch_timeout_s"],
            "formal_wall_time_max_s": math.ceil(
                report["telemetry"]["p95_group_runtime_s"]
                * manifest["design"]["expected_group_count"]
                * 1.25
            ),
            "formal_artifact_bytes_max": math.ceil(
                report["telemetry"]["p95_group_artifact_bytes"]
                * manifest["design"]["expected_group_count"]
                * 1.15
            ),
            "process_rss_max_bytes": manifest["budgets"]["process_rss_max_bytes"],
            "extra_attempt_fraction_max": manifest["budgets"][
                "extra_attempt_fraction_max"
            ],
            "thresholds": manifest["thresholds"],
            "basis": {
                "group_wall_multiplier": 1.25,
                "group_artifact_multiplier": 1.15,
                "preflight_excluded_from_formal_cohort": True,
            },
        }
    if mode == "formal":
        budgets = manifest["budgets"]
        if wall_seconds > budgets["formal_wall_time_max_s"]:
            raise RuntimeError("formal cohort exceeded wall-time budget")
        if total_bytes > budgets["formal_artifact_bytes_max"]:
            raise RuntimeError("formal cohort exceeded artifact budget")
        if max_rss_bytes > budgets["process_rss_max_bytes"]:
            raise RuntimeError("formal cohort exceeded RSS budget")
        if report["split_counts"] != {
            "train": manifest["split_policy"]["counts"]["train_groups"],
            "validation": manifest["split_policy"]["counts"]["validation_groups"],
            "test": manifest["split_policy"]["counts"]["test_groups"],
            "preflight": 0,
        }:
            raise RuntimeError("formal split counts differ from manifest")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(strict_json_bytes(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_cohort(
            mode=args.mode,
            output_root=args.output_root.resolve(),
            report_path=args.report.resolve(),
            spec_path=args.spec.resolve(),
            experiment_path=args.experiment.resolve(),
            lock_path=args.lock.resolve(),
            validator=args.validator.resolve(),
            node=args.node,
            source_commit=args.source_commit,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "verdict": "invalid",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                sort_keys=True,
            )
        )
        return 4
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "mode": report["mode"],
                "counts": report["counts"],
                "telemetry": report["telemetry"],
                "report": str(args.report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
