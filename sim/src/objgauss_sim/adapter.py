"""PR-01C adapter from real primitive simulation to contract-ready branch candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import time
from pathlib import Path
from typing import Any

from .runtime import assert_offline_asset_gate, assert_versions, install_network_guard
from .writer import BranchCandidate, publish_branch, sha256_bytes, strict_json_bytes


CONTRACT_BRANCH_IDS = {
    "hold": "hold",
    "push_pos_x_weak": "push-pos-x-weak",
    "push_pos_x_strong": "push-pos-x-strong",
    "push_neg_x_weak": "push-neg-x-weak",
    "push_pos_y_weak": "push-pos-y-weak",
}
RNG_ALGORITHM = "numpy.random.RandomState.MT19937"
PRODUCER_VERSION = "0.1.0"
SOURCE_COMMIT_POLICY = "runtime-current-clean-git-head"
SOURCE_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40,64}$")


def validated_source_commit(value: str) -> str:
    if SOURCE_COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError("source commit must be 40-64 lowercase hexadecimal characters")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def producer_tree_sha256(validator: Path) -> str:
    package = Path(__file__).resolve().parent
    sources = {
        name: file_sha256(package / name)
        for name in (
            "adapter.py",
            "canonical.py",
            "cohort.py",
            "primitive.py",
            "runtime.py",
            "writer.py",
        )
    }
    sources["validate-pr01-document.mjs"] = file_sha256(validator)
    return sha256_bytes(strict_json_bytes(sources))


def load_inputs(
    config_path: Path, experiment_path: Path, lock_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    config_file_sha256 = file_sha256(config_path)
    hashes = {
        "config": config.get("experiment_spec_sha256", config_file_sha256),
        "group_config": config_file_sha256,
        "experiment": file_sha256(experiment_path),
        "runtime_lock": file_sha256(lock_path),
    }
    if experiment["identity"]["experiment_spec_sha256"] != hashes["config"]:
        raise RuntimeError("experiment spec hash does not match golden group config")
    if experiment["runtime"]["lock_sha256"] != hashes["runtime_lock"]:
        raise RuntimeError("experiment runtime lock hash does not match sim/uv.lock")
    if experiment["identity"]["experiment_id"] != config["experiment_id"]:
        raise RuntimeError("experiment_id differs between config and experiment manifest")
    if experiment["identity"]["fixture_id"] != config["fixture_id"]:
        raise RuntimeError("fixture_id differs between config and experiment manifest")
    return config, experiment, hashes


def _action_document(branch_id: str, vector: list[float]) -> dict[str, Any]:
    return {
        "kind": "hold" if branch_id == "hold" else "push",
        "vector_W_N": vector,
        "duration_s": 0.1,
        "sim_frequency_hz": 100,
        "applied_steps": 10,
    }


def _episode_document(
    *,
    config: dict[str, Any],
    experiment: dict[str, Any],
    hashes: dict[str, str],
    source_tree_sha256: str,
    source_hashes: dict[str, str],
    branch_id: str,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    contract_branch_id = CONTRACT_BRANCH_IDS[branch_id]
    attempt_id = f"attempt-{config['group_id']}-{contract_branch_id}-1"
    episode_id = f"episode-{config['group_id']}-{contract_branch_id}"
    vector = [float(value) for value in outcome["commanded_action"]["force_n"]]
    action = _action_document(branch_id, vector)
    final_state = outcome["final_target_state"]
    final_linear_speed = sum(value * value for value in final_state[7:10]) ** 0.5
    final_angular_speed = sum(value * value for value in final_state[10:13]) ** 0.5
    thresholds = experiment["thresholds"]
    return {
        "schema_version": "0.2.0",
        "contract_kind": "objgauss.episode",
        "identity": {
            "experiment_id": config["experiment_id"],
            "fixture_id": config["fixture_id"],
            "group_id": config["group_id"],
            "branch_id": contract_branch_id,
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "split": config["split"],
        },
        "initialization": {
            "snapshot_id": config["snapshot_id"],
            "snapshot_sha256": source_hashes["full"],
            "initial_state_sha256": source_hashes["physical"],
            "reset_seed": config["reset_seed"],
            "rng_algorithm": RNG_ALGORITHM,
            "restored_rng_sha256": source_hashes["rng"],
        },
        "intervention": {
            "changed_variable": "/intervention/commanded_action",
            "target_object_id": config["target_object_id"],
            "commanded_action": action,
            "executed_action": dict(action),
            "control_ledger": {
                "application_mode": "force_at_center_of_mass",
                "requested_steps": 10,
                "applied_steps": 10,
                "force_samples_sha256": sha256_bytes(
                    strict_json_bytes([vector for _ in range(10)])
                ),
            },
        },
        "environment": {
            "object_spec_id": config["object_spec_id"],
            "layout_id": config["layout_id"],
            "start_pose_id": config["start_pose_id"],
            "target_object_id": config["target_object_id"],
            "simulator": {
                "name": "mani-skill",
                "version": "3.0.1",
                "physics_engine": "sapien-physx",
                "physics_engine_version": "3.0.3",
                "backend": "physx_cpu",
            },
            "physics": {
                "gravity_W_m_s2": [0.0, 0.0, -9.81],
                "static_friction": 0.5,
                "dynamic_friction": 0.5,
                "restitution": 0.0,
            },
            "contact": {
                "enabled": True,
                "sample_frequency_hz": 100,
                "trace_fields": [
                    "body_pair",
                    "position_W_m",
                    "normal_W",
                    "impulse_W_N_s",
                    "separation_m",
                ],
            },
            "settling": {
                "duration_s": 1.0,
                "linear_speed_max_m_s": thresholds["final_linear_speed_max_m_s"],
                "angular_speed_max_rad_s": thresholds[
                    "final_angular_speed_max_rad_s"
                ],
            },
            "coordinate_convention": {
                "id": "robotics-opencv-v1",
                "world_handedness": "right",
                "world_up": "+Z",
                "length_unit": "meter",
                "force_frame": "world",
                "force_unit": "newton",
                "time_unit": "second",
                "quaternion_order": "wxyz",
            },
        },
        "evidence": {
            "terminal_state_sha256": outcome["final_physical_state_sha256"],
            "settling_result": {
                "settled": final_linear_speed
                <= thresholds["final_linear_speed_max_m_s"]
                and final_angular_speed
                <= thresholds["final_angular_speed_max_rad_s"],
                "evaluation_time_s": 1.1,
                "final_linear_speed_m_s": final_linear_speed,
                "final_angular_speed_rad_s": final_angular_speed,
            },
        },
        "provenance": {
            "parent_snapshot_id": config["snapshot_id"],
            "previous_attempt_id": {
                "availability": "missing",
                "reason": "not_applicable",
            },
            "source_gate_report_sha256": experiment["source_gate"]["report_sha256"],
            "source_commit": config["source_commit"],
            "source_tree_sha256": source_tree_sha256,
            "config_sha256": hashes["config"],
            "runtime_lock_sha256": hashes["runtime_lock"],
            "asset_manifest_sha256": config["asset_manifest_sha256"],
            "producer": {"name": "objgauss.pr01-writer", "version": PRODUCER_VERSION},
        },
    }


def _candidate(
    *,
    config: dict[str, Any],
    experiment: dict[str, Any],
    hashes: dict[str, str],
    source_tree_sha256: str,
    source_hashes: dict[str, str],
    branch_id: str,
    outcome: dict[str, Any],
    started: float,
    finished: float,
) -> BranchCandidate:
    episode = _episode_document(
        config=config,
        experiment=experiment,
        hashes=hashes,
        source_tree_sha256=source_tree_sha256,
        source_hashes=source_hashes,
        branch_id=branch_id,
        outcome=outcome,
    )
    trajectory = {
        "artifact_version": "0.1.0",
        "artifact_kind": "objgauss.trajectory",
        "coordinate_convention": "robotics-opencv-v1",
        "group_id": config["group_id"],
        "branch_id": CONTRACT_BRANCH_IDS[branch_id],
        "records": outcome["trajectory_records"],
    }
    contacts = {
        "artifact_version": "0.1.0",
        "artifact_kind": "objgauss.contact_ledger",
        "coordinate_convention": "robotics-opencv-v1",
        "group_id": config["group_id"],
        "branch_id": CONTRACT_BRANCH_IDS[branch_id],
        "records": outcome["contact_records"],
    }
    return BranchCandidate(
        episode=episode,
        trajectory=trajectory,
        contact_ledger=contacts,
        attempt_timing={
            "started_monotonic_s": started,
            "finished_monotonic_s": finished,
            "wall_seconds": finished - started,
        },
        timeout_s=experiment["budgets"]["branch_timeout_s"],
        attempt_ordinal=1,
        previous_attempt_id={"availability": "missing", "reason": "not_applicable"},
        attempt_provenance={
            "experiment_manifest_sha256": hashes["experiment"],
            "config_sha256": hashes["config"],
            "runtime_lock_sha256": hashes["runtime_lock"],
            "source_tree_sha256": source_tree_sha256,
        },
    )


def run_group(
    *,
    order: str,
    output_root: Path,
    config_path: Path,
    experiment_path: Path,
    lock_path: Path,
    validator: Path,
    node: str,
    source_commit: str,
) -> dict[str, Any]:
    install_network_guard()
    from .canonical import capture_snapshot, snapshot_hashes
    from .primitive import (
        SIBLINGS,
        SPEC,
        PrimitiveActionEnv,
        run_branch,
    )

    asset_before = assert_offline_asset_gate()
    versions = assert_versions()
    config, experiment, hashes = load_inputs(config_path, experiment_path, lock_path)
    if config.get("source_commit_policy") != SOURCE_COMMIT_POLICY:
        raise RuntimeError(
            f"group config must use source_commit_policy={SOURCE_COMMIT_POLICY}"
        )
    config["source_commit"] = validated_source_commit(source_commit)
    source_tree = producer_tree_sha256(validator)

    env = PrimitiveActionEnv(config.get("scene_spec"))
    candidates: dict[str, BranchCandidate] = {}
    try:
        env.reset(seed=config["reset_seed"])
        for _ in range(SPEC["warmup_steps"]):
            env.scene.step()
        source = capture_snapshot(env)
        source_hashes = snapshot_hashes(source)
        execution_order = list(SIBLINGS)
        if order == "reverse":
            execution_order.reverse()
        publications: dict[str, dict[str, Any]] = {}
        for branch_id in execution_order:
            started = time.monotonic()
            outcome = run_branch(
                env,
                source,
                branch_id,
                SIBLINGS.index(branch_id),
                capture_artifacts=True,
                reset_seed=config["reset_seed"],
            )
            finished = time.monotonic()
            candidate = _candidate(
                config=config,
                experiment=experiment,
                hashes=hashes,
                source_tree_sha256=source_tree,
                source_hashes=source_hashes,
                branch_id=branch_id,
                outcome=outcome,
                started=started,
                finished=finished,
            )
            candidates[branch_id] = candidate
            publications[CONTRACT_BRANCH_IDS[branch_id]] = publish_branch(
                candidate,
                root=output_root,
                validator=validator,
                node=node,
            )
        idempotent_replay = {
            CONTRACT_BRANCH_IDS[branch_id]: publish_branch(
                candidates[branch_id],
                root=output_root,
                validator=validator,
                node=node,
            )["status"]
            for branch_id in SIBLINGS
        }
    finally:
        env.close()

    asset_after = assert_offline_asset_gate()
    branch_semantics = {
        branch_id: publications[branch_id]["semantic_sha256"]
        for branch_id in sorted(publications)
    }
    stable_evidence = {
        "config_sha256": hashes["config"],
        "experiment_manifest_sha256": hashes["experiment"],
        "runtime_lock_sha256": hashes["runtime_lock"],
        "source_tree_sha256": source_tree,
        "versions": versions,
        "source_snapshot_hashes": source_hashes,
        "branch_semantic_sha256": branch_semantics,
        "asset_gate": {
            "empty": asset_after["empty"],
            "mode_octal": asset_after["mode_octal"],
        },
        "all_five_branches": len(branch_semantics) == 5,
        "idempotent_replay_all_noop": all(
            status == "noop" for status in idempotent_replay.values()
        ),
        "asset_gate_unchanged": asset_after == asset_before,
    }
    return {
        **stable_evidence,
        "evidence_sha256": sha256_bytes(strict_json_bytes(stable_evidence)),
        "execution_order": [CONTRACT_BRANCH_IDS[item] for item in execution_order],
        "publication_status": {
            branch: publication["status"] for branch, publication in publications.items()
        },
        "idempotent_replay": idempotent_replay,
        "local_verdict": "supported" if all(
            [
                stable_evidence["all_five_branches"],
                stable_evidence["idempotent_replay_all_noop"],
                stable_evidence["asset_gate_unchanged"],
            ]
        ) else "rejected",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("canonical", "reverse"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    exit_code = 0
    try:
        report = run_group(
            order=args.order,
            output_root=args.output_root.resolve(),
            config_path=args.config.resolve(),
            experiment_path=args.experiment.resolve(),
            lock_path=args.lock.resolve(),
            validator=args.validator.resolve(),
            node=args.node,
            source_commit=args.source_commit,
        )
        if report["local_verdict"] != "supported":
            report["verdict"] = "rejected"
            exit_code = 2
        elif args.compare is None:
            report["verdict"] = "pending_repeat"
        else:
            previous = json.loads(args.compare.read_text(encoding="utf-8"))
            comparison = {
                "previous_evidence_sha256": previous.get("evidence_sha256"),
                "matches": previous.get("evidence_sha256")
                == report["evidence_sha256"],
                "opposite_execution_orders": previous.get("execution_order")
                == list(reversed(report["execution_order"])),
                "previous_baseline_valid": previous.get("verdict") == "pending_repeat"
                and previous.get("local_verdict") == "supported",
            }
            report["repeat_comparison"] = comparison
            passed = all(
                comparison[key]
                for key in (
                    "matches",
                    "opposite_execution_orders",
                    "previous_baseline_valid",
                )
            )
            report["verdict"] = "supported" if passed else "rejected"
            exit_code = 0 if passed else 2
    except Exception as error:
        report = {
            "verdict": "invalid",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        exit_code = 4
    report["runtime_telemetry"] = {
        "wall_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "verdict": report["verdict"],
                "evidence_sha256": report.get("evidence_sha256"),
                "runtime_telemetry": report["runtime_telemetry"],
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
