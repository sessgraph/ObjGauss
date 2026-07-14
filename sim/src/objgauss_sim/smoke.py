"""Real five-branch, cross-process PR-01B runtime smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

from .runtime import (
    assert_offline_asset_gate,
    assert_versions,
    install_network_guard,
)


ERROR_SPEC = {
    "smoke_id": "objgauss-pr01-five-branch-runtime-v0",
    "smoke_version": "0.1.0",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def producer_sources(lock_path: Path) -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {
        name: file_sha256(package_dir / name)
        for name in ("canonical.py", "primitive.py", "runtime.py", "smoke.py")
    } | {"uv.lock": file_sha256(lock_path)}


def run_smoke(order: str, lock_path: Path) -> dict[str, Any]:
    install_network_guard()
    import numpy as np

    from .canonical import capture_snapshot, digest, snapshot_hashes
    from .primitive import (
        SIBLINGS,
        SPEC,
        PrimitiveActionEnv,
        actor_state,
        direction_checks,
        negative_controls,
        observed_backend,
        paired_effects,
        run_branch,
    )

    asset_before = assert_offline_asset_gate()
    versions = assert_versions()
    if not lock_path.is_file():
        raise RuntimeError(f"runtime lock is missing: {lock_path}")

    env = PrimitiveActionEnv()
    try:
        backend = observed_backend(env)
        env.reset(seed=SPEC["seed"])
        for _ in range(SPEC["warmup_steps"]):
            env.scene.step()
        source = capture_snapshot(env)
        source_hashes = snapshot_hashes(source)
        source_target = actor_state(env.target)

        execution_order = list(SIBLINGS)
        if order == "reverse":
            execution_order.reverse()
        outcomes = {
            branch_id: run_branch(env, source, branch_id, SIBLINGS.index(branch_id))
            for branch_id in execution_order
        }
        effects = paired_effects(outcomes, source_target)
        checks = {
            "fixed_versions": True,
            "no_agent": backend["agent_is_none"],
            "no_sensors": backend["sensor_count"] == 0,
            "no_human_render_cameras": backend["human_render_camera_count"] == 0,
            "rendering_disabled": backend["scene_can_render"] is False,
            "cpu_physics_backend": backend["sim_backend"] == "physx_cpu",
            "registered_actors_exact": backend["actors"]
            == ["context", "floor", "target"],
            "all_siblings_present": sorted(outcomes) == sorted(SIBLINGS),
            "only_commanded_action_is_declared_changed_variable": all(
                outcome["changed_variable"] == SPEC["changed_variable"]
                for outcome in outcomes.values()
            ),
            "all_pre_action_full_hashes_equal_source": all(
                outcome["pre_action_hashes"]["full"] == source_hashes["full"]
                for outcome in outcomes.values()
            ),
            "all_pre_action_target_states_equal_source": all(
                digest(outcome["pre_action_target_state"]) == digest(source_target)
                for outcome in outcomes.values()
            ),
            "all_executed_ledgers_match_commands": all(
                digest(outcome["commanded_action"])
                == digest(outcome["executed_action"])
                for outcome in outcomes.values()
            ),
            "all_branches_have_target_floor_contact": all(
                outcome["contact_summary"]["target_floor_point_count"] > 0
                for outcome in outcomes.values()
            ),
            "no_branch_has_target_context_contact": all(
                outcome["contact_summary"]["target_context_point_count"] == 0
                for outcome in outcomes.values()
            ),
            "all_branches_settle_linear_speed": all(
                float(np.linalg.norm(outcome["final_target_state"][7:10]))
                <= SPEC["thresholds"]["final_linear_speed_max_m_s"]
                for outcome in outcomes.values()
            ),
            "all_branches_settle_angular_speed": all(
                float(np.linalg.norm(outcome["final_target_state"][10:13]))
                <= SPEC["thresholds"]["final_angular_speed_max_rad_s"]
                for outcome in outcomes.values()
            ),
        }
        checks.update(direction_checks(effects))
        checks.update(negative_controls(outcomes, effects))
        asset_after = assert_offline_asset_gate()
        checks["asset_gate_unchanged"] = asset_after == asset_before
        local_verdict = "supported" if all(checks.values()) else "rejected"
        stable_evidence = {
            "spec": SPEC,
            "spec_sha256": digest(SPEC),
            "producer_sources": producer_sources(lock_path),
            "versions": versions,
            "backend": backend,
            "asset_gate": {
                "empty": asset_after["empty"],
                "mode_octal": asset_after["mode_octal"],
            },
            "source_hashes": source_hashes,
            "source_target_state": source_target,
            "outcomes": outcomes,
            "paired_effects": effects,
            "checks": checks,
            "local_verdict": local_verdict,
            "claim_boundary": (
                "isolated offline programmatic physx_cpu five-branch runtime smoke only; "
                "no writer, formal cohort, external asset, RGB/GPU renderer, model, "
                "training, Gaussian dynamics, robot controller, causal, or planning claim"
            ),
        }
        return {
            **stable_evidence,
            "evidence_sha256": digest(stable_evidence),
            "execution_order": execution_order,
        }
    finally:
        env.close()


def compare_reports(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {
        "previous_evidence_sha256": previous.get("evidence_sha256"),
        "matches": previous.get("evidence_sha256") == current.get("evidence_sha256"),
        "opposite_execution_orders": previous.get("execution_order")
        == list(reversed(current.get("execution_order", []))),
        "previous_baseline_valid": previous.get("verdict") == "pending_repeat"
        and previous.get("local_verdict") == "supported"
        and previous.get("runtime_telemetry", {}).get("within_budget") is True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--order", choices=("canonical", "reverse"), default="canonical")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    exit_code = 0
    try:
        report = run_smoke(args.order, args.lock.resolve())
        if report["local_verdict"] != "supported":
            report["verdict"] = "rejected"
            exit_code = 2
        elif args.compare is None:
            report["verdict"] = "pending_repeat"
        else:
            previous = json.loads(args.compare.read_text(encoding="utf-8"))
            comparison = compare_reports(previous, report)
            report["repeat_comparison"] = comparison
            comparison_passes = (
                comparison["matches"]
                and comparison["opposite_execution_orders"]
                and comparison["previous_baseline_valid"]
            )
            report["verdict"] = "supported" if comparison_passes else "rejected"
            exit_code = 0 if report["verdict"] == "supported" else 2
    except Exception as error:
        report = {
            "spec": ERROR_SPEC,
            "spec_sha256": hashlib.sha256(
                json.dumps(ERROR_SPEC, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "verdict": "invalid",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        exit_code = 4

    report["runtime_telemetry"] = {
        "wall_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    within_budget = (
        report["runtime_telemetry"]["wall_seconds"] <= 15 * 60
        and report["runtime_telemetry"]["max_rss_kib"] <= 8 * 1024 * 1024
    )
    report["runtime_telemetry"]["within_budget"] = within_budget
    if not within_budget:
        report["verdict"] = "invalid"
        exit_code = 4

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
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
