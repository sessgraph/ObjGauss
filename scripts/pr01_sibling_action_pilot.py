#!/usr/bin/env python3
"""PR-01 programmatic sibling action/contact pilot for ManiSkill 3.0.1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import SimConfig

from res001_snapshot_pilot import (
    assert_asset_gate,
    capture_snapshot,
    clone_value,
    digest,
    restore_rng,
    snapshot_hashes,
)


PILOT_VERSION = "0.1.0"
SEED = 24_071_401
SIM_FREQUENCY_HZ = 100
WARMUP_STEPS = 20
ACTION_STEPS = 10
SETTLING_STEPS = 100
SIBLINGS = (
    "hold",
    "push_pos_x_weak",
    "push_pos_x_strong",
    "push_neg_x_weak",
    "push_pos_y_weak",
)
FORCES_N = {
    "hold": [0.0, 0.0, 0.0],
    "push_pos_x_weak": [0.35, 0.0, 0.0],
    "push_pos_x_strong": [0.70, 0.0, 0.0],
    "push_neg_x_weak": [-0.35, 0.0, 0.0],
    "push_pos_y_weak": [0.0, 0.35, 0.0],
}
SPEC = {
    "pilot_id": "pr01-maniskill-sibling-action-v0",
    "pilot_version": PILOT_VERSION,
    "seed": SEED,
    "sim_backend": "physx_cpu",
    "render_backend": "none",
    "num_envs": 1,
    "obs_mode": "none",
    "reward_mode": "none",
    "agent": None,
    "sim_frequency_hz": SIM_FREQUENCY_HZ,
    "warmup_steps": WARMUP_STEPS,
    "action_steps": ACTION_STEPS,
    "settling_steps": SETTLING_STEPS,
    "material": {
        "static_friction": 0.5,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
    },
    "actors": {
        "floor": {"half_size_m": [1.0, 1.0, 0.05], "top_z_m": 0.0},
        "target": {"half_size_m": [0.03, 0.02, 0.025], "density_kg_m3": 500.0},
        "context": {"half_size_m": [0.02, 0.025, 0.015], "density_kg_m3": 700.0},
    },
    "siblings": [
        {"branch_id": branch_id, "force_n": FORCES_N[branch_id]}
        for branch_id in SIBLINGS
    ],
    "thresholds": {
        "hold_horizontal_drift_max_m": 0.002,
        "directional_effect_min_m": 0.005,
        "strong_over_weak_min_m": 0.005,
        "final_linear_speed_max_m_s": 0.01,
        "final_angular_speed_max_rad_s": 0.01,
    },
    "primary_endpoint": (
        "canonical and reverse process runs must exactly match each branch's pre-action "
        "full snapshot, executed-action ledger, final physical state, and target-related "
        "contact trace hashes; every non-hold branch must also pass its preregistered "
        "paired direction/effect threshold"
    ),
}


class PrimitiveActionEnv(BaseEnv):
    """Collision-only floor, target, and context without agent, renderer, or assets."""

    def __init__(self) -> None:
        super().__init__(
            num_envs=1,
            obs_mode="none",
            reward_mode="none",
            robot_uids="none",
            sim_backend="physx_cpu",
            render_backend="none",
            enhanced_determinism=True,
        )

    @property
    def _default_sim_config(self) -> SimConfig:
        return SimConfig(sim_freq=SIM_FREQUENCY_HZ, control_freq=20)

    def _load_lighting(self, options: dict) -> None:
        return None

    def _load_scene(self, options: dict) -> None:
        material = sapien.physx.PhysxMaterial(
            SPEC["material"]["static_friction"],
            SPEC["material"]["dynamic_friction"],
            SPEC["material"]["restitution"],
        )

        floor_builder = self.scene.create_actor_builder()
        floor_builder.add_box_collision(
            half_size=SPEC["actors"]["floor"]["half_size_m"], material=material
        )
        floor_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -0.05])
        self.floor = floor_builder.build_static(name="floor")

        target_builder = self.scene.create_actor_builder()
        target_builder.add_box_collision(
            half_size=SPEC["actors"]["target"]["half_size_m"],
            material=material,
            density=SPEC["actors"]["target"]["density_kg_m3"],
        )
        target_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.025])
        self.target = target_builder.build_dynamic(name="target")

        context_builder = self.scene.create_actor_builder()
        context_builder.add_box_collision(
            half_size=SPEC["actors"]["context"]["half_size_m"],
            material=material,
            density=SPEC["actors"]["context"]["density_kg_m3"],
        )
        context_builder.initial_pose = sapien.Pose(p=[0.55, 0.55, 0.015])
        self.context = context_builder.build_dynamic(name="context")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        if len(env_idx) != 1:
            raise RuntimeError("PR-01 pilot is preregistered for exactly one environment")

        target_xy = self._batched_episode_rng[env_idx].uniform(-0.02, 0.02, size=2)
        target_state = torch.zeros((1, 13), dtype=torch.float32, device=self.device)
        target_state[:, :2] = torch.as_tensor(target_xy, dtype=torch.float32, device=self.device)
        target_state[:, 2] = SPEC["actors"]["target"]["half_size_m"][2]
        target_state[:, 3] = 1.0
        self.target.set_state(target_state, env_idx)

        context_state = torch.zeros((1, 13), dtype=torch.float32, device=self.device)
        context_state[:, :3] = torch.tensor(
            [[0.55, 0.55, SPEC["actors"]["context"]["half_size_m"][2]]],
            dtype=torch.float32,
            device=self.device,
        )
        context_state[:, 3] = 1.0
        self.context.set_state(context_state, env_idx)

    def get_state_dict(self) -> dict:
        return self.scene.get_sim_state()

    def set_state_dict(self, state: dict, env_idx: torch.Tensor | None = None) -> None:
        self.scene.set_sim_state(state, env_idx)


def required_versions() -> dict[str, str]:
    return {
        "python": "3.10.20",
        "mani_skill": "3.0.1",
        "sapien": "3.0.3",
        "torch": "2.13.0+cu130",
    }


def observed_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "mani_skill": importlib.metadata.version("mani-skill"),
        "sapien": importlib.metadata.version("sapien"),
        "torch": str(torch.__version__),
    }


def producer_sources() -> dict[str, str]:
    current = Path(__file__).resolve()
    snapshot_helper = current.with_name("res001_snapshot_pilot.py")
    return {
        current.name: hashlib.sha256(current.read_bytes()).hexdigest(),
        snapshot_helper.name: hashlib.sha256(snapshot_helper.read_bytes()).hexdigest(),
    }


def observed_backend(env: PrimitiveActionEnv) -> dict[str, Any]:
    return {
        "sim_backend": str(env.backend.sim_backend),
        "render_backend": str(env.backend.render_backend),
        "device": str(env.device),
        "agent_is_none": env.agent is None,
        "sensor_count": len(env._sensors),
        "human_render_camera_count": len(env._human_render_cameras),
        "scene_can_render": bool(env.scene.can_render()),
        "actors": sorted(env.scene.state_dict_registry.actors),
    }


def actor_state(actor: Any) -> list[float]:
    raw = actor.get_state().detach().cpu().numpy()[0]
    if not np.isfinite(raw).all():
        raise ValueError(f"non-finite actor state for {actor.name}")
    return [float(value) for value in raw]


def entity_name(body: Any) -> str:
    name = str(body.entity.name)
    prefix = "scene-0_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def vector(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("non-finite contact value")
    return [float(item) for item in array]


def target_contact_step(env: PrimitiveActionEnv, phase: str, step: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for contact in env.scene.get_contacts():
        pair = sorted(entity_name(body) for body in contact.bodies)
        if "target" not in pair:
            continue
        points = [
            {
                "position_m": vector(point.position),
                "normal": vector(point.normal),
                "impulse_n_s": vector(point.impulse),
                "separation_m": float(point.separation),
            }
            for point in contact.points
        ]
        points.sort(key=lambda item: digest(item))
        records.append({"body_pair": pair, "points": points})
    records.sort(key=lambda item: digest(item))
    return {"phase": phase, "step": step, "contacts": records}


def summarize_contacts(trace: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "target_floor_contact_steps": 0,
        "target_floor_contact_count": 0,
        "target_floor_point_count": 0,
        "target_context_contact_count": 0,
        "target_context_point_count": 0,
        "target_related_impulse_norm_sum_n_s": 0.0,
    }
    for step in trace:
        floor_on_step = False
        for contact in step["contacts"]:
            pair = contact["body_pair"]
            point_count = len(contact["points"])
            if pair == ["floor", "target"]:
                floor_on_step = True
                summary["target_floor_contact_count"] += 1
                summary["target_floor_point_count"] += point_count
            if pair == ["context", "target"]:
                summary["target_context_contact_count"] += 1
                summary["target_context_point_count"] += point_count
            for point in contact["points"]:
                summary["target_related_impulse_norm_sum_n_s"] += float(
                    np.linalg.norm(point["impulse_n_s"])
                )
        if floor_on_step:
            summary["target_floor_contact_steps"] += 1
    return summary


def commanded_action(branch_id: str) -> dict[str, Any]:
    return {
        "kind": "external_force_at_target_center_of_mass",
        "force_n": FORCES_N[branch_id],
        "sim_frequency_hz": SIM_FREQUENCY_HZ,
        "action_steps": ACTION_STEPS,
        "action_duration_s": ACTION_STEPS / SIM_FREQUENCY_HZ,
        "settling_steps": SETTLING_STEPS,
        "settling_duration_s": SETTLING_STEPS / SIM_FREQUENCY_HZ,
    }


def ledger_matches(command: dict[str, Any], executed: dict[str, Any]) -> bool:
    return digest(command) == digest(executed)


def run_branch(
    env: PrimitiveActionEnv, source: dict[str, Any], branch_id: str, branch_index: int
) -> dict[str, Any]:
    env.reset(
        seed=SEED + 1_000 + branch_index,
        options={"reset_to_env_states": {"env_states": source["physical_state"]}},
    )
    restore_rng(env, source["rng_state"])
    pre_action = capture_snapshot(env)
    pre_target_state = actor_state(env.target)
    command = commanded_action(branch_id)

    trace: list[dict[str, Any]] = []
    applied_forces: list[list[float]] = []
    for step in range(ACTION_STEPS):
        applied_force = [float(value) for value in FORCES_N[branch_id]]
        env.target.apply_force(applied_force)
        applied_forces.append(applied_force)
        env.scene.step()
        trace.append(target_contact_step(env, "action", step))
    settled_steps = 0
    for step in range(SETTLING_STEPS):
        env.scene.step()
        settled_steps += 1
        trace.append(target_contact_step(env, "settling", step))

    executed = {
        "kind": "external_force_at_target_center_of_mass",
        "force_n": applied_forces[0] if applied_forces else None,
        "sim_frequency_hz": SIM_FREQUENCY_HZ,
        "action_steps": len(applied_forces),
        "action_duration_s": len(applied_forces) / SIM_FREQUENCY_HZ,
        "settling_steps": settled_steps,
        "settling_duration_s": settled_steps / SIM_FREQUENCY_HZ,
    }
    if any(force != executed["force_n"] for force in applied_forces):
        raise RuntimeError(f"non-constant executed force ledger for {branch_id}")
    final_physical_state = clone_value(env.get_state_dict())
    final_target_state = actor_state(env.target)
    return {
        "branch_id": branch_id,
        "changed_variable": "target_external_force_n",
        "pre_action_hashes": snapshot_hashes(pre_action),
        "pre_action_target_state": pre_target_state,
        "commanded_action": command,
        "executed_action": executed,
        "executed_action_sha256": digest(executed),
        "final_physical_state_sha256": digest(final_physical_state),
        "final_target_state": final_target_state,
        "contact_trace_sha256": digest(trace),
        "contact_summary": summarize_contacts(trace),
    }


def paired_effects(outcomes: dict[str, dict[str, Any]], source_target: list[float]) -> dict[str, Any]:
    hold_position = np.asarray(outcomes["hold"]["final_target_state"][:3], dtype=np.float64)
    source_position = np.asarray(source_target[:3], dtype=np.float64)
    effects: dict[str, Any] = {
        "hold_horizontal_drift_m": float(np.linalg.norm((hold_position - source_position)[:2])),
        "paired_displacement_m": {},
    }
    for branch_id in SIBLINGS:
        final_position = np.asarray(outcomes[branch_id]["final_target_state"][:3], dtype=np.float64)
        effects["paired_displacement_m"][branch_id] = [
            float(value) for value in final_position - hold_position
        ]
    return effects


def direction_checks(effects: dict[str, Any]) -> dict[str, bool]:
    threshold = SPEC["thresholds"]["directional_effect_min_m"]
    strong_margin = SPEC["thresholds"]["strong_over_weak_min_m"]
    displacement = effects["paired_displacement_m"]
    weak_x = displacement["push_pos_x_weak"][0]
    strong_x = displacement["push_pos_x_strong"][0]
    return {
        "hold_horizontal_drift_within_limit": (
            effects["hold_horizontal_drift_m"]
            <= SPEC["thresholds"]["hold_horizontal_drift_max_m"]
        ),
        "push_pos_x_weak_direction_and_effect": weak_x >= threshold,
        "push_pos_x_strong_direction_and_effect": strong_x >= threshold,
        "push_pos_x_strong_exceeds_weak": strong_x - weak_x >= strong_margin,
        "push_neg_x_weak_direction_and_effect": displacement["push_neg_x_weak"][0]
        <= -threshold,
        "push_pos_y_weak_direction_and_effect": displacement["push_pos_y_weak"][1]
        >= threshold,
    }


def run_pilot(order: str) -> dict[str, Any]:
    asset_before = assert_asset_gate()
    versions = observed_versions()
    if versions != required_versions():
        raise RuntimeError(
            f"runtime version drift: expected {required_versions()}, got {versions}"
        )

    env = PrimitiveActionEnv()
    try:
        backend = observed_backend(env)
        env.reset(seed=SEED)
        for _ in range(WARMUP_STEPS):
            env.scene.step()
        source = capture_snapshot(env)
        source_hashes = snapshot_hashes(source)
        source_target = actor_state(env.target)

        execution_order = list(SIBLINGS)
        if order == "reverse":
            execution_order.reverse()
        outcomes: dict[str, dict[str, Any]] = {}
        for branch_id in execution_order:
            outcomes[branch_id] = run_branch(
                env, source, branch_id, SIBLINGS.index(branch_id)
            )

        effects = paired_effects(outcomes, source_target)
        checks: dict[str, bool] = {
            "fixed_versions": versions == required_versions(),
            "no_agent": backend["agent_is_none"],
            "no_sensors": backend["sensor_count"] == 0,
            "no_human_render_cameras": backend["human_render_camera_count"] == 0,
            "rendering_disabled": backend["scene_can_render"] is False,
            "registered_actors_exact": backend["actors"]
            == ["context", "floor", "target"],
            "all_siblings_present": sorted(outcomes) == sorted(SIBLINGS),
            "all_pre_action_full_hashes_equal_source": all(
                outcome["pre_action_hashes"]["full"] == source_hashes["full"]
                for outcome in outcomes.values()
            ),
            "all_pre_action_target_states_equal_source": all(
                digest(outcome["pre_action_target_state"]) == digest(source_target)
                for outcome in outcomes.values()
            ),
            "all_executed_ledgers_match_commands": all(
                ledger_matches(outcome["commanded_action"], outcome["executed_action"])
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

        sign_control = copy.deepcopy(effects)
        sign_control["paired_displacement_m"]["push_pos_x_weak"][0] = -0.01
        checks["direction_evaluator_rejects_sign_flip"] = not direction_checks(
            sign_control
        )["push_pos_x_weak_direction_and_effect"]
        ledger_control = copy.deepcopy(outcomes["push_pos_x_weak"]["executed_action"])
        ledger_control["action_steps"] -= 1
        checks["ledger_evaluator_rejects_missing_step"] = not ledger_matches(
            outcomes["push_pos_x_weak"]["commanded_action"], ledger_control
        )

        asset_after = assert_asset_gate()
        checks["asset_gate_unchanged"] = asset_after == asset_before
        local_verdict = "supported" if all(checks.values()) else "rejected"
        stable_evidence = {
            "spec": SPEC,
            "spec_sha256": digest(SPEC),
            "producer_sources": producer_sources(),
            "versions": versions,
            "backend": backend,
            "asset_gate": asset_after,
            "source_hashes": source_hashes,
            "source_target_state": source_target,
            "outcomes": outcomes,
            "paired_effects": effects,
            "checks": checks,
            "local_verdict": local_verdict,
            "claim_boundary": (
                "programmatic CPU external-force sibling source for the primitive push "
                "slice only; no robot controller, renderer, GPU simulator, external asset, "
                "adapter/schema writer, dataset, training, Gaussian dynamics, causal, or "
                "planning claim"
            ),
        }
        return {
            **stable_evidence,
            "evidence_sha256": digest(stable_evidence),
            "execution_order": execution_order,
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/res001/evidence/sibling-action-pilot.json"),
    )
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--order", choices=("canonical", "reverse"), default="canonical")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    report: dict[str, Any]
    exit_code = 0
    try:
        report = run_pilot(args.order)
        if report["local_verdict"] != "supported":
            report["verdict"] = "rejected"
            exit_code = 1
        elif args.compare is None:
            report["verdict"] = "pending_repeat"
        else:
            previous = json.loads(args.compare.read_text(encoding="utf-8"))
            matches = previous.get("evidence_sha256") == report["evidence_sha256"]
            opposite_orders = previous.get("execution_order") == list(
                reversed(report["execution_order"])
            )
            previous_baseline_valid = (
                previous.get("verdict") == "pending_repeat"
                and previous.get("local_verdict") == "supported"
                and previous.get("runtime_telemetry", {}).get(
                    "within_preregistered_budget"
                )
                is True
            )
            report["repeat_comparison"] = {
                "previous_evidence_sha256": previous.get("evidence_sha256"),
                "matches": matches,
                "opposite_execution_orders": opposite_orders,
                "previous_baseline_valid": previous_baseline_valid,
            }
            report["verdict"] = (
                "supported"
                if matches and opposite_orders and previous_baseline_valid
                else "rejected"
            )
            if report["verdict"] != "supported":
                exit_code = 1
    except Exception as error:
        report = {
            "spec": SPEC,
            "spec_sha256": digest(SPEC),
            "verdict": "invalid",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        exit_code = 1

    report["runtime_telemetry"] = {
        "wall_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    telemetry_ok = (
        report["runtime_telemetry"]["wall_seconds"] <= 15 * 60
        and report["runtime_telemetry"]["max_rss_kib"] <= 8 * 1024 * 1024
    )
    report["runtime_telemetry"]["within_preregistered_budget"] = telemetry_ok
    if not telemetry_ok:
        report["verdict"] = "invalid"
        exit_code = 1

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
