"""Programmatic collision-only ManiSkill task used by the PR-01B smoke."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import SimConfig

from .canonical import (
    capture_snapshot,
    clone_value,
    digest,
    restore_rng,
    snapshot_hashes,
)


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
    "smoke_id": "objgauss-pr01-five-branch-runtime-v0",
    "smoke_version": "0.1.0",
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
    "changed_variable": "/intervention/commanded_action",
    "thresholds": {
        "hold_horizontal_drift_max_m": 0.002,
        "directional_effect_min_m": 0.005,
        "strong_over_weak_min_m": 0.005,
        "final_linear_speed_max_m_s": 0.01,
        "final_angular_speed_max_rad_s": 0.01,
    },
}


class PrimitiveActionEnv(BaseEnv):
    """Floor and two boxes; no robot, sensor, renderer, or external asset."""

    def __init__(self, scene_spec: dict[str, Any] | None = None) -> None:
        self.scene_spec = copy.deepcopy(scene_spec or {})
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

    def _load_lighting(self, options: dict[str, Any]) -> None:
        return None

    def _load_scene(self, options: dict[str, Any]) -> None:
        target_spec = self.scene_spec.get("target", SPEC["actors"]["target"])
        context_spec = self.scene_spec.get("context", SPEC["actors"]["context"])
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
            half_size=target_spec["half_size_m"],
            material=material,
            density=target_spec["density_kg_m3"],
        )
        target_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, target_spec["half_size_m"][2]])
        self.target = target_builder.build_dynamic(name="target")

        context_builder = self.scene.create_actor_builder()
        context_builder.add_box_collision(
            half_size=context_spec["half_size_m"],
            material=material,
            density=context_spec["density_kg_m3"],
        )
        context_xy = self.scene_spec.get("context_xy_m", [0.55, 0.55])
        context_builder.initial_pose = sapien.Pose(
            p=[context_xy[0], context_xy[1], context_spec["half_size_m"][2]]
        )
        self.context = context_builder.build_dynamic(name="context")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        if len(env_idx) != 1:
            raise RuntimeError("PR-01 smoke requires exactly one environment")
        target_spec = self.scene_spec.get("target", SPEC["actors"]["target"])
        context_spec = self.scene_spec.get("context", SPEC["actors"]["context"])
        target_base_xy = np.asarray(self.scene_spec.get("target_base_xy_m", [0.0, 0.0]))
        jitter_m = float(self.scene_spec.get("target_jitter_m", 0.02))
        target_xy = target_base_xy + self._batched_episode_rng[env_idx].uniform(
            -jitter_m, jitter_m, size=2
        )
        target_state = torch.zeros((1, 13), dtype=torch.float32, device=self.device)
        target_state[:, :2] = torch.as_tensor(
            target_xy, dtype=torch.float32, device=self.device
        )
        target_state[:, 2] = target_spec["half_size_m"][2]
        target_state[:, 3] = 1.0
        self.target.set_state(target_state, env_idx)

        context_state = torch.zeros((1, 13), dtype=torch.float32, device=self.device)
        context_xy = self.scene_spec.get("context_xy_m", [0.55, 0.55])
        context_state[:, :3] = torch.tensor(
            [[context_xy[0], context_xy[1], context_spec["half_size_m"][2]]],
            dtype=torch.float32,
            device=self.device,
        )
        context_state[:, 3] = 1.0
        self.context.set_state(context_state, env_idx)

    def get_state_dict(self) -> dict[str, Any]:
        return self.scene.get_sim_state()

    def set_state_dict(
        self, state: dict[str, Any], env_idx: torch.Tensor | None = None
    ) -> None:
        self.scene.set_sim_state(state, env_idx)


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


def actor_record(actor: Any) -> dict[str, list[float]]:
    state = actor_state(actor)
    return {
        "position_W_m": state[:3],
        "quaternion_WO_wxyz": state[3:7],
        "linear_velocity_W_m_s": state[7:10],
        "angular_velocity_W_rad_s": state[10:13],
    }


def trajectory_record(
    env: PrimitiveActionEnv, phase: str, phase_step: int, episode_time_s: float
) -> dict[str, Any]:
    return {
        "episode_time_s": episode_time_s,
        "phase": phase,
        "phase_step": phase_step,
        "actors": {
            "target": actor_record(env.target),
            "context": actor_record(env.context),
        },
    }


def _entity_name(body: Any) -> str:
    name = str(body.entity.name)
    prefix = "scene-0_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _vector(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("non-finite contact value")
    return [float(item) for item in array]


def contact_step(
    env: PrimitiveActionEnv, phase: str, step: int, episode_time_s: float
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for contact in env.scene.get_contacts():
        pair = sorted(_entity_name(body) for body in contact.bodies)
        if "target" not in pair:
            continue
        points = [
            {
                "position_W_m": _vector(point.position),
                "normal_W": _vector(point.normal),
                "impulse_W_N_s": _vector(point.impulse),
                "separation_m": float(point.separation),
            }
            for point in contact.points
        ]
        points.sort(key=digest)
        records.append({"body_pair": pair, "points": points})
    records.sort(key=digest)
    return {
        "episode_time_s": episode_time_s,
        "phase": phase,
        "phase_step": step,
        "contacts": records,
    }


def summarize_contacts(trace: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
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
                    np.linalg.norm(point["impulse_W_N_s"])
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


def run_branch(
    env: PrimitiveActionEnv,
    source: dict[str, Any],
    branch_id: str,
    branch_index: int,
    *,
    capture_artifacts: bool = False,
    reset_seed: int = SEED,
) -> dict[str, Any]:
    env.reset(
        seed=reset_seed + 1_000 + branch_index,
        options={"reset_to_env_states": {"env_states": source["physical_state"]}},
    )
    restore_rng(env, source["rng_state"])
    pre_action = capture_snapshot(env)
    pre_target_state = actor_state(env.target)
    command = commanded_action(branch_id)

    trace: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = [
        trajectory_record(env, "pre_action", 0, 0.0)
    ]
    applied_forces: list[list[float]] = []
    for step in range(ACTION_STEPS):
        applied_force = [float(value) for value in FORCES_N[branch_id]]
        env.target.apply_force(applied_force)
        applied_forces.append(applied_force)
        env.scene.step()
        episode_time_s = (step + 1) / SIM_FREQUENCY_HZ
        trace.append(contact_step(env, "action", step, episode_time_s))
        if capture_artifacts:
            trajectory.append(
                trajectory_record(env, "action", step, episode_time_s)
            )
    for step in range(SETTLING_STEPS):
        env.scene.step()
        episode_time_s = (ACTION_STEPS + step + 1) / SIM_FREQUENCY_HZ
        trace.append(contact_step(env, "settling", step, episode_time_s))
        if capture_artifacts:
            trajectory.append(
                trajectory_record(env, "settling", step, episode_time_s)
            )

    executed = {
        "kind": "external_force_at_target_center_of_mass",
        "force_n": applied_forces[0],
        "sim_frequency_hz": SIM_FREQUENCY_HZ,
        "action_steps": len(applied_forces),
        "action_duration_s": len(applied_forces) / SIM_FREQUENCY_HZ,
        "settling_steps": SETTLING_STEPS,
        "settling_duration_s": SETTLING_STEPS / SIM_FREQUENCY_HZ,
    }
    if any(force != executed["force_n"] for force in applied_forces):
        raise RuntimeError(f"non-constant executed force ledger for {branch_id}")
    final_physical_state = clone_value(env.get_state_dict())
    outcome = {
        "branch_id": branch_id,
        "changed_variable": SPEC["changed_variable"],
        "pre_action_hashes": snapshot_hashes(pre_action),
        "pre_action_target_state": pre_target_state,
        "commanded_action": command,
        "executed_action": executed,
        "executed_action_sha256": digest(executed),
        "final_physical_state_sha256": digest(final_physical_state),
        "final_target_state": actor_state(env.target),
        "contact_trace_sha256": digest(trace),
        "contact_summary": summarize_contacts(trace),
    }
    if capture_artifacts:
        outcome["trajectory_records"] = trajectory
        outcome["contact_records"] = trace
    return outcome


def paired_effects(
    outcomes: dict[str, dict[str, Any]], source_target: list[float]
) -> dict[str, Any]:
    hold_position = np.asarray(outcomes["hold"]["final_target_state"][:3])
    source_position = np.asarray(source_target[:3])
    return {
        "hold_horizontal_drift_m": float(
            np.linalg.norm((hold_position - source_position)[:2])
        ),
        "paired_displacement_m": {
            branch_id: [
                float(value)
                for value in np.asarray(outcomes[branch_id]["final_target_state"][:3])
                - hold_position
            ]
            for branch_id in SIBLINGS
        },
    }


def direction_checks(effects: dict[str, Any]) -> dict[str, bool]:
    threshold = SPEC["thresholds"]["directional_effect_min_m"]
    strong_margin = SPEC["thresholds"]["strong_over_weak_min_m"]
    displacement = effects["paired_displacement_m"]
    weak_x = displacement["push_pos_x_weak"][0]
    strong_x = displacement["push_pos_x_strong"][0]
    return {
        "hold_horizontal_drift_within_limit": effects["hold_horizontal_drift_m"]
        <= SPEC["thresholds"]["hold_horizontal_drift_max_m"],
        "push_pos_x_weak_direction_and_effect": weak_x >= threshold,
        "push_pos_x_strong_direction_and_effect": strong_x >= threshold,
        "push_pos_x_strong_exceeds_weak": strong_x - weak_x >= strong_margin,
        "push_neg_x_weak_direction_and_effect": displacement["push_neg_x_weak"][0]
        <= -threshold,
        "push_pos_y_weak_direction_and_effect": displacement["push_pos_y_weak"][1]
        >= threshold,
    }


def negative_controls(
    outcomes: dict[str, dict[str, Any]], effects: dict[str, Any]
) -> dict[str, bool]:
    sign_control = copy.deepcopy(effects)
    sign_control["paired_displacement_m"]["push_pos_x_weak"][0] = -0.01
    ledger_control = copy.deepcopy(outcomes["push_pos_x_weak"]["executed_action"])
    ledger_control["action_steps"] -= 1
    return {
        "direction_evaluator_rejects_sign_flip": not direction_checks(sign_control)[
            "push_pos_x_weak_direction_and_effect"
        ],
        "ledger_evaluator_rejects_missing_step": digest(ledger_control)
        != digest(outcomes["push_pos_x_weak"]["commanded_action"]),
    }
