#!/usr/bin/env python3
"""RES-001 state-only snapshot/reset feasibility pilot for ManiSkill 3.0.1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import stat
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
from mani_skill.utils.structs.types import SimConfig


PILOT_VERSION = "0.1.0"
SEED = 24_071_401
SIBLINGS = (
    "hold",
    "push_pos_x_weak",
    "push_pos_x_strong",
    "push_neg_x_weak",
    "push_pos_y_weak",
)
SPEC = {
    "pilot_id": "res001-maniskill-snapshot-v0",
    "pilot_version": PILOT_VERSION,
    "seed": SEED,
    "sim_backend": "physx_cpu",
    "render_backend": "none",
    "num_envs": 1,
    "obs_mode": "none",
    "reward_mode": "none",
    "agent": None,
    "actors": {
        "target": {"half_size_m": [0.03, 0.02, 0.025], "density_kg_m3": 500.0},
        "context": {"half_size_m": [0.02, 0.025, 0.015], "density_kg_m3": 700.0},
    },
    "siblings": list(SIBLINGS),
    "primary_endpoint": (
        "pre-action SHA-256 over physical state plus ManiSkill main/episode RNG; "
        "all five sibling hashes must exactly equal the source snapshot hash"
    ),
}


class PrimitiveSnapshotEnv(BaseEnv):
    """Two collision-only actors and no agent, sensors, renderer, or external assets."""

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
        return SimConfig(sim_freq=100, control_freq=20)

    def _load_lighting(self, options: dict) -> None:
        # Rendering is deliberately disabled for this pilot.
        return None

    def _load_scene(self, options: dict) -> None:
        target_builder = self.scene.create_actor_builder()
        target_builder.add_box_collision(
            half_size=SPEC["actors"]["target"]["half_size_m"],
            density=SPEC["actors"]["target"]["density_kg_m3"],
        )
        target_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.30])
        self.target = target_builder.build_dynamic(name="target")

        context_builder = self.scene.create_actor_builder()
        context_builder.add_box_collision(
            half_size=SPEC["actors"]["context"]["half_size_m"],
            density=SPEC["actors"]["context"]["density_kg_m3"],
        )
        context_builder.initial_pose = sapien.Pose(p=[0.25, 0.0, 0.45])
        self.context = context_builder.build_dynamic(name="context")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        batch_size = len(env_idx)
        if batch_size != 1:
            raise RuntimeError("RES-001 pilot is preregistered for exactly one environment")

        # Consume the environment-owned episode RNG so a physical-state-only restore
        # has a detectable RNG gap. The full snapshot adapter must close that gap.
        target_xy = self._batched_episode_rng[env_idx].uniform(-0.05, 0.05, size=2)
        context_xy = self._batched_episode_rng[env_idx].uniform(-0.03, 0.03, size=2)
        velocities = self._batched_episode_rng[env_idx].uniform(-0.2, 0.2, size=6)

        target_state = torch.zeros((1, 13), dtype=torch.float32, device=self.device)
        target_state[:, :2] = torch.as_tensor(target_xy, dtype=torch.float32, device=self.device)
        target_state[:, 2] = 0.30
        target_state[:, 3] = 1.0
        target_state[:, 7:10] = torch.as_tensor(
            velocities[:, :3], dtype=torch.float32, device=self.device
        )
        target_state[:, 10:13] = torch.as_tensor(
            velocities[:, 3:], dtype=torch.float32, device=self.device
        )
        self.target.set_state(target_state, env_idx)

        context_state = torch.zeros((1, 13), dtype=torch.float32, device=self.device)
        context_state[:, :2] = torch.as_tensor(context_xy, dtype=torch.float32, device=self.device)
        context_state[:, 2] = 0.45
        context_state[:, 3] = 1.0
        self.context.set_state(context_state, env_idx)

    def get_state_dict(self) -> dict:
        # BaseEnv assumes an agent/controller. This no-agent task delegates directly
        # to ManiSkillScene while preserving the public task-level API.
        return self.scene.get_sim_state()

    def set_state_dict(self, state: dict, env_idx: torch.Tensor | None = None) -> None:
        self.scene.set_sim_state(state, env_idx)


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    return copy.deepcopy(value)


def canonical_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("non-finite array value")
        dtype = value.dtype.newbyteorder("<")
        array = np.ascontiguousarray(value.astype(dtype, copy=False))
        if np.issubdtype(array.dtype, np.floating):
            array = array.copy()
            array[array == 0] = 0
        return {
            "kind": "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_hex": array.tobytes(order="C").hex(),
        }
    if isinstance(value, np.generic):
        return canonical_value(value.item())
    if isinstance(value, dict):
        return {str(key): canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite scalar value")
        return 0.0 if value == 0 else value.hex()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def digest(value: Any) -> str:
    payload = json.dumps(
        canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def random_state_from(snapshot: tuple) -> np.random.RandomState:
    rng = np.random.RandomState()
    rng.set_state(clone_value(snapshot))
    return rng


def capture_rng(env: PrimitiveSnapshotEnv) -> dict:
    return {
        "main_seed": list(env._main_seed),
        "main_rng": clone_value(env._main_rng.get_state()),
        "batched_main_rng": [
            clone_value(rng.get_state()) for rng in env._batched_main_rng.rngs
        ],
        "episode_seed": clone_value(env._episode_seed),
        "batched_episode_rng": [
            clone_value(rng.get_state()) for rng in env._batched_episode_rng.rngs
        ],
    }


def restore_rng(env: PrimitiveSnapshotEnv, snapshot: dict) -> None:
    env._main_seed = list(snapshot["main_seed"])
    env._main_rng = random_state_from(snapshot["main_rng"])
    env._batched_main_rng = BatchedRNG.from_rngs(
        [random_state_from(state) for state in snapshot["batched_main_rng"]]
    )
    env._episode_seed = clone_value(snapshot["episode_seed"])
    env._batched_episode_rng = BatchedRNG.from_rngs(
        [random_state_from(state) for state in snapshot["batched_episode_rng"]]
    )
    env._episode_rng = env._batched_episode_rng[0]


def capture_snapshot(env: PrimitiveSnapshotEnv) -> dict:
    return {
        "physical_state": clone_value(env.get_state_dict()),
        "rng_state": capture_rng(env),
    }


def snapshot_hashes(snapshot: dict) -> dict[str, str]:
    return {
        "physical": digest(snapshot["physical_state"]),
        "rng": digest(snapshot["rng_state"]),
        "full": digest(snapshot),
    }


def assert_asset_gate() -> dict:
    if os.getenv("MS_SKIP_ASSET_DOWNLOAD_PROMPT") is not None:
        raise RuntimeError("MS_SKIP_ASSET_DOWNLOAD_PROMPT must be unset")
    raw_path = os.getenv("MS_ASSET_DIR")
    if not raw_path:
        raise RuntimeError("MS_ASSET_DIR must point to the preregistered no-assets directory")
    asset_dir = Path(raw_path).resolve()
    if not asset_dir.is_dir():
        raise RuntimeError(f"asset gate directory is missing: {asset_dir}")
    entries = list(asset_dir.iterdir())
    if entries:
        raise RuntimeError(f"asset gate is not empty: {[entry.name for entry in entries]}")
    mode = stat.S_IMODE(asset_dir.stat().st_mode)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(f"asset gate must be read-only, got mode {mode:o}")
    return {"empty": True, "mode_octal": f"{mode:o}"}


def observed_backend(env: PrimitiveSnapshotEnv) -> dict:
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


def run_pilot() -> dict:
    asset_before = assert_asset_gate()
    versions = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "mani_skill": importlib.metadata.version("mani-skill"),
        "sapien": importlib.metadata.version("sapien"),
        "torch": torch.__version__,
    }
    required_versions = {
        "python": "3.10.20",
        "mani_skill": "3.0.1",
        "sapien": "3.0.3",
        "torch": "2.13.0+cu130",
    }
    if versions != required_versions:
        raise RuntimeError(f"runtime version drift: expected {required_versions}, got {versions}")

    env = PrimitiveSnapshotEnv()
    try:
        backend = observed_backend(env)
        env.reset(seed=SEED)
        source = capture_snapshot(env)
        source_hashes = snapshot_hashes(source)

        env.reset(seed=SEED)
        same_seed = capture_snapshot(env)
        same_seed_hashes = snapshot_hashes(same_seed)

        env.reset(seed=SEED + 1)
        different_seed = capture_snapshot(env)
        different_seed_hashes = snapshot_hashes(different_seed)

        mutated_state = clone_value(source["physical_state"])
        mutated_target = mutated_state["actors"]["target"]
        mutated_target[0, 0] += 0.125
        mutated_target[0, 7] += 0.375
        env.set_state_dict(mutated_state)
        mutated = capture_snapshot(env)
        mutated_hashes = snapshot_hashes(mutated)

        env.set_state_dict(source["physical_state"])
        set_restored = capture_snapshot(env)
        set_restored_hashes = snapshot_hashes(set_restored)

        env.reset(
            seed=SEED + 99,
            options={"reset_to_env_states": {"env_states": source["physical_state"]}},
        )
        reset_state_only = capture_snapshot(env)
        reset_state_only_hashes = snapshot_hashes(reset_state_only)
        restore_rng(env, source["rng_state"])
        reset_full = capture_snapshot(env)
        reset_full_hashes = snapshot_hashes(reset_full)

        sibling_hashes: dict[str, dict[str, str]] = {}
        for index, sibling in enumerate(SIBLINGS):
            env.reset(
                seed=SEED + 1_000 + index,
                options={
                    "reset_to_env_states": {"env_states": source["physical_state"]}
                },
            )
            restore_rng(env, source["rng_state"])
            sibling_hashes[sibling] = snapshot_hashes(capture_snapshot(env))

        checks = {
            "fixed_versions": versions == required_versions,
            "no_agent": backend["agent_is_none"],
            "no_sensors": backend["sensor_count"] == 0,
            "no_human_render_cameras": backend["human_render_camera_count"] == 0,
            "rendering_disabled": backend["scene_can_render"] is False,
            "registered_actors_exact": backend["actors"] == ["context", "target"],
            "same_seed_full_repeat": same_seed_hashes["full"] == source_hashes["full"],
            "different_seed_negative_control": (
                different_seed_hashes["full"] != source_hashes["full"]
            ),
            "mutation_negative_control": mutated_hashes["physical"]
            != source_hashes["physical"],
            "set_state_dict_physical_roundtrip": (
                set_restored_hashes["physical"] == source_hashes["physical"]
            ),
            "reset_to_env_states_physical_roundtrip": (
                reset_state_only_hashes["physical"] == source_hashes["physical"]
            ),
            "state_only_restore_exposes_rng_gap": (
                reset_state_only_hashes["full"] != source_hashes["full"]
            ),
            "explicit_rng_restore_full_roundtrip": (
                reset_full_hashes["full"] == source_hashes["full"]
            ),
            "five_sibling_full_hashes_equal_source": all(
                hashes["full"] == source_hashes["full"]
                for hashes in sibling_hashes.values()
            ),
            "five_sibling_hashes_present": tuple(sibling_hashes) == SIBLINGS,
        }
        asset_after = assert_asset_gate()
        checks["asset_gate_unchanged"] = asset_after == asset_before

        verdict = "supported" if all(checks.values()) else "rejected"
        stable_evidence = {
            "spec": SPEC,
            "spec_sha256": digest(SPEC),
            "versions": versions,
            "backend": backend,
            "asset_gate": asset_after,
            "source_hashes": source_hashes,
            "same_seed_hashes": same_seed_hashes,
            "different_seed_hashes": different_seed_hashes,
            "mutated_hashes": mutated_hashes,
            "set_restored_hashes": set_restored_hashes,
            "reset_state_only_hashes": reset_state_only_hashes,
            "reset_full_hashes": reset_full_hashes,
            "sibling_hashes": sibling_hashes,
            "checks": checks,
            "verdict": verdict,
            "claim_boundary": (
                "snapshot fork feasibility only; no action, contact, rendering, dataset, "
                "training, dynamics, causal, or planning claim"
            ),
        }
        return {
            **stable_evidence,
            "evidence_sha256": digest(stable_evidence),
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/res001/evidence/snapshot-pilot.json"),
    )
    parser.add_argument("--compare", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    report: dict[str, Any]
    exit_code = 0
    try:
        report = run_pilot()
        if args.compare is not None:
            previous = json.loads(args.compare.read_text(encoding="utf-8"))
            matches = previous.get("evidence_sha256") == report["evidence_sha256"]
            report["repeat_comparison"] = {
                "previous_evidence_sha256": previous.get("evidence_sha256"),
                "matches": matches,
            }
            if not matches:
                report["verdict"] = "rejected"
                exit_code = 1
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
