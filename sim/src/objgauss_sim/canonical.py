"""Canonical serialization and ManiSkill snapshot helpers for PR-01."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG


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


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def random_state_from(snapshot: tuple[Any, ...]) -> np.random.RandomState:
    rng = np.random.RandomState()
    rng.set_state(clone_value(snapshot))
    return rng


def capture_rng(env: Any) -> dict[str, Any]:
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


def restore_rng(env: Any, snapshot: dict[str, Any]) -> None:
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


def capture_snapshot(env: Any) -> dict[str, Any]:
    return {
        "physical_state": clone_value(env.get_state_dict()),
        "rng_state": capture_rng(env),
    }


def snapshot_hashes(snapshot: dict[str, Any]) -> dict[str, str]:
    return {
        "physical": digest(snapshot["physical_state"]),
        "rng": digest(snapshot["rng_state"]),
        "full": digest(snapshot),
    }
