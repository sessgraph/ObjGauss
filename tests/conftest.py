from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io import write_ply


@dataclass(frozen=True)
class ProceduralRealSampleV2Scenario:
    name: str
    cloud: GaussianCloud

    @property
    def source(self) -> str:
        return f"fixture://real-sample-v2/{self.name}"

    def write(self, directory: Path) -> Path:
        path = directory / f"{self.name}.ply"
        write_ply(path, self.cloud, fmt="ascii")
        return path


class ProceduralRealSampleV2Scenarios:
    def __init__(self) -> None:
        self._scenarios = {
            "temperature-boundary": ProceduralRealSampleV2Scenario(
                "temperature-boundary",
                _temperature_boundary_cloud(),
            ),
            "coverage-boundary": ProceduralRealSampleV2Scenario(
                "coverage-boundary",
                _coverage_boundary_cloud(),
            ),
            "weight-fix": ProceduralRealSampleV2Scenario(
                "weight-fix",
                _weight_boundary_cloud(mode="fix", boundary_per_slot=16),
            ),
            "weight-regression-a": ProceduralRealSampleV2Scenario(
                "weight-regression-a",
                _weight_boundary_cloud(mode="regress", boundary_per_slot=52),
            ),
            "weight-regression-b": ProceduralRealSampleV2Scenario(
                "weight-regression-b",
                _weight_boundary_cloud(mode="regress", boundary_per_slot=68),
            ),
            "no-safe-candidate": ProceduralRealSampleV2Scenario(
                "no-safe-candidate",
                _no_safe_candidate_cloud(),
            ),
        }

    def __getitem__(self, name: str) -> ProceduralRealSampleV2Scenario:
        return self._scenarios[name]


@pytest.fixture(scope="session")
def real_sample_v2_scenarios() -> ProceduralRealSampleV2Scenarios:
    """License-safe, deterministic inputs for real-sample-v2 behavior tests.

    The scenarios exercise the same solver and policy branches as the local research
    assets without embedding, deriving, or claiming provenance from those assets.
    """

    return ProceduralRealSampleV2Scenarios()


def _temperature_boundary_cloud() -> GaussianCloud:
    cloud = _base_cloud(count_per_slot=64, position_separation=0.25, jitter=0.50)
    vertices = cloud.vertices.copy()
    phase = np.arange(cloud.count, dtype=np.float32)
    vertices["red"] = np.clip(
        128.0 + np.where(vertices["object_id"] == 0, 12.0, -12.0) + 45.0 * np.sin(phase),
        0.0,
        255.0,
    ).astype(np.uint8)
    vertices["green"] = np.clip(
        128.0 + np.where(vertices["object_id"] == 0, -12.0, 12.0) + 45.0 * np.cos(phase * 0.7),
        0.0,
        255.0,
    ).astype(np.uint8)
    vertices["blue"] = np.clip(128.0 + 45.0 * np.sin(phase * 0.37), 0.0, 255.0).astype(
        np.uint8
    )
    return cloud.with_vertices(vertices)


def _coverage_boundary_cloud() -> GaussianCloud:
    cloud = _base_cloud(count_per_slot=256, position_separation=1.0, jitter=0.04)
    vertices = cloud.vertices.copy()
    selected = _balanced_selected_indices(
        count_per_slot=256,
        slots=2,
        quota_per_slot=64,
        seed=4,
    )
    for slot in range(2):
        slot_selected = selected[slot]
        biased = slot_selected[:8]
        other = 1 - slot
        sign = -1.0 if other == 0 else 1.0
        vertices["x"][biased] = sign
        vertices["y"][biased] = sign * 0.18
        vertices["z"][biased] = 0.25
    return cloud.with_vertices(vertices)


def _weight_boundary_cloud(*, mode: str, boundary_per_slot: int) -> GaussianCloud:
    if mode not in {"fix", "regress"}:
        raise ValueError(f"unsupported weight-boundary mode: {mode}")
    cloud = _base_cloud(count_per_slot=256, position_separation=1.0, jitter=0.035)
    vertices = cloud.vertices.copy()
    selected = _balanced_selected_indices(
        count_per_slot=256,
        slots=2,
        quota_per_slot=64,
        seed=4,
    )
    for slot in range(2):
        start = slot * 256
        stop = start + 256
        available = np.setdiff1d(
            np.arange(start, stop, dtype=np.int64),
            selected[slot],
            assume_unique=False,
        )
        boundary = available[:boundary_per_slot]
        other = 1 - slot
        if mode == "fix":
            own_x = -1.0 if slot == 0 else 1.0
            other_x = -1.0 if other == 0 else 1.0
            vertices["x"][boundary] = 0.46 * own_x + 0.54 * other_x
            vertices["y"][boundary] *= 0.25
        else:
            sign = -1.0 if slot == 0 else 1.0
            vertices["x"][boundary] = sign * 0.10
            vertices["y"][boundary] *= 0.25
            red, green, blue = _cluster_color(other)
            vertices["red"][boundary] = red
            vertices["green"][boundary] = green
            vertices["blue"][boundary] = blue
    return cloud.with_vertices(vertices)


def _no_safe_candidate_cloud() -> GaussianCloud:
    cloud = _base_cloud(count_per_slot=256, position_separation=1.0, jitter=0.035)
    vertices = cloud.vertices.copy()
    selected = _balanced_selected_indices(
        count_per_slot=256,
        slots=2,
        quota_per_slot=64,
        seed=4,
    )
    for slot in range(2):
        start = slot * 256
        stop = start + 256
        unselected = np.setdiff1d(
            np.arange(start, stop, dtype=np.int64),
            selected[slot],
            assume_unique=False,
        )
        phase = np.arange(unselected.shape[0], dtype=np.float32)
        vertices["x"][unselected] = 0.15 * np.sin(phase * 0.13)
        vertices["y"][unselected] = 0.15 * np.cos(phase * 0.17)
        vertices["z"][unselected] = 0.10 * np.sin(phase * 0.19)
        vertices["red"][unselected] = np.asarray(
            128.0 + 20.0 * np.sin(phase * 0.11), dtype=np.uint8
        )
        vertices["green"][unselected] = np.asarray(
            128.0 + 20.0 * np.cos(phase * 0.07), dtype=np.uint8
        )
        vertices["blue"][unselected] = np.asarray(
            128.0 + 20.0 * np.sin(phase * 0.05), dtype=np.uint8
        )
    return cloud.with_vertices(vertices)


def _base_cloud(
    *,
    count_per_slot: int,
    position_separation: float,
    jitter: float,
) -> GaussianCloud:
    count = count_per_slot * 2
    vertices = np.zeros(
        count,
        dtype=np.dtype(
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("opacity", "f4"),
                ("object_id", "i4"),
            ]
        ),
    )
    local = np.arange(count_per_slot, dtype=np.float32)
    for slot in range(2):
        indices = np.arange(slot * count_per_slot, (slot + 1) * count_per_slot)
        sign = -1.0 if slot == 0 else 1.0
        vertices["x"][indices] = sign * position_separation + jitter * np.sin(local * 0.31)
        vertices["y"][indices] = sign * 0.18 + jitter * np.cos(local * 0.23)
        vertices["z"][indices] = 0.25 + jitter * np.sin(local * 0.17 + slot)
        red, green, blue = _cluster_color(slot)
        vertices["red"][indices] = red
        vertices["green"][indices] = green
        vertices["blue"][indices] = blue
        vertices["object_id"][indices] = slot
    vertices["opacity"] = 1.0
    return GaussianCloud(vertices=vertices, source_format="procedural-fixture")


def _cluster_color(slot: int) -> tuple[int, int, int]:
    return (230, 35, 45) if slot == 0 else (30, 220, 210)


def _balanced_selected_indices(
    *,
    count_per_slot: int,
    slots: int,
    quota_per_slot: int,
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: dict[int, np.ndarray] = {}
    for slot in range(slots):
        indices = np.arange(slot * count_per_slot, (slot + 1) * count_per_slot, dtype=np.int64)
        rng.shuffle(indices)
        selected[slot] = indices[:quota_per_slot].copy()
    return selected
