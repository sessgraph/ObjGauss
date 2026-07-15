"""PR-02B isolated calibration, power design, and experiment freeze producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from .writer import strict_json_bytes


COMPONENTS = ("position", "orientation", "linear_velocity", "angular_velocity")
REQUIRED_SOURCE_BRANCHES = (
    "hold",
    "push-neg-x-weak",
    "push-pos-x-strong",
    "push-pos-x-weak",
    "push-pos-y-weak",
)
Z_95_ONE_SIDED = 1.6448536269514722


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def vector_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def normalized_quaternion(value: list[float]) -> list[float]:
    if len(value) != 4:
        raise ValueError("quaternion must have four wxyz components")
    magnitude = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    return [component / magnitude for component in value]


def quaternion_distance(
    left: list[float], right: list[float], symmetries: list[list[float]]
) -> float:
    left_normalized = normalized_quaternion(left)
    right_normalized = normalized_quaternion(right)
    candidates = []
    for symmetry in symmetries:
        equivalent = normalized_quaternion(
            quaternion_multiply(right_normalized, normalized_quaternion(symmetry))
        )
        dot = abs(sum(a * b for a, b in zip(left_normalized, equivalent)))
        candidates.append(2.0 * math.acos(max(-1.0, min(1.0, dot))))
    if not candidates:
        raise ValueError("orientation metric requires explicit object symmetries")
    return min(candidates)


def component_distance(
    left: dict[str, Any], right: dict[str, Any], symmetries: list[list[float]]
) -> dict[str, float]:
    return {
        "position": vector_distance(left["position_W_m"], right["position_W_m"]),
        "orientation": quaternion_distance(
            left["quaternion_WO_wxyz"], right["quaternion_WO_wxyz"], symmetries
        ),
        "linear_velocity": vector_distance(
            left["linear_velocity_W_m_s"], right["linear_velocity_W_m_s"]
        ),
        "angular_velocity": vector_distance(
            left["angular_velocity_W_rad_s"], right["angular_velocity_W_rad_s"]
        ),
    }


def record_at(records: list[dict[str, Any]], time_s: float) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if math.isclose(float(record["episode_time_s"]), time_s, abs_tol=1e-9)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one trajectory record at {time_s}, found {len(matches)}")
    return matches[0]


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_repeat(root: Path, experiment_id: str) -> dict[str, dict[str, Any]]:
    experiment_root = root / "dataset" / experiment_id
    if not experiment_root.is_dir():
        raise ValueError(f"pilot repeat lacks dataset root: {experiment_root}")
    groups: dict[str, dict[str, Any]] = {}
    for group_path in sorted(item for item in experiment_root.iterdir() if item.is_dir()):
        branches = {}
        for branch_path in sorted(item for item in group_path.iterdir() if item.is_dir()):
            trajectory = json.loads(
                (branch_path / "trajectory.json").read_text(encoding="utf-8")
            )
            episode = json.loads((branch_path / "episode.json").read_text(encoding="utf-8"))
            publication = json.loads(
                (branch_path / "publication.json").read_text(encoding="utf-8")
            )
            branches[branch_path.name] = {
                "trajectory": trajectory,
                "episode": episode,
                "publication": publication,
            }
        if tuple(sorted(branches)) != REQUIRED_SOURCE_BRANCHES:
            raise ValueError(f"pilot group {group_path.name} has the wrong branch set")
        groups[group_path.name] = branches
    if not groups:
        raise ValueError("pilot repeat has no groups")
    return groups


def explicit_symmetries(
    spec: dict[str, Any], group: dict[str, Any]
) -> list[list[float]]:
    object_id = group["hold"]["episode"]["environment"]["object_spec_id"]
    symmetry = spec["object_specs"][object_id]["symmetry"]
    if symmetry.get("kind") != "finite_wxyz":
        raise ValueError(f"unsupported pilot symmetry for {object_id}")
    return symmetry["rotations"]


def calibrate(
    *,
    spec: dict[str, Any],
    repeat_a: dict[str, dict[str, Any]],
    repeat_b: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if sorted(repeat_a) != sorted(repeat_b):
        raise ValueError("pilot repeats have different group identities")
    calibration = spec["calibration"]
    times = [float(value) for value in calibration["scoring_times_s"]]
    effects = {component: [] for component in COMPONENTS}
    noise = {component: [] for component in COMPONENTS}
    raw_by_group: dict[str, dict[str, list[dict[str, float]]]] = {}
    semantic_hashes_match = True

    for group_id in sorted(repeat_a):
        group_a = repeat_a[group_id]
        group_b = repeat_b[group_id]
        symmetries = explicit_symmetries(spec, group_a)
        raw_by_group[group_id] = {}
        for branch_id in REQUIRED_SOURCE_BRANCHES:
            if (
                group_a[branch_id]["publication"]["semantic_sha256"]
                != group_b[branch_id]["publication"]["semantic_sha256"]
            ):
                semantic_hashes_match = False
            records_a = group_a[branch_id]["trajectory"]["records"]
            records_b = group_b[branch_id]["trajectory"]["records"]
            for time_s in times:
                state_a = record_at(records_a, time_s)["actors"]["target"]
                state_b = record_at(records_b, time_s)["actors"]["target"]
                observed_noise = component_distance(state_a, state_b, symmetries)
                for component in COMPONENTS:
                    noise[component].append(observed_noise[component])

        hold_records = group_a[calibration["hold_branch"]]["trajectory"]["records"]
        for branch_id in calibration["effect_branches"]:
            branch_records = group_a[branch_id]["trajectory"]["records"]
            raw_by_group[group_id][branch_id] = []
            for time_s in times:
                hold_state = record_at(hold_records, time_s)["actors"]["target"]
                action_state = record_at(branch_records, time_s)["actors"]["target"]
                observed_effect = component_distance(action_state, hold_state, symmetries)
                raw_by_group[group_id][branch_id].append(observed_effect)
                for component in COMPONENTS:
                    effects[component].append(observed_effect[component])

    scales = {}
    scale_details = {}
    for component in COMPONENTS:
        robust = quantile(effects[component], calibration["robust_scale_quantile"])
        noise_floor = quantile(noise[component], calibration["evaluator_noise_quantile"])
        scale = max(robust, noise_floor, calibration["scale_epsilon"][component])
        scales[component] = scale
        scale_details[component] = {
            "robust_scale": robust,
            "evaluator_noise_floor": noise_floor,
            "epsilon": calibration["scale_epsilon"][component],
            "frozen_scale": scale,
            "effect_observation_count": len(effects[component]),
            "noise_observation_count": len(noise[component]),
        }

    group_effects = []
    direction_checks = []
    for group_id in sorted(raw_by_group):
        normalized = []
        for branch_values in raw_by_group[group_id].values():
            for observed in branch_values:
                normalized.extend(observed[component] / scales[component] for component in COMPONENTS)
        group_effects.append(sum(normalized) / len(normalized))

        hold_final = record_at(
            repeat_a[group_id][calibration["hold_branch"]]["trajectory"]["records"],
            calibration["horizon_duration_s"],
        )["actors"]["target"]["position_W_m"][0]
        for branch_id, expected_sign in (("push-pos-x-weak", 1), ("push-neg-x-weak", -1)):
            action_final = record_at(
                repeat_a[group_id][branch_id]["trajectory"]["records"],
                calibration["horizon_duration_s"],
            )["actors"]["target"]["position_W_m"][0]
            observed = action_final - hold_final
            direction_checks.append(
                {
                    "group_id": group_id,
                    "branch_id": branch_id,
                    "expected_sign": expected_sign,
                    "observed_effect_x_m": observed,
                    "passed": observed * expected_sign > 0.0,
                }
            )

    return {
        "semantic_hashes_match_across_orders": semantic_hashes_match,
        "horizon": {
            "unit": "physical_seconds",
            "duration_s": calibration["horizon_duration_s"],
            "scoring_times_s": calibration["scoring_times_s"],
            "covers_action": min(times) <= 0.1,
            "covers_settling": max(times) >= 1.1,
        },
        "normalization_scales": scales,
        "scale_details": scale_details,
        "normalized_group_effects": group_effects,
        "direction_checks": direction_checks,
    }


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def power_design(spec: dict[str, Any], group_effects: list[float]) -> dict[str, Any]:
    policy = spec["power"]
    median_effect = statistics.median(group_effects)
    delta = max(0.05, min(0.10, 0.10 * median_effect))
    delta_shuffle = max(0.03, 0.60 * delta)
    raw_group_sigma = (
        statistics.stdev(group_effects) if len(group_effects) > 1 else 0.0
    )

    randomizer = random.Random(policy["proxy_bootstrap_seed"])
    bootstrap_means = []
    for _ in range(policy["proxy_bootstrap_replicates"]):
        sample = [randomizer.choice(group_effects) for _ in group_effects]
        bootstrap_means.append(statistics.fmean(sample))
    raw_seed_proxy_sigma = (
        statistics.stdev(bootstrap_means) if len(bootstrap_means) > 1 else 0.0
    )
    effect_scale = abs(median_effect)
    effect_scale_supported = effect_scale > 0.0
    proxy_denominator = effect_scale if effect_scale_supported else 1.0
    group_sigma = delta * raw_group_sigma / proxy_denominator
    seed_proxy_sigma = delta * raw_seed_proxy_sigma / proxy_denominator
    alternative = policy["alternative_multiplier_over_delta"] * delta
    candidates = []
    for groups in policy["candidate_test_group_counts"]:
        for seeds in policy["candidate_training_seed_counts"]:
            standard_error = math.sqrt(
                group_sigma * group_sigma / groups
                + seed_proxy_sigma * seed_proxy_sigma / seeds
            )
            if not effect_scale_supported:
                power = 0.0
            elif standard_error == 0.0:
                power = 1.0
            else:
                power = normal_cdf(
                    (alternative - delta) / standard_error - Z_95_ONE_SIDED
                )
            candidates.append(
                {
                    "test_groups": groups,
                    "training_seeds": seeds,
                    "standard_error_proxy": standard_error,
                    "power": power,
                    "supported": power >= policy["target_power"],
                }
            )
    supported = [item for item in candidates if item["supported"]]
    supported.sort(
        key=lambda item: (
            item["test_groups"] * item["training_seeds"],
            item["test_groups"],
            item["training_seeds"],
        )
    )
    selected = supported[0] if supported else None
    return {
        "method": policy["method"],
        "target_power": policy["target_power"],
        "confidence_level": policy["confidence_level"],
        "alternative_error_reduction": alternative,
        "median_normalized_effect": median_effect,
        "positive_effect_scale_supported": effect_scale_supported,
        "raw_group_effect_sigma": raw_group_sigma,
        "raw_bootstrap_group_mean_sigma": raw_seed_proxy_sigma,
        "group_error_reduction_sigma_proxy": group_sigma,
        "training_seed_error_reduction_sigma_proxy": seed_proxy_sigma,
        "variance_proxy_formula": "delta*(effect_sigma/abs(median_normalized_effect))",
        "proxy_boundary": "bootstrap group-mean sensitivity only; not observed GNN performance",
        "delta": delta,
        "delta_shuffle": delta_shuffle,
        "candidates": candidates,
        "selected": selected,
    }


def formal_object_catalog(split: str, count: int) -> dict[str, Any]:
    catalog = {}
    split_offset = {"train": 0, "validation": 4, "test": 6}[split]
    for index in range(1, count + 1):
        offset = split_offset + index
        catalog[f"pr02-{split}-object-{index:02d}"] = {
            "half_size_m": [
                0.020 + 0.0004 * offset,
                0.024 + 0.0003 * offset,
                0.018 + 0.00035 * offset,
            ],
            "density_kg_m3": 440.0 + 5.0 * offset,
            "symmetry": {
                "kind": "finite_wxyz",
                "rotations": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            },
        }
    return catalog


def formal_layout_catalog(split: str, count: int) -> dict[str, Any]:
    base = {"train": 0.42, "validation": 0.50, "test": 0.58}[split]
    return {
        f"pr02-{split}-layout-{index:02d}": {
            "context_xy_m": [
                (-1 if index % 2 == 0 else 1) * (base + 0.015 * index),
                (-1 if index % 3 == 0 else 1) * (base - 0.01 * index),
            ]
        }
        for index in range(1, count + 1)
    }


def cuboid_mass_kg(specification: dict[str, Any]) -> float:
    half_size = specification["half_size_m"]
    return 8.0 * math.prod(half_size) * specification["density_kg_m3"]


def build_formal_data_spec(
    spec: dict[str, Any], source_experiment: dict[str, Any], power: dict[str, Any]
) -> dict[str, Any]:
    selected = power["selected"]
    if selected is None:
        raise ValueError("power design has no supported formal size")
    test_groups = selected["test_groups"]
    group_counts = {
        "train": test_groups * spec["power"]["train_to_test_group_ratio"],
        "validation": test_groups
        * spec["power"]["validation_to_test_group_ratio"],
        "test": test_groups,
    }
    partitions = {}
    all_groups = []
    for split in ("train", "validation", "test"):
        design = spec["formal_design"]
        objects = formal_object_catalog(split, design["objects_per_split"][split])
        layouts = formal_layout_catalog(split, design["layouts_per_split"][split])
        starts = {
            f"pr02-{split}-start-01": {
                "target_base_xy_m": [0.0, 0.0],
                "target_jitter_m": 0.02,
            }
        }
        divisor = len(objects) * len(layouts) * len(starts)
        if group_counts[split] % divisor != 0:
            raise ValueError(f"{split} group count cannot be factored by frozen strata")
        seed_count = group_counts[split] // divisor
        seed_base = {"train": 27071000, "validation": 27072000, "test": 27073000}[split]
        reset_seeds = [seed_base + index for index in range(1, seed_count + 1)]
        groups = []
        for object_id in objects:
            for layout_id in layouts:
                for start_id in starts:
                    for reset_seed in reset_seeds:
                        group_id = (
                            f"group-{object_id}-{layout_id}-{start_id}-seed-{reset_seed}"
                        )
                        groups.append(
                            {
                                "group_id": group_id,
                                "object_identity_id": object_id,
                                "layout_id": layout_id,
                                "start_pose_id": start_id,
                                "reset_seed": reset_seed,
                            }
                        )
        if len(groups) != group_counts[split]:
            raise AssertionError("formal group design count drifted")
        partitions[split] = {
            "objects": objects,
            "layouts": layouts,
            "starts": starts,
            "reset_seeds": reset_seeds,
            "groups": groups,
        }
        all_groups.extend({"split": split, **group} for group in groups)
    pilot_max_mass = max(cuboid_mass_kg(item) for item in spec["object_specs"].values())
    formal_masses = [
        cuboid_mass_kg(item)
        for partition in partitions.values()
        for item in partition["objects"].values()
    ]
    formal_max_mass = max(formal_masses)
    return {
        "spec_version": "0.1.0",
        "spec_kind": "objgauss.pr02-formal-data-freeze",
        "experiment_id": "pr02-objectstate-baseline-v0",
        "source_episode_schema_version": "0.2.0",
        "source_kind": "maniskill-programmatic-cpu-primitive",
        "pilot_exclusion": {
            "experiment_id": spec["experiment_id"],
            "object_identity_ids": sorted(spec["object_specs"]),
            "layout_ids": sorted(spec["layouts"]),
            "reset_seeds": spec["preflight_reset_seeds"],
            "excluded_from_training": True,
            "excluded_from_final_statistics": True,
        },
        "action_support_envelope": {
            "basis": "same-material-cuboid-mass-no-greater-than-audited-pilot-maximum",
            "pilot_max_mass_kg": pilot_max_mass,
            "formal_max_mass_kg": formal_max_mass,
            "formal_within_pilot_mass_envelope": formal_max_mass <= pilot_max_mass,
        },
        "partitions": partitions,
        "group_counts": group_counts,
        "total_groups": len(all_groups),
        "actions": source_experiment["actions"],
        "split_unit": "sibling_group",
        "final_access": {
            "trainer_loader_rejects_test": True,
            "inference_reads_gt_future": False,
            "evaluator_only_reads_gt_future": True,
            "max_formal_runs": 1,
        },
    }


def gpu_probe(spec: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {
            "status": "blocked",
            "reason_code": "cuda_unavailable",
            "torch": metadata.version("torch"),
            "torch_cuda": torch.version.cuda,
            "gpu_compute_hours": 0.0,
            "claim_boundary": "availability/reserve probe only; no trainer or performance claim",
        }
    device = torch.device("cuda:0")
    free_before, total = torch.cuda.mem_get_info(device)
    reserve = spec["budgets"]["display_vram_reserve_bytes_min"]
    process_cap = min(
        spec["budgets"]["training_peak_vram_bytes_max"], free_before - reserve
    )
    if process_cap <= 0:
        return {
            "status": "blocked",
            "reason_code": "display_reserve_unavailable",
            "torch": metadata.version("torch"),
            "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_bytes": int(total),
            "free_before_bytes": int(free_before),
            "display_reserve_bytes": reserve,
            "training_allocation_cap_bytes": 0,
            "gpu_compute_hours": 0.0,
            "claim_boundary": "availability/reserve probe only; no trainer or performance claim",
        }
    probe_bytes = min(16 * 1024 * 1024, max(1, process_cap // 64))
    probe = torch.ones(probe_bytes, dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)
    observed = int(probe.numel() * probe.element_size())
    del probe
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_after, total_after = torch.cuda.mem_get_info(device)
    reserve_supported = total_after == total and min(free_before, free_after) >= reserve
    return {
        "status": "supported" if reserve_supported else "blocked",
        "reason_code": (
            "display_reserve_supported"
            if reserve_supported
            else "display_reserve_unavailable"
        ),
        "torch": metadata.version("torch"),
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_bytes": int(total),
        "free_before_bytes": int(free_before),
        "free_after_bytes": int(free_after),
        "display_reserve_bytes": reserve,
        "training_allocation_cap_bytes": int(process_cap),
        "probe_allocation_bytes": observed,
        "gpu_compute_hours": 0.0,
        "claim_boundary": "availability/reserve probe only; no trainer or performance claim",
    }


def build_dynamics_experiment(
    *,
    spec: dict[str, Any],
    source_experiment: dict[str, Any],
    spec_sha256: str,
    pilot_report_sha256: str,
    formal_data_spec: dict[str, Any],
    power: dict[str, Any],
    calibration: dict[str, Any],
    grid_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    selected = power["selected"]
    if selected is None:
        raise ValueError("cannot freeze experiment without supported power design")
    calibration_group_ids = []
    for object_id in spec["object_specs"]:
        for layout_id in spec["layouts"]:
            for start_id in spec["preflight_start_pose_ids"]:
                for reset_seed in spec["preflight_reset_seeds"]:
                    calibration_group_ids.append(
                        f"group-{object_id}-{layout_id}-{start_id}-seed-{reset_seed}"
                    )

    partitions = {}
    for split in ("train", "validation", "test"):
        partition = formal_data_spec["partitions"][split]
        partitions[split] = {
            "object_identity_ids": sorted(partition["objects"]),
            "layout_ids": sorted(partition["layouts"]),
            "group_ids": [group["group_id"] for group in partition["groups"]],
            "group_count": len(partition["groups"]),
        }
    grid_bytes = grid_path.read_bytes()
    return {
        "schema_version": "0.3.0",
        "contract_kind": "objgauss.dynamics_experiment",
        "identity": {
            "experiment_id": "pr02-objectstate-baseline-v0",
            "fixture_id": "pr02b-freeze-v0",
            "preregistration_sha256": spec_sha256,
        },
        "contract_versions": {"source_episode": "0.2.0", "dynamics_records": "0.3.0"},
        "source": {
            "simulator": "mani-skill==3.0.1",
            "sapien": "sapien==3.0.3",
            "runtime_lock_sha256": file_sha256(lock_path),
            "source_gate_report_sha256": pilot_report_sha256,
            "source_gate_status": "supported",
        },
        "data_policy": {
            "source_kind": "maniskill-programmatic-cpu-primitive",
            "pr01_groups_excluded": True,
            "new_cohort_required": True,
            "split_unit": "sibling_group",
            "isolation_keys": ["object_identity_id", "layout_id", "group_id"],
            "pilot_freeze": {
                "report_sha256": pilot_report_sha256,
                "excluded_from_training": True,
                "excluded_from_final_statistics": True,
                "freezes": [
                    "horizon_and_scoring_times",
                    "normalization_scales",
                    "delta",
                    "delta_shuffle",
                    "split_group_counts",
                    "training_seeds",
                    "training_configuration",
                ],
            },
            "calibration": {
                "object_identity_ids": sorted(spec["object_specs"]),
                "layout_ids": sorted(spec["layouts"]),
                "group_ids": calibration_group_ids,
                "group_count": len(calibration_group_ids),
            },
            **partitions,
            "final_access": formal_data_spec["final_access"],
        },
        "endpoint": {
            "name": "target-object-multistep-effect-vs-hold-objectstate-error",
            "target_scope": "intervention.target_object_id",
            "components": [
                "position",
                "orientation_symmetry_corrected",
                "linear_velocity",
                "angular_velocity",
            ],
            "component_weights": {
                "position": 0.25,
                "orientation": 0.25,
                "linear_velocity": 0.25,
                "angular_velocity": 0.25,
            },
            "normalization": "max-pilot-robust-scale-evaluator-noise-floor",
            "normalization_scales": calibration["normalization_scales"],
            "aggregation": "group-first-paired-hierarchical-bootstrap",
            "confidence_level": 0.95,
            "baselines": ["copy_state", "constant_velocity", "action_free"],
            "delta": power["delta"],
            "delta_shuffle": power["delta_shuffle"],
        },
        "horizon": calibration["horizon"],
        "training": {
            "model_family": "minimal-object-gnn",
            "message_passing_rounds": 1,
            "prediction_action": "commanded_action",
            "executed_action_is_feature": False,
            "loss": "normalized-open-loop-branch-rollout",
            "teacher_forcing": "initial-state-only",
            "arms": ["copy_state", "constant_velocity", "action_free", "action_conditioned"],
            "training_seeds": spec["power"]["training_seed_values"][: selected["training_seeds"]],
            "checkpoint_selection": "minimum-validation-primary-error-per-seed",
            "hyperparameter_grid": {
                "uri": "contracts/fixtures/pr02b/hyperparameter-grid.json",
                "media_type": "application/json",
                "sha256": sha256_bytes(grid_bytes),
                "byte_size": len(grid_bytes),
            },
        },
        "budgets": {
            "gpu_hours_total_max": spec["budgets"]["gpu_hours_total_max"],
            "gpu_hours_pilot_hpo_max": spec["budgets"]["gpu_hours_pilot_hpo_max"],
            "gpu_hours_formal_max": spec["budgets"]["gpu_hours_formal_max"],
            "training_peak_vram_bytes_max": spec["budgets"]["training_peak_vram_bytes_max"],
            "display_vram_reserve_bytes_min": spec["budgets"]["display_vram_reserve_bytes_min"],
            "cohort_cpu_wall_hours_max": spec["budgets"]["cohort_cpu_wall_hours_max"],
            "artifact_bytes_max": spec["budgets"]["artifact_bytes_max"],
        },
        "retry_policy": {
            "max_attempts_per_task": 2,
            "max_extra_attempt_fraction": source_experiment["budgets"][
                "extra_attempt_fraction_max"
            ],
            "same_seed_config": True,
            "eligible_failures": ["process_crash", "io_failure", "transient_oom"],
            "scientific_retry_allowed": False,
        },
        "claim_boundary": {
            "hypothesis": "commanded-action-conditioned-objectstate-dynamics-beats-all-preregistered-baselines",
            "excluded_claims": ["gaussian-dynamics-value", "external-data-generalization", "robot-control"],
        },
    }


def validate_inputs(
    spec: dict[str, Any], source_experiment: dict[str, Any], spec_sha256: str
) -> None:
    if source_experiment["identity"]["experiment_spec_sha256"] != spec_sha256:
        raise ValueError("source experiment does not freeze the PR-02B pilot spec")
    if spec["calibration"]["repeat_orders"] != ["canonical", "reverse"]:
        raise ValueError("pilot must use canonical/reverse branch-order repeats")
    if not all(value.startswith("pr02b-cal-") for value in spec["object_specs"]):
        raise ValueError("pilot object IDs are not isolated")
    if not all(value.startswith("pr02b-cal-") for value in spec["layouts"]):
        raise ValueError("pilot layout IDs are not isolated")
    forbidden_seeds = {
        24071401,
        24071402,
        24071403,
        24071404,
        24071501,
        24071502,
    }
    if forbidden_seeds.intersection(spec["preflight_reset_seeds"]):
        raise ValueError("pilot reset seeds overlap PR-01")


def validate_run_lineage(
    *,
    spec: dict[str, Any],
    repeat_reports: list[dict[str, Any]],
    source_audits: list[dict[str, Any]],
    spec_sha256: str,
    source_experiment_sha256: str,
    source_commit: str,
) -> None:
    if [report.get("branch_order") for report in repeat_reports] != spec["calibration"][
        "repeat_orders"
    ]:
        raise ValueError("pilot repeats are not canonical/reverse in frozen order")
    expected_counts = {"groups": 12, "episodes": 60, "attempts": 60}
    for report in repeat_reports:
        if report.get("mode") != "preflight" or report.get("verdict") != "supported":
            raise ValueError("pilot repeat report is not a supported preflight")
        if report.get("experiment_id") != spec["experiment_id"]:
            raise ValueError("pilot repeat experiment identity drifted")
        if report.get("experiment_spec_sha256") != spec_sha256:
            raise ValueError("pilot repeat spec lineage drifted")
        if report.get("experiment_manifest_sha256") != source_experiment_sha256:
            raise ValueError("pilot repeat source experiment lineage drifted")
        if report.get("source_commit") != source_commit:
            raise ValueError("pilot repeat source commit drifted")
        if any(report["counts"].get(key) != value for key, value in expected_counts.items()):
            raise ValueError("pilot repeat counts drifted")
    for audit in source_audits:
        if audit["identity"].get("experiment_id") != spec["experiment_id"]:
            raise ValueError("source audit experiment identity drifted")
        if audit["inputs"].get("experiment_manifest_sha256") != source_experiment_sha256:
            raise ValueError("source audit manifest lineage drifted")
        if audit["counts"].get("expected_groups") != 12 or audit["counts"].get(
            "expected_episodes"
        ) != 60:
            raise ValueError("source audit expected counts drifted")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_status(statuses: list[str]) -> str:
    allowed = {"supported", "rejected", "blocked", "invalid"}
    if not statuses or any(status not in allowed for status in statuses):
        return "invalid"
    for candidate in ("invalid", "rejected", "blocked"):
        if candidate in statuses:
            return candidate
    return "supported"


def decide_verdict(
    *,
    source_status: str,
    repeat_supported: bool,
    direction_supported: bool,
    horizon_supported: bool,
    power_supported: bool,
    resources_supported: bool,
    gpu_supported: bool,
) -> tuple[str, str]:
    if source_status == "invalid" or not repeat_supported:
        return "invalid", "structural_evidence_invalid"
    if source_status == "rejected" or not direction_supported or not horizon_supported:
        return "rejected", "scientific_gate_failed"
    if (
        source_status == "blocked"
        or not power_supported
        or not resources_supported
        or not gpu_supported
    ):
        return "blocked", "evidence_incomplete"
    return "supported", "all_hard_gates_passed"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    spec = read_json(args.spec)
    source_experiment = read_json(args.source_experiment)
    spec_sha256 = file_sha256(args.spec)
    validate_inputs(spec, source_experiment, spec_sha256)
    repeat_reports = [read_json(args.repeat_a_report), read_json(args.repeat_b_report)]
    source_audits = [read_json(args.audit_a), read_json(args.audit_b)]
    validate_run_lineage(
        spec=spec,
        repeat_reports=repeat_reports,
        source_audits=source_audits,
        spec_sha256=spec_sha256,
        source_experiment_sha256=file_sha256(args.source_experiment),
        source_commit=args.source_commit,
    )
    grid = read_json(args.grid)
    repeat_a = load_repeat(args.repeat_a, spec["experiment_id"])
    repeat_b = load_repeat(args.repeat_b, spec["experiment_id"])
    calibration = calibrate(spec=spec, repeat_a=repeat_a, repeat_b=repeat_b)
    power = power_design(spec, calibration["normalized_group_effects"])
    power_supported = power["selected"] is not None
    formal_data_spec = (
        build_formal_data_spec(spec, source_experiment, power)
        if power_supported
        else None
    )
    formal_bytes = (
        strict_json_bytes(formal_data_spec) if formal_data_spec is not None else None
    )
    observed_gpu = gpu_probe(spec)

    source_statuses = [audit["verdict"]["status"] for audit in source_audits]
    source_status = aggregate_status(source_statuses)
    direction_supported = all(
        item["passed"] for item in calibration["direction_checks"]
    )
    horizon_supported = (
        calibration["horizon"]["covers_action"]
        and calibration["horizon"]["covers_settling"]
    )
    repeat_supported = calibration["semantic_hashes_match_across_orders"]
    total_pilot_wall = sum(report["telemetry"]["wall_seconds"] for report in repeat_reports)
    total_pilot_bytes = directory_bytes(args.repeat_a) + directory_bytes(args.repeat_b)
    max_pilot_rss = max(
        report["telemetry"]["max_rss_bytes"] for report in repeat_reports
    )
    total_formal_groups = (
        formal_data_spec["total_groups"] if formal_data_spec is not None else 0
    )
    p95_group_seconds = max(
        report["telemetry"]["p95_group_runtime_s"] for report in repeat_reports
    )
    p95_group_bytes = max(
        report["telemetry"]["p95_group_artifact_bytes"] for report in repeat_reports
    )
    projected_cpu_hours = (
        p95_group_seconds * total_formal_groups * 1.25 / 3600.0
        if formal_data_spec is not None
        else None
    )
    projected_artifact_bytes = (
        math.ceil(p95_group_bytes * total_formal_groups * 1.15)
        if formal_data_spec is not None
        else None
    )
    selected_training_seeds = (
        power["selected"]["training_seeds"] if power["selected"] is not None else 0
    )
    hpo_gpu_hours_base = (
        2
        * grid["search"]["configuration_count"]
        * selected_training_seeds
        * grid["per_task_limits"]["hpo_wall_seconds_max"]
        / 3600.0
    )
    formal_gpu_hours_base = (
        2
        * selected_training_seeds
        * grid["per_task_limits"]["formal_wall_seconds_max"]
        / 3600.0
    )
    retry_reserve_fraction = source_experiment["budgets"][
        "extra_attempt_fraction_max"
    ]
    hpo_gpu_hours_max = hpo_gpu_hours_base * (1.0 + retry_reserve_fraction)
    formal_gpu_hours_max = formal_gpu_hours_base * (1.0 + retry_reserve_fraction)
    total_gpu_hours_max = hpo_gpu_hours_max + formal_gpu_hours_max
    resources_supported = (
        formal_data_spec is not None
        and total_pilot_wall <= spec["budgets"]["pilot_wall_seconds_max"]
        and total_pilot_bytes <= spec["budgets"]["pilot_artifact_bytes_max"]
        and max_pilot_rss <= spec["budgets"]["pilot_process_rss_bytes_max"]
        and projected_cpu_hours is not None
        and projected_cpu_hours <= spec["budgets"]["cohort_cpu_wall_hours_max"]
        and projected_artifact_bytes is not None
        and projected_artifact_bytes <= spec["budgets"]["artifact_bytes_max"]
        and hpo_gpu_hours_max <= spec["budgets"]["gpu_hours_pilot_hpo_max"]
        and formal_gpu_hours_max <= spec["budgets"]["gpu_hours_formal_max"]
        and total_gpu_hours_max <= spec["budgets"]["gpu_hours_total_max"]
    )
    gpu_supported = observed_gpu["status"] == "supported"
    hard_gates = {
        "source_audits_supported": source_status,
        "canonical_reverse_semantics_match": (
            "supported" if repeat_supported else "invalid"
        ),
        "direction_controls_supported": (
            "supported" if direction_supported else "rejected"
        ),
        "horizon_covers_action_and_settling": (
            "supported" if horizon_supported else "rejected"
        ),
        "power_target_reached": "supported" if power_supported else "blocked",
        "resources_within_hard_budgets": (
            "supported" if resources_supported else "blocked"
        ),
        "gpu_display_reserve_supported": (
            "supported" if gpu_supported else "blocked"
        ),
    }
    verdict, reason = decide_verdict(
        source_status=source_status,
        repeat_supported=repeat_supported,
        direction_supported=direction_supported,
        horizon_supported=horizon_supported,
        power_supported=power_supported,
        resources_supported=resources_supported,
        gpu_supported=gpu_supported,
    )

    analyzer_sha256 = file_sha256(Path(__file__))
    report = {
        "report_version": "0.1.0",
        "report_kind": "objgauss.pr02b-calibration-power-freeze",
        "identity": {
            "report_id": "pr02b-calibration-power-v0",
            "experiment_id": "pr02-objectstate-baseline-v0",
            "pilot_experiment_id": spec["experiment_id"],
        },
        "verdict": {"status": verdict, "reason_code": reason},
        "inputs": {
            "pilot_spec_sha256": spec_sha256,
            "source_experiment_sha256": file_sha256(args.source_experiment),
            "runtime_lock_sha256": file_sha256(args.lock),
            "hyperparameter_grid_sha256": file_sha256(args.grid),
            "repeat_report_sha256": [
                file_sha256(args.repeat_a_report),
                file_sha256(args.repeat_b_report),
            ],
            "source_audit_sha256": [file_sha256(args.audit_a), file_sha256(args.audit_b)],
        },
        "producer": {
            "name": "objgauss.pr02b-pilot",
            "version": "0.1.0",
            "source_sha256": analyzer_sha256,
            "source_commit": args.source_commit,
        },
        "counts": {
            "pilot_groups_per_repeat": len(repeat_a),
            "pilot_repeats": 2,
            "pilot_episodes": len(repeat_a) * len(REQUIRED_SOURCE_BRANCHES) * 2,
            "formal_groups": total_formal_groups,
            "training_seeds": power["selected"]["training_seeds"] if power["selected"] else 0,
        },
        "source_audits": [
            {
                "status": audit["verdict"]["status"],
                "reason_code": audit["verdict"]["reason_code"],
                "observed_groups": audit["counts"]["observed_groups"],
                "observed_episodes": audit["counts"]["observed_episodes"],
            }
            for audit in source_audits
        ],
        "gpu_probe": observed_gpu,
        "calibration": calibration,
        "power": power,
        "freeze": {
            "formal_data_spec_sha256": (
                sha256_bytes(formal_bytes) if formal_bytes is not None else None
            ),
            "hyperparameter_grid_sha256": file_sha256(args.grid),
            "horizon": calibration["horizon"],
            "normalization_scales": calibration["normalization_scales"],
            "delta": power["delta"],
            "delta_shuffle": power["delta_shuffle"],
            "training_seed_values": spec["power"]["training_seed_values"][
                : power["selected"]["training_seeds"] if power["selected"] else 0
            ],
            "formal_group_counts": (
                formal_data_spec["group_counts"]
                if formal_data_spec is not None
                else None
            ),
        },
        "resources": {
            "pilot_wall_seconds": total_pilot_wall,
            "pilot_artifact_bytes": total_pilot_bytes,
            "pilot_process_rss_peak_bytes": max_pilot_rss,
            "projected_formal_cohort_cpu_wall_hours": projected_cpu_hours,
            "projected_formal_artifact_bytes": projected_artifact_bytes,
            "scheduled_hpo_gpu_hours_base": hpo_gpu_hours_base,
            "scheduled_formal_gpu_hours_base": formal_gpu_hours_base,
            "retry_gpu_reserve_fraction": retry_reserve_fraction,
            "scheduled_hpo_gpu_hours_max": hpo_gpu_hours_max,
            "scheduled_formal_gpu_hours_max": formal_gpu_hours_max,
            "scheduled_total_gpu_hours_max": total_gpu_hours_max,
            "analysis_wall_seconds": time.monotonic() - started,
            "hard_budgets": spec["budgets"],
        },
        "hard_gates": [
            {"gate_id": gate_id, "status": status}
            for gate_id, status in hard_gates.items()
        ],
        "claim_boundary": {
            "supported_claim": "isolated-pilot-can-freeze-pr02-experiment-design",
            "excluded_claims": [
                "trained-model-performance",
                "gaussian-dynamics-value",
                "external-data-generalization",
                "robot-control",
            ],
        },
    }
    report_bytes = strict_json_bytes(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(report_bytes)
    report_sha256 = sha256_bytes(report_bytes)

    if verdict == "supported" and formal_data_spec is not None and formal_bytes is not None:
        args.formal_data_spec.parent.mkdir(parents=True, exist_ok=True)
        args.formal_data_spec.write_bytes(formal_bytes)
        dynamics_experiment = build_dynamics_experiment(
            spec=spec,
            source_experiment=source_experiment,
            spec_sha256=spec_sha256,
            pilot_report_sha256=report_sha256,
            formal_data_spec=formal_data_spec,
            power=power,
            calibration=calibration,
            grid_path=args.grid,
            lock_path=args.lock,
        )
        args.dynamics_experiment.parent.mkdir(parents=True, exist_ok=True)
        args.dynamics_experiment.write_bytes(strict_json_bytes(dynamics_experiment))
    return {
        "verdict": verdict,
        "reason_code": reason,
        "report_sha256": report_sha256,
        "report": str(args.report),
        "formal_data_spec": (
            str(args.formal_data_spec) if verdict == "supported" else None
        ),
        "dynamics_experiment": (
            str(args.dynamics_experiment) if verdict == "supported" else None
        ),
        "selected_power_design": power["selected"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repeat-a", type=Path, required=True)
    parser.add_argument("--repeat-b", type=Path, required=True)
    parser.add_argument("--repeat-a-report", type=Path, required=True)
    parser.add_argument("--repeat-b-report", type=Path, required=True)
    parser.add_argument("--audit-a", type=Path, required=True)
    parser.add_argument("--audit-b", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--formal-data-spec", type=Path, required=True)
    parser.add_argument("--dynamics-experiment", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as error:
        result = {
            "verdict": "invalid",
            "reason_code": "structural_evidence_invalid",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        print(json.dumps(result, sort_keys=True))
        return 4
    print(json.dumps(result, sort_keys=True))
    return {"supported": 0, "rejected": 2, "blocked": 3, "invalid": 4}[
        result["verdict"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
