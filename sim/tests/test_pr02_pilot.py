from __future__ import annotations

import copy
import unittest

from objgauss_sim.pr02_pilot import (
    REQUIRED_SOURCE_BRANCHES,
    aggregate_status,
    build_formal_data_spec,
    calibrate,
    decide_verdict,
    power_design,
    quantile,
    quaternion_distance,
    validate_run_lineage,
)


def state(*, x: float = 0.0, y: float = 0.0, velocity: float = 0.0) -> dict:
    return {
        "position_W_m": [x, y, 0.02],
        "quaternion_WO_wxyz": [1.0, 0.0, 0.0, 0.0],
        "linear_velocity_W_m_s": [velocity, 0.0, 0.0],
        "angular_velocity_W_rad_s": [0.0, velocity * 0.1, 0.0],
    }


def branch(branch_id: str, semantic: str) -> dict:
    records = []
    for time_s in (0.1, 0.2, 0.5, 1.1):
        if branch_id == "hold":
            target = state()
        elif branch_id == "push-pos-x-strong":
            target = state(x=0.02 * time_s, velocity=0.02)
        elif branch_id == "push-pos-x-weak":
            target = state(x=0.01 * time_s, velocity=0.01)
        elif branch_id == "push-neg-x-weak":
            target = state(x=-0.01 * time_s, velocity=-0.01)
        else:
            target = state(y=0.01 * time_s, velocity=0.01)
        records.append({"episode_time_s": time_s, "actors": {"target": target}})
    return {
        "trajectory": {"records": records},
        "episode": {"environment": {"object_spec_id": "pr02b-cal-object-a"}},
        "publication": {"semantic_sha256": semantic},
    }


def repeat() -> dict:
    return {
        f"pr02b-group-{index}": {
            branch_id: branch(branch_id, f"{index}-{branch_id}")
            for branch_id in REQUIRED_SOURCE_BRANCHES
        }
        for index in range(3)
    }


def spec() -> dict:
    return {
        "experiment_id": "pr02b-calibration-v0",
        "object_specs": {
            "pr02b-cal-object-a": {
                "half_size_m": [0.03, 0.02, 0.025],
                "density_kg_m3": 500.0,
                "symmetry": {
                    "kind": "finite_wxyz",
                    "rotations": [[1, 0, 0, 0], [0, 1, 0, 0]],
                }
            }
        },
        "layouts": {"pr02b-cal-layout-a": {}},
        "preflight_reset_seeds": [1],
        "calibration": {
            "hold_branch": "hold",
            "effect_branches": [
                "push-pos-x-weak",
                "push-pos-x-strong",
                "push-neg-x-weak",
                "push-pos-y-weak",
            ],
            "horizon_duration_s": 1.1,
            "scoring_times_s": [0.1, 0.2, 0.5, 1.1],
            "robust_scale_quantile": 0.75,
            "evaluator_noise_quantile": 0.95,
            "scale_epsilon": {
                "position": 0.000001,
                "orientation": 0.000001,
                "linear_velocity": 0.00001,
                "angular_velocity": 0.00001,
            },
        },
        "power": {
            "method": "test-power-proxy",
            "target_power": 0.8,
            "confidence_level": 0.95,
            "alternative_multiplier_over_delta": 2.0,
            "candidate_test_group_counts": [12, 24],
            "candidate_training_seed_counts": [3, 5],
            "training_seed_values": [101, 102, 103, 104, 105],
            "proxy_bootstrap_replicates": 32,
            "proxy_bootstrap_seed": 99,
            "train_to_test_group_ratio": 4,
            "validation_to_test_group_ratio": 1,
        },
        "formal_design": {
            "objects_per_split": {"train": 4, "validation": 2, "test": 2},
            "layouts_per_split": {"train": 3, "validation": 3, "test": 3},
            "starts_per_split": {"train": 1, "validation": 1, "test": 1},
        },
    }


class Pr02PilotTests(unittest.TestCase):
    def test_quantile_interpolates_and_rejects_empty_input(self) -> None:
        self.assertEqual(quantile([1.0, 3.0], 0.5), 2.0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            quantile([], 0.5)

    def test_quaternion_distance_respects_explicit_symmetry(self) -> None:
        distance = quaternion_distance(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        self.assertAlmostEqual(distance, 0.0)

    def test_calibration_is_repeatable_and_freezes_positive_scales(self) -> None:
        calibration = calibrate(spec=spec(), repeat_a=repeat(), repeat_b=repeat())
        self.assertTrue(calibration["semantic_hashes_match_across_orders"])
        self.assertTrue(all(calibration["normalization_scales"][key] > 0 for key in calibration["normalization_scales"]))
        self.assertTrue(all(item["passed"] for item in calibration["direction_checks"]))

        mutated = repeat()
        mutated["pr02b-group-0"]["hold"]["publication"]["semantic_sha256"] = "changed"
        changed = calibrate(spec=spec(), repeat_a=repeat(), repeat_b=mutated)
        self.assertFalse(changed["semantic_hashes_match_across_orders"])

    def test_power_and_formal_design_are_deterministic_and_split_isolated(self) -> None:
        policy = spec()
        power = power_design(policy, [1.0, 1.0, 1.0])
        self.assertEqual(power["selected"]["test_groups"], 12)
        self.assertEqual(power["selected"]["training_seeds"], 3)

        source_experiment = {"actions": [{"branch_id": "hold"}]}
        frozen = build_formal_data_spec(policy, source_experiment, power)
        self.assertEqual(frozen["group_counts"], {"train": 48, "validation": 12, "test": 12})
        self.assertTrue(
            frozen["action_support_envelope"][
                "formal_within_pilot_mass_envelope"
            ]
        )
        object_sets = [
            set(frozen["partitions"][split]["objects"])
            for split in ("train", "validation", "test")
        ]
        self.assertTrue(object_sets[0].isdisjoint(object_sets[1]))
        self.assertTrue(object_sets[0].isdisjoint(object_sets[2]))
        self.assertTrue(object_sets[1].isdisjoint(object_sets[2]))

        repeated = build_formal_data_spec(copy.deepcopy(policy), source_experiment, power)
        self.assertEqual(frozen, repeated)

    def test_insufficient_power_is_blocked_before_formal_data_freeze(self) -> None:
        policy = spec()
        power = power_design(policy, [-100.0, 0.0, 100.0])
        self.assertIsNone(power["selected"])
        self.assertTrue(all(not candidate["supported"] for candidate in power["candidates"]))
        with self.assertRaisesRegex(ValueError, "no supported formal size"):
            build_formal_data_spec(policy, {"actions": []}, power)

    def test_four_state_verdicts_preserve_source_and_resource_failures(self) -> None:
        self.assertEqual(aggregate_status(["supported", "rejected"]), "rejected")
        self.assertEqual(aggregate_status(["blocked", "invalid"]), "invalid")
        self.assertEqual(aggregate_status(["unexpected"]), "invalid")
        common = {
            "repeat_supported": True,
            "direction_supported": True,
            "horizon_supported": True,
            "power_supported": True,
            "resources_supported": True,
            "gpu_supported": True,
        }
        self.assertEqual(
            decide_verdict(source_status="rejected", **common),
            ("rejected", "scientific_gate_failed"),
        )
        self.assertEqual(
            decide_verdict(source_status="blocked", **common),
            ("blocked", "evidence_incomplete"),
        )
        self.assertEqual(
            decide_verdict(
                source_status="supported", **{**common, "gpu_supported": False}
            ),
            ("blocked", "evidence_incomplete"),
        )

    def test_repeat_and_audit_lineage_requires_opposite_orders(self) -> None:
        policy = spec()
        policy["calibration"]["repeat_orders"] = ["canonical", "reverse"]
        reports = [
            {
                "branch_order": order,
                "mode": "preflight",
                "verdict": "supported",
                "experiment_id": policy["experiment_id"],
                "experiment_spec_sha256": "spec",
                "experiment_manifest_sha256": "manifest",
                "source_commit": "commit",
                "counts": {"groups": 12, "episodes": 60, "attempts": 60},
            }
            for order in ("canonical", "reverse")
        ]
        audits = [
            {
                "identity": {"experiment_id": policy["experiment_id"]},
                "inputs": {"experiment_manifest_sha256": "manifest"},
                "counts": {"expected_groups": 12, "expected_episodes": 60},
            }
            for _ in range(2)
        ]
        validate_run_lineage(
            spec=policy,
            repeat_reports=reports,
            source_audits=audits,
            spec_sha256="spec",
            source_experiment_sha256="manifest",
            source_commit="commit",
        )
        reports[1]["branch_order"] = "canonical"
        with self.assertRaisesRegex(ValueError, "canonical/reverse"):
            validate_run_lineage(
                spec=policy,
                repeat_reports=reports,
                source_audits=audits,
                spec_sha256="spec",
                source_experiment_sha256="manifest",
                source_commit="commit",
            )


if __name__ == "__main__":
    unittest.main()
