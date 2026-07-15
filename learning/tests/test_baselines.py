from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from objgauss_learning import baselines
from objgauss_learning.runtime import file_sha256, strict_json_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "a" * 40


def initial_state(
    *,
    object_id: str = "target",
    position: list[float] | None = None,
    quaternion: list[float] | None = None,
    linear: list[float] | None = None,
    angular: list[float] | None = None,
) -> dict:
    return {
        "object_id": object_id,
        "position_W_m": position or [1.0, 2.0, 3.0],
        "quaternion_WO_wxyz": quaternion or [1.0, 0.0, 0.0, 0.0],
        "linear_velocity_W_m_s": linear or [0.5, -0.25, 0.0],
        "angular_velocity_W_rad_s": angular or [0.0, 0.0, 0.0],
    }


def sample(group_id: str = "group-00", branch_id: str = "hold") -> dict:
    states = [initial_state(), initial_state(object_id="context", position=[0.0, 0.0, 0.0])]
    command = {
        "kind": "hold",
        "vector_W_N": [0.0, 0.0, 0.0],
        "duration_s": 0.1,
        "sim_frequency_hz": 100,
        "applied_steps": 10,
    }
    return {
        "group_id": group_id,
        "branch_id": branch_id,
        "split": "validation",
        "source_episode": {
            "uri": f"generated/pr02c/data/{group_id}/{branch_id}/episode.json",
            "media_type": "application/json",
            "sha256": "1" * 64,
            "schema_version": "0.2.0",
            "lineage_sha256": "2" * 64,
        },
        "initial_objectstate_sha256": "",
        "commanded_action_sha256": "",
        "target_object_id": "target",
        "initial_object_states": states,
        "commanded_action": command,
        "rollout_times_s": list(baselines.SCORING_TIMES),
    }


def bundle() -> dict:
    manifest = json.loads(
        (REPO_ROOT / "learning/baseline-manifest.json").read_text(encoding="utf-8")
    )
    samples = [
        sample(f"group-{group:02d}", branch)
        for group in range(12)
        for branch in baselines.BRANCH_IDS
    ]
    samples.sort(key=lambda item: (item["group_id"], item["branch_id"]))
    values = [
        value
        for item in samples
        for value in (item["initial_object_states"], item["commanded_action"])
    ]
    hashes = baselines.canonical_sha256s("node", values)
    for index, item in zip(range(0, len(hashes), 2), samples, strict=True):
        item["initial_objectstate_sha256"] = hashes[index]
        item["commanded_action_sha256"] = hashes[index + 1]
    return {
        "bundle_version": "0.1.0",
        "bundle_kind": baselines.BUNDLE_KIND,
        "source_commit": SOURCE_COMMIT,
        "experiment_id": "pr02-objectstate-baseline-v0",
        "split": "validation",
        "inputs": {
            "data_boundary_manifest_sha256": manifest["frozen_inputs"][
                "data_boundary_manifest"
            ]["sha256"],
            "formal_data_spec_sha256": manifest["frozen_inputs"]["formal_data_spec"][
                "sha256"
            ],
            "dynamics_experiment_sha256": "5" * 64,
            "source_plan_sha256": "6" * 64,
            "source_report_sha256": "7" * 64,
            "data_index_sha256": "8" * 64,
            "model_input_index_sha256": "9" * 64,
            "loader_report_sha256": "a" * 64,
            "runtime_lock_sha256": file_sha256(REPO_ROOT / "learning/uv.lock"),
        },
        "samples": samples,
        "sample_payload_sha256": baselines.canonical_payload_sha256("node", samples),
        "isolation": {
            "visible_fields": [
                "initial_objectstate",
                "commanded_action_schedule",
                "non_future_metadata",
            ],
            "executed_action_is_feature": False,
            "gt_future_read": False,
            "test_materialized": False,
        },
        "claim_boundary": {
            "supported_claim": "sanitized-validation-model-inputs-are-published",
            "excluded_claims": [
                "prediction-produced",
                "trainer-implemented",
                "model-performance",
                "scientific-baseline-comparison",
            ],
        },
    }


class DeterministicBaselineTests(unittest.TestCase):
    def test_copy_state_copies_all_components_at_every_frozen_time(self) -> None:
        source = sample()
        predictions = baselines.predict(source, "copy_state")
        self.assertEqual([item["time_s"] for item in predictions], list(baselines.SCORING_TIMES))
        target = next(item for item in predictions[-1]["objects"] if item["object_id"] == "target")
        self.assertEqual(target["position_W_m"], [1.0, 2.0, 3.0])
        self.assertEqual(target["linear_velocity_W_m_s"], [0.5, -0.25, 0.0])
        self.assertEqual(target["angular_velocity_W_rad_s"], [0.0, 0.0, 0.0])

    def test_constant_velocity_uses_absolute_physical_time(self) -> None:
        predictions = baselines.predict(sample(), "constant_velocity")
        target = next(item for item in predictions[-1]["objects"] if item["object_id"] == "target")
        self.assertEqual(target["position_W_m"], [1.55, 1.725, 3.0])
        self.assertEqual(target["linear_velocity_W_m_s"], [0.5, -0.25, 0.0])

    def test_constant_world_angular_velocity_left_multiplies_orientation(self) -> None:
        quaternion = baselines.integrate_world_quaternion(
            [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, math.pi], 0.5
        )
        self.assertAlmostEqual(quaternion[0], math.sqrt(0.5))
        self.assertAlmostEqual(quaternion[3], math.sqrt(0.5))
        self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0)

    def test_quaternion_sign_is_canonical_and_action_is_not_used(self) -> None:
        source = sample()
        source["initial_object_states"][0]["quaternion_WO_wxyz"] = [-1.0, 0.0, 0.0, 0.0]
        first = baselines.predict(source, "copy_state")
        mutated = deepcopy(source)
        mutated["commanded_action"]["vector_W_N"] = [99.0, 0.0, 0.0]
        second = baselines.predict(mutated, "copy_state")
        self.assertEqual(first, second)
        target = next(item for item in first[0]["objects"] if item["object_id"] == "target")
        self.assertEqual(target["quaternion_WO_wxyz"], [1.0, 0.0, 0.0, 0.0])

    def test_bundle_rejects_test_future_and_checksum_drift(self) -> None:
        valid = bundle()
        self.assertEqual(
            len(baselines.validate_bundle(valid, SOURCE_COMMIT, node="node")), 60
        )
        for mutation, message in (
            (("split", "test"), "validation"),
            (("future_object_states", []), "forbidden"),
            (("sample_payload_sha256", "0" * 64), "checksum"),
        ):
            invalid = deepcopy(valid)
            key, value = mutation
            if key == "future_object_states":
                invalid["samples"][0][key] = value
            else:
                invalid[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                baselines.BaselineInvalidError, message
            ):
                baselines.validate_bundle(invalid, SOURCE_COMMIT, node="node")

        unsafe = deepcopy(valid)
        unsafe["samples"][0]["group_id"] = "../escape"
        unsafe["sample_payload_sha256"] = baselines.canonical_payload_sha256(
            "node", unsafe["samples"]
        )
        with self.assertRaisesRegex(baselines.BaselineInvalidError, "identity"):
            baselines.validate_bundle(unsafe, SOURCE_COMMIT, node="node")

    def test_canonical_payload_hash_matches_node_contract_algorithm(self) -> None:
        predictions = baselines.predict(sample(), "constant_velocity")
        first = baselines.canonical_payload_sha256("node", predictions)
        second = baselines.canonical_payload_sha256("node", json.loads(json.dumps(predictions)))
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[a-f0-9]{64}$")

    def test_missing_node_canonicalizer_is_blocked(self) -> None:
        with self.assertRaises(baselines.CanonicalHashError):
            baselines.canonical_sha256s("/definitely-missing-node", [[]])

    def test_canonical_and_reverse_publication_have_same_semantic_index(self) -> None:
        payload = bundle()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_path = root / "bundle.json"
            bundle_path.write_bytes(strict_json_bytes(payload))
            with patch.dict(os.environ, {"OBJGAUSS_LEARNING_OFFLINE": "1"}):
                canonical = baselines.produce(
                    repo_root=REPO_ROOT,
                    bundle_path=bundle_path,
                    manifest_path=REPO_ROOT / "learning/baseline-manifest.json",
                    output_root=root / "canonical",
                    source_commit=SOURCE_COMMIT,
                    node="node",
                    order="canonical",
                )
                reverse = baselines.produce(
                    repo_root=REPO_ROOT,
                    bundle_path=bundle_path,
                    manifest_path=REPO_ROOT / "learning/baseline-manifest.json",
                    output_root=root / "reverse",
                    source_commit=SOURCE_COMMIT,
                    node="node",
                    order="reverse",
                )
            self.assertEqual(canonical["counts"]["predictions"], 120)
            self.assertEqual(canonical["semantic_index_sha256"], reverse["semantic_index_sha256"])
            self.assertEqual(
                (root / "canonical/index.json").read_bytes(),
                (root / "reverse/index.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
