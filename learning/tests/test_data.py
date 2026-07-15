from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from objgauss_learning import data
from objgauss_learning.runtime import strict_json_bytes


def state() -> dict:
    return {
        "position_W_m": [0.0, 0.0, 0.02],
        "quaternion_WO_wxyz": [1.0, 0.0, 0.0, 0.0],
        "linear_velocity_W_m_s": [0.0, 0.0, 0.0],
        "angular_velocity_W_rad_s": [0.0, 0.0, 0.0],
    }


def trajectory() -> dict:
    times = [0.0, 0.1, 0.2, 0.5, 1.1]
    return {
        "records": [
            {"episode_time_s": time, "actors": {"context": state(), "target": state()}}
            for time in times
        ]
    }


def episode(split: str = "train") -> dict:
    action = {
        "kind": "hold",
        "vector_W_N": [0.0, 0.0, 0.0],
        "duration_s": 0.1,
        "sim_frequency_hz": 100,
        "applied_steps": 10,
    }
    return {
        "identity": {"group_id": "group-a", "branch_id": "hold", "split": split},
        "initialization": {
            "reset_seed": 7,
            "snapshot_sha256": "1" * 64,
            "initial_state_sha256": "2" * 64,
            "restored_rng_sha256": "3" * 64,
        },
        "intervention": {
            "target_object_id": "target",
            "commanded_action": action,
            "executed_action": deepcopy(action),
            "control_ledger": {"applied_steps": 10},
        },
        "environment": {
            "object_spec_id": "object-a",
            "layout_id": "layout-a",
            "start_pose_id": "start-a",
        },
        "evidence": {},
        "provenance": {"source_commit": "a" * 40},
    }


class DataBoundaryTests(unittest.TestCase):
    def test_test_split_is_rejected(self) -> None:
        for values in (("test",), ("train", "test"), tuple()):
            with self.subTest(values=values), self.assertRaises(data.DataInvalidError):
                data.require_splits(values)

    def test_model_inputs_exclude_executed_action_control_and_future(self) -> None:
        source = episode()
        first = data.model_payload(data.build_model_inputs(source, trajectory()))
        source["intervention"]["executed_action"]["vector_W_N"] = [9, 9, 9]
        source["intervention"]["control_ledger"]["applied_steps"] = 0
        second = data.model_payload(data.build_model_inputs(source, trajectory()))
        self.assertEqual(first, second)
        serialized = json.dumps(first)
        for forbidden in data.FORBIDDEN_MODEL_FIELDS:
            self.assertNotIn(forbidden, serialized)

    def test_model_payload_rejects_future_or_executed_fields(self) -> None:
        payload = data.model_payload(data.build_model_inputs(episode(), trajectory()))
        for forbidden in data.FORBIDDEN_MODEL_FIELDS:
            mutated = deepcopy(payload)
            mutated[forbidden] = []
            with self.subTest(forbidden=forbidden), self.assertRaises(data.DataInvalidError):
                data.assert_model_payload(mutated)

    def test_commanded_action_is_required(self) -> None:
        source = episode()
        del source["intervention"]["commanded_action"]
        with self.assertRaisesRegex(data.DataInvalidError, "commanded action"):
            data.build_model_inputs(source, trajectory())

    def test_labels_use_only_the_four_frozen_physical_times(self) -> None:
        labels = data.build_labels(trajectory())
        self.assertEqual(
            tuple(item.episode_time_s for item in labels.future_object_states),
            data.SCORING_TIMES,
        )

    def test_missing_score_or_post_horizon_record_is_invalid(self) -> None:
        missing = trajectory()
        missing["records"] = [
            item for item in missing["records"] if item["episode_time_s"] != 0.5
        ]
        with self.assertRaisesRegex(data.DataInvalidError, "scoring time"):
            data.build_labels(missing)
        future = trajectory()
        future["records"].append(
            {"episode_time_s": 1.2, "actors": {"context": state(), "target": state()}}
        )
        with self.assertRaisesRegex(data.DataInvalidError, "horizon"):
            data.build_labels(future)

    def test_non_normalized_quaternion_is_invalid(self) -> None:
        bad = trajectory()
        bad["records"][0]["actors"]["target"]["quaternion_WO_wxyz"] = [2, 0, 0, 0]
        with self.assertRaisesRegex(data.DataInvalidError, "not normalized"):
            data.build_model_inputs(episode(), bad)

    def test_branch_loader_checks_checksums_and_keeps_labels_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "dataset/experiment/group-a/hold"
            directory.mkdir(parents=True)
            source_episode = episode()
            source_trajectory = trajectory()
            contacts = {"records": []}
            attempt = {"provenance": {"experiment_manifest_sha256": "b" * 64}}
            trajectory_bytes = strict_json_bytes(source_trajectory)
            contact_bytes = strict_json_bytes(contacts)
            source_episode["evidence"] = {
                "trajectory": {
                    "uri": "dataset/experiment/group-a/hold/trajectory.json",
                    "sha256": data.sha256_bytes(trajectory_bytes),
                    "byte_length": len(trajectory_bytes),
                    "record_count": len(source_trajectory["records"]),
                },
                "contact_ledger": {
                    "uri": "dataset/experiment/group-a/hold/contact-ledger.json",
                    "sha256": data.sha256_bytes(contact_bytes),
                    "byte_length": len(contact_bytes),
                    "record_count": 0,
                },
            }
            episode_bytes = strict_json_bytes(source_episode)
            attempt_bytes = strict_json_bytes(attempt)
            publication = {
                "episode_sha256": data.sha256_bytes(episode_bytes),
                "trajectory_sha256": data.sha256_bytes(trajectory_bytes),
                "contact_ledger_sha256": data.sha256_bytes(contact_bytes),
                "attempt_sha256": data.sha256_bytes(attempt_bytes),
            }
            publication["semantic_sha256"] = data._semantic_sha(publication)
            for name, payload in (
                ("episode.json", episode_bytes),
                ("trajectory.json", trajectory_bytes),
                ("contact-ledger.json", contact_bytes),
                ("attempt.json", attempt_bytes),
                ("publication.json", strict_json_bytes(publication)),
            ):
                (directory / name).write_bytes(payload)
            expected_group = {
                "object_identity_id": "object-a",
                "layout_id": "layout-a",
                "start_pose_id": "start-a",
                "reset_seed": 7,
            }
            expected_action = {
                "branch_id": "hold",
                "kind": "hold",
                "vector_W_N": [0.0, 0.0, 0.0],
                "duration_s": 0.1,
                "sim_frequency_hz": 100,
                "applied_steps": 10,
            }
            sample, _, payload = data.load_branch(
                data_root=root,
                directory=directory,
                group_id="group-a",
                split="train",
                branch_id="hold",
                expected_group=expected_group,
                expected_action=expected_action,
                source_commit="a" * 40,
                source_plan_sha256="b" * 64,
            )
            self.assertEqual(sample.labels.future_object_states[0].episode_time_s, 0.1)
            self.assertNotIn("future_object_states", payload)
            (directory / "trajectory.json").write_text("{}\n")
            with self.assertRaisesRegex(data.DataInvalidError, "checksum mismatch"):
                data.load_branch(
                    data_root=root,
                    directory=directory,
                    group_id="group-a",
                    split="train",
                    branch_id="hold",
                    expected_group=expected_group,
                    expected_action=expected_action,
                    source_commit="a" * 40,
                    source_plan_sha256="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
