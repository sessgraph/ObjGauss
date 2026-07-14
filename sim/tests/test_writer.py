from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from objgauss_sim.writer import (
    BranchCandidate,
    ImmutableConflict,
    PublicationError,
    publish_branch,
)


class AtomicWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["OBJGAUSS_REPO_ROOT"]).resolve()
        cls.validator = cls.root / "scripts/validate-pr01-document.mjs"
        cls.node = os.environ.get("OBJGAUSS_NODE", "node")
        cls.fixture = json.loads(
            (cls.root / "contracts/fixtures/pr01a/episode.valid.json").read_text()
        )

    def candidate(self, branch_id: str = "hold") -> BranchCandidate:
        episode = copy.deepcopy(self.fixture)
        identity = episode["identity"]
        identity["experiment_id"] = "pr01c-writer-test"
        identity["group_id"] = "group-writer-test"
        identity["branch_id"] = branch_id
        identity["episode_id"] = f"episode-writer-test-{branch_id}"
        identity["attempt_id"] = f"attempt-writer-test-{branch_id}-1"
        return BranchCandidate(
            episode=episode,
            trajectory={
                "artifact_version": "0.1.0",
                "artifact_kind": "objgauss.trajectory",
                "records": [{"episode_time_s": 0.0, "position_W_m": [0.0, 0.0, 0.0]}],
            },
            contact_ledger={
                "artifact_version": "0.1.0",
                "artifact_kind": "objgauss.contact_ledger",
                "records": [{"episode_time_s": 0.01, "contacts": []}],
            },
            attempt_timing={
                "started_monotonic_s": 1.0,
                "finished_monotonic_s": 1.25,
                "wall_seconds": 0.25,
            },
            timeout_s=10.0,
            attempt_ordinal=1,
            previous_attempt_id={"availability": "missing", "reason": "not_applicable"},
            attempt_provenance={
                "experiment_manifest_sha256": "1" * 64,
                "config_sha256": "2" * 64,
                "runtime_lock_sha256": "3" * 64,
                "source_tree_sha256": "4" * 64,
            },
        )

    def publish(self, candidate: BranchCandidate, root: Path, **kwargs):
        return publish_branch(
            candidate,
            root=root,
            validator=self.validator,
            node=self.node,
            **kwargs,
        )

    def test_publish_then_identical_replay_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.publish(self.candidate(), root)
            second = self.publish(self.candidate(), root)
            self.assertEqual(first["status"], "published")
            self.assertEqual(second["status"], "noop")
            self.assertEqual(first["semantic_sha256"], second["semantic_sha256"])

    def test_different_content_for_same_key_is_immutable_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.candidate()
            self.publish(baseline, root)
            changed = self.candidate()
            changed.episode["evidence"]["terminal_state_sha256"] = "a" * 64
            with self.assertRaises(ImmutableConflict):
                self.publish(changed, root)

    def test_injected_mid_write_failure_publishes_no_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.candidate("push-pos-x-weak")
            with self.assertRaises(PublicationError):
                self.publish(candidate, root, fail_after_artifacts=True)
            final = (
                root
                / "dataset/pr01c-writer-test/group-writer-test/push-pos-x-weak"
            )
            self.assertFalse(final.exists())
            attempts = list((root / "attempts").rglob("*.json"))
            self.assertEqual(len(attempts), 1)
            attempt = json.loads(attempts[0].read_text())
            self.assertEqual(attempt["outcome"]["reason_code"], "atomic_write_failure")
            self.assertFalse(attempt["publication"]["final_episode_published"])
            self.assertFalse(any(path.name.startswith(".tmp-") for path in root.rglob("*")))

    def test_non_finite_artifact_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.candidate()
            candidate.trajectory["records"][0]["position_W_m"][0] = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.publish(candidate, root)
            self.assertFalse((root / "dataset").exists())

    def test_contract_failure_is_validation_and_not_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.candidate("push-pos-y-weak")
            candidate.episode["environment"]["target_object_id"] = "different-target"
            with self.assertRaises(PublicationError):
                self.publish(candidate, root)
            self.assertFalse(
                (
                    root
                    / "dataset/pr01c-writer-test/group-writer-test/push-pos-y-weak"
                ).exists()
            )
            attempts = list((root / "attempts").rglob("*.json"))
            self.assertEqual(len(attempts), 1)
            attempt = json.loads(attempts[0].read_text())
            self.assertEqual(attempt["outcome"]["classification"], "validation")
            self.assertEqual(attempt["outcome"]["reason_code"], "schema_invalid")
            self.assertFalse(attempt["retry"]["eligible"])

    def test_infrastructure_failure_remains_visible_after_one_same_seed_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.candidate("push-pos-x-weak")
            first_id = first.episode["identity"]["attempt_id"]
            with self.assertRaises(PublicationError):
                self.publish(first, root, fail_after_artifacts=True)

            failed_path = next((root / "attempts").rglob("*.json"))
            failed = json.loads(failed_path.read_text())
            self.assertEqual(failed["identity"]["attempt_id"], first_id)
            self.assertTrue(failed["retry"]["eligible"])
            self.assertTrue(failed["retry"]["seed_reused"])

            retry = self.candidate("push-pos-x-weak")
            retry_id = first_id.removesuffix("-1") + "-2"
            retry.episode["identity"]["attempt_id"] = retry_id
            retry = replace(
                retry,
                attempt_ordinal=2,
                previous_attempt_id={"availability": "present", "value": first_id},
            )
            result = self.publish(retry, root)
            self.assertEqual(result["status"], "published")

            success_path = (
                root
                / "dataset/pr01c-writer-test/group-writer-test/push-pos-x-weak/attempt.json"
            )
            success = json.loads(success_path.read_text())
            self.assertEqual(success["identity"]["attempt_id"], retry_id)
            self.assertEqual(success["retry"]["ordinal"], 2)
            self.assertFalse(success["retry"]["eligible"])
            self.assertEqual(
                success["retry"]["previous_attempt_id"],
                {"availability": "present", "value": first_id},
            )
            self.assertTrue(failed_path.is_file(), "first failure must not be deleted")


if __name__ == "__main__":
    unittest.main()
