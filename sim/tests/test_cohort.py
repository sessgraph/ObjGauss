from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from collections import Counter
from pathlib import Path

from objgauss_sim.adapter import SOURCE_COMMIT_POLICY, validated_source_commit
from objgauss_sim.cohort import assigned_split, design_groups, validate_spec


class CohortDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(os.environ["OBJGAUSS_REPO_ROOT"])
        cls.spec_path = root / "contracts/fixtures/pr01e/cohort-spec.json"
        cls.manifest_path = root / "contracts/fixtures/pr01e/experiment.formal.json"
        cls.spec = json.loads(cls.spec_path.read_text())
        cls.manifest = json.loads(cls.manifest_path.read_text())
        cls.measured_spec_path = (
            root / "contracts/fixtures/pr01e/cohort-spec-preflight-measured.json"
        )
        cls.measured_spec = json.loads(cls.measured_spec_path.read_text())
        cls.freeze = json.loads(
            (root / "contracts/fixtures/pr01e/preflight-freeze.json").read_text()
        )

    def test_spec_and_manifest_are_exactly_aligned(self) -> None:
        import hashlib

        validate_spec(
            self.spec,
            self.manifest,
            hashlib.sha256(self.spec_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(self.spec["source_commit_policy"], SOURCE_COMMIT_POLICY)
        self.assertNotIn("source_commit", self.spec)

    def test_source_commit_is_runtime_bound_and_strict(self) -> None:
        commit = "a" * 40
        self.assertEqual(validated_source_commit(commit), commit)
        for invalid in ("", "A" * 40, "a" * 39, "g" * 40, "HEAD"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validated_source_commit(invalid)

    def test_runtime_lineage_policy_does_not_change_frozen_scientific_spec(self) -> None:
        import hashlib

        active = deepcopy(self.spec)
        measured = deepcopy(self.measured_spec)
        self.assertEqual(active.pop("source_commit_policy"), SOURCE_COMMIT_POLICY)
        self.assertEqual(
            measured.pop("source_commit"),
            "b4107fa4fb2747294daf616a01bede8def99a378",
        )
        self.assertEqual(active, measured)
        lineage = self.freeze["lineage"]
        self.assertEqual(
            hashlib.sha256(self.measured_spec_path.read_bytes()).hexdigest(),
            lineage["measurement_experiment_spec_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(self.spec_path.read_bytes()).hexdigest(),
            lineage["active_experiment_spec_sha256"],
        )
        self.assertFalse(
            lineage["provenance_only_supersession"][
                "scientific_configuration_changed"
            ]
        )

    def test_preflight_is_twelve_groups_and_excluded(self) -> None:
        groups = design_groups(self.spec, self.manifest, "preflight")
        self.assertEqual(len(groups), 12)
        self.assertEqual({group["split"] for group in groups}, {"preflight"})
        self.assertTrue(
            set(self.spec["preflight_reset_seeds"]).isdisjoint(
                self.spec["formal_reset_seeds"]
            )
        )

    def test_formal_design_is_group_first_24_12_12(self) -> None:
        groups = design_groups(self.spec, self.manifest, "formal")
        self.assertEqual(len(groups), 48)
        self.assertEqual(
            Counter(group["split"] for group in groups),
            Counter({"train": 24, "validation": 12, "test": 12}),
        )
        self.assertEqual(len({group["group_id"] for group in groups}), 48)

    def test_seed_rank_split_is_order_independent(self) -> None:
        kwargs = {
            "object_spec_id": "box-a",
            "layout_id": "layout-a",
            "start_pose_id": "start-a",
            "seed": 24071401,
            "allocation": {"train": 2, "validation": 1, "test": 1},
        }
        forward = assigned_split(seeds=[24071401, 24071402, 24071403, 24071404], **kwargs)
        reverse = assigned_split(seeds=[24071404, 24071403, 24071402, 24071401], **kwargs)
        self.assertEqual(forward, reverse)


if __name__ == "__main__":
    unittest.main()
