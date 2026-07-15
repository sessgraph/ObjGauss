from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from objgauss_sim.pr02_data import (
    ALLOWED_SPLITS,
    BRANCH_IDS,
    build_source_plan,
    group_config,
    validate_requested_splits,
)


class Pr02DataDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["OBJGAUSS_REPO_ROOT"])
        cls.manifest = json.loads(
            (cls.root / "learning/data-boundary-manifest.json").read_text()
        )

    def test_only_train_and_validation_are_allowed(self) -> None:
        self.assertEqual(validate_requested_splits(["validation", "train"]), ALLOWED_SPLITS)
        for values in (["test"], ["train", "test"], ["train", "train"], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_requested_splits(values)

    def test_manifest_freezes_counts_branches_and_final_exclusion(self) -> None:
        materialization = self.manifest["materialization"]
        self.assertEqual(materialization["allowed_splits"], list(ALLOWED_SPLITS))
        self.assertEqual(materialization["forbidden_split"], "test")
        self.assertEqual(materialization["group_counts"], {"train": 48, "validation": 12})
        self.assertEqual(materialization["branch_ids"], list(BRANCH_IDS))

    def test_manifest_committed_input_hashes_match(self) -> None:
        import hashlib

        for name in ("pilot_spec", "source_experiment", "simulator_lock"):
            entry = self.manifest["frozen_inputs"][name]
            observed = hashlib.sha256((self.root / entry["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, entry["sha256"], name)

    def test_source_plan_never_materializes_test(self) -> None:
        formal = {
            "experiment_id": "experiment",
            "actions": [],
            "partitions": {"train": {"groups": [{}]}, "validation": {"groups": [{}]}},
        }
        source = {
            "runtime": {"lock_sha256": "0" * 64},
            "thresholds": {},
            "budgets": {"branch_timeout_s": 10},
        }
        plan = build_source_plan(
            formal=formal,
            formal_sha256="1" * 64,
            pilot_sha256="2" * 64,
            source_experiment=source,
            lock_sha256="3" * 64,
            requested_splits=ALLOWED_SPLITS,
        )
        self.assertFalse(plan["materialization"]["test_materialized"])
        self.assertEqual(plan["materialization"]["splits"], list(ALLOWED_SPLITS))

    def test_group_config_rejects_test_and_projects_frozen_identity(self) -> None:
        group = {
            "group_id": "group-a",
            "object_identity_id": "object-a",
            "layout_id": "layout-a",
            "start_pose_id": "start-a",
            "reset_seed": 7,
        }
        partition = {
            "objects": {"object-a": {"half_size_m": [0.02, 0.02, 0.02], "density_kg_m3": 500}},
            "layouts": {"layout-a": {"context_xy_m": [0.4, 0.4]}},
            "starts": {"start-a": {"target_base_xy_m": [0, 0], "target_jitter_m": 0.02}},
        }
        formal = {"partitions": {"train": partition}}
        plan = {"identity": {"experiment_id": "experiment", "fixture_id": "fixture"}}
        config = group_config(
            formal=formal,
            formal_sha256="a" * 64,
            plan=plan,
            split="train",
            group=group,
            source_commit="b" * 40,
            asset_manifest_sha256="c" * 64,
        )
        self.assertEqual(config["object_spec_id"], "object-a")
        self.assertEqual(config["split"], "train")
        with self.assertRaises(ValueError):
            group_config(
                formal={"partitions": {"test": partition}},
                formal_sha256="a" * 64,
                plan=plan,
                split="test",
                group=group,
                source_commit="b" * 40,
                asset_manifest_sha256="c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
