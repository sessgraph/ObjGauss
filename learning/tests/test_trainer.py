from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from objgauss_learning import checkpoint, trainer
from objgauss_learning.model import MinimalObjectGNN


REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainerTests(unittest.TestCase):
    def test_manifest_freezes_golden_without_selecting_hpo_or_formal(self) -> None:
        manifest = trainer.read_json(REPO_ROOT / "learning/trainer-manifest.json")
        trainer.validate_manifest(REPO_ROOT, manifest)
        self.assertEqual(manifest["arms"], ["action_free", "action_conditioned"])
        self.assertEqual(manifest["rollout"]["delta_t_s"], [0.1, 0.1, 0.3, 0.6])
        self.assertFalse(manifest["golden"]["hpo_config_selected"])
        self.assertFalse(manifest["golden"]["formal_checkpoint_frozen"])
        self.assertIn("test-source-or-prediction-produced", manifest["claim_boundary"]["excluded_claims"])

    def test_cpu_tiny_trains_both_arms_with_fair_counts_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"OBJGAUSS_LEARNING_OFFLINE": "1"}
        ):
            report = trainer.run_tiny(
                repo_root=REPO_ROOT,
                fixture_path=REPO_ROOT / "learning/fixtures/pr02c-tiny-training.json",
                manifest_path=REPO_ROOT / "learning/trainer-manifest.json",
                output=Path(temporary) / "tiny-report.json",
            )
        self.assertEqual(report["verdict"], "supported")
        self.assertEqual([item["model_arm"] for item in report["arms"]], list(trainer.ARMS))
        self.assertTrue(all(report["fairness"].values()))
        self.assertEqual(report["rollout"]["delta_t_s"], [0.1, 0.1, 0.3, 0.6])

    def test_trainer_and_independent_checkpoint_semantics_match(self) -> None:
        torch.manual_seed(23)
        model = MinimalObjectGNN(hidden_width=64, arm="action_conditioned")
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        produced = trainer.tensor_state_semantic_sha256(state)
        inspected = checkpoint.inspect_state_dict(state)
        self.assertEqual(produced, inspected["semantic_sha256"])
        self.assertEqual(inspected["parameter_count"], sum(value.numel() for value in state.values()))

    def test_checkpoint_inspector_rejects_non_tensor_or_non_finite_state(self) -> None:
        with self.assertRaises(checkpoint.CheckpointInvalidError):
            checkpoint.inspect_state_dict({"weight": "not-a-tensor"})
        with self.assertRaises(checkpoint.CheckpointInvalidError):
            checkpoint.inspect_state_dict({"weight": torch.tensor([float("nan")])})

    def test_cli_rejects_test_split_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"OBJGAUSS_LEARNING_OFFLINE": "1"}
        ):
            output = Path(temporary) / "report.json"
            exit_code = trainer.main(
                [
                    "--repo-root",
                    str(REPO_ROOT),
                    "--mode",
                    "tiny",
                    "--output",
                    str(output),
                    "--splits",
                    "test",
                ]
            )
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 4)
        self.assertEqual(report["verdict"], "invalid")
        self.assertIn("final test", report["message"])


if __name__ == "__main__":
    unittest.main()
