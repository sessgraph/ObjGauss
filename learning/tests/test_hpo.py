"""Behavior and failure tests for the frozen C6 planner and independent selector."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from objgauss_learning import hpo, selector
from objgauss_learning.model import canonicalize_quaternion_tensor
from objgauss_learning.trainer import _tiny_branch


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "learning/hpo-manifest.json"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tiny_pair_inputs() -> tuple[dict, dict, dict[str, dict[str, tuple]]]:
    fixture = json.loads((ROOT / "learning/fixtures/pr02c-tiny-training.json").read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    trainer_manifest = json.loads((ROOT / "learning/trainer-manifest.json").read_text())
    symmetries = canonicalize_quaternion_tensor(
        torch.tensor(fixture["symmetry_rotations_wxyz"], dtype=torch.float32)
    )
    groups: dict[str, dict[str, tuple]] = {"train": {}, "validation": {}}
    for group in fixture["groups"]:
        groups[group["split"]][group["group_id"]] = tuple(
            _tiny_branch(
                raw,
                group_id=group["group_id"],
                split=group["split"],
                symmetries=symmetries,
                device=torch.device("cpu"),
            )
            for raw in group["branches"]
        )
    return manifest, trainer_manifest, groups


def valid_index(root: Path) -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    validation_ids = [f"validation-group-{index:02d}" for index in range(12)]
    tasks = []
    configs = {item["config_id"]: item for item in manifest["matrix"]["configurations"]}
    for config_rank, pair in enumerate(manifest["matrix"]["fairness_pairs"]):
        pair_digest = digest(pair["pair_id"].encode())
        order_digest = digest(f"order:{pair['training_seed']}".encode())
        for arm_rank, arm in enumerate(("action_free", "action_conditioned")):
            task_id = pair["task_ids"][arm]
            artifact = root / f"{task_id}.json"
            artifact.write_text(json.dumps({"task_id": task_id}))
            config_id = pair["config_id"]
            base = list(configs).index(config_id) + arm_rank * 0.01
            group_errors = [
                {"group_id": group_id, "primary_error": base + index * 0.001}
                for index, group_id in enumerate(validation_ids)
            ]
            score = math.fsum(item["primary_error"] for item in group_errors) / 12.0
            tasks.append(
                {
                    "task_id": task_id,
                    "pair_id": pair["pair_id"],
                    "model_arm": arm,
                    "config_id": config_id,
                    "config_sha256": configs[config_id]["config_sha256"],
                    "training_seed": pair["training_seed"],
                    "status": "completed",
                    "attempt_ids": [f"attempt-{task_id}-a01"],
                    "hpo_data_index_sha256": "d" * 64,
                    "validation_group_errors": group_errors,
                    "validation_primary_error": score,
                    "fairness": {
                        "initialization_seed": pair["training_seed"],
                        "initialization_algorithm": "torch-manual-seed-deterministic-v1",
                        "common_parameter_names_sha256": "1" * 64,
                        "common_parameter_subtree_sha256": pair_digest,
                        "arm_specific_parameter_names_sha256": "2" * 64,
                        "arm_specific_parameter_subtree_sha256": digest(f"{pair['pair_id']}:{arm}".encode()),
                        "data_order_sha256": "3" * 64,
                        "batch_group_sequence_sha256": order_digest,
                        "optimizer_updates": 80,
                        "epochs": 20,
                        "training_budget_sha256": "4" * 64,
                        "checkpoint_policy": "minimum-validation-primary-error-per-seed",
                    },
                    "artifacts": [
                        {"uri": artifact.name, "sha256": digest(artifact.read_bytes())}
                    ],
                }
            )
    tasks.sort(key=lambda item: item["task_id"])
    index = {
        "index_version": "0.1.0",
        "index_kind": "objgauss.pr02c-c6-task-index",
        "runner_commit": "a" * 40,
        "workflow_commit": "a" * 40,
        "hpo_data_build_commit": "a" * 40,
        "trainer_contract_commit": manifest["lineage"]["trainer_contract_commit"],
        "hpo_manifest_sha256": digest(MANIFEST_PATH.read_bytes()),
        "hpo_data_index_sha256": "d" * 64,
        "validation_group_ids": validation_ids,
        "splits": ["train", "validation"],
        "test_materialized": False,
        "tasks": tasks,
    }
    return manifest, index


class HpoContractTests(unittest.TestCase):
    def test_manifest_expands_to_frozen_24_tasks_and_12_pairs(self) -> None:
        contract = hpo.validate_hpo_manifest(ROOT, MANIFEST_PATH, "node")
        self.assertEqual(len(contract["configs"]), 4)
        self.assertEqual(len(contract["pairs"]), 12)
        self.assertEqual(
            len({task_id for pair in contract["pairs"] for task_id in pair["task_ids"].values()}),
            24,
        )

    def test_sealed_config_mutation_is_invalid(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text())
        manifest["matrix"]["configurations"][0]["values"]["learning_rate"] = 0.2
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(hpo.TrainerInvalidError):
                hpo.validate_hpo_manifest(ROOT, path, "node")

    def test_rehashed_config_or_task_id_mutation_is_invalid(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text())
        config = manifest["matrix"]["configurations"][0]
        config["values"]["learning_rate"] = 0.2
        config["config_sha256"] = hpo.canonical_sha256s("node", [config["values"]])[0]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(hpo.TrainerInvalidError):
                hpo.validate_hpo_manifest(ROOT, path, "node")
        manifest = json.loads(MANIFEST_PATH.read_text())
        manifest["matrix"]["fairness_pairs"][0]["task_ids"]["action_free"] += "-forged"
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(hpo.TrainerInvalidError):
                hpo.validate_hpo_manifest(ROOT, path, "node")

    def test_runner_rejects_final_test_before_reading_inputs(self) -> None:
        with patch.object(hpo.sys, "stdout", SimpleNamespace(buffer=BytesIO())):
            code = hpo.main(["--mode", "contract", "--splits", "test"])
        self.assertEqual(code, 4)

    def test_cpu_tiny_pair_locks_initialization_updates_and_batch_order(self) -> None:
        manifest, trainer_manifest, groups = tiny_pair_inputs()
        result = hpo.fit_pair(
            config=manifest["matrix"]["configurations"][0],
            seed=2026071501,
            train_groups=groups["train"],
            validation_groups=groups["validation"],
            scales=trainer_manifest["loss"]["normalization_scales"],
            device=torch.device("cpu"),
        )
        fits = list(result.fits.values())
        self.assertEqual(fits[0].optimizer_updates, fits[1].optimizer_updates)
        self.assertEqual(fits[0].epochs, fits[1].epochs)
        self.assertEqual(
            result.initialization["action_free"]["common_parameter_subtree_sha256"],
            result.initialization["action_conditioned"]["common_parameter_subtree_sha256"],
        )
        self.assertEqual(
            result.initialization["action_free"]["arm_specific_parameter_names_sha256"],
            digest(hpo.strict_json_bytes(["action_free_mask_token"])),
        )
        self.assertEqual(
            fits[0].training_log["batch_group_sequence_sha256"],
            fits[1].training_log["batch_group_sequence_sha256"],
        )

    def test_pair_retry_keeps_the_same_frozen_config_and_seed(self) -> None:
        sentinel = object()
        manifest = json.loads(MANIFEST_PATH.read_text())
        config = manifest["matrix"]["configurations"][0]
        arguments = {
            "config": config,
            "seed": 2026071501,
            "train_groups": {},
            "validation_groups": {},
            "scales": {},
            "device": torch.device("cpu"),
        }
        with patch.object(
            hpo,
            "fit_pair",
            side_effect=[torch.OutOfMemoryError("transient"), sentinel],
        ) as mocked:
            result, failures = hpo.fit_pair_with_retry(**arguments)
        self.assertIs(result, sentinel)
        self.assertEqual([item.reason_code for item in failures], ["transient_oom"])
        self.assertEqual(mocked.call_count, 2)
        for call in mocked.call_args_list:
            self.assertIs(call.kwargs["config"], config)
            self.assertEqual(call.kwargs["seed"], 2026071501)

    def test_scientific_failure_is_never_retried(self) -> None:
        with patch.object(
            hpo,
            "fit_pair",
            side_effect=hpo.TrainerRejectedError("non-finite"),
        ) as mocked:
            with self.assertRaises(hpo.TrainerRejectedError):
                hpo.fit_pair_with_retry(
                    config={},
                    seed=2026071501,
                    train_groups={},
                    validation_groups={},
                    scales={},
                    device=torch.device("cpu"),
                )
        self.assertEqual(mocked.call_count, 1)

    def test_hpo_checkpoint_is_auditable_but_not_promoted(self) -> None:
        manifest, trainer_manifest, groups = tiny_pair_inputs()
        config = manifest["matrix"]["configurations"][0]
        pair = manifest["matrix"]["fairness_pairs"][0]
        result = hpo.fit_pair(
            config=config,
            seed=pair["training_seed"],
            train_groups=groups["train"],
            validation_groups=groups["validation"],
            scales=trainer_manifest["loss"]["normalization_scales"],
            device=torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = hpo.publish_task(
                repo_root=ROOT,
                root=root,
                task_id=pair["task_ids"]["action_free"],
                pair_id=pair["pair_id"],
                arm="action_free",
                config=config,
                seed=pair["training_seed"],
                fit=result.fits["action_free"],
                pair_fit=result,
                validation_groups=groups["validation"],
                runner_commit="a" * 40,
                dynamics_sha256="b" * 64,
                runtime_lock_sha256="c" * 64,
                grid_sha256="d" * 64,
                hpo_data_index_sha256="e" * 64,
                node="node",
                failed_attempts=(
                    hpo.FailedPairAttempt(
                        reason_code="transient_oom",
                        message="C6 retryable transient_oom on fairness pair attempt 1.",
                        started_monotonic_s=1.0,
                        finished_monotonic_s=2.0,
                        minimum_display_vram_free_bytes=hpo.DISPLAY_RESERVE_BYTES,
                        peak_vram_bytes=0,
                    ),
                ),
            )
            task_root = root / "tasks" / task["task_id"]
            self.assertFalse((task_root / "checkpoint-manifests").exists())
            self.assertFalse(task["checkpoint_manifest_published"])
            checkpoint = next(task_root.glob("checkpoints/*.pt"))
            self.assertEqual(
                task["attempt_ids"],
                [
                    f"attempt-{task['task_id']}-a01",
                    f"attempt-{task['task_id']}-a02",
                ],
            )
            failed_attempt = json.loads(
                (task_root / "attempts" / f"{task['attempt_ids'][0]}.json").read_text()
            )
            attempt = json.loads(
                (task_root / "attempts" / f"{task['attempt_ids'][1]}.json").read_text()
            )
            self.assertEqual(failed_attempt["outcome"]["reason_code"], "transient_oom")
            self.assertTrue(failed_attempt["retry"]["eligible"])
            self.assertEqual(attempt["identity"]["ordinal"], 2)
            self.assertEqual(
                attempt["retry"]["previous_attempt_id"]["value"], task["attempt_ids"][0]
            )
            self.assertEqual(task["checkpoint_sha256"], hpo.file_sha256(checkpoint))
            self.assertEqual(
                attempt["outputs"]["checkpoint"]["value"]["sha256"],
                task["checkpoint_sha256"],
            )
            public_documents = [
                *task_root.glob("attempts/*.json"),
                *task_root.glob("trials/*.json"),
            ]
            schema_probe = subprocess.run(
                [
                    "node",
                    "--input-type=module",
                    "--eval",
                    (
                        "import {readFileSync} from 'node:fs';"
                        "import {validateContract} from './src/pr01/contract-dispatch.mjs';"
                        "for (const path of process.argv.slice(1)) {"
                        "const result=validateContract(JSON.parse(readFileSync(path,'utf8')));"
                        "if (!result.valid) { console.error(JSON.stringify({path,result})); process.exit(1); }"
                        "}"
                    ),
                    *(str(path) for path in public_documents),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(schema_probe.returncode, 0, schema_probe.stderr)


class IndependentSelectorTests(unittest.TestCase):
    def test_canonical_reverse_replay_freezes_same_unique_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, index = valid_index(root)
            canonical, report_a = selector.select_records(
                manifest=manifest, index=index, artifact_root=root, input_order="canonical"
            )
            reverse, report_b = selector.select_records(
                manifest=manifest, index=index, artifact_root=root, input_order="reverse"
            )
            self.assertEqual(canonical, reverse)
            self.assertEqual(
                report_a["selection_semantic_sha256"], report_b["selection_semantic_sha256"]
            )
            self.assertEqual(
                canonical["mapping"],
                {
                    "action_free": {
                        "config_id": "hpo-h064-lr0p0003",
                        "config_sha256": "7197afb5f09087ae6e01699be1fe9382e5016a3f78d4e3f160cf4f15c52bb5f2",
                        "selection_score": 0.0055000000000000005,
                    },
                    "action_conditioned": {
                        "config_id": "hpo-h064-lr0p0003",
                        "config_sha256": "7197afb5f09087ae6e01699be1fe9382e5016a3f78d4e3f160cf4f15c52bb5f2",
                        "selection_score": 0.0155,
                    },
                },
            )

    def test_pair_initialization_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, index = valid_index(root)
            pair_id = index["tasks"][0]["pair_id"]
            pair = [task for task in index["tasks"] if task["pair_id"] == pair_id]
            pair[1]["fairness"]["common_parameter_subtree_sha256"] = "f" * 64
            with self.assertRaises(selector.SelectionRejectedError):
                selector.select_records(
                    manifest=manifest, index=index, artifact_root=root, input_order="canonical"
                )

    def test_missing_duplicate_or_forged_task_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, index = valid_index(root)
            missing = copy.deepcopy(index)
            missing["tasks"].pop()
            with self.assertRaises(selector.SelectionInvalidError):
                selector.select_records(
                    manifest=manifest, index=missing, artifact_root=root, input_order="canonical"
                )
            duplicate = copy.deepcopy(index)
            duplicate["tasks"][-1] = copy.deepcopy(duplicate["tasks"][0])
            with self.assertRaises(selector.SelectionInvalidError):
                selector.select_records(
                    manifest=manifest, index=duplicate, artifact_root=root, input_order="canonical"
                )
            forged = copy.deepcopy(index)
            forged["tasks"][0]["training_seed"] = 99
            with self.assertRaises(selector.SelectionInvalidError):
                selector.select_records(
                    manifest=manifest, index=forged, artifact_root=root, input_order="canonical"
                )

    def test_checksum_or_test_split_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, index = valid_index(root)
            corrupted = copy.deepcopy(index)
            corrupted["tasks"][0]["artifacts"][0]["sha256"] = "e" * 64
            with self.assertRaises(selector.SelectionInvalidError):
                selector.select_records(
                    manifest=manifest, index=corrupted, artifact_root=root, input_order="canonical"
                )
            leaked = copy.deepcopy(index)
            leaked["splits"] = ["train", "validation", "test"]
            leaked["test_materialized"] = True
            with self.assertRaises(selector.SelectionInvalidError):
                selector.select_records(
                    manifest=manifest, index=leaked, artifact_root=root, input_order="canonical"
                )

    def test_exhausted_arm_is_blocked_not_force_selected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, index = valid_index(root)
            for task in index["tasks"]:
                if task["model_arm"] == "action_conditioned":
                    task["status"] = "failed"
            with self.assertRaises(selector.SelectionBlockedError):
                selector.select_records(
                    manifest=manifest, index=index, artifact_root=root, input_order="canonical"
                )

    def test_selector_cli_rejects_test_with_exit_four(self) -> None:
        with patch.object(selector.sys, "stdout", SimpleNamespace(buffer=BytesIO())):
            code = selector.main(
                [
                    "--task-index",
                    "missing.json",
                    "--artifact-root",
                    ".",
                    "--output-root",
                    "unused",
                    "--splits",
                    "test",
                ]
            )
        self.assertEqual(code, 4)


if __name__ == "__main__":
    unittest.main()
