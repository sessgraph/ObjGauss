"""Deterministic paired runner for the frozen PR-02C C6 HPO matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .canonical import CanonicalHashError, canonical_sha256s
from .data import DataBlockedError, DataInvalidError, GroupSample, load_dataset, model_payload
from .model import ARMS, ROLLOUT_TIMES, MinimalObjectGNN
from .runtime import (
    RuntimeInvalidError,
    assert_clean_git_head,
    atomic_write,
    file_sha256,
    package_tree_sha256,
    strict_json_bytes,
)
from .trainer import (
    DISPLAY_RESERVE_BYTES,
    METRIC_NAME,
    SCHEMA_VERSION,
    TRAINING_CAP_BYTES,
    FitResult,
    TensorBranch,
    TrainerBlockedError,
    TrainerInvalidError,
    TrainerRejectedError,
    _artifact,
    _atomic_torch_save,
    _clone_state_dict,
    _formal_symmetries,
    _gpu_free,
    _prediction_payloads,
    configure_determinism,
    open_loop_loss,
    read_json,
    require_offline,
    tensor_branch,
    tensor_state_semantic_sha256,
    validation_effect_error,
)


VERSION = "0.1.0"
MANIFEST_KIND = "objgauss.pr02c-c6-hpo-selection"
TASK_INDEX_KIND = "objgauss.pr02c-c6-task-index"
EXPERIMENT_ID = "pr02-objectstate-baseline-v0"
PAIR_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class PairFit:
    fits: dict[str, FitResult]
    validation_group_errors: dict[str, list[dict[str, Any]]]
    initialization: dict[str, dict[str, str]]
    batch_group_sequence_sha256: str
    training_budget_sha256: str
    started_monotonic_s: float
    finished_monotonic_s: float


@dataclass(frozen=True)
class FailedPairAttempt:
    reason_code: str
    message: str
    started_monotonic_s: float
    finished_monotonic_s: float
    minimum_display_vram_free_bytes: int
    peak_vram_bytes: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_matrix(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if manifest.get("manifest_version") != VERSION or manifest.get("manifest_kind") != MANIFEST_KIND:
        raise TrainerInvalidError("C6 HPO manifest kind or version drift")
    matrix = manifest.get("matrix", {})
    configurations = matrix.get("configurations")
    pairs = matrix.get("fairness_pairs")
    if (
        matrix.get("learned_arms") != list(ARMS)
        or matrix.get("training_seeds") != [2026071501, 2026071502, 2026071503]
        or matrix.get("task_count") != 24
        or matrix.get("pair_count") != 12
        or not isinstance(configurations, list)
        or len(configurations) != 4
        or not isinstance(pairs, list)
        or len(pairs) != 12
    ):
        raise TrainerInvalidError("C6 HPO matrix drift")
    configs = {item.get("config_id"): item for item in configurations if isinstance(item, dict)}
    if len(configs) != 4 or None in configs:
        raise TrainerInvalidError("C6 config IDs are missing or duplicated")
    task_ids: set[str] = set()
    pair_ids: set[str] = set()
    for pair in pairs:
        pair_id = pair.get("pair_id")
        task_map = pair.get("task_ids")
        if (
            not isinstance(pair_id, str)
            or pair_id in pair_ids
            or pair.get("config_id") not in configs
            or pair.get("training_seed") not in matrix["training_seeds"]
            or not isinstance(task_map, dict)
            or set(task_map) != set(ARMS)
        ):
            raise TrainerInvalidError("C6 fairness pair identity drift")
        pair_ids.add(pair_id)
        for task_id in task_map.values():
            if not isinstance(task_id, str) or task_id in task_ids:
                raise TrainerInvalidError("C6 task IDs are missing or duplicated")
            task_ids.add(task_id)
    if len(task_ids) != 24:
        raise TrainerInvalidError("C6 manifest does not expand to 24 tasks")
    return configs, pairs


def validate_hpo_manifest(repo_root: Path, manifest_path: Path, node: str) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    configs, pairs = _manifest_matrix(manifest)
    lineage = manifest.get("lineage", {})
    if lineage.get("trainer_contract_commit") != "4498bd603558ecf80ad0591edab0413ea52acb18":
        raise TrainerInvalidError("C6 trainer contract commit drift")
    for name, entry in lineage.get("frozen_inputs", {}).items():
        path = entry.get("path")
        if not isinstance(path, str) or file_sha256(repo_root / path) != entry.get("sha256"):
            raise TrainerInvalidError(f"C6 frozen input checksum drift: {name}")
    grid = read_json(repo_root / lineage["frozen_inputs"]["hyperparameter_grid"]["path"])
    optimization = grid.get("optimization", {})
    widths = grid.get("architecture", {}).get("hidden_width")
    rates = optimization.get("learning_rate")
    if widths != [64, 128] or rates != [0.001, 0.0003]:
        raise TrainerInvalidError("C6 frozen hyperparameter grid drift")
    expected_values: dict[str, dict[str, Any]] = {}
    for hidden_width in widths:
        for learning_rate in sorted(rates):
            rate_id = "0p0003" if learning_rate == 0.0003 else "0p0010"
            config_id = f"hpo-h{hidden_width:03d}-lr{rate_id}"
            expected_values[config_id] = {
                "batch_size": optimization["batch_size"][0],
                "checkpoint_selection": optimization["checkpoint_selection"],
                "early_stopping_patience_epochs": optimization[
                    "early_stopping_patience_epochs"
                ],
                "epochs_max": optimization["epochs_max"],
                "gradient_clip_norm": optimization["gradient_clip_norm"],
                "hidden_width": hidden_width,
                "learning_rate": learning_rate,
                "optimizer": optimization["optimizer"],
                "optimizer_updates_max": optimization["optimizer_updates_max"],
                "weight_decay": optimization["weight_decay"][0],
            }
    if set(configs) != set(expected_values) or any(
        configs[config_id].get("values") != values
        for config_id, values in expected_values.items()
    ):
        raise TrainerInvalidError("C6 sealed configuration set drift")
    expected_pairs = []
    for config_id in expected_values:
        for seed in manifest["matrix"]["training_seeds"]:
            pair_id = f"pair-{config_id}-s{seed}"
            expected_pairs.append(
                {
                    "pair_id": pair_id,
                    "config_id": config_id,
                    "training_seed": seed,
                    "task_ids": {
                        arm: f"task-{arm.replace('_', '-')}-{config_id}-s{seed}"
                        for arm in ARMS
                    },
                }
            )
    if pairs != expected_pairs:
        raise TrainerInvalidError("C6 sealed fairness pair or task IDs drift")
    expected_config_hashes = canonical_sha256s(
        node, [configs[config_id]["values"] for config_id in sorted(configs)]
    )
    for config_id, expected in zip(sorted(configs), expected_config_hashes, strict=True):
        if configs[config_id].get("config_sha256") != expected:
            raise TrainerInvalidError(f"C6 config digest drift: {config_id}")
    data = manifest.get("data_build", {})
    if (
        data.get("allowed_splits") != ["train", "validation"]
        or data.get("forbidden_split") != "test"
        or data.get("build_count") != 1
        or data.get("shared_by_all_tasks") is not True
    ):
        raise TrainerInvalidError("C6 HPO data build policy drift")
    selector = manifest.get("selector", {})
    if (
        selector.get("forbidden_split") != "test"
        or selector.get("performance_promotion_threshold") is not None
        or selector.get("tie_breaker") != "exact-score-tie-then-config-id-lexicographic"
    ):
        raise TrainerInvalidError("C6 selector policy drift")
    return {
        "manifest": manifest,
        "configs": configs,
        "pairs": pairs,
        "manifest_sha256": file_sha256(manifest_path),
    }


def _convert_groups(
    *,
    groups: tuple[GroupSample, ...],
    formal: dict[str, Any],
    node: str,
    device: torch.device,
) -> dict[str, dict[str, tuple[TensorBranch, ...]]]:
    branches = [branch for group in groups for branch in group.branches]
    payloads = [model_payload(branch.model_inputs) for branch in branches]
    hashes = canonical_sha256s(
        node,
        [
            value
            for payload in payloads
            for value in (payload["initial_object_states"], payload["commanded_action"])
        ],
    )
    converted: dict[str, dict[str, list[TensorBranch]]] = {"train": {}, "validation": {}}
    for index, branch in enumerate(branches):
        if branch.split not in converted:
            raise TrainerInvalidError(f"HPO source split is forbidden: {branch.split}")
        symmetries = _formal_symmetries(formal, branch.split, branch.group_id, device=device)
        item = tensor_branch(
            branch,
            symmetries=symmetries,
            initial_sha256=hashes[index * 2],
            command_sha256=hashes[index * 2 + 1],
            device=device,
        )
        converted[branch.split].setdefault(branch.group_id, []).append(item)
    result: dict[str, dict[str, tuple[TensorBranch, ...]]] = {"train": {}, "validation": {}}
    expected_counts = {"train": 48, "validation": 12}
    for split, by_group in converted.items():
        if len(by_group) != expected_counts[split]:
            raise TrainerInvalidError(f"HPO {split} group count drift")
        for group_id, values in by_group.items():
            values.sort(key=lambda item: item.branch_id)
            if len(values) != 5 or values[0].branch_id != "hold":
                raise TrainerInvalidError("HPO sibling group must contain five sorted branches and hold")
            result[split][group_id] = tuple(values)
    return result


def _group_order(group_ids: tuple[str, ...], seed: int, epoch: int) -> list[str]:
    return sorted(
        group_ids,
        key=lambda group_id: sha256_bytes(strict_json_bytes([seed, epoch, group_id])),
    )


def _initialization(model: MinimalObjectGNN) -> dict[str, str]:
    state = _clone_state_dict(model)
    arm_specific_names = ["action_free_mask_token"]
    if not all(name in state for name in arm_specific_names):
        raise TrainerInvalidError("arm-specific initialization parameter set drift")
    common = {name: value for name, value in state.items() if name not in arm_specific_names}
    arm_specific = {name: state[name] for name in arm_specific_names}
    return {
        "common_parameter_names_sha256": sha256_bytes(strict_json_bytes(sorted(common))),
        "common_parameter_subtree_sha256": tensor_state_semantic_sha256(common),
        "arm_specific_parameter_names_sha256": sha256_bytes(
            strict_json_bytes(arm_specific_names)
        ),
        "arm_specific_parameter_subtree_sha256": tensor_state_semantic_sha256(arm_specific),
    }


def _validation_scores(
    model: MinimalObjectGNN,
    validation_groups: dict[str, tuple[TensorBranch, ...]],
    scales: dict[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    values = []
    for group_id in sorted(validation_groups):
        score = validation_effect_error(model, validation_groups[group_id], scales=scales)
        if not torch.isfinite(score):
            raise TrainerRejectedError("HPO validation primary error is non-finite")
        values.append({"group_id": group_id, "primary_error": float(score.detach().cpu())})
    aggregate = math.fsum(item["primary_error"] for item in values) / len(values)
    return aggregate, values


def fit_pair(
    *,
    config: dict[str, Any],
    seed: int,
    train_groups: dict[str, tuple[TensorBranch, ...]],
    validation_groups: dict[str, tuple[TensorBranch, ...]],
    scales: dict[str, float],
    device: torch.device,
) -> PairFit:
    values = config["values"]
    max_epochs = values["epochs_max"]
    max_updates = values["optimizer_updates_max"]
    patience = values["early_stopping_patience_epochs"]
    groups_per_batch = values["batch_size"] // 5
    if max_epochs != 200 or max_updates != 20000 or patience != 20 or groups_per_batch != 12:
        raise TrainerInvalidError("HPO optimization budget drift")
    models: dict[str, MinimalObjectGNN] = {}
    initialization: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        configure_determinism(seed)
        model = MinimalObjectGNN(hidden_width=values["hidden_width"], arm=arm).to(device)
        models[arm] = model
        initialization[arm] = _initialization(model)
    if initialization[ARMS[0]]["common_parameter_subtree_sha256"] != initialization[ARMS[1]]["common_parameter_subtree_sha256"]:
        raise TrainerInvalidError("fairness pair common initialization digest differs")
    optimizers = {
        arm: torch.optim.AdamW(
            models[arm].parameters(),
            lr=values["learning_rate"],
            weight_decay=values["weight_decay"],
        )
        for arm in ARMS
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    minimum_free = _gpu_free(device)
    if minimum_free < DISPLAY_RESERVE_BYTES:
        raise TrainerBlockedError("HPO cannot preserve 1 GiB display VRAM before training")
    best_scores = {arm: math.inf for arm in ARMS}
    best_states: dict[str, dict[str, Tensor] | None] = {arm: None for arm in ARMS}
    best_epochs = {arm: 0 for arm in ARMS}
    best_updates = {arm: 0 for arm in ARMS}
    best_group_errors: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    entries: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    update_count = 0
    batch_sequence: list[list[str]] = []
    train_ids = tuple(sorted(train_groups))
    started = time.monotonic()
    completed_epoch = 0
    for epoch in range(1, max_epochs + 1):
        order = _group_order(train_ids, seed, epoch)
        epoch_losses: dict[str, list[float]] = {arm: [] for arm in ARMS}
        for offset in range(0, len(order), groups_per_batch):
            batch_ids = order[offset : offset + groups_per_batch]
            batch_sequence.append(batch_ids)
            if update_count >= max_updates:
                break
            for arm in ARMS:
                model = models[arm]
                model.train()
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                losses = [open_loop_loss(model, train_groups[group_id], scales=scales) for group_id in batch_ids]
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise TrainerRejectedError(f"HPO training loss is non-finite: {arm}")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), values["gradient_clip_norm"])
                if not torch.isfinite(gradient):
                    raise TrainerRejectedError(f"HPO gradient norm is non-finite: {arm}")
                optimizer.step()
                epoch_losses[arm].append(float(loss.detach().cpu()))
                minimum_free = min(minimum_free, _gpu_free(device))
            update_count += 1
            if time.monotonic() - started > PAIR_TIMEOUT_SECONDS:
                raise TrainerBlockedError("HPO fairness pair exceeded the 900 second task timeout")
        completed_epoch = epoch
        for arm in ARMS:
            models[arm].eval()
            with torch.no_grad():
                score, group_errors = _validation_scores(models[arm], validation_groups, scales)
            if score < best_scores[arm]:
                best_scores[arm] = score
                best_states[arm] = _clone_state_dict(models[arm])
                best_epochs[arm] = epoch
                best_updates[arm] = update_count
                best_group_errors[arm] = group_errors
            entries[arm].append(
                {
                    "epoch": epoch,
                    "optimizer_updates": update_count,
                    "training_loss": math.fsum(epoch_losses[arm]) / len(epoch_losses[arm]),
                    "validation_primary_error": score,
                }
            )
        if all(epoch - best_epochs[arm] >= patience for arm in ARMS):
            break
        if update_count >= max_updates:
            break
    if any(best_states[arm] is None for arm in ARMS):
        raise TrainerRejectedError("HPO pair did not select both validation checkpoints")
    finished = time.monotonic()
    wall_seconds = finished - started
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if peak_vram > TRAINING_CAP_BYTES:
        raise TrainerInvalidError("HPO pair exceeded the 12 GiB training cap")
    if minimum_free < DISPLAY_RESERVE_BYTES:
        raise TrainerInvalidError("HPO pair violated the 1 GiB display VRAM reserve")
    data_order = [
        f"{group_id}/{branch.branch_id}"
        for group_id in train_ids
        for branch in train_groups[group_id]
    ]
    data_order_sha256 = sha256_bytes(strict_json_bytes(data_order))
    batch_sequence_sha256 = sha256_bytes(strict_json_bytes(batch_sequence))
    budget = {
        "batch_size": values["batch_size"],
        "groups_per_batch": groups_per_batch,
        "optimizer_updates_max": max_updates,
        "epochs_max": max_epochs,
        "early_stopping_patience_epochs": patience,
        "pair_stop": "both-arms-exhaust-patience-or-frozen-max",
        "task_timeout_seconds": 900,
    }
    budget_sha256 = sha256_bytes(strict_json_bytes(budget))
    fits: dict[str, FitResult] = {}
    for arm in ARMS:
        state = best_states[arm]
        assert state is not None
        models[arm].load_state_dict(state, strict=True)
        log = {
            "log_version": VERSION,
            "log_kind": "objgauss.pr02c-c6-hpo-training-log",
            "model_arm": arm,
            "config_id": config["config_id"],
            "training_seed": seed,
            "teacher_forcing": "initial-state-only",
            "rollout_times_s": list(ROLLOUT_TIMES),
            "optimizer": "adamw",
            "optimizer_updates": update_count,
            "epochs": completed_epoch,
            "data_order_sha256": data_order_sha256,
            "batch_group_sequence_sha256": batch_sequence_sha256,
            "training_budget": budget,
            "training_budget_sha256": budget_sha256,
            "best_epoch": best_epochs[arm],
            "best_optimizer_update": best_updates[arm],
            "best_validation_primary_error": best_scores[arm],
            "tensor_state_semantic_sha256": tensor_state_semantic_sha256(state),
            "initialization": initialization[arm],
            "entries": entries[arm],
            "visibility": {
                "commanded_action_present": True,
                "executed_action_is_feature": False,
                "gt_future_is_model_input": False,
                "test_materialized": False,
            },
        }
        fits[arm] = FitResult(
            model=models[arm],
            best_state_dict=state,
            best_epoch=best_epochs[arm],
            best_update=best_updates[arm],
            validation_primary_error=best_scores[arm],
            optimizer_updates=update_count,
            epochs=completed_epoch,
            training_log=log,
            minimum_display_vram_free_bytes=minimum_free,
            peak_vram_bytes=peak_vram,
            wall_seconds=wall_seconds,
        )
    return PairFit(
        fits=fits,
        validation_group_errors=best_group_errors,
        initialization=initialization,
        batch_group_sequence_sha256=batch_sequence_sha256,
        training_budget_sha256=budget_sha256,
        started_monotonic_s=started,
        finished_monotonic_s=finished,
    )


def fit_pair_with_retry(
    *,
    config: dict[str, Any],
    seed: int,
    train_groups: dict[str, tuple[TensorBranch, ...]],
    validation_groups: dict[str, tuple[TensorBranch, ...]],
    scales: dict[str, float],
    device: torch.device,
) -> tuple[PairFit, tuple[FailedPairAttempt, ...]]:
    failures: list[FailedPairAttempt] = []
    for ordinal in (1, 2):
        started = time.monotonic()
        try:
            return (
                fit_pair(
                    config=config,
                    seed=seed,
                    train_groups=train_groups,
                    validation_groups=validation_groups,
                    scales=scales,
                    device=device,
                ),
                tuple(failures),
            )
        except (torch.OutOfMemoryError, OSError) as error:
            finished = time.monotonic()
            reason_code = (
                "transient_oom" if isinstance(error, torch.OutOfMemoryError) else "io_failure"
            )
            minimum_free = _gpu_free(device)
            peak_vram = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            if minimum_free < DISPLAY_RESERVE_BYTES:
                raise TrainerInvalidError(
                    "HPO retry refused because the 1 GiB display VRAM reserve was violated"
                ) from error
            failures.append(
                FailedPairAttempt(
                    reason_code=reason_code,
                    message=f"C6 retryable {reason_code} on fairness pair attempt {ordinal}.",
                    started_monotonic_s=started,
                    finished_monotonic_s=finished,
                    minimum_display_vram_free_bytes=minimum_free,
                    peak_vram_bytes=peak_vram,
                )
            )
            if ordinal == 2:
                raise TrainerBlockedError(
                    f"HPO fairness pair exhausted two legal attempts after {reason_code}"
                ) from error
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise AssertionError("unreachable HPO retry state")


def _provenance(
    *, repo_root: Path, runner_commit: str, dynamics_sha256: str, runtime_lock_sha256: str
) -> dict[str, str]:
    return {
        "source_commit": runner_commit,
        "source_tree_sha256": package_tree_sha256(repo_root),
        "experiment_spec_sha256": dynamics_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
    }


def publish_task(
    *,
    repo_root: Path,
    root: Path,
    task_id: str,
    pair_id: str,
    arm: str,
    config: dict[str, Any],
    seed: int,
    fit: FitResult,
    pair_fit: PairFit,
    validation_groups: dict[str, tuple[TensorBranch, ...]],
    runner_commit: str,
    dynamics_sha256: str,
    runtime_lock_sha256: str,
    grid_sha256: str,
    hpo_data_index_sha256: str,
    node: str,
    failed_attempts: tuple[FailedPairAttempt, ...] = (),
) -> dict[str, Any]:
    task_root = root / "tasks" / task_id
    artifact_prefix = f"tasks/{task_id}"
    if len(failed_attempts) > 1:
        raise TrainerInvalidError("C6 task exceeded the frozen two-attempt limit")
    ordinal = len(failed_attempts) + 1
    attempt_ids = [f"attempt-{task_id}-a{index:02d}" for index in range(1, ordinal + 1)]
    attempt_id = attempt_ids[-1]
    trial_id = f"trial-{task_id}"
    checkpoint_id = f"checkpoint-{task_id}"
    checkpoint_relative = Path("checkpoints") / f"{checkpoint_id}.pt"
    log_relative = Path("logs") / f"{trial_id}.json"
    _atomic_torch_save(task_root / checkpoint_relative, fit.best_state_dict)
    atomic_write(task_root / log_relative, strict_json_bytes(fit.training_log))
    checkpoint_artifact = _artifact(
        artifact_prefix,
        checkpoint_relative,
        task_root / checkpoint_relative,
        "application/vnd.objgauss.pytorch-state-dict",
    )
    log_artifact = _artifact(
        artifact_prefix, log_relative, task_root / log_relative, "application/json"
    )
    attempt_resources = {
        "gpu_hours": fit.wall_seconds / 3600.0,
        "peak_vram_bytes": fit.peak_vram_bytes,
        "minimum_display_vram_free_bytes": fit.minimum_display_vram_free_bytes,
        "artifact_bytes": checkpoint_artifact["byte_size"] + log_artifact["byte_size"],
    }
    resources = {
        "gpu_hours": attempt_resources["gpu_hours"]
        + math.fsum(
            (item.finished_monotonic_s - item.started_monotonic_s) / 3600.0
            for item in failed_attempts
        ),
        "peak_vram_bytes": max(
            [fit.peak_vram_bytes, *(item.peak_vram_bytes for item in failed_attempts)]
        ),
        "minimum_display_vram_free_bytes": min(
            [
                fit.minimum_display_vram_free_bytes,
                *(item.minimum_display_vram_free_bytes for item in failed_attempts),
            ]
        ),
        "artifact_bytes": attempt_resources["artifact_bytes"],
    }
    provenance = _provenance(
        repo_root=repo_root,
        runner_commit=runner_commit,
        dynamics_sha256=dynamics_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
    )
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "contract_kind": "objgauss.training_attempt",
        "identity": {
            "experiment_id": EXPERIMENT_ID,
            "trial_id": trial_id,
            "attempt_id": attempt_id,
            "ordinal": ordinal,
            "model_arm": arm,
            "training_seed": seed,
            "config_sha256": config["config_sha256"],
        },
        "timing": {
            "started_monotonic_s": pair_fit.started_monotonic_s,
            "finished_monotonic_s": pair_fit.finished_monotonic_s,
            "wall_seconds": fit.wall_seconds,
        },
        "outcome": {
            "status": "succeeded",
            "classification": "none",
            "reason_code": "none",
            "message": (
                "C6 preregistered HPO task completed without retry."
                if not failed_attempts
                else "C6 preregistered HPO task completed on the second legal attempt."
            ),
        },
        "retry": {
            "max_attempts": 2,
            "eligible": False,
            "same_seed": True,
            "same_config": True,
            "previous_attempt_id": (
                {"availability": "missing", "reason": "not_applicable"}
                if not failed_attempts
                else {"availability": "present", "value": attempt_ids[-2]}
            ),
        },
        "resources": attempt_resources,
        "outputs": {
            "training_log": {"availability": "present", "value": log_artifact},
            "checkpoint": {"availability": "present", "value": checkpoint_artifact},
        },
        "provenance": provenance,
    }
    trial = {
        "schema_version": SCHEMA_VERSION,
        "contract_kind": "objgauss.training_trial",
        "identity": {
            "experiment_id": EXPERIMENT_ID,
            "trial_id": trial_id,
            "model_arm": arm,
            "config_id": config["config_id"],
            "training_seed": seed,
        },
        "configuration": {
            "config_sha256": config["config_sha256"],
            "hyperparameter_grid_sha256": grid_sha256,
            "optimizer_updates_max": config["values"]["optimizer_updates_max"],
            "epochs_max": config["values"]["epochs_max"],
        },
        "outcome": {
            "status": "completed",
            "reason_code": "completed",
            "optimizer_updates_completed": fit.optimizer_updates,
            "epochs_completed": fit.epochs,
        },
        "selection": {
            "selected": False,
            "split": "validation",
            "metric": METRIC_NAME,
            "validation_primary_error": {"availability": "present", "value": fit.validation_primary_error},
            "checkpoint_id": {"availability": "present", "value": checkpoint_id},
        },
        "attempt_ids": attempt_ids,
        "resources": resources,
        "provenance": provenance,
    }
    failed_documents = []
    for index, failure in enumerate(failed_attempts, 1):
        failed_id = attempt_ids[index - 1]
        failed_documents.append(
            (
                Path("attempts") / f"{failed_id}.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract_kind": "objgauss.training_attempt",
                    "identity": {
                        "experiment_id": EXPERIMENT_ID,
                        "trial_id": trial_id,
                        "attempt_id": failed_id,
                        "ordinal": index,
                        "model_arm": arm,
                        "training_seed": seed,
                        "config_sha256": config["config_sha256"],
                    },
                    "timing": {
                        "started_monotonic_s": failure.started_monotonic_s,
                        "finished_monotonic_s": failure.finished_monotonic_s,
                        "wall_seconds": (
                            failure.finished_monotonic_s - failure.started_monotonic_s
                        ),
                    },
                    "outcome": {
                        "status": "failed",
                        "classification": "infrastructure",
                        "reason_code": failure.reason_code,
                        "message": failure.message,
                    },
                    "retry": {
                        "max_attempts": 2,
                        "eligible": True,
                        "same_seed": True,
                        "same_config": True,
                        "previous_attempt_id": (
                            {"availability": "missing", "reason": "not_applicable"}
                            if index == 1
                            else {"availability": "present", "value": attempt_ids[index - 2]}
                        ),
                    },
                    "resources": {
                        "gpu_hours": (
                            failure.finished_monotonic_s - failure.started_monotonic_s
                        )
                        / 3600.0,
                        "peak_vram_bytes": failure.peak_vram_bytes,
                        "minimum_display_vram_free_bytes": (
                            failure.minimum_display_vram_free_bytes
                        ),
                        "artifact_bytes": 0,
                    },
                    "outputs": {
                        "training_log": {"availability": "missing", "reason": "not_produced"},
                        "checkpoint": {"availability": "missing", "reason": "not_produced"},
                    },
                    "provenance": provenance,
                },
            )
        )
    documents = (
        *failed_documents,
        (Path("attempts") / f"{attempt_id}.json", attempt),
        (Path("trials") / f"{trial_id}.json", trial),
    )
    for relative, document in documents:
        atomic_write(task_root / relative, strict_json_bytes(document))
    validation_branches = tuple(
        branch for group_id in sorted(validation_groups) for branch in validation_groups[group_id]
    )
    prediction_jobs = _prediction_payloads(fit.model, validation_branches)
    prediction_hashes = canonical_sha256s(node, [payload for _, payload in prediction_jobs])
    prediction_artifacts = []
    for (branch, payload), payload_sha256 in zip(prediction_jobs, prediction_hashes, strict=True):
        prediction_id = f"prediction-{sha256_bytes(strict_json_bytes([task_id, branch.group_id, branch.branch_id]))[:32]}"
        document = {
            "schema_version": SCHEMA_VERSION,
            "contract_kind": "objgauss.dynamics_prediction",
            "identity": {
                "experiment_id": EXPERIMENT_ID,
                "prediction_id": prediction_id,
                "group_id": branch.group_id,
                "branch_id": branch.branch_id,
                "split": "validation",
                "model_arm": arm,
                "trial_id": {"availability": "present", "value": trial_id},
                "checkpoint_id": {"availability": "present", "value": checkpoint_id},
                "training_seed": {"availability": "present", "value": seed},
            },
            "inputs": {
                "source_episode": branch.source_episode,
                "initial_objectstate_sha256": branch.initial_objectstate_sha256,
                "commanded_action_sha256": branch.commanded_action_sha256,
                "target_object_id": branch.object_ids[branch.target_index],
                "visible_fields": ["initial_objectstate", "commanded_action_schedule", "non_future_metadata"],
                "commanded_action_present": True,
                "executed_action_is_feature": False,
                "gt_future_read": False,
            },
            "horizon": {
                "unit": "physical_seconds",
                "duration_s": ROLLOUT_TIMES[-1],
                "scoring_times_s": list(ROLLOUT_TIMES),
            },
            "predictions": payload,
            "prediction_payload_sha256": payload_sha256,
            "publication": {"atomic_publish": True, "immutable_after_publish": True},
            "provenance": provenance,
        }
        relative = Path("predictions") / branch.group_id / f"{branch.branch_id}.json"
        atomic_write(task_root / relative, strict_json_bytes(document))
        prediction_artifacts.append(relative)
    artifact_paths = [
        checkpoint_relative,
        log_relative,
        *(relative for relative, _ in documents),
        *prediction_artifacts,
    ]
    artifacts = [
        {"uri": f"tasks/{task_id}/{path.as_posix()}", "sha256": file_sha256(task_root / path)}
        for path in sorted(artifact_paths)
    ]
    initialization = pair_fit.initialization[arm]
    fairness = {
        "initialization_seed": seed,
        "initialization_algorithm": "torch-manual-seed-deterministic-v1",
        **initialization,
        "data_order_sha256": fit.training_log["data_order_sha256"],
        "batch_group_sequence_sha256": pair_fit.batch_group_sequence_sha256,
        "optimizer_updates": fit.optimizer_updates,
        "epochs": fit.epochs,
        "training_budget_sha256": pair_fit.training_budget_sha256,
        "checkpoint_policy": "minimum-validation-primary-error-per-seed",
    }
    return {
        "task_id": task_id,
        "pair_id": pair_id,
        "model_arm": arm,
        "config_id": config["config_id"],
        "config_sha256": config["config_sha256"],
        "training_seed": seed,
        "status": "completed",
        "attempt_ids": attempt_ids,
        "hpo_data_index_sha256": hpo_data_index_sha256,
        "validation_group_errors": pair_fit.validation_group_errors[arm],
        "validation_primary_error": fit.validation_primary_error,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_artifact["sha256"],
        "checkpoint_semantic_sha256": fit.training_log["tensor_state_semantic_sha256"],
        "checkpoint_manifest_published": False,
        "fairness": fairness,
        "resources": resources,
        "artifacts": artifacts,
    }


def produce_hpo(
    *,
    repo_root: Path,
    manifest_path: Path,
    data_root: Path,
    formal_path: Path,
    dynamics_path: Path,
    output_root: Path,
    runner_commit: str,
    node: str,
    device_name: str,
) -> dict[str, Any]:
    require_offline()
    assert_clean_git_head(repo_root, runner_commit)
    if device_name != "cuda" or not torch.cuda.is_available():
        raise TrainerBlockedError("C6 HPO requires the frozen CUDA runtime")
    device = torch.device("cuda")
    contract = validate_hpo_manifest(repo_root, manifest_path, node)
    manifest = contract["manifest"]
    if output_root.exists():
        raise TrainerInvalidError("C6 HPO output root already exists")
    formal = read_json(formal_path)
    dynamics = read_json(dynamics_path)
    if dynamics.get("contract_kind") != "objgauss.dynamics_experiment":
        raise TrainerInvalidError("C6 dynamics experiment contract drift")
    groups, loader_report = load_dataset(
        repo_root=repo_root,
        data_root=data_root,
        manifest_path=repo_root / "learning/data-boundary-manifest.json",
        formal_path=formal_path,
        dynamics_path=dynamics_path,
        source_commit=runner_commit,
        splits=("train", "validation"),
    )
    converted = _convert_groups(groups=groups, formal=formal, node=node, device=device)
    temporary = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    failed = output_root.with_name(f"{output_root.name}.failed-{os.getpid()}")
    if temporary.exists() or failed.exists():
        raise TrainerInvalidError("C6 HPO temporary output collision")
    temporary.mkdir(parents=True)
    try:
        hpo_data_index = {
            "index_version": VERSION,
            "index_kind": "objgauss.pr02c-c6-hpo-data-index",
            "runner_commit": runner_commit,
            "source_data_index_sha256": loader_report["data_index_sha256"],
            "model_input_index_sha256": loader_report["model_input_index_sha256"],
            "loader_report_sha256": sha256_bytes(strict_json_bytes(loader_report)),
            "group_counts": {"train": 48, "validation": 12},
            "branch_count": 300,
            "splits": ["train", "validation"],
            "test_materialized": False,
            "build_count": 1,
            "shared_by_all_tasks": True,
        }
        atomic_write(temporary / "hpo-data-index.json", strict_json_bytes(hpo_data_index))
        hpo_data_index_sha256 = file_sha256(temporary / "hpo-data-index.json")
        task_records = []
        pair_ledgers = []
        scales = read_json(repo_root / "learning/trainer-manifest.json")["loss"]["normalization_scales"]
        for pair in contract["pairs"]:
            config = contract["configs"][pair["config_id"]]
            pair_fit, failed_attempts = fit_pair_with_retry(
                config=config,
                seed=pair["training_seed"],
                train_groups=converted["train"],
                validation_groups=converted["validation"],
                scales=scales,
                device=device,
            )
            pair_tasks = []
            for arm in ARMS:
                task = publish_task(
                    repo_root=repo_root,
                    root=temporary,
                    task_id=pair["task_ids"][arm],
                    pair_id=pair["pair_id"],
                    arm=arm,
                    config=config,
                    seed=pair["training_seed"],
                    fit=pair_fit.fits[arm],
                    pair_fit=pair_fit,
                    validation_groups=converted["validation"],
                    runner_commit=runner_commit,
                    dynamics_sha256=file_sha256(dynamics_path),
                    runtime_lock_sha256=manifest["lineage"]["frozen_inputs"]["runtime_lock"]["sha256"],
                    grid_sha256=manifest["lineage"]["frozen_inputs"]["hyperparameter_grid"]["sha256"],
                    hpo_data_index_sha256=hpo_data_index_sha256,
                    node=node,
                    failed_attempts=failed_attempts,
                )
                task_records.append(task)
                pair_tasks.append(task)
            shared_fields = (
                "initialization_seed",
                "initialization_algorithm",
                "common_parameter_names_sha256",
                "common_parameter_subtree_sha256",
                "data_order_sha256",
                "batch_group_sequence_sha256",
                "optimizer_updates",
                "epochs",
                "training_budget_sha256",
                "checkpoint_policy",
            )
            if any(
                pair_tasks[0]["fairness"][name] != pair_tasks[1]["fairness"][name]
                for name in shared_fields
            ):
                raise TrainerInvalidError(f"C6 fairness pair ledger differs: {pair['pair_id']}")
            pair_ledgers.append(
                {
                    "pair_id": pair["pair_id"],
                    "config_id": pair["config_id"],
                    "training_seed": pair["training_seed"],
                    "task_ids": pair["task_ids"],
                    "shared": {name: pair_tasks[0]["fairness"][name] for name in shared_fields},
                    "arm_specific_initialization": {
                        task["model_arm"]: {
                            "parameter_names_sha256": task["fairness"]["arm_specific_parameter_names_sha256"],
                            "parameter_subtree_sha256": task["fairness"]["arm_specific_parameter_subtree_sha256"],
                        }
                        for task in pair_tasks
                    },
                }
            )
        task_records.sort(key=lambda item: item["task_id"])
        pair_ledgers.sort(key=lambda item: item["pair_id"])
        total_gpu_hours = math.fsum(item["resources"]["gpu_hours"] for item in task_records)
        if total_gpu_hours > manifest["execution"]["hpo_gpu_hours_with_retry_reserve_max"]:
            raise TrainerBlockedError("C6 HPO exceeded the 6.3 GPU-hour hard budget")
        validation_group_ids = sorted(converted["validation"])
        task_index = {
            "index_version": VERSION,
            "index_kind": TASK_INDEX_KIND,
            "runner_commit": runner_commit,
            "workflow_commit": runner_commit,
            "hpo_data_build_commit": runner_commit,
            "trainer_contract_commit": manifest["lineage"]["trainer_contract_commit"],
            "hpo_manifest_sha256": contract["manifest_sha256"],
            "hpo_data_index_sha256": hpo_data_index_sha256,
            "validation_group_ids": validation_group_ids,
            "splits": ["train", "validation"],
            "test_materialized": False,
            "tasks": task_records,
            "resources": {
                "gpu_hours": total_gpu_hours,
                "peak_vram_bytes": max(item["resources"]["peak_vram_bytes"] for item in task_records),
                "minimum_display_vram_free_bytes": min(
                    item["resources"]["minimum_display_vram_free_bytes"] for item in task_records
                ),
            },
        }
        fairness_ledger = {
            "ledger_version": VERSION,
            "ledger_kind": "objgauss.pr02c-c6-fairness-pair-ledger",
            "pair_count": 12,
            "pairs": pair_ledgers,
        }
        atomic_write(temporary / "task-index.json", strict_json_bytes(task_index))
        atomic_write(temporary / "fairness-pair-ledger.json", strict_json_bytes(fairness_ledger))
        report = {
            "report_version": VERSION,
            "report_kind": "objgauss.pr02c-c6-hpo-runner-report",
            "verdict": {"status": "supported", "reason_code": "frozen-task-matrix-complete"},
            "runner_commit": runner_commit,
            "counts": {"tasks": 24, "fairness_pairs": 12, "configs": 4, "seeds": 3},
            "hpo_data_index_sha256": hpo_data_index_sha256,
            "task_index_sha256": file_sha256(temporary / "task-index.json"),
            "fairness_pair_ledger_sha256": file_sha256(temporary / "fairness-pair-ledger.json"),
            "resources": task_index["resources"],
            "test_materialized": False,
            "claim_boundary": manifest["claim_boundary"],
        }
        atomic_write(temporary / "runner-report.json", strict_json_bytes(report))
        os.replace(temporary, output_root)
        directory = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return report
    except Exception:
        if temporary.exists():
            os.replace(temporary, failed)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--mode", choices=("contract", "run"), required=True)
    value.add_argument("--manifest", type=Path, default=Path("learning/hpo-manifest.json"))
    value.add_argument("--output", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--formal-spec", type=Path)
    value.add_argument("--dynamics-experiment", type=Path)
    value.add_argument("--output-root", type=Path)
    value.add_argument("--source-commit")
    value.add_argument("--node", default="node")
    value.add_argument("--device", choices=("cpu", "cuda"))
    value.add_argument("--splits", nargs="+", default=["train", "validation"])
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    resolve = lambda path: path if path.is_absolute() else repo_root / path
    try:
        if args.splits != ["train", "validation"]:
            raise TrainerInvalidError("HPO runner accepts train/validation only; final test is forbidden")
        manifest_path = resolve(args.manifest).resolve()
        if args.mode == "contract":
            contract = validate_hpo_manifest(repo_root, manifest_path, args.node)
            report = {
                "report_version": VERSION,
                "report_kind": "objgauss.pr02c-c6-cpu-contract-report",
                "verdict": "supported",
                "counts": {"tasks": 24, "fairness_pairs": 12, "configs": 4, "seeds": 3},
                "hpo_manifest_sha256": contract["manifest_sha256"],
                "test_visible": False,
            }
            if args.output is not None:
                atomic_write(resolve(args.output), strict_json_bytes(report))
        else:
            required = {
                "data_root": args.data_root,
                "formal_spec": args.formal_spec,
                "dynamics_experiment": args.dynamics_experiment,
                "output_root": args.output_root,
                "source_commit": args.source_commit,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise TrainerInvalidError(f"HPO run is missing arguments: {','.join(missing)}")
            report = produce_hpo(
                repo_root=repo_root,
                manifest_path=manifest_path,
                data_root=resolve(args.data_root).resolve(),
                formal_path=resolve(args.formal_spec).resolve(),
                dynamics_path=resolve(args.dynamics_experiment).resolve(),
                output_root=resolve(args.output_root).resolve(),
                runner_commit=args.source_commit,
                node=args.node,
                device_name=args.device or "cuda",
            )
        sys.stdout.buffer.write(strict_json_bytes(report))
        return 0
    except (CanonicalHashError, DataBlockedError, TrainerBlockedError) as error:
        report = {"verdict": "blocked", "reason_code": "required-input-resource-or-budget-missing", "message": str(error)}
        exit_code = 3
    except TrainerRejectedError as error:
        report = {"verdict": "rejected", "reason_code": "valid-hpo-training-failed", "message": str(error)}
        exit_code = 2
    except (
        DataInvalidError,
        RuntimeInvalidError,
        TrainerInvalidError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        report = {"verdict": "invalid", "reason_code": "hpo-protocol-invalid", "message": str(error)}
        exit_code = 4
    if args.output is not None and args.mode == "contract":
        atomic_write(resolve(args.output), strict_json_bytes(report))
    sys.stdout.buffer.write(strict_json_bytes(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
