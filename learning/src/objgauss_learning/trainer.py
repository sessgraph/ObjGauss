"""C3 learned-arm trainer, golden lineage writer, and CPU tiny smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from .canonical import CanonicalHashError, canonical_sha256, canonical_sha256s
from .data import (
    BranchSample,
    DataBlockedError,
    DataInvalidError,
    GroupSample,
    actor_states,
    load_dataset,
    model_payload,
)
from .model import (
    ARMS,
    ROLLOUT_TIMES,
    STATE_WIDTH,
    MinimalObjectGNN,
    ModelInvalidError,
    architecture_document,
    assert_frozen_delta_t,
    canonicalize_quaternion_tensor,
    parameter_count,
    quaternion_inverse_tensor,
    quaternion_multiply_tensor,
    state_payload,
    state_tensor,
    symmetry_quaternion_distance,
)
from .runtime import atomic_write, file_sha256, package_tree_sha256, strict_json_bytes


MANIFEST_KIND = "objgauss.pr02c-golden-trainer"
REPORT_KIND = "objgauss.pr02c-golden-training-report"
REPORT_VERSION = "0.1.0"
SOURCE_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40}$")
DISPLAY_RESERVE_BYTES = 1024**3
TRAINING_CAP_BYTES = 12 * 1024**3
SCHEMA_VERSION = "0.3.0"
METRIC_NAME = "target-object-multistep-effect-vs-hold-objectstate-error"
COMPONENT_NAMES = ("position", "orientation", "linear_velocity", "angular_velocity")


class TrainerBlockedError(RuntimeError):
    """The frozen runtime, data, or device is unavailable."""


class TrainerInvalidError(RuntimeError):
    """Training inputs, fairness, lineage, or publication are invalid."""


class TrainerRejectedError(RuntimeError):
    """A valid training attempt produced non-finite output or did not complete."""


@dataclass(frozen=True)
class TensorBranch:
    group_id: str
    split: str
    branch_id: str
    object_ids: tuple[str, ...]
    initial_state: Tensor
    commanded_action: Tensor
    target_index: int
    labels: Tensor
    symmetries_wxyz: Tensor
    source_episode: dict[str, Any]
    initial_objectstate_sha256: str
    commanded_action_sha256: str


@dataclass
class FitResult:
    model: MinimalObjectGNN
    best_state_dict: dict[str, Tensor]
    best_epoch: int
    best_update: int
    validation_primary_error: float
    optimizer_updates: int
    epochs: int
    training_log: dict[str, Any]
    minimum_display_vram_free_bytes: int
    peak_vram_bytes: int
    wall_seconds: float


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(path.read_bytes(), parse_constant=reject_constant)
    except FileNotFoundError as error:
        raise TrainerBlockedError(f"required trainer input is missing: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise TrainerInvalidError(f"invalid trainer JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise TrainerInvalidError(f"trainer JSON input must be an object: {path}")
    return value


def require_offline() -> None:
    if os.environ.get("OBJGAUSS_LEARNING_OFFLINE") != "1":
        raise TrainerInvalidError("OBJGAUSS_LEARNING_OFFLINE must be exactly '1'")


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def validate_manifest(repo_root: Path, manifest: dict[str, Any]) -> None:
    required = {
        "manifest_version",
        "manifest_kind",
        "arms",
        "architecture",
        "rollout",
        "loss",
        "golden",
        "tiny",
        "resources",
        "frozen_inputs",
        "outputs",
        "claim_boundary",
    }
    if set(manifest) != required or manifest.get("manifest_version") != REPORT_VERSION:
        raise TrainerInvalidError("trainer manifest top-level contract drift")
    if manifest.get("manifest_kind") != MANIFEST_KIND or manifest.get("arms") != list(ARMS):
        raise TrainerInvalidError("trainer manifest kind or learned arm set drift")
    architecture = manifest["architecture"]
    expected_architecture = {
        "model_family": "minimal-object-gnn",
        "state_width": 13,
        "action_feature_width": 9,
        "hidden_width": 64,
        "object_encoder_layers": 2,
        "pairwise_message_layers": 2,
        "message_passing_rounds": 1,
        "aggregation": "mean",
        "shared_residual_head_layers": 2,
        "activation": "silu",
        "action_injection": "target-object-only",
        "action_application_point": "target-object-center-of-mass-zero-in-object-frame",
        "action_free_input": "learned-mask-token",
        "action_conditioned_input": "learned-mask-token-plus-interval-action",
        "parameter_parity_required": True,
    }
    if architecture != expected_architecture:
        raise TrainerInvalidError("minimal Object GNN architecture drift")
    if manifest["rollout"] != {
        "unit": "physical_seconds",
        "boundaries_s": [0.0, 0.1, 0.2, 0.5, 1.1],
        "delta_t_s": [0.1, 0.1, 0.3, 0.6],
        "transition_parameters": "shared-across-all-intervals",
        "teacher_forcing": "initial-state-only",
        "commanded_action_clipping": "per-interval-active-duration-and-fraction",
        "executed_action_is_feature": False,
        "gt_future_is_model_input": False,
        "quaternion_normalization": "every-transition",
        "non_finite_fallback_allowed": False,
    }:
        raise TrainerInvalidError("four-interval rollout policy drift")
    golden = manifest["golden"]
    if golden != {
        "config_id": "golden-h64-lr1e3",
        "training_seed": 2026071501,
        "hidden_width": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "batch_size_max": 64,
        "gradient_clip_norm": 1.0,
        "optimizer_updates": 8,
        "epochs": 8,
        "train_group_id": "group-pr02-train-object-01-pr02-train-layout-01-pr02-train-start-01-seed-27071001",
        "validation_group_id": "group-pr02-validation-object-01-pr02-validation-layout-01-pr02-validation-start-01-seed-27072001",
        "hpo_config_selected": False,
        "formal_checkpoint_frozen": False,
    }:
        raise TrainerInvalidError("golden smoke configuration drift")
    if manifest["resources"] != {
        "training_peak_vram_bytes_max": TRAINING_CAP_BYTES,
        "display_vram_reserve_bytes_min": DISPLAY_RESERVE_BYTES,
        "gpu_hours_charged_to_hpo": False,
        "gpu_hours_charged_to_formal": False,
    }:
        raise TrainerInvalidError("golden resource boundary drift")
    for name, entry in manifest["frozen_inputs"].items():
        if name == "schemas":
            continue
        path = entry.get("path")
        if path is not None and file_sha256(repo_root / path) != entry.get("sha256"):
            raise TrainerInvalidError(f"frozen trainer input checksum drift: {name}")
    schema_paths = {
        "common": "contracts/objgauss/0.3.0/common.schema.json",
        "training_trial": "contracts/objgauss/0.3.0/training-trial.schema.json",
        "training_attempt": "contracts/objgauss/0.3.0/training-attempt.schema.json",
        "checkpoint_manifest": "contracts/objgauss/0.3.0/checkpoint-manifest.schema.json",
        "dynamics_prediction": "contracts/objgauss/0.3.0/dynamics-prediction.schema.json",
    }
    for name, relative in schema_paths.items():
        if file_sha256(repo_root / relative) != manifest["frozen_inputs"]["schemas"].get(name):
            raise TrainerInvalidError(f"frozen trainer schema checksum drift: {name}")
    assert_frozen_delta_t()


def _state_from_actor(value: Any, *, device: torch.device) -> Tensor:
    payload = {
        "position_W_m": list(value.position_W_m),
        "quaternion_WO_wxyz": list(value.quaternion_WO_wxyz),
        "linear_velocity_W_m_s": list(value.linear_velocity_W_m_s),
        "angular_velocity_W_rad_s": list(value.angular_velocity_W_rad_s),
    }
    return state_tensor(payload, device=device)


def _formal_symmetries(
    formal: dict[str, Any], split: str, group_id: str, *, device: torch.device
) -> Tensor:
    if split not in {"train", "validation"}:
        raise TrainerInvalidError(f"trainer split is forbidden: {split}")
    partition = formal.get("partitions", {}).get(split, {})
    group = next((item for item in partition.get("groups", []) if item.get("group_id") == group_id), None)
    if group is None:
        raise TrainerInvalidError(f"training group is absent from frozen spec: {group_id}")
    object_spec = partition.get("objects", {}).get(group.get("object_identity_id"), {})
    symmetry = object_spec.get("symmetry", {})
    if symmetry.get("kind") != "finite_wxyz":
        raise TrainerInvalidError("C3 golden requires explicit finite object symmetries")
    rotations = torch.tensor(symmetry.get("rotations"), dtype=torch.float32, device=device)
    if rotations.ndim != 2 or rotations.shape[-1] != 4 or rotations.shape[0] == 0:
        raise TrainerInvalidError("frozen object symmetry rotations are invalid")
    return canonicalize_quaternion_tensor(rotations)


def tensor_branch(
    branch: BranchSample,
    *,
    symmetries: Tensor,
    initial_sha256: str,
    command_sha256: str,
    device: torch.device,
) -> TensorBranch:
    object_ids = tuple(state.actor_id for state in branch.model_inputs.initial_object_states)
    if len(set(object_ids)) != len(object_ids):
        raise TrainerInvalidError("training ObjectState IDs are not unique")
    try:
        target_index = object_ids.index(branch.model_inputs.target_object_id)
    except ValueError as error:
        raise TrainerInvalidError("training target object is absent") from error
    initial = torch.stack(
        [_state_from_actor(state, device=device) for state in branch.model_inputs.initial_object_states]
    )
    label_rows = []
    for timed in branch.labels.future_object_states:
        by_id = {state.actor_id: state for state in timed.object_states}
        if set(by_id) != set(object_ids):
            raise TrainerInvalidError("label ObjectState set differs from initial state")
        label_rows.append(torch.stack([_state_from_actor(by_id[object_id], device=device) for object_id in object_ids]))
    labels = torch.stack(label_rows)
    if labels.shape != (4, len(object_ids), STATE_WIDTH):
        raise TrainerInvalidError("training labels do not cover four frozen scoring times")
    action = branch.model_inputs.commanded_action
    commanded = torch.tensor(
        [*action.vector_W_N, action.duration_s, 1.0 if action.kind == "push" else 0.0],
        dtype=torch.float32,
        device=device,
    )
    return TensorBranch(
        group_id=branch.group_id,
        split=branch.split,
        branch_id=branch.branch_id,
        object_ids=object_ids,
        initial_state=initial,
        commanded_action=commanded,
        target_index=target_index,
        labels=labels,
        symmetries_wxyz=symmetries,
        source_episode={
            "uri": branch.source_episode.uri,
            "media_type": "application/json",
            "sha256": branch.source_episode.sha256,
            "schema_version": "0.2.0",
            "lineage_sha256": branch.source_episode.lineage_sha256,
        },
        initial_objectstate_sha256=initial_sha256,
        commanded_action_sha256=command_sha256,
    )


def select_golden_groups(
    *,
    groups: tuple[GroupSample, ...],
    formal: dict[str, Any],
    train_group_id: str,
    validation_group_id: str,
    node: str,
    device: torch.device,
) -> tuple[tuple[TensorBranch, ...], tuple[TensorBranch, ...]]:
    selected = {
        (group.split, group.group_id): group
        for group in groups
        if group.group_id in {train_group_id, validation_group_id}
    }
    expected_keys = {("train", train_group_id), ("validation", validation_group_id)}
    if set(selected) != expected_keys:
        raise TrainerInvalidError("golden train/validation group set differs from manifest")
    branches = [
        branch
        for key in sorted(selected)
        for branch in selected[key].branches
    ]
    payloads = [model_payload(branch.model_inputs) for branch in branches]
    hashes = canonical_sha256s(
        node,
        [
            value
            for payload in payloads
            for value in (payload["initial_object_states"], payload["commanded_action"])
        ],
    )
    converted: dict[str, list[TensorBranch]] = {"train": [], "validation": []}
    for index, (branch, payload) in enumerate(zip(branches, payloads, strict=True)):
        del payload
        symmetries = _formal_symmetries(formal, branch.split, branch.group_id, device=device)
        converted[branch.split].append(
            tensor_branch(
                branch,
                symmetries=symmetries,
                initial_sha256=hashes[index * 2],
                command_sha256=hashes[index * 2 + 1],
                device=device,
            )
        )
    for split in converted:
        converted[split].sort(key=lambda item: (item.group_id, item.branch_id))
        if len(converted[split]) != 5:
            raise TrainerInvalidError(f"golden {split} group must contain five siblings")
    return tuple(converted["train"]), tuple(converted["validation"])


def _stack(branches: tuple[TensorBranch, ...]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not branches:
        raise TrainerInvalidError("training batch is empty")
    object_ids = branches[0].object_ids
    if any(branch.object_ids != object_ids for branch in branches):
        raise TrainerInvalidError("training batch ObjectState order drift")
    initial = torch.stack([branch.initial_state for branch in branches])
    actions = torch.stack([branch.commanded_action for branch in branches])
    labels = torch.stack([branch.labels for branch in branches])
    target_mask = torch.zeros(
        (len(branches), len(object_ids), 1), dtype=initial.dtype, device=initial.device
    )
    for index, branch in enumerate(branches):
        target_mask[index, branch.target_index, 0] = 1.0
    return initial, actions, target_mask, labels


def _component_errors(
    prediction: Tensor,
    target: Tensor,
    *,
    symmetries: Tensor,
    scales: dict[str, float],
) -> Tensor:
    position = torch.linalg.vector_norm(prediction[..., 0:3] - target[..., 0:3], dim=-1)
    orientation = symmetry_quaternion_distance(
        prediction[..., 3:7], target[..., 3:7], symmetries
    )
    linear = torch.linalg.vector_norm(prediction[..., 7:10] - target[..., 7:10], dim=-1)
    angular = torch.linalg.vector_norm(prediction[..., 10:13] - target[..., 10:13], dim=-1)
    return (
        position / scales["position"]
        + orientation / scales["orientation"]
        + linear / scales["linear_velocity"]
        + angular / scales["angular_velocity"]
    ) * 0.25


def open_loop_loss(
    model: MinimalObjectGNN,
    branches: tuple[TensorBranch, ...],
    *,
    scales: dict[str, float],
) -> Tensor:
    initial, actions, target_mask, labels = _stack(branches)
    predictions = model.rollout(initial, commanded_action=actions, target_mask=target_mask)
    values = []
    for batch_index, branch in enumerate(branches):
        target_index = branch.target_index
        values.append(
            _component_errors(
                predictions[batch_index, :, target_index],
                labels[batch_index, :, target_index],
                symmetries=branch.symmetries_wxyz,
                scales=scales,
            ).mean()
        )
    return torch.stack(values).mean()


def predict_tensor_branches(
    model: MinimalObjectGNN, branches: tuple[TensorBranch, ...]
) -> dict[str, Tensor]:
    initial, actions, target_mask, _ = _stack(branches)
    with torch.no_grad():
        predictions = model.rollout(initial, commanded_action=actions, target_mask=target_mask)
    return {branch.branch_id: predictions[index] for index, branch in enumerate(branches)}


def validation_effect_error(
    model: MinimalObjectGNN,
    branches: tuple[TensorBranch, ...],
    *,
    scales: dict[str, float],
) -> Tensor:
    by_id = {branch.branch_id: branch for branch in branches}
    if len(by_id) != len(branches) or "hold" not in by_id:
        raise TrainerInvalidError("validation sibling group must contain one unique hold branch")
    predicted = predict_tensor_branches(model, branches)
    hold = by_id["hold"]
    values = []
    for branch_id in sorted(set(by_id) - {"hold"}):
        branch = by_id[branch_id]
        if branch.target_index != hold.target_index or branch.object_ids != hold.object_ids:
            raise TrainerInvalidError("validation sibling target identity drift")
        target_index = branch.target_index
        pred_branch = predicted[branch_id][:, target_index]
        pred_hold = predicted["hold"][:, target_index]
        gt_branch = branch.labels[:, target_index]
        gt_hold = hold.labels[:, target_index]
        position = torch.linalg.vector_norm(
            (pred_branch[:, 0:3] - pred_hold[:, 0:3])
            - (gt_branch[:, 0:3] - gt_hold[:, 0:3]),
            dim=-1,
        )
        linear = torch.linalg.vector_norm(
            (pred_branch[:, 7:10] - pred_hold[:, 7:10])
            - (gt_branch[:, 7:10] - gt_hold[:, 7:10]),
            dim=-1,
        )
        angular = torch.linalg.vector_norm(
            (pred_branch[:, 10:13] - pred_hold[:, 10:13])
            - (gt_branch[:, 10:13] - gt_hold[:, 10:13]),
            dim=-1,
        )
        pred_effect = quaternion_multiply_tensor(
            pred_branch[:, 3:7], quaternion_inverse_tensor(pred_hold[:, 3:7])
        )
        gt_effect = quaternion_multiply_tensor(
            gt_branch[:, 3:7], quaternion_inverse_tensor(gt_hold[:, 3:7])
        )
        orientation = symmetry_quaternion_distance(
            pred_effect, gt_effect, branch.symmetries_wxyz
        )
        normalized = (
            position / scales["position"]
            + orientation / scales["orientation"]
            + linear / scales["linear_velocity"]
            + angular / scales["angular_velocity"]
        ) * 0.25
        values.append(normalized.mean())
    return torch.stack(values).mean()


def _clone_state_dict(model: MinimalObjectGNN) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _tensor_bytes(tensor: Tensor) -> bytes:
    return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())


def tensor_state_semantic_sha256(state_dict: dict[str, Tensor]) -> str:
    semantic = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        payload = _tensor_bytes(tensor)
        entry = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "tensor_sha256": hashlib.sha256(payload).hexdigest(),
        }
        encoded = strict_json_bytes(entry)
        semantic.update(len(encoded).to_bytes(8, "big"))
        semantic.update(encoded)
        semantic.update(len(payload).to_bytes(8, "big"))
        semantic.update(payload)
    return semantic.hexdigest()


def _gpu_free(device: torch.device) -> int:
    if device.type != "cuda":
        return DISPLAY_RESERVE_BYTES
    free, _ = torch.cuda.mem_get_info(device)
    return int(free)


def fit_model(
    *,
    arm: str,
    hidden_width: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    optimizer_updates: int,
    epochs: int,
    seed: int,
    train_branches: tuple[TensorBranch, ...],
    validation_branches: tuple[TensorBranch, ...],
    scales: dict[str, float],
    device: torch.device,
) -> FitResult:
    configure_determinism(seed)
    if optimizer_updates != epochs:
        raise TrainerInvalidError("golden smoke requires exactly one optimizer update per epoch")
    started = time.monotonic()
    model = MinimalObjectGNN(hidden_width=hidden_width, arm=arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    minimum_free = _gpu_free(device)
    best_score = math.inf
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_update = 0
    log_entries = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = open_loop_loss(model, train_branches, scales=scales)
        if not torch.isfinite(loss):
            raise TrainerRejectedError("training loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise TrainerRejectedError("gradient norm is non-finite")
        optimizer.step()
        minimum_free = min(minimum_free, _gpu_free(device))
        model.eval()
        with torch.no_grad():
            validation = validation_effect_error(
                model, validation_branches, scales=scales
            )
        if not torch.isfinite(validation):
            raise TrainerRejectedError("validation primary error is non-finite")
        score = float(validation.detach().cpu())
        if score < best_score:
            best_score = score
            best_state = _clone_state_dict(model)
            best_epoch = epoch
            best_update = epoch
        log_entries.append(
            {
                "epoch": epoch,
                "optimizer_update": epoch,
                "training_loss": float(loss.detach().cpu()),
                "validation_primary_error": score,
            }
        )
    if best_state is None or best_epoch < 1:
        raise TrainerRejectedError("golden training did not select a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    finished = time.monotonic()
    if peak_vram > TRAINING_CAP_BYTES:
        raise TrainerInvalidError("training peak VRAM exceeded 12 GiB")
    if minimum_free < DISPLAY_RESERVE_BYTES:
        raise TrainerInvalidError("training violated the 1 GiB display VRAM reserve")
    data_order = [f"{branch.group_id}/{branch.branch_id}" for branch in train_branches]
    training_log = {
        "log_version": REPORT_VERSION,
        "log_kind": "objgauss.pr02c-golden-training-log",
        "model_arm": arm,
        "training_seed": seed,
        "teacher_forcing": "initial-state-only",
        "rollout_times_s": list(ROLLOUT_TIMES),
        "delta_t_s": [0.1, 0.1, 0.3, 0.6],
        "optimizer": "adamw",
        "optimizer_updates": optimizer_updates,
        "epochs": epochs,
        "data_order": data_order,
        "data_order_sha256": sha256_bytes(strict_json_bytes(data_order)),
        "best_epoch": best_epoch,
        "best_optimizer_update": best_update,
        "best_validation_primary_error": best_score,
        "tensor_state_semantic_sha256": tensor_state_semantic_sha256(best_state),
        "entries": log_entries,
        "visibility": {
            "commanded_action_present": True,
            "executed_action_is_feature": False,
            "gt_future_is_model_input": False,
            "test_materialized": False,
        },
    }
    return FitResult(
        model=model,
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_update=best_update,
        validation_primary_error=best_score,
        optimizer_updates=optimizer_updates,
        epochs=epochs,
        training_log=training_log,
        minimum_display_vram_free_bytes=minimum_free,
        peak_vram_bytes=peak_vram,
        wall_seconds=finished - started,
    )


def _atomic_torch_save(path: Path, state_dict: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise TrainerInvalidError(f"checkpoint temporary path already exists: {temporary}")
    try:
        torch.save(state_dict, temporary, _use_new_zipfile_serialization=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(prefix: str, relative: Path, path: Path, media_type: str) -> dict[str, Any]:
    return {
        "uri": f"{prefix.rstrip('/')}/{relative.as_posix()}",
        "media_type": media_type,
        "sha256": file_sha256(path),
        "byte_size": path.stat().st_size,
    }


def _provenance(
    *, repo_root: Path, source_commit: str, experiment_spec_sha256: str, runtime_lock_sha256: str
) -> dict[str, str]:
    return {
        "source_commit": source_commit,
        "source_tree_sha256": package_tree_sha256(repo_root),
        "experiment_spec_sha256": experiment_spec_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
    }


def _prediction_payloads(
    model: MinimalObjectGNN, branches: tuple[TensorBranch, ...]
) -> list[tuple[TensorBranch, list[dict[str, Any]]]]:
    predicted = predict_tensor_branches(model, branches)
    documents = []
    for branch in branches:
        time_points = []
        tensor = predicted[branch.branch_id]
        for time_index, time_s in enumerate(ROLLOUT_TIMES):
            time_points.append(
                {
                    "time_s": time_s,
                    "objects": [
                        state_payload(object_id, tensor[time_index, object_index])
                        for object_index, object_id in enumerate(branch.object_ids)
                    ],
                }
            )
        documents.append((branch, time_points))
    return documents


def publish_arm(
    *,
    repo_root: Path,
    output_root: Path,
    artifact_prefix: str,
    source_commit: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    dynamics_sha256: str,
    arm: str,
    fit: FitResult,
    validation_branches: tuple[TensorBranch, ...],
    node: str,
    device: torch.device,
    started_monotonic_s: float,
) -> dict[str, Any]:
    golden = manifest["golden"]
    config = {
        "config_id": golden["config_id"],
        "model_family": "minimal-object-gnn",
        "hidden_width": golden["hidden_width"],
        "learning_rate": golden["learning_rate"],
        "weight_decay": golden["weight_decay"],
        "batch_size_max": golden["batch_size_max"],
        "gradient_clip_norm": golden["gradient_clip_norm"],
        "optimizer_updates": golden["optimizer_updates"],
        "epochs": golden["epochs"],
    }
    config_sha256 = sha256_bytes(strict_json_bytes(config))
    architecture_sha256 = sha256_bytes(
        strict_json_bytes(architecture_document(golden["hidden_width"]))
    )
    seed = golden["training_seed"]
    trial_id = f"trial-golden-{arm}-{seed}"
    attempt_id = f"attempt-golden-{arm}-{seed}-1"
    checkpoint_id = f"checkpoint-golden-{arm}-{seed}"
    checkpoint_relative = Path("checkpoints") / f"{checkpoint_id}.pt"
    log_relative = Path("logs") / f"{trial_id}.json"
    _atomic_torch_save(output_root / checkpoint_relative, fit.best_state_dict)
    atomic_write(output_root / log_relative, strict_json_bytes(fit.training_log))
    checkpoint_artifact = _artifact(
        artifact_prefix,
        checkpoint_relative,
        output_root / checkpoint_relative,
        "application/vnd.objgauss.pytorch-state-dict",
    )
    log_artifact = _artifact(
        artifact_prefix, log_relative, output_root / log_relative, "application/json"
    )
    artifact_bytes = checkpoint_artifact["byte_size"] + log_artifact["byte_size"]
    resources = {
        "gpu_hours": fit.wall_seconds / 3600.0 if device.type == "cuda" else 0.0,
        "peak_vram_bytes": fit.peak_vram_bytes,
        "minimum_display_vram_free_bytes": fit.minimum_display_vram_free_bytes,
        "artifact_bytes": artifact_bytes,
    }
    provenance = _provenance(
        repo_root=repo_root,
        source_commit=source_commit,
        experiment_spec_sha256=dynamics_sha256,
        runtime_lock_sha256=manifest["frozen_inputs"]["runtime_lock"]["sha256"],
    )
    finished_monotonic_s = started_monotonic_s + fit.wall_seconds
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "contract_kind": "objgauss.training_attempt",
        "identity": {
            "experiment_id": "pr02-objectstate-baseline-v0",
            "trial_id": trial_id,
            "attempt_id": attempt_id,
            "ordinal": 1,
            "model_arm": arm,
            "training_seed": seed,
            "config_sha256": config_sha256,
        },
        "timing": {
            "started_monotonic_s": started_monotonic_s,
            "finished_monotonic_s": finished_monotonic_s,
            "wall_seconds": fit.wall_seconds,
        },
        "outcome": {
            "status": "succeeded",
            "classification": "none",
            "reason_code": "none",
            "message": "C3 golden smoke training completed without retry.",
        },
        "retry": {
            "max_attempts": 2,
            "eligible": False,
            "same_seed": True,
            "same_config": True,
            "previous_attempt_id": {"availability": "missing", "reason": "not_applicable"},
        },
        "resources": resources,
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
            "experiment_id": "pr02-objectstate-baseline-v0",
            "trial_id": trial_id,
            "model_arm": arm,
            "config_id": golden["config_id"],
            "training_seed": seed,
        },
        "configuration": {
            "config_sha256": config_sha256,
            "hyperparameter_grid_sha256": manifest["frozen_inputs"]["hyperparameter_grid"]["sha256"],
            "optimizer_updates_max": 20000,
            "epochs_max": 200,
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
            "validation_primary_error": {
                "availability": "present",
                "value": fit.validation_primary_error,
            },
            "checkpoint_id": {"availability": "present", "value": checkpoint_id},
        },
        "attempt_ids": [attempt_id],
        "resources": resources,
        "provenance": provenance,
    }
    checkpoint_manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_kind": "objgauss.checkpoint_manifest",
        "identity": {
            "experiment_id": "pr02-objectstate-baseline-v0",
            "trial_id": trial_id,
            "checkpoint_id": checkpoint_id,
            "model_arm": arm,
            "training_seed": seed,
        },
        "selection": {
            "split": "validation",
            "metric": METRIC_NAME,
            "validation_primary_error": fit.validation_primary_error,
            "epoch": fit.best_epoch,
            "optimizer_update": fit.best_update,
        },
        "configuration": {
            "config_id": golden["config_id"],
            "config_sha256": config_sha256,
            "architecture_sha256": architecture_sha256,
            "parameter_count": parameter_count(fit.model),
            "optimizer_state_included": False,
        },
        "payload": checkpoint_artifact,
        "compatibility": {
            "python_version": "3.10.20",
            "torch_version": torch.__version__,
            "runtime_lock_sha256": manifest["frozen_inputs"]["runtime_lock"]["sha256"],
            "device_kind": device.type,
        },
        "final_evaluation_eligible": True,
        "provenance": provenance,
    }
    for relative, document in (
        (Path("attempts") / f"{attempt_id}.json", attempt),
        (Path("trials") / f"{trial_id}.json", trial),
        (Path("checkpoint-manifests") / f"{checkpoint_id}.json", checkpoint_manifest),
    ):
        atomic_write(output_root / relative, strict_json_bytes(document))
    prediction_jobs = _prediction_payloads(fit.model, validation_branches)
    prediction_hashes = canonical_sha256s(node, [payload for _, payload in prediction_jobs])
    predictions = []
    for (branch, payload), payload_sha256 in zip(prediction_jobs, prediction_hashes, strict=True):
        prediction_id = f"prediction-golden-{arm}-{sha256_bytes(strict_json_bytes([branch.group_id, branch.branch_id, seed]))[:20]}"
        document = {
            "schema_version": SCHEMA_VERSION,
            "contract_kind": "objgauss.dynamics_prediction",
            "identity": {
                "experiment_id": "pr02-objectstate-baseline-v0",
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
                "visible_fields": [
                    "initial_objectstate",
                    "commanded_action_schedule",
                    "non_future_metadata",
                ],
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
        relative = Path("predictions") / arm / f"{branch.branch_id}.json"
        atomic_write(output_root / relative, strict_json_bytes(document))
        predictions.append(
            {
                "group_id": branch.group_id,
                "branch_id": branch.branch_id,
                "model_arm": arm,
                "uri": relative.as_posix(),
                "sha256": file_sha256(output_root / relative),
                "prediction_payload_sha256": payload_sha256,
            }
        )
    predictions.sort(key=lambda item: (item["group_id"], item["branch_id"], item["model_arm"]))
    prediction_semantic = sha256_bytes(
        strict_json_bytes(
            [
                {
                    "group_id": item["group_id"],
                    "branch_id": item["branch_id"],
                    "prediction_payload_sha256": item["prediction_payload_sha256"],
                }
                for item in predictions
            ]
        )
    )
    return {
        "model_arm": arm,
        "trial_id": trial_id,
        "attempt_id": attempt_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config_sha256,
        "architecture_sha256": architecture_sha256,
        "parameter_count": parameter_count(fit.model),
        "optimizer_updates": fit.optimizer_updates,
        "epochs": fit.epochs,
        "data_order_sha256": fit.training_log["data_order_sha256"],
        "tensor_state_semantic_sha256": fit.training_log["tensor_state_semantic_sha256"],
        "validation_prediction_semantic_sha256": prediction_semantic,
        "validation_primary_error": fit.validation_primary_error,
        "minimum_display_vram_free_bytes": fit.minimum_display_vram_free_bytes,
        "peak_vram_bytes": fit.peak_vram_bytes,
        "gpu_hours": resources["gpu_hours"],
        "checkpoint_uri": checkpoint_relative.as_posix(),
        "checkpoint_sha256": checkpoint_artifact["sha256"],
        "training_log_uri": log_relative.as_posix(),
        "predictions": predictions,
        "manifest_sha256": manifest_sha256,
    }


def produce_golden(
    *,
    repo_root: Path,
    data_root: Path,
    formal_path: Path,
    dynamics_path: Path,
    manifest_path: Path,
    source_commit: str,
    output_root: Path,
    artifact_prefix: str,
    node: str,
    device_name: str,
    order: str,
) -> dict[str, Any]:
    require_offline()
    if SOURCE_COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise TrainerInvalidError("source commit must be 40 lowercase hexadecimal characters")
    if order not in {"canonical", "reverse"}:
        raise TrainerInvalidError("learned arm execution order must be canonical or reverse")
    if device_name != "cuda":
        raise TrainerInvalidError("golden training requires the frozen CUDA runtime")
    if not torch.cuda.is_available():
        raise TrainerBlockedError("CUDA is unavailable for golden training")
    device = torch.device("cuda")
    manifest = read_json(manifest_path)
    validate_manifest(repo_root, manifest)
    formal = read_json(formal_path)
    dynamics = read_json(dynamics_path)
    if file_sha256(formal_path) != manifest["frozen_inputs"]["formal_data_spec"]["sha256"]:
        raise TrainerInvalidError("formal data spec checksum drift")
    if dynamics.get("contract_kind") != "objgauss.dynamics_experiment":
        raise TrainerInvalidError("dynamics experiment contract drift")
    if output_root.exists() and any(output_root.iterdir()):
        raise TrainerInvalidError("golden output root must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    groups, loader_report = load_dataset(
        repo_root=repo_root,
        data_root=data_root,
        manifest_path=repo_root / "learning/data-boundary-manifest.json",
        formal_path=formal_path,
        dynamics_path=dynamics_path,
        source_commit=source_commit,
        splits=("train", "validation"),
    )
    golden = manifest["golden"]
    train_branches, validation_branches = select_golden_groups(
        groups=groups,
        formal=formal,
        train_group_id=golden["train_group_id"],
        validation_group_id=golden["validation_group_id"],
        node=node,
        device=device,
    )
    scales = manifest["loss"]["normalization_scales"]
    sequence = list(ARMS) if order == "canonical" else list(reversed(ARMS))
    arms = []
    for arm in sequence:
        started = time.monotonic()
        fit = fit_model(
            arm=arm,
            hidden_width=golden["hidden_width"],
            learning_rate=golden["learning_rate"],
            weight_decay=golden["weight_decay"],
            gradient_clip_norm=golden["gradient_clip_norm"],
            optimizer_updates=golden["optimizer_updates"],
            epochs=golden["epochs"],
            seed=golden["training_seed"],
            train_branches=train_branches,
            validation_branches=validation_branches,
            scales=scales,
            device=device,
        )
        arms.append(
            publish_arm(
                repo_root=repo_root,
                output_root=output_root,
                artifact_prefix=artifact_prefix,
                source_commit=source_commit,
                manifest=manifest,
                manifest_sha256=file_sha256(manifest_path),
                dynamics_sha256=file_sha256(dynamics_path),
                arm=arm,
                fit=fit,
                validation_branches=validation_branches,
                node=node,
                device=device,
                started_monotonic_s=started,
            )
        )
    arms.sort(key=lambda item: item["model_arm"])
    counts = {item["parameter_count"] for item in arms}
    updates = {item["optimizer_updates"] for item in arms}
    data_orders = {item["data_order_sha256"] for item in arms}
    if len(counts) != 1 or len(updates) != 1 or len(data_orders) != 1:
        raise TrainerInvalidError("learned arm fairness parity failed")
    semantic_index = sha256_bytes(
        strict_json_bytes(
            [
                {
                    "model_arm": item["model_arm"],
                    "tensor_state_semantic_sha256": item["tensor_state_semantic_sha256"],
                    "validation_prediction_semantic_sha256": item[
                        "validation_prediction_semantic_sha256"
                    ],
                    "optimizer_updates": item["optimizer_updates"],
                    "data_order_sha256": item["data_order_sha256"],
                }
                for item in arms
            ]
        )
    )
    index = {
        "index_version": REPORT_VERSION,
        "index_kind": "objgauss.pr02c-golden-training-index",
        "source_commit": source_commit,
        "execution_order": order,
        "trainer_manifest_sha256": file_sha256(manifest_path),
        "loader_report_sha256": sha256_bytes(strict_json_bytes(loader_report)),
        "data_index_sha256": loader_report["data_index_sha256"],
        "arms": arms,
        "semantic_index_sha256": semantic_index,
        "isolation": {
            "train_groups": [golden["train_group_id"]],
            "validation_groups": [golden["validation_group_id"]],
            "test_materialized": False,
            "hpo_config_selected": False,
            "formal_checkpoint_frozen": False,
        },
    }
    atomic_write(output_root / "index.json", strict_json_bytes(index))
    report = {
        "report_version": REPORT_VERSION,
        "report_kind": REPORT_KIND,
        "verdict": {"status": "supported", "reason_code": "golden_training_published"},
        "source_commit": source_commit,
        "execution_order": order,
        "counts": {
            "arms": 2,
            "train_groups": 1,
            "validation_groups": 1,
            "training_branches": len(train_branches),
            "validation_branches": len(validation_branches),
            "predictions": sum(len(item["predictions"]) for item in arms),
            "optimizer_updates_per_arm": golden["optimizer_updates"],
        },
        "fairness": {
            "parameter_count_equal": len(counts) == 1,
            "optimizer_updates_equal": len(updates) == 1,
            "data_order_equal": len(data_orders) == 1,
            "grid_equal": True,
            "seed_equal": True,
            "transition_shared_across_intervals": True,
        },
        "semantic_index_sha256": semantic_index,
        "resources": {
            "gpu_hours": sum(item["gpu_hours"] for item in arms),
            "peak_vram_bytes": max(item["peak_vram_bytes"] for item in arms),
            "minimum_display_vram_free_bytes": min(
                item["minimum_display_vram_free_bytes"] for item in arms
            ),
            "charged_to_hpo": False,
            "charged_to_formal": False,
        },
        "isolation": index["isolation"],
        "claim_boundary": manifest["claim_boundary"],
    }
    atomic_write(output_root / "report.json", strict_json_bytes(report))
    return report


def _tiny_branch(
    raw: dict[str, Any], *, group_id: str, split: str, symmetries: Tensor, device: torch.device
) -> TensorBranch:
    object_ids = tuple(item["object_id"] for item in raw["initial_object_states"])
    initial = torch.stack([state_tensor(item, device=device) for item in raw["initial_object_states"]])
    target_index = object_ids.index(raw["target_object_id"])
    label_rows = []
    for target_state in raw["future_target_states"]:
        row = initial.clone()
        row[target_index] = state_tensor(target_state, device=device)
        label_rows.append(row)
    command = raw["commanded_action"]
    action = torch.tensor(
        [
            *command["vector_W_N"],
            command["duration_s"],
            1.0 if command["kind"] == "push" else 0.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    return TensorBranch(
        group_id=group_id,
        split=split,
        branch_id=raw["branch_id"],
        object_ids=object_ids,
        initial_state=initial,
        commanded_action=action,
        target_index=target_index,
        labels=torch.stack(label_rows),
        symmetries_wxyz=symmetries,
        source_episode={},
        initial_objectstate_sha256="0" * 64,
        commanded_action_sha256="0" * 64,
    )


def run_tiny(
    *, repo_root: Path, fixture_path: Path, manifest_path: Path, output: Path
) -> dict[str, Any]:
    require_offline()
    manifest = read_json(manifest_path)
    validate_manifest(repo_root, manifest)
    fixture = read_json(fixture_path)
    if fixture.get("fixture_kind") != "objgauss.pr02c-tiny-training":
        raise TrainerInvalidError("tiny fixture kind drift")
    if fixture.get("rollout_times_s") != list(ROLLOUT_TIMES):
        raise TrainerInvalidError("tiny fixture rollout times drift")
    device = torch.device("cpu")
    symmetries = canonicalize_quaternion_tensor(
        torch.tensor(fixture["symmetry_rotations_wxyz"], dtype=torch.float32)
    )
    groups: dict[str, list[TensorBranch]] = {"train": [], "validation": []}
    for group in fixture.get("groups", []):
        split = group.get("split")
        if split not in groups:
            raise TrainerInvalidError(f"tiny trainer split is forbidden: {split}")
        groups[split].extend(
            _tiny_branch(
                raw,
                group_id=group["group_id"],
                split=split,
                symmetries=symmetries,
                device=device,
            )
            for raw in group["branches"]
        )
    train = tuple(sorted(groups["train"], key=lambda item: item.branch_id))
    validation = tuple(sorted(groups["validation"], key=lambda item: item.branch_id))
    scales = manifest["loss"]["normalization_scales"]
    results = []
    for arm in ARMS:
        fit = fit_model(
            arm=arm,
            hidden_width=64,
            learning_rate=0.001,
            weight_decay=0.00001,
            gradient_clip_norm=1.0,
            optimizer_updates=2,
            epochs=2,
            seed=2026071501,
            train_branches=train,
            validation_branches=validation,
            scales=scales,
            device=device,
        )
        results.append(
            {
                "model_arm": arm,
                "parameter_count": parameter_count(fit.model),
                "optimizer_updates": fit.optimizer_updates,
                "data_order_sha256": fit.training_log["data_order_sha256"],
                "tensor_state_semantic_sha256": fit.training_log[
                    "tensor_state_semantic_sha256"
                ],
                "validation_primary_error": fit.validation_primary_error,
            }
        )
    report = {
        "report_version": REPORT_VERSION,
        "report_kind": "objgauss.pr02c-tiny-training-report",
        "verdict": "supported",
        "device": "cpu",
        "arms": results,
        "fairness": {
            "parameter_count_equal": len({item["parameter_count"] for item in results}) == 1,
            "optimizer_updates_equal": len({item["optimizer_updates"] for item in results}) == 1,
            "data_order_equal": len({item["data_order_sha256"] for item in results}) == 1,
        },
        "rollout": {
            "times_s": list(ROLLOUT_TIMES),
            "delta_t_s": [0.1, 0.1, 0.3, 0.6],
            "teacher_forcing": "initial-state-only",
        },
        "claim_boundary": "CPU tiny fixture validates smoke behavior only",
    }
    if not all(report["fairness"].values()):
        raise TrainerInvalidError("CPU tiny learned arm fairness failed")
    atomic_write(output, strict_json_bytes(report))
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--mode", choices=("tiny", "golden"), required=True)
    value.add_argument("--manifest", type=Path, default=Path("learning/trainer-manifest.json"))
    value.add_argument("--tiny-fixture", type=Path, default=Path("learning/fixtures/pr02c-tiny-training.json"))
    value.add_argument("--output", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--formal-spec", type=Path)
    value.add_argument("--dynamics-experiment", type=Path)
    value.add_argument("--source-commit")
    value.add_argument("--output-root", type=Path)
    value.add_argument("--artifact-prefix", default="generated/pr02c/trainer")
    value.add_argument("--node", default="node")
    value.add_argument("--device", choices=("cpu", "cuda"))
    value.add_argument("--order", choices=("canonical", "reverse"), default="canonical")
    value.add_argument("--splits", nargs="+", default=["train", "validation"])
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    resolve = lambda path: path if path.is_absolute() else repo_root / path
    try:
        if any(split not in {"train", "validation"} for split in args.splits):
            raise TrainerInvalidError("trainer split is forbidden; final test is evaluator-only")
        if args.mode == "tiny":
            if args.device not in {None, "cpu"} or args.output is None:
                raise TrainerInvalidError("tiny mode requires --output and CPU device")
            report = run_tiny(
                repo_root=repo_root,
                fixture_path=resolve(args.tiny_fixture).resolve(),
                manifest_path=resolve(args.manifest).resolve(),
                output=resolve(args.output).resolve(),
            )
        else:
            required = {
                "data_root": args.data_root,
                "formal_spec": args.formal_spec,
                "dynamics_experiment": args.dynamics_experiment,
                "source_commit": args.source_commit,
                "output_root": args.output_root,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise TrainerInvalidError(f"golden mode is missing arguments: {','.join(missing)}")
            report = produce_golden(
                repo_root=repo_root,
                data_root=resolve(args.data_root).resolve(),
                formal_path=resolve(args.formal_spec).resolve(),
                dynamics_path=resolve(args.dynamics_experiment).resolve(),
                manifest_path=resolve(args.manifest).resolve(),
                source_commit=args.source_commit,
                output_root=resolve(args.output_root).resolve(),
                artifact_prefix=args.artifact_prefix,
                node=args.node,
                device_name=args.device or "cuda",
                order=args.order,
            )
        sys.stdout.buffer.write(strict_json_bytes(report))
        return 0
    except (CanonicalHashError, DataBlockedError, TrainerBlockedError) as error:
        report = {"verdict": "blocked", "reason_code": "required_input_or_device_missing", "message": str(error)}
        exit_code = 3
    except (TrainerRejectedError,) as error:
        report = {"verdict": "rejected", "reason_code": "golden_training_failed", "message": str(error)}
        exit_code = 2
    except (
        DataInvalidError,
        ModelInvalidError,
        TrainerInvalidError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        report = {"verdict": "invalid", "reason_code": "trainer_protocol_invalid", "message": str(error)}
        exit_code = 4
    output = args.output if args.mode == "tiny" else None
    if output is not None:
        atomic_write(resolve(output), strict_json_bytes(report))
    sys.stdout.buffer.write(strict_json_bytes(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
