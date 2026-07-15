"""Fail-closed PR-02C train/validation loader with separated inputs and labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from .canonical import CanonicalHashError, canonical_sha256, canonical_sha256s
from .runtime import atomic_write, file_sha256, strict_json_bytes


REPORT_VERSION = "0.1.0"
REPORT_KIND = "objgauss.pr02c-loader-report"
INPUT_BUNDLE_VERSION = "0.1.0"
INPUT_BUNDLE_KIND = "objgauss.pr02c-model-input-bundle"
ALLOWED_SPLITS = ("train", "validation")
BRANCH_IDS = (
    "hold",
    "push-neg-x-weak",
    "push-pos-x-strong",
    "push-pos-x-weak",
    "push-pos-y-weak",
)
FILES = (
    "attempt.json",
    "contact-ledger.json",
    "episode.json",
    "publication.json",
    "trajectory.json",
)
SCORING_TIMES = (0.1, 0.2, 0.5, 1.1)
FORBIDDEN_MODEL_FIELDS = frozenset(
    {"executed_action", "control_ledger", "future_object_states", "gt_future"}
)


class DataBlockedError(RuntimeError):
    """Required source data is not available."""


class DataInvalidError(RuntimeError):
    """Source structure, lineage, checksums, or feature boundaries are invalid."""


@dataclass(frozen=True)
class ActorState:
    actor_id: str
    position_W_m: tuple[float, float, float]
    quaternion_WO_wxyz: tuple[float, float, float, float]
    linear_velocity_W_m_s: tuple[float, float, float]
    angular_velocity_W_rad_s: tuple[float, float, float]


@dataclass(frozen=True)
class CommandedAction:
    kind: str
    vector_W_N: tuple[float, float, float]
    duration_s: float
    sim_frequency_hz: int
    applied_steps: int


@dataclass(frozen=True)
class ModelInputs:
    initial_object_states: tuple[ActorState, ...]
    commanded_action: CommandedAction
    target_object_id: str
    rollout_times_s: tuple[float, float, float, float]


@dataclass(frozen=True)
class TimedStates:
    episode_time_s: float
    object_states: tuple[ActorState, ...]


@dataclass(frozen=True)
class TrainingLabels:
    future_object_states: tuple[TimedStates, ...]


@dataclass(frozen=True)
class SourceEpisodeReference:
    uri: str
    sha256: str
    lineage_sha256: str


@dataclass(frozen=True)
class BranchSample:
    group_id: str
    split: str
    branch_id: str
    model_inputs: ModelInputs
    labels: TrainingLabels
    semantic_sha256: str
    source_episode: SourceEpisodeReference


@dataclass(frozen=True)
class GroupSample:
    group_id: str
    split: str
    branches: tuple[BranchSample, ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise DataBlockedError(f"required source artifact is missing: {path}") from error
    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DataInvalidError(f"invalid JSON source artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataInvalidError(f"source JSON must be an object: {path}")
    return payload, value


def require_splits(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not values or len(values) != len(set(values)):
        raise DataInvalidError("loader splits must be non-empty and unique")
    forbidden = [value for value in values if value not in ALLOWED_SPLITS]
    if forbidden:
        raise DataInvalidError(f"loader split is forbidden: {','.join(sorted(forbidden))}")
    return tuple(split for split in ALLOWED_SPLITS if split in values)


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise DataInvalidError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise DataInvalidError(f"{name} contains a non-finite value")
    return result


def actor_states(raw: Any) -> tuple[ActorState, ...]:
    if not isinstance(raw, dict) or not raw:
        raise DataInvalidError("object state map must be a non-empty object")
    states = []
    for actor_id in sorted(raw):
        state = raw[actor_id]
        if not isinstance(state, dict):
            raise DataInvalidError(f"actor state must be an object: {actor_id}")
        quaternion = _finite_vector(state.get("quaternion_WO_wxyz"), 4, "quaternion")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise DataInvalidError(f"actor quaternion is not normalized: {actor_id}")
        states.append(
            ActorState(
                actor_id=actor_id,
                position_W_m=_finite_vector(state.get("position_W_m"), 3, "position"),
                quaternion_WO_wxyz=quaternion,
                linear_velocity_W_m_s=_finite_vector(
                    state.get("linear_velocity_W_m_s"), 3, "linear velocity"
                ),
                angular_velocity_W_rad_s=_finite_vector(
                    state.get("angular_velocity_W_rad_s"), 3, "angular velocity"
                ),
            )
        )
    return tuple(states)


def build_model_inputs(
    episode: dict[str, Any], trajectory: dict[str, Any]
) -> ModelInputs:
    intervention = episode.get("intervention")
    if not isinstance(intervention, dict) or "commanded_action" not in intervention:
        raise DataInvalidError("commanded action is required")
    command = intervention["commanded_action"]
    if not isinstance(command, dict):
        raise DataInvalidError("commanded action must be an object")
    records = trajectory.get("records")
    if not isinstance(records, list) or not records or records[0].get("episode_time_s") != 0:
        raise DataInvalidError("trajectory must start at physical time 0.0")
    return ModelInputs(
        initial_object_states=actor_states(records[0].get("actors")),
        commanded_action=CommandedAction(
            kind=str(command.get("kind")),
            vector_W_N=_finite_vector(command.get("vector_W_N"), 3, "command vector"),
            duration_s=float(command.get("duration_s")),
            sim_frequency_hz=int(command.get("sim_frequency_hz")),
            applied_steps=int(command.get("applied_steps")),
        ),
        target_object_id=str(intervention.get("target_object_id")),
        rollout_times_s=SCORING_TIMES,
    )


def build_labels(trajectory: dict[str, Any]) -> TrainingLabels:
    records = trajectory.get("records")
    if not isinstance(records, list):
        raise DataInvalidError("trajectory records are missing")
    times = [record.get("episode_time_s") for record in records]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in times
    ):
        raise DataInvalidError("trajectory time is non-finite")
    if times != sorted(times) or len(times) != len(set(times)):
        raise DataInvalidError("trajectory times must be strictly increasing")
    by_time = {float(record["episode_time_s"]): record for record in records}
    if max(by_time, default=0.0) > SCORING_TIMES[-1]:
        raise DataInvalidError("trajectory exceeds the frozen horizon")
    try:
        labels = tuple(
            TimedStates(time, actor_states(by_time[time].get("actors")))
            for time in SCORING_TIMES
        )
    except KeyError as error:
        raise DataInvalidError(f"frozen scoring time is missing: {error.args[0]}") from error
    return TrainingLabels(future_object_states=labels)


def model_payload(inputs: ModelInputs) -> dict[str, Any]:
    def state_payload(state: ActorState) -> dict[str, Any]:
        return {
            "actor_id": state.actor_id,
            "position_W_m": list(state.position_W_m),
            "quaternion_WO_wxyz": list(state.quaternion_WO_wxyz),
            "linear_velocity_W_m_s": list(state.linear_velocity_W_m_s),
            "angular_velocity_W_rad_s": list(state.angular_velocity_W_rad_s),
        }

    payload = {
        "initial_object_states": [
            state_payload(state) for state in inputs.initial_object_states
        ],
        "commanded_action": {
            "kind": inputs.commanded_action.kind,
            "vector_W_N": list(inputs.commanded_action.vector_W_N),
            "duration_s": inputs.commanded_action.duration_s,
            "sim_frequency_hz": inputs.commanded_action.sim_frequency_hz,
            "applied_steps": inputs.commanded_action.applied_steps,
        },
        "target_object_id": inputs.target_object_id,
        "rollout_times_s": list(inputs.rollout_times_s),
    }
    assert_model_payload(payload)
    return payload


def assert_model_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise DataInvalidError("model input payload must be an object")
    if set(payload) != {
        "initial_object_states",
        "commanded_action",
        "target_object_id",
        "rollout_times_s",
    }:
        raise DataInvalidError("model input field set drift")

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in FORBIDDEN_MODEL_FIELDS:
                    raise DataInvalidError(f"forbidden model input field: {key}")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)


def _semantic_sha(publication: dict[str, Any]) -> str:
    document = {
        "contact_ledger_sha256": publication["contact_ledger_sha256"],
        "episode_sha256": publication["episode_sha256"],
        "trajectory_sha256": publication["trajectory_sha256"],
    }
    return sha256_bytes(strict_json_bytes(document))


def load_branch(
    *,
    repo_root: Path,
    data_root: Path,
    directory: Path,
    group_id: str,
    split: str,
    branch_id: str,
    expected_group: dict[str, Any],
    expected_action: dict[str, Any],
    source_commit: str,
    source_plan_sha256: str,
) -> tuple[BranchSample, dict[str, str], dict[str, Any]]:
    names = tuple(sorted(path.name for path in directory.iterdir() if path.is_file()))
    if names != FILES:
        raise DataInvalidError(f"branch file set drift: {group_id}/{branch_id}: {names}")
    loaded = {name: read_json(directory / name) for name in FILES}
    publication = loaded["publication.json"][1]
    for name, key in (
        ("episode.json", "episode_sha256"),
        ("trajectory.json", "trajectory_sha256"),
        ("contact-ledger.json", "contact_ledger_sha256"),
        ("attempt.json", "attempt_sha256"),
    ):
        if publication.get(key) != sha256_bytes(loaded[name][0]):
            raise DataInvalidError(f"publication checksum mismatch: {group_id}/{branch_id}/{name}")
    if publication.get("semantic_sha256") != _semantic_sha(publication):
        raise DataInvalidError(f"semantic checksum mismatch: {group_id}/{branch_id}")
    episode = loaded["episode.json"][1]
    trajectory = loaded["trajectory.json"][1]
    attempt = loaded["attempt.json"][1]
    identity = episode.get("identity", {})
    if identity.get("group_id") != group_id or identity.get("branch_id") != branch_id:
        raise DataInvalidError("episode identity differs from source path")
    if identity.get("split") != split or split not in ALLOWED_SPLITS:
        raise DataInvalidError("episode split is forbidden or differs from frozen partition")
    environment = episode.get("environment", {})
    if (
        environment.get("object_spec_id") != expected_group["object_identity_id"]
        or environment.get("layout_id") != expected_group["layout_id"]
        or environment.get("start_pose_id") != expected_group["start_pose_id"]
        or episode.get("initialization", {}).get("reset_seed")
        != expected_group["reset_seed"]
    ):
        raise DataInvalidError("episode group design differs from frozen formal spec")
    command = episode.get("intervention", {}).get("commanded_action")
    expected_command = {key: value for key, value in expected_action.items() if key != "branch_id"}
    if command != expected_command:
        raise DataInvalidError("commanded action differs from frozen schedule")
    if episode.get("provenance", {}).get("source_commit") != source_commit:
        raise DataInvalidError("episode source commit differs from accepted HEAD")
    if attempt.get("provenance", {}).get("experiment_manifest_sha256") != source_plan_sha256:
        raise DataInvalidError("attempt source plan lineage differs")
    trajectory_descriptor = episode.get("evidence", {}).get("trajectory", {})
    contact_descriptor = episode.get("evidence", {}).get("contact_ledger", {})
    for descriptor, name in (
        (trajectory_descriptor, "trajectory.json"),
        (contact_descriptor, "contact-ledger.json"),
    ):
        expected_uri = (directory / name).relative_to(data_root).as_posix()
        if (
            descriptor.get("uri") != expected_uri
            or descriptor.get("sha256") != sha256_bytes(loaded[name][0])
            or descriptor.get("byte_length") != len(loaded[name][0])
            or descriptor.get("record_count") != len(loaded[name][1].get("records", []))
        ):
            raise DataInvalidError(f"episode descriptor mismatch: {group_id}/{branch_id}/{name}")
    inputs = build_model_inputs(episode, trajectory)
    payload = model_payload(inputs)
    labels = build_labels(trajectory)
    episode_uri = (directory / "episode.json").resolve().relative_to(
        repo_root.resolve()
    ).as_posix()
    lineage_document = {
        "source_commit": source_commit,
        "source_plan_sha256": source_plan_sha256,
        "episode_sha256": publication["episode_sha256"],
        "semantic_sha256": publication["semantic_sha256"],
    }
    sample = BranchSample(
        group_id=group_id,
        split=split,
        branch_id=branch_id,
        model_inputs=inputs,
        labels=labels,
        semantic_sha256=publication["semantic_sha256"],
        source_episode=SourceEpisodeReference(
            uri=episode_uri,
            sha256=publication["episode_sha256"],
            lineage_sha256=sha256_bytes(strict_json_bytes(lineage_document)),
        ),
    )
    index = {
        "group_id": group_id,
        "branch_id": branch_id,
        "episode_sha256": publication["episode_sha256"],
        "trajectory_sha256": publication["trajectory_sha256"],
        "semantic_sha256": publication["semantic_sha256"],
    }
    return sample, index, payload


def validate_manifest_inputs(repo_root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("manifest_kind") != "objgauss.pr02c-data-boundary":
        raise DataInvalidError("data boundary manifest kind drift")
    for name in ("pilot_spec", "source_experiment", "simulator_lock"):
        entry = manifest.get("frozen_inputs", {}).get(name)
        if not isinstance(entry, dict):
            raise DataInvalidError(f"missing frozen input: {name}")
        path = (repo_root / entry["path"]).resolve()
        if file_sha256(path) != entry.get("sha256"):
            raise DataInvalidError(f"frozen input checksum drift: {name}")


def load_dataset(
    *,
    repo_root: Path,
    data_root: Path,
    manifest_path: Path,
    formal_path: Path,
    dynamics_path: Path,
    source_commit: str,
    splits: tuple[str, ...],
) -> tuple[tuple[GroupSample, ...], dict[str, Any]]:
    splits = require_splits(splits)
    _, manifest = read_json(manifest_path)
    formal_bytes, formal = read_json(formal_path)
    dynamics_bytes, dynamics = read_json(dynamics_path)
    plan_bytes, plan = read_json(data_root / "source-plan.json")
    report_bytes, source_report = read_json(data_root / "source-report.json")
    del report_bytes
    validate_manifest_inputs(repo_root, manifest)
    formal_sha256 = sha256_bytes(formal_bytes)
    expected_formal_sha = manifest["frozen_inputs"]["formal_data_spec"]["sha256"]
    if formal_sha256 != expected_formal_sha:
        raise DataInvalidError("formal data spec checksum drift")
    if plan.get("plan_kind") != "objgauss.pr02c-source-plan":
        raise DataInvalidError("source plan kind drift")
    if plan.get("identity", {}).get("experiment_spec_sha256") != formal_sha256:
        raise DataInvalidError("source plan does not bind the formal data spec")
    if plan.get("materialization", {}).get("splits") != list(splits):
        raise DataInvalidError("source plan split set differs from loader request")
    if plan.get("materialization", {}).get("test_materialized") is not False:
        raise DataInvalidError("source plan materialized final test")
    if source_report.get("producer", {}).get("source_commit") != source_commit:
        raise DataInvalidError("source report commit differs from accepted HEAD")
    if source_report.get("inputs", {}).get("formal_data_spec_sha256") != formal_sha256:
        raise DataInvalidError("source report formal lineage drift")
    if source_report.get("inputs", {}).get("dynamics_experiment_sha256") != sha256_bytes(dynamics_bytes):
        raise DataInvalidError("source report dynamics lineage drift")
    if source_report.get("inputs", {}).get("source_plan_sha256") != sha256_bytes(plan_bytes):
        raise DataInvalidError("source report plan lineage drift")
    if source_report.get("isolation", {}).get("test_materialized") is not False:
        raise DataInvalidError("source report claims final test materialization")
    experiment_id = formal.get("experiment_id")
    group_root = data_root / "dataset" / str(experiment_id)
    if not group_root.is_dir():
        raise DataBlockedError("materialized source dataset root is missing")
    expected_groups = {
        group["group_id"]: (split, group)
        for split in splits
        for group in formal["partitions"][split]["groups"]
    }
    actual_group_ids = {path.name for path in group_root.iterdir() if path.is_dir()}
    if actual_group_ids != set(expected_groups):
        raise DataInvalidError("materialized group set differs from train/validation freeze")
    test_partition = formal["partitions"]["test"]
    forbidden_tokens = {
        *test_partition["objects"],
        *test_partition["layouts"],
        *(group["group_id"] for group in test_partition["groups"]),
    }
    if actual_group_ids.intersection(forbidden_tokens):
        raise DataInvalidError("final test identity was materialized")
    actions = {item["branch_id"]: item for item in formal["actions"]}
    source_plan_sha256 = sha256_bytes(plan_bytes)
    groups: list[GroupSample] = []
    index: list[dict[str, str]] = []
    model_payload_hashes = []
    lineage_by_split: dict[str, set[tuple[str, str, str]]] = {
        split: set() for split in splits
    }
    for group_id in sorted(expected_groups):
        split, expected_group = expected_groups[group_id]
        directory = group_root / group_id
        actual_branches = tuple(sorted(path.name for path in directory.iterdir() if path.is_dir()))
        if actual_branches != BRANCH_IDS:
            raise DataInvalidError(f"branch set differs for {group_id}")
        branches = []
        initializations = []
        for branch_id in BRANCH_IDS:
            branch, branch_index, payload = load_branch(
                repo_root=repo_root,
                data_root=data_root,
                directory=directory / branch_id,
                group_id=group_id,
                split=split,
                branch_id=branch_id,
                expected_group=expected_group,
                expected_action=actions[branch_id],
                source_commit=source_commit,
                source_plan_sha256=source_plan_sha256,
            )
            branches.append(branch)
            index.append(branch_index)
            model_payload_hashes.append(sha256_bytes(strict_json_bytes(payload)))
            _, episode = read_json(directory / branch_id / "episode.json")
            initialization = episode["initialization"]
            initializations.append(initialization)
        if any(item != initializations[0] for item in initializations[1:]):
            raise DataInvalidError(f"sibling initialization drift: {group_id}")
        lineage = initializations[0]
        lineage_key = (
            lineage["snapshot_sha256"],
            lineage["initial_state_sha256"],
            lineage["restored_rng_sha256"],
        )
        for other_split, values in lineage_by_split.items():
            if other_split != split and lineage_key in values:
                raise DataInvalidError("initialization lineage crosses train/validation")
        lineage_by_split[split].add(lineage_key)
        groups.append(GroupSample(group_id, split, tuple(branches)))
    index.sort(key=lambda item: (item["group_id"], item["branch_id"]))
    data_index_sha256 = sha256_bytes(strict_json_bytes(index))
    if source_report.get("data_index_sha256") != data_index_sha256:
        raise DataInvalidError("source report data index differs from loader recomputation")
    report = {
        "report_version": REPORT_VERSION,
        "report_kind": REPORT_KIND,
        "verdict": {"status": "supported", "reason_code": "all_c1_loader_gates_passed"},
        "source_commit": source_commit,
        "inputs": {
            "data_boundary_manifest_sha256": file_sha256(manifest_path),
            "formal_data_spec_sha256": formal_sha256,
            "dynamics_experiment_sha256": sha256_bytes(dynamics_bytes),
            "source_plan_sha256": source_plan_sha256,
            "source_report_sha256": file_sha256(data_root / "source-report.json"),
        },
        "counts": {
            "groups": len(groups),
            "branches": len(index),
            "split_groups": {
                split: sum(group.split == split for group in groups)
                for split in ALLOWED_SPLITS
            },
        },
        "isolation": {
            "test_materialized": False,
            "executed_action_is_model_input": False,
            "future_gt_is_model_input": False,
            "model_input_fields": list(manifest["loader_boundary"]["model_input_fields"]),
            "label_fields": list(manifest["loader_boundary"]["label_fields"]),
        },
        "data_index_sha256": data_index_sha256,
        "model_input_index_sha256": sha256_bytes(
            strict_json_bytes(sorted(model_payload_hashes))
        ),
        "claim_boundary": {
            "supported_claim": "train-validation-loader-boundary-is-valid",
            "excluded_claims": [
                "test-source-materialized",
                "trainer-implemented",
                "model-performance",
                "scientific-baseline-comparison",
            ],
        },
    }
    return tuple(groups), report


def build_model_input_bundle(
    *,
    repo_root: Path,
    groups: tuple[GroupSample, ...],
    loader_report: dict[str, Any],
    source_commit: str,
    node: str,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for group in groups:
        if group.split != "validation":
            continue
        for branch in group.branches:
            inputs = branch.model_inputs
            initial_object_states = [
                {
                    "object_id": state.actor_id,
                    "position_W_m": list(state.position_W_m),
                    "quaternion_WO_wxyz": list(state.quaternion_WO_wxyz),
                    "linear_velocity_W_m_s": list(state.linear_velocity_W_m_s),
                    "angular_velocity_W_rad_s": list(state.angular_velocity_W_rad_s),
                }
                for state in inputs.initial_object_states
            ]
            commanded_action = {
                "kind": inputs.commanded_action.kind,
                "vector_W_N": list(inputs.commanded_action.vector_W_N),
                "duration_s": inputs.commanded_action.duration_s,
                "sim_frequency_hz": inputs.commanded_action.sim_frequency_hz,
                "applied_steps": inputs.commanded_action.applied_steps,
            }
            samples.append(
                {
                    "group_id": branch.group_id,
                    "branch_id": branch.branch_id,
                    "split": branch.split,
                    "source_episode": {
                        "uri": branch.source_episode.uri,
                        "media_type": "application/json",
                        "sha256": branch.source_episode.sha256,
                        "schema_version": "0.2.0",
                        "lineage_sha256": branch.source_episode.lineage_sha256,
                    },
                    "initial_objectstate_sha256": "",
                    "commanded_action_sha256": "",
                    "target_object_id": inputs.target_object_id,
                    "initial_object_states": initial_object_states,
                    "commanded_action": commanded_action,
                    "rollout_times_s": list(inputs.rollout_times_s),
                }
            )
    samples.sort(key=lambda item: (item["group_id"], item["branch_id"]))
    if len(samples) != 60:
        raise DataInvalidError("validation input bundle must contain exactly 60 branches")
    hashes = canonical_sha256s(
        node,
        [
            value
            for sample in samples
            for value in (sample["initial_object_states"], sample["commanded_action"])
        ],
    )
    hash_by_identity = {
        (sample["group_id"], sample["branch_id"]): (hashes[index], hashes[index + 1])
        for index, sample in zip(range(0, len(hashes), 2), samples, strict=True)
    }
    for sample in samples:
        initial_sha256, command_sha256 = hash_by_identity[
            (sample["group_id"], sample["branch_id"])
        ]
        sample["initial_objectstate_sha256"] = initial_sha256
        sample["commanded_action_sha256"] = command_sha256
    bundle = {
        "bundle_version": INPUT_BUNDLE_VERSION,
        "bundle_kind": INPUT_BUNDLE_KIND,
        "source_commit": source_commit,
        "experiment_id": "pr02-objectstate-baseline-v0",
        "split": "validation",
        "inputs": {
            **loader_report["inputs"],
            "data_index_sha256": loader_report["data_index_sha256"],
            "model_input_index_sha256": loader_report["model_input_index_sha256"],
            "loader_report_sha256": sha256_bytes(strict_json_bytes(loader_report)),
            "runtime_lock_sha256": file_sha256(repo_root / "learning/uv.lock"),
        },
        "samples": samples,
        "sample_payload_sha256": canonical_sha256(node, samples),
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
    assert_model_input_bundle(bundle, node=node)
    return bundle


def assert_model_input_bundle(bundle: Any, *, node: str) -> None:
    if not isinstance(bundle, dict):
        raise DataInvalidError("model input bundle must be an object")
    forbidden = FORBIDDEN_MODEL_FIELDS | {"labels", "training_labels"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden:
                    raise DataInvalidError(f"forbidden model input bundle field: {key}")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(bundle.get("samples"))
    if bundle.get("bundle_kind") != INPUT_BUNDLE_KIND:
        raise DataInvalidError("model input bundle kind drift")
    if bundle.get("split") != "validation":
        raise DataInvalidError("model input bundle split must be validation")
    samples = bundle.get("samples")
    if not isinstance(samples, list) or len(samples) != 60:
        raise DataInvalidError("model input bundle sample count drift")
    if any(sample.get("split") != "validation" for sample in samples):
        raise DataInvalidError("model input bundle contains a forbidden split")
    if bundle.get("sample_payload_sha256") != canonical_sha256(node, samples):
        raise DataInvalidError("model input bundle payload checksum drift")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("learning/data-boundary-manifest.json"))
    parser.add_argument("--formal-spec", type=Path, required=True)
    parser.add_argument("--dynamics-experiment", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-input-output", type=Path)
    parser.add_argument("--node", default="node")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    resolve = lambda path: path if path.is_absolute() else repo_root / path
    output = resolve(args.output)
    try:
        groups, report = load_dataset(
            repo_root=repo_root,
            data_root=resolve(args.data_root).resolve(),
            manifest_path=resolve(args.manifest).resolve(),
            formal_path=resolve(args.formal_spec).resolve(),
            dynamics_path=resolve(args.dynamics_experiment).resolve(),
            source_commit=args.source_commit,
            splits=require_splits(args.splits),
        )
        if args.model_input_output is not None:
            bundle = build_model_input_bundle(
                repo_root=repo_root,
                groups=groups,
                loader_report=report,
                source_commit=args.source_commit,
                node=args.node,
            )
            atomic_write(
                resolve(args.model_input_output), strict_json_bytes(bundle)
            )
        exit_code = 0
    except (CanonicalHashError, DataBlockedError) as error:
        report = {
            "report_version": REPORT_VERSION,
            "report_kind": REPORT_KIND,
            "verdict": {"status": "blocked", "reason_code": "source_missing"},
            "message": str(error),
            "claim_boundary": {"supported_claim": "none"},
        }
        exit_code = 3
    except (DataInvalidError, OSError, ValueError, KeyError, TypeError) as error:
        report = {
            "report_version": REPORT_VERSION,
            "report_kind": REPORT_KIND,
            "verdict": {"status": "invalid", "reason_code": "data_boundary_invalid"},
            "message": str(error),
            "claim_boundary": {"supported_claim": "none"},
        }
        exit_code = 4
    atomic_write(output, strict_json_bytes(report))
    sys.stdout.buffer.write(strict_json_bytes(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
