"""Deterministic PR-02C validation baselines over sanitized model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, NoReturn

from .canonical import CanonicalHashError, canonical_sha256, canonical_sha256s
from .runtime import atomic_write, file_sha256, package_tree_sha256, strict_json_bytes


MANIFEST_KIND = "objgauss.pr02c-deterministic-baselines"
BUNDLE_KIND = "objgauss.pr02c-model-input-bundle"
REPORT_KIND = "objgauss.pr02c-deterministic-baseline-report"
REPORT_VERSION = "0.1.0"
ARMS = ("copy_state", "constant_velocity")
BRANCH_IDS = (
    "hold",
    "push-neg-x-weak",
    "push-pos-x-strong",
    "push-pos-x-weak",
    "push-pos-y-weak",
)
SCORING_TIMES = (0.1, 0.2, 0.5, 1.1)
SOURCE_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40}$")
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
FORBIDDEN_FIELDS = frozenset(
    {
        "executed_action",
        "control_ledger",
        "future_object_states",
        "gt_future",
        "labels",
        "training_labels",
    }
)


class BaselineBlockedError(RuntimeError):
    """A required frozen input or runtime component is unavailable."""


class BaselineInvalidError(RuntimeError):
    """The sanitized input, lineage, or deterministic output is invalid."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(path.read_bytes(), parse_constant=reject_constant)
    except FileNotFoundError as error:
        raise BaselineBlockedError(f"required input is missing: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BaselineInvalidError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise BaselineInvalidError(f"JSON input must be an object: {path}")
    return value


def require_offline() -> None:
    if os.environ.get("OBJGAUSS_LEARNING_OFFLINE") != "1":
        raise BaselineInvalidError("OBJGAUSS_LEARNING_OFFLINE must be exactly '1'")


def finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise BaselineInvalidError(f"{name} must contain exactly {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise BaselineInvalidError(f"{name} contains a non-finite value")
    return result


def canonical_quaternion(value: Any) -> list[float]:
    quaternion = finite_vector(value, 4, "quaternion")
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise BaselineInvalidError("input quaternion is not normalized")
    normalized = [component / norm for component in quaternion]
    first = next((item for item in normalized if abs(item) > 1e-12), None)
    if first is not None and first < 0:
        normalized = [-item for item in normalized]
    return [0.0 if item == 0 else item for item in normalized]


def quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def integrate_world_quaternion(
    quaternion_wxyz: Any, angular_velocity_W_rad_s: Any, time_s: float
) -> list[float]:
    initial = canonical_quaternion(quaternion_wxyz)
    omega = finite_vector(angular_velocity_W_rad_s, 3, "angular velocity")
    speed = math.sqrt(sum(component * component for component in omega))
    if speed == 0:
        return initial
    half_angle = 0.5 * speed * time_s
    scale = math.sin(half_angle) / speed
    delta = [math.cos(half_angle), *(component * scale for component in omega)]
    return canonical_quaternion(quaternion_multiply(delta, initial))


def state_prediction(state: dict[str, Any], *, arm: str, time_s: float) -> dict[str, Any]:
    if set(state) != {
        "object_id",
        "position_W_m",
        "quaternion_WO_wxyz",
        "linear_velocity_W_m_s",
        "angular_velocity_W_rad_s",
    }:
        raise BaselineInvalidError("initial ObjectState field set drift")
    object_id = state["object_id"]
    if not isinstance(object_id, str) or not object_id:
        raise BaselineInvalidError("object_id must be a non-empty string")
    position = finite_vector(state["position_W_m"], 3, "position")
    linear = finite_vector(state["linear_velocity_W_m_s"], 3, "linear velocity")
    angular = finite_vector(state["angular_velocity_W_rad_s"], 3, "angular velocity")
    if arm == "copy_state":
        predicted_position = list(position)
        predicted_quaternion = canonical_quaternion(state["quaternion_WO_wxyz"])
    elif arm == "constant_velocity":
        predicted_position = [
            coordinate + velocity * time_s
            for coordinate, velocity in zip(position, linear, strict=True)
        ]
        predicted_quaternion = integrate_world_quaternion(
            state["quaternion_WO_wxyz"], angular, time_s
        )
    else:
        raise BaselineInvalidError(f"unknown deterministic arm: {arm}")
    return {
        "object_id": object_id,
        "position_W_m": [0.0 if item == 0 else item for item in predicted_position],
        "quaternion_WO_wxyz": predicted_quaternion,
        "linear_velocity_W_m_s": [0.0 if item == 0 else item for item in linear],
        "angular_velocity_W_rad_s": [0.0 if item == 0 else item for item in angular],
    }


def predict(sample: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    states = sample.get("initial_object_states")
    if not isinstance(states, list) or not states:
        raise BaselineInvalidError("initial_object_states must be a non-empty array")
    object_ids = [state.get("object_id") for state in states if isinstance(state, dict)]
    if len(object_ids) != len(states) or len(set(object_ids)) != len(states):
        raise BaselineInvalidError("initial object IDs must be unique")
    if sample.get("target_object_id") not in object_ids:
        raise BaselineInvalidError("target object is absent from initial state")
    return [
        {
            "time_s": time_s,
            "objects": [
                state_prediction(state, arm=arm, time_s=time_s)
                for state in sorted(states, key=lambda item: item["object_id"])
            ],
        }
        for time_s in SCORING_TIMES
    ]


def canonical_payload_sha256(node: str, value: Any) -> str:
    return canonical_sha256(node, value)


def validate_manifest(repo_root: Path, manifest: dict[str, Any]) -> None:
    if set(manifest) != {
        "manifest_version",
        "manifest_kind",
        "arms",
        "inference_boundary",
        "horizon",
        "semantics",
        "frozen_inputs",
        "outputs",
        "claim_boundary",
    } or manifest.get("manifest_version") != "0.1.0":
        raise BaselineInvalidError("baseline manifest top-level contract drift")
    if manifest.get("manifest_kind") != MANIFEST_KIND:
        raise BaselineInvalidError("baseline manifest kind drift")
    if manifest.get("arms") != list(ARMS):
        raise BaselineInvalidError("deterministic arm set drift")
    boundary = manifest.get("inference_boundary", {})
    if boundary != {
        "bundle_kind": BUNDLE_KIND,
        "allowed_split": "validation",
        "expected_groups": 12,
        "branches_per_group": 5,
        "expected_source_samples": 60,
        "expected_predictions": 120,
        "visible_fields": [
            "initial_objectstate",
            "commanded_action_schedule",
            "non_future_metadata",
        ],
        "executed_action_is_feature": False,
        "gt_future_read": False,
        "test_materialized": False,
    }:
        raise BaselineInvalidError("baseline inference boundary drift")
    if manifest.get("horizon") != {
        "unit": "physical_seconds",
        "duration_s": 1.1,
        "scoring_times_s": list(SCORING_TIMES),
    }:
        raise BaselineInvalidError("baseline horizon drift")
    if manifest.get("semantics") != {
        "copy_state": {
            "position": "copy-initial",
            "orientation": "copy-initial-and-canonicalize-sign",
            "linear_velocity": "copy-initial",
            "angular_velocity": "copy-initial",
        },
        "constant_velocity": {
            "position": "position-0-plus-linear-velocity-W-times-absolute-time",
            "orientation": "left-multiply-world-angular-velocity-exponential-and-canonicalize-sign",
            "linear_velocity": "copy-initial",
            "angular_velocity": "copy-initial",
        },
        "quaternion": {
            "layout": "wxyz",
            "frame": "quaternion-WO-with-angular-velocity-W",
            "normalization": "unit-norm-required",
            "canonical_sign": "first-component-with-absolute-value-above-1e-12-is-positive",
        },
    }:
        raise BaselineInvalidError("baseline semantics drift")
    if manifest.get("claim_boundary") != {
        "supported_claim": "deterministic-validation-baselines-are-reproducible-and-auditable",
        "excluded_claims": [
            "test-prediction-produced",
            "trainer-implemented",
            "learned-model-performance",
            "scientific-baseline-comparison",
            "gaussian-dynamics-value",
        ],
    }:
        raise BaselineInvalidError("baseline claim boundary drift")
    if manifest.get("outputs") != {
        "root": "generated/pr02c/baselines/",
        "prediction_contract_kind": "objgauss.dynamics_prediction",
        "prediction_schema_version": "0.3.0",
        "atomic_publish_required": True,
        "git_ignored": True,
    }:
        raise BaselineInvalidError("baseline output contract drift")
    frozen_inputs = manifest.get("frozen_inputs", {})
    if set(frozen_inputs) != {
        "data_boundary_manifest",
        "runtime_lock",
        "common_schema",
        "prediction_schema",
        "formal_data_spec",
        "dynamics_experiment",
    }:
        raise BaselineInvalidError("baseline frozen input set drift")
    if frozen_inputs["dynamics_experiment"] != {
        "runtime_path": "generated/pr02b/evidence/freeze/dynamics-experiment.json",
        "contract_kind": "objgauss.dynamics_experiment",
        "schema_version": "0.3.0",
    }:
        raise BaselineInvalidError("dynamics experiment contract drift")
    for name, entry in manifest.get("frozen_inputs", {}).items():
        path = entry.get("path")
        if path is not None and file_sha256(repo_root / path) != entry.get("sha256"):
            raise BaselineInvalidError(f"frozen baseline input checksum drift: {name}")


def walk_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_FIELDS:
                raise BaselineInvalidError(f"forbidden inference field: {key}")
            walk_forbidden(item)
    elif isinstance(value, list):
        for item in value:
            walk_forbidden(item)


def validate_bundle(
    bundle: dict[str, Any], source_commit: str, *, node: str
) -> list[dict[str, Any]]:
    if set(bundle) != {
        "bundle_version",
        "bundle_kind",
        "source_commit",
        "experiment_id",
        "split",
        "inputs",
        "samples",
        "sample_payload_sha256",
        "isolation",
        "claim_boundary",
    }:
        raise BaselineInvalidError("model input bundle top-level fields drift")
    if bundle.get("bundle_kind") != BUNDLE_KIND or bundle.get("bundle_version") != "0.1.0":
        raise BaselineInvalidError("model input bundle version or kind drift")
    if bundle.get("source_commit") != source_commit:
        raise BaselineInvalidError("model input bundle source commit drift")
    if bundle.get("experiment_id") != "pr02-objectstate-baseline-v0":
        raise BaselineInvalidError("model input bundle experiment identity drift")
    if bundle.get("split") != "validation":
        raise BaselineInvalidError("deterministic baselines only accept validation inputs")
    inputs = bundle.get("inputs")
    expected_input_keys = {
        "data_boundary_manifest_sha256",
        "formal_data_spec_sha256",
        "dynamics_experiment_sha256",
        "source_plan_sha256",
        "source_report_sha256",
        "data_index_sha256",
        "model_input_index_sha256",
        "loader_report_sha256",
        "runtime_lock_sha256",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_keys:
        raise BaselineInvalidError("model input bundle lineage field set drift")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
        for value in inputs.values()
    ):
        raise BaselineInvalidError("model input bundle lineage checksum is invalid")
    isolation = bundle.get("isolation")
    if isolation != {
        "visible_fields": [
            "initial_objectstate",
            "commanded_action_schedule",
            "non_future_metadata",
        ],
        "executed_action_is_feature": False,
        "gt_future_read": False,
        "test_materialized": False,
    }:
        raise BaselineInvalidError("model input isolation boundary drift")
    if bundle.get("claim_boundary") != {
        "supported_claim": "sanitized-validation-model-inputs-are-published",
        "excluded_claims": [
            "prediction-produced",
            "trainer-implemented",
            "model-performance",
            "scientific-baseline-comparison",
        ],
    }:
        raise BaselineInvalidError("model input bundle claim boundary drift")
    samples = bundle.get("samples")
    if not isinstance(samples, list) or len(samples) != 60:
        raise BaselineInvalidError("model input bundle must contain 60 validation branches")
    walk_forbidden(samples)
    if bundle.get("sample_payload_sha256") != canonical_sha256(node, samples):
        raise BaselineInvalidError("model input bundle checksum drift")
    expected_sample_keys = {
        "group_id",
        "branch_id",
        "split",
        "source_episode",
        "initial_objectstate_sha256",
        "commanded_action_sha256",
        "target_object_id",
        "initial_object_states",
        "commanded_action",
        "rollout_times_s",
    }
    seen: set[tuple[str, str]] = set()
    groups: dict[str, set[str]] = {}
    canonical_values: list[Any] = []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != expected_sample_keys:
            raise BaselineInvalidError("model input sample field set drift")
        if sample.get("split") != "validation" or sample.get(
            "rollout_times_s"
        ) != list(SCORING_TIMES):
            raise BaselineInvalidError("sample split or scoring horizon drift")
        key = (sample.get("group_id"), sample.get("branch_id"))
        if key in seen or not all(
            isinstance(value, str) and IDENTIFIER_PATTERN.fullmatch(value) is not None
            for value in key
        ):
            raise BaselineInvalidError("sample identity is invalid or duplicated")
        seen.add(key)
        groups.setdefault(key[0], set()).add(key[1])
        canonical_values.extend(
            (sample["initial_object_states"], sample["commanded_action"])
        )
        source = sample.get("source_episode")
        if not isinstance(source, dict) or set(source) != {
            "uri",
            "media_type",
            "sha256",
            "schema_version",
            "lineage_sha256",
        }:
            raise BaselineInvalidError("source episode reference field set drift")
        if source.get("media_type") != "application/json" or source.get(
            "schema_version"
        ) != "0.2.0":
            raise BaselineInvalidError("source episode reference contract drift")
    hashes = canonical_sha256s(node, canonical_values)
    for index, sample in zip(range(0, len(hashes), 2), samples, strict=True):
        if hashes[index] != sample.get("initial_objectstate_sha256"):
            raise BaselineInvalidError("initial ObjectState checksum drift")
        if hashes[index + 1] != sample.get("commanded_action_sha256"):
            raise BaselineInvalidError("commanded action checksum drift")
    if len(groups) != 12 or any(branches != set(BRANCH_IDS) for branches in groups.values()):
        raise BaselineInvalidError("validation group or branch coverage drift")
    return samples


def prediction_document(
    *,
    repo_root: Path,
    bundle: dict[str, Any],
    sample: dict[str, Any],
    arm: str,
    source_commit: str,
    predictions: list[dict[str, Any]],
    prediction_payload_sha256: str,
) -> dict[str, Any]:
    prediction_key = strict_json_bytes(
        [sample["group_id"], sample["branch_id"], arm]
    )
    prediction_id = f"prediction-{arm}-{sha256_bytes(prediction_key)[:24]}"
    document = {
        "schema_version": "0.3.0",
        "contract_kind": "objgauss.dynamics_prediction",
        "identity": {
            "experiment_id": bundle["experiment_id"],
            "prediction_id": prediction_id,
            "group_id": sample["group_id"],
            "branch_id": sample["branch_id"],
            "split": "validation",
            "model_arm": arm,
            "trial_id": {"availability": "missing", "reason": "not_applicable"},
            "checkpoint_id": {"availability": "missing", "reason": "not_applicable"},
            "training_seed": {"availability": "missing", "reason": "not_applicable"},
        },
        "inputs": {
            "source_episode": sample["source_episode"],
            "initial_objectstate_sha256": sample["initial_objectstate_sha256"],
            "commanded_action_sha256": sample["commanded_action_sha256"],
            "target_object_id": sample["target_object_id"],
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
            "duration_s": SCORING_TIMES[-1],
            "scoring_times_s": list(SCORING_TIMES),
        },
        "predictions": predictions,
        "prediction_payload_sha256": prediction_payload_sha256,
        "publication": {"atomic_publish": True, "immutable_after_publish": True},
        "provenance": {
            "source_commit": source_commit,
            "source_tree_sha256": package_tree_sha256(repo_root),
            "experiment_spec_sha256": bundle["inputs"]["dynamics_experiment_sha256"],
            "runtime_lock_sha256": bundle["inputs"]["runtime_lock_sha256"],
        },
    }
    return document


def produce(
    *,
    repo_root: Path,
    bundle_path: Path,
    manifest_path: Path,
    output_root: Path,
    source_commit: str,
    node: str,
    order: str,
) -> dict[str, Any]:
    require_offline()
    if SOURCE_COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise BaselineInvalidError("source commit must be 40 lowercase hex characters")
    if order not in {"canonical", "reverse"}:
        raise BaselineInvalidError("execution order must be canonical or reverse")
    manifest = read_json(manifest_path)
    validate_manifest(repo_root, manifest)
    manifest_sha256 = file_sha256(manifest_path)
    bundle = read_json(bundle_path)
    samples = validate_bundle(bundle, source_commit, node=node)
    frozen = manifest["frozen_inputs"]
    for bundle_key, manifest_key in (
        ("data_boundary_manifest_sha256", "data_boundary_manifest"),
        ("formal_data_spec_sha256", "formal_data_spec"),
        ("runtime_lock_sha256", "runtime_lock"),
    ):
        if bundle["inputs"][bundle_key] != frozen[manifest_key]["sha256"]:
            raise BaselineInvalidError(f"model input lineage drift: {bundle_key}")
    if output_root.exists() and any(output_root.iterdir()):
        raise BaselineInvalidError("baseline output root must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    sequence = samples if order == "canonical" else list(reversed(samples))
    prediction_jobs = [
        (sample, arm, predict(sample, arm))
        for sample in sequence
        for arm in ARMS
    ]
    payload_hashes = canonical_sha256s(
        node, [predictions for _, _, predictions in prediction_jobs]
    )
    index = []
    for (sample, arm, predictions), payload_sha256 in zip(
        prediction_jobs, payload_hashes, strict=True
    ):
        document = prediction_document(
            repo_root=repo_root,
            bundle=bundle,
            sample=sample,
            arm=arm,
            source_commit=source_commit,
            predictions=predictions,
            prediction_payload_sha256=payload_sha256,
        )
        relative = (
            Path("predictions")
            / sample["group_id"]
            / sample["branch_id"]
            / f"{arm}.json"
        )
        payload = strict_json_bytes(document)
        atomic_write(output_root / relative, payload)
        index.append(
            {
                "group_id": sample["group_id"],
                "branch_id": sample["branch_id"],
                "model_arm": arm,
                "uri": relative.as_posix(),
                "sha256": sha256_bytes(payload),
                "prediction_payload_sha256": document["prediction_payload_sha256"],
            }
        )
    index.sort(key=lambda item: (item["group_id"], item["branch_id"], item["model_arm"]))
    index_payload = {
        "index_version": REPORT_VERSION,
        "index_kind": "objgauss.pr02c-deterministic-baseline-index",
        "source_commit": source_commit,
        "baseline_manifest_sha256": manifest_sha256,
        "model_input_bundle_sha256": file_sha256(bundle_path),
        "predictions": index,
        "semantic_index_sha256": sha256_bytes(
            strict_json_bytes(
                [
                    {
                        "group_id": item["group_id"],
                        "branch_id": item["branch_id"],
                        "model_arm": item["model_arm"],
                        "prediction_payload_sha256": item["prediction_payload_sha256"],
                    }
                    for item in index
                ]
            )
        ),
    }
    atomic_write(output_root / "index.json", strict_json_bytes(index_payload))
    report = {
        "report_version": REPORT_VERSION,
        "report_kind": REPORT_KIND,
        "verdict": {"status": "supported", "reason_code": "deterministic_baselines_published"},
        "source_commit": source_commit,
        "execution_order": order,
        "inputs": {
            "baseline_manifest_sha256": manifest_sha256,
            "model_input_bundle_sha256": file_sha256(bundle_path),
            "data_index_sha256": bundle["inputs"]["data_index_sha256"],
            "runtime_lock_sha256": bundle["inputs"]["runtime_lock_sha256"],
        },
        "counts": {
            "groups": 12,
            "branches": 60,
            "arms": 2,
            "predictions": len(index),
            "failed_predictions": 0,
        },
        "semantic_index_sha256": index_payload["semantic_index_sha256"],
        "isolation": {
            "split": "validation",
            "test_materialized": False,
            "gt_future_read": False,
            "executed_action_is_feature": False,
            "trial_records_created": False,
            "checkpoint_records_created": False,
        },
        "claim_boundary": manifest["claim_boundary"],
    }
    atomic_write(output_root / "report.json", strict_json_bytes(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("learning/baseline-manifest.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--order", choices=("canonical", "reverse"), default="canonical")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    resolve = lambda path: path if path.is_absolute() else repo_root / path
    try:
        report = produce(
            repo_root=repo_root,
            bundle_path=resolve(args.bundle).resolve(),
            manifest_path=resolve(args.manifest).resolve(),
            output_root=resolve(args.output_root).resolve(),
            source_commit=args.source_commit,
            node=args.node,
            order=args.order,
        )
        exit_code = 0
    except (BaselineBlockedError, CanonicalHashError) as error:
        report = {
            "report_version": REPORT_VERSION,
            "report_kind": REPORT_KIND,
            "verdict": {"status": "blocked", "reason_code": "required_input_missing"},
            "message": str(error),
            "claim_boundary": {"supported_claim": "none"},
        }
        exit_code = 3
    except (BaselineInvalidError, OSError, ValueError, KeyError, TypeError) as error:
        report = {
            "report_version": REPORT_VERSION,
            "report_kind": REPORT_KIND,
            "verdict": {"status": "invalid", "reason_code": "baseline_evidence_invalid"},
            "message": str(error),
            "claim_boundary": {"supported_claim": "none"},
        }
        exit_code = 4
    sys.stdout.buffer.write(strict_json_bytes(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
