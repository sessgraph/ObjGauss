"""Independent PR-02C C6 selector for the frozen 24-task HPO ledger."""

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


HEX64 = re.compile(r"^[a-f0-9]{64}$")
COMMIT = re.compile(r"^[a-f0-9]{40}$")
ARMS = ("action_free", "action_conditioned")
VERSION = "0.1.0"
INDEX_KIND = "objgauss.pr02c-c6-task-index"


class SelectionBlockedError(RuntimeError):
    """No eligible config remains for at least one learned arm."""


class SelectionRejectedError(RuntimeError):
    """Complete valid evidence fails the frozen reproducibility/fairness endpoint."""


class SelectionInvalidError(RuntimeError):
    """The manifest, task ledger, lineage, or artifact set is invalid."""


def strict_json_bytes(value: Any) -> bytes:
    return f"{json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}\n".encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise SelectionInvalidError(f"referenced artifact is missing: {path}") from error
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(path.read_bytes(), parse_constant=reject_constant)
    except FileNotFoundError as error:
        raise SelectionInvalidError(f"selector input is missing: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SelectionInvalidError(f"selector input is invalid: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionInvalidError(f"selector input must be an object: {path}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise SelectionInvalidError(f"selector refuses to overwrite output: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise SelectionInvalidError(f"{name} must be a lowercase SHA-256")
    return value


def _expected(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if manifest.get("manifest_kind") != "objgauss.pr02c-c6-hpo-selection":
        raise SelectionInvalidError("HPO manifest kind drift")
    matrix = manifest.get("matrix")
    if not isinstance(matrix, dict) or matrix.get("task_count") != 24 or matrix.get("pair_count") != 12:
        raise SelectionInvalidError("HPO matrix cardinality drift")
    configurations = matrix.get("configurations")
    pairs = matrix.get("fairness_pairs")
    if not isinstance(configurations, list) or len(configurations) != 4:
        raise SelectionInvalidError("HPO configuration set drift")
    if not isinstance(pairs, list) or len(pairs) != 12:
        raise SelectionInvalidError("HPO fairness pair set drift")
    config_by_id: dict[str, dict[str, Any]] = {}
    for config in configurations:
        if not isinstance(config, dict):
            raise SelectionInvalidError("HPO configuration must be an object")
        config_id = config.get("config_id")
        if not isinstance(config_id, str) or config_id in config_by_id:
            raise SelectionInvalidError("HPO config IDs must be unique strings")
        _require_sha(config.get("config_sha256"), f"config {config_id} digest")
        config_by_id[config_id] = config
    expected: dict[str, dict[str, Any]] = {}
    pair_by_id: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise SelectionInvalidError("HPO fairness pair must be an object")
        pair_id = pair.get("pair_id")
        config_id = pair.get("config_id")
        seed = pair.get("training_seed")
        task_ids = pair.get("task_ids")
        if (
            not isinstance(pair_id, str)
            or pair_id in pair_by_id
            or config_id not in config_by_id
            or seed not in matrix.get("training_seeds", [])
            or not isinstance(task_ids, dict)
            or set(task_ids) != set(ARMS)
        ):
            raise SelectionInvalidError("HPO fairness pair identity drift")
        pair_by_id[pair_id] = pair
        for arm in ARMS:
            task_id = task_ids[arm]
            if not isinstance(task_id, str) or task_id in expected:
                raise SelectionInvalidError("HPO task IDs must be unique strings")
            expected[task_id] = {
                "task_id": task_id,
                "pair_id": pair_id,
                "model_arm": arm,
                "config_id": config_id,
                "config_sha256": config_by_id[config_id]["config_sha256"],
                "training_seed": seed,
            }
    if len(expected) != 24:
        raise SelectionInvalidError("HPO manifest must expand to exactly 24 tasks")
    return expected, pair_by_id


def _artifact_checks(task: dict[str, Any], artifact_root: Path) -> None:
    artifacts = task.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SelectionInvalidError(f"task has no artifact ledger: {task.get('task_id')}")
    uris: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"uri", "sha256"}:
            raise SelectionInvalidError("task artifact descriptor drift")
        uri = artifact["uri"]
        if not isinstance(uri, str) or uri.startswith("/") or ".." in Path(uri).parts or uri in uris:
            raise SelectionInvalidError("task artifact URI is unsafe or duplicated")
        uris.add(uri)
        expected = _require_sha(artifact["sha256"], f"artifact {uri}")
        if file_sha256(artifact_root / uri) != expected:
            raise SelectionInvalidError(f"task artifact checksum mismatch: {uri}")


def _task_score(task: dict[str, Any], validation_group_ids: tuple[str, ...]) -> float:
    values = task.get("validation_group_errors")
    if not isinstance(values, list) or len(values) != 12:
        raise SelectionInvalidError("task must report exactly 12 validation group errors")
    observed: dict[str, float] = {}
    for item in values:
        if not isinstance(item, dict) or set(item) != {"group_id", "primary_error"}:
            raise SelectionInvalidError("validation group score record drift")
        group_id = item["group_id"]
        score = item["primary_error"]
        if (
            group_id in observed
            or group_id not in validation_group_ids
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) < 0
        ):
            raise SelectionInvalidError("validation group score is invalid")
        observed[group_id] = float(score)
    if set(observed) != set(validation_group_ids):
        raise SelectionInvalidError("validation group score coverage drift")
    recomputed = math.fsum(observed[group_id] for group_id in validation_group_ids) / 12.0
    claimed = task.get("validation_primary_error")
    if (
        isinstance(claimed, bool)
        or not isinstance(claimed, (int, float))
        or not math.isfinite(float(claimed))
        or float(claimed) != recomputed
    ):
        raise SelectionInvalidError("task validation primary error was not group-first recomputed")
    return recomputed


def _fairness(task: dict[str, Any]) -> dict[str, Any]:
    fairness = task.get("fairness")
    required = {
        "initialization_seed",
        "initialization_algorithm",
        "common_parameter_names_sha256",
        "common_parameter_subtree_sha256",
        "arm_specific_parameter_names_sha256",
        "arm_specific_parameter_subtree_sha256",
        "data_order_sha256",
        "batch_group_sequence_sha256",
        "optimizer_updates",
        "epochs",
        "training_budget_sha256",
        "checkpoint_policy",
    }
    if not isinstance(fairness, dict) or set(fairness) != required:
        raise SelectionInvalidError("task fairness ledger drift")
    for name in (
        "common_parameter_names_sha256",
        "common_parameter_subtree_sha256",
        "arm_specific_parameter_names_sha256",
        "arm_specific_parameter_subtree_sha256",
        "data_order_sha256",
        "batch_group_sequence_sha256",
        "training_budget_sha256",
    ):
        _require_sha(fairness[name], f"fairness {name}")
    if (
        fairness["initialization_seed"] != task.get("training_seed")
        or fairness["initialization_algorithm"] != "torch-manual-seed-deterministic-v1"
        or fairness["checkpoint_policy"] != "minimum-validation-primary-error-per-seed"
        or not isinstance(fairness["optimizer_updates"], int)
        or fairness["optimizer_updates"] < 1
        or not isinstance(fairness["epochs"], int)
        or fairness["epochs"] < 1
    ):
        raise SelectionInvalidError("task fairness policy drift")
    return fairness


def select_records(
    *,
    manifest: dict[str, Any],
    index: dict[str, Any],
    artifact_root: Path,
    input_order: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if input_order not in {"canonical", "reverse"}:
        raise SelectionInvalidError("selector input order must be canonical or reverse")
    expected, pair_by_id = _expected(manifest)
    if index.get("index_version") != VERSION or index.get("index_kind") != INDEX_KIND:
        raise SelectionInvalidError("HPO task index kind or version drift")
    if index.get("splits") != ["train", "validation"] or index.get("test_materialized") is not False:
        raise SelectionInvalidError("selector detected final test access")
    runner_commit = index.get("runner_commit")
    if not isinstance(runner_commit, str) or COMMIT.fullmatch(runner_commit) is None:
        raise SelectionInvalidError("runner commit lineage is invalid")
    if runner_commit != index.get("workflow_commit") or runner_commit != index.get("hpo_data_build_commit"):
        raise SelectionInvalidError("runner/workflow/data build commit lineage differs")
    if index.get("trainer_contract_commit") != manifest["lineage"]["trainer_contract_commit"]:
        raise SelectionInvalidError("trainer contract commit lineage drift")
    _require_sha(index.get("hpo_manifest_sha256"), "HPO manifest digest")
    hpo_data_index_sha256 = _require_sha(index.get("hpo_data_index_sha256"), "HPO data index digest")
    validation_group_ids = index.get("validation_group_ids")
    if (
        not isinstance(validation_group_ids, list)
        or len(validation_group_ids) != 12
        or validation_group_ids != sorted(validation_group_ids)
        or len(set(validation_group_ids)) != 12
    ):
        raise SelectionInvalidError("validation group ID set drift")
    validation_ids = tuple(validation_group_ids)
    tasks = index.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 24:
        raise SelectionInvalidError("task ledger must contain exactly 24 records")
    sequence = tasks if input_order == "canonical" else list(reversed(tasks))
    observed: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    fairness: dict[str, dict[str, Any]] = {}
    for task in sequence:
        if not isinstance(task, dict):
            raise SelectionInvalidError("task ledger record must be an object")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or task_id in observed or task_id not in expected:
            raise SelectionInvalidError("task ledger contains duplicate or unregistered task ID")
        identity = expected[task_id]
        for name, value in identity.items():
            if task.get(name) != value:
                raise SelectionInvalidError(f"task identity drift for {task_id}: {name}")
        if task.get("hpo_data_index_sha256") != hpo_data_index_sha256:
            raise SelectionInvalidError("tasks did not share the sealed HPO data index")
        attempt_ids = task.get("attempt_ids")
        if (
            not isinstance(attempt_ids, list)
            or not 1 <= len(attempt_ids) <= 2
            or len(set(attempt_ids)) != len(attempt_ids)
            or any(
                attempt_id != f"attempt-{task_id}-a{ordinal:02d}"
                for ordinal, attempt_id in enumerate(attempt_ids, 1)
            )
        ):
            raise SelectionInvalidError("task attempt lineage drift")
        status = task.get("status")
        if status not in {"completed", "failed"}:
            raise SelectionInvalidError("task status is invalid")
        if status == "completed":
            scores[task_id] = _task_score(task, validation_ids)
            fairness[task_id] = _fairness(task)
            _artifact_checks(task, artifact_root)
        observed[task_id] = task
    if set(observed) != set(expected):
        raise SelectionInvalidError("task ledger does not exactly match the frozen matrix")

    pair_records = []
    for pair_id in sorted(pair_by_id):
        pair = pair_by_id[pair_id]
        members = {arm: observed[pair["task_ids"][arm]] for arm in ARMS}
        complete = all(member["status"] == "completed" for member in members.values())
        if complete:
            first = fairness[members[ARMS[0]]["task_id"]]
            second = fairness[members[ARMS[1]]["task_id"]]
            shared = (
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
            if any(first[name] != second[name] for name in shared):
                raise SelectionRejectedError(f"fairness pair shared ledger differs: {pair_id}")
        pair_records.append(
            {
                "pair_id": pair_id,
                "config_id": pair["config_id"],
                "training_seed": pair["training_seed"],
                "complete": complete,
                "task_ids": pair["task_ids"],
            }
        )

    selections: dict[str, dict[str, Any]] = {}
    config_scores: list[dict[str, Any]] = []
    seeds = tuple(manifest["matrix"]["training_seeds"])
    configs = tuple(item["config_id"] for item in manifest["matrix"]["configurations"])
    config_by_id = {item["config_id"]: item for item in manifest["matrix"]["configurations"]}
    for arm in ARMS:
        eligible: list[tuple[float, str]] = []
        for config_id in configs:
            seed_tasks = [
                task
                for task in observed.values()
                if task["model_arm"] == arm and task["config_id"] == config_id
            ]
            seed_tasks.sort(key=lambda item: item["training_seed"])
            complete = (
                [item["training_seed"] for item in seed_tasks] == list(seeds)
                and all(item["status"] == "completed" for item in seed_tasks)
            )
            score = None
            if complete:
                score = math.fsum(scores[item["task_id"]] for item in seed_tasks) / len(seeds)
                eligible.append((score, config_id))
            config_scores.append(
                {
                    "model_arm": arm,
                    "config_id": config_id,
                    "eligible": complete,
                    "selection_score": score,
                    "task_ids": [item["task_id"] for item in seed_tasks],
                }
            )
        if not eligible:
            raise SelectionBlockedError(f"no complete eligible config remains for {arm}")
        score, selected_id = min(eligible, key=lambda item: (item[0], item[1]))
        selections[arm] = {
            "config_id": selected_id,
            "config_sha256": config_by_id[selected_id]["config_sha256"],
            "selection_score": score,
        }

    semantic = sha256_bytes(strict_json_bytes(selections))
    selected = {
        "selection_version": VERSION,
        "selection_kind": "objgauss.pr02c-c6-selected-configs",
        "verdict": "supported",
        "runner_commit": runner_commit,
        "trainer_contract_commit": manifest["lineage"]["trainer_contract_commit"],
        "hpo_data_index_sha256": hpo_data_index_sha256,
        "mapping": selections,
        "selection_semantic_sha256": semantic,
        "test_visible": False,
        "formal_checkpoint_frozen": False,
    }
    report = {
        "report_version": VERSION,
        "report_kind": "objgauss.pr02c-c6-selection-report",
        "verdict": {"status": "supported", "reason_code": "unique-arm-config-mapping-frozen"},
        "counts": {"tasks": 24, "fairness_pairs": 12, "configs": 4, "seeds": 3},
        "config_scores": sorted(config_scores, key=lambda item: (item["model_arm"], item["config_id"])),
        "fairness_pairs": pair_records,
        "selected_configs_sha256": sha256_bytes(strict_json_bytes(selected)),
        "selection_semantic_sha256": semantic,
        "selection_policy": manifest["selector"],
        "claim_boundary": manifest["claim_boundary"],
    }
    return selected, report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=Path("learning/hpo-manifest.json"))
    value.add_argument("--task-index", type=Path, required=True)
    value.add_argument("--artifact-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--order", choices=("canonical", "reverse"), default="canonical")
    value.add_argument("--splits", nargs="+", default=["validation"])
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.splits != ["validation"]:
            raise SelectionInvalidError("selector accepts validation only; final test is forbidden")
        manifest = read_json(args.manifest.resolve())
        index = read_json(args.task_index.resolve())
        if index.get("hpo_manifest_sha256") != file_sha256(args.manifest.resolve()):
            raise SelectionInvalidError("task index HPO manifest checksum drift")
        output_root = args.output_root.resolve()
        if output_root.exists() and any(output_root.iterdir()):
            raise SelectionInvalidError("selector output root must be empty")
        selected, report = select_records(
            manifest=manifest,
            index=index,
            artifact_root=args.artifact_root.resolve(),
            input_order=args.order,
        )
        atomic_write(output_root / "selected-configs.json", strict_json_bytes(selected))
        atomic_write(output_root / "selection-report.json", strict_json_bytes(report))
        sys.stdout.buffer.write(strict_json_bytes(report))
        return 0
    except SelectionRejectedError as error:
        report = {"verdict": "rejected", "reason_code": "fairness-or-reproducibility-failed", "message": str(error)}
        exit_code = 2
    except SelectionBlockedError as error:
        report = {"verdict": "blocked", "reason_code": "no-eligible-config", "message": str(error)}
        exit_code = 3
    except (SelectionInvalidError, OSError, KeyError, TypeError, ValueError) as error:
        report = {"verdict": "invalid", "reason_code": "selection-protocol-invalid", "message": str(error)}
        exit_code = 4
    sys.stdout.buffer.write(strict_json_bytes(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
