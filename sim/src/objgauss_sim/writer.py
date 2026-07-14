"""Immutable, atomic, idempotent branch writer for PR-01 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class PublicationError(RuntimeError):
    """A candidate could not be safely published."""


class ImmutableConflict(PublicationError):
    """An idempotency key already owns different immutable content."""


class ContractValidationError(PublicationError):
    """A staged episode or attempt violates the machine contract."""


class ChecksumValidationError(PublicationError):
    """A staged artifact differs from its computed descriptor."""


@dataclass(frozen=True)
class BranchCandidate:
    episode: dict[str, Any]
    trajectory: dict[str, Any]
    contact_ledger: dict[str, Any]
    attempt_timing: dict[str, float]
    timeout_s: float
    attempt_ordinal: int
    previous_attempt_id: dict[str, Any]
    attempt_provenance: dict[str, str]


def _assert_finite(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string object key at {path}")
            _assert_finite(item, f"{path}.{key}")
        return
    raise TypeError(f"unsupported JSON value {type(value).__name__} at {path}")


def strict_json_bytes(value: Any) -> bytes:
    _assert_finite(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"unsafe {name}: {value!r}")
    return value


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate(path: Path, validator: Path, node: str) -> None:
    completed = subprocess.run(
        [node, str(validator), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ContractValidationError(
            f"validator returned non-JSON output for {path}: {completed.stdout!r}"
        ) from error
    if completed.returncode != 0 or result.get("valid") is not True:
        raise ContractValidationError(
            f"contract validation failed for {path}: "
            f"{result.get('schema_errors', []) + result.get('semantic_errors', [])}"
        )


def _artifact_descriptor(
    *, uri: str, media_type: str, payload: bytes, record_count: int
) -> dict[str, Any]:
    if record_count <= 0:
        raise ValueError("artifact record_count must be positive")
    return {
        "uri": uri,
        "media_type": media_type,
        "sha256": sha256_bytes(payload),
        "byte_length": len(payload),
        "record_count": record_count,
    }


def _attempt_document(
    candidate: BranchCandidate,
    *,
    succeeded: bool,
    episode_uri: str | None,
    episode_sha256: str | None,
    failure_classification: str = "infrastructure",
    failure_reason_code: str = "atomic_write_failure",
    failure_message: str | None = None,
) -> dict[str, Any]:
    identity = candidate.episode["identity"]
    if succeeded:
        outcome = {
            "status": "succeeded",
            "classification": "none",
            "reason_code": "none",
            "message": "Complete branch episode published atomically.",
        }
        publication = {
            "temporary_output_removed": True,
            "final_episode_published": True,
            "episode_artifact": {
                "availability": "present",
                "value": {"uri": episode_uri, "sha256": episode_sha256},
            },
        }
        eligible = False
    else:
        outcome = {
            "status": "failed",
            "classification": failure_classification,
            "reason_code": failure_reason_code,
            "message": failure_message or "Atomic publication failed before episode publication.",
        }
        publication = {
            "temporary_output_removed": True,
            "final_episode_published": False,
            "episode_artifact": {
                "availability": "missing",
                "reason": "not_produced",
            },
        }
        eligible = (
            failure_reason_code
            in {"simulator_crash", "startup_timeout", "atomic_write_failure"}
            and candidate.attempt_ordinal < 2
        )
    return {
        "schema_version": "0.2.0",
        "contract_kind": "objgauss.attempt",
        "identity": {
            "experiment_id": identity["experiment_id"],
            "group_id": identity["group_id"],
            "branch_id": identity["branch_id"],
            "attempt_id": identity["attempt_id"],
            "split": identity["split"],
        },
        "timing": {
            **candidate.attempt_timing,
            "timeout_s": candidate.timeout_s,
        },
        "outcome": outcome,
        "retry": {
            "ordinal": candidate.attempt_ordinal,
            "max_attempts": 2,
            "eligible": eligible,
            "previous_attempt_id": candidate.previous_attempt_id,
            "seed_reused": True,
        },
        "publication": publication,
        "provenance": candidate.attempt_provenance,
    }


def _atomic_failure_attempt(
    root: Path,
    attempt: dict[str, Any],
    validator: Path,
    node: str,
) -> None:
    identity = attempt["identity"]
    directory = (
        root
        / "attempts"
        / _identifier(identity["experiment_id"], "experiment_id")
        / _identifier(identity["group_id"], "group_id")
        / _identifier(identity["branch_id"], "branch_id")
    )
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / f"{_identifier(identity['attempt_id'], 'attempt_id')}.json"
    payload = strict_json_bytes(attempt)
    if final_path.exists():
        if final_path.read_bytes() != payload:
            raise ImmutableConflict(f"attempt already exists with different bytes: {final_path}")
        return
    temporary = directory / f".{final_path.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_new_file(temporary, payload)
        _validate(temporary, validator, node)
        os.rename(temporary, final_path)
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_result(final_directory: Path, semantic_sha256: str) -> dict[str, Any] | None:
    if not final_directory.exists():
        return None
    publication_path = final_directory / "publication.json"
    if not publication_path.is_file():
        raise ImmutableConflict(f"existing branch lacks publication metadata: {final_directory}")
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    if publication.get("semantic_sha256") != semantic_sha256:
        raise ImmutableConflict(
            f"idempotency key already owns different content: {final_directory}"
        )
    return {"status": "noop", **publication}


def publish_branch(
    candidate: BranchCandidate,
    *,
    root: Path,
    validator: Path,
    node: str = "node",
    fail_after_artifacts: bool = False,
) -> dict[str, Any]:
    """Publish one complete branch directory or one immutable failed attempt."""

    identity = candidate.episode["identity"]
    experiment_id = _identifier(identity["experiment_id"], "experiment_id")
    group_id = _identifier(identity["group_id"], "group_id")
    branch_id = _identifier(identity["branch_id"], "branch_id")
    _identifier(identity["attempt_id"], "attempt_id")
    relative_directory = Path("dataset") / experiment_id / group_id / branch_id
    final_directory = root / relative_directory
    parent = final_directory.parent

    trajectory_payload = strict_json_bytes(candidate.trajectory)
    contact_payload = strict_json_bytes(candidate.contact_ledger)
    trajectory_records = candidate.trajectory.get("records")
    contact_records = candidate.contact_ledger.get("records")
    if not isinstance(trajectory_records, list) or not isinstance(contact_records, list):
        raise ValueError("trajectory and contact artifacts require records arrays")

    episode = json.loads(json.dumps(candidate.episode, allow_nan=False))
    episode_uri = (relative_directory / "episode.json").as_posix()
    episode["evidence"]["trajectory"] = _artifact_descriptor(
        uri=(relative_directory / "trajectory.json").as_posix(),
        media_type="application/vnd.objgauss.trajectory+json",
        payload=trajectory_payload,
        record_count=len(trajectory_records),
    )
    episode["evidence"]["contact_ledger"] = _artifact_descriptor(
        uri=(relative_directory / "contact-ledger.json").as_posix(),
        media_type="application/vnd.objgauss.contact-ledger+json",
        payload=contact_payload,
        record_count=len(contact_records),
    )
    episode_payload = strict_json_bytes(episode)
    semantic_document = {
        "episode_sha256": sha256_bytes(episode_payload),
        "trajectory_sha256": sha256_bytes(trajectory_payload),
        "contact_ledger_sha256": sha256_bytes(contact_payload),
    }
    semantic_sha256 = sha256_bytes(strict_json_bytes(semantic_document))
    existing = _existing_result(final_directory, semantic_sha256)
    if existing is not None:
        return existing

    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".tmp-{branch_id}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_new_file(temporary / "trajectory.json", trajectory_payload)
        _write_new_file(temporary / "contact-ledger.json", contact_payload)
        if fail_after_artifacts:
            raise OSError("injected failure after artifact writes")
        _write_new_file(temporary / "episode.json", episode_payload)
        _validate(temporary / "episode.json", validator, node)

        attempt = _attempt_document(
            candidate,
            succeeded=True,
            episode_uri=episode_uri,
            episode_sha256=sha256_bytes(episode_payload),
        )
        attempt_payload = strict_json_bytes(attempt)
        _write_new_file(temporary / "attempt.json", attempt_payload)
        _validate(temporary / "attempt.json", validator, node)

        for descriptor, filename in (
            (episode["evidence"]["trajectory"], "trajectory.json"),
            (episode["evidence"]["contact_ledger"], "contact-ledger.json"),
        ):
            payload = (temporary / filename).read_bytes()
            if len(payload) != descriptor["byte_length"] or sha256_bytes(payload) != descriptor["sha256"]:
                raise ChecksumValidationError(
                    f"local artifact integrity failed: {filename}"
                )

        publication = {
            "idempotency_key": f"{experiment_id}+{group_id}+{branch_id}",
            "semantic_sha256": semantic_sha256,
            "episode_sha256": sha256_bytes(episode_payload),
            "trajectory_sha256": sha256_bytes(trajectory_payload),
            "contact_ledger_sha256": sha256_bytes(contact_payload),
            "attempt_sha256": sha256_bytes(attempt_payload),
        }
        _write_new_file(temporary / "publication.json", strict_json_bytes(publication))
        _fsync_directory(temporary)
        try:
            os.rename(temporary, final_directory)
        except OSError:
            existing = _existing_result(final_directory, semantic_sha256)
            if existing is None:
                raise
            shutil.rmtree(temporary, ignore_errors=True)
            return existing
        _fsync_directory(parent)
        return {"status": "published", **publication}
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(error, ContractValidationError):
            failure_classification = "validation"
            failure_reason_code = "schema_invalid"
        elif isinstance(error, ChecksumValidationError):
            failure_classification = "validation"
            failure_reason_code = "checksum_mismatch"
        else:
            failure_classification = "infrastructure"
            failure_reason_code = "atomic_write_failure"
        failure = _attempt_document(
            candidate,
            succeeded=False,
            episode_uri=None,
            episode_sha256=None,
            failure_classification=failure_classification,
            failure_reason_code=failure_reason_code,
            failure_message=f"Atomic publication failed: {type(error).__name__}.",
        )
        _atomic_failure_attempt(root, failure, validator, node)
        if isinstance(error, PublicationError):
            raise
        raise PublicationError(str(error)) from error
