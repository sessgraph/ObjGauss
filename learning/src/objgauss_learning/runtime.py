"""Fail-closed PR-02C runtime, isolation, lineage, and GPU reserve probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, NoReturn


REPORT_VERSION = "0.1.0"
REPORT_KIND = "objgauss.pr02c-runtime-probe"
MANIFEST_KIND = "objgauss.pr02c-runtime-manifest"
SOURCE_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40,64}$")
DISPLAY_RESERVE_BYTES = 1024**3
TRAINING_CAP_BYTES = 12 * 1024**3
PROBE_ALLOCATION_BYTES = 16 * 1024**2
REQUIRED_VERSIONS = {
    "python": "3.10.20",
    "torch_distribution": "2.13.0",
    "torch_runtime": "2.13.0+cu130",
    "torch_cuda": "13.0",
}
FORBIDDEN_DISTRIBUTIONS = ("mani-skill", "sapien", "objgauss-sim")
FORBIDDEN_MODULES = ("mani_skill", "sapien", "objgauss_sim")


class RuntimeBlockedError(RuntimeError):
    """The frozen runtime cannot execute with the available local resources."""


class RuntimeInvalidError(RuntimeError):
    """The runtime inputs, isolation boundary, or lineage are invalid."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def strict_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeInvalidError(f"cannot read JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeInvalidError(f"JSON input must be an object: {path}")
    return value


def require_relative_path(root: Path, raw: str, *, prefix: str | None = None) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        raise RuntimeInvalidError(f"path must be repository-relative: {raw}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeInvalidError(f"path escapes repository root: {raw}")
    relative = resolved.relative_to(resolved_root).as_posix()
    if prefix is not None and not relative.startswith(prefix.rstrip("/") + "/"):
        raise RuntimeInvalidError(f"path must be below {prefix}/: {raw}")
    return resolved


def validate_source_commit(value: str) -> str:
    if SOURCE_COMMIT_PATTERN.fullmatch(value) is None:
        raise RuntimeInvalidError(
            "source commit must be 40-64 lowercase hexadecimal characters"
        )
    return value


def assert_clean_git_head(repo_root: Path, source_commit: str) -> dict[str, Any]:
    source_commit = validate_source_commit(source_commit)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeInvalidError(f"cannot inspect git worktree: {status.stderr.strip()}")
    if status.stdout:
        raise RuntimeInvalidError("PR-02C runtime requires a clean git worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        raise RuntimeInvalidError(f"cannot resolve git HEAD: {head.stderr.strip()}")
    observed = head.stdout.strip()
    if observed != source_commit:
        raise RuntimeInvalidError(
            f"source commit differs from HEAD: expected {observed}, got {source_commit}"
        )
    return {"worktree_clean": True, "head_matches_source_commit": True}


def assert_offline_mode() -> dict[str, Any]:
    if os.environ.get("OBJGAUSS_LEARNING_OFFLINE") != "1":
        raise RuntimeInvalidError("OBJGAUSS_LEARNING_OFFLINE must be exactly '1'")
    return {"explicit_offline_mode": True}


def package_tree_sha256(repo_root: Path) -> str:
    learning_root = repo_root / "learning"
    package_root = learning_root / "src" / "objgauss_learning"
    sources = [
        learning_root / "pyproject.toml",
        learning_root / "runtime-manifest.json",
        *sorted(package_root.glob("*.py")),
    ]
    if any(not path.is_file() for path in sources):
        missing = [str(path) for path in sources if not path.is_file()]
        raise RuntimeInvalidError(f"runtime source set is incomplete: {missing}")
    entries = {
        path.relative_to(repo_root).as_posix(): file_sha256(path) for path in sources
    }
    return sha256_bytes(strict_json_bytes(entries))


def validate_manifest(
    *, repo_root: Path, manifest_path: Path, lock_path: Path, grid_path: Path
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    expected_keys = {
        "manifest_version",
        "manifest_kind",
        "runtime",
        "inputs",
        "isolation",
        "resources",
        "outputs",
        "claim_boundary",
    }
    if set(manifest) != expected_keys:
        raise RuntimeInvalidError("runtime manifest top-level fields drift")
    if manifest.get("manifest_version") != REPORT_VERSION:
        raise RuntimeInvalidError("runtime manifest version drift")
    if manifest.get("manifest_kind") != MANIFEST_KIND:
        raise RuntimeInvalidError("runtime manifest kind drift")
    if manifest.get("runtime") != {
        "python": "3.10.20",
        "torch_distribution": "2.13.0",
        "torch_runtime": "2.13.0+cu130",
        "torch_cuda": "13.0",
        "dependency_policy": "pure-pytorch-no-pyg-lightning-hydra",
    }:
        raise RuntimeInvalidError("runtime manifest versions or dependency policy drift")
    isolation = manifest.get("isolation")
    if isolation != {
        "forbidden_distributions": list(FORBIDDEN_DISTRIBUTIONS),
        "forbidden_modules": list(FORBIDDEN_MODULES),
        "simulator_import_allowed": False,
        "network_policy": "offline-during-runtime",
    }:
        raise RuntimeInvalidError("runtime manifest isolation policy drift")
    resources = manifest.get("resources")
    if resources != {
        "display_vram_reserve_bytes_min": DISPLAY_RESERVE_BYTES,
        "training_peak_vram_bytes_max": TRAINING_CAP_BYTES,
        "probe_allocation_bytes": PROBE_ALLOCATION_BYTES,
    }:
        raise RuntimeInvalidError("runtime manifest resource policy drift")
    if manifest.get("outputs") != {
        "root": "generated/pr02c/",
        "git_ignored": True,
        "atomic_publish_required": True,
    }:
        raise RuntimeInvalidError("runtime manifest output policy drift")
    if manifest.get("claim_boundary") != {
        "supported_claim": "clean-isolated-pytorch-runtime-is-available",
        "excluded_claims": [
            "formal-cohort-generated",
            "trainer-implemented",
            "model-performance",
            "scientific-baseline-comparison",
            "gaussian-dynamics-value",
        ],
    }:
        raise RuntimeInvalidError("runtime manifest claim boundary drift")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeInvalidError("runtime manifest inputs must be an object")
    expected_paths = {
        "runtime_lock": lock_path.relative_to(repo_root).as_posix(),
        "hyperparameter_grid": grid_path.relative_to(repo_root).as_posix(),
    }
    for name, path in (("runtime_lock", lock_path), ("hyperparameter_grid", grid_path)):
        entry = inputs.get(name)
        if not isinstance(entry, dict):
            raise RuntimeInvalidError(f"runtime manifest input is missing: {name}")
        if entry.get("path") != expected_paths[name]:
            raise RuntimeInvalidError(f"runtime manifest path drift: {name}")
        if entry.get("sha256") != file_sha256(path):
            raise RuntimeInvalidError(f"runtime manifest checksum drift: {name}")
    return manifest


def _deny_network(*args: Any, **kwargs: Any) -> NoReturn:
    del args, kwargs
    raise RuntimeInvalidError("network access is forbidden during PR-02C runtime")


def install_network_guard() -> None:
    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network

    def audit(event: str, args: tuple[Any, ...]) -> None:
        del args
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeInvalidError(f"network audit event is forbidden: {event}")

    sys.addaudithook(audit)


def observed_versions() -> dict[str, str]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Failed to initialize NumPy:.*", category=UserWarning
        )
        import torch

    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_runtime": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def assert_versions() -> dict[str, str]:
    versions = observed_versions()
    if versions != REQUIRED_VERSIONS:
        raise RuntimeBlockedError(
            f"runtime version drift: expected {REQUIRED_VERSIONS}, got {versions}"
        )
    return versions


def assert_isolation() -> dict[str, Any]:
    present_distributions: list[str] = []
    for name in FORBIDDEN_DISTRIBUTIONS:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        present_distributions.append(name)

    present_modules = [
        name for name in FORBIDDEN_MODULES if importlib.util.find_spec(name) is not None
    ]
    loaded_modules = [
        name
        for name in FORBIDDEN_MODULES
        if name in sys.modules
        or any(module.startswith(name + ".") for module in sys.modules)
    ]
    if present_distributions or present_modules or loaded_modules:
        raise RuntimeInvalidError(
            "learning runtime isolation violated: "
            f"distributions={present_distributions}, modules={present_modules}, "
            f"loaded={loaded_modules}"
        )
    return {
        "forbidden_distributions_absent": True,
        "forbidden_modules_absent": True,
        "forbidden_modules_loaded": False,
        "network_policy": "offline-during-runtime",
    }


def gpu_reserve_probe() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeBlockedError("CUDA is unavailable")
    free_before, total = torch.cuda.mem_get_info()
    if free_before <= DISPLAY_RESERVE_BYTES + PROBE_ALLOCATION_BYTES:
        raise RuntimeBlockedError("insufficient free VRAM for probe plus display reserve")
    allocation = torch.empty(
        PROBE_ALLOCATION_BYTES // 4, dtype=torch.float32, device="cuda"
    )
    allocation.fill_(1.0)
    torch.cuda.synchronize()
    free_during, _ = torch.cuda.mem_get_info()
    observed = float(allocation[0].item())
    del allocation
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    minimum_free = min(free_during, free_after)
    if not math.isclose(observed, 1.0):
        raise RuntimeInvalidError("CUDA probe allocation produced the wrong value")
    if minimum_free < DISPLAY_RESERVE_BYTES:
        raise RuntimeBlockedError("CUDA probe violated the 1 GiB display reserve")
    training_cap = min(
        TRAINING_CAP_BYTES, max(0, free_after - DISPLAY_RESERVE_BYTES)
    )
    if training_cap <= 0:
        raise RuntimeBlockedError("no VRAM remains for training after display reserve")
    return {
        "status": "supported",
        "reason_code": "display_reserve_supported",
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_bytes": int(total),
        "free_before_bytes": int(free_before),
        "free_during_probe_bytes": int(free_during),
        "free_after_bytes": int(free_after),
        "minimum_free_bytes": int(minimum_free),
        "display_reserve_bytes": DISPLAY_RESERVE_BYTES,
        "probe_allocation_bytes": PROBE_ALLOCATION_BYTES,
        "training_allocation_cap_bytes": int(training_cap),
    }


def build_report(
    *,
    repo_root: Path,
    source_commit: str,
    manifest_path: Path,
    lock_path: Path,
    grid_path: Path,
) -> dict[str, Any]:
    validate_manifest(
        repo_root=repo_root,
        manifest_path=manifest_path,
        lock_path=lock_path,
        grid_path=grid_path,
    )
    source_commit = validate_source_commit(source_commit)
    source = assert_clean_git_head(repo_root, source_commit)
    offline = assert_offline_mode()
    install_network_guard()
    versions = assert_versions()
    isolation = assert_isolation()
    gpu = gpu_reserve_probe()
    return {
        "report_version": REPORT_VERSION,
        "report_kind": REPORT_KIND,
        "verdict": {"status": "supported", "reason_code": "all_c0_gates_passed"},
        "producer": {
            "name": "objgauss.pr02c-runtime",
            "version": "0.1.0",
            "source_commit": source_commit,
            "source_tree_sha256": package_tree_sha256(repo_root),
        },
        "inputs": {
            "runtime_manifest_sha256": file_sha256(manifest_path),
            "runtime_lock_sha256": file_sha256(lock_path),
            "hyperparameter_grid_sha256": file_sha256(grid_path),
        },
        "runtime": versions,
        "source": source,
        "isolation": {**isolation, **offline},
        "gpu_probe": gpu,
        "claim_boundary": {
            "supported_claim": "clean-isolated-pytorch-runtime-is-available",
            "excluded_claims": [
                "formal-cohort-generated",
                "trainer-implemented",
                "model-performance",
                "scientific-baseline-comparison",
                "gaussian-dynamics-value",
            ],
        },
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def failure_report(*, status: str, reason_code: str, message: str) -> dict[str, Any]:
    return {
        "report_version": REPORT_VERSION,
        "report_kind": REPORT_KIND,
        "verdict": {"status": status, "reason_code": reason_code},
        "message": message,
        "claim_boundary": {
            "supported_claim": "none",
            "excluded_claims": ["runtime-available", "trainer-implemented"],
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", default=".")
    value.add_argument("--source-commit", required=True)
    value.add_argument("--manifest", default="learning/runtime-manifest.json")
    value.add_argument("--lock", default="learning/uv.lock")
    value.add_argument(
        "--grid", default="contracts/fixtures/pr02b/hyperparameter-grid.json"
    )
    value.add_argument("--output", default="generated/pr02c/runtime/report.json")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output: Path | None = None
    try:
        output = require_relative_path(
            repo_root, args.output, prefix="generated/pr02c"
        )
        manifest_path = require_relative_path(repo_root, args.manifest)
        lock_path = require_relative_path(repo_root, args.lock)
        grid_path = require_relative_path(repo_root, args.grid)
        report = build_report(
            repo_root=repo_root,
            source_commit=args.source_commit,
            manifest_path=manifest_path,
            lock_path=lock_path,
            grid_path=grid_path,
        )
        exit_code = 0
    except RuntimeBlockedError as error:
        report = failure_report(
            status="blocked", reason_code="runtime_unavailable", message=str(error)
        )
        exit_code = 3
    except (RuntimeInvalidError, OSError, ValueError) as error:
        report = failure_report(
            status="invalid", reason_code="runtime_or_lineage_invalid", message=str(error)
        )
        exit_code = 4
    payload = strict_json_bytes(report)
    if output is not None:
        atomic_write(output, payload)
    sys.stdout.buffer.write(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
