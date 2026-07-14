"""Fail-closed runtime version and offline-boundary probe for PR-01."""

from __future__ import annotations

import importlib.metadata
import json
import os
import socket
import stat
import sys
from pathlib import Path
from typing import Any


REQUIRED_VERSIONS = {
    "python": "3.10.20",
    "mani_skill": "3.0.1",
    "sapien": "3.0.3",
    "torch_distribution": "2.13.0",
    "torch_runtime": "2.13.0+cu130",
    "torch_cuda": "13.0",
}


def _deny_network(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("network access is forbidden during PR-01 runtime execution")


def install_network_guard() -> None:
    """Block Python-level outbound networking before runtime imports."""

    socket.create_connection = _deny_network
    socket.getaddrinfo = _deny_network
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError(f"network audit event is forbidden: {event}")

    sys.addaudithook(audit)


def observed_versions() -> dict[str, str]:
    """Return distribution and imported runtime versions."""

    import torch

    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "mani_skill": importlib.metadata.version("mani-skill"),
        "sapien": importlib.metadata.version("sapien"),
        "torch_distribution": importlib.metadata.version("torch"),
        "torch_runtime": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def assert_versions() -> dict[str, str]:
    versions = observed_versions()
    if versions != REQUIRED_VERSIONS:
        raise RuntimeError(
            f"runtime version drift: expected {REQUIRED_VERSIONS}, got {versions}"
        )
    return versions


def assert_offline_asset_gate() -> dict[str, Any]:
    """Require an empty read-only asset directory and explicit offline mode."""

    if os.environ.get("OBJGAUSS_RUNTIME_OFFLINE") != "1":
        raise RuntimeError("OBJGAUSS_RUNTIME_OFFLINE must be exactly '1'")
    if os.getenv("MS_SKIP_ASSET_DOWNLOAD_PROMPT") is not None:
        raise RuntimeError("MS_SKIP_ASSET_DOWNLOAD_PROMPT must be unset")
    raw_path = os.getenv("MS_ASSET_DIR")
    if not raw_path:
        raise RuntimeError("MS_ASSET_DIR must point to the no-assets directory")
    asset_dir = Path(raw_path).resolve()
    if not asset_dir.is_dir():
        raise RuntimeError(f"asset gate directory is missing: {asset_dir}")
    entries = sorted(entry.name for entry in asset_dir.iterdir())
    if entries:
        raise RuntimeError(f"asset gate is not empty: {entries}")
    mode = stat.S_IMODE(asset_dir.stat().st_mode)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(f"asset gate must be read-only, got mode {mode:o}")
    return {"path": str(asset_dir), "empty": True, "mode_octal": f"{mode:o}"}


def runtime_probe() -> dict[str, Any]:
    install_network_guard()
    return {
        "runtime_id": "objgauss-pr01-runtime-v0",
        "versions": assert_versions(),
        "asset_gate": assert_offline_asset_gate(),
        "network_policy": "offline",
    }


def main() -> int:
    try:
        report = runtime_probe()
    except Exception as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 3
    print(json.dumps({"status": "supported", **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
