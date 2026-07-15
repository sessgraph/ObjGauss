"""Independent semantic inspection for PR-02C PyTorch state-dict checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

from .runtime import strict_json_bytes


class CheckpointInvalidError(RuntimeError):
    """A checkpoint is unsafe, incomplete, or not a tensor-only state dict."""


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return bytes(value.view(torch.uint8).flatten().tolist())


def inspect_state_dict(state_dict: Any) -> dict[str, Any]:
    if not isinstance(state_dict, dict) or not state_dict:
        raise CheckpointInvalidError("checkpoint must contain a non-empty state dict")
    entries = []
    semantic = hashlib.sha256()
    parameter_count = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_.]+", name) is None:
            raise CheckpointInvalidError("checkpoint parameter name is invalid")
        if not isinstance(tensor, torch.Tensor):
            raise CheckpointInvalidError("checkpoint state dict must contain tensors only")
        if not torch.isfinite(tensor).all():
            raise CheckpointInvalidError(f"checkpoint tensor is non-finite: {name}")
        payload = tensor_bytes(tensor)
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
        parameter_count += tensor.numel()
        entries.append(entry)
    return {
        "semantic_sha256": semantic.hexdigest(),
        "parameter_count": parameter_count,
        "parameter_structure_sha256": hashlib.sha256(
            strict_json_bytes(
                [
                    {"name": item["name"], "dtype": item["dtype"], "shape": item["shape"]}
                    for item in entries
                ]
            )
        ).hexdigest(),
        "entries": entries,
    }


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise CheckpointInvalidError(f"cannot load checkpoint: {error}") from error
    return inspect_state_dict(state_dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = {"verdict": "supported", **inspect_checkpoint(args.checkpoint)}
        exit_code = 0
    except (CheckpointInvalidError, KeyError, TypeError) as error:
        report = {"verdict": "invalid", "message": str(error)}
        exit_code = 4
    sys.stdout.buffer.write(strict_json_bytes(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
