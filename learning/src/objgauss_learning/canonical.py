"""Node-contract-compatible canonical JSON hashes for cross-language artifacts."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from .runtime import strict_json_bytes


CANONICAL_HASH_PROGRAM = r"""
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const values = JSON.parse(Buffer.concat(chunks).toString("utf8"));
if (!Array.isArray(values)) throw new Error("canonical hash input must be an array");
function canonicalize(item) {
  if (Array.isArray(item)) return item.map(canonicalize);
  if (item !== null && typeof item === "object") {
    return Object.fromEntries(
      Object.keys(item).sort().map((key) => [key, canonicalize(item[key])])
    );
  }
  return item;
}
const { createHash } = await import("node:crypto");
const hashes = values.map((value) => {
  const canonical = `${JSON.stringify(canonicalize(value), null, 2)}\n`;
  return createHash("sha256").update(canonical).digest("hex");
});
process.stdout.write(JSON.stringify(hashes));
"""


class CanonicalHashError(RuntimeError):
    """The frozen Node canonicalizer is unavailable or returned invalid output."""


def canonical_sha256s(node: str, values: list[Any]) -> list[str]:
    try:
        process = subprocess.run(
            [node, "--input-type=module", "-e", CANONICAL_HASH_PROGRAM],
            input=strict_json_bytes(values),
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise CanonicalHashError(
            f"Node canonical hash runtime is unavailable: {node}"
        ) from error
    if process.returncode != 0:
        raise CanonicalHashError(
            "Node canonical hash failed: "
            + process.stderr.decode("utf-8", errors="replace").strip()
        )
    try:
        hashes = json.loads(process.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalHashError("Node canonical hash returned invalid JSON") from error
    if (
        not isinstance(hashes, list)
        or len(hashes) != len(values)
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"[a-f0-9]{64}", item) is None
            for item in hashes
        )
    ):
        raise CanonicalHashError("Node canonical hash returned an invalid digest set")
    return hashes


def canonical_sha256(node: str, value: Any) -> str:
    return canonical_sha256s(node, [value])[0]
