import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const LEARNING_ROOT = resolve(ROOT, "learning");

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

test("PR-02C runtime manifest freezes the independent lock and PR-02B grid", () => {
  const manifest = readJson(resolve(LEARNING_ROOT, "runtime-manifest.json"));
  assert.equal(manifest.manifest_version, "0.1.0");
  assert.equal(manifest.manifest_kind, "objgauss.pr02c-runtime-manifest");
  for (const input of Object.values(manifest.inputs)) {
    assert.equal(sha256(resolve(ROOT, input.path)), input.sha256, input.path);
  }
  assert.deepEqual(manifest.runtime, {
    python: "3.10.20",
    torch_distribution: "2.13.0",
    torch_runtime: "2.13.0+cu130",
    torch_cuda: "13.0",
    dependency_policy: "pure-pytorch-no-pyg-lightning-hydra",
  });
});

test("learning package has one production dependency and no simulator runtime", () => {
  const project = readFileSync(resolve(LEARNING_ROOT, "pyproject.toml"), "utf8");
  const lock = readFileSync(resolve(LEARNING_ROOT, "uv.lock"), "utf8");
  assert.match(project, /dependencies = \[\s*"torch==2\.13\.0",?\s*\]/);
  assert.match(lock, /name = "torch"\nversion = "2\.13\.0"/);
  for (const forbidden of ["mani-skill", "sapien", "objgauss-sim"]) {
    assert.equal(lock.includes(`name = "${forbidden}"`), false, forbidden);
  }
});

test("C0 gate is executable, clean-head guarded, and runs the independent verifier", () => {
  const gatePath = resolve(ROOT, "scripts/check-pr02c-runtime");
  const gate = readFileSync(gatePath, "utf8");
  assert.ok((statSync(gatePath).mode & 0o111) !== 0, "C0 gate must be executable");
  assert.match(gate, /git status --porcelain=v1 --untracked-files=all/);
  assert.match(gate, /npm run check/);
  assert.match(gate, /verify-pr02c-runtime\.mjs/);
  assert.match(gate, /UV_OFFLINE=1/);
});

test("C0 verifier freezes runtime, source, isolation, GPU reserve, and claims", () => {
  const verifier = readFileSync(resolve(ROOT, "scripts/verify-pr02c-runtime.mjs"), "utf8");
  for (const check of [
    "source-tree",
    "runtime-versions",
    "lock-lineage",
    "grid-lineage",
    "simulator-isolation",
    "gpu-display-reserve",
    "gpu-training-cap",
    "claim-boundary",
  ]) {
    assert.ok(verifier.includes(`check("${check}"`), check);
  }
  assert.match(verifier, /trainer-implemented/);
  assert.match(verifier, /model-performance/);
});
