import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

test("C1 manifest freezes train/validation counts and the deterministic formal spec", () => {
  const manifest = JSON.parse(readFileSync(resolve(ROOT, "learning/data-boundary-manifest.json")));
  assert.equal(manifest.manifest_kind, "objgauss.pr02c-data-boundary");
  assert.deepEqual(manifest.materialization.allowed_splits, ["train", "validation"]);
  assert.equal(manifest.materialization.forbidden_split, "test");
  assert.deepEqual(manifest.materialization.group_counts, { train: 48, validation: 12 });
  for (const name of ["pilot_spec", "source_experiment", "simulator_lock"]) {
    const input = manifest.frozen_inputs[name];
    assert.equal(sha256(resolve(ROOT, input.path)), input.sha256, name);
  }
});

test("source and loader CLIs are installed without adding a learning dependency", () => {
  const simProject = readFileSync(resolve(ROOT, "sim/pyproject.toml"), "utf8");
  const learningProject = readFileSync(resolve(ROOT, "learning/pyproject.toml"), "utf8");
  assert.match(simProject, /objgauss-pr02-data = "objgauss_sim\.pr02_data:main"/);
  assert.match(learningProject, /objgauss-pr02c-data = "objgauss_learning\.data:main"/);
  assert.match(learningProject, /dependencies = \[\s*"torch==2\.13\.0",?\s*\]/);
});

test("learning data package imports no simulator and exposes no executed-action feature", () => {
  const source = readFileSync(resolve(ROOT, "learning/src/objgauss_learning/data.py"), "utf8");
  for (const forbidden of ["mani_skill", "sapien", "objgauss_sim"]) {
    assert.equal(source.includes(`import ${forbidden}`), false, forbidden);
    assert.equal(source.includes(`from ${forbidden}`), false, forbidden);
  }
  assert.match(source, /FORBIDDEN_MODEL_FIELDS/);
  assert.match(source, /executed_action_is_model_input/);
  assert.match(source, /future_gt_is_model_input/);
});

test("C1 verifier is independent of producer and loader implementation modules", () => {
  const verifier = readFileSync(resolve(ROOT, "scripts/verify-pr02c-data.mjs"), "utf8");
  assert.doesNotMatch(verifier, /(?:import|from)[^\n]*objgauss_sim/);
  assert.doesNotMatch(verifier, /(?:import|from)[^\n]*objgauss_learning/);
  for (const check of [
    "final-test-not-materialized",
    "episode-attempt-contracts",
    "artifact-checksums",
    "commanded-action-only-input-source",
    "split-identity-isolation",
    "loader-feature-isolation",
  ]) assert.ok(verifier.includes(`check("${check}"`), check);
});

test("C1 clean gate is executable and runs C0, PR-02B, producer, loader, and verifier", () => {
  const path = resolve(ROOT, "scripts/check-pr02c-data");
  const gate = readFileSync(path, "utf8");
  assert.ok((statSync(path).mode & 0o111) !== 0, "C1 gate must be executable");
  for (const required of [
    "check-pr02b-pilot",
    "check-pr02c-runtime",
    "objgauss-pr02-data",
    "objgauss-pr02c-data",
    "verify-pr02c-data.mjs",
    "--splits test",
  ]) assert.ok(gate.includes(required), required);

  const simProvision = gate.indexOf('uv sync --project "$SIM_ROOT"');
  const learningProvision = gate.indexOf('uv sync --project "$LEARNING_ROOT"');
  const pilotGate = gate.indexOf("./scripts/check-pr02b-pilot");
  const runtimeGate = gate.indexOf("./scripts/check-pr02c-runtime");
  assert.ok(simProvision >= 0 && simProvision < pilotGate, "sim venv must precede nested gates");
  assert.ok(
    learningProvision >= 0 && learningProvision < pilotGate,
    "learning venv must precede nested gates",
  );
  assert.ok(pilotGate < runtimeGate, "PR-02B must precede the C0 runtime gate");
});
