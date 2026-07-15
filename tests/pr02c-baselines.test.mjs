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

test("C2 manifest freezes the two deterministic validation baselines", () => {
  const manifest = JSON.parse(readFileSync(resolve(ROOT, "learning/baseline-manifest.json")));
  assert.equal(manifest.manifest_kind, "objgauss.pr02c-deterministic-baselines");
  assert.deepEqual(manifest.arms, ["copy_state", "constant_velocity"]);
  assert.deepEqual(manifest.horizon.scoring_times_s, [0.1, 0.2, 0.5, 1.1]);
  assert.equal(manifest.inference_boundary.expected_source_samples, 60);
  assert.equal(manifest.inference_boundary.expected_predictions, 120);
  assert.equal(manifest.inference_boundary.test_materialized, false);
  for (const input of Object.values(manifest.frozen_inputs)) {
    if (input.path) assert.equal(sha256(resolve(ROOT, input.path)), input.sha256, input.path);
  }
});

test("C2 producer is installed without simulator or trainer dependencies", () => {
  const project = readFileSync(resolve(ROOT, "learning/pyproject.toml"), "utf8");
  const source = readFileSync(resolve(ROOT, "learning/src/objgauss_learning/baselines.py"), "utf8");
  assert.match(project, /objgauss-pr02c-baselines = "objgauss_learning\.baselines:main"/);
  for (const forbidden of ["mani_skill", "sapien", "objgauss_sim"]) {
    assert.equal(source.includes(`import ${forbidden}`), false, forbidden);
    assert.equal(source.includes(`from ${forbidden}`), false, forbidden);
  }
  assert.doesNotMatch(source, /(?:torch\.optim|DataLoader|backward\(|optimizer\s*=)/);
  assert.match(source, /OBJGAUSS_LEARNING_OFFLINE/);
});

test("C2 verifier independently audits contracts, math, lineage, and repeatability", () => {
  const verifier = readFileSync(resolve(ROOT, "scripts/verify-pr02c-baselines.mjs"), "utf8");
  assert.doesNotMatch(verifier, /(?:import|from)[^\n]*objgauss_learning/);
  for (const check of [
    "05-sanitized-projection",
    "06-final-isolation",
    "08-contract-valid",
    "11-visible-field-isolation",
    "13-copy-state-semantics",
    "14-constant-velocity-semantics",
    "16-provenance",
    "18-reverse-repeat",
  ]) assert.ok(verifier.includes(`check("${check}"`), check);
  assert.match(verifier, /verifier_sha256: sha256\(await readFile\(VERIFIER_PATH\)\)/);
});

test("C2 clean gate is executable and orders data, offline inference, and verification", () => {
  const path = resolve(ROOT, "scripts/check-pr02c-baselines");
  const gate = readFileSync(path, "utf8");
  assert.ok((statSync(path).mode & 0o111) !== 0, "C2 gate must be executable");
  for (const required of [
    "check-pr02c-data",
    "objgauss-pr02c-data",
    "objgauss-pr02c-baselines",
    "verify-pr02c-baselines.mjs",
    "--order canonical",
    "--order reverse",
    "MUTATION_EXIT",
  ]) assert.ok(gate.includes(required), required);

  const provision = gate.indexOf('uv sync --project "$LEARNING_ROOT"');
  const dataGate = gate.indexOf("./scripts/check-pr02c-data");
  const offlineBoundary = gate.indexOf("export UV_OFFLINE=1");
  const producer = gate.indexOf('"$LEARNING_VENV/bin/objgauss-pr02c-baselines"');
  assert.ok(provision >= 0 && provision < dataGate, "C2 venv must precede the nested C1 gate");
  assert.ok(dataGate < offlineBoundary && offlineBoundary < producer, "project code must run offline");
});
