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

test("C3 manifest freezes a narrow golden smoke without selecting HPO or formal", () => {
  const manifest = JSON.parse(readFileSync(resolve(ROOT, "learning/trainer-manifest.json")));
  assert.equal(manifest.manifest_kind, "objgauss.pr02c-golden-trainer");
  assert.deepEqual(manifest.arms, ["action_free", "action_conditioned"]);
  assert.deepEqual(manifest.rollout.delta_t_s, [0.1, 0.1, 0.3, 0.6]);
  assert.equal(manifest.architecture.action_feature_width, 9);
  assert.equal(
    manifest.architecture.action_application_point,
    "target-object-center-of-mass-zero-in-object-frame",
  );
  assert.equal(manifest.golden.optimizer_updates, 8);
  assert.equal(manifest.golden.hpo_config_selected, false);
  assert.equal(manifest.golden.formal_checkpoint_frozen, false);
  assert.equal(manifest.resources.display_vram_reserve_bytes_min, 1024 ** 3);
  for (const input of Object.values(manifest.frozen_inputs)) {
    if (input.path) assert.equal(sha256(resolve(ROOT, input.path)), input.sha256, input.path);
  }
});

test("C3 model and trainer remain pure PyTorch and simulator-free", () => {
  const project = readFileSync(resolve(ROOT, "learning/pyproject.toml"), "utf8");
  const model = readFileSync(resolve(ROOT, "learning/src/objgauss_learning/model.py"), "utf8");
  const trainer = readFileSync(resolve(ROOT, "learning/src/objgauss_learning/trainer.py"), "utf8");
  assert.match(project, /objgauss-pr02c-trainer = "objgauss_learning\.trainer:main"/);
  assert.match(project, /objgauss-pr02c-checkpoint-audit = "objgauss_learning\.checkpoint:main"/);
  for (const source of [model, trainer]) {
    for (const forbidden of ["mani_skill", "sapien", "objgauss_sim"] ) {
      assert.equal(source.includes(`import ${forbidden}`), false, forbidden);
      assert.equal(source.includes(`from ${forbidden}`), false, forbidden);
    }
  }
  assert.match(trainer, /OBJGAUSS_LEARNING_OFFLINE/);
  assert.match(model, /class MinimalObjectGNN/);
  assert.match(model, /self\.pairwise_message/);
  assert.match(model, /self\.shared_residual_head/);
});

test("C3 verifier is independent from trainer and audits semantic checkpoints", () => {
  const verifier = readFileSync(resolve(ROOT, "scripts/verify-pr02c-trainer.mjs"), "utf8");
  const checkpoint = readFileSync(
    resolve(ROOT, "learning/src/objgauss_learning/checkpoint.py"), "utf8",
  );
  assert.doesNotMatch(verifier, /(?:import|from)[^\n]*objgauss_learning/);
  assert.doesNotMatch(checkpoint, /from \.trainer|from \.model|import objgauss_learning\.trainer/);
  assert.match(verifier, /const REPORT_VERSION = "0\.1\.0";/);
  for (const check of [
    "09-parameter-parity",
    "10-update-parity",
    "11-data-order-parity",
    "13-public-contracts",
    "18-checkpoint-semantic-audit",
    "20-semantic-repeat",
    "23-resource-display-reserve",
  ]) assert.ok(verifier.includes(`check("${check}"`), check);
});

test("C3 clean gate is executable and keeps HPO and final test out of scope", () => {
  const path = resolve(ROOT, "scripts/check-pr02c-trainer");
  const gate = readFileSync(path, "utf8");
  assert.ok((statSync(path).mode & 0o111) !== 0, "C3 gate must be executable");
  for (const required of [
    "check-pr02c-baselines",
    "--mode tiny",
    "--mode golden",
    "--order canonical",
    "--order reverse",
    "verify-pr02c-trainer.mjs",
    "TEST_SPLIT_EXIT",
    "MUTATION_EXIT",
  ]) assert.ok(gate.includes(required), required);
  assert.doesNotMatch(gate, /(?:24-task|formal-training|--split test|--splits train validation test)/);
});
