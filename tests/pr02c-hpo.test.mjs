import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import test from "node:test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";


const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = (path) => readFileSync(resolve(ROOT, path), "utf8");
const json = (path) => JSON.parse(read(path));


test("C6 manifest freezes 24 tasks, 12 fairness pairs, and two-arm mapping", () => {
  const manifest = json("learning/hpo-manifest.json");
  assert.equal(manifest.manifest_kind, "objgauss.pr02c-c6-hpo-selection");
  assert.deepEqual(manifest.matrix.learned_arms, ["action_free", "action_conditioned"]);
  assert.equal(manifest.matrix.configurations.length, 4);
  assert.equal(manifest.matrix.training_seeds.length, 3);
  assert.equal(manifest.matrix.fairness_pairs.length, 12);
  const tasks = manifest.matrix.fairness_pairs.flatMap((pair) => Object.values(pair.task_ids));
  assert.equal(tasks.length, 24);
  assert.equal(new Set(tasks).size, 24);
  assert.equal(manifest.selector.performance_promotion_threshold, null);
  assert.equal(manifest.formal_training_boundary.fit_split, "train");
  assert.equal(manifest.formal_training_boundary.checkpoint_selection_split, "validation");
});

test("independent selector imports no trainer, model, runner, or simulator logic", () => {
  const source = read("learning/src/objgauss_learning/selector.py");
  assert.doesNotMatch(source, /from \.trainer|from \.model|from \.hpo/);
  assert.doesNotMatch(source, /mani_skill|sapien|objgauss_sim/);
  assert.match(source, /final test is forbidden/);
  assert.match(source, /exactly 24 records/);
});

test("C6 runner exposes contract/run modes and pair-locked fairness", () => {
  const source = read("learning/src/objgauss_learning/hpo.py");
  const project = read("learning/pyproject.toml");
  assert.match(project, /objgauss-pr02c-hpo = "objgauss_learning\.hpo:main"/);
  assert.match(project, /objgauss-pr02c-selector = "objgauss_learning\.selector:main"/);
  assert.match(source, /pair_stop/);
  assert.match(source, /common initialization digest differs/);
  assert.match(source, /build_count.*1/s);
  assert.match(source, /final test is forbidden/);
  assert.doesNotMatch(source, /"contract_kind": "objgauss\.checkpoint_manifest"/);
  assert.match(source, /"checkpoint_manifest_published": False/);
});

test("C6 verifier is independent and checks public contracts and checkpoints", () => {
  const source = read("scripts/verify-pr02c-hpo.mjs");
  assert.doesNotMatch(source, /^import .*objgauss_learning|^import .*hpo|^import .*trainer/m);
  assert.match(source, /validateContract/);
  assert.match(source, /checkpointAudit/);
  for (const check of [
    "04-data-build-once",
    "05-final-test-isolation",
    "06-exact-task-set",
    "13-fairness-pair-ledger",
    "17-selector-order-repeat",
    "19-checkpoint-not-promoted",
    "20-claim-boundary",
  ]) assert.match(source, new RegExp(check));
});

test("C6 clean gate and remote workflow keep GPU HPO out of CPU CI", () => {
  const gate = read("scripts/check-pr02c-hpo");
  const workflow = read(".github/workflows/pr02c-cpu.yml");
  assert.equal((statSync(resolve(ROOT, "scripts/check-pr02c-hpo")).mode & 0o111) !== 0, true);
  assert.match(gate, /check-pr02c-data/);
  assert.match(gate, /--mode run/);
  assert.match(gate, /--device cuda/);
  assert.match(gate, /--order canonical/);
  assert.match(gate, /--order reverse/);
  assert.match(workflow, /--mode contract/);
  assert.match(workflow, /--mode tiny/);
  assert.doesNotMatch(workflow, /--mode run|--device cuda|check-pr02c-hpo/);
});
