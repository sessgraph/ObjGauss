import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { validateContract } from "../src/pr01/contract-dispatch.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURE_ROOT = resolve(ROOT, "contracts/fixtures/pr02b");

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function disjoint(left, right) {
  return [...left].every((item) => !right.has(item));
}

test("the PR-02B manifest freezes every pilot input", () => {
  const manifest = readJson(resolve(FIXTURE_ROOT, "manifest.json"));
  assert.equal(manifest.fixture_kind, "objgauss.pr02b-pilot-input-manifest");
  assert.equal(manifest.inputs.length, 4);
  for (const item of manifest.inputs) {
    assert.equal(sha256(resolve(ROOT, item.path)), item.sha256, item.path);
  }
  assert.deepEqual(manifest.pilot_exclusion, {
    pr01_experiment_id: "pr01e-primitive-cohort-v0",
    pr02b_experiment_id: "pr02b-calibration-v0",
    ids_disjoint: true,
    seeds_disjoint: true,
    excluded_from_training: true,
    excluded_from_final_statistics: true,
  });
});

test("the source experiment is valid, finite, and exactly matches the isolated pilot", () => {
  const spec = readJson(resolve(FIXTURE_ROOT, "pilot-spec.json"));
  const experiment = readJson(resolve(FIXTURE_ROOT, "source-experiment.json"));
  const result = validateContract(experiment);
  assert.equal(result.valid, true, JSON.stringify(result));
  assert.equal(experiment.identity.experiment_spec_sha256, sha256(resolve(FIXTURE_ROOT, "pilot-spec.json")));
  assert.deepEqual(experiment.design.object_spec_ids, Object.keys(spec.object_specs));
  assert.deepEqual(experiment.design.layout_ids, Object.keys(spec.layouts));
  assert.deepEqual(experiment.design.start_pose_ids, Object.keys(spec.start_poses));
  assert.equal(experiment.preflight.expected_group_count,
    Object.keys(spec.object_specs).length
      * Object.keys(spec.layouts).length
      * spec.preflight_start_pose_ids.length
      * spec.preflight_reset_seeds.length);
  assert.equal(experiment.preflight.expected_episode_count,
    experiment.preflight.expected_group_count * experiment.actions.length);
  assert.deepEqual(spec.calibration.repeat_orders, ["canonical", "reverse"]);
  assert.deepEqual(experiment.actions.map((item) => item.branch_id), [
    "hold",
    "push-pos-x-weak",
    "push-pos-x-strong",
    "push-neg-x-weak",
    "push-pos-y-weak",
  ]);
});

test("PR-02B identities, layouts, and reset seeds are disjoint from PR-01", () => {
  const pr02 = readJson(resolve(FIXTURE_ROOT, "pilot-spec.json"));
  const pr01 = readJson(resolve(ROOT, "contracts/fixtures/pr01e/cohort-spec.json"));
  assert.equal(disjoint(new Set(Object.keys(pr02.object_specs)), new Set(Object.keys(pr01.object_specs))), true);
  assert.equal(disjoint(new Set(Object.keys(pr02.layouts)), new Set(Object.keys(pr01.layouts))), true);
  const pr02Seeds = new Set([...pr02.formal_reset_seeds, ...pr02.preflight_reset_seeds]);
  const pr01Seeds = new Set([...pr01.formal_reset_seeds, ...pr01.preflight_reset_seeds]);
  assert.equal(disjoint(pr02Seeds, pr01Seeds), true);
});

test("the fixed grid and minimum power candidate fit all GPU-hour ceilings", () => {
  const grid = readJson(resolve(FIXTURE_ROOT, "hyperparameter-grid.json"));
  const spec = readJson(resolve(FIXTURE_ROOT, "pilot-spec.json"));
  const source = readJson(resolve(FIXTURE_ROOT, "source-experiment.json"));
  const configurationCount = grid.architecture.hidden_width.length
    * grid.optimization.learning_rate.length
    * grid.optimization.weight_decay.length
    * grid.optimization.batch_size.length;
  assert.equal(configurationCount, grid.search.configuration_count);
  const trainingSeeds = Math.min(...spec.power.candidate_training_seed_counts);
  const hpoHours = 2 * configurationCount * trainingSeeds
    * grid.per_task_limits.hpo_wall_seconds_max / 3600;
  const formalHours = 2 * trainingSeeds
    * grid.per_task_limits.formal_wall_seconds_max / 3600;
  assert.equal(hpoHours, 6);
  assert.equal(formalHours, 4);
  const retryMultiplier = 1 + source.budgets.extra_attempt_fraction_max;
  assert.ok(hpoHours * retryMultiplier <= spec.budgets.gpu_hours_pilot_hpo_max);
  assert.ok(formalHours * retryMultiplier <= spec.budgets.gpu_hours_formal_max);
  assert.ok((hpoHours + formalHours) * retryMultiplier <= spec.budgets.gpu_hours_total_max);
  assert.equal(grid.per_task_limits.display_vram_reserve_bytes_min, 1024 ** 3);
});

test("the pure PR-02B calibration and power unit suite passes without simulator assets", () => {
  const result = spawnSync(process.env.PYTHON ?? "python3", [
    "-m",
    "unittest",
    "sim.tests.test_pr02_pilot",
  ], {
    cwd: ROOT,
    encoding: "utf8",
    env: {
      ...process.env,
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONPATH: resolve(ROOT, "sim/src"),
    },
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});
