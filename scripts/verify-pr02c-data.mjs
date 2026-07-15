#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createContractDispatcher } from "../src/pr01/contract-dispatch.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const BRANCHES = [
  "hold",
  "push-neg-x-weak",
  "push-pos-x-strong",
  "push-pos-x-weak",
  "push-pos-y-weak",
];
const FILES = [
  "attempt.json",
  "contact-ledger.json",
  "episode.json",
  "publication.json",
  "trajectory.json",
];
const SCORING_TIMES = [0.1, 0.2, 0.5, 1.1];

function sha256(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

function stableBytes(value) {
  if (Array.isArray(value)) return `[${value.map(stableBytes).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => (
      `${JSON.stringify(key)}:${stableBytes(value[key])}`
    )).join(",")}}`;
  }
  assert(!(typeof value === "number" && !Number.isFinite(value)), "non-finite JSON value");
  return JSON.stringify(value);
}

function digestDocument(value) {
  return sha256(Buffer.from(`${stableBytes(value)}\n`));
}

function load(path) {
  const bytes = readFileSync(path);
  return { bytes, document: JSON.parse(bytes.toString("utf8")) };
}

function names(path, kind) {
  return readdirSync(path, { withFileTypes: true })
    .filter((item) => item[kind]())
    .map((item) => item.name)
    .sort();
}

function atomicJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}`;
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
  try {
    renameSync(temporary, path);
  } finally {
    rmSync(temporary, { force: true });
  }
}

function parseArgs(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    assert(name?.startsWith("--") && value, `invalid argument pair: ${name}`);
    values[name.slice(2)] = value;
  }
  for (const name of [
    "root",
    "formal-spec",
    "dynamics-experiment",
    "manifest",
    "loader-report",
    "source-commit",
    "output",
  ]) assert(values[name], `missing --${name}`);
  assert(/^[a-f0-9]{40}$/.test(values["source-commit"]), "source commit must be exact SHA-1");
  return values;
}

function repoPath(raw, prefix = null) {
  assert(!raw.startsWith("/"), `path must be repository-relative: ${raw}`);
  const path = resolve(ROOT, raw);
  assert(path.startsWith(`${ROOT}/`), `path escapes repository root: ${raw}`);
  if (prefix) assert(raw.startsWith(`${prefix}/`), `path must be below ${prefix}/: ${raw}`);
  return path;
}

function expectedProjection(formal, splits) {
  return new Map(splits.flatMap((split) => formal.partitions[split].groups.map((group) => [
    group.group_id,
    { split, ...group },
  ])));
}

function producerTreeSha256() {
  const packageRoot = resolve(ROOT, "sim/src/objgauss_sim");
  const entries = Object.fromEntries([
    "adapter.py",
    "canonical.py",
    "cohort.py",
    "pr02_data.py",
    "primitive.py",
    "runtime.py",
    "writer.py",
  ].map((name) => [name, sha256(readFileSync(resolve(packageRoot, name))) ]));
  entries["validate-pr01-document.mjs"] = sha256(
    readFileSync(resolve(ROOT, "scripts/validate-pr01-document.mjs")),
  );
  return digestDocument(entries);
}

function validateStateMap(actors) {
  assert(actors && typeof actors === "object" && !Array.isArray(actors), "actor state map missing");
  for (const state of Object.values(actors)) {
    for (const [name, length] of [
      ["position_W_m", 3],
      ["quaternion_WO_wxyz", 4],
      ["linear_velocity_W_m_s", 3],
      ["angular_velocity_W_rad_s", 3],
    ]) {
      assert(Array.isArray(state[name]) && state[name].length === length, `state field drift: ${name}`);
      assert(state[name].every(Number.isFinite), `non-finite state field: ${name}`);
    }
    const norm = Math.hypot(...state.quaternion_WO_wxyz);
    assert(Math.abs(norm - 1) <= 1e-5, "quaternion is not normalized");
  }
}

function auditBranch({
  dataRoot,
  directory,
  groupId,
  branchId,
  expected,
  action,
  sourceCommit,
  planSha256,
  formalSha256,
  simulatorLockSha256,
  sourceTreeSha256,
  validate,
}) {
  assert.deepEqual(names(directory, "isFile"), FILES, `branch file set drift: ${groupId}/${branchId}`);
  const loaded = Object.fromEntries(FILES.map((name) => [name, load(resolve(directory, name))]));
  const episode = loaded["episode.json"].document;
  const attempt = loaded["attempt.json"].document;
  const trajectory = loaded["trajectory.json"].document;
  const contacts = loaded["contact-ledger.json"].document;
  const publication = loaded["publication.json"].document;
  assert(validate(episode).valid, `episode contract invalid: ${groupId}/${branchId}`);
  assert(validate(attempt).valid, `attempt contract invalid: ${groupId}/${branchId}`);
  const hashes = {
    episode_sha256: sha256(loaded["episode.json"].bytes),
    trajectory_sha256: sha256(loaded["trajectory.json"].bytes),
    contact_ledger_sha256: sha256(loaded["contact-ledger.json"].bytes),
    attempt_sha256: sha256(loaded["attempt.json"].bytes),
  };
  for (const [key, value] of Object.entries(hashes)) {
    assert.equal(publication[key], value, `publication ${key} drift: ${groupId}/${branchId}`);
  }
  assert.equal(publication.semantic_sha256, digestDocument({
    contact_ledger_sha256: hashes.contact_ledger_sha256,
    episode_sha256: hashes.episode_sha256,
    trajectory_sha256: hashes.trajectory_sha256,
  }), `semantic checksum drift: ${groupId}/${branchId}`);
  assert.equal(episode.identity.group_id, groupId);
  assert.equal(episode.identity.branch_id, branchId);
  assert.equal(episode.identity.split, expected.split);
  assert(["train", "validation"].includes(episode.identity.split), "test episode materialized");
  assert.equal(episode.environment.object_spec_id, expected.object_identity_id);
  assert.equal(episode.environment.layout_id, expected.layout_id);
  assert.equal(episode.environment.start_pose_id, expected.start_pose_id);
  assert.equal(episode.initialization.reset_seed, expected.reset_seed);
  assert.deepEqual(episode.intervention.commanded_action, Object.fromEntries(
    Object.entries(action).filter(([key]) => key !== "branch_id"),
  ), `commanded action drift: ${groupId}/${branchId}`);
  assert.equal(episode.provenance.source_commit, sourceCommit, "source commit drift");
  assert.equal(attempt.provenance.experiment_manifest_sha256, planSha256, "source plan lineage drift");
  assert.equal(episode.provenance.config_sha256, formalSha256, "episode formal spec lineage drift");
  assert.equal(attempt.provenance.config_sha256, formalSha256, "attempt formal spec lineage drift");
  assert.equal(episode.provenance.runtime_lock_sha256, simulatorLockSha256, "episode runtime lock drift");
  assert.equal(attempt.provenance.runtime_lock_sha256, simulatorLockSha256, "attempt runtime lock drift");
  assert.equal(episode.provenance.source_tree_sha256, sourceTreeSha256, "episode source tree drift");
  assert.equal(attempt.provenance.source_tree_sha256, sourceTreeSha256, "attempt source tree drift");
  for (const [descriptor, filename] of [
    [episode.evidence.trajectory, "trajectory.json"],
    [episode.evidence.contact_ledger, "contact-ledger.json"],
  ]) {
    const loadedArtifact = loaded[filename];
    assert.equal(descriptor.uri, relative(dataRoot, resolve(directory, filename)).replaceAll("\\", "/"));
    assert.equal(descriptor.sha256, sha256(loadedArtifact.bytes));
    assert.equal(descriptor.byte_length, loadedArtifact.bytes.length);
    assert.equal(descriptor.record_count, loadedArtifact.document.records.length);
  }
  const times = trajectory.records.map((record) => record.episode_time_s);
  assert(times.every(Number.isFinite), "non-finite trajectory time");
  assert(times.every((time, index) => index === 0 || time > times[index - 1]), "trajectory time not increasing");
  assert.equal(times[0], 0);
  assert.equal(times.at(-1), 1.1);
  for (const time of SCORING_TIMES) assert(times.includes(time), `scoring time missing: ${time}`);
  for (const record of trajectory.records) validateStateMap(record.actors);
  assert.equal(trajectory.records.length, contacts.records.length + 1, "trajectory/contact count drift");
  return {
    index: {
      group_id: groupId,
      branch_id: branchId,
      episode_sha256: hashes.episode_sha256,
      trajectory_sha256: hashes.trajectory_sha256,
      semantic_sha256: publication.semantic_sha256,
    },
    initialization: episode.initialization,
  };
}

const args = parseArgs(process.argv.slice(2));
const dataRoot = repoPath(args.root, "generated/pr02c");
const formalLoaded = load(repoPath(args["formal-spec"]));
const dynamicsLoaded = load(repoPath(args["dynamics-experiment"]));
const manifestLoaded = load(repoPath(args.manifest));
const loaderLoaded = load(repoPath(args["loader-report"], "generated/pr02c"));
const planLoaded = load(resolve(dataRoot, "source-plan.json"));
const sourceLoaded = load(resolve(dataRoot, "source-report.json"));
const output = repoPath(args.output, "generated/pr02c");
const formal = formalLoaded.document;
const dynamics = dynamicsLoaded.document;
const manifest = manifestLoaded.document;
const plan = planLoaded.document;
const sourceReport = sourceLoaded.document;
const loaderReport = loaderLoaded.document;
const sourceCommit = args["source-commit"];
const checks = [];
const validate = createContractDispatcher({ root: ROOT });

function check(checkId, predicate, detail = {}) {
  assert(predicate, `PR-02C C1 verification failed: ${checkId}`);
  checks.push({ check_id: checkId, status: "supported", ...detail });
}

check("manifest-freeze",
  manifest.manifest_kind === "objgauss.pr02c-data-boundary"
  && manifest.frozen_inputs.formal_data_spec.sha256 === sha256(formalLoaded.bytes)
  && manifest.frozen_inputs.simulator_lock.sha256
    === sha256(readFileSync(repoPath(manifest.frozen_inputs.simulator_lock.path)))
  && manifest.materialization.forbidden_split === "test"
  && JSON.stringify(manifest.materialization.allowed_splits) === JSON.stringify(["train", "validation"]));
check("dynamics-contract-lineage",
  validate(dynamics).valid
  && dynamics.source.source_gate_report_sha256 === sourceReport.inputs.pilot_report_sha256
  && sourceReport.inputs.dynamics_experiment_sha256 === sha256(dynamicsLoaded.bytes));
check("source-plan-boundary",
  plan.plan_kind === "objgauss.pr02c-source-plan"
  && plan.identity.experiment_spec_sha256 === sha256(formalLoaded.bytes)
  && JSON.stringify(plan.materialization.splits) === JSON.stringify(["train", "validation"])
  && plan.materialization.test_materialized === false);
check("source-report-lineage",
  sourceReport.verdict?.status === "supported"
  && sourceReport.producer?.source_commit === sourceCommit
  && sourceReport.inputs?.source_plan_sha256 === sha256(planLoaded.bytes)
  && sourceReport.inputs?.formal_data_spec_sha256 === sha256(formalLoaded.bytes));
check("source-tree-lineage",
  sourceReport.producer?.source_tree_sha256 === producerTreeSha256());

const expected = expectedProjection(formal, ["train", "validation"]);
const groupRoot = resolve(dataRoot, "dataset", formal.experiment_id);
const actualGroups = names(groupRoot, "isDirectory");
check("group-set-exact", actualGroups.length === 60
  && JSON.stringify(actualGroups) === JSON.stringify([...expected.keys()].sort()));
const testTokens = new Set([
  ...Object.keys(formal.partitions.test.objects),
  ...Object.keys(formal.partitions.test.layouts),
  ...formal.partitions.test.groups.map((group) => group.group_id),
]);
check("final-test-not-materialized", actualGroups.every((group) => !testTokens.has(group))
  && !actualGroups.some((group) => group.includes("pr02-test-")));

const actions = new Map(formal.actions.map((action) => [action.branch_id, action]));
const index = [];
const lineageBySplit = new Map([["train", new Set()], ["validation", new Set()]]);
for (const groupId of actualGroups) {
  const expectedGroup = expected.get(groupId);
  const directory = resolve(groupRoot, groupId);
  assert.deepEqual(names(directory, "isDirectory"), BRANCHES, `branch set drift: ${groupId}`);
  const initializations = [];
  for (const branchId of BRANCHES) {
    const result = auditBranch({
      dataRoot,
      directory: resolve(directory, branchId),
      groupId,
      branchId,
      expected: expectedGroup,
      action: actions.get(branchId),
      sourceCommit,
      planSha256: sha256(planLoaded.bytes),
      formalSha256: sha256(formalLoaded.bytes),
      simulatorLockSha256: manifest.frozen_inputs.simulator_lock.sha256,
      sourceTreeSha256: sourceReport.producer.source_tree_sha256,
      validate,
    });
    index.push(result.index);
    initializations.push(result.initialization);
  }
  assert(initializations.slice(1).every((value) => stableBytes(value) === stableBytes(initializations[0])),
    `sibling initialization drift: ${groupId}`);
  const lineage = stableBytes([
    initializations[0].snapshot_sha256,
    initializations[0].initial_state_sha256,
    initializations[0].restored_rng_sha256,
  ]);
  const other = expectedGroup.split === "train" ? "validation" : "train";
  assert(!lineageBySplit.get(other).has(lineage), `initialization lineage crosses split: ${groupId}`);
  lineageBySplit.get(expectedGroup.split).add(lineage);
}
index.sort((left, right) => stableBytes([left.group_id, left.branch_id])
  .localeCompare(stableBytes([right.group_id, right.branch_id])));
const dataIndexSha256 = digestDocument(index);
check("episode-attempt-contracts", index.length === 300);
check("artifact-checksums", index.every((item) => /^[a-f0-9]{64}$/.test(item.semantic_sha256)));
check("commanded-action-only-input-source", formal.actions.length === 5
  && dynamics.training.prediction_action === "commanded_action"
  && dynamics.training.executed_action_is_feature === false);
check("scoring-times-physical", JSON.stringify(manifest.materialization.scoring_times_s)
  === JSON.stringify(SCORING_TIMES));
check("split-identity-isolation",
  [...lineageBySplit.get("train")].every((value) => !lineageBySplit.get("validation").has(value))
  && Object.keys(formal.partitions.train.objects).every((id) => !(id in formal.partitions.validation.objects))
  && Object.keys(formal.partitions.train.layouts).every((id) => !(id in formal.partitions.validation.layouts)));
check("source-index-recomputed", sourceReport.data_index_sha256 === dataIndexSha256
  && sourceReport.counts.groups === 60
  && sourceReport.counts.branches === 300
  && sourceReport.counts.failed_attempts === 0);
check("loader-index-recomputed", loaderReport.verdict?.status === "supported"
  && loaderReport.source_commit === sourceCommit
  && loaderReport.data_index_sha256 === dataIndexSha256
  && loaderReport.counts.groups === 60
  && loaderReport.counts.branches === 300);
check("loader-feature-isolation",
  loaderReport.isolation?.test_materialized === false
  && loaderReport.isolation?.executed_action_is_model_input === false
  && loaderReport.isolation?.future_gt_is_model_input === false
  && JSON.stringify(loaderReport.isolation?.model_input_fields)
    === JSON.stringify(manifest.loader_boundary.model_input_fields));
check("claim-boundary",
  loaderReport.claim_boundary?.excluded_claims?.includes("trainer-implemented")
  && sourceReport.claim_boundary?.excluded_claims?.includes("model-performance"));

const verification = {
  report_version: "0.1.0",
  report_kind: "objgauss.pr02c-data-verification",
  verdict: "supported",
  source_commit: sourceCommit,
  inputs: {
    formal_data_spec_sha256: sha256(formalLoaded.bytes),
    dynamics_experiment_sha256: sha256(dynamicsLoaded.bytes),
    data_boundary_manifest_sha256: sha256(manifestLoaded.bytes),
    source_report_sha256: sha256(sourceLoaded.bytes),
    loader_report_sha256: sha256(loaderLoaded.bytes),
  },
  counts: { groups: 60, branches: 300, checks: checks.length },
  data_index_sha256: dataIndexSha256,
  checks,
  claim_boundary: "PR-02C C1 train/validation materialization and loader isolation only",
};
atomicJson(output, verification);
process.stdout.write(`${JSON.stringify({
  verdict: verification.verdict,
  check_count: checks.length,
  data_index_sha256: dataIndexSha256,
  output: args.output,
})}\n`);
