import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { canonicalStringify } from "../src/pr00/canonical-json.mjs";
import { validateContract } from "../src/pr01/contract-dispatch.mjs";

const VERIFIER_PATH = fileURLToPath(import.meta.url);
const REPO_ROOT = resolve(dirname(VERIFIER_PATH), "..");
const ARMS = ["copy_state", "constant_velocity"];
const BRANCH_IDS = [
  "hold",
  "push-neg-x-weak",
  "push-pos-x-strong",
  "push-pos-x-weak",
  "push-pos-y-weak",
];
const SCORING_TIMES = [0.1, 0.2, 0.5, 1.1];
const CLAIM_BOUNDARY = {
  supported_claim: "deterministic-validation-baselines-are-reproducible-and-auditable",
  excluded_claims: [
    "test-prediction-produced",
    "trainer-implemented",
    "learned-model-performance",
    "scientific-baseline-comparison",
    "gaussian-dynamics-value",
  ],
};
const FORBIDDEN_FIELDS = new Set([
  "executed_action",
  "control_ledger",
  "future_object_states",
  "gt_future",
  "labels",
  "training_labels",
]);

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) throw new Error(`invalid argument: ${key}`);
    result[key.slice(2)] = value;
  }
  for (const required of [
    "root",
    "repeat-root",
    "bundle",
    "loader-report",
    "manifest",
    "source-commit",
    "output",
  ]) {
    if (!(required in result)) throw new Error(`missing --${required}`);
  }
  return result;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]));
  }
  return value;
}

function compactCanonical(value) {
  return `${JSON.stringify(canonicalize(value))}\n`;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function sameJson(left, right) {
  return canonicalStringify(left) === canonicalStringify(right);
}

function safeRepoPath(raw, prefix = null) {
  assert(typeof raw === "string" && raw.length > 0 && !raw.startsWith("/"), `unsafe path: ${raw}`);
  const path = resolve(REPO_ROOT, raw);
  const rel = relative(REPO_ROOT, path);
  assert(rel !== ".." && !rel.startsWith(`..${sep}`), `path escapes repository: ${raw}`);
  const normalized = rel.split(sep).join("/");
  if (prefix !== null) assert(normalized.startsWith(`${prefix.replace(/\/$/, "")}/`), `path outside ${prefix}: ${raw}`);
  return path;
}

function safePredictionPath(root, raw) {
  assert(typeof raw === "string" && raw.length > 0 && !raw.startsWith("/"), `unsafe prediction path: ${raw}`);
  const path = resolve(root, raw);
  const rel = relative(root, path);
  assert(rel !== ".." && !rel.startsWith(`..${sep}`), `prediction path escapes output root: ${raw}`);
  const normalized = rel.split(sep).join("/");
  assert(normalized.startsWith("predictions/"), `prediction path is outside predictions/: ${raw}`);
  return path;
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

function walkForbidden(value) {
  if (Array.isArray(value)) {
    for (const item of value) walkForbidden(item);
  } else if (value !== null && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      assert(!FORBIDDEN_FIELDS.has(key), `forbidden field: ${key}`);
      walkForbidden(item);
    }
  }
}

function canonicalQuaternion(raw) {
  assert(Array.isArray(raw) && raw.length === 4 && raw.every(Number.isFinite), "invalid quaternion");
  const norm = Math.sqrt(raw.reduce((sum, value) => sum + value * value, 0));
  assert(Math.abs(norm - 1) <= 1e-6, "non-unit input quaternion");
  let result = raw.map((value) => value / norm);
  const first = result.find((value) => Math.abs(value) > 1e-12);
  if (first !== undefined && first < 0) result = result.map((value) => -value);
  return result.map((value) => Object.is(value, -0) || value === 0 ? 0 : value);
}

function multiplyQuaternion(left, right) {
  const [lw, lx, ly, lz] = left;
  const [rw, rx, ry, rz] = right;
  return [
    lw * rw - lx * rx - ly * ry - lz * rz,
    lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw,
  ];
}

function integrateWorldQuaternion(quaternion, omega, time) {
  const initial = canonicalQuaternion(quaternion);
  const speed = Math.sqrt(omega.reduce((sum, value) => sum + value * value, 0));
  if (speed === 0) return initial;
  const halfAngle = 0.5 * speed * time;
  const scale = Math.sin(halfAngle) / speed;
  const delta = [Math.cos(halfAngle), ...omega.map((value) => value * scale)];
  return canonicalQuaternion(multiplyQuaternion(delta, initial));
}

function predictedState(state, arm, time) {
  const position = state.position_W_m;
  const linear = state.linear_velocity_W_m_s;
  const angular = state.angular_velocity_W_rad_s;
  assert([position, linear, angular].every((value) => (
    Array.isArray(value) && value.length === 3 && value.every(Number.isFinite)
  )), "invalid ObjectState vector");
  const predictedPosition = arm === "copy_state"
    ? [...position]
    : position.map((value, index) => value + linear[index] * time);
  const quaternion = arm === "copy_state"
    ? canonicalQuaternion(state.quaternion_WO_wxyz)
    : integrateWorldQuaternion(state.quaternion_WO_wxyz, angular, time);
  return {
    object_id: state.object_id,
    position_W_m: predictedPosition.map((value) => Object.is(value, -0) || value === 0 ? 0 : value),
    quaternion_WO_wxyz: quaternion,
    linear_velocity_W_m_s: linear.map((value) => Object.is(value, -0) || value === 0 ? 0 : value),
    angular_velocity_W_rad_s: angular.map((value) => Object.is(value, -0) || value === 0 ? 0 : value),
  };
}

function expectedPrediction(sample, arm) {
  const states = [...sample.initial_object_states].sort((left, right) => (
    left.object_id < right.object_id ? -1 : left.object_id > right.object_id ? 1 : 0
  ));
  return SCORING_TIMES.map((time) => ({
    time_s: time,
    objects: states.map((state) => predictedState(state, arm, time)),
  }));
}

async function packageTreeSha256() {
  const learningRoot = resolve(REPO_ROOT, "learning");
  const packageRoot = resolve(learningRoot, "src/objgauss_learning");
  const jsonFiles = (await readdir(learningRoot)).filter((name) => name.endsWith(".json")).sort();
  const pythonFiles = (await readdir(packageRoot)).filter((name) => name.endsWith(".py")).sort();
  const paths = [
    resolve(learningRoot, "pyproject.toml"),
    ...jsonFiles.map((name) => resolve(learningRoot, name)),
    ...pythonFiles.map((name) => resolve(packageRoot, name)),
  ];
  const entries = {};
  for (const path of paths) {
    entries[relative(REPO_ROOT, path).split(sep).join("/")] = sha256(await readFile(path));
  }
  return sha256(Buffer.from(compactCanonical(entries)));
}

async function listPredictionFiles(root) {
  const result = [];
  async function visit(path) {
    for (const entry of await readdir(path, { withFileTypes: true })) {
      const child = resolve(path, entry.name);
      if (entry.isDirectory()) await visit(child);
      else if (entry.isFile() && entry.name.endsWith(".json")) result.push(relative(root, child).split(sep).join("/"));
    }
  }
  await visit(resolve(root, "predictions"));
  return result.sort();
}

const args = parseArgs(process.argv.slice(2));
const root = safeRepoPath(args.root, "generated/pr02c");
const repeatRoot = safeRepoPath(args["repeat-root"], "generated/pr02c");
const bundlePath = safeRepoPath(args.bundle, "generated/pr02c");
const loaderReportPath = safeRepoPath(args["loader-report"], "generated/pr02c");
const manifestPath = safeRepoPath(args.manifest);
const outputPath = safeRepoPath(args.output, "generated/pr02c");
const sourceCommit = args["source-commit"];
assert(/^[a-f0-9]{40}$/.test(sourceCommit), "invalid source commit");

const checks = [];
async function check(id, action) {
  try {
    const evidence = await action();
    checks.push({ check_id: id, status: "supported", evidence });
  } catch (error) {
    checks.push({ check_id: id, status: "invalid", message: error.message });
  }
}

const manifest = await readJson(manifestPath);
const bundle = await readJson(bundlePath);
const loaderReport = await readJson(loaderReportPath);
const index = await readJson(resolve(root, "index.json"));
const report = await readJson(resolve(root, "report.json"));
const repeatIndexBytes = await readFile(resolve(repeatRoot, "index.json"));
const repeatIndex = JSON.parse(repeatIndexBytes);
const repeatReport = await readJson(resolve(repeatRoot, "report.json"));
const predictionDocuments = new Map();
for (const item of index.predictions) {
  predictionDocuments.set(
    `${item.group_id}:${item.branch_id}:${item.model_arm}`,
    await readJson(safePredictionPath(root, item.uri)),
  );
}

await check("01-manifest-frozen", async () => {
  assert(sameJson(Object.keys(manifest).sort(), [
    "arms",
    "claim_boundary",
    "frozen_inputs",
    "horizon",
    "inference_boundary",
    "manifest_kind",
    "manifest_version",
    "outputs",
    "semantics",
  ]), "manifest top-level field drift");
  assert(manifest.manifest_version === "0.1.0", "manifest version drift");
  assert(manifest.manifest_kind === "objgauss.pr02c-deterministic-baselines", "manifest kind drift");
  assert(sameJson(manifest.arms, ARMS), "arm set drift");
  assert(sameJson(manifest.horizon.scoring_times_s, SCORING_TIMES), "horizon drift");
  assert(sameJson(manifest.inference_boundary, {
    bundle_kind: "objgauss.pr02c-model-input-bundle",
    allowed_split: "validation",
    expected_groups: 12,
    branches_per_group: 5,
    expected_source_samples: 60,
    expected_predictions: 120,
    visible_fields: ["initial_objectstate", "commanded_action_schedule", "non_future_metadata"],
    executed_action_is_feature: false,
    gt_future_read: false,
    test_materialized: false,
  }), "inference boundary drift");
  assert(sameJson(manifest.semantics, {
    copy_state: {
      position: "copy-initial",
      orientation: "copy-initial-and-canonicalize-sign",
      linear_velocity: "copy-initial",
      angular_velocity: "copy-initial",
    },
    constant_velocity: {
      position: "position-0-plus-linear-velocity-W-times-absolute-time",
      orientation: "left-multiply-world-angular-velocity-exponential-and-canonicalize-sign",
      linear_velocity: "copy-initial",
      angular_velocity: "copy-initial",
    },
    quaternion: {
      layout: "wxyz",
      frame: "quaternion-WO-with-angular-velocity-W",
      normalization: "unit-norm-required",
      canonical_sign: "first-component-with-absolute-value-above-1e-12-is-positive",
    },
  }), "baseline semantics drift");
  assert(sameJson(manifest.claim_boundary, CLAIM_BOUNDARY), "manifest claim boundary drift");
  assert(sameJson(manifest.outputs, {
    root: "generated/pr02c/baselines/",
    prediction_contract_kind: "objgauss.dynamics_prediction",
    prediction_schema_version: "0.3.0",
    atomic_publish_required: true,
    git_ignored: true,
  }), "output contract drift");
  assert(sameJson(manifest.frozen_inputs.dynamics_experiment, {
    runtime_path: "generated/pr02b/evidence/freeze/dynamics-experiment.json",
    contract_kind: "objgauss.dynamics_experiment",
    schema_version: "0.3.0",
  }), "dynamics experiment contract drift");
  for (const [name, entry] of Object.entries(manifest.frozen_inputs)) {
    const path = entry.path ?? entry.runtime_path;
    if (path && entry.sha256) {
      assert(sha256(await readFile(safeRepoPath(path))) === entry.sha256, `frozen hash drift: ${name}`);
    }
  }
  return { manifest_sha256: sha256(await readFile(manifestPath)) };
});

await check("02-bundle-boundary", async () => {
  assert(bundle.bundle_kind === "objgauss.pr02c-model-input-bundle", "bundle kind drift");
  assert(bundle.bundle_version === "0.1.0", "bundle version drift");
  assert(bundle.experiment_id === "pr02-objectstate-baseline-v0", "bundle experiment identity drift");
  assert(bundle.source_commit === sourceCommit && bundle.split === "validation", "bundle source or split drift");
  assert(bundle.samples.length === 60, "bundle sample count drift");
  assert(loaderReport.source_commit === sourceCommit, "loader report source commit drift");
  assert(loaderReport.verdict?.status === "supported", "loader report is not supported");
  assert(loaderReport.counts?.groups === 60 && loaderReport.counts?.branches === 300, "loader counts drift");
  const expectedInputs = {
    ...loaderReport.inputs,
    data_index_sha256: loaderReport.data_index_sha256,
    model_input_index_sha256: loaderReport.model_input_index_sha256,
    loader_report_sha256: sha256(await readFile(loaderReportPath)),
    runtime_lock_sha256: manifest.frozen_inputs.runtime_lock.sha256,
  };
  assert(sameJson(bundle.inputs, expectedInputs), "bundle/loader lineage drift");
  assert(bundle.inputs.data_boundary_manifest_sha256 === manifest.frozen_inputs.data_boundary_manifest.sha256, "data boundary lineage drift");
  assert(bundle.inputs.formal_data_spec_sha256 === manifest.frozen_inputs.formal_data_spec.sha256, "formal spec lineage drift");
  assert(sameJson(bundle.claim_boundary, {
    supported_claim: "sanitized-validation-model-inputs-are-published",
    excluded_claims: [
      "prediction-produced",
      "trainer-implemented",
      "model-performance",
      "scientific-baseline-comparison",
    ],
  }), "bundle claim boundary drift");
  walkForbidden(bundle.samples);
  assert(bundle.sample_payload_sha256 === sha256(Buffer.from(canonicalStringify(bundle.samples))), "bundle payload hash drift");
  assert(bundle.isolation.gt_future_read === false && bundle.isolation.executed_action_is_feature === false, "bundle isolation drift");
  return { samples: bundle.samples.length, payload_sha256: bundle.sample_payload_sha256 };
});

await check("03-bundle-input-hashes", async () => {
  for (const sample of bundle.samples) {
    assert(sample.initial_objectstate_sha256 === sha256(Buffer.from(canonicalStringify(sample.initial_object_states))), "initial state hash drift");
    assert(sample.commanded_action_sha256 === sha256(Buffer.from(canonicalStringify(sample.commanded_action))), "command hash drift");
    assert(sample.rollout_times_s.every((time, index_) => time === SCORING_TIMES[index_]), "sample horizon drift");
  }
  return { initial_and_action_hashes: bundle.samples.length * 2 };
});

await check("04-source-episode-lineage", async () => {
  for (const sample of bundle.samples) {
    const sourcePath = safeRepoPath(sample.source_episode.uri, "generated/pr02c/data");
    const episodeBytes = await readFile(sourcePath);
    const episode = JSON.parse(episodeBytes);
    const publication = await readJson(resolve(dirname(sourcePath), "publication.json"));
    assert(sha256(episodeBytes) === sample.source_episode.sha256, "source episode hash drift");
    assert(publication.episode_sha256 === sample.source_episode.sha256, "publication episode hash drift");
    assert(episode.identity.group_id === sample.group_id && episode.identity.branch_id === sample.branch_id, "source identity drift");
    assert(episode.identity.split === "validation" && episode.provenance.source_commit === sourceCommit, "source split/commit drift");
    const lineage = {
      source_commit: sourceCommit,
      source_plan_sha256: bundle.inputs.source_plan_sha256,
      episode_sha256: publication.episode_sha256,
      semantic_sha256: publication.semantic_sha256,
    };
    assert(sha256(Buffer.from(compactCanonical(lineage))) === sample.source_episode.lineage_sha256, "source lineage hash drift");
  }
  return { source_episodes: bundle.samples.length };
});

await check("05-sanitized-projection", async () => {
  for (const sample of bundle.samples) {
    const sourcePath = safeRepoPath(sample.source_episode.uri, "generated/pr02c/data");
    const episode = await readJson(sourcePath);
    const trajectoryPath = resolve(dirname(sourcePath), "trajectory.json");
    const trajectoryBytes = await readFile(trajectoryPath);
    const trajectory = JSON.parse(trajectoryBytes);
    assert(sha256(trajectoryBytes) === episode.evidence.trajectory.sha256, "trajectory descriptor hash drift");
    const first = trajectory.records.find((record) => record.episode_time_s === 0);
    assert(first, "initial trajectory record missing");
    const projected = Object.keys(first.actors).sort().map((objectId) => ({
      object_id: objectId,
      position_W_m: first.actors[objectId].position_W_m,
      quaternion_WO_wxyz: first.actors[objectId].quaternion_WO_wxyz,
      linear_velocity_W_m_s: first.actors[objectId].linear_velocity_W_m_s,
      angular_velocity_W_rad_s: first.actors[objectId].angular_velocity_W_rad_s,
    }));
    assert(sameJson(projected, sample.initial_object_states), "initial state projection drift");
    assert(sameJson(episode.intervention.commanded_action, sample.commanded_action), "command projection drift");
    assert(episode.intervention.target_object_id === sample.target_object_id, "target projection drift");
  }
  return { sanitized_projections: bundle.samples.length };
});

await check("06-final-isolation", async () => {
  assert(bundle.samples.every((sample) => sample.split === "validation" && !sample.group_id.includes("-test-")), "test sample exposed");
  assert(index.predictions.every((item) => !item.uri.includes("test") && item.group_id.includes("validation")), "test prediction exposed");
  return { test_materialized: false };
});

await check("07-output-file-set", async () => {
  const files = await listPredictionFiles(root);
  assert(files.length === 120, "prediction file count drift");
  assert(index.predictions.length === 120, "index prediction count drift");
  assert(sameJson(files, index.predictions.map((item) => item.uri).sort()), "prediction file/index mismatch");
  return { prediction_files: files.length };
});

await check("08-contract-valid", async () => {
  for (const document of predictionDocuments.values()) {
    const validation = validateContract(document);
    assert(validation.valid, `prediction contract invalid: ${validation.reason_code}`);
  }
  return { contract_valid_predictions: predictionDocuments.size };
});

await check("09-identity-complete", async () => {
  const expected = new Set(bundle.samples.flatMap((sample) => ARMS.map((arm) => `${sample.group_id}:${sample.branch_id}:${arm}`)));
  assert(predictionDocuments.size === expected.size, "prediction identity count drift");
  for (const key of expected) assert(predictionDocuments.has(key), `missing prediction: ${key}`);
  for (const item of index.predictions) {
    const key = `${item.group_id}:${item.branch_id}:${item.model_arm}`;
    const document = predictionDocuments.get(key);
    const sample = bundle.samples.find((candidate) => (
      candidate.group_id === item.group_id && candidate.branch_id === item.branch_id
    ));
    assert(sample, `prediction sample missing: ${key}`);
    assert(document.identity.experiment_id === bundle.experiment_id, "prediction experiment identity drift");
    assert(document.identity.group_id === item.group_id, "prediction group identity drift");
    assert(document.identity.branch_id === item.branch_id, "prediction branch identity drift");
    assert(document.identity.model_arm === item.model_arm, "prediction arm identity drift");
    const expectedId = `prediction-${item.model_arm}-${sha256(Buffer.from(compactCanonical([
      item.group_id,
      item.branch_id,
      item.model_arm,
    ]))).slice(0, 24)}`;
    assert(document.identity.prediction_id === expectedId, "prediction ID drift");
    assert(sameJson(document.inputs.source_episode, sample.source_episode), "source episode reference drift");
    assert(document.inputs.initial_objectstate_sha256 === sample.initial_objectstate_sha256, "initial input hash drift");
    assert(document.inputs.commanded_action_sha256 === sample.commanded_action_sha256, "action input hash drift");
    assert(document.inputs.target_object_id === sample.target_object_id, "target object drift");
    assert(sameJson(document.horizon.scoring_times_s, SCORING_TIMES), "prediction horizon drift");
    assert(document.horizon.duration_s === 1.1, "prediction duration drift");
    assert(item.prediction_payload_sha256 === document.prediction_payload_sha256, "index payload hash drift");
  }
  return { identities: expected.size };
});

await check("10-no-learned-lineage", async () => {
  for (const document of predictionDocuments.values()) {
    assert([document.identity.trial_id, document.identity.checkpoint_id, document.identity.training_seed].every((value) => (
      value.availability === "missing" && value.reason === "not_applicable"
    )), "deterministic prediction has learned lineage");
  }
  return { trial_records: 0, checkpoint_records: 0 };
});

await check("11-visible-field-isolation", async () => {
  for (const document of predictionDocuments.values()) {
    assert(document.inputs.gt_future_read === false && document.inputs.executed_action_is_feature === false, "prediction feature isolation drift");
    walkForbidden(document);
  }
  return { gt_future_read: false, executed_action_is_feature: false };
});

await check("12-payload-checksums", async () => {
  for (const [key, document] of predictionDocuments) {
    const item = index.predictions.find((candidate) => `${candidate.group_id}:${candidate.branch_id}:${candidate.model_arm}` === key);
    assert(document.prediction_payload_sha256 === sha256(Buffer.from(canonicalStringify(document.predictions))), "prediction payload hash drift");
    assert(item.sha256 === sha256(await readFile(safePredictionPath(root, item.uri))), "prediction artifact hash drift");
  }
  return { payload_checksums: predictionDocuments.size };
});

await check("13-copy-state-semantics", async () => {
  for (const sample of bundle.samples) {
    const observed = predictionDocuments.get(`${sample.group_id}:${sample.branch_id}:copy_state`).predictions;
    assert(sameJson(observed, expectedPrediction(sample, "copy_state")), "copy-state semantic drift");
  }
  return { copy_state_predictions: bundle.samples.length };
});

await check("14-constant-velocity-semantics", async () => {
  for (const sample of bundle.samples) {
    const observed = predictionDocuments.get(`${sample.group_id}:${sample.branch_id}:constant_velocity`).predictions;
    assert(sameJson(observed, expectedPrediction(sample, "constant_velocity")), "constant-velocity semantic drift");
  }
  return { constant_velocity_predictions: bundle.samples.length };
});

await check("15-sibling-initial-invariance", async () => {
  const groups = new Map();
  for (const sample of bundle.samples) {
    const hash = sha256(Buffer.from(canonicalStringify(sample.initial_object_states)));
    if (!groups.has(sample.group_id)) groups.set(sample.group_id, new Set());
    groups.get(sample.group_id).add(hash);
  }
  assert([...groups.values()].every((hashes) => hashes.size === 1), "sibling initial states drift");
  for (const groupId of groups.keys()) {
    for (const arm of ARMS) {
      const hashes = BRANCH_IDS.map((branch) => predictionDocuments.get(`${groupId}:${branch}:${arm}`).prediction_payload_sha256);
      assert(new Set(hashes).size === 1, "deterministic sibling predictions differ");
    }
  }
  return { groups: groups.size, branches_per_group: 5 };
});

await check("16-provenance", async () => {
  const treeSha = await packageTreeSha256();
  for (const document of predictionDocuments.values()) {
    assert(document.provenance.source_commit === sourceCommit, "prediction source commit drift");
    assert(document.provenance.source_tree_sha256 === treeSha, "prediction source tree drift");
    assert(document.provenance.runtime_lock_sha256 === bundle.inputs.runtime_lock_sha256, "prediction runtime lock drift");
    assert(document.provenance.experiment_spec_sha256 === bundle.inputs.dynamics_experiment_sha256, "prediction experiment lineage drift");
  }
  return { source_tree_sha256: treeSha };
});

await check("17-report-and-index", async () => {
  const manifestSha = sha256(await readFile(manifestPath));
  const bundleSha = sha256(await readFile(bundlePath));
  assert(index.index_version === "0.1.0", "index version drift");
  assert(index.index_kind === "objgauss.pr02c-deterministic-baseline-index", "index kind drift");
  assert(index.source_commit === sourceCommit, "index source commit drift");
  assert(index.baseline_manifest_sha256 === manifestSha, "index manifest lineage drift");
  assert(index.model_input_bundle_sha256 === bundleSha, "index bundle lineage drift");
  const semanticEntries = index.predictions.map((item) => ({
    group_id: item.group_id,
    branch_id: item.branch_id,
    model_arm: item.model_arm,
    prediction_payload_sha256: item.prediction_payload_sha256,
  }));
  const semanticSha = sha256(Buffer.from(compactCanonical(semanticEntries)));
  assert(index.semantic_index_sha256 === semanticSha, "semantic index hash drift");
  assert(report.semantic_index_sha256 === semanticSha, "report semantic index drift");
  assert(report.source_commit === sourceCommit, "report source commit drift");
  assert(report.inputs.baseline_manifest_sha256 === manifestSha, "report manifest lineage drift");
  assert(report.inputs.model_input_bundle_sha256 === bundleSha, "report bundle lineage drift");
  assert(report.inputs.data_index_sha256 === bundle.inputs.data_index_sha256, "report data lineage drift");
  assert(report.inputs.runtime_lock_sha256 === bundle.inputs.runtime_lock_sha256, "report runtime lineage drift");
  assert(report.verdict.status === "supported" && report.counts.predictions === 120 && report.counts.failed_predictions === 0, "report status/count drift");
  assert(sameJson(report.isolation, {
    split: "validation",
    test_materialized: false,
    gt_future_read: false,
    executed_action_is_feature: false,
    trial_records_created: false,
    checkpoint_records_created: false,
  }), "report isolation drift");
  assert(sameJson(report.claim_boundary, manifest.claim_boundary), "report claim boundary drift");
  return { semantic_index_sha256: semanticSha };
});

await check("18-reverse-repeat", async () => {
  assert(repeatReport.execution_order === "reverse" && report.execution_order === "canonical", "repeat execution orders drift");
  assert(sameJson(repeatReport, { ...report, execution_order: "reverse" }), "repeat report drift");
  assert(repeatIndex.semantic_index_sha256 === index.semantic_index_sha256, "repeat semantic index mismatch");
  assert(Buffer.compare(repeatIndexBytes, await readFile(resolve(root, "index.json"))) === 0, "repeat index bytes differ");
  const repeatFiles = await listPredictionFiles(repeatRoot);
  assert(sameJson(repeatFiles, index.predictions.map((item) => item.uri).sort()), "repeat prediction file set drift");
  for (const item of index.predictions) {
    const canonicalBytes = await readFile(safePredictionPath(root, item.uri));
    const repeatBytes = await readFile(safePredictionPath(repeatRoot, item.uri));
    assert(Buffer.compare(canonicalBytes, repeatBytes) === 0, `repeat artifact bytes differ: ${item.uri}`);
    assert(sha256(repeatBytes) === item.sha256, `repeat artifact checksum drift: ${item.uri}`);
  }
  return {
    semantic_index_sha256: index.semantic_index_sha256,
    byte_identical_predictions: index.predictions.length,
  };
});

const supported = checks.every((item) => item.status === "supported");
const verification = {
  report_version: "0.1.0",
  report_kind: "objgauss.pr02c-deterministic-baseline-verification",
  verdict: supported ? "supported" : "invalid",
  source_commit: sourceCommit,
  counts: {
    groups: 12,
    branches: 60,
    arms: 2,
    predictions: 120,
    failed_predictions: 0,
    checks: checks.length,
  },
  data_index_sha256: bundle.inputs.data_index_sha256,
  semantic_index_sha256: index.semantic_index_sha256,
  provenance: {
    verifier_sha256: sha256(await readFile(VERIFIER_PATH)),
    baseline_manifest_sha256: sha256(await readFile(manifestPath)),
    model_input_bundle_sha256: sha256(await readFile(bundlePath)),
    loader_report_sha256: sha256(await readFile(loaderReportPath)),
  },
  checks,
  claim_boundary: CLAIM_BOUNDARY,
};
await writeFile(outputPath, compactCanonical(verification), "utf8");
console.log(JSON.stringify({
  verdict: verification.verdict,
  check_count: checks.length,
  predictions: verification.counts.predictions,
  semantic_index_sha256: verification.semantic_index_sha256,
  output: relative(REPO_ROOT, outputPath).split(sep).join("/"),
}));
if (!supported) process.exit(4);
