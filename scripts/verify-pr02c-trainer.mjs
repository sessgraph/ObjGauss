import { createHash } from "node:crypto";
import {
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { validateContract } from "../src/pr01/contract-dispatch.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const VERIFIER_PATH = fileURLToPath(import.meta.url);
const REPORT_VERSION = "0.1.0";
const DISPLAY_RESERVE = 1024 ** 3;
const TRAINING_CAP = 12 * 1024 ** 3;
const ARMS = ["action_conditioned", "action_free"];
const BRANCHES = [
  "hold",
  "push-neg-x-weak",
  "push-pos-x-strong",
  "push-pos-x-weak",
  "push-pos-y-weak",
];

function sha256Bytes(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

function sha256File(path) {
  return sha256Bytes(readFileSync(path));
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function stableBytes(value) {
  if (Array.isArray(value)) return `[${value.map(stableBytes).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${stableBytes(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function packageTreeSha256() {
  const paths = [
    resolve(ROOT, "learning/pyproject.toml"),
    ...readdirSync(resolve(ROOT, "learning"))
      .filter((name) => name.endsWith(".json"))
      .sort()
      .map((name) => resolve(ROOT, "learning", name)),
    ...readdirSync(resolve(ROOT, "learning/src/objgauss_learning"))
      .filter((name) => name.endsWith(".py"))
      .sort()
      .map((name) => resolve(ROOT, "learning/src/objgauss_learning", name)),
  ];
  const entries = Object.fromEntries(paths
    .map((path) => [relative(ROOT, path).replaceAll("\\", "/"), sha256File(path)])
    .sort(([left], [right]) => left.localeCompare(right)));
  return sha256Bytes(`${stableBytes(entries)}\n`);
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
    if (!value) throw new Error(`missing value for ${name}`);
    if (name === "--root") values.root = value;
    else if (name === "--repeat-root") values.repeatRoot = value;
    else if (name === "--manifest") values.manifest = value;
    else if (name === "--runtime-report") values.runtimeReport = value;
    else if (name === "--source-commit") values.sourceCommit = value;
    else if (name === "--checkpoint-auditor") values.checkpointAuditor = value;
    else if (name === "--output") values.output = value;
    else throw new Error(`unknown argument: ${name}`);
  }
  for (const name of [
    "root",
    "repeatRoot",
    "manifest",
    "runtimeReport",
    "sourceCommit",
    "checkpointAuditor",
    "output",
  ]) {
    if (!values[name]) throw new Error(`missing --${name.replace(/[A-Z]/g, (item) => `-${item.toLowerCase()}`)}`);
  }
  return values;
}

function repoPath(raw, prefix) {
  if (raw.startsWith("/")) throw new Error(`path must be repository-relative: ${raw}`);
  const path = resolve(ROOT, raw);
  if (!path.startsWith(`${ROOT}/`)) throw new Error(`path escapes repository root: ${raw}`);
  if (prefix && !raw.startsWith(`${prefix}/`)) {
    throw new Error(`path must be below ${prefix}/: ${raw}`);
  }
  return path;
}

function artifactPath(evidenceRoot, uri) {
  const prefix = "generated/pr02c/trainer/";
  if (typeof uri !== "string" || !uri.startsWith(prefix)) return null;
  const suffix = uri.slice(prefix.length);
  const path = resolve(evidenceRoot, suffix);
  return path.startsWith(`${evidenceRoot}/`) ? path : null;
}

function checkDescriptor(descriptor, evidenceRoot) {
  const path = artifactPath(evidenceRoot, descriptor?.uri);
  return path !== null
    && descriptor.byte_size === statSync(path).size
    && descriptor.sha256 === sha256File(path);
}

function loadArm(root, item) {
  const trial = readJson(resolve(root, "trials", `${item.trial_id}.json`));
  const attempt = readJson(resolve(root, "attempts", `${item.attempt_id}.json`));
  const checkpoint = readJson(resolve(root, "checkpoint-manifests", `${item.checkpoint_id}.json`));
  const predictions = item.predictions.map((entry) => ({
    entry,
    document: readJson(resolve(root, entry.uri)),
  }));
  return { item, trial, attempt, checkpoint, predictions };
}

function inspectCheckpoint(auditor, root, item) {
  const path = resolve(root, item.checkpoint_uri);
  const process = spawnSync(auditor, ["--checkpoint", path], {
    cwd: ROOT,
    encoding: "utf8",
    env: {
      ...processEnv(),
      OBJGAUSS_LEARNING_OFFLINE: "1",
      UV_OFFLINE: "1",
    },
  });
  if (process.status !== 0) return { verdict: "invalid", stderr: process.stderr };
  return JSON.parse(process.stdout);
}

function processEnv() {
  return Object.fromEntries(Object.entries(globalThis.process.env));
}

const args = parseArgs(process.argv.slice(2));
const root = repoPath(args.root, "generated/pr02c");
const repeatRoot = repoPath(args.repeatRoot, "generated/pr02c");
const manifestPath = repoPath(args.manifest, "learning");
const runtimeReportPath = repoPath(args.runtimeReport, "generated/pr02c");
const outputPath = repoPath(args.output, "generated/pr02c");
const checkpointAuditor = resolve(args.checkpointAuditor);
const manifest = readJson(manifestPath);
const runtime = readJson(runtimeReportPath);
const index = readJson(resolve(root, "index.json"));
const report = readJson(resolve(root, "report.json"));
const repeatIndex = readJson(resolve(repeatRoot, "index.json"));
const repeatReport = readJson(resolve(repeatRoot, "report.json"));
const checks = [];

function check(checkId, predicate) {
  checks.push({
    check_id: checkId,
    status: predicate ? "supported" : "invalid",
  });
}

check("01-report-kind-version",
  report.report_version === "0.1.0"
  && report.report_kind === "objgauss.pr02c-golden-training-report"
  && report.verdict?.status === "supported");
check("02-source-commit",
  /^[a-f0-9]{40}$/.test(args.sourceCommit)
  && report.source_commit === args.sourceCommit
  && repeatReport.source_commit === args.sourceCommit
  && index.source_commit === args.sourceCommit
  && repeatIndex.source_commit === args.sourceCommit);
check("03-manifest-lineage",
  manifest.manifest_kind === "objgauss.pr02c-golden-trainer"
  && index.trainer_manifest_sha256 === sha256File(manifestPath)
  && repeatIndex.trainer_manifest_sha256 === sha256File(manifestPath)
  && Object.entries(manifest.frozen_inputs)
    .filter(([, value]) => value.path)
    .every(([, value]) => sha256File(resolve(ROOT, value.path)) === value.sha256));
check("04-four-interval-rollout",
  stableBytes(manifest.rollout.boundaries_s) === stableBytes([0, 0.1, 0.2, 0.5, 1.1])
  && stableBytes(manifest.rollout.delta_t_s) === stableBytes([0.1, 0.1, 0.3, 0.6])
  && manifest.rollout.transition_parameters === "shared-across-all-intervals"
  && report.fairness?.transition_shared_across_intervals === true);
check("05-final-isolation",
  index.isolation?.test_materialized === false
  && index.isolation?.hpo_config_selected === false
  && index.isolation?.formal_checkpoint_frozen === false
  && repeatIndex.isolation?.test_materialized === false
  && report.isolation?.test_materialized === false
  && report.claim_boundary?.excluded_claims?.includes("test-source-or-prediction-produced"));
check("06-counts",
  report.counts?.arms === 2
  && report.counts?.train_groups === 1
  && report.counts?.validation_groups === 1
  && report.counts?.training_branches === 5
  && report.counts?.validation_branches === 5
  && report.counts?.predictions === 10
  && report.counts?.optimizer_updates_per_arm === 8);
check("07-execution-order",
  index.execution_order === "canonical"
  && report.execution_order === "canonical"
  && repeatIndex.execution_order === "reverse"
  && repeatReport.execution_order === "reverse");

const armNames = index.arms.map((item) => item.model_arm).sort();
const repeatArmNames = repeatIndex.arms.map((item) => item.model_arm).sort();
check("08-arm-set", stableBytes(armNames) === stableBytes(ARMS) && stableBytes(repeatArmNames) === stableBytes(ARMS));

const arms = index.arms.map((item) => loadArm(root, item));
const repeatArms = repeatIndex.arms.map((item) => loadArm(repeatRoot, item));
const armByName = new Map(arms.map((item) => [item.item.model_arm, item]));
const repeatByName = new Map(repeatArms.map((item) => [item.item.model_arm, item]));

check("09-parameter-parity",
  new Set(index.arms.map((item) => item.parameter_count)).size === 1
  && new Set(index.arms.map((item) => item.architecture_sha256)).size === 1
  && index.arms.every((item) => item.parameter_count > 0)
  && report.fairness?.parameter_count_equal === true);
check("10-update-parity",
  index.arms.every((item) => item.optimizer_updates === 8 && item.epochs === 8)
  && new Set(index.arms.map((item) => item.optimizer_updates)).size === 1
  && report.fairness?.optimizer_updates_equal === true);
check("11-data-order-parity",
  new Set(index.arms.map((item) => item.data_order_sha256)).size === 1
  && report.fairness?.data_order_equal === true);
check("12-grid-seed-parity",
  report.fairness?.grid_equal === true
  && report.fairness?.seed_equal === true
  && arms.every(({ trial }) => trial.identity.training_seed === 2026071501)
  && arms.every(({ trial }) => trial.configuration.hyperparameter_grid_sha256
    === manifest.frozen_inputs.hyperparameter_grid.sha256));

const publicDocuments = arms.flatMap(({ trial, attempt, checkpoint, predictions }) => [
  trial,
  attempt,
  checkpoint,
  ...predictions.map((item) => item.document),
]);
const contractResults = publicDocuments.map((document) => validateContract(document));
check("13-public-contracts", contractResults.length === 16 && contractResults.every((item) => item.valid));

check("14-trial-attempt-checkpoint-lineage", arms.every(({ item, trial, attempt, checkpoint }) => (
  trial.identity.trial_id === item.trial_id
  && trial.identity.model_arm === item.model_arm
  && trial.selection.selected === false
  && trial.attempt_ids.length === 1
  && trial.attempt_ids[0] === item.attempt_id
  && attempt.identity.trial_id === item.trial_id
  && attempt.identity.attempt_id === item.attempt_id
  && attempt.identity.ordinal === 1
  && attempt.retry.eligible === false
  && checkpoint.identity.trial_id === item.trial_id
  && checkpoint.identity.checkpoint_id === item.checkpoint_id
  && checkpoint.configuration.parameter_count === item.parameter_count
  && checkpoint.compatibility.runtime_lock_sha256 === manifest.frozen_inputs.runtime_lock.sha256
)));

check("15-artifact-checksums", arms.every(({ item, attempt, checkpoint }) => (
  checkDescriptor(attempt.outputs.training_log.value, root)
  && checkDescriptor(attempt.outputs.checkpoint.value, root)
  && checkDescriptor(checkpoint.payload, root)
  && checkpoint.payload.sha256 === item.checkpoint_sha256
  && sha256File(resolve(root, item.training_log_uri)) === attempt.outputs.training_log.value.sha256
)));

const sourceTree = packageTreeSha256();
check("16-provenance", publicDocuments.every((document) => (
  document.provenance.source_commit === args.sourceCommit
  && document.provenance.source_tree_sha256 === sourceTree
  && document.provenance.experiment_spec_sha256 === sha256File(
    resolve(ROOT, manifest.frozen_inputs.dynamics_experiment.runtime_path)
  )
  && document.provenance.runtime_lock_sha256 === manifest.frozen_inputs.runtime_lock.sha256
)));

check("17-prediction-coverage-isolation", arms.every(({ item, predictions }) => (
  predictions.length === 5
  && stableBytes(predictions.map(({ entry }) => entry.branch_id).sort()) === stableBytes(BRANCHES)
  && predictions.every(({ entry, document }) => (
    entry.model_arm === item.model_arm
    && entry.sha256 === sha256File(resolve(root, entry.uri))
    && document.identity.split === "validation"
    && document.identity.model_arm === item.model_arm
    && document.identity.trial_id.value === item.trial_id
    && document.identity.checkpoint_id.value === item.checkpoint_id
    && document.inputs.executed_action_is_feature === false
    && document.inputs.gt_future_read === false
  ))
)));

const checkpointAudits = new Map();
const repeatCheckpointAudits = new Map();
for (const item of index.arms) checkpointAudits.set(item.model_arm, inspectCheckpoint(checkpointAuditor, root, item));
for (const item of repeatIndex.arms) {
  repeatCheckpointAudits.set(item.model_arm, inspectCheckpoint(checkpointAuditor, repeatRoot, item));
}
check("18-checkpoint-semantic-audit", ARMS.every((arm) => {
  const item = armByName.get(arm).item;
  const repeatItem = repeatByName.get(arm).item;
  const observed = checkpointAudits.get(arm);
  const repeatObserved = repeatCheckpointAudits.get(arm);
  return observed.verdict === "supported"
    && repeatObserved.verdict === "supported"
    && observed.semantic_sha256 === item.tensor_state_semantic_sha256
    && repeatObserved.semantic_sha256 === repeatItem.tensor_state_semantic_sha256
    && observed.parameter_count === item.parameter_count
    && repeatObserved.parameter_count === repeatItem.parameter_count;
}));
check("19-checkpoint-structure-parity",
  new Set([...checkpointAudits.values()].map((item) => item.parameter_structure_sha256)).size === 1
  && new Set([...repeatCheckpointAudits.values()].map((item) => item.parameter_structure_sha256)).size === 1);

check("20-semantic-repeat",
  index.semantic_index_sha256 === repeatIndex.semantic_index_sha256
  && report.semantic_index_sha256 === index.semantic_index_sha256
  && repeatReport.semantic_index_sha256 === repeatIndex.semantic_index_sha256
  && ARMS.every((arm) => {
    const first = armByName.get(arm).item;
    const second = repeatByName.get(arm).item;
    return first.tensor_state_semantic_sha256 === second.tensor_state_semantic_sha256
      && first.validation_prediction_semantic_sha256 === second.validation_prediction_semantic_sha256
      && first.optimizer_updates === second.optimizer_updates
      && first.data_order_sha256 === second.data_order_sha256;
  }));
check("21-prediction-byte-repeat", ARMS.every((arm) => {
  const first = armByName.get(arm).item.predictions;
  const second = repeatByName.get(arm).item.predictions;
  return first.length === second.length && first.every((item, indexValue) => (
    item.prediction_payload_sha256 === second[indexValue].prediction_payload_sha256
    && sha256File(resolve(root, item.uri)) === sha256File(resolve(repeatRoot, second[indexValue].uri))
  ));
}));
check("22-training-log-repeat", ARMS.every((arm) => (
  sha256File(resolve(root, armByName.get(arm).item.training_log_uri))
  === sha256File(resolve(repeatRoot, repeatByName.get(arm).item.training_log_uri))
)));

check("23-resource-display-reserve",
  runtime.verdict?.status === "supported"
  && runtime.gpu_probe?.minimum_free_bytes >= DISPLAY_RESERVE
  && report.resources?.minimum_display_vram_free_bytes >= DISPLAY_RESERVE
  && repeatReport.resources?.minimum_display_vram_free_bytes >= DISPLAY_RESERVE
  && report.resources?.peak_vram_bytes <= TRAINING_CAP
  && repeatReport.resources?.peak_vram_bytes <= TRAINING_CAP
  && report.resources?.charged_to_hpo === false
  && report.resources?.charged_to_formal === false);
check("24-claim-boundary",
  manifest.claim_boundary?.supported_claim
    === "minimal-learned-arms-train-reproducibly-with-fair-golden-lineage"
  && manifest.claim_boundary?.excluded_claims?.includes("hpo-config-selected")
  && manifest.claim_boundary?.excluded_claims?.includes("formal-checkpoint-frozen")
  && manifest.claim_boundary?.excluded_claims?.includes("learned-model-performance")
  && manifest.claim_boundary?.excluded_claims?.includes("scientific-baseline-comparison"));

const supported = checks.every((item) => item.status === "supported");
const verification = {
  report_version: REPORT_VERSION,
  report_kind: "objgauss.pr02c-golden-training-verification",
  verdict: supported ? "supported" : "invalid",
  source_commit: args.sourceCommit,
  counts: {
    arms: index.arms.length,
    predictions: index.arms.reduce((total, item) => total + item.predictions.length, 0),
    checks: checks.length,
  },
  semantic_index_sha256: index.semantic_index_sha256,
  verifier_sha256: sha256File(VERIFIER_PATH),
  checks,
  claim_boundary: "C3 tiny/golden reproducibility, fairness, lineage, isolation, and resources only",
};
atomicJson(outputPath, verification);
process.stdout.write(`${JSON.stringify({
  verdict: verification.verdict,
  check_count: checks.length,
  predictions: verification.counts.predictions,
  semantic_index_sha256: verification.semantic_index_sha256,
  output: args.output,
})}\n`);
if (!supported) process.exit(4);
