import { spawnSync } from "node:child_process";
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
import { validateContract } from "../src/pr01/contract-dispatch.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const VERIFIER = fileURLToPath(import.meta.url);
const ARMS = ["action_conditioned", "action_free"];
const DISPLAY_RESERVE = 1024 ** 3;
const TRAINING_CAP = 12 * 1024 ** 3;

function sha256Bytes(value) {
  return createHash("sha256").update(value).digest("hex");
}

function sha256File(path) {
  return sha256Bytes(readFileSync(path));
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stableValue(value[key])]));
  }
  return value;
}

function canonicalSha256(value) {
  return sha256Bytes(`${JSON.stringify(stableValue(value), null, 2)}\n`);
}

function stableSha256(value) {
  return sha256Bytes(`${JSON.stringify(stableValue(value))}\n`);
}

function fsum(values) {
  const partials = [];
  for (const raw of values) {
    let value = raw;
    let index = 0;
    for (const partialRaw of partials) {
      let partial = partialRaw;
      if (Math.abs(value) < Math.abs(partial)) [value, partial] = [partial, value];
      const high = value + partial;
      const low = partial - (high - value);
      if (low !== 0) {
        partials[index] = low;
        index += 1;
      }
      value = high;
    }
    partials.length = index;
    partials.push(value);
  }
  return partials.reduceRight((sum, value) => sum + value, 0);
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

function parseArgs(values) {
  const result = {};
  for (let index = 0; index < values.length; index += 2) {
    const name = values[index];
    const value = values[index + 1];
    if (!value) throw new Error(`missing value for ${name}`);
    if (name === "--root") result.root = value;
    else if (name === "--repeat-root") result.repeatRoot = value;
    else if (name === "--manifest") result.manifest = value;
    else if (name === "--source-commit") result.sourceCommit = value;
    else if (name === "--checkpoint-auditor") result.checkpointAuditor = value;
    else if (name === "--output") result.output = value;
    else throw new Error(`unknown argument: ${name}`);
  }
  for (const name of ["root", "repeatRoot", "manifest", "sourceCommit", "checkpointAuditor", "output"]) {
    if (!result[name]) throw new Error(`missing required verifier argument: ${name}`);
  }
  return result;
}

function repoPath(raw, prefix) {
  if (raw.startsWith("/")) throw new Error(`path must be repository-relative: ${raw}`);
  const path = resolve(ROOT, raw);
  if (!path.startsWith(`${ROOT}/`)) throw new Error(`path escapes repository root: ${raw}`);
  if (prefix && !raw.startsWith(`${prefix}/`)) throw new Error(`path must be below ${prefix}: ${raw}`);
  return path;
}

function artifactPath(root, uri) {
  if (typeof uri !== "string" || uri.startsWith("/")) return null;
  const path = resolve(root, uri);
  return path.startsWith(`${root}/`) ? path : null;
}

function walkFiles(root, current = root) {
  return readdirSync(current, { withFileTypes: true })
    .flatMap((entry) => {
      const path = resolve(current, entry.name);
      return entry.isDirectory() ? walkFiles(root, path) : [relative(root, path).replaceAll("\\", "/")];
    })
    .sort();
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
  return stableSha256(entries);
}

function checkpointAudit(executable, path) {
  const child = spawnSync(executable, ["--checkpoint", path], {
    cwd: ROOT,
    encoding: "utf8",
    env: { ...process.env, OBJGAUSS_LEARNING_OFFLINE: "1", UV_OFFLINE: "1" },
  });
  if (child.status !== 0) return { verdict: "invalid", stderr: child.stderr };
  return JSON.parse(child.stdout);
}

const args = parseArgs(process.argv.slice(2));
const evidenceRoot = repoPath(args.root, "generated/pr02c");
const repeatRoot = repoPath(args.repeatRoot, "generated/pr02c");
const manifestPath = repoPath(args.manifest, "learning");
const outputPath = repoPath(args.output, "generated/pr02c");
const manifest = readJson(manifestPath);
const taskIndex = readJson(resolve(evidenceRoot, "task-index.json"));
const dataIndex = readJson(resolve(evidenceRoot, "hpo-data-index.json"));
const pairLedger = readJson(resolve(evidenceRoot, "fairness-pair-ledger.json"));
const selected = readJson(resolve(evidenceRoot, "selected-configs.json"));
const selection = readJson(resolve(evidenceRoot, "selection-report.json"));
const repeatSelected = readJson(resolve(repeatRoot, "selected-configs.json"));
const repeatSelection = readJson(resolve(repeatRoot, "selection-report.json"));
const checks = [];

function check(id, predicate) {
  checks.push({ check_id: id, status: predicate ? "supported" : "invalid" });
}

check("01-manifest-kind-matrix",
  manifest.manifest_version === "0.1.0"
  && manifest.manifest_kind === "objgauss.pr02c-c6-hpo-selection"
  && manifest.matrix?.task_count === 24
  && manifest.matrix?.pair_count === 12
  && manifest.matrix?.configurations?.length === 4
  && manifest.matrix?.training_seeds?.length === 3);

check("02-frozen-inputs", Object.values(manifest.lineage?.frozen_inputs ?? {}).every((entry) => (
  sha256File(resolve(ROOT, entry.path)) === entry.sha256
)) && manifest.matrix.configurations.every((entry) => canonicalSha256(entry.values) === entry.config_sha256));

check("03-runner-lineage",
  /^[a-f0-9]{40}$/.test(args.sourceCommit)
  && taskIndex.runner_commit === args.sourceCommit
  && taskIndex.workflow_commit === args.sourceCommit
  && taskIndex.hpo_data_build_commit === args.sourceCommit
  && taskIndex.trainer_contract_commit === manifest.lineage.trainer_contract_commit
  && taskIndex.hpo_manifest_sha256 === sha256File(manifestPath));

check("04-data-build-once",
  dataIndex.runner_commit === args.sourceCommit
  && dataIndex.build_count === 1
  && dataIndex.shared_by_all_tasks === true
  && dataIndex.group_counts?.train === 48
  && dataIndex.group_counts?.validation === 12
  && dataIndex.branch_count === 300
  && sha256File(resolve(evidenceRoot, "hpo-data-index.json")) === taskIndex.hpo_data_index_sha256);

check("05-final-test-isolation",
  JSON.stringify(taskIndex.splits) === JSON.stringify(["train", "validation"])
  && taskIndex.test_materialized === false
  && dataIndex.test_materialized === false
  && selected.test_visible === false
  && manifest.selector?.forbidden_split === "test"
  && manifest.selector?.performance_promotion_threshold === null);

const expectedTasks = new Map();
for (const pair of manifest.matrix.fairness_pairs) {
  for (const [arm, taskId] of Object.entries(pair.task_ids)) {
    expectedTasks.set(taskId, {
      pair_id: pair.pair_id,
      config_id: pair.config_id,
      training_seed: pair.training_seed,
      model_arm: arm,
    });
  }
}
const observedTaskIds = taskIndex.tasks.map((task) => task.task_id);
check("06-exact-task-set",
  taskIndex.tasks.length === 24
  && new Set(observedTaskIds).size === 24
  && observedTaskIds.every((taskId) => expectedTasks.has(taskId))
  && [...expectedTasks].every(([taskId, identity]) => {
    const task = taskIndex.tasks.find((item) => item.task_id === taskId);
    return Object.entries(identity).every(([name, value]) => task?.[name] === value);
  }));

check("07-attempt-lineage", taskIndex.tasks.every((task) => (
  task.status === "completed"
  && [1, 2].includes(task.attempt_ids?.length)
  && task.attempt_ids.every((attemptId, index) => (
    attemptId === `attempt-${task.task_id}-a${String(index + 1).padStart(2, "0")}`
  ))
  && task.hpo_data_index_sha256 === taskIndex.hpo_data_index_sha256
)));

const artifactChecks = [];
const documents = [];
const checkpoints = [];
const logs = new Map();
for (const task of taskIndex.tasks) {
  for (const artifact of task.artifacts ?? []) {
    const path = artifactPath(evidenceRoot, artifact.uri);
    artifactChecks.push(path !== null && statSync(path).isFile() && sha256File(path) === artifact.sha256);
    if (path?.endsWith(".json")) {
      const document = readJson(path);
      if (document.contract_kind) documents.push({ task, document, path });
      if (document.log_kind === "objgauss.pr02c-c6-hpo-training-log") logs.set(task.task_id, document);
    } else if (path?.endsWith(".pt")) {
      checkpoints.push({ task, path });
    }
  }
}
check("08-artifact-checksums", artifactChecks.length > 0 && artifactChecks.every(Boolean));

const contractChecks = documents.map(({ document }) => validateContract(document));
check("09-public-contracts",
  documents.length === 24 * 61
    + taskIndex.tasks.reduce((count, task) => count + task.attempt_ids.length, 0)
  && contractChecks.every((result) => result.valid));

const predictions = documents.filter(({ document }) => document.contract_kind === "objgauss.dynamics_prediction");
check("10-validation-predictions",
  predictions.length === 24 * 60
  && predictions.every(({ task, document }) => (
    document.identity.split === "validation"
    && document.identity.model_arm === task.model_arm
    && document.inputs.executed_action_is_feature === false
    && document.inputs.gt_future_read === false
    && document.predictions.length === 4
  )));

const sourceTree = packageTreeSha256();
check("11-public-lineage", documents.every(({ document }) => (
  document.provenance?.source_commit === args.sourceCommit
  && document.provenance?.source_tree_sha256 === sourceTree
  && document.provenance?.runtime_lock_sha256 === manifest.lineage.frozen_inputs.runtime_lock.sha256
)));

check("12-group-first-scores", taskIndex.tasks.every((task) => (
  task.validation_group_errors?.length === 12
  && new Set(task.validation_group_errors.map((item) => item.group_id)).size === 12
  && task.validation_primary_error
    === fsum(task.validation_group_errors.map((item) => item.primary_error)) / 12
)));

check("13-fairness-pair-ledger",
  pairLedger.ledger_kind === "objgauss.pr02c-c6-fairness-pair-ledger"
  && pairLedger.pair_count === 12
  && pairLedger.pairs?.length === 12
  && pairLedger.pairs.every((pair) => {
    const members = taskIndex.tasks.filter((task) => task.pair_id === pair.pair_id);
    if (members.length !== 2) return false;
    const fields = [
      "initialization_seed",
      "initialization_algorithm",
      "common_parameter_names_sha256",
      "common_parameter_subtree_sha256",
      "data_order_sha256",
      "batch_group_sequence_sha256",
      "optimizer_updates",
      "epochs",
      "training_budget_sha256",
      "checkpoint_policy",
    ];
    return fields.every((name) => members[0].fairness[name] === members[1].fairness[name]);
  }));

check("14-fairness-training-logs", taskIndex.tasks.every((task) => {
  const log = logs.get(task.task_id);
  return log
    && log.training_seed === task.training_seed
    && log.optimizer_updates === task.fairness.optimizer_updates
    && log.epochs === task.fairness.epochs
    && log.data_order_sha256 === task.fairness.data_order_sha256
    && log.batch_group_sequence_sha256 === task.fairness.batch_group_sequence_sha256
    && log.visibility?.test_materialized === false;
}));

const checkpointResults = checkpoints.map(({ task, path }) => ({
  task,
  result: checkpointAudit(resolve(args.checkpointAuditor), path),
}));
check("15-checkpoint-semantic-audit",
  checkpointResults.length === 24
  && checkpointResults.every(({ task, result }) => (
    result.verdict === "supported"
    && result.semantic_sha256 === logs.get(task.task_id)?.tensor_state_semantic_sha256
  )));

check("16-selector-output",
  selected.selection_kind === "objgauss.pr02c-c6-selected-configs"
  && selected.verdict === "supported"
  && JSON.stringify(Object.keys(selected.mapping).sort()) === JSON.stringify(ARMS)
  && selection.verdict?.status === "supported"
  && selection.counts?.tasks === 24
  && selection.counts?.fairness_pairs === 12
  && selection.selection_semantic_sha256 === selected.selection_semantic_sha256);

check("17-selector-order-repeat",
  stableSha256(selected) === stableSha256(repeatSelected)
  && selection.selection_semantic_sha256 === repeatSelection.selection_semantic_sha256
  && selection.selected_configs_sha256 === repeatSelection.selected_configs_sha256);

check("18-resource-budget",
  taskIndex.resources?.gpu_hours <= 6.3
  && taskIndex.resources?.peak_vram_bytes <= TRAINING_CAP
  && taskIndex.resources?.minimum_display_vram_free_bytes >= DISPLAY_RESERVE
  && taskIndex.tasks.every((task) => task.resources?.gpu_hours <= 0.25));

const checkpointManifests = documents
  .filter(({ document }) => document.contract_kind === "objgauss.checkpoint_manifest");
const attemptDocuments = documents
  .filter(({ document }) => document.contract_kind === "objgauss.training_attempt");
const attemptById = new Map(attemptDocuments.map(({ document }) => [
  document.identity.attempt_id,
  document,
]));
const retryableReasons = new Set(["process_crash", "io_failure", "transient_oom"]);
check("19-checkpoint-not-promoted",
  checkpointManifests.length === 0
  && attemptDocuments.length
    === taskIndex.tasks.reduce((count, task) => count + task.attempt_ids.length, 0)
  && taskIndex.tasks.every((task) => (
    task.checkpoint_manifest_published === false
    && task.checkpoint_sha256?.length === 64
    && task.checkpoint_semantic_sha256 === logs.get(task.task_id)?.tensor_state_semantic_sha256
    && task.attempt_ids.every((attemptId, index) => {
      const attempt = attemptById.get(attemptId);
      const previous = index === 0
        ? { availability: "missing", reason: "not_applicable" }
        : { availability: "present", value: task.attempt_ids[index - 1] };
      const isFinal = index === task.attempt_ids.length - 1;
      return attempt?.identity?.ordinal === index + 1
        && attempt.identity.model_arm === task.model_arm
        && attempt.identity.training_seed === task.training_seed
        && attempt.identity.config_sha256 === task.config_sha256
        && JSON.stringify(attempt.retry?.previous_attempt_id) === JSON.stringify(previous)
        && (isFinal
          ? attempt.outcome?.status === "succeeded"
            && attempt.retry?.eligible === false
            && attempt.outputs?.checkpoint?.availability === "present"
            && attempt.outputs.checkpoint.value.sha256 === task.checkpoint_sha256
          : attempt.outcome?.status === "failed"
            && attempt.outcome?.classification === "infrastructure"
            && retryableReasons.has(attempt.outcome?.reason_code)
            && attempt.retry?.eligible === true
            && attempt.outputs?.checkpoint?.availability === "missing");
    })
  )));

check("20-claim-boundary",
  manifest.claim_boundary?.supported_claim
    === "a-deterministic-test-isolated-config-mapping-is-frozen-for-both-learned-arms"
  && manifest.claim_boundary?.excluded_claims?.includes("formal-training-complete")
  && manifest.claim_boundary?.excluded_claims?.includes("learned-model-performance")
  && selected.formal_checkpoint_frozen === false);

const supported = checks.every((item) => item.status === "supported");
const verification = {
  report_version: "0.1.0",
  report_kind: "objgauss.pr02c-c6-hpo-verification",
  verdict: supported ? "supported" : "invalid",
  runner_commit: args.sourceCommit,
  counts: { tasks: taskIndex.tasks.length, fairness_pairs: pairLedger.pairs.length, checks: checks.length },
  hpo_data_index_sha256: taskIndex.hpo_data_index_sha256,
  selection_semantic_sha256: selected.selection_semantic_sha256,
  selector_repeat: {
    canonical_selected_configs_sha256: sha256File(resolve(evidenceRoot, "selected-configs.json")),
    reverse_selected_configs_sha256: sha256File(resolve(repeatRoot, "selected-configs.json")),
    canonical_selection_report_sha256: sha256File(resolve(evidenceRoot, "selection-report.json")),
    reverse_selection_report_sha256: sha256File(resolve(repeatRoot, "selection-report.json")),
  },
  verifier_sha256: sha256File(VERIFIER),
  checks,
  claim_boundary: "C6 deterministic HPO config mapping only; no formal or final-test claim",
};
atomicJson(outputPath, verification);

const checksumPath = resolve(evidenceRoot, "checksum-index.json");
if (checksumPath !== outputPath) {
  rmSync(checksumPath, { force: true });
  const entries = walkFiles(evidenceRoot)
    .filter((uri) => uri !== "checksum-index.json")
    .map((uri) => ({ uri, sha256: sha256File(resolve(evidenceRoot, uri)) }));
  atomicJson(checksumPath, {
    index_version: "0.1.0",
    index_kind: "objgauss.pr02c-c6-checksum-index",
    entries,
  });
}

process.stdout.write(`${JSON.stringify({
  verdict: verification.verdict,
  checks: checks.length,
  tasks: taskIndex.tasks.length,
  selection_semantic_sha256: selected.selection_semantic_sha256,
})}\n`);
if (!supported) process.exit(4);
