import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { createContractDispatcher } from "../contract-dispatch.mjs";

const SOURCE_PATH = fileURLToPath(import.meta.url);
const ROOT = resolve(dirname(SOURCE_PATH), "../../..");
const REQUIRED_FILES = [
  "attempt.json",
  "contact-ledger.json",
  "episode.json",
  "publication.json",
  "trajectory.json",
];
const PRIORITY = { supported: 0, blocked: 1, rejected: 2, invalid: 3 };
const VERDICT_REASON = {
  supported: "all_hard_gates_passed",
  rejected: "scientific_gate_failed",
  blocked: "evidence_incomplete",
  invalid: "structural_evidence_invalid",
};
const REQUIRED_NEGATIVE_CASES = [
  "snapshot_changed",
  "rng_changed",
  "physics_changed",
  "cross_split_leakage",
  "action_missing_or_duplicate",
  "ledger_missing",
  "contact_tampered",
  "checksum_mismatch",
  "lineage_broken",
  "attempt_timeout",
  "non_finite_value",
];

export function sha256(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

export function stableJson(value) {
  if (value === null || typeof value !== "object") {
    if (typeof value === "number" && !Number.isFinite(value)) {
      throw new TypeError("non-finite JSON value");
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(",")}]`;
  }
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
}

function digestDocument(value) {
  return sha256(Buffer.from(`${stableJson(value)}\n`));
}

function equal(left, right) {
  return stableJson(left) === stableJson(right);
}

function finite(value) {
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(finite);
  if (value !== null && typeof value === "object") return Object.values(value).every(finite);
  return true;
}

class AuditFailure extends Error {
  constructor(status, reasonCode, message, details = {}) {
    super(message);
    this.name = "AuditFailure";
    this.status = status;
    this.reasonCode = reasonCode;
    this.details = details;
  }
}

function fail(status, reasonCode, message, details) {
  throw new AuditFailure(status, reasonCode, message, details);
}

async function jsonFile(path) {
  const bytes = await readFile(path);
  const text = bytes.toString("utf8");
  if (/(?:^|[\s[:,])(?:NaN|Infinity|-Infinity)(?=$|[\s,}\]])/.test(text)) {
    fail("invalid", "non_finite_value", `non-finite JSON token in ${path}`);
  }
  let document;
  try {
    document = JSON.parse(text);
  } catch (error) {
    fail("invalid", "schema_valid", `invalid JSON in ${path}: ${error.message}`);
  }
  if (!finite(document)) {
    fail("invalid", "non_finite_value", `non-finite value in ${path}`);
  }
  return { bytes, document };
}

function safeRelative(root, uri) {
  if (typeof uri !== "string" || uri.length === 0 || uri.startsWith("/") || uri.includes("\\")) {
    return false;
  }
  const target = resolve(root, uri);
  return target === root || target.startsWith(`${root}${sep}`);
}

async function directories(path) {
  try {
    return (await readdir(path, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
      .map((entry) => entry.name)
      .sort();
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
}

async function files(path) {
  try {
    return (await readdir(path, { withFileTypes: true }))
      .filter((entry) => entry.isFile() && !entry.name.startsWith("."))
      .map((entry) => entry.name)
      .sort();
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
}

async function loadFailedAttempts(root, validate) {
  const attempts = [];
  const base = resolve(root, "attempts");
  async function walk(path) {
    let entries;
    try {
      entries = await readdir(path, { withFileTypes: true });
    } catch (error) {
      if (error.code === "ENOENT") return;
      throw error;
    }
    for (const entry of entries) {
      const child = resolve(path, entry.name);
      if (entry.isDirectory()) await walk(child);
      if (entry.isFile() && entry.name.endsWith(".json")) {
        const loaded = await jsonFile(child);
        const result = validate(loaded.document);
        if (!result.valid || loaded.document.contract_kind !== "objgauss.attempt") {
          fail("invalid", "schema_valid", `failed attempt is contract-invalid: ${child}`);
        }
        attempts.push({ path: child, ...loaded });
      }
    }
  }
  await walk(base);
  return attempts.sort((a, b) => a.path.localeCompare(b.path));
}

async function loadDataset(root, manifest, validate) {
  const experimentRoot = resolve(root, "dataset", manifest.identity.experiment_id);
  const groups = [];
  for (const groupId of await directories(experimentRoot)) {
    const groupPath = resolve(experimentRoot, groupId);
    const branches = [];
    for (const branchId of await directories(groupPath)) {
      const branchPath = resolve(groupPath, branchId);
      const names = await files(branchPath);
      if (!names.includes("attempt.json")) {
        fail("invalid", "ledger_missing", `success attempt ledger is missing for ${groupId}/${branchId}`);
      }
      if (!equal(names, REQUIRED_FILES)) {
        fail("invalid", "schema_valid", `unexpected branch file set for ${groupId}/${branchId}: ${names}`);
      }
      const loaded = {};
      for (const name of REQUIRED_FILES) loaded[name] = await jsonFile(resolve(branchPath, name));
      for (const name of ["episode.json", "attempt.json"]) {
        const result = validate(loaded[name].document);
        if (!result.valid) {
          const semanticReasons = result.semantic_errors.map((item) => item.reason_code);
          if (semanticReasons.some((reason) => reason.includes("lineage") || reason.includes("snapshot"))) {
            fail("invalid", "lineage_broken", `${name} lineage semantics are invalid`, { result });
          }
          fail("invalid", "schema_valid", `${name} is contract-invalid`, { result });
        }
      }
      branches.push({ groupId, branchId, path: branchPath, loaded });
    }
    groups.push({ groupId, path: groupPath, branches });
  }
  return { experimentRoot, groups, failedAttempts: await loadFailedAttempts(root, validate) };
}

function branchDocument(branch, name) {
  return branch.loaded[name].document;
}

function auditPaths(root, dataset) {
  for (const group of dataset.groups) {
    for (const branch of group.branches) {
      const episode = branchDocument(branch, "episode.json");
      const expectedDirectory = relative(root, branch.path).replaceAll("\\", "/");
      const expected = {
        trajectory: `${expectedDirectory}/trajectory.json`,
        contact_ledger: `${expectedDirectory}/contact-ledger.json`,
      };
      for (const [key, uri] of Object.entries(expected)) {
        const actual = episode.evidence[key].uri;
        if (!safeRelative(root, actual) || actual !== uri) {
          fail("invalid", "path_safe", `unsafe or non-canonical ${key} path: ${actual}`);
        }
      }
      const attemptUri = branchDocument(branch, "attempt.json").publication.episode_artifact.value.uri;
      if (!safeRelative(root, attemptUri) || attemptUri !== `${expectedDirectory}/episode.json`) {
        fail("invalid", "path_safe", `unsafe or non-canonical attempt episode path: ${attemptUri}`);
      }
    }
  }
}

function auditChecksums(dataset) {
  for (const group of dataset.groups) {
    for (const branch of group.branches) {
      const episode = branchDocument(branch, "episode.json");
      const attempt = branchDocument(branch, "attempt.json");
      const publication = branchDocument(branch, "publication.json");
      for (const [key, filename] of [["trajectory", "trajectory.json"], ["contact_ledger", "contact-ledger.json"]]) {
        const descriptor = episode.evidence[key];
        const artifact = branch.loaded[filename];
        if (
          descriptor.sha256 !== sha256(artifact.bytes)
          || descriptor.byte_length !== artifact.bytes.length
          || descriptor.record_count !== artifact.document.records?.length
        ) {
          fail("invalid", "checksum_mismatch", `${key} descriptor differs for ${group.groupId}/${branch.branchId}`);
        }
      }
      const hashes = {
        episode_sha256: sha256(branch.loaded["episode.json"].bytes),
        trajectory_sha256: sha256(branch.loaded["trajectory.json"].bytes),
        contact_ledger_sha256: sha256(branch.loaded["contact-ledger.json"].bytes),
      };
      const semantic = digestDocument(hashes);
      if (
        publication.episode_sha256 !== hashes.episode_sha256
        || publication.trajectory_sha256 !== hashes.trajectory_sha256
        || publication.contact_ledger_sha256 !== hashes.contact_ledger_sha256
        || publication.attempt_sha256 !== sha256(branch.loaded["attempt.json"].bytes)
        || publication.semantic_sha256 !== semantic
        || attempt.publication.episode_artifact.value.sha256 !== hashes.episode_sha256
      ) {
        fail("invalid", "checksum_mismatch", `publication checksum chain differs for ${group.groupId}/${branch.branchId}`);
      }
    }
  }
}

function auditIds(dataset, manifest) {
  const episodeIds = new Set();
  const attemptIds = new Set();
  for (const group of dataset.groups) {
    for (const branch of group.branches) {
      const episode = branchDocument(branch, "episode.json");
      const attempt = branchDocument(branch, "attempt.json");
      if (
        episode.identity.experiment_id !== manifest.identity.experiment_id
        || episode.identity.group_id !== group.groupId
        || episode.identity.branch_id !== branch.branchId
        || attempt.identity.experiment_id !== manifest.identity.experiment_id
        || attempt.identity.group_id !== group.groupId
        || attempt.identity.branch_id !== branch.branchId
        || attempt.identity.attempt_id !== episode.identity.attempt_id
        || attempt.identity.split !== episode.identity.split
      ) {
        fail("invalid", "ids_unique", `identity/path mismatch for ${group.groupId}/${branch.branchId}`);
      }
      if (episodeIds.has(episode.identity.episode_id) || attemptIds.has(attempt.identity.attempt_id)) {
        fail("invalid", "ids_unique", `duplicate episode or attempt identity in ${group.groupId}`);
      }
      episodeIds.add(episode.identity.episode_id);
      attemptIds.add(attempt.identity.attempt_id);
    }
  }
  for (const item of dataset.failedAttempts) {
    if (attemptIds.has(item.document.identity.attempt_id)) {
      fail("invalid", "ids_unique", `duplicate failed attempt identity ${item.document.identity.attempt_id}`);
    }
    attemptIds.add(item.document.identity.attempt_id);
  }
}

function auditLineage(dataset, manifest, manifestSha256) {
  let invariant = null;
  for (const group of dataset.groups) {
    for (const branch of group.branches) {
      const episode = branchDocument(branch, "episode.json");
      const attempt = branchDocument(branch, "attempt.json");
      const current = {
        source_gate_report_sha256: episode.provenance.source_gate_report_sha256,
        source_commit: episode.provenance.source_commit,
        source_tree_sha256: episode.provenance.source_tree_sha256,
        config_sha256: episode.provenance.config_sha256,
        runtime_lock_sha256: episode.provenance.runtime_lock_sha256,
        asset_manifest_sha256: episode.provenance.asset_manifest_sha256,
      };
      if (
        episode.provenance.parent_snapshot_id !== episode.initialization.snapshot_id
        || episode.provenance.config_sha256 !== manifest.identity.experiment_spec_sha256
        || episode.provenance.runtime_lock_sha256 !== manifest.runtime.lock_sha256
        || episode.provenance.source_gate_report_sha256 !== manifest.source_gate.report_sha256
        || attempt.provenance.experiment_manifest_sha256 !== manifestSha256
        || attempt.provenance.config_sha256 !== episode.provenance.config_sha256
        || attempt.provenance.runtime_lock_sha256 !== episode.provenance.runtime_lock_sha256
        || attempt.provenance.source_tree_sha256 !== episode.provenance.source_tree_sha256
      ) {
        fail("invalid", "lineage_broken", `lineage chain differs for ${group.groupId}/${branch.branchId}`);
      }
      if (invariant === null) invariant = current;
      if (!equal(invariant, current)) {
        fail("invalid", "lineage_broken", `producer lineage changes within audited fixture`);
      }
    }
  }
}

function timeoutForMissingBranch(dataset, groupId, branchId) {
  return dataset.failedAttempts.find((item) => (
    item.document.identity.group_id === groupId
    && item.document.identity.branch_id === branchId
    && item.document.outcome.reason_code === "startup_timeout"
    && item.document.publication.final_episode_published === false
  ));
}

function auditBranches(dataset, manifest) {
  const expected = manifest.actions.map((action) => action.branch_id).sort();
  if (dataset.groups.length === 0) fail("blocked", "evidence_missing", "no sibling groups found");
  for (const group of dataset.groups) {
    const actual = group.branches.map((branch) => branch.branchId).sort();
    if (!equal(actual, expected)) {
      const missing = expected.filter((branchId) => !actual.includes(branchId));
      if (missing.length === 1 && timeoutForMissingBranch(dataset, group.groupId, missing[0])) {
        fail("blocked", "attempt_timeout", `branch ${missing[0]} has only a recorded timeout attempt`);
      }
      fail("invalid", "action_missing_or_duplicate", `five-branch set differs for ${group.groupId}`);
    }
  }
}

function expectedFormalSplit(episode, manifest) {
  const { object_spec_id: objectId, layout_id: layoutId, start_pose_id: startId } = episode.environment;
  const ranked = [...manifest.design.reset_seeds].sort((left, right) => {
    const leftHash = sha256(Buffer.from(`${objectId}\0${layoutId}\0${startId}\0${left}`));
    const rightHash = sha256(Buffer.from(`${objectId}\0${layoutId}\0${startId}\0${right}`));
    return leftHash.localeCompare(rightHash);
  });
  const rank = ranked.indexOf(episode.initialization.reset_seed);
  if (rank < 0) fail("invalid", "cross_split_leakage", `reset seed is outside the formal design`);
  const allocation = manifest.split_policy.seed_rank_allocation;
  if (rank < allocation.train) return "train";
  if (rank < allocation.train + allocation.validation) return "validation";
  return "test";
}

function auditSplits(dataset, manifest) {
  const splitByGroup = new Map();
  const lineageSplits = new Map();
  for (const group of dataset.groups) {
    const splits = new Set(group.branches.map((branch) => branchDocument(branch, "episode.json").identity.split));
    if (splits.size !== 1) fail("invalid", "cross_split_leakage", `group ${group.groupId} crosses splits`);
    const split = [...splits][0];
    if (splitByGroup.has(group.groupId) && splitByGroup.get(group.groupId) !== split) {
      fail("invalid", "cross_split_leakage", `group ${group.groupId} is indexed in multiple splits`);
    }
    splitByGroup.set(group.groupId, split);
    const episode = branchDocument(group.branches[0], "episode.json");
    if (split === "preflight") {
      if (!manifest.preflight.reserved_reset_seeds.includes(episode.initialization.reset_seed)) {
        fail("invalid", "cross_split_leakage", `preflight group uses a formal seed: ${group.groupId}`);
      }
    } else if (split !== "fixture" && split !== expectedFormalSplit(episode, manifest)) {
      fail("invalid", "cross_split_leakage", `stable seed-rank split differs for ${group.groupId}`);
    }
    const lineageKey = stableJson([
      episode.initialization.snapshot_sha256,
      episode.initialization.initial_state_sha256,
      episode.initialization.restored_rng_sha256,
    ]);
    if (lineageSplits.has(lineageKey) && lineageSplits.get(lineageKey) !== split) {
      fail("invalid", "cross_split_leakage", `initialization lineage appears across splits`);
    }
    lineageSplits.set(lineageKey, split);
  }
}

function groupValues(group, selector) {
  return group.branches.map((branch) => selector(branchDocument(branch, "episode.json")));
}

function auditInitialization(dataset) {
  for (const group of dataset.groups) {
    const values = groupValues(group, (episode) => episode.initialization);
    if (!values.every((value) => equal(value, values[0]))) {
      const snapshots = values.map((value) => [value.snapshot_id, value.snapshot_sha256, value.initial_state_sha256]);
      const rng = values.map((value) => [value.reset_seed, value.rng_algorithm, value.restored_rng_sha256]);
      if (!snapshots.every((value) => equal(value, snapshots[0]))) {
        fail("rejected", "snapshot_changed", `snapshot/initial state changes within ${group.groupId}`);
      }
      if (!rng.every((value) => equal(value, rng[0]))) {
        fail("rejected", "rng_changed", `seed/RNG changes within ${group.groupId}`);
      }
    }
  }
}

function auditPhysics(dataset) {
  for (const group of dataset.groups) {
    const environments = groupValues(group, (episode) => episode.environment);
    if (!environments.every((value) => equal(value, environments[0]))) {
      fail("rejected", "physics_changed", `environment/physics changes within ${group.groupId}`);
    }
  }
}

function auditOnlyAction(dataset) {
  for (const group of dataset.groups) {
    const inputs = groupValues(group, (episode) => ({
      identity: {
        experiment_id: episode.identity.experiment_id,
        fixture_id: episode.identity.fixture_id,
        group_id: episode.identity.group_id,
        split: episode.identity.split,
      },
      initialization: episode.initialization,
      intervention: {
        changed_variable: episode.intervention.changed_variable,
        target_object_id: episode.intervention.target_object_id,
        application_mode: episode.intervention.control_ledger.application_mode,
        requested_steps: episode.intervention.control_ledger.requested_steps,
      },
      environment: episode.environment,
      provenance: {
        parent_snapshot_id: episode.provenance.parent_snapshot_id,
        source_gate_report_sha256: episode.provenance.source_gate_report_sha256,
        source_commit: episode.provenance.source_commit,
        source_tree_sha256: episode.provenance.source_tree_sha256,
        config_sha256: episode.provenance.config_sha256,
        runtime_lock_sha256: episode.provenance.runtime_lock_sha256,
        asset_manifest_sha256: episode.provenance.asset_manifest_sha256,
        producer: episode.provenance.producer,
      },
    }));
    if (!inputs.every((value) => equal(value, inputs[0]))) {
      fail("rejected", "physics_changed", `a non-commanded input changes within ${group.groupId}`);
    }
  }
}

function auditActionLedger(dataset) {
  for (const group of dataset.groups) {
    for (const branch of group.branches) {
      const intervention = branchDocument(branch, "episode.json").intervention;
      const commanded = intervention.commanded_action;
      const executed = intervention.executed_action;
      const ledger = intervention.control_ledger;
      if (
        !equal(commanded, executed)
        || ledger.requested_steps !== commanded.applied_steps
        || ledger.applied_steps !== executed.applied_steps
        || !/^[a-f0-9]{64}$/.test(ledger.force_samples_sha256)
      ) {
        fail("rejected", "ledger_missing", `action ledger is inconsistent for ${group.groupId}/${branch.branchId}`);
      }
    }
  }
}

function magnitude(vector) {
  return Math.hypot(...vector);
}

function auditContacts(dataset) {
  for (const group of dataset.groups) {
    for (const branch of group.branches) {
      const trajectory = branchDocument(branch, "trajectory.json");
      const ledger = branchDocument(branch, "contact-ledger.json");
      if (
        trajectory.group_id !== group.groupId || trajectory.branch_id !== branch.branchId
        || ledger.group_id !== group.groupId || ledger.branch_id !== branch.branchId
        || trajectory.coordinate_convention !== "robotics-opencv-v1"
        || ledger.coordinate_convention !== "robotics-opencv-v1"
        || trajectory.records.length !== ledger.records.length + 1
      ) {
        fail("rejected", "contact_tampered", `trajectory/contact identity or length differs for ${group.groupId}/${branch.branchId}`);
      }
      let previous = 0;
      for (let index = 0; index < ledger.records.length; index += 1) {
        const record = ledger.records[index];
        const trajectoryRecord = trajectory.records[index + 1];
        if (
          !Number.isFinite(record.episode_time_s)
          || record.episode_time_s <= previous
          || record.episode_time_s !== trajectoryRecord.episode_time_s
          || record.phase !== trajectoryRecord.phase
          || record.phase_step !== trajectoryRecord.phase_step
          || !Array.isArray(record.contacts)
        ) {
          fail("rejected", "contact_tampered", `contact timeline is invalid for ${group.groupId}/${branch.branchId}`);
        }
        previous = record.episode_time_s;
        for (const contact of record.contacts) {
          if (!Array.isArray(contact.body_pair) || contact.body_pair.length !== 2 || !Array.isArray(contact.points)) {
            fail("rejected", "contact_tampered", `contact shape is invalid for ${group.groupId}/${branch.branchId}`);
          }
          for (const point of contact.points) {
            if (
              !finite(point)
              || point.position_W_m?.length !== 3
              || point.normal_W?.length !== 3
              || point.impulse_W_N_s?.length !== 3
              || Math.abs(magnitude(point.normal_W) - 1) > 1e-6
            ) {
              fail("rejected", "contact_tampered", `contact point is invalid for ${group.groupId}/${branch.branchId}`);
            }
          }
        }
      }
    }
  }
}

function auditSettling(dataset) {
  for (const group of dataset.groups) {
    for (const branch of group.branches) {
      const episode = branchDocument(branch, "episode.json");
      const records = branchDocument(branch, "trajectory.json").records;
      const final = records.at(-1);
      const target = final?.actors?.[episode.environment.target_object_id];
      if (!target) fail("rejected", "settling_recomputed", `terminal target state is missing`);
      const linear = magnitude(target.linear_velocity_W_m_s);
      const angular = magnitude(target.angular_velocity_W_rad_s);
      const settled = linear <= episode.environment.settling.linear_speed_max_m_s
        && angular <= episode.environment.settling.angular_speed_max_rad_s;
      const claimed = episode.evidence.settling_result;
      if (
        Math.abs(linear - claimed.final_linear_speed_m_s) > 1e-12
        || Math.abs(angular - claimed.final_angular_speed_rad_s) > 1e-12
        || settled !== claimed.settled
        || final.episode_time_s !== claimed.evaluation_time_s
      ) {
        fail("rejected", "settling_recomputed", `settling does not recompute for ${group.groupId}/${branch.branchId}`);
      }
      if (!settled) fail("rejected", "settling_recomputed", `branch did not settle: ${group.groupId}/${branch.branchId}`);
    }
  }
}

function terminalPosition(branch) {
  const episode = branchDocument(branch, "episode.json");
  return branchDocument(branch, "trajectory.json").records.at(-1).actors[episode.environment.target_object_id].position_W_m;
}

function horizontalDistance(left, right) {
  return Math.hypot(left[0] - right[0], left[1] - right[1]);
}

function auditEffects(dataset, manifest) {
  const actions = new Map(manifest.actions.map((action) => [action.branch_id, action]));
  const thresholds = manifest.thresholds;
  for (const group of dataset.groups) {
    const branches = new Map(group.branches.map((branch) => [branch.branchId, branch]));
    const hold = branches.get("hold");
    const holdEpisode = branchDocument(hold, "episode.json");
    const holdRecords = branchDocument(hold, "trajectory.json").records;
    const target = holdEpisode.environment.target_object_id;
    const initial = holdRecords[0].actors[target].position_W_m;
    const holdFinal = terminalPosition(hold);
    if (horizontalDistance(initial, holdFinal) > thresholds.hold_horizontal_drift_max_m) {
      fail("rejected", "paired_effect_valid", `hold drift exceeds threshold in ${group.groupId}`);
    }
    const effects = new Map();
    for (const [branchId, branch] of branches) {
      if (branchId === "hold") continue;
      const vector = actions.get(branchId).vector_W_N;
      const norm = Math.hypot(vector[0], vector[1]);
      const final = terminalPosition(branch);
      const directional = ((final[0] - holdFinal[0]) * vector[0] + (final[1] - holdFinal[1]) * vector[1]) / norm;
      effects.set(branchId, directional);
      if (directional < thresholds.directional_effect_min_m) {
        fail("rejected", "paired_effect_valid", `${branchId} lacks paired directional effect in ${group.groupId}`);
      }
    }
    if (
      effects.get("push-pos-x-strong") - effects.get("push-pos-x-weak")
      < thresholds.strong_over_weak_min_m
    ) {
      fail("rejected", "paired_effect_valid", `strong push does not exceed weak push in ${group.groupId}`);
    }
  }
}

function check(checkId, status, reasonCode, evidence) {
  return {
    check_id: checkId,
    status,
    reason_code: reasonCode,
    group_id: { availability: "missing", reason: "not_applicable" },
    branch_id: { availability: "missing", reason: "not_applicable" },
    evidence_sha256: digestDocument(evidence),
  };
}

function indexes(dataset) {
  const episodes = [];
  const attempts = [];
  for (const group of dataset.groups) {
    for (const branch of group.branches) {
      const episode = branchDocument(branch, "episode.json");
      const attempt = branchDocument(branch, "attempt.json");
      episodes.push({
        group_id: group.groupId,
        branch_id: branch.branchId,
        episode_id: episode.identity.episode_id,
        sha256: sha256(branch.loaded["episode.json"].bytes),
      });
      attempts.push({
        attempt_id: attempt.identity.attempt_id,
        status: attempt.outcome.status,
        sha256: sha256(branch.loaded["attempt.json"].bytes),
      });
    }
  }
  for (const item of dataset.failedAttempts) {
    attempts.push({
      attempt_id: item.document.identity.attempt_id,
      status: item.document.outcome.status,
      sha256: sha256(item.bytes),
    });
  }
  episodes.sort((a, b) => stableJson(a).localeCompare(stableJson(b)));
  attempts.sort((a, b) => stableJson(a).localeCompare(stableJson(b)));
  return { episodes, attempts };
}

export function validateNegativeCases(negativeCases) {
  if (!Array.isArray(negativeCases)) throw new TypeError("negative mutation matrix must be an array");
  const ids = negativeCases.map((item) => item.case_id).sort();
  if (!equal(ids, [...REQUIRED_NEGATIVE_CASES].sort())) {
    throw new TypeError("negative mutation matrix is incomplete or duplicated");
  }
  for (const item of negativeCases) {
    if (item.outcome !== "passed" || !["rejected", "blocked", "invalid"].includes(item.expected_status)) {
      throw new TypeError(`negative mutation did not pass: ${item.case_id}`);
    }
  }
  return negativeCases;
}

export async function evaluateCohort({ root, manifestPath }) {
  const absoluteRoot = resolve(root);
  const absoluteManifest = resolve(manifestPath);
  const validate = createContractDispatcher({ root: ROOT });
  const manifestLoaded = await jsonFile(absoluteManifest);
  const manifest = manifestLoaded.document;
  const manifestResult = validate(manifest);
  const checks = [];
  let dataset = { groups: [], failedAttempts: [] };
  let failure = null;
  const gates = [
    ["01-schema", "schema_valid", async () => { dataset = await loadDataset(absoluteRoot, manifest, validate); if (!manifestResult.valid) fail("invalid", "schema_valid", "experiment manifest is contract-invalid"); }],
    ["02-path", "path_safe", async () => auditPaths(absoluteRoot, dataset)],
    ["03-checksum", "checksum_valid", async () => auditChecksums(dataset)],
    ["04-ids", "ids_unique", async () => auditIds(dataset, manifest)],
    ["05-lineage", "lineage_complete", async () => auditLineage(dataset, manifest, sha256(manifestLoaded.bytes))],
    ["06-branches", "five_branches_complete", async () => auditBranches(dataset, manifest)],
    ["07-split", "group_split_isolated", async () => auditSplits(dataset, manifest)],
    ["08-initialization", "initialization_invariant", async () => auditInitialization(dataset)],
    ["09-physics", "physics_invariant", async () => auditPhysics(dataset)],
    ["10-only-action", "only_commanded_action_changed", async () => auditOnlyAction(dataset)],
    ["11-action-ledger", "action_ledger_consistent", async () => auditActionLedger(dataset)],
    ["12-contact", "contact_trace_valid", async () => auditContacts(dataset)],
    ["13-settling", "settling_recomputed", async () => auditSettling(dataset)],
    ["14-effect", "paired_effect_valid", async () => auditEffects(dataset, manifest)],
  ];
  for (const [checkId, successReason, gate] of gates) {
    if (failure !== null) break;
    try {
      await gate();
      checks.push(check(checkId, "supported", successReason, { check_id: checkId, status: "supported" }));
    } catch (error) {
      if (!(error instanceof AuditFailure)) throw error;
      failure = error;
      checks.push(check(checkId, error.status, error.reasonCode, {
        check_id: checkId,
        status: error.status,
        reason_code: error.reasonCode,
        message: error.message,
      }));
    }
  }
  const status = checks.reduce((current, item) => PRIORITY[item.status] > PRIORITY[current] ? item.status : current, "supported");
  const index = indexes(dataset);
  const fixtureScope = dataset.groups.length === 1
    && dataset.groups.every((group) => group.branches.every((branch) => branchDocument(branch, "episode.json").identity.split === "fixture"));
  const preflightScope = dataset.groups.length > 0
    && dataset.groups.every((group) => group.branches.every((branch) => branchDocument(branch, "episode.json").identity.split === "preflight"));
  return {
    status,
    reason_code: VERDICT_REASON[status],
    checks,
    counts: {
      expected_groups: fixtureScope ? 1 : preflightScope ? manifest.preflight.expected_group_count : manifest.design.expected_group_count,
      observed_groups: dataset.groups.length,
      expected_episodes: fixtureScope ? 5 : preflightScope ? manifest.preflight.expected_episode_count : manifest.design.expected_episode_count,
      observed_episodes: index.episodes.length,
      attempts: index.attempts.length,
      failed_attempts: dataset.failedAttempts.length,
    },
    inputs: {
      experiment_manifest_sha256: sha256(manifestLoaded.bytes),
      episode_index_sha256: digestDocument(index.episodes),
      attempt_index_sha256: digestDocument(index.attempts),
      source_gate_report_sha256: manifest.source_gate.report_sha256,
    },
    identity: {
      experiment_id: manifest.identity.experiment_id,
      fixture_id: manifest.identity.fixture_id,
    },
    failure: failure === null ? null : {
      status: failure.status,
      reason_code: failure.reasonCode,
      message: failure.message,
    },
  };
}

export async function buildReport({ evaluation, negativeCases, artifacts }) {
  const sourceBytes = await readFile(SOURCE_PATH);
  return {
    schema_version: "0.2.0",
    contract_kind: "objgauss.invariance_report",
    audit_kind: "sibling_invariance",
    identity: {
      report_id: `report-${evaluation.identity.experiment_id}`,
      experiment_id: evaluation.identity.experiment_id,
      fixture_id: evaluation.identity.fixture_id,
    },
    evaluator: {
      name: "objgauss.pr01-invariance-audit",
      version: "0.1.0",
      source_sha256: sha256(sourceBytes),
      imports_writer_logic: false,
    },
    inputs: evaluation.inputs,
    verdict: {
      status: evaluation.status,
      reason_code: evaluation.reason_code,
      aggregation_priority: ["invalid", "rejected", "blocked", "supported"],
    },
    counts: evaluation.counts,
    checks: evaluation.checks,
    negative_cases: validateNegativeCases(negativeCases),
    artifacts,
    claim_boundary: {
      supported_claim: "controlled-sibling-evidence-is-structurally-and-semantically-auditable",
      excluded_claims: ["causal-model-understanding", "gaussian-dynamics", "robot-planning-value"],
    },
  };
}

export const EXIT_CODES = { supported: 0, rejected: 2, blocked: 3, invalid: 4 };
