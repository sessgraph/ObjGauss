#!/usr/bin/env node

import { readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { resolve, relative } from "node:path";
import { createContractDispatcher } from "../src/pr01/contract-dispatch.mjs";

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

function sha256(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

async function bytes(path) {
  return readFile(path);
}

async function document(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function validateContract(validate, value, path) {
  const result = validate(value);
  assert(result.valid, `${path} is contract-invalid: ${JSON.stringify(result)}`);
}

async function branchAudit(validate, root, branch) {
  const directory = resolve(root, "dataset/pr01c-writer-fixture-v0/group-box-a-layout-a-start-a-seed-24071401", branch);
  const names = (await readdir(directory)).sort();
  assert(JSON.stringify(names) === JSON.stringify(FILES), `${branch} file set differs: ${names}`);

  const episodePath = resolve(directory, "episode.json");
  const attemptPath = resolve(directory, "attempt.json");
  const trajectoryPath = resolve(directory, "trajectory.json");
  const contactPath = resolve(directory, "contact-ledger.json");
  const publicationPath = resolve(directory, "publication.json");
  const episode = await document(episodePath);
  const attempt = await document(attemptPath);
  const trajectory = await document(trajectoryPath);
  const contacts = await document(contactPath);
  const publication = await document(publicationPath);
  validateContract(validate, episode, episodePath);
  validateContract(validate, attempt, attemptPath);

  const trajectoryBytes = await bytes(trajectoryPath);
  const contactBytes = await bytes(contactPath);
  const episodeBytes = await bytes(episodePath);
  const attemptBytes = await bytes(attemptPath);
  const trajectoryDescriptor = episode.evidence.trajectory;
  const contactDescriptor = episode.evidence.contact_ledger;
  assert(trajectoryDescriptor.uri === relative(root, trajectoryPath).replaceAll("\\", "/"), `${branch} trajectory URI differs`);
  assert(contactDescriptor.uri === relative(root, contactPath).replaceAll("\\", "/"), `${branch} contact URI differs`);
  assert(trajectoryDescriptor.sha256 === sha256(trajectoryBytes), `${branch} trajectory checksum differs`);
  assert(contactDescriptor.sha256 === sha256(contactBytes), `${branch} contact checksum differs`);
  assert(trajectoryDescriptor.byte_length === trajectoryBytes.length, `${branch} trajectory byte length differs`);
  assert(contactDescriptor.byte_length === contactBytes.length, `${branch} contact byte length differs`);
  assert(trajectoryDescriptor.record_count === trajectory.records.length && trajectory.records.length === 111, `${branch} trajectory count differs`);
  assert(contactDescriptor.record_count === contacts.records.length && contacts.records.length === 110, `${branch} contact count differs`);
  assert(attempt.publication.episode_artifact.value.sha256 === sha256(episodeBytes), `${branch} attempt episode checksum differs`);
  assert(attempt.publication.final_episode_published === true, `${branch} attempt did not publish`);

  const semanticDocument = {
    contact_ledger_sha256: sha256(contactBytes),
    episode_sha256: sha256(episodeBytes),
    trajectory_sha256: sha256(trajectoryBytes),
  };
  const semanticBytes = Buffer.from(`${JSON.stringify(semanticDocument)}\n`);
  assert(publication.semantic_sha256 === sha256(semanticBytes), `${branch} semantic digest differs`);
  assert(publication.attempt_sha256 === sha256(attemptBytes), `${branch} attempt checksum differs`);
  return {
    semantic_sha256: publication.semantic_sha256,
    episode_sha256: sha256(episodeBytes),
    trajectory_sha256: sha256(trajectoryBytes),
    contact_ledger_sha256: sha256(contactBytes),
  };
}

async function main() {
  const [canonicalRoot, reverseRoot, canonicalReportPath, reverseReportPath] = process.argv.slice(2);
  assert(canonicalRoot && reverseRoot && canonicalReportPath && reverseReportPath,
    "usage: audit-pr01c-golden.mjs <canonical-root> <reverse-root> <canonical-report> <reverse-report>");
  const validate = createContractDispatcher();
  const canonical = {};
  const reverse = {};
  for (const branch of BRANCHES) {
    canonical[branch] = await branchAudit(validate, resolve(canonicalRoot), branch);
    reverse[branch] = await branchAudit(validate, resolve(reverseRoot), branch);
  }
  assert(JSON.stringify(canonical) === JSON.stringify(reverse), "canonical/reverse raw semantic artifacts differ");
  const canonicalReport = await document(canonicalReportPath);
  const reverseReport = await document(reverseReportPath);
  assert(canonicalReport.verdict === "pending_repeat", "canonical report did not stop at pending_repeat");
  assert(reverseReport.verdict === "supported", "reverse report is not supported");
  assert(canonicalReport.evidence_sha256 === reverseReport.evidence_sha256, "group evidence digest differs");
  assert(reverseReport.repeat_comparison.matches === true, "repeat comparison did not match");
  assert(reverseReport.repeat_comparison.opposite_execution_orders === true, "orders are not opposite");
  process.stdout.write(`${JSON.stringify({
    verdict: "supported",
    branch_count: BRANCHES.length,
    trajectory_records_per_branch: 111,
    contact_records_per_branch: 110,
    evidence_sha256: reverseReport.evidence_sha256,
    branch_semantics: canonical,
  })}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 4;
});
