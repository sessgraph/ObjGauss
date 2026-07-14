#!/usr/bin/env node

import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { tmpdir } from "node:os";
import { evaluateCohort, sha256, stableJson } from "../src/pr01/audit/evaluator.mjs";

const CASES = [
  ["snapshot_changed", "rejected", "snapshot_changed"],
  ["rng_changed", "rejected", "rng_changed"],
  ["physics_changed", "rejected", "physics_changed"],
  ["cross_split_leakage", "invalid", "cross_split_leakage"],
  ["action_missing_or_duplicate", "invalid", "action_missing_or_duplicate"],
  ["ledger_missing", "invalid", "ledger_missing"],
  ["contact_tampered", "rejected", "contact_tampered"],
  ["checksum_mismatch", "invalid", "checksum_mismatch"],
  ["lineage_broken", "invalid", "lineage_broken"],
  ["attempt_timeout", "blocked", "attempt_timeout"],
  ["non_finite_value", "invalid", "non_finite_value"],
];

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function writeJson(path, value) {
  await writeFile(path, `${stableJson(value)}\n`);
}

async function groupPath(root, manifest) {
  const base = resolve(root, "dataset", manifest.identity.experiment_id);
  const groups = (await readdir(base, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  if (groups.length !== 1) throw new Error(`mutation fixture requires one group, found ${groups.length}`);
  return resolve(base, groups[0]);
}

async function resealBranch(path) {
  const trajectoryPath = resolve(path, "trajectory.json");
  const contactPath = resolve(path, "contact-ledger.json");
  const episodePath = resolve(path, "episode.json");
  const attemptPath = resolve(path, "attempt.json");
  const publicationPath = resolve(path, "publication.json");
  const trajectoryBytes = await readFile(trajectoryPath);
  const contactBytes = await readFile(contactPath);
  const trajectory = JSON.parse(trajectoryBytes);
  const contacts = JSON.parse(contactBytes);
  const episode = await readJson(episodePath);
  episode.evidence.trajectory.sha256 = sha256(trajectoryBytes);
  episode.evidence.trajectory.byte_length = trajectoryBytes.length;
  episode.evidence.trajectory.record_count = trajectory.records.length;
  episode.evidence.contact_ledger.sha256 = sha256(contactBytes);
  episode.evidence.contact_ledger.byte_length = contactBytes.length;
  episode.evidence.contact_ledger.record_count = contacts.records.length;
  await writeJson(episodePath, episode);
  const episodeBytes = await readFile(episodePath);
  const attempt = await readJson(attemptPath);
  attempt.publication.episode_artifact.value.sha256 = sha256(episodeBytes);
  await writeJson(attemptPath, attempt);
  const attemptBytes = await readFile(attemptPath);
  const hashes = {
    episode_sha256: sha256(episodeBytes),
    trajectory_sha256: sha256(trajectoryBytes),
    contact_ledger_sha256: sha256(contactBytes),
  };
  const publication = await readJson(publicationPath);
  Object.assign(publication, hashes, {
    attempt_sha256: sha256(attemptBytes),
    semantic_sha256: sha256(Buffer.from(`${stableJson(hashes)}\n`)),
  });
  await writeJson(publicationPath, publication);
}

async function mutation(caseId, root, manifest) {
  const group = await groupPath(root, manifest);
  const branch = resolve(group, "push-pos-x-weak");
  const episodePath = resolve(branch, "episode.json");
  const attemptPath = resolve(branch, "attempt.json");
  if (caseId === "snapshot_changed") {
    const episode = await readJson(episodePath);
    episode.initialization.snapshot_sha256 = "f".repeat(64);
    await writeJson(episodePath, episode);
    await resealBranch(branch);
  } else if (caseId === "rng_changed") {
    const episode = await readJson(episodePath);
    episode.initialization.restored_rng_sha256 = "e".repeat(64);
    await writeJson(episodePath, episode);
    await resealBranch(branch);
  } else if (caseId === "physics_changed") {
    const episode = await readJson(episodePath);
    episode.environment.physics.static_friction = 0.6;
    await writeJson(episodePath, episode);
    await resealBranch(branch);
  } else if (caseId === "cross_split_leakage") {
    const episode = await readJson(episodePath);
    const attempt = await readJson(attemptPath);
    episode.identity.split = "validation";
    attempt.identity.split = "validation";
    await writeJson(episodePath, episode);
    await writeJson(attemptPath, attempt);
    await resealBranch(branch);
  } else if (caseId === "action_missing_or_duplicate") {
    await rm(branch, { recursive: true, force: true });
  } else if (caseId === "ledger_missing") {
    await rm(attemptPath);
  } else if (caseId === "contact_tampered") {
    const contactPath = resolve(branch, "contact-ledger.json");
    const contact = await readJson(contactPath);
    contact.records[0].contacts[0].points[0].normal_W = [2, 0, 0];
    await writeJson(contactPath, contact);
    await resealBranch(branch);
  } else if (caseId === "checksum_mismatch") {
    const contactPath = resolve(branch, "contact-ledger.json");
    const payload = await readFile(contactPath);
    await writeFile(contactPath, Buffer.concat([payload, Buffer.from(" ")]));
  } else if (caseId === "lineage_broken") {
    const attempt = await readJson(attemptPath);
    attempt.provenance.source_tree_sha256 = "d".repeat(64);
    await writeJson(attemptPath, attempt);
    await resealBranch(branch);
  } else if (caseId === "attempt_timeout") {
    const attempt = await readJson(attemptPath);
    attempt.identity.attempt_id = `${attempt.identity.attempt_id}-timeout`;
    attempt.timing = { started_monotonic_s: 0, finished_monotonic_s: 10, wall_seconds: 10, timeout_s: 10 };
    attempt.outcome = {
      status: "failed",
      classification: "infrastructure",
      reason_code: "startup_timeout",
      message: "Injected startup timeout mutation.",
    };
    attempt.retry.eligible = true;
    attempt.publication = {
      temporary_output_removed: true,
      final_episode_published: false,
      episode_artifact: { availability: "missing", reason: "not_produced" },
    };
    const target = resolve(root, "attempts", manifest.identity.experiment_id, attempt.identity.group_id, attempt.identity.branch_id, `${attempt.identity.attempt_id}.json`);
    await mkdir(dirname(target), { recursive: true });
    await writeJson(target, attempt);
    await rm(branch, { recursive: true, force: true });
  } else if (caseId === "non_finite_value") {
    const trajectoryPath = resolve(branch, "trajectory.json");
    const text = await readFile(trajectoryPath, "utf8");
    const changed = text.replace(/"episode_time_s":(?:0(?:\.0)?)/, '"episode_time_s":NaN');
    if (changed === text) throw new Error("could not inject non-finite trajectory value");
    await writeFile(trajectoryPath, changed);
  } else {
    throw new Error(`unknown mutation ${caseId}`);
  }
}

async function main() {
  const [baselineRootArg, manifestArg, outputArg, workParentArg] = process.argv.slice(2);
  if (!baselineRootArg || !manifestArg || !outputArg) {
    throw new Error("usage: run-pr01d-mutations.mjs <baseline-root> <manifest> <output> [work-parent]");
  }
  const baselineRoot = resolve(baselineRootArg);
  const manifestPath = resolve(manifestArg);
  const output = resolve(outputArg);
  const parent = workParentArg ? resolve(workParentArg) : tmpdir();
  await mkdir(parent, { recursive: true });
  const work = await mkdtemp(resolve(parent, "pr01d-mutations-"));
  const manifest = await readJson(manifestPath);
  const results = [];
  try {
    const baseline = await evaluateCohort({ root: baselineRoot, manifestPath });
    if (baseline.status !== "supported") {
      throw new Error(`baseline is not supported: ${JSON.stringify(baseline.failure)}`);
    }
    for (const [caseId, expectedStatus, expectedReason] of CASES) {
      const target = resolve(work, caseId);
      await cp(baselineRoot, target, { recursive: true, force: false, errorOnExist: true });
      await mutation(caseId, target, manifest);
      const evaluation = await evaluateCohort({ root: target, manifestPath });
      const actualReason = evaluation.failure?.reason_code;
      if (evaluation.status !== expectedStatus || actualReason !== expectedReason) {
        throw new Error(`${caseId}: expected ${expectedStatus}/${expectedReason}, got ${evaluation.status}/${actualReason}`);
      }
      results.push({
        case_id: caseId,
        outcome: "passed",
        expected_status: expectedStatus,
        expected_reason_code: expectedReason,
      });
    }
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${stableJson(results)}\n`);
    process.stdout.write(`${JSON.stringify({ verdict: "supported", cases: results.length, output: basename(output) })}\n`);
  } finally {
    await rm(work, { recursive: true, force: true });
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
