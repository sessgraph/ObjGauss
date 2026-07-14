#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import { dirname, resolve, sep } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT = resolve(ROOT, "artifacts/pr01");

function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
async function json(path) { return JSON.parse(await readFile(path, "utf8")); }
function assert(value, message) { if (!value) throw new Error(message); }

async function allFiles(path) {
  const entries = await readdir(path, { withFileTypes: true });
  const result = [];
  for (const entry of entries) {
    const child = resolve(path, entry.name);
    if (entry.isDirectory()) result.push(...await allFiles(child));
    if (entry.isFile()) result.push(child);
  }
  return result.sort();
}

async function main() {
  const [delivery, audit, cohort, demo, checksums, app, html] = await Promise.all([
    json(resolve(OUTPUT, "delivery-report.json")),
    json(resolve(OUTPUT, "audit-report.json")),
    json(resolve(OUTPUT, "cohort-report.json")),
    json(resolve(OUTPUT, "demo/demo-manifest.json")),
    readFile(resolve(OUTPUT, "checksums.sha256"), "utf8"),
    readFile(resolve(OUTPUT, "demo/app.mjs"), "utf8"),
    readFile(resolve(OUTPUT, "demo/index.html"), "utf8"),
  ]);
  assert(delivery.verdict === "supported", "delivery report is not supported");
  assert(audit.verdict.status === "supported", "machine audit is not supported");
  assert(cohort.verdict === "supported", "cohort report is not supported");
  assert(audit.counts.observed_groups === 48 && audit.counts.observed_episodes === 240, "formal counts differ");
  assert(cohort.counts.failed_attempts === 0 && cohort.counts.extra_attempts === 0, "failed or extra attempts exist");
  assert(delivery.counts.attempts === 240, "delivery attempt count differs");
  assert(JSON.stringify(cohort.split_counts) === JSON.stringify({ preflight: 0, test: 12, train: 24, validation: 12 }), "split counts differ");
  assert(demo.branches.length === 5 && new Set(demo.branches.map((item) => item.branch_id)).size === 5, "demo does not contain five unique branches");
  assert(demo.display_boundary.rgb === false && demo.display_boundary.simulator_in_browser === false, "demo boundary expanded");
  assert(!/(WebGL|cdn\.|https?:\/\/|rgb_uri|rgb-card)/i.test(`${app}\n${html}`), "demo source contains forbidden WebGL/CDN/RGB path");

  const [datasetFiles, standaloneAttemptFiles] = await Promise.all([
    allFiles(resolve(OUTPUT, "dataset")),
    allFiles(resolve(OUTPUT, "attempts")),
  ]);
  const successfulAttemptFiles = datasetFiles.filter((path) => path.endsWith(`${sep}attempt.json`));
  assert(successfulAttemptFiles.length === 240, "successful attempt ledger count differs");
  assert(
    standaloneAttemptFiles.filter((path) => path.endsWith(".json")).length === cohort.counts.failed_attempts,
    "standalone failed attempt ledger count differs",
  );
  for (const path of successfulAttemptFiles) {
    const attempt = await json(path);
    assert(attempt.outcome.status === "succeeded", `non-success attempt in formal dataset: ${path}`);
    assert(attempt.publication.final_episode_published === true, `attempt lacks published episode: ${path}`);
    assert(attempt.retry.ordinal === 1 && attempt.retry.seed_reused === true, `attempt retry/seed ledger differs: ${path}`);
  }

  const status = spawnSync(
    "git",
    ["status", "--porcelain=v1", "--untracked-files=all"],
    { cwd: ROOT, encoding: "utf8" },
  );
  assert(status.status === 0, "cannot inspect worktree state");
  assert(status.stdout.trim() === "", "delivery verification requires a clean checkout");
  const git = spawnSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" });
  assert(git.status === 0 && delivery.source_commit === git.stdout.trim(), "delivery source commit differs from HEAD");
  const lines = checksums.trim().split("\n");
  assert(lines.length >= 1210, `checksum index is unexpectedly small: ${lines.length}`);
  for (const line of lines) {
    const match = /^([a-f0-9]{64})  (artifacts\/pr01\/[a-zA-Z0-9._/-]+)$/.exec(line);
    assert(match, `invalid checksum line: ${line}`);
    const path = resolve(ROOT, match[2]);
    assert(path.startsWith(`${OUTPUT}${sep}`), `checksum path escapes delivery root: ${match[2]}`);
    assert(sha256(await readFile(path)) === match[1], `checksum mismatch: ${match[2]}`);
  }
  process.stdout.write(`${JSON.stringify({
    verdict: "supported",
    source_commit: delivery.source_commit,
    groups: audit.counts.observed_groups,
    episodes: audit.counts.observed_episodes,
    checksum_entries: lines.length,
    checksums_sha256: sha256(Buffer.from(checksums)),
    demo_group_id: demo.group_id,
  })}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
