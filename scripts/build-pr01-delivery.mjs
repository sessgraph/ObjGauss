#!/usr/bin/env node

import { createHash } from "node:crypto";
import { cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { BRANCH_ORDER } from "../viewer/pr01/render.mjs";
import { stableJson } from "../src/pr01/audit/evaluator.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE = resolve(ROOT, "generated/pr01e/formal");
const OUTPUT = resolve(ROOT, "artifacts/pr01");
const EXPERIMENT = resolve(ROOT, "contracts/fixtures/pr01e/experiment.formal.json");
const VIEWER = resolve(ROOT, "viewer/pr01");

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function json(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

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

function cleanGitHead() {
  const status = spawnSync(
    "git",
    ["status", "--porcelain=v1", "--untracked-files=all"],
    { cwd: ROOT, encoding: "utf8" },
  );
  if (status.status !== 0) throw new Error(`cannot inspect worktree: ${status.stderr}`);
  if (status.stdout.trim() !== "") {
    throw new Error("delivery requires a clean checkout; dirty worktree cannot prove final commit lineage");
  }
  const result = spawnSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" });
  if (result.status !== 0) throw new Error(`cannot resolve source commit: ${result.stderr}`);
  return result.stdout.trim();
}

async function main() {
  const audit = await json(resolve(SOURCE, "audit-report.json"));
  const cohort = await json(resolve(SOURCE, "cohort-report.json"));
  if (audit.verdict.status !== "supported" || cohort.verdict !== "supported") {
    throw new Error("delivery refuses a non-supported machine audit or cohort report");
  }
  if (audit.counts.observed_groups !== 48 || audit.counts.observed_episodes !== 240) {
    throw new Error("delivery requires the complete frozen formal cohort");
  }
  const sourceCommit = cleanGitHead();
  const sourceTreeHashes = new Set();
  const episodePaths = (await allFiles(resolve(SOURCE, "dataset"))).filter((path) => path.endsWith("/episode.json"));
  for (const path of episodePaths) {
    const episode = await json(path);
    if (episode.provenance.source_commit !== sourceCommit) {
      throw new Error(`${path} was not produced from current HEAD ${sourceCommit}`);
    }
    sourceTreeHashes.add(episode.provenance.source_tree_sha256);
  }
  if (episodePaths.length !== 240 || sourceTreeHashes.size !== 1) {
    throw new Error("formal episode lineage is incomplete or inconsistent");
  }

  await rm(OUTPUT, { recursive: true, force: true });
  await mkdir(resolve(OUTPUT, "attempts"), { recursive: true });
  await cp(resolve(SOURCE, "dataset"), resolve(OUTPUT, "dataset"), { recursive: true });
  await cp(VIEWER, resolve(OUTPUT, "demo"), { recursive: true });
  await Promise.all([
    cp(EXPERIMENT, resolve(OUTPUT, "experiment-manifest.json")),
    cp(resolve(SOURCE, "audit-report.json"), resolve(OUTPUT, "audit-report.json")),
    cp(resolve(SOURCE, "audit-report.md"), resolve(OUTPUT, "audit-report.md")),
    cp(resolve(SOURCE, "cohort-report.json"), resolve(OUTPUT, "cohort-report.json")),
  ]);

  const selected = cohort.groups.filter((group) => group.split === "test").sort((left, right) => left.group_id.localeCompare(right.group_id))[0];
  if (!selected) throw new Error("formal cohort has no test group for the demo");
  const branchRoot = resolve(OUTPUT, "dataset", cohort.experiment_id, selected.group_id);
  const branches = [];
  for (const branchId of BRANCH_ORDER) {
    const directory = resolve(branchRoot, branchId);
    const episodeBytes = await readFile(resolve(directory, "episode.json"));
    const episode = JSON.parse(episodeBytes);
    branches.push({
      branch_id: branchId,
      episode_uri: `../dataset/${cohort.experiment_id}/${selected.group_id}/${branchId}/episode.json`,
      episode_sha256: sha256(episodeBytes),
      trajectory_uri: `../dataset/${cohort.experiment_id}/${selected.group_id}/${branchId}/trajectory.json`,
      contact_uri: `../dataset/${cohort.experiment_id}/${selected.group_id}/${branchId}/contact-ledger.json`,
      trajectory_sha256: episode.evidence.trajectory.sha256,
      contact_sha256: episode.evidence.contact_ledger.sha256,
    });
  }
  const auditBytes = await readFile(resolve(OUTPUT, "audit-report.json"));
  const demoManifest = {
    manifest_version: "0.1.0",
    audit_report_uri: "../audit-report.json",
    audit_report_sha256: sha256(auditBytes),
    experiment_id: cohort.experiment_id,
    group_id: selected.group_id,
    split: selected.split,
    shared_time_range_s: [0, 1.1],
    branches,
    display_boundary: {
      renderer: "canvas-2d-state-only",
      rgb: false,
      simulator_in_browser: false,
      external_assets: false,
    },
  };
  await writeFile(resolve(OUTPUT, "demo/demo-manifest.json"), `${stableJson(demoManifest)}\n`);

  const deliveryReport = {
    report_version: "0.1.0",
    report_kind: "objgauss.pr01-delivery",
    verdict: "supported",
    source_commit: sourceCommit,
    source_tree_sha256: [...sourceTreeHashes][0],
    experiment_manifest_sha256: sha256(await readFile(EXPERIMENT)),
    cohort_report_sha256: sha256(await readFile(resolve(OUTPUT, "cohort-report.json"))),
    audit_report_sha256: sha256(auditBytes),
    counts: audit.counts,
    demo_group_id: selected.group_id,
    demo_branch_count: branches.length,
    checksums_uri: "checksums.sha256",
    claim_boundary: {
      supported: "frozen-primitive-sibling-evidence-delivery-is-locally-reproducible",
      excluded: ["causal-model-understanding", "gaussian-dynamics", "robot-planning-value", "remote-ci"],
    },
  };
  await writeFile(resolve(OUTPUT, "delivery-report.json"), `${stableJson(deliveryReport)}\n`);

  const files = (await allFiles(OUTPUT)).filter((path) => path !== resolve(OUTPUT, "checksums.sha256"));
  const lines = [];
  for (const path of files) {
    const uri = relative(ROOT, path).replaceAll("\\", "/");
    lines.push(`${sha256(await readFile(path))}  ${uri}`);
  }
  const checksumBytes = Buffer.from(`${lines.sort().join("\n")}\n`);
  await writeFile(resolve(OUTPUT, "checksums.sha256"), checksumBytes);
  process.stdout.write(`${JSON.stringify({
    verdict: "supported",
    source_commit: sourceCommit,
    groups: audit.counts.observed_groups,
    episodes: audit.counts.observed_episodes,
    demo_group_id: selected.group_id,
    checksum_entries: lines.length,
    checksums_sha256: sha256(checksumBytes),
  })}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
