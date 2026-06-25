import { readFileSync } from "node:fs";

import { validateReviewedAllowlistManifest } from "./lib/remap-reviewed-allowlist.mjs";

const MODE = "object-boundary-remap-reviewed-allowlist-manifest-audit-v1";
const DEFAULT_ALLOWLIST = "docs/rendering/object-boundary-remap-reviewed-allowlist.json";

const args = parseArgs(process.argv.slice(2));
const allowlistPath = String(
  args.allowlist ??
    args["reviewed-allowlist"] ??
    args.reviewedAllowlist ??
    args["reviewed-allowlist-path"] ??
    DEFAULT_ALLOWLIST,
);
const requireExistingEvidence = !flagEnabled(
  args.skipExistingEvidence ?? args["skip-existing-evidence"],
);

const summary = {
  mode: MODE,
  allowlistPath,
  requireExistingEvidence,
  targetCount: 0,
  failures: [],
  passed: false,
};

try {
  const raw = JSON.parse(readFileSync(allowlistPath, "utf-8"));
  const validated = validateReviewedAllowlistManifest(raw, {
    manifestPath: allowlistPath,
    requireExistingEvidence,
    rootDir: process.cwd(),
  });
  summary.targetCount = validated.targets.length;
  summary.passed = true;
} catch (error) {
  summary.failures.push(error?.message ?? String(error));
  summary.passed = false;
}

console.log(
  `object_boundary_remap_reviewed_allowlist_manifest=${summary.passed ? "passed" : "failed"} ` +
    `targets=${summary.targetCount} evidence=${summary.requireExistingEvidence ? "required" : "not-required"} ` +
    `path=${JSON.stringify(allowlistPath)}`,
);

if (!summary.passed) {
  for (const failure of summary.failures) {
    console.error(`object_boundary_remap_reviewed_allowlist_manifest_failure=${JSON.stringify(failure)}`);
  }
  process.exit(1);
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const entry = argv[index];
    if (!entry.startsWith("--")) continue;
    const key = entry.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}

function flagEnabled(value) {
  return value === true || value === "true" || value === "1" || value === "yes";
}
