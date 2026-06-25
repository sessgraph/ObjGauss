import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const MODE = "object-boundary-remap-reviewed-allowlist-fixture-v1";
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-object-boundary-remap-reviewed-allowlist";
const DEFAULT_ASSET = "nerf-lego-alpha-closure-local";

const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const assetId = String(args.asset ?? args.assets ?? DEFAULT_ASSET).split(",")[0].trim();
const candidateDir = path.join(outputDir, "candidate");
const positiveDir = path.join(outputDir, "positive");
const policyPath = path.join(outputDir, "fixture-policy.json");
const allowlistPath = path.join(outputDir, "fixture-reviewed-allowlist.json");

mkdirSync(outputDir, { recursive: true });

const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  assetId,
  candidateDir,
  positiveDir,
  policyPath,
  allowlistPath,
  candidate: null,
  target: null,
  positive: null,
  failures: [],
  passed: false,
};

try {
  runCommand([
    "node",
    "scripts/export-object-boundary-remap-preview.mjs",
    "--assets",
    assetId,
    "--dry-run",
    "--output-dir",
    candidateDir,
  ]);
  const candidate = readJson(path.join(candidateDir, "summary.json"));
  const result = candidate.results?.[0];
  if (!result) throw new Error(`candidate summary contains no result for ${assetId}`);
  const targetObjectId = firstCandidateTarget(result);
  if (!Number.isFinite(targetObjectId)) {
    throw new Error(`candidate summary contains no remappable target for ${assetId}`);
  }
  summary.candidate = {
    rawCandidateRemapGaussians: result.rawCandidateRemapGaussians ?? result.remappedGaussians ?? 0,
    targetObjectId,
  };
  summary.target = { assetId, targetObjectId };

  writeFileSync(policyPath, `${JSON.stringify(fixturePolicy({ assetId, targetObjectId }), null, 2)}\n`);
  writeFileSync(
    allowlistPath,
    `${JSON.stringify(fixtureAllowlist({ assetId, targetObjectId }), null, 2)}\n`,
  );

  runCommand([
    "node",
    "scripts/export-object-boundary-remap-preview.mjs",
    "--assets",
    assetId,
    "--policy",
    policyPath,
    "--reviewed-allowlist",
    allowlistPath,
    "--output-dir",
    positiveDir,
  ]);
  const positive = readJson(path.join(positiveDir, "summary.json"));
  const positiveResult = positive.results?.[0];
  if (!positiveResult) throw new Error(`positive summary contains no result for ${assetId}`);
  const applied = positiveResult.remappedGaussians ?? 0;
  const blocked = positiveResult.policyGate?.blockedRemaps ?? 0;
  const appliedTargets = positiveResult.policyGate?.appliedTargetObjectIds ?? [];
  if (applied <= 0) {
    throw new Error(`positive fixture did not apply any remaps for ${assetId}:${targetObjectId}`);
  }
  if (!appliedTargets.includes(targetObjectId)) {
    throw new Error(`positive fixture did not apply target ${assetId}:${targetObjectId}`);
  }
  if (!existsSync(positiveResult.outputPly)) {
    throw new Error(`positive fixture output PLY is missing: ${positiveResult.outputPly}`);
  }
  summary.positive = {
    outputPly: positiveResult.outputPly,
    rawCandidateRemapGaussians: positiveResult.rawCandidateRemapGaussians,
    appliedRemaps: applied,
    blockedRemaps: blocked,
    appliedTargetObjectIds: appliedTargets,
  };
  summary.passed = true;
} catch (error) {
  summary.failures.push(error?.message ?? String(error));
  summary.passed = false;
}

writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary));

console.log(
  `object_boundary_remap_reviewed_allowlist=${summary.passed ? "passed" : "failed"} ` +
    `asset=${JSON.stringify(assetId)} target=${JSON.stringify(summary.target?.targetObjectId ?? null)} ` +
    `applied=${summary.positive?.appliedRemaps ?? 0} blocked=${summary.positive?.blockedRemaps ?? 0} ` +
    `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
);

if (!summary.passed) {
  for (const failure of summary.failures) {
    console.error(`object_boundary_remap_reviewed_allowlist_failure=${JSON.stringify(failure)}`);
  }
  process.exit(1);
}

function firstCandidateTarget(result) {
  const pairTarget = (result.rawRemapPairs ?? result.remapPairs ?? []).find((row) =>
    Number.isFinite(row?.fromObject),
  )?.fromObject;
  if (Number.isFinite(pairTarget)) return pairTarget;
  return (result.rawByObject ?? result.byObject ?? []).find((row) =>
    Number.isFinite(row?.objectId),
  )?.objectId;
}

function fixturePolicy({ assetId: targetAssetId, targetObjectId }) {
  return {
    mode: "object-boundary-remap-decision-policy-v1",
    generatedAt: new Date().toISOString(),
    defaultAction: "keep-hard-mask",
    applyMode: "manual-target-allowlist-only",
    globalPromotion: false,
    recommendation: "fixture-only",
    reason: "synthetic positive fixture for reviewed allowlist gate",
    counts: {
      "allowlist-candidate": 1,
      totalTargets: 1,
    },
    allowlistCandidates: [fixtureTarget({ assetId: targetAssetId, targetObjectId })],
    riskyTargets: [],
    reviewOnlyTargets: [],
    targets: [fixtureTarget({ assetId: targetAssetId, targetObjectId })],
  };
}

function fixtureTarget({ assetId: targetAssetId, targetObjectId }) {
  return {
    assetId: targetAssetId,
    targetObjectId,
    decision: "allowlist-candidate",
    action: "manual-review-before-apply",
    reason: "synthetic fixture candidate",
    promotionCandidate: true,
    passed: true,
    hiddenGaussianDelta: -1,
    hiddenGaussianDeltaShare: -0.000001,
    afterDelta: {
      coverageRatio: 1,
      lumaDelta: 0,
      chromaDelta: 0,
    },
    deleteDeltaChange: {
      coverageRatio: 0,
      lumaDelta: 0,
      chromaDelta: 0,
    },
  };
}

function fixtureAllowlist({ assetId: targetAssetId, targetObjectId }) {
  return {
    mode: "object-boundary-remap-reviewed-allowlist-v1",
    version: 1,
    defaultAction: "keep-hard-mask",
    targets: [
      {
        assetId: targetAssetId,
        targetObjectId,
        reviewer: "fixture",
        reason: "synthetic positive fixture for reviewed allowlist gate",
      },
    ],
  };
}

function runCommand(command) {
  const result = spawnSync(command[0], command.slice(1), {
    cwd: process.cwd(),
    encoding: "utf-8",
    stdio: "pipe",
  });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.status !== 0) {
    throw new Error(`command failed (${result.status}): ${command.join(" ")}`);
  }
}

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf-8"));
}

function renderMarkdown(payload) {
  const lines = [
    "# Object Boundary Remap Reviewed Allowlist Fixture",
    "",
    `- Status: ${payload.passed ? "passed" : "failed"}`,
    `- Mode: ${payload.mode}`,
    `- Generated: ${payload.generatedAt}`,
    `- Asset: ${payload.assetId}`,
    `- Target object: ${payload.target?.targetObjectId ?? ""}`,
    `- Applied remaps: ${payload.positive?.appliedRemaps ?? 0}`,
    `- Blocked remaps: ${payload.positive?.blockedRemaps ?? 0}`,
    "",
    "This is a synthetic positive fixture. It writes a temporary policy and reviewed allowlist under `/tmp` to prove that export only applies remaps when both files agree. It does not approve any real repository target.",
    "",
  ];
  if (payload.failures.length > 0) {
    lines.push("## Failures", "");
    for (const failure of payload.failures) lines.push(`- ${failure}`);
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
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
