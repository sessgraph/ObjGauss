import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import {
  webGpuAccumulationWorkgroups,
  webGpuComputeWorkgroups,
  webGpuPixelResolveWorkgroups,
} from "../src/webgpuTileComputeShader.js";
import { estimateWebGpuTileRuntimeStorage } from "../src/webgpuTileStorage.js";

const KiB = 1024;
const MiB = 1024 * KiB;
const args = parseArgs(process.argv.slice(2));
const outputDir = String(
  args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-webgpu-edit-cost-budget",
);
const maxFullUploadMiB = positiveNumber(args.maxFullUploadMiB ?? args["max-full-upload-mib"], 256);
const maxObjectStateUpdateKiB = positiveNumber(
  args.maxObjectStateUpdateKiB ?? args["max-object-state-update-kib"],
  1024,
);
const maxObjectStateUploadShare = positiveNumber(
  args.maxObjectStateUploadShare ?? args["max-object-state-upload-share"],
  0.01,
);
const maxTileReferences = positiveNumber(
  args.maxTileReferences ?? args["max-tile-references"],
  32_000_000,
);
const maxPixelCandidateChecksG = positiveNumber(
  args.maxPixelCandidateChecksG ?? args["max-pixel-candidate-checks-g"],
  10,
);
const maxPixelWorkgroups = positiveNumber(args.maxPixelWorkgroups ?? args["max-pixel-workgroups"], 4096);
const maxTotalWorkgroups = positiveNumber(args.maxTotalWorkgroups ?? args["max-total-workgroups"], 4352);

const profiles = [
  {
    id: "c-path-100k-interactive",
    gaussians: 100_000,
    objectCount: 64,
    viewportSize: 512,
    tileReferenceMultiplier: 24,
  },
  {
    id: "c-path-300k-medium",
    gaussians: 300_000,
    objectCount: 128,
    viewportSize: 384,
    tileReferenceMultiplier: 28,
  },
  {
    id: "c-path-1m-budget",
    gaussians: 1_000_000,
    objectCount: 256,
    viewportSize: 320,
    tileReferenceMultiplier: 32,
  },
];

const rows = profiles.map((profile) => auditProfile(profile));
const summary = {
  status: rows.every((row) => row.status === "passed") ? "passed" : "failed",
  generatedAt: new Date().toISOString(),
  mode: "webgpu-edit-cost-budget-v1",
  outputDir,
  assumptions: {
    storageLayout: "webgpu-tile-storage-v1",
    tileListMode: "object-state-filtered",
    tileSize: 16,
    maxFullUploadMiB,
    maxObjectStateUpdateKiB,
    maxObjectStateUploadShare,
    maxTileReferences,
    maxPixelCandidateChecksG,
    maxPixelWorkgroups,
    maxTotalWorkgroups,
    note:
      "Budget audit estimates objectState-only edit upload bytes, compute dispatch shape, and candidate scan upper bounds; it is not a browser FPS proof.",
  },
  rows,
};

writeReport(summary);

console.log(
  [
    `webgpu_edit_cost_budget=${summary.status}`,
    `profiles=${rows.length}`,
    `objectStateMaxKiB=${maxObjectStateUpdateKiB}`,
    `candidateMaxG=${maxPixelCandidateChecksG}`,
    `rows=${rows
      .map(
        (row) =>
          `${row.id}:${row.status}:${row.gaussians}:edit=${row.objectStateUpdateKiB.toFixed(2)}KiB/full=${row.fullUploadMiB.toFixed(2)}MiB/candidates=${row.pixelCandidateChecksG.toFixed(3)}G`,
      )
      .join(",")}`,
    `outputDir=${JSON.stringify(outputDir)}`,
  ].join(" "),
);

if (summary.status !== "passed") {
  process.exitCode = 1;
}

function auditProfile(profile) {
  const tileSmoke = syntheticTileSmoke(profile);
  const estimate = estimateWebGpuTileRuntimeStorage(tileSmoke);
  const objectState = estimate.descriptors.find((descriptor) => descriptor.key === "objectState");
  const objectStateBytes = objectState?.allocatedByteLength ?? 0;
  const fullUploadBytes = estimate.totalByteLength;
  const objectStateUploadShare = fullUploadBytes > 0 ? objectStateBytes / fullUploadBytes : 1;
  const accumulationWorkgroups = webGpuAccumulationWorkgroups(tileSmoke);
  const resolveWorkgroups = webGpuComputeWorkgroups(tileSmoke);
  const pixelWorkgroups = webGpuPixelResolveWorkgroups(tileSmoke);
  const totalWorkgroups = accumulationWorkgroups + resolveWorkgroups + pixelWorkgroups;
  const tileReferenceCount = tileSmoke.tileReferenceCount;
  const accumulationSampleChecks = tileReferenceCount * 4;
  const pixelCandidateChecksUpperBound = tileReferenceCount * tileSmoke.tileSize * tileSmoke.tileSize;
  const failures = [];

  if (estimate.layoutVersion !== "webgpu-tile-storage-v1") {
    failures.push(`layout=${estimate.layoutVersion}`);
  }
  if (!objectState) {
    failures.push("missing objectState buffer");
  }
  if (fullUploadBytes > maxFullUploadMiB * MiB) {
    failures.push(`fullUpload=${formatMiB(fullUploadBytes)}MiB>${maxFullUploadMiB}MiB`);
  }
  if (objectStateBytes > maxObjectStateUpdateKiB * KiB) {
    failures.push(
      `objectStateUpdate=${formatKiB(objectStateBytes)}KiB>${maxObjectStateUpdateKiB}KiB`,
    );
  }
  if (objectStateUploadShare > maxObjectStateUploadShare) {
    failures.push(
      `objectStateShare=${objectStateUploadShare.toFixed(6)}>${maxObjectStateUploadShare}`,
    );
  }
  if (tileReferenceCount > maxTileReferences) {
    failures.push(`tileReferences=${tileReferenceCount}>${maxTileReferences}`);
  }
  if (pixelCandidateChecksUpperBound > maxPixelCandidateChecksG * 1_000_000_000) {
    failures.push(
      `pixelCandidateChecks=${formatG(pixelCandidateChecksUpperBound)}G>${maxPixelCandidateChecksG}G`,
    );
  }
  if (pixelWorkgroups > maxPixelWorkgroups) {
    failures.push(`pixelWorkgroups=${pixelWorkgroups}>${maxPixelWorkgroups}`);
  }
  if (totalWorkgroups > maxTotalWorkgroups) {
    failures.push(`totalWorkgroups=${totalWorkgroups}>${maxTotalWorkgroups}`);
  }

  return {
    id: profile.id,
    status: failures.length === 0 ? "passed" : "failed",
    failures,
    gaussians: profile.gaussians,
    objectCount: profile.objectCount,
    viewportSize: profile.viewportSize,
    pixelCount: tileSmoke.pixelCount,
    tileSize: tileSmoke.tileSize,
    tileCount: tileSmoke.tileCount,
    tileReferenceMultiplier: profile.tileReferenceMultiplier,
    tileReferenceCount,
    fullUploadMiB: Number(formatMiB(fullUploadBytes)),
    objectStateUpdateKiB: Number(formatKiB(objectStateBytes)),
    objectStateUploadShare: Number(objectStateUploadShare.toFixed(8)),
    staticUploadExcludedMiB: Number(formatMiB(Math.max(0, fullUploadBytes - objectStateBytes))),
    accumulationWorkgroups,
    resolveWorkgroups,
    pixelWorkgroups,
    totalWorkgroups,
    accumulationSampleChecks,
    accumulationSampleChecksM: Number(formatM(accumulationSampleChecks)),
    pixelCandidateChecksUpperBound,
    pixelCandidateChecksG: Number(formatG(pixelCandidateChecksUpperBound)),
    editUpdateMode: "object-state-only",
  };
}

function syntheticTileSmoke(profile) {
  const tileSize = 16;
  const tileColumns = Math.ceil(profile.viewportSize / tileSize);
  const tileRows = tileColumns;
  const tileCount = tileColumns * tileRows;
  const pixelCount = profile.viewportSize * profile.viewportSize;
  const tileEntryCapacity = profile.gaussians * profile.tileReferenceMultiplier;
  return {
    layoutVersion: "webgpu-tile-smoke-v1",
    tileCapacityGate: "pass",
    packedGaussians: profile.gaussians,
    visibleGaussians: profile.gaussians,
    binnedGaussians: profile.gaussians,
    objectCount: profile.objectCount,
    objectStateStrideUint32: 4,
    viewportWidth: profile.viewportSize,
    viewportHeight: profile.viewportSize,
    pixelCount,
    tileSize,
    tileColumns,
    tileRows,
    tileCount,
    tileEntryCapacity,
    tileReferenceCount: tileEntryCapacity,
    tileEntryStoredCount: tileEntryCapacity,
    tileEntryLayout: "compact-offset-list",
    tileEntryOffsetCount: tileCount,
    tileCapacityMode: "compact-offset-list",
    tileCapacityStatus: "ok",
    tileListMode: "object-state-filtered",
  };
}

function writeReport(summaryToWrite) {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(
    path.join(outputDir, "summary.json"),
    `${JSON.stringify(summaryToWrite, null, 2)}\n`,
  );
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summaryToWrite));
}

function renderMarkdown(summaryToRender) {
  return [
    "# WebGPU Edit Cost Budget Audit",
    "",
    `- Status: ${summaryToRender.status}`,
    `- Generated: ${summaryToRender.generatedAt}`,
    `- Layout: ${summaryToRender.assumptions.storageLayout}`,
    `- Tile list mode: ${summaryToRender.assumptions.tileListMode}`,
    `- Max object-state update: ${summaryToRender.assumptions.maxObjectStateUpdateKiB} KiB`,
    `- Max object-state upload share: ${summaryToRender.assumptions.maxObjectStateUploadShare}`,
    `- Max pixel candidate checks: ${summaryToRender.assumptions.maxPixelCandidateChecksG} G`,
    `- Note: ${summaryToRender.assumptions.note}`,
    "",
    "| Profile | Status | Gaussians | Objects | Viewport | Tile refs | Full upload | Edit upload | Share | Workgroups | Pixel candidates |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...summaryToRender.rows.map(
      (row) =>
        `| ${row.id} | ${row.status} | ${row.gaussians} | ${row.objectCount} | ${row.viewportSize} | ${row.tileReferenceCount} | ${row.fullUploadMiB} MiB | ${row.objectStateUpdateKiB} KiB | ${row.objectStateUploadShare} | ${row.totalWorkgroups} | ${row.pixelCandidateChecksG} G |`,
    ),
    "",
    "This gate checks the C-path edit update shape. Passing means compatible object edits can avoid full static storage re-upload and remain inside the configured dispatch/candidate-scan budgets. It does not prove real browser FPS.",
    "",
  ].join("\n");
}

function formatKiB(bytes) {
  return (bytes / KiB).toFixed(2);
}

function formatMiB(bytes) {
  return (bytes / MiB).toFixed(2);
}

function formatM(value) {
  return (value / 1_000_000).toFixed(3);
}

function formatG(value) {
  return (value / 1_000_000_000).toFixed(3);
}

function positiveNumber(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
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
