import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { editRendererContract } from "../src/webgpuCapability.js";
import { estimateWebGpuTileRuntimeStorage } from "../src/webgpuTileStorage.js";

const MiB = 1024 * 1024;
const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-webgpu-scale-budget");
const maxBufferMiB = positiveNumber(args.maxBufferMiB ?? args["max-buffer-mib"], 128);
const maxTotalMiB = positiveNumber(args.maxTotalMiB ?? args["max-total-mib"], 256);
const maxStorageBuffersPerStage = positiveNumber(
  args.maxStorageBuffersPerStage ?? args["max-storage-buffers-per-stage"],
  12,
);

const desktopBudget = Object.freeze({
  status: "available",
  reason: "synthetic-desktop-webgpu-budget",
  label: "Synthetic desktop WebGPU budget",
  maxBufferSize: 256 * MiB,
  maxStorageBufferBindingSize: Math.floor(maxBufferMiB * MiB),
  maxStorageBuffersPerShaderStage: Math.floor(maxStorageBuffersPerStage),
});

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
  mode: "webgpu-scale-budget-v1",
  outputDir,
  assumptions: {
    maxBufferMiB,
    maxTotalMiB,
    maxStorageBuffersPerStage,
    tileSize: 16,
    storageLayout: "webgpu-tile-storage-v1",
    note:
      "Budget audit estimates storage and route gates; it is not a browser FPS or visual-quality proof.",
  },
  rows,
};

writeReport(summary);

console.log(
  [
    `webgpu_scale_budget=${summary.status}`,
    `profiles=${rows.length}`,
    `maxBufferMiB=${maxBufferMiB}`,
    `maxTotalMiB=${maxTotalMiB}`,
    `rows=${rows
      .map(
        (row) =>
          `${row.id}:${row.status}:${row.gaussians}:${row.maxBufferMiB.toFixed(2)}/${row.totalMiB.toFixed(2)}`,
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
  const contract = editRendererContract(desktopBudget, tileSmoke);
  const maxBufferBytes = Math.floor(maxBufferMiB * MiB);
  const maxTotalBytes = Math.floor(maxTotalMiB * MiB);
  const failures = [];

  if (estimate.layoutVersion !== "webgpu-tile-storage-v1") {
    failures.push(`layout=${estimate.layoutVersion}`);
  }
  if (estimate.bufferCount !== 11) {
    failures.push(`bufferCount=${estimate.bufferCount}`);
  }
  if (!estimate.tileEntriesIncluded || !estimate.tileOffsetsIncluded || !estimate.pixelOutputIncluded) {
    failures.push("missing compact tile/pixel buffers");
  }
  if (estimate.maxBufferByteLength > maxBufferBytes) {
    failures.push(
      `maxBuffer=${formatMiB(estimate.maxBufferByteLength)}MiB>${formatMiB(maxBufferBytes)}MiB`,
    );
  }
  if (estimate.totalByteLength > maxTotalBytes) {
    failures.push(
      `total=${formatMiB(estimate.totalByteLength)}MiB>${formatMiB(maxTotalBytes)}MiB`,
    );
  }
  if (contract.targetGate !== "pass" || contract.rendererId !== "webgpu-tile") {
    failures.push(`targetGate=${contract.targetGate}:${contract.targetGateReason}`);
  }
  if (contract.objectFilter !== "gpu-object-state-buffer") {
    failures.push(`objectFilter=${contract.objectFilter}`);
  }

  return {
    id: profile.id,
    status: failures.length === 0 ? "passed" : "failed",
    failures,
    gaussians: profile.gaussians,
    objectCount: profile.objectCount,
    viewportSize: profile.viewportSize,
    tileSize: tileSmoke.tileSize,
    tileCount: tileSmoke.tileCount,
    pixelCount: tileSmoke.pixelCount,
    tileReferenceMultiplier: profile.tileReferenceMultiplier,
    tileEntryCapacity: tileSmoke.tileEntryCapacity,
    storageLayout: estimate.layoutVersion,
    bufferCount: estimate.bufferCount,
    maxBufferKey: estimate.maxBufferKey,
    maxBufferMiB: Number(formatMiB(estimate.maxBufferByteLength)),
    totalMiB: Number(formatMiB(estimate.totalByteLength)),
    targetGate: contract.targetGate,
    targetGateReason: contract.targetGateReason,
    objectFilter: contract.objectFilter,
    descriptors: estimate.descriptors.map((descriptor) => ({
      key: descriptor.key,
      elementCount: descriptor.elementCount,
      allocatedMiB: Number(formatMiB(descriptor.allocatedByteLength)),
    })),
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
    tileCount,
    tileEntryCapacity,
    tileReferenceCount: tileEntryCapacity,
    tileEntryStoredCount: tileEntryCapacity,
    tileEntryLayout: "compact-offset-list",
    tileEntryOffsetCount: tileCount,
    tileCapacityMode: "compact-offset-list",
    tileCapacityStatus: "ok",
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
    "# WebGPU Scale Budget Audit",
    "",
    `- Status: ${summaryToRender.status}`,
    `- Generated: ${summaryToRender.generatedAt}`,
    `- Max storage buffer binding: ${summaryToRender.assumptions.maxBufferMiB} MiB`,
    `- Total storage budget: ${summaryToRender.assumptions.maxTotalMiB} MiB`,
    `- Storage buffers per shader stage: ${summaryToRender.assumptions.maxStorageBuffersPerStage}`,
    `- Note: ${summaryToRender.assumptions.note}`,
    "",
    "| Profile | Status | Gaussians | Viewport | Tile refs/G | Max buffer | Total | Gate |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ...summaryToRender.rows.map(
      (row) =>
        `| ${row.id} | ${row.status} | ${row.gaussians} | ${row.viewportSize} | ${row.tileReferenceMultiplier} | ${row.maxBufferKey} ${row.maxBufferMiB} MiB | ${row.totalMiB} MiB | ${row.targetGate}:${row.targetGateReason} |`,
    ),
    "",
  ].join("\n");
}

function formatMiB(bytes) {
  return (bytes / MiB).toFixed(2);
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
