import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5180;
const DEFAULT_ASSETS = [
  {
    id: "plush-semantic-closure-local",
    name: "Plush 2D 语义 Mask 闭环样例",
    fileName: "plush_semantic_objects.ply",
  },
  {
    id: "plush-v1-closure-local",
    name: "ObjGauss v1 闭环样例",
    fileName: "plush_v1_objects.ply",
  },
  {
    id: "nerf-lego-alpha-closure-local",
    name: "NeRF Lego 闭环代理样例",
    fileName: "lego_alpha_v1_objects.ply",
  },
];
const KNOWN_ASSETS = [
  ...DEFAULT_ASSETS,
  {
    id: "nerf-lego-trained-output-local",
    name: "NeRF Lego 训练输出样例",
    fileName: "nerf_lego_trained_objects.ply",
  },
];

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = args.url ?? `http://127.0.0.1:${port}/`;
const assets = args.asset ? KNOWN_ASSETS.filter((asset) => asset.id === args.asset) : DEFAULT_ASSETS;
const auditOptions = {
  requireWebGpu: Boolean(args.requireWebgpu ?? args["require-webgpu"]),
  webGpuFlags: String(args.webgpuFlags ?? args["webgpu-flags"] ?? process.env.OBJGAUSS_WEBGPU_FLAGS ?? "none"),
};

if (assets.length === 0) {
  throw new Error(`unknown asset id: ${args.asset}`);
}

const server = args.url || args.noServer ? null : startDevServer(port);
try {
  await waitForApp(baseUrl);
  const results = await runAudit(baseUrl, assets, auditOptions);
  for (const result of results) {
    console.log(
        `asset=${result.assetId} title=${JSON.stringify(result.title)} ` +
        `splatPixels=${result.splatPixels} splatRendererId=${JSON.stringify(result.splatRendererId)} ` +
        `editRenderer=${JSON.stringify(result.editRenderer)} ` +
        `editRendererId=${JSON.stringify(result.editRendererId)} ` +
        `firstFrame=${JSON.stringify(result.webGpuFirstFrameStatus)}:${result.webGpuFirstFramePixels} ` +
        `accumulation=${JSON.stringify(result.webGpuAccumulationStatus)}:${JSON.stringify(result.webGpuAccumulationSource)}:${result.webGpuAccumulationWorkgroups} ` +
        `compute=${JSON.stringify(result.webGpuComputeStatus)}:${JSON.stringify(result.webGpuComputeSource)}:${result.webGpuComputeWorkgroups} ` +
        `pixel=${JSON.stringify(result.webGpuPixelStatus)}:${JSON.stringify(result.webGpuPixelSource)}:${result.webGpuPixelWorkgroups} ` +
        `resolveSource=${JSON.stringify(result.webGpuResolveSource)} ` +
        `storage=${JSON.stringify(result.webGpuStorageStatus)}:${JSON.stringify(result.webGpuStorageChecksum)} ` +
        `storageLimit=${JSON.stringify(result.storageLimitGate)}:${JSON.stringify(result.storageLimitBlocker)}:${JSON.stringify(result.storageEstimatedMaxBufferKey)}:${result.storageEstimatedMaxBufferByteSize} ` +
        `rendererTarget=${JSON.stringify(result.rendererTarget)} ` +
        `targetGate=${JSON.stringify(result.targetGate)}:${JSON.stringify(result.targetGateBlocker)} ` +
        `webgpuStatus=${JSON.stringify(result.webgpuStatus)} ` +
        `fallbackReason=${JSON.stringify(result.fallbackReason)} ` +
        `tileSmokeLayout=${JSON.stringify(result.tileSmokeLayout)} ` +
        `packedGaussians=${result.packedGaussians} ` +
        `binnedGaussians=${result.binnedGaussians} ` +
        `activeTiles=${result.activeTileCount}/${result.tileCount} ` +
        `tileReferences=${result.tileReferenceCount} ` +
        `maxTileOccupancy=${result.maxTileOccupancy} ` +
        `tileCapacity=${JSON.stringify(result.tileCapacityMode)}:${JSON.stringify(result.tileCapacityStatus)}:${result.tileOverflowTileCount} ` +
        `tileEntryLayout=${JSON.stringify(result.tileEntryLayout)}:${result.tileEntryOffsetCount} ` +
        `resolveLayout=${JSON.stringify(result.resolveLayout)} ` +
        `resolvedTiles=${result.resolvedTileCount} ` +
        `resolveChecksum=${JSON.stringify(result.resolveChecksum)} ` +
        `objectState=${JSON.stringify(result.objectStateLayout)}:${JSON.stringify(result.objectStateChecksum)} ` +
        `objectStateAfterDelete=${JSON.stringify(result.objectStateChecksumAfterDelete)} ` +
        `tileOverflowCount=${result.tileOverflowCount} ` +
        `objectFilter=${JSON.stringify(result.objectFilter)} ` +
        `objectFilterTarget=${JSON.stringify(result.objectFilterTarget)} ` +
        `editPixels=${result.editPixels} ` +
        `canvasSelectedObject=${result.canvasSelectedObject} ` +
        `visibleAfterIsolate=${result.visibleAfterIsolate} ` +
        `visibleAfterDelete=${result.visibleAfterDelete} ` +
        `renderModeAfterDelete=${JSON.stringify(result.renderModeAfterDelete)} ` +
        `deletedObjects=${result.deletedObjects} screenshot=${result.screenshotPath}`,
    );
  }
  console.log(
    `browser_audit=passed assets=${results.length} url=${baseUrl} ` +
      `requireWebGpu=${auditOptions.requireWebGpu} webGpuFlags=${JSON.stringify(auditOptions.webGpuFlags)}`,
  );
} finally {
  if (server) {
    stopDevServer(server);
  }
}

async function runAudit(url, assetsToCheck, options) {
  const browser = await chromium.launch(launchOptions(options));
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  const results = [];
  try {
    for (const asset of assetsToCheck) {
      await page.goto(url, { waitUntil: "networkidle" });
      const title = await page.title();
      if (title !== "ObjGauss 查看器") {
        throw new Error(`unexpected page title: ${title}`);
      }
      await expectText(page, "素材库");
      await expectNoFrameworkOverlay(page);
      const card = page.locator("article.assetCard").filter({ hasText: asset.name }).first();
      await card.getByRole("button", { name: "加载" }).click();
      await page.waitForFunction(
        (fileName) => document.body.innerText.includes(fileName),
        asset.fileName,
        { timeout: 15000 },
      );
      await page.waitForTimeout(1800);
      const splatPixels = await waitForNonBackgroundPixels(page);
      if (splatPixels <= 0) {
        throw new Error(`${asset.id} splat canvas appears blank: ${splatPixels}`);
      }
      const splatRendererId = await page.locator(".splatViewport").first().getAttribute("data-renderer");
      if (splatRendererId !== "spark-splat") {
        throw new Error(`${asset.id} did not expose Spark renderer id: ${splatRendererId}`);
      }

      await page.getByLabel("渲染模式").selectOption("clustered");
      const editRenderer = await labeledValue(page, "渲染器");
      if (!["Gaussian OIT 编辑", "WebGPU Tile 编辑"].includes(editRenderer)) {
        throw new Error(`${asset.id} did not enter a known edit renderer: ${editRenderer}`);
      }
      await page.waitForFunction(() => {
        const viewport = document.querySelector(".viewport");
        return viewport?.getAttribute("data-webgpu-status") !== "pending";
      }, undefined, { timeout: 15000 });
      const viewport = page.locator(".viewport").first();
      const editRendererId = await viewport.getAttribute("data-renderer");
      if (!["gaussian-oit", "webgpu-tile"].includes(editRendererId ?? "")) {
        throw new Error(`${asset.id} did not expose a known edit renderer id: ${editRendererId}`);
      }
      const rendererTarget = await viewport.getAttribute("data-renderer-target");
      if (rendererTarget !== "webgpu-tile") {
        throw new Error(`${asset.id} did not expose WebGPU tile target: ${rendererTarget}`);
      }
      const targetGate = await viewport.getAttribute("data-webgpu-target-gate");
      const targetGateReason = await viewport.getAttribute("data-webgpu-target-gate-reason");
      const targetGateBlocker = await viewport.getAttribute("data-webgpu-target-gate-blocker");
      const webgpuStatus = await viewport.getAttribute("data-webgpu-status");
      const storageLimitGate = await viewport.getAttribute("data-webgpu-storage-limit-gate");
      const storageLimitReason = await viewport.getAttribute("data-webgpu-storage-limit-reason");
      const storageLimitBlocker = await viewport.getAttribute("data-webgpu-storage-limit-blocker");
      const storageLimitMaxBufferSize = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-max-buffer-size") ?? "0");
      const storageLimitMaxBindingSize = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-max-binding-size") ?? "0");
      const storageLimitEffectiveMaxBufferSize = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-effective-max-buffer-size") ?? "0");
      const storageEstimatedLayout = await viewport.getAttribute("data-webgpu-storage-estimated-layout");
      const storageEstimatedBufferCount = numericValue(await viewport.getAttribute("data-webgpu-storage-estimated-buffer-count") ?? "0");
      const storageEstimatedByteSize = numericValue(await viewport.getAttribute("data-webgpu-storage-estimated-byte-size") ?? "0");
      const storageEstimatedMaxBufferByteSize = numericValue(await viewport.getAttribute("data-webgpu-storage-estimated-max-buffer-byte-size") ?? "0");
      const storageEstimatedMaxBufferKey = await viewport.getAttribute("data-webgpu-storage-estimated-max-buffer-key");
      if (!["available", "unavailable"].includes(webgpuStatus)) {
        throw new Error(`${asset.id} WebGPU capability status was not resolved: ${webgpuStatus}`);
      }
      const fallbackReason = await viewport.getAttribute("data-renderer-fallback-reason");
      if (editRendererId !== "webgpu-tile" && !fallbackReason) {
        throw new Error(`${asset.id} did not expose a renderer fallback reason`);
      }
      if (!["blocked", "pass"].includes(targetGate ?? "") || !targetGateReason) {
        throw new Error(
          `${asset.id} did not expose hardened WebGPU target gate: gate=${targetGate} reason=${targetGateReason} blocker=${targetGateBlocker}`,
        );
      }
      if (editRendererId === "webgpu-tile" && (targetGate !== "pass" || targetGateBlocker)) {
        throw new Error(
          `${asset.id} entered WebGPU renderer without a pass gate: gate=${targetGate} blocker=${targetGateBlocker}`,
        );
      }
      if (editRendererId !== "webgpu-tile" && (targetGate !== "blocked" || !targetGateBlocker)) {
        throw new Error(
          `${asset.id} fallback renderer did not expose blocked gate: gate=${targetGate} blocker=${targetGateBlocker}`,
        );
      }
      if (webgpuStatus !== "available" && targetGateBlocker !== "webgpu-capability") {
        throw new Error(
          `${asset.id} WebGPU unavailable but target gate was not capability-blocked: status=${webgpuStatus} blocker=${targetGateBlocker}`,
        );
      }
      if (options.requireWebGpu) {
        if (
          webgpuStatus !== "available" ||
          editRendererId !== "webgpu-tile" ||
          targetGate !== "pass" ||
          targetGateBlocker ||
          fallbackReason
        ) {
          throw new Error(
            `${asset.id} did not satisfy required WebGPU runtime route: status=${webgpuStatus} renderer=${editRendererId} gate=${targetGate} blocker=${targetGateBlocker} fallback=${fallbackReason} flags=${options.webGpuFlags}`,
          );
        }
      }
      if (
        !["unknown", "pass", "blocked"].includes(storageLimitGate ?? "") ||
        !storageLimitReason ||
        storageEstimatedLayout !== "webgpu-tile-storage-v1" ||
        storageEstimatedBufferCount < 11 ||
        storageEstimatedByteSize <= 0 ||
        storageEstimatedMaxBufferByteSize <= 0 ||
        !storageEstimatedMaxBufferKey
      ) {
        throw new Error(
          `${asset.id} invalid WebGPU storage limit telemetry: gate=${storageLimitGate} reason=${storageLimitReason} blocker=${storageLimitBlocker} layout=${storageEstimatedLayout} buffers=${storageEstimatedBufferCount} bytes=${storageEstimatedByteSize} max=${storageEstimatedMaxBufferKey}:${storageEstimatedMaxBufferByteSize}`,
        );
      }
      if (webgpuStatus !== "available") {
        if (storageLimitGate !== "unknown" || storageLimitBlocker !== "webgpu-capability") {
          throw new Error(
            `${asset.id} WebGPU unavailable but storage limit gate was not capability-unknown: gate=${storageLimitGate} blocker=${storageLimitBlocker}`,
          );
        }
      } else if (storageLimitGate === "blocked") {
        if (targetGate !== "blocked" || targetGateBlocker !== "webgpu-buffer-limit" || fallbackReason !== "webgpu-buffer-limit") {
          throw new Error(
            `${asset.id} storage limit blocked but target gate/fallback did not expose buffer-limit: gate=${targetGate} blocker=${targetGateBlocker} fallback=${fallbackReason}`,
          );
        }
      } else if (storageLimitGate !== "pass") {
        throw new Error(`${asset.id} unexpected WebGPU storage limit gate for available device: ${storageLimitGate}`);
      }
      const tileSmokeLayout = await viewport.getAttribute("data-webgpu-pack-layout");
      if (tileSmokeLayout !== "webgpu-tile-smoke-v1") {
        throw new Error(`${asset.id} did not expose WebGPU tile smoke layout: ${tileSmokeLayout}`);
      }
      const packedGaussians = numericValue(await viewport.getAttribute("data-webgpu-packed-gaussians") ?? "0");
      const binnedGaussians = numericValue(await viewport.getAttribute("data-webgpu-binned-gaussians") ?? "0");
      const visibleGaussians = numericValue(await viewport.getAttribute("data-webgpu-visible-gaussians") ?? "0");
      const tileSize = numericValue(await viewport.getAttribute("data-webgpu-tile-size") ?? "0");
      const tileCount = numericValue(await viewport.getAttribute("data-webgpu-tile-count") ?? "0");
      const activeTileCount = numericValue(await viewport.getAttribute("data-webgpu-active-tile-count") ?? "0");
      const tileReferenceCount = numericValue(await viewport.getAttribute("data-webgpu-tile-reference-count") ?? "0");
      const tileOverflowCount = numericValue(await viewport.getAttribute("data-tile-overflow-count") ?? "0");
      const tileOverflowTileCount = numericValue(await viewport.getAttribute("data-webgpu-tile-overflow-tile-count") ?? "0");
      const tileOverflowRatio = Number(await viewport.getAttribute("data-webgpu-tile-overflow-ratio") ?? "0");
      const tileOverflowMaxExcess = numericValue(await viewport.getAttribute("data-webgpu-tile-overflow-max-excess") ?? "0");
      const tileEntryStoredCount = numericValue(await viewport.getAttribute("data-webgpu-tile-entry-stored-count") ?? "0");
      const tileEntryCapacity = numericValue(await viewport.getAttribute("data-webgpu-tile-entry-capacity") ?? "0");
      const tileEntryUtilization = Number(await viewport.getAttribute("data-webgpu-tile-entry-utilization") ?? "0");
      const tileEntryLayout = await viewport.getAttribute("data-webgpu-tile-entry-layout");
      const tileEntryOffsetCount = numericValue(await viewport.getAttribute("data-webgpu-tile-entry-offset-count") ?? "0");
      const tileCapacityMode = await viewport.getAttribute("data-webgpu-tile-capacity-mode");
      const tileCapacityStatus = await viewport.getAttribute("data-webgpu-tile-capacity-status");
      const tileCapacityGate = await viewport.getAttribute("data-webgpu-tile-capacity-gate");
      const maxTileOccupancy = numericValue(await viewport.getAttribute("data-webgpu-max-tile-occupancy") ?? "0");
      const resolveLayout = await viewport.getAttribute("data-webgpu-resolve-layout");
      const resolveMode = await viewport.getAttribute("data-webgpu-resolve-mode");
      const resolvedTileCount = numericValue(await viewport.getAttribute("data-webgpu-resolved-tile-count") ?? "0");
      const resolveWeightSum = Number(await viewport.getAttribute("data-webgpu-resolve-weight-sum") ?? "0");
      const resolveAlphaMean = Number(await viewport.getAttribute("data-webgpu-resolve-alpha-mean") ?? "0");
      const resolveLumaMean = Number(await viewport.getAttribute("data-webgpu-resolve-luma-mean") ?? "0");
      const resolveChecksum = await viewport.getAttribute("data-webgpu-resolve-checksum");
      if (tileSize !== 16) {
        throw new Error(`${asset.id} unexpected WebGPU tile size: ${tileSize}`);
      }
      if (packedGaussians <= 0 || binnedGaussians <= 0 || visibleGaussians <= 0) {
        throw new Error(
          `${asset.id} did not expose positive WebGPU pack/bin counts: packed=${packedGaussians} visible=${visibleGaussians} binned=${binnedGaussians}`,
        );
      }
      if (tileCount <= 0 || activeTileCount <= 0 || tileReferenceCount < binnedGaussians) {
        throw new Error(
          `${asset.id} invalid WebGPU tile occupancy: tiles=${activeTileCount}/${tileCount} refs=${tileReferenceCount} binned=${binnedGaussians}`,
        );
      }
      if (maxTileOccupancy <= 0) {
        throw new Error(`${asset.id} did not expose positive max tile occupancy: ${maxTileOccupancy}`);
      }
      if (
        !["compact-offset-list", "fixed-cap-smoke"].includes(tileCapacityMode ?? "") ||
        !["compact-offset-list", "fixed-cap-smoke"].includes(tileEntryLayout ?? "") ||
        !["ok", "overflow"].includes(tileCapacityStatus ?? "") ||
        !["pass", "blocked"].includes(tileCapacityGate ?? "") ||
        tileEntryStoredCount <= 0 ||
        tileEntryCapacity <= 0 ||
        (tileEntryLayout === "compact-offset-list" && tileEntryStoredCount !== tileEntryCapacity) ||
        (tileEntryLayout === "compact-offset-list" && tileEntryOffsetCount !== 0 && tileEntryOffsetCount !== tileCount) ||
        tileEntryUtilization <= 0 ||
        tileEntryUtilization > 1
      ) {
        throw new Error(
          `${asset.id} invalid tile capacity telemetry: mode=${tileCapacityMode} layout=${tileEntryLayout} offsets=${tileEntryOffsetCount} status=${tileCapacityStatus} gate=${tileCapacityGate} stored=${tileEntryStoredCount} capacity=${tileEntryCapacity} utilization=${tileEntryUtilization}`,
        );
      }
      if (tileOverflowCount > 0) {
        if (
          tileCapacityStatus !== "overflow" ||
          tileCapacityGate !== "blocked" ||
          tileOverflowTileCount <= 0 ||
          tileOverflowRatio <= 0 ||
          tileOverflowMaxExcess <= 0
        ) {
          throw new Error(
            `${asset.id} overflow telemetry did not block tile capacity: overflow=${tileOverflowCount} overflowTiles=${tileOverflowTileCount} ratio=${tileOverflowRatio} maxExcess=${tileOverflowMaxExcess} status=${tileCapacityStatus} gate=${tileCapacityGate}`,
          );
        }
        if (webgpuStatus === "available" && targetGateBlocker !== "tile-overflow") {
          throw new Error(`${asset.id} WebGPU available with overflow but target gate blocker was ${targetGateBlocker}`);
        }
      } else if (
        tileCapacityStatus !== "ok" ||
        tileCapacityGate !== "pass" ||
        tileOverflowTileCount !== 0 ||
        tileOverflowRatio !== 0 ||
        tileOverflowMaxExcess !== 0
      ) {
        throw new Error(
          `${asset.id} non-overflow telemetry was not pass/ok: overflowTiles=${tileOverflowTileCount} ratio=${tileOverflowRatio} maxExcess=${tileOverflowMaxExcess} status=${tileCapacityStatus} gate=${tileCapacityGate}`,
        );
      }
      if (resolveLayout !== "webgpu-tile-resolve-v1" || resolveMode !== "tile-2x2-covariance-weighted-oit") {
        throw new Error(
          `${asset.id} did not expose WebGPU tile resolve contract: layout=${resolveLayout} mode=${resolveMode}`,
        );
      }
      if (
        resolvedTileCount <= 0 ||
        resolveWeightSum <= 0 ||
        resolveAlphaMean <= 0 ||
        resolveLumaMean <= 0 ||
        !/^[0-9a-f]{8}$/.test(resolveChecksum ?? "")
      ) {
        throw new Error(
          `${asset.id} invalid WebGPU tile resolve telemetry: resolved=${resolvedTileCount} weight=${resolveWeightSum} alpha=${resolveAlphaMean} luma=${resolveLumaMean} checksum=${resolveChecksum}`,
        );
      }
      const objectFilter = await viewport.getAttribute("data-object-filter");
      const expectedObjectFilter =
        editRendererId === "webgpu-tile" ? "gpu-object-state-buffer" : "gpu-object-state-texture";
      if (objectFilter !== expectedObjectFilter) {
        throw new Error(`${asset.id} did not expose expected object filtering: ${objectFilter}`);
      }
      const objectFilterTarget = await viewport.getAttribute("data-webgpu-object-filter-target");
      if (objectFilterTarget !== "gpu-object-state-buffer") {
        throw new Error(`${asset.id} did not expose WebGPU object-state buffer target: ${objectFilterTarget}`);
      }
      const objectStateLayout = await viewport.getAttribute("data-webgpu-object-state-layout");
      const objectStateStride = numericValue(await viewport.getAttribute("data-webgpu-object-state-stride") ?? "0");
      const objectStateVisibleObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-visible-objects") ?? "0");
      const objectStateHiddenObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-hidden-objects") ?? "0");
      const objectStateRemovedObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-removed-objects") ?? "0");
      const objectStateSelectedObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-selected-objects") ?? "0");
      const objectStateIsolatedObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-isolated-objects") ?? "0");
      const objectStateChecksum = await viewport.getAttribute("data-webgpu-object-state-checksum");
      if (
        objectStateLayout !== "webgpu-object-state-v1" ||
        objectStateStride !== 4 ||
        objectStateVisibleObjects <= 0 ||
        objectStateHiddenObjects !== 0 ||
        objectStateRemovedObjects !== 0 ||
        objectStateSelectedObjects !== 0 ||
        objectStateIsolatedObjects !== 0 ||
        !/^[0-9a-f]{8}$/.test(objectStateChecksum ?? "")
      ) {
        throw new Error(
          `${asset.id} invalid initial WebGPU object-state buffer: layout=${objectStateLayout} stride=${objectStateStride} visible=${objectStateVisibleObjects} hidden=${objectStateHiddenObjects} removed=${objectStateRemovedObjects} selected=${objectStateSelectedObjects} isolated=${objectStateIsolatedObjects} checksum=${objectStateChecksum}`,
        );
      }
      const webGpuFirstFrameStatus = await viewport.getAttribute("data-webgpu-first-frame-status");
      const webGpuFirstFrameReason = await viewport.getAttribute("data-webgpu-first-frame-reason");
      const webGpuFirstFramePixels = numericValue(await viewport.getAttribute("data-webgpu-first-frame-pixels") ?? "0");
      const webGpuFirstFrameChecksum = await viewport.getAttribute("data-webgpu-first-frame-checksum");
      const webGpuResolveSource = await viewport.getAttribute("data-webgpu-resolve-source");
      const webGpuAccumulationSource = await viewport.getAttribute("data-webgpu-accumulation-source");
      const webGpuAccumulationStatus = await viewport.getAttribute("data-webgpu-accumulation-status");
      const webGpuAccumulationReason = await viewport.getAttribute("data-webgpu-accumulation-reason");
      const webGpuAccumulationWorkgroups = numericValue(await viewport.getAttribute("data-webgpu-accumulation-workgroups") ?? "0");
      const webGpuComputeSource = await viewport.getAttribute("data-webgpu-compute-source");
      const webGpuComputeStatus = await viewport.getAttribute("data-webgpu-compute-status");
      const webGpuComputeReason = await viewport.getAttribute("data-webgpu-compute-reason");
      const webGpuComputeWorkgroups = numericValue(await viewport.getAttribute("data-webgpu-compute-workgroups") ?? "0");
      const webGpuPixelSource = await viewport.getAttribute("data-webgpu-pixel-source");
      const webGpuPixelStatus = await viewport.getAttribute("data-webgpu-pixel-status");
      const webGpuPixelReason = await viewport.getAttribute("data-webgpu-pixel-reason");
      const webGpuPixelWorkgroups = numericValue(await viewport.getAttribute("data-webgpu-pixel-workgroups") ?? "0");
      const webGpuStorageLayout = await viewport.getAttribute("data-webgpu-storage-layout");
      const webGpuStorageStatus = await viewport.getAttribute("data-webgpu-storage-status");
      const webGpuStorageReason = await viewport.getAttribute("data-webgpu-storage-reason");
      const webGpuStorageBufferCount = numericValue(await viewport.getAttribute("data-webgpu-storage-buffer-count") ?? "0");
      const webGpuStorageByteSize = numericValue(await viewport.getAttribute("data-webgpu-storage-byte-size") ?? "0");
      const webGpuStorageChecksum = await viewport.getAttribute("data-webgpu-storage-checksum");
      const webGpuStorageTileEntries = await viewport.getAttribute("data-webgpu-storage-tile-entries");
      const webGpuStorageTileOffsets = await viewport.getAttribute("data-webgpu-storage-tile-offsets");
      const webGpuStoragePixelOutput = await viewport.getAttribute("data-webgpu-storage-pixel-output");
      if (editRendererId === "webgpu-tile") {
        if (
          webGpuAccumulationSource !== "webgpu-compute-covariance-accumulation-v1" ||
          webGpuAccumulationStatus !== "dispatched" ||
          webGpuAccumulationWorkgroups <= 0 ||
          webGpuComputeSource !== "webgpu-compute-resolve-v1" ||
          webGpuComputeStatus !== "dispatched" ||
          webGpuComputeWorkgroups <= 0 ||
          webGpuPixelSource !== "webgpu-compute-pixel-accumulation-v1" ||
          webGpuPixelStatus !== "dispatched" ||
          webGpuPixelWorkgroups <= 0 ||
          webGpuFirstFrameStatus !== "rendered" ||
          webGpuFirstFramePixels <= 0 ||
          !/^[0-9a-f]{8}$/.test(webGpuFirstFrameChecksum ?? "") ||
          webGpuResolveSource !== "webgpu-pixel-storage-resolve-v1"
        ) {
          throw new Error(
            `${asset.id} WebGPU first frame did not render through accumulation/compute/pixel/storage resolve: accumulation=${webGpuAccumulationStatus}:${webGpuAccumulationSource}:${webGpuAccumulationWorkgroups}:${webGpuAccumulationReason} compute=${webGpuComputeStatus}:${webGpuComputeSource}:${webGpuComputeWorkgroups}:${webGpuComputeReason} pixel=${webGpuPixelStatus}:${webGpuPixelSource}:${webGpuPixelWorkgroups}:${webGpuPixelReason} frame=${webGpuFirstFrameStatus}:${webGpuFirstFrameReason} pixels=${webGpuFirstFramePixels} checksum=${webGpuFirstFrameChecksum} source=${webGpuResolveSource}`,
          );
        }
        if (
          webGpuStorageLayout !== "webgpu-tile-storage-v1" ||
          webGpuStorageStatus !== "uploaded" ||
          webGpuStorageBufferCount < 11 ||
          webGpuStorageByteSize <= 0 ||
          webGpuStorageTileEntries !== "true" ||
          webGpuStorageTileOffsets !== "true" ||
          webGpuStoragePixelOutput !== "true" ||
          !/^[0-9a-f]{8}$/.test(webGpuStorageChecksum ?? "")
        ) {
          throw new Error(
            `${asset.id} WebGPU storage buffers were not uploaded with tile entries, offsets, and pixel output: layout=${webGpuStorageLayout} status=${webGpuStorageStatus} reason=${webGpuStorageReason} buffers=${webGpuStorageBufferCount} bytes=${webGpuStorageByteSize} tileEntries=${webGpuStorageTileEntries} tileOffsets=${webGpuStorageTileOffsets} pixelOutput=${webGpuStoragePixelOutput} checksum=${webGpuStorageChecksum}`,
          );
        }
      }
      const editPixels =
        editRendererId === "webgpu-tile"
          ? webGpuFirstFramePixels
          : await waitForNonBackgroundPixels(page);
      if (editPixels <= 0) {
        throw new Error(`${asset.id} point-edit canvas appears blank: ${editPixels}`);
      }
      const canvasSelectedObject = await selectObjectFromCanvas(page, asset.id);
      await page.getByRole("button", { name: "只看所选" }).click();
      await page.waitForTimeout(300);
      const visibleAfterIsolate = await labeledValue(page, "可见");
      const objectStateChecksumAfterIsolate = await viewport.getAttribute("data-webgpu-object-state-checksum");
      const objectStateVisibleAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-object-state-visible-objects") ?? "0");
      const objectStateHiddenAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-object-state-hidden-objects") ?? "0");
      const objectStateSelectedAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-object-state-selected-objects") ?? "0");
      const objectStateIsolatedAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-object-state-isolated-objects") ?? "0");
      const webGpuStorageChecksumAfterIsolate = await viewport.getAttribute("data-webgpu-storage-checksum");
      if (
        objectStateChecksumAfterIsolate === objectStateChecksum ||
        objectStateVisibleAfterIsolate !== 1 ||
        objectStateHiddenAfterIsolate <= 0 ||
        objectStateSelectedAfterIsolate !== 1 ||
        objectStateIsolatedAfterIsolate !== 1
      ) {
        throw new Error(
          `${asset.id} isolate did not update WebGPU object-state buffer: checksum=${objectStateChecksumAfterIsolate} visible=${objectStateVisibleAfterIsolate} hidden=${objectStateHiddenAfterIsolate} selected=${objectStateSelectedAfterIsolate} isolated=${objectStateIsolatedAfterIsolate}`,
        );
      }
      if (editRendererId === "webgpu-tile" && webGpuStorageChecksumAfterIsolate === webGpuStorageChecksum) {
        throw new Error(`${asset.id} isolate did not update WebGPU storage checksum`);
      }
      await page.getByRole("button", { name: "预览删除" }).click();
      await page.waitForTimeout(300);
      const deletedObjects = await labeledValue(page, "已删除对象");
      const visibleAfterDelete = await labeledValue(page, "可见");
      const renderModeAfterDelete = await labeledValue(page, "模式");
      const objectStateChecksumAfterDelete = await viewport.getAttribute("data-webgpu-object-state-checksum");
      const objectStateVisibleAfterDelete = numericValue(await viewport.getAttribute("data-webgpu-object-state-visible-objects") ?? "0");
      const objectStateRemovedAfterDelete = numericValue(await viewport.getAttribute("data-webgpu-object-state-removed-objects") ?? "0");
      const objectStateIsolatedAfterDelete = numericValue(await viewport.getAttribute("data-webgpu-object-state-isolated-objects") ?? "0");
      const webGpuStorageChecksumAfterDelete = await viewport.getAttribute("data-webgpu-storage-checksum");
      if (deletedObjects !== "1") {
        throw new Error(`${asset.id} delete preview did not update: ${deletedObjects}`);
      }
      if (numericValue(visibleAfterDelete) <= 0) {
        throw new Error(`${asset.id} delete preview did not show remaining scene`);
      }
      if (renderModeAfterDelete !== "原始颜色（编辑预览）") {
        throw new Error(`${asset.id} delete preview did not restore edit-preview original colors`);
      }
      if (
        objectStateChecksumAfterDelete === objectStateChecksumAfterIsolate ||
        objectStateVisibleAfterDelete !== objectStateVisibleObjects - 1 ||
        objectStateRemovedAfterDelete !== 1 ||
        objectStateIsolatedAfterDelete !== 0
      ) {
        throw new Error(
          `${asset.id} delete did not update WebGPU object-state buffer: checksum=${objectStateChecksumAfterDelete} visible=${objectStateVisibleAfterDelete} removed=${objectStateRemovedAfterDelete} isolated=${objectStateIsolatedAfterDelete}`,
        );
      }
      if (editRendererId === "webgpu-tile" && webGpuStorageChecksumAfterDelete === webGpuStorageChecksumAfterIsolate) {
        throw new Error(`${asset.id} delete did not update WebGPU storage checksum`);
      }
      const screenshotPath = `/tmp/objgauss-audit-${asset.id}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      results.push({
        assetId: asset.id,
        title,
        splatPixels,
        splatRendererId,
        editPixels,
        editRenderer,
        editRendererId,
        webGpuFirstFrameStatus,
        webGpuFirstFramePixels,
        webGpuFirstFrameChecksum,
        webGpuResolveSource,
        webGpuAccumulationSource,
        webGpuAccumulationStatus,
        webGpuAccumulationWorkgroups,
        webGpuComputeSource,
        webGpuComputeStatus,
        webGpuComputeWorkgroups,
        webGpuPixelSource,
        webGpuPixelStatus,
        webGpuPixelWorkgroups,
        webGpuStorageLayout,
        webGpuStorageStatus,
        webGpuStorageBufferCount,
        webGpuStorageByteSize,
        webGpuStorageTileEntries,
        webGpuStorageTileOffsets,
        webGpuStoragePixelOutput,
        webGpuStorageChecksum,
        webGpuStorageChecksumAfterIsolate,
        webGpuStorageChecksumAfterDelete,
        storageLimitGate,
        storageLimitReason,
        storageLimitBlocker,
        storageLimitMaxBufferSize,
        storageLimitMaxBindingSize,
        storageLimitEffectiveMaxBufferSize,
        storageEstimatedLayout,
        storageEstimatedBufferCount,
        storageEstimatedByteSize,
        storageEstimatedMaxBufferByteSize,
        storageEstimatedMaxBufferKey,
        rendererTarget,
        targetGate,
        targetGateReason,
        targetGateBlocker,
        webgpuStatus,
        fallbackReason,
        tileSmokeLayout,
        packedGaussians,
        visibleGaussians,
        binnedGaussians,
        tileSize,
        tileCount,
        activeTileCount,
        tileReferenceCount,
        maxTileOccupancy,
        tileOverflowTileCount,
        tileOverflowRatio,
        tileOverflowMaxExcess,
        tileEntryStoredCount,
        tileEntryCapacity,
        tileEntryUtilization,
        tileEntryLayout,
        tileEntryOffsetCount,
        tileCapacityMode,
        tileCapacityStatus,
        tileCapacityGate,
        resolveLayout,
        resolveMode,
        resolvedTileCount,
        resolveWeightSum,
        resolveAlphaMean,
        resolveLumaMean,
        resolveChecksum,
        tileOverflowCount,
        objectFilter,
        objectFilterTarget,
        objectStateLayout,
        objectStateStride,
        objectStateVisibleObjects,
        objectStateHiddenObjects,
        objectStateRemovedObjects,
        objectStateSelectedObjects,
        objectStateIsolatedObjects,
        objectStateChecksum,
        objectStateChecksumAfterIsolate,
        objectStateChecksumAfterDelete,
        canvasSelectedObject,
        visibleAfterIsolate,
        visibleAfterDelete,
        renderModeAfterDelete,
        deletedObjects,
        screenshotPath,
      });
    }
    const relevantIssues = consoleIssues.filter(
      (issue) =>
        !issue.includes("THREE.WebGLRenderer") &&
        !issue.includes("GPU stall due to ReadPixels") &&
        !issue.includes("No available adapters."),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }
    return results;
  } finally {
    await browser.close();
  }
}

function launchOptions(options = {}) {
  const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ?? firstExisting([
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ]);
  const args = ["--no-sandbox", ...webGpuLaunchArgs(options.webGpuFlags)];
  return executablePath ? { executablePath, args } : { args };
}

function webGpuLaunchArgs(mode) {
  if (mode === "none" || mode === "false" || mode === "") return [];
  if (mode === "unsafe") {
    return ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"];
  }
  if (mode === "vulkan") {
    return [
      "--enable-unsafe-webgpu",
      "--ignore-gpu-blocklist",
      "--enable-features=Vulkan,DefaultANGLEVulkan,WebGPUDeveloperFeatures",
      "--use-angle=vulkan",
    ];
  }
  throw new Error(`unsupported WebGPU launch flags mode: ${mode}`);
}

function firstExisting(paths) {
  return paths.find((path) => existsSync(path));
}

function startDevServer(port) {
  const child = spawn(
    "npm",
    ["run", "dev", "--", "--port", String(port), "--strictPort"],
    { detached: true, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

function stopDevServer(child) {
  if (!child.pid) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    child.kill("SIGTERM");
  }
}

async function waitForApp(url) {
  const deadline = Date.now() + 30000;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  throw new Error(`app did not become ready at ${url}: ${lastError?.message ?? "timeout"}`);
}

async function expectText(page, text) {
  await page.getByText(text).first().waitFor({ timeout: 15000 });
}

async function expectNoFrameworkOverlay(page) {
  const bodyText = await page.locator("body").innerText();
  const overlaySignals = ["Vite Error", "Internal server error", "Failed to resolve import"];
  for (const signal of overlaySignals) {
    if (bodyText.includes(signal)) {
      throw new Error(`framework overlay detected: ${signal}`);
    }
  }
}

async function nonBackgroundPixels(page) {
  const webGpuPixels = await page.locator(".viewport").first().evaluate((viewport) => {
    if (viewport?.getAttribute("data-renderer") !== "webgpu-tile") return 0;
    return Number(viewport.getAttribute("data-webgpu-first-frame-pixels") ?? "0");
  }).catch(() => 0);
  if (webGpuPixels > 0) return webGpuPixels;

  return page.locator("canvas").evaluateAll((canvases) => {
    let maxCount = -1;
    for (const canvas of canvases) {
      const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
      if (!gl) continue;
      const width = canvas.width;
      const height = canvas.height;
      const pixels = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      let count = 0;
      for (let index = 0; index < pixels.length; index += 4) {
        if (pixels[index] !== 16 || pixels[index + 1] !== 19 || pixels[index + 2] !== 22) {
          count += 1;
        }
      }
      maxCount = Math.max(maxCount, count);
    }
    return maxCount;
  });
}

async function waitForNonBackgroundPixels(page, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  let pixels = 0;
  while (Date.now() < deadline) {
    pixels = await nonBackgroundPixels(page);
    if (pixels > 0) {
      return pixels;
    }
    await page.waitForTimeout(250);
  }
  return pixels;
}

async function labeledValue(page, label) {
  const value = await page.locator(".metric, .stateRow").evaluateAll((rows, targetLabel) => {
    const row = rows.find(
      (candidate) => candidate.querySelector("span")?.textContent?.trim() === targetLabel,
    );
    return row?.querySelector("strong")?.textContent?.trim() ?? "";
  }, label);
  if (!value) {
    throw new Error(`missing labeled value: ${label}`);
  }
  return value;
}

function numericValue(value) {
  return Number(value.replace(/[^\d]/g, ""));
}

async function selectObjectFromCanvas(page, assetId) {
  const canvas = page.locator(".viewport canvas").first();
  const box = await canvas.boundingBox();
  if (!box) {
    throw new Error(`${assetId} point-edit canvas is missing`);
  }

  const clickPoints = [
    [0.5, 0.5],
    [0.45, 0.48],
    [0.55, 0.48],
    [0.4, 0.55],
    [0.6, 0.55],
    [0.5, 0.4],
    [0.35, 0.48],
    [0.65, 0.48],
  ];
  for (const [xRatio, yRatio] of clickPoints) {
    await page.mouse.click(box.x + box.width * xRatio, box.y + box.height * yRatio);
    await page.waitForTimeout(250);
    const selectedObject = await selectedObjectValue(page);
    if (selectedObject !== "无") {
      return selectedObject;
    }
  }
  throw new Error(`${assetId} canvas selection did not choose an object`);
}

async function selectedObjectValue(page) {
  const status = await page.locator(".statusBar").innerText();
  const match = status.match(/所选：([^\n]+)/);
  return match?.[1] ?? "无";
}

function parseArgs(rawArgs) {
  const parsed = {};
  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = rawArgs[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}
