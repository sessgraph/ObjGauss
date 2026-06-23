import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { inflateSync } from "node:zlib";

import { chromium } from "playwright";
import { WEBGPU_PIXEL_RESOLVE_SOURCE } from "../src/webgpuTileComputeShader.js";
import {
  WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR,
  WEBGPU_TILE_ALPHA_PRESENTATION_MODE,
} from "../src/webgpuTileResolveShader.js";
import {
  normalizeWebGpuRuntimeProbe,
  WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY,
  WEBGPU_RUNTIME_PROBE_CLEAR_ONLY,
  WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY,
  WEBGPU_RUNTIME_PROBE_FULL,
  WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY,
  WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY,
  WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY,
  WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY,
  WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY,
  WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT,
} from "../src/webgpuRuntimeProbe.js";

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
const DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE = 320;
const VISUAL_RESIDUAL_MODE = "spark-edit-visual-residual-v1";

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = args.url ?? `http://127.0.0.1:${port}/`;
const assets = args.asset ? KNOWN_ASSETS.filter((asset) => asset.id === args.asset) : DEFAULT_ASSETS;
const auditOptions = {
  requireWebGpu: Boolean(args.requireWebgpu ?? args["require-webgpu"]),
  webGpuFlags: String(args.webgpuFlags ?? args["webgpu-flags"] ?? process.env.OBJGAUSS_WEBGPU_FLAGS ?? "none"),
  headed: flagEnabled(args.headed ?? args.headful ?? process.env.OBJGAUSS_PLAYWRIGHT_HEADED),
  browserChannel: optionalString(args.browserChannel ?? args["browser-channel"] ?? process.env.PLAYWRIGHT_BROWSER_CHANNEL),
  executablePath: optionalString(args.executablePath ?? args["executable-path"] ?? process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE),
  slowMo: Number(args.slowMo ?? args["slow-mo"] ?? 0),
  webGpuProbe: normalizeWebGpuRuntimeProbe(
    args.webgpuProbe ?? args["webgpu-probe"] ?? process.env.OBJGAUSS_WEBGPU_PROBE,
  ),
  webGpuViewportSize: optionalPositiveInteger(
    args.webgpuViewportSize ??
      args["webgpu-viewport-size"] ??
      process.env.OBJGAUSS_WEBGPU_VIEWPORT_SIZE,
  ),
  allowWebGpuDeviceLost: Boolean(
    args.allowWebgpuDeviceLost ?? args["allow-webgpu-device-lost"],
  ),
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
        `visualResidual=${JSON.stringify(result.visualResidualMode)}:${result.sparkVisualCoverage}/${result.editOriginalVisualCoverage}:${result.sparkEditCoverageRatio}:${result.sparkEditLumaDelta}:${result.sparkEditChromaDelta} ` +
        `editRenderer=${JSON.stringify(result.editRenderer)} ` +
        `editRendererId=${JSON.stringify(result.editRendererId)} ` +
        `runtimeProbe=${JSON.stringify(result.webGpuRuntimeProbe)} ` +
        `firstFrame=${JSON.stringify(result.webGpuFirstFrameStatus)}:${result.webGpuFirstFramePixels} ` +
        `webgpuViewport=${result.webGpuViewportWidth}x${result.webGpuViewportHeight}:${result.webGpuPixelCount}:${JSON.stringify(result.webGpuViewportAspectMode)}:${JSON.stringify(result.webGpuViewportQuality)}:${result.webGpuViewportPixelBudget} ` +
        `display=${result.webGpuDisplayWidth}x${result.webGpuDisplayHeight} boundsFit=${JSON.stringify(result.webGpuBoundsFitMode)}:${result.webGpuBoundsWorldAspect}/${result.webGpuBoundsViewportAspect} ` +
        `projection=${JSON.stringify(result.webGpuProjectionMode)}:${result.webGpuProjectionCameraFov} ` +
        `depthWeight=${JSON.stringify(result.webGpuDepthWeightMode)}:${result.webGpuProjectionDepthMin}/${result.webGpuProjectionDepthMax}/${result.webGpuProjectionDepthSpan} ` +
        `pixelDepthSort=${JSON.stringify(result.webGpuPixelDepthSortMode)}:${result.webGpuPixelDepthGateStrength}/${result.webGpuPixelDepthGateFloor}:${result.webGpuPixelDepthBinCount} ` +
        `pixelCoverage=${JSON.stringify(result.webGpuPixelCoverageMode)}:${result.webGpuPixelCoverageWeightFloor}:${result.webGpuPixelCoverageFootprintScale} ` +
        `colorFidelity=${JSON.stringify(result.webGpuColorFidelityMode)}:${result.webGpuColorSourceRgbGaussians}/${result.webGpuColorSourceShDcGaussians}/${result.webGpuColorSourceFallbackGaussians}/${result.webGpuColorSourceObjectGaussians}:${result.webGpuColorOpacityMean} ` +
        `colorAfterDelete=${result.webGpuColorSourceRgbGaussiansAfterDelete}/${result.webGpuColorSourceShDcGaussiansAfterDelete}/${result.webGpuColorSourceFallbackGaussiansAfterDelete}/${result.webGpuColorSourceObjectGaussiansAfterDelete} ` +
        `screenCovariance=${JSON.stringify(result.webGpuScreenCovarianceMode)}:${result.webGpuScreenCovarianceGaussians}/${result.webGpuScreenCovarianceFallbackGaussians}/${result.webGpuScreenCovarianceClampedGaussians}:${result.webGpuScreenCovarianceMaxAnisotropy}:${result.webGpuScreenCovarianceSigmaMean} ` +
        `deviceLost=${JSON.stringify(result.webGpuDeviceLostStatus)}:${JSON.stringify(result.webGpuDeviceLostReason)} ` +
        `deviceError=${JSON.stringify(result.webGpuDeviceErrorStatus)}:${JSON.stringify(result.webGpuDeviceErrorType)} ` +
        `queue=${JSON.stringify(result.webGpuQueueStatus)}:${JSON.stringify(result.webGpuQueueReason)} ` +
        `accumulation=${JSON.stringify(result.webGpuAccumulationStatus)}:${JSON.stringify(result.webGpuAccumulationSource)}:${result.webGpuAccumulationWorkgroups} ` +
        `compute=${JSON.stringify(result.webGpuComputeStatus)}:${JSON.stringify(result.webGpuComputeSource)}:${result.webGpuComputeWorkgroups} ` +
        `pixel=${JSON.stringify(result.webGpuPixelStatus)}:${JSON.stringify(result.webGpuPixelSource)}:${result.webGpuPixelWorkgroups} ` +
        `resolveSource=${JSON.stringify(result.webGpuResolveSource)}:${JSON.stringify(result.webGpuResolveFilter)}:${JSON.stringify(result.webGpuAlphaPresentationMode)}:${result.webGpuAlphaPresentationFloor} ` +
        `storage=${JSON.stringify(result.webGpuStorageStatus)}:${JSON.stringify(result.webGpuStorageChecksum)} ` +
        `storageLimit=${JSON.stringify(result.storageLimitGate)}:${JSON.stringify(result.storageLimitBlocker)}:${JSON.stringify(result.storageEstimatedMaxBufferKey)}:${result.storageEstimatedMaxBufferByteSize}:${result.storageLimitRequiredStorageBuffersPerStage}/${result.storageLimitMaxStorageBuffersPerStage} ` +
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
      `requireWebGpu=${auditOptions.requireWebGpu} webGpuFlags=${JSON.stringify(auditOptions.webGpuFlags)} ` +
      `webGpuProbe=${JSON.stringify(auditOptions.webGpuProbe)} webGpuViewportSize=${auditOptions.webGpuViewportSize ?? "default"} ` +
      `allowWebGpuDeviceLost=${auditOptions.allowWebGpuDeviceLost}`,
  );
} finally {
  if (server) {
    stopDevServer(server);
  }
}

async function runAudit(url, assetsToCheck, options) {
  const browser = await chromium.launch(launchOptions(options));
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const auditUrl = urlWithWebGpuOptions(url, options);
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
      await page.goto(auditUrl, { waitUntil: "networkidle" });
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
      const sparkVisualStats = await canvasVisualStats(page, ".splatViewport canvas");
      validateCanvasVisualStats(asset.id, "Spark", sparkVisualStats);

      await page.locator(".modeTabs").getByRole("button", { name: "对象编辑" }).click();
      await waitForEditViewportReady(page);
      const editOriginalVisualStats = await canvasVisualStats(page, ".viewport canvas");
      validateCanvasVisualStats(asset.id, "edit original", editOriginalVisualStats);
      const visualResidual = compareVisualStats(sparkVisualStats, editOriginalVisualStats);
      validateVisualResidual(asset.id, visualResidual);

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
      if (editRendererId === "webgpu-tile") {
        await page.waitForFunction(() => {
          const activeViewport = document.querySelector(".viewport");
          return activeViewport?.getAttribute("data-webgpu-first-frame-status") !== "pending";
        }, undefined, { timeout: 15000 });
        await waitForWebGpuQueueTelemetry(page);
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
      const storageLimitMaxStorageBuffersPerStage = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-max-storage-buffers-per-stage") ?? "0");
      const storageLimitRequiredStorageBuffersPerStage = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-required-storage-buffers-per-stage") ?? "0");
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
        !storageEstimatedMaxBufferKey ||
        storageLimitRequiredStorageBuffersPerStage <= 0
      ) {
        throw new Error(
          `${asset.id} invalid WebGPU storage limit telemetry: gate=${storageLimitGate} reason=${storageLimitReason} blocker=${storageLimitBlocker} layout=${storageEstimatedLayout} buffers=${storageEstimatedBufferCount} bytes=${storageEstimatedByteSize} max=${storageEstimatedMaxBufferKey}:${storageEstimatedMaxBufferByteSize} storageBuffersPerStage=${storageLimitRequiredStorageBuffersPerStage}/${storageLimitMaxStorageBuffersPerStage}`,
        );
      }
      if (webgpuStatus !== "available") {
        if (storageLimitGate !== "unknown" || storageLimitBlocker !== "webgpu-capability") {
          throw new Error(
            `${asset.id} WebGPU unavailable but storage limit gate was not capability-unknown: gate=${storageLimitGate} blocker=${storageLimitBlocker}`,
          );
        }
      } else if (storageLimitGate === "blocked") {
        if (
          targetGate !== "blocked" ||
          !["webgpu-buffer-limit", "webgpu-binding-limit"].includes(targetGateBlocker ?? "") ||
          fallbackReason !== targetGateBlocker
        ) {
          throw new Error(
            `${asset.id} storage limit blocked but target gate/fallback did not expose buffer-limit: gate=${targetGate} blocker=${targetGateBlocker} fallback=${fallbackReason}`,
          );
        }
      } else if (storageLimitGate !== "pass") {
        throw new Error(`${asset.id} unexpected WebGPU storage limit gate for available device: ${storageLimitGate}`);
      } else if (
        storageLimitMaxStorageBuffersPerStage > 0 &&
        storageLimitMaxStorageBuffersPerStage < storageLimitRequiredStorageBuffersPerStage
      ) {
        throw new Error(
          `${asset.id} WebGPU available but storage buffers per stage are below renderer requirement: ${storageLimitRequiredStorageBuffersPerStage}/${storageLimitMaxStorageBuffersPerStage}`,
        );
      }
      const tileSmokeLayout = await viewport.getAttribute("data-webgpu-pack-layout");
      if (tileSmokeLayout !== "webgpu-tile-smoke-v1") {
        throw new Error(`${asset.id} did not expose WebGPU tile smoke layout: ${tileSmokeLayout}`);
      }
      const webGpuViewportWidth = numericValue(await viewport.getAttribute("data-webgpu-viewport-width") ?? "0");
      const webGpuViewportHeight = numericValue(await viewport.getAttribute("data-webgpu-viewport-height") ?? "0");
      const webGpuPixelCount = numericValue(await viewport.getAttribute("data-webgpu-pixel-count") ?? "0");
      const webGpuViewportAspectMode = await viewport.getAttribute("data-webgpu-viewport-aspect-mode");
      const webGpuViewportQuality = await viewport.getAttribute("data-webgpu-viewport-quality");
      const webGpuViewportPixelBudget = numericValue(await viewport.getAttribute("data-webgpu-viewport-pixel-budget") ?? "0");
      const webGpuDisplayWidth = numericValue(await viewport.getAttribute("data-webgpu-display-width") ?? "0");
      const webGpuDisplayHeight = numericValue(await viewport.getAttribute("data-webgpu-display-height") ?? "0");
      const webGpuBoundsFitMode = await viewport.getAttribute("data-webgpu-bounds-fit-mode");
      const webGpuBoundsPaddingRatio = Number(await viewport.getAttribute("data-webgpu-bounds-padding-ratio") ?? "0");
      const webGpuBoundsViewportAspect = Number(await viewport.getAttribute("data-webgpu-bounds-viewport-aspect") ?? "0");
      const webGpuBoundsWorldAspect = Number(await viewport.getAttribute("data-webgpu-bounds-world-aspect") ?? "0");
      const webGpuProjectionMode = await viewport.getAttribute("data-webgpu-projection-mode");
      const webGpuProjectionCameraFov = Number(await viewport.getAttribute("data-webgpu-projection-camera-fov") ?? "0");
      const webGpuDepthWeightMode = await viewport.getAttribute("data-webgpu-depth-weight-mode");
      const webGpuPixelDepthSortMode = await viewport.getAttribute("data-webgpu-pixel-depth-sort-mode");
      const webGpuPixelDepthGateStrength = Number(await viewport.getAttribute("data-webgpu-pixel-depth-gate-strength") ?? "0");
      const webGpuPixelDepthGateFloor = Number(await viewport.getAttribute("data-webgpu-pixel-depth-gate-floor") ?? "0");
      const webGpuPixelDepthBinCount = Number(await viewport.getAttribute("data-webgpu-pixel-depth-bin-count") ?? "0");
      const webGpuPixelCoverageMode = await viewport.getAttribute("data-webgpu-pixel-coverage-mode");
      const webGpuPixelCoverageWeightFloor = Number(await viewport.getAttribute("data-webgpu-pixel-coverage-weight-floor") ?? "0");
      const webGpuPixelCoverageFootprintScale = Number(await viewport.getAttribute("data-webgpu-pixel-coverage-footprint-scale") ?? "0");
      const webGpuProjectionDepthMin = Number(await viewport.getAttribute("data-webgpu-projection-depth-min") ?? "0");
      const webGpuProjectionDepthMax = Number(await viewport.getAttribute("data-webgpu-projection-depth-max") ?? "0");
      const webGpuProjectionDepthSpan = Number(await viewport.getAttribute("data-webgpu-projection-depth-span") ?? "0");
      const webGpuColorFidelityMode = await viewport.getAttribute("data-webgpu-color-fidelity-mode");
      const webGpuColorSourceRgbGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-source-rgb-gaussians") ?? "0");
      const webGpuColorSourceShDcGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-source-sh-dc-gaussians") ?? "0");
      const webGpuColorSourceFallbackGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-source-fallback-gaussians") ?? "0");
      const webGpuColorSourceObjectGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-source-object-gaussians") ?? "0");
      const webGpuColorOpacityMean = Number(await viewport.getAttribute("data-webgpu-color-opacity-mean") ?? "0");
      const webGpuScreenCovarianceMode = await viewport.getAttribute("data-webgpu-screen-covariance-mode");
      const webGpuScreenCovarianceGaussians = numericValue(await viewport.getAttribute("data-webgpu-screen-covariance-gaussians") ?? "0");
      const webGpuScreenCovarianceFallbackGaussians = numericValue(await viewport.getAttribute("data-webgpu-screen-covariance-fallback-gaussians") ?? "0");
      const webGpuScreenCovarianceClampedGaussians = numericValue(await viewport.getAttribute("data-webgpu-screen-covariance-clamped-gaussians") ?? "0");
      const webGpuScreenCovarianceMaxAnisotropy = Number(await viewport.getAttribute("data-webgpu-screen-covariance-max-anisotropy") ?? "0");
      const webGpuScreenCovarianceSigmaMean = Number(await viewport.getAttribute("data-webgpu-screen-covariance-sigma-mean") ?? "0");
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
      if (editRendererId === "webgpu-tile") {
        if (webGpuDisplayWidth <= 0 || webGpuDisplayHeight <= 0) {
          throw new Error(
            `${asset.id} WebGPU display size was not measured: ${webGpuDisplayWidth}x${webGpuDisplayHeight}`,
          );
        }
        if (options.webGpuViewportSize) {
          if (
            webGpuViewportWidth !== options.webGpuViewportSize ||
            webGpuViewportHeight !== options.webGpuViewportSize ||
            webGpuViewportAspectMode !== "explicit-square"
          ) {
            throw new Error(
              `${asset.id} WebGPU viewport did not honor requested size: requested=${options.webGpuViewportSize} actual=${webGpuViewportWidth}x${webGpuViewportHeight} mode=${webGpuViewportAspectMode}`,
            );
          }
        } else if (
          options.webGpuProbe === WEBGPU_RUNTIME_PROBE_FULL &&
          webGpuPixelCount <
            DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE *
              DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE
        ) {
          throw new Error(
            `${asset.id} WebGPU full runtime pixel budget is below visual audit floor: ${webGpuViewportWidth}x${webGpuViewportHeight}:${webGpuPixelCount}`,
          );
        }
        if (
          !options.webGpuViewportSize &&
          options.webGpuProbe === WEBGPU_RUNTIME_PROBE_FULL &&
          webGpuViewportAspectMode !== "display-aspect-adaptive"
        ) {
          throw new Error(
            `${asset.id} WebGPU full runtime viewport did not use adaptive display aspect matching: ${webGpuViewportAspectMode}`,
          );
        }
        if (
          !options.webGpuViewportSize &&
          options.webGpuProbe === WEBGPU_RUNTIME_PROBE_FULL &&
          (!String(webGpuViewportQuality ?? "").startsWith("adaptive-") ||
            webGpuViewportPixelBudget <
              DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE *
                DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE)
        ) {
          throw new Error(
            `${asset.id} WebGPU full runtime did not expose an adaptive quality budget: quality=${webGpuViewportQuality} budget=${webGpuViewportPixelBudget}`,
          );
        }
        if (webGpuPixelCount !== webGpuViewportWidth * webGpuViewportHeight) {
          throw new Error(
            `${asset.id} WebGPU pixel count does not match viewport: ${webGpuPixelCount} vs ${webGpuViewportWidth}x${webGpuViewportHeight}`,
          );
        }
        if (
          webGpuBoundsFitMode !== "aspect-fit-padding" ||
          webGpuBoundsPaddingRatio < 0.079 ||
          Math.abs(webGpuBoundsWorldAspect - webGpuBoundsViewportAspect) > 0.02
        ) {
          throw new Error(
            `${asset.id} WebGPU bounds were not aspect-fit with padding: mode=${webGpuBoundsFitMode} padding=${webGpuBoundsPaddingRatio} worldAspect=${webGpuBoundsWorldAspect} viewportAspect=${webGpuBoundsViewportAspect}`,
          );
        }
        if (webGpuProjectionMode !== "edit-perspective-camera-v1" || Math.abs(webGpuProjectionCameraFov - 52) > 0.001) {
          throw new Error(
            `${asset.id} WebGPU projection did not use edit perspective camera: mode=${webGpuProjectionMode} fov=${webGpuProjectionCameraFov}`,
          );
        }
        if (
          webGpuDepthWeightMode !== "front-weighted-oit-v1" ||
          webGpuPixelDepthSortMode !== "depth-binned-alpha-composite-v1" ||
          webGpuPixelDepthGateStrength <= 1 ||
          webGpuPixelDepthGateFloor <= 0 ||
          webGpuPixelDepthGateFloor >= 1 ||
          webGpuPixelDepthBinCount !== 8 ||
          webGpuPixelCoverageMode !== "footprint-weight-floor-calibrated-v1" ||
          webGpuPixelCoverageWeightFloor < 0.003 ||
          webGpuPixelCoverageWeightFloor > 0.005 ||
          webGpuPixelCoverageFootprintScale < 2.1 ||
          webGpuPixelCoverageFootprintScale > 2.3 ||
          webGpuProjectionDepthSpan <= 0 ||
          webGpuProjectionDepthMax <= webGpuProjectionDepthMin
        ) {
          throw new Error(
            `${asset.id} WebGPU depth/coverage weighting did not expose a valid depth-binned alpha contract: mode=${webGpuDepthWeightMode} pixelSort=${webGpuPixelDepthSortMode} strength=${webGpuPixelDepthGateStrength} floor=${webGpuPixelDepthGateFloor} bins=${webGpuPixelDepthBinCount} coverage=${webGpuPixelCoverageMode}:${webGpuPixelCoverageWeightFloor}:${webGpuPixelCoverageFootprintScale} min=${webGpuProjectionDepthMin} max=${webGpuProjectionDepthMax} span=${webGpuProjectionDepthSpan}`,
          );
        }
        if (
          webGpuColorFidelityMode !== "source-color-fidelity-v1" ||
          webGpuColorSourceRgbGaussians +
            webGpuColorSourceShDcGaussians +
            webGpuColorSourceFallbackGaussians +
            webGpuColorSourceObjectGaussians !==
            packedGaussians ||
          webGpuColorOpacityMean <= 0 ||
          webGpuColorOpacityMean > 1
        ) {
          throw new Error(
            `${asset.id} WebGPU color fidelity contract is invalid: mode=${webGpuColorFidelityMode} rgb=${webGpuColorSourceRgbGaussians} shDc=${webGpuColorSourceShDcGaussians} fallback=${webGpuColorSourceFallbackGaussians} object=${webGpuColorSourceObjectGaussians} packed=${packedGaussians} opacityMean=${webGpuColorOpacityMean}`,
          );
        }
        if (
          webGpuColorSourceObjectGaussians > 0 &&
          (webGpuColorSourceObjectGaussians !== packedGaussians ||
            webGpuColorSourceRgbGaussians !== 0 ||
            webGpuColorSourceShDcGaussians !== 0 ||
            webGpuColorSourceFallbackGaussians !== 0)
        ) {
          throw new Error(
            `${asset.id} WebGPU object-color mode mixed source colors unexpectedly: rgb=${webGpuColorSourceRgbGaussians} shDc=${webGpuColorSourceShDcGaussians} fallback=${webGpuColorSourceFallbackGaussians} object=${webGpuColorSourceObjectGaussians} packed=${packedGaussians}`,
          );
        }
        if (
          webGpuScreenCovarianceMode !== "camera-jacobian-covariance-v1" ||
          webGpuScreenCovarianceGaussians + webGpuScreenCovarianceFallbackGaussians !== packedGaussians ||
          webGpuScreenCovarianceClampedGaussians < 0 ||
          webGpuScreenCovarianceMaxAnisotropy < 1 ||
          webGpuScreenCovarianceSigmaMean <= 0
        ) {
          throw new Error(
            `${asset.id} WebGPU screen covariance contract is invalid: mode=${webGpuScreenCovarianceMode} full=${webGpuScreenCovarianceGaussians} fallback=${webGpuScreenCovarianceFallbackGaussians} clamped=${webGpuScreenCovarianceClampedGaussians} maxAnisotropy=${webGpuScreenCovarianceMaxAnisotropy} packed=${packedGaussians} sigmaMean=${webGpuScreenCovarianceSigmaMean}`,
          );
        }
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
      const webGpuResolveFilter = await viewport.getAttribute("data-webgpu-resolve-filter");
      const webGpuAlphaPresentationMode = await viewport.getAttribute("data-webgpu-alpha-presentation-mode");
      const webGpuAlphaPresentationFloor = Number(await viewport.getAttribute("data-webgpu-alpha-presentation-floor") ?? "0");
      const webGpuRuntimeProbe = await viewport.getAttribute("data-webgpu-runtime-probe");
      const webGpuDeviceLostStatus = await viewport.getAttribute("data-webgpu-device-lost-status");
      const webGpuDeviceLostReason = await viewport.getAttribute("data-webgpu-device-lost-reason");
      const webGpuDeviceLostMessage = await viewport.getAttribute("data-webgpu-device-lost-message");
      const webGpuDeviceErrorStatus = await viewport.getAttribute("data-webgpu-device-error-status");
      const webGpuDeviceErrorType = await viewport.getAttribute("data-webgpu-device-error-type");
      const webGpuDeviceErrorMessage = await viewport.getAttribute("data-webgpu-device-error-message");
      const webGpuQueueStatus = await viewport.getAttribute("data-webgpu-queue-status");
      const webGpuQueueReason = await viewport.getAttribute("data-webgpu-queue-reason");
      const webGpuQueueMessage = await viewport.getAttribute("data-webgpu-queue-message");
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
        validateWebGpuRuntimeProbe({
          assetId: asset.id,
          expectedProbe: options.webGpuProbe,
          actualProbe: webGpuRuntimeProbe,
          webGpuAccumulationSource,
          webGpuAccumulationStatus,
          webGpuAccumulationReason,
          webGpuAccumulationWorkgroups,
          webGpuComputeSource,
          webGpuComputeStatus,
          webGpuComputeReason,
          webGpuComputeWorkgroups,
          webGpuPixelSource,
          webGpuPixelStatus,
          webGpuPixelReason,
          webGpuPixelWorkgroups,
          webGpuFirstFrameStatus,
          webGpuFirstFrameReason,
          webGpuFirstFramePixels,
          webGpuFirstFrameChecksum,
          webGpuResolveSource,
          webGpuResolveFilter,
          webGpuAlphaPresentationMode,
          webGpuAlphaPresentationFloor,
        });
        if (webGpuDeviceLostStatus === "lost" && !options.allowWebGpuDeviceLost) {
          throw new Error(
            `${asset.id} WebGPU device was lost after first-frame submission: probe=${webGpuRuntimeProbe} reason=${webGpuDeviceLostReason} message=${webGpuDeviceLostMessage} deviceError=${webGpuDeviceErrorStatus}:${webGpuDeviceErrorType}:${webGpuDeviceErrorMessage} queue=${webGpuQueueStatus}:${webGpuQueueReason}:${webGpuQueueMessage}`,
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
        if (options.webGpuProbe !== WEBGPU_RUNTIME_PROBE_FULL) {
          const screenshotPath = `/tmp/objgauss-audit-${asset.id}-${options.webGpuProbe}.png`;
          await page.screenshot({ path: screenshotPath, fullPage: false });
          results.push({
            assetId: asset.id,
            title,
            splatPixels,
            splatRendererId,
            ...visualResidualResultFields(sparkVisualStats, editOriginalVisualStats, visualResidual),
            editPixels: webGpuFirstFramePixels,
            editRenderer,
            editRendererId,
            webGpuFirstFrameStatus,
            webGpuFirstFramePixels,
            webGpuFirstFrameChecksum,
            webGpuResolveSource,
            webGpuResolveFilter,
            webGpuAlphaPresentationMode,
            webGpuAlphaPresentationFloor,
            webGpuRuntimeProbe,
            webGpuViewportWidth,
            webGpuViewportHeight,
            webGpuPixelCount,
            webGpuViewportAspectMode,
            webGpuViewportQuality,
            webGpuViewportPixelBudget,
            webGpuDisplayWidth,
            webGpuDisplayHeight,
            webGpuBoundsFitMode,
            webGpuBoundsPaddingRatio,
            webGpuBoundsViewportAspect,
            webGpuBoundsWorldAspect,
            webGpuProjectionMode,
            webGpuProjectionCameraFov,
            webGpuDepthWeightMode,
            webGpuPixelDepthSortMode,
            webGpuPixelDepthGateStrength,
            webGpuPixelDepthGateFloor,
            webGpuPixelDepthBinCount,
            webGpuPixelCoverageMode,
            webGpuPixelCoverageWeightFloor,
            webGpuPixelCoverageFootprintScale,
            webGpuProjectionDepthMin,
            webGpuProjectionDepthMax,
            webGpuProjectionDepthSpan,
            webGpuColorFidelityMode,
            webGpuColorSourceRgbGaussians,
            webGpuColorSourceShDcGaussians,
            webGpuColorSourceFallbackGaussians,
            webGpuColorSourceObjectGaussians,
            webGpuColorOpacityMean,
            webGpuColorSourceRgbGaussiansAfterDelete: "probe-skipped",
            webGpuColorSourceShDcGaussiansAfterDelete: "probe-skipped",
            webGpuColorSourceFallbackGaussiansAfterDelete: "probe-skipped",
            webGpuColorSourceObjectGaussiansAfterDelete: "probe-skipped",
            webGpuScreenCovarianceMode,
            webGpuScreenCovarianceGaussians,
            webGpuScreenCovarianceFallbackGaussians,
            webGpuScreenCovarianceClampedGaussians,
            webGpuScreenCovarianceMaxAnisotropy,
            webGpuScreenCovarianceSigmaMean,
            webGpuDeviceLostStatus,
            webGpuDeviceLostReason,
            webGpuDeviceLostMessage,
            webGpuDeviceErrorStatus,
            webGpuDeviceErrorType,
            webGpuDeviceErrorMessage,
            webGpuQueueStatus,
            webGpuQueueReason,
            webGpuQueueMessage,
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
            webGpuStorageChecksumAfterIsolate: webGpuStorageChecksum,
            webGpuStorageChecksumAfterDelete: webGpuStorageChecksum,
            storageLimitGate,
            storageLimitReason,
            storageLimitBlocker,
            storageLimitMaxBufferSize,
            storageLimitMaxBindingSize,
            storageLimitMaxStorageBuffersPerStage,
            storageLimitRequiredStorageBuffersPerStage,
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
            objectStateChecksumAfterIsolate: objectStateChecksum,
            objectStateChecksumAfterDelete: objectStateChecksum,
            canvasSelectedObject: "probe-skipped",
            visibleAfterIsolate: "probe-skipped",
            visibleAfterDelete: "probe-skipped",
            renderModeAfterDelete: "probe-skipped",
            deletedObjects: "probe-skipped",
            screenshotPath,
          });
          continue;
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
      const webGpuColorSourceRgbGaussiansAfterDelete = numericValue(await viewport.getAttribute("data-webgpu-color-source-rgb-gaussians") ?? "0");
      const webGpuColorSourceShDcGaussiansAfterDelete = numericValue(await viewport.getAttribute("data-webgpu-color-source-sh-dc-gaussians") ?? "0");
      const webGpuColorSourceFallbackGaussiansAfterDelete = numericValue(await viewport.getAttribute("data-webgpu-color-source-fallback-gaussians") ?? "0");
      const webGpuColorSourceObjectGaussiansAfterDelete = numericValue(await viewport.getAttribute("data-webgpu-color-source-object-gaussians") ?? "0");
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
        webGpuColorSourceRgbGaussiansAfterDelete +
          webGpuColorSourceShDcGaussiansAfterDelete <=
          0 ||
        webGpuColorSourceFallbackGaussiansAfterDelete !== 0 ||
        webGpuColorSourceObjectGaussiansAfterDelete !== 0
      ) {
        throw new Error(
          `${asset.id} delete preview did not return to source original colors: rgb=${webGpuColorSourceRgbGaussiansAfterDelete} shDc=${webGpuColorSourceShDcGaussiansAfterDelete} fallback=${webGpuColorSourceFallbackGaussiansAfterDelete} object=${webGpuColorSourceObjectGaussiansAfterDelete}`,
        );
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
        ...visualResidualResultFields(sparkVisualStats, editOriginalVisualStats, visualResidual),
        editPixels,
        editRenderer,
        editRendererId,
        webGpuFirstFrameStatus,
        webGpuFirstFramePixels,
        webGpuFirstFrameChecksum,
        webGpuResolveSource,
        webGpuResolveFilter,
        webGpuAlphaPresentationMode,
        webGpuAlphaPresentationFloor,
        webGpuRuntimeProbe,
        webGpuViewportWidth,
        webGpuViewportHeight,
        webGpuPixelCount,
        webGpuViewportAspectMode,
        webGpuViewportQuality,
        webGpuViewportPixelBudget,
        webGpuDisplayWidth,
        webGpuDisplayHeight,
        webGpuBoundsFitMode,
        webGpuBoundsPaddingRatio,
        webGpuBoundsViewportAspect,
        webGpuBoundsWorldAspect,
        webGpuProjectionMode,
        webGpuProjectionCameraFov,
        webGpuDepthWeightMode,
        webGpuPixelDepthSortMode,
        webGpuPixelDepthGateStrength,
        webGpuPixelDepthGateFloor,
        webGpuPixelDepthBinCount,
        webGpuPixelCoverageMode,
        webGpuPixelCoverageWeightFloor,
        webGpuPixelCoverageFootprintScale,
        webGpuProjectionDepthMin,
        webGpuProjectionDepthMax,
        webGpuProjectionDepthSpan,
        webGpuColorFidelityMode,
        webGpuColorSourceRgbGaussians,
        webGpuColorSourceShDcGaussians,
        webGpuColorSourceFallbackGaussians,
        webGpuColorSourceObjectGaussians,
        webGpuColorOpacityMean,
        webGpuColorSourceRgbGaussiansAfterDelete,
        webGpuColorSourceShDcGaussiansAfterDelete,
        webGpuColorSourceFallbackGaussiansAfterDelete,
        webGpuColorSourceObjectGaussiansAfterDelete,
        webGpuScreenCovarianceMode,
        webGpuScreenCovarianceGaussians,
        webGpuScreenCovarianceFallbackGaussians,
        webGpuScreenCovarianceClampedGaussians,
        webGpuScreenCovarianceMaxAnisotropy,
        webGpuScreenCovarianceSigmaMean,
        webGpuDeviceLostStatus,
        webGpuDeviceLostReason,
        webGpuDeviceLostMessage,
        webGpuDeviceErrorStatus,
        webGpuDeviceErrorType,
        webGpuDeviceErrorMessage,
        webGpuQueueStatus,
        webGpuQueueReason,
        webGpuQueueMessage,
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
        storageLimitMaxStorageBuffersPerStage,
        storageLimitRequiredStorageBuffersPerStage,
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
        !issue.includes("No available adapters.") &&
        !(options.allowWebGpuDeviceLost && issue.includes("CONTEXT_LOST_WEBGL")),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }
    return results;
  } finally {
    await browser.close();
  }
}

function validateWebGpuRuntimeProbe({
  assetId,
  expectedProbe,
  actualProbe,
  webGpuAccumulationSource,
  webGpuAccumulationStatus,
  webGpuAccumulationReason,
  webGpuAccumulationWorkgroups,
  webGpuComputeSource,
  webGpuComputeStatus,
  webGpuComputeReason,
  webGpuComputeWorkgroups,
  webGpuPixelSource,
  webGpuPixelStatus,
  webGpuPixelReason,
  webGpuPixelWorkgroups,
  webGpuFirstFrameStatus,
  webGpuFirstFrameReason,
  webGpuFirstFramePixels,
  webGpuFirstFrameChecksum,
  webGpuResolveSource,
  webGpuResolveFilter,
  webGpuAlphaPresentationMode,
  webGpuAlphaPresentationFloor,
}) {
  if (actualProbe !== expectedProbe) {
    throw new Error(`${assetId} WebGPU runtime probe mismatch: expected=${expectedProbe} actual=${actualProbe}`);
  }
  if (
    webGpuResolveSource === "webgpu-pixel-storage-resolve-v1" &&
    (
      webGpuAlphaPresentationMode !== WEBGPU_TILE_ALPHA_PRESENTATION_MODE ||
      Math.abs(webGpuAlphaPresentationFloor - WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR) > 0.000001
    )
  ) {
    throw new Error(
      `${assetId} WebGPU storage resolve did not expose alpha presentation gate: mode=${webGpuAlphaPresentationMode} floor=${webGpuAlphaPresentationFloor}`,
    );
  }

  const stageState = {
    accumulation: {
      status: webGpuAccumulationStatus,
      source: webGpuAccumulationSource,
      workgroups: webGpuAccumulationWorkgroups,
      expectedSource: "webgpu-compute-covariance-accumulation-v1",
      reason: webGpuAccumulationReason,
    },
    compute: {
      status: webGpuComputeStatus,
      source: webGpuComputeSource,
      workgroups: webGpuComputeWorkgroups,
      expectedSource: "webgpu-compute-resolve-v1",
      reason: webGpuComputeReason,
    },
    pixel: {
      status: webGpuPixelStatus,
      source: webGpuPixelSource,
      workgroups: webGpuPixelWorkgroups,
      expectedSource: WEBGPU_PIXEL_RESOLVE_SOURCE,
      reason: webGpuPixelReason,
    },
  };

  const expected = expectedProbeStages(expectedProbe);
  for (const [stage, state] of Object.entries(stageState)) {
    if (expected.dispatched.has(stage)) {
      if (
        state.status !== "dispatched" ||
        state.source !== state.expectedSource ||
        state.workgroups <= 0
      ) {
        throw new Error(
          `${assetId} WebGPU ${expectedProbe} did not dispatch ${stage}: ${state.status}:${state.source}:${state.workgroups}:${state.reason}`,
        );
      }
    } else if (state.status !== "skipped" || state.source || state.workgroups !== 0) {
      throw new Error(
        `${assetId} WebGPU ${expectedProbe} did not skip ${stage}: ${state.status}:${state.source}:${state.workgroups}:${state.reason}`,
      );
    }
  }

  if (
    expectedProbe === WEBGPU_RUNTIME_PROBE_FULL ||
    expectedProbe === WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY ||
    expectedProbe === WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY ||
    expectedProbe === WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT
  ) {
    validateRenderedProbeFrame({
      assetId,
      expectedProbe,
      webGpuFirstFrameStatus,
      webGpuFirstFrameReason,
      webGpuFirstFramePixels,
      webGpuFirstFrameChecksum,
      webGpuResolveSource,
      webGpuResolveFilter,
      expectedSource: "webgpu-pixel-storage-resolve-v1",
      expectedFilter: "bilinear-storage",
    });
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY) {
    validateRenderedProbeFrame({
      assetId,
      expectedProbe,
      webGpuFirstFrameStatus,
      webGpuFirstFrameReason,
      webGpuFirstFramePixels,
      webGpuFirstFrameChecksum,
      webGpuResolveSource,
      webGpuResolveFilter,
      expectedSource: "webgpu-sampled-texture-resolve-v1",
      expectedFilter: "nearest-sampled-texture",
    });
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY) {
    validateRenderedProbeFrame({
      assetId,
      expectedProbe,
      webGpuFirstFrameStatus,
      webGpuFirstFrameReason,
      webGpuFirstFramePixels,
      webGpuFirstFrameChecksum,
      webGpuResolveSource,
      webGpuResolveFilter,
      expectedSource: "webgpu-buffer-copy-texture-resolve-v1",
      expectedFilter: "nearest-texture-load",
    });
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_CLEAR_ONLY) {
    validateRenderedProbeFrame({
      assetId,
      expectedProbe,
      webGpuFirstFrameStatus,
      webGpuFirstFrameReason,
      webGpuFirstFramePixels,
      webGpuFirstFrameChecksum,
      webGpuResolveSource,
      webGpuResolveFilter,
      expectedSource: "webgpu-clear-pass-v1",
      expectedFilter: "clear-pass",
    });
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY) {
    if (
      webGpuFirstFrameStatus !== "probed" ||
      webGpuFirstFramePixels <= 0 ||
      !/^[0-9a-f]{8}$/.test(webGpuFirstFrameChecksum ?? "") ||
      webGpuResolveSource !== WEBGPU_PIXEL_RESOLVE_SOURCE
    ) {
      throw new Error(
        `${assetId} WebGPU pixel-compute-only probe did not submit pixel compute: frame=${webGpuFirstFrameStatus}:${webGpuFirstFrameReason} pixels=${webGpuFirstFramePixels} checksum=${webGpuFirstFrameChecksum} source=${webGpuResolveSource}`,
      );
    }
  } else if (
    webGpuFirstFrameStatus !== "probed" ||
    webGpuFirstFramePixels <= 0 ||
    !/^[0-9a-f]{8}$/.test(webGpuFirstFrameChecksum ?? "")
  ) {
    throw new Error(
      `${assetId} WebGPU ${expectedProbe} did not expose a submitted probe frame fact: frame=${webGpuFirstFrameStatus}:${webGpuFirstFrameReason} pixels=${webGpuFirstFramePixels} checksum=${webGpuFirstFrameChecksum} source=${webGpuResolveSource}`,
    );
  }
}

function validateRenderedProbeFrame({
  assetId,
  expectedProbe,
  webGpuFirstFrameStatus,
  webGpuFirstFrameReason,
  webGpuFirstFramePixels,
  webGpuFirstFrameChecksum,
  webGpuResolveSource,
  webGpuResolveFilter,
  expectedSource,
  expectedFilter,
}) {
  if (
    webGpuFirstFrameStatus !== "rendered" ||
    webGpuFirstFramePixels <= 0 ||
    !/^[0-9a-f]{8}$/.test(webGpuFirstFrameChecksum ?? "") ||
    webGpuResolveSource !== expectedSource
  ) {
    throw new Error(
      `${assetId} WebGPU ${expectedProbe} route did not render through ${expectedSource}: frame=${webGpuFirstFrameStatus}:${webGpuFirstFrameReason} pixels=${webGpuFirstFramePixels} checksum=${webGpuFirstFrameChecksum} source=${webGpuResolveSource}`,
    );
  }
  if (webGpuResolveFilter !== expectedFilter) {
    throw new Error(
      `${assetId} WebGPU ${expectedProbe} resolve filter mismatch: expected=${expectedFilter} actual=${webGpuResolveFilter}`,
    );
  }
}

function expectedProbeStages(probe) {
  if (probe === WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY) {
    return { dispatched: new Set(["accumulation"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY) {
    return { dispatched: new Set(["compute"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY) {
    return { dispatched: new Set() };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY) {
    return { dispatched: new Set() };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_CLEAR_ONLY) {
    return { dispatched: new Set() };
  }
  return { dispatched: new Set(["accumulation", "compute", "pixel"]) };
}

function launchOptions(options = {}) {
  const executablePath = options.executablePath ?? firstExisting([
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ]);
  const args = ["--no-sandbox", ...webGpuLaunchArgs(options.webGpuFlags)];
  const launch = {
    args,
    headless: !options.headed,
  };
  if (Number.isFinite(options.slowMo) && options.slowMo > 0) {
    launch.slowMo = options.slowMo;
  }
  if (options.browserChannel) {
    launch.channel = options.browserChannel;
    return launch;
  }
  if (executablePath) {
    launch.executablePath = executablePath;
  }
  return launch;
}

function urlWithWebGpuOptions(url, options) {
  if (options.webGpuProbe === WEBGPU_RUNTIME_PROBE_FULL && !options.webGpuViewportSize) {
    return url;
  }
  const parsed = new URL(url);
  if (options.webGpuProbe !== WEBGPU_RUNTIME_PROBE_FULL) {
    parsed.searchParams.set("webgpu-probe", options.webGpuProbe);
  }
  if (options.webGpuViewportSize) {
    parsed.searchParams.set("webgpu-viewport-size", String(options.webGpuViewportSize));
  }
  return parsed.toString();
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

function flagEnabled(value) {
  if (value === true) return true;
  if (value === false || value === undefined || value === null) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const text = String(value).trim();
  return text ? text : undefined;
}

function optionalPositiveInteger(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  return Math.round(parsed);
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

async function waitForWebGpuQueueTelemetry(page, timeoutMs = 8000) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    const queueStatus = viewport?.getAttribute("data-webgpu-queue-status");
    const deviceLostStatus = viewport?.getAttribute("data-webgpu-device-lost-status");
    return (
      queueStatus === "done" ||
      queueStatus === "failed" ||
      queueStatus === "unavailable" ||
      deviceLostStatus === "lost"
    );
  }, undefined, { timeout: timeoutMs }).catch(() => {});
}

async function waitForEditViewportReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    const renderer = viewport?.getAttribute("data-renderer");
    const webGpuStatus = viewport?.getAttribute("data-webgpu-status");
    return (
      (renderer === "gaussian-oit" || renderer === "webgpu-tile") &&
      webGpuStatus !== "pending"
    );
  }, undefined, { timeout: 15000 });
  const rendererId = await page.locator(".viewport").first().getAttribute("data-renderer");
  if (rendererId === "webgpu-tile") {
    await page.waitForFunction(() => {
      const viewport = document.querySelector(".viewport");
      return viewport?.getAttribute("data-webgpu-first-frame-status") !== "pending";
    }, undefined, { timeout: 15000 });
    await waitForWebGpuQueueTelemetry(page);
  }
  await waitForNonBackgroundPixels(page);
}

async function canvasVisualStats(page, selector) {
  const locator = page.locator(selector).first();
  await locator.waitFor({ timeout: 15000 });
  const buffer = await locator.screenshot({ animations: "disabled" });
  return visualStatsFromPng(buffer);
}

function visualStatsFromPng(buffer) {
  const image = decodePng(buffer);
  const totalPixels = image.width * image.height;
  let nonBackgroundPixels = 0;
  let lumaSum = 0;
  let chromaSum = 0;
  let checksum = 2166136261;

  for (let pixelIndex = 0; pixelIndex < totalPixels; pixelIndex += 1) {
    const pixel = pngPixelAt(image, pixelIndex);
    checksum = checksumByte(checksum, pixel.red);
    checksum = checksumByte(checksum, pixel.green);
    checksum = checksumByte(checksum, pixel.blue);
    checksum = checksumByte(checksum, pixel.alpha);
    if (
      pixel.alpha <= 0 ||
      Math.abs(pixel.red - 16) + Math.abs(pixel.green - 19) + Math.abs(pixel.blue - 22) <= 10
    ) {
      continue;
    }
    const luma = (0.2126 * pixel.red + 0.7152 * pixel.green + 0.0722 * pixel.blue) / 255;
    const chroma = (Math.max(pixel.red, pixel.green, pixel.blue) - Math.min(pixel.red, pixel.green, pixel.blue)) / 255;
    nonBackgroundPixels += 1;
    lumaSum += luma;
    chromaSum += chroma;
  }

  return {
    width: image.width,
    height: image.height,
    pixels: totalPixels,
    nonBackgroundPixels,
    coverage: roundMetric(totalPixels > 0 ? nonBackgroundPixels / totalPixels : 0),
    lumaMean: roundMetric(nonBackgroundPixels > 0 ? lumaSum / nonBackgroundPixels : 0),
    chromaMean: roundMetric(nonBackgroundPixels > 0 ? chromaSum / nonBackgroundPixels : 0),
    checksum: checksum.toString(16).padStart(8, "0"),
  };
}

function validateCanvasVisualStats(assetId, label, stats) {
  if (
    !stats ||
    stats.width <= 0 ||
    stats.height <= 0 ||
    stats.pixels <= 0 ||
    stats.nonBackgroundPixels <= 0 ||
    stats.coverage <= 0 ||
    !/^[0-9a-f]{8}$/.test(stats.checksum)
  ) {
    throw new Error(
      `${assetId} ${label} canvas visual stats are invalid: ${JSON.stringify(stats)}`,
    );
  }
}

function compareVisualStats(sparkStats, editStats) {
  return {
    coverageRatio: roundMetric(editStats.coverage / Math.max(sparkStats.coverage, 0.000001)),
    lumaDelta: roundMetric(Math.abs(editStats.lumaMean - sparkStats.lumaMean)),
    chromaDelta: roundMetric(Math.abs(editStats.chromaMean - sparkStats.chromaMean)),
  };
}

function validateVisualResidual(assetId, residual) {
  if (
    !Number.isFinite(residual.coverageRatio) ||
    residual.coverageRatio <= 0 ||
    !Number.isFinite(residual.lumaDelta) ||
    !Number.isFinite(residual.chromaDelta)
  ) {
    throw new Error(`${assetId} visual residual is invalid: ${JSON.stringify(residual)}`);
  }
}

function visualResidualResultFields(sparkStats, editStats, residual) {
  return {
    visualResidualMode: VISUAL_RESIDUAL_MODE,
    sparkVisualWidth: sparkStats.width,
    sparkVisualHeight: sparkStats.height,
    sparkVisualPixels: sparkStats.pixels,
    sparkVisualNonBackgroundPixels: sparkStats.nonBackgroundPixels,
    sparkVisualCoverage: sparkStats.coverage,
    sparkVisualLumaMean: sparkStats.lumaMean,
    sparkVisualChromaMean: sparkStats.chromaMean,
    sparkVisualChecksum: sparkStats.checksum,
    editOriginalVisualWidth: editStats.width,
    editOriginalVisualHeight: editStats.height,
    editOriginalVisualPixels: editStats.pixels,
    editOriginalVisualNonBackgroundPixels: editStats.nonBackgroundPixels,
    editOriginalVisualCoverage: editStats.coverage,
    editOriginalVisualLumaMean: editStats.lumaMean,
    editOriginalVisualChromaMean: editStats.chromaMean,
    editOriginalVisualChecksum: editStats.checksum,
    sparkEditCoverageRatio: residual.coverageRatio,
    sparkEditLumaDelta: residual.lumaDelta,
    sparkEditChromaDelta: residual.chromaDelta,
  };
}

function decodePng(buffer) {
  const source = Buffer.from(buffer);
  const signature = "89504e470d0a1a0a";
  if (source.subarray(0, 8).toString("hex") !== signature) {
    throw new Error("unsupported screenshot format: expected PNG signature");
  }

  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idatChunks = [];
  while (offset < source.length) {
    const length = source.readUInt32BE(offset);
    const type = source.toString("ascii", offset + 4, offset + 8);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (type === "IHDR") {
      width = source.readUInt32BE(dataStart);
      height = source.readUInt32BE(dataStart + 4);
      bitDepth = source[dataStart + 8];
      colorType = source[dataStart + 9];
    } else if (type === "IDAT") {
      idatChunks.push(source.subarray(dataStart, dataEnd));
    } else if (type === "IEND") {
      break;
    }
    offset = dataEnd + 4;
  }

  if (width <= 0 || height <= 0 || bitDepth !== 8) {
    throw new Error(`unsupported PNG dimensions or bit depth: ${width}x${height}:${bitDepth}`);
  }
  const bytesPerPixel = pngBytesPerPixel(colorType);
  const inflated = inflateSync(Buffer.concat(idatChunks));
  const stride = width * bytesPerPixel;
  const data = new Uint8Array(height * stride);
  let sourceOffset = 0;
  let targetOffset = 0;
  let previous = new Uint8Array(stride);
  for (let y = 0; y < height; y += 1) {
    const filter = inflated[sourceOffset];
    sourceOffset += 1;
    const row = inflated.subarray(sourceOffset, sourceOffset + stride);
    sourceOffset += stride;
    const output = data.subarray(targetOffset, targetOffset + stride);
    unfilterPngRow({ filter, row, output, previous, bytesPerPixel });
    previous = output;
    targetOffset += stride;
  }
  return { width, height, colorType, bytesPerPixel, data };
}

function pngBytesPerPixel(colorType) {
  if (colorType === 0) return 1;
  if (colorType === 2) return 3;
  if (colorType === 4) return 2;
  if (colorType === 6) return 4;
  throw new Error(`unsupported PNG color type: ${colorType}`);
}

function pngPixelAt(image, pixelIndex) {
  const offset = pixelIndex * image.bytesPerPixel;
  if (image.colorType === 0) {
    const value = image.data[offset];
    return { red: value, green: value, blue: value, alpha: 255 };
  }
  if (image.colorType === 2) {
    return {
      red: image.data[offset],
      green: image.data[offset + 1],
      blue: image.data[offset + 2],
      alpha: 255,
    };
  }
  if (image.colorType === 4) {
    const value = image.data[offset];
    return { red: value, green: value, blue: value, alpha: image.data[offset + 1] };
  }
  return {
    red: image.data[offset],
    green: image.data[offset + 1],
    blue: image.data[offset + 2],
    alpha: image.data[offset + 3],
  };
}

function unfilterPngRow({ filter, row, output, previous, bytesPerPixel }) {
  for (let index = 0; index < row.length; index += 1) {
    const left = index >= bytesPerPixel ? output[index - bytesPerPixel] : 0;
    const up = previous[index] ?? 0;
    const upLeft = index >= bytesPerPixel ? previous[index - bytesPerPixel] : 0;
    let predictor = 0;
    if (filter === 1) predictor = left;
    else if (filter === 2) predictor = up;
    else if (filter === 3) predictor = Math.floor((left + up) / 2);
    else if (filter === 4) predictor = paethPredictor(left, up, upLeft);
    else if (filter !== 0) throw new Error(`unsupported PNG filter: ${filter}`);
    output[index] = (row[index] + predictor) & 0xff;
  }
}

function paethPredictor(left, up, upLeft) {
  const estimate = left + up - upLeft;
  const leftDistance = Math.abs(estimate - left);
  const upDistance = Math.abs(estimate - up);
  const upLeftDistance = Math.abs(estimate - upLeft);
  if (leftDistance <= upDistance && leftDistance <= upLeftDistance) return left;
  if (upDistance <= upLeftDistance) return up;
  return upLeft;
}

function checksumByte(checksum, value) {
  const next = checksum ^ (value & 0xff);
  return Math.imul(next, 16777619) >>> 0;
}

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) return 0;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
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
