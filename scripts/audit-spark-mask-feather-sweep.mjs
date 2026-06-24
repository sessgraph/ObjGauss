import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

import { ASSET_LIBRARY } from "../src/assetLibrary.js";
import {
  canvasVisualStats,
  compareVisualStats,
  validateCanvasVisualStats,
} from "./lib/visual-stats.mjs";

const MODE = "spark-mask-feather-sweep-v1";
const DEFAULT_ASSETS = "nerf-lego-alpha-closure-local,plush-semantic-closure-local";
const DEFAULT_VARIANTS = "hard:off,feather55:0.55";
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-spark-mask-feather-sweep";
const DEFAULT_PORT = 5380;
const SPARK_OBJECT_FILTER = "spark-object-opacity-mask";
const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";
const SPARK_FEATHER_MODE = "spatial-neighbor-feather-v1";

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const assets = selectAssets(args.assets ?? DEFAULT_ASSETS);
const variants = parseVariants(String(args.variants ?? DEFAULT_VARIANTS));
const skipBuild = flagEnabled(args.skipBuild ?? args["skip-build"]);
const noServer = flagEnabled(args.noServer ?? args["no-server"]) || Boolean(args.url);
const headed = flagEnabled(args.headed ?? args.headful);
const includeVisualStats = !flagEnabled(args.skipVisualStats ?? args["skip-visual-stats"]);
const control = normalizeControl(args.control ?? "url");

const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  url: baseUrl,
  outputDir,
  control,
  includeVisualStats,
  assets: assets.map((asset) => asset.id),
  variants,
  rows: [],
  comparisons: [],
  commands: [],
  failures: [],
  passed: false,
};

mkdirSync(outputDir, { recursive: true });

let server = null;
try {
  if (!skipBuild) {
    await runCommand({
      label: "Build viewer",
      command: ["npm", "run", "build"],
    });
  }
  if (!noServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before preview audit");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  summary.rows = await runSweep({
    baseUrl,
    outputDir,
    control,
    assets,
    variants,
    headed,
    includeVisualStats,
  });
  summary.comparisons = buildComparisons(summary.rows, variants);
  summary.passed =
    summary.failures.length === 0 &&
    summary.rows.length === assets.length * variants.length &&
    summary.rows.every((row) => row.passed);
} catch (error) {
  summary.failures.push(error?.message ?? String(error));
  summary.passed = false;
} finally {
  if (server) stopPreviewServer(server);
}

writeReport(outputDir, summary);
printSummary(summary);

if (!summary.passed) {
  process.exit(1);
}

async function runSweep({ baseUrl, outputDir, control, assets, variants, headed, includeVisualStats }) {
  const browser = await chromium.launch(launchOptions({ headed }));
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  try {
    const rows = [];
    for (const variant of variants) {
      const variantUrl = urlForVariant(baseUrl, variant, control);
      for (const asset of assets) {
        console.log(
          `spark_mask_feather_sweep_asset_start asset=${JSON.stringify(asset.id)} variant=${JSON.stringify(variant.id)}`,
        );
        const row = await auditAssetVariant({
          page,
          url: variantUrl,
          outputDir,
          control,
          asset,
          variant,
          includeVisualStats,
        });
        rows.push(row);
      }
    }

    const relevantIssues = consoleIssues.filter(
      (issue) =>
        !issue.includes("THREE.WebGLRenderer") &&
        !issue.includes("GPU stall due to ReadPixels") &&
        !issue.includes("No available adapters.") &&
        !issue.includes("Worker terminate"),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }
    return rows;
  } finally {
    await closeBrowserWithTimeout(browser);
  }
}

async function auditAssetVariant({ page, url, outputDir, control, asset, variant, includeVisualStats }) {
  await loadAsset(page, url, asset);
  await setFeatherControl(page, { control, variant });
  await page.waitForTimeout(750);
  const beforeStats = includeVisualStats
    ? await readCanvasStats(page, asset.id, "before delete", ".splatViewport canvas")
    : null;

  await enterEditModeAndDeleteObject(page);
  await waitForSparkDeleteReady(page);
  await page.waitForTimeout(750);
  const afterStats = includeVisualStats
    ? await readCanvasStats(page, asset.id, "after delete", ".viewport canvas")
    : null;

  const stats = await readRouteStats(page);
  const failures = validateStats({ assetId: asset.id, variant, stats });
  const screenshotPath = path.join(outputDir, `${safeFileName(asset.id)}-${safeFileName(variant.id)}.png`);
  await page.screenshot({ path: screenshotPath, fullPage: false });
  const visualDelta =
    beforeStats && afterStats
      ? compareVisualStats(beforeStats, afterStats)
      : { coverageRatio: 0, lumaDelta: 0, chromaDelta: 0 };

  return {
    assetId: asset.id,
    assetName: asset.name,
    variantId: variant.id,
    control,
    featherEnabled: variant.enabled,
    requestedOpacity: variant.opacity,
    requestedRadius: variant.radius,
    passed: failures.length === 0,
    failures,
    route: {
      id: stats.appRoute,
      kind: stats.appKind,
      colorRole: stats.appColorRole,
      boundary: stats.appBoundary,
      result: stats.appResult,
      quality: stats.appQuality,
      qualityInterpretation: stats.appHardMaskQuality,
    },
    spark: {
      renderer: stats.renderer,
      objectFilter: stats.objectFilter,
      filterStatus: stats.filterStatus,
      filterMode: stats.filterMode,
      maskSource: stats.maskSource,
      reconstructSource: stats.reconstructSource,
      visibleGaussians: stats.visibleGaussians,
      baseGaussians: stats.baseGaussians,
      removedObjects: stats.removedObjects,
      colorMode: stats.colorMode,
    },
    objectMask: {
      mode: stats.objectMaskMode,
      size: stats.objectMaskSize,
      visibleGaussians: stats.objectMaskVisible,
      hiddenGaussians: stats.objectMaskHidden,
      updates: stats.objectMaskUpdates,
    },
    feather: {
      mode: stats.featherMode,
      featheredGaussians: stats.featheredGaussians,
      radius: stats.featherRadius,
      opacity: stats.featherOpacity,
      opacityMean: stats.opacityMean,
      minOpacity: stats.minOpacity,
    },
    visual: {
      before: beforeStats,
      after: afterStats,
      delta: visualDelta,
    },
    screenshotPath,
  };
}

async function setFeatherControl(page, { control, variant }) {
  if (control === "url") {
    await waitForAppFeatherState(page, variant.enabled);
    return;
  }
  if (
    variant.enabled &&
    (Math.abs(variant.opacity - 0.55) > 0.00001 || variant.radius !== null)
  ) {
    throw new Error(
      `ui control supports the built-in feather55 candidate only; received ${variant.id}:${variant.opacity}:${variant.radius}`,
    );
  }
  const checkbox = page.getByRole("checkbox", { name: "柔化删除边界" });
  await checkbox.waitFor({ timeout: 15000 });
  const checked = await checkbox.isChecked();
  if (checked !== variant.enabled) {
    await checkbox.click();
  }
  await waitForAppFeatherState(page, variant.enabled);
}

async function waitForAppFeatherState(page, enabled) {
  await page.waitForFunction(
    (expected) => {
      const app = document.querySelector(".appShell");
      return app?.getAttribute("data-spark-object-mask-feather-enabled") === String(expected);
    },
    enabled,
    { timeout: 15000 },
  );
}

async function loadAsset(page, url, asset) {
  await page.goto(url, { waitUntil: "networkidle" });
  const title = await page.title();
  if (title !== "ObjGauss 查看器") {
    throw new Error(`unexpected page title: ${title}`);
  }
  await expectNoFrameworkOverlay(page);
  await page.getByText("素材库").first().waitFor({ timeout: 15000 });

  const card = page.locator("article.assetCard").filter({ hasText: asset.name }).first();
  await card.getByRole("button", { name: "加载" }).click();
  await page.waitForFunction(
    (fileName) => document.body.innerText.includes(fileName),
    asset.fileName,
    { timeout: 30000 },
  );
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".splatViewport");
    return (
      viewport?.getAttribute("data-renderer") === "spark-splat" &&
      viewport?.getAttribute("data-spark-filter-status") === "ready"
    );
  }, undefined, { timeout: 120000 });
}

async function enterEditModeAndDeleteObject(page) {
  await page.locator(".modeTabs").getByRole("button", { name: "对象编辑" }).click();
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    return Boolean(viewport?.getAttribute("data-renderer"));
  }, undefined, { timeout: 30000 });

  const selectButton = page.locator(".objectSelectButton").first();
  await selectButton.waitFor({ timeout: 15000 });
  await selectButton.click();
  await page.waitForFunction(() => {
    const status = document.querySelector(".statusBar")?.textContent ?? "";
    return /所选：\S+/.test(status) && !status.includes("所选：无");
  }, undefined, { timeout: 15000 });

  await page.getByRole("button", { name: "预览删除" }).click();
}

async function waitForSparkDeleteReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    const app = document.querySelector(".appShell");
    return (
      viewport?.getAttribute("data-renderer") === "spark-splat" &&
      viewport?.getAttribute("data-object-filter") === "spark-object-opacity-mask" &&
      viewport?.getAttribute("data-spark-filter-status") === "ready" &&
      app?.getAttribute("data-source-preview-boundary") === "hard-object-mask-no-reoptimize"
    );
  }, undefined, { timeout: 120000 });
}

async function readCanvasStats(page, assetId, label, selector) {
  const stats = await canvasVisualStats(page, selector, { timeoutMs: 60000, usePageClip: true });
  validateCanvasVisualStats(assetId, label, stats);
  return stats;
}

async function readRouteStats(page) {
  return page.locator(".viewport").first().evaluate((viewport) => {
    const app = document.querySelector(".appShell");
    const numberAttr = (name) => {
      const parsed = Number(viewport.getAttribute(name) ?? "0");
      return Number.isFinite(parsed) ? parsed : 0;
    };
    const finiteAttr = (name) => {
      const parsed = Number(viewport.getAttribute(name) ?? "0");
      return Number.isFinite(parsed) ? parsed : 0;
    };
    const appAttr = (name) => app?.getAttribute(name) ?? "";
    return {
      appRoute: appAttr("data-renderer-route"),
      appKind: appAttr("data-renderer-route-kind"),
      appColorRole: appAttr("data-color-mode-role"),
      appBoundary: appAttr("data-source-preview-boundary"),
      appResult: appAttr("data-source-preview-result"),
      appQuality: appAttr("data-preview-quality"),
      appHardMaskQuality: appAttr("data-hard-mask-quality-interpretation"),
      renderer: viewport.getAttribute("data-renderer"),
      objectFilter: viewport.getAttribute("data-object-filter"),
      filterStatus: viewport.getAttribute("data-spark-filter-status"),
      filterMode: viewport.getAttribute("data-spark-filter-mode"),
      maskSource: viewport.getAttribute("data-spark-mask-source"),
      reconstructSource: viewport.getAttribute("data-spark-reconstruct-source"),
      visibleGaussians: numberAttr("data-spark-visible-gaussians"),
      removedObjects: numberAttr("data-spark-removed-objects"),
      colorMode: viewport.getAttribute("data-spark-color-mode"),
      baseGaussians: numberAttr("data-spark-packed-base-gaussians"),
      objectMaskMode: viewport.getAttribute("data-spark-object-mask-mode"),
      objectMaskSize: viewport.getAttribute("data-spark-object-mask-size"),
      objectMaskVisible: numberAttr("data-spark-object-mask-visible-gaussians"),
      objectMaskHidden: numberAttr("data-spark-object-mask-hidden-gaussians"),
      objectMaskUpdates: numberAttr("data-spark-object-mask-updates"),
      featherMode: viewport.getAttribute("data-spark-object-mask-feather-mode"),
      featherRadius: finiteAttr("data-spark-object-mask-feather-radius"),
      featherOpacity: finiteAttr("data-spark-object-mask-feather-opacity"),
      featheredGaussians: numberAttr("data-spark-object-mask-feathered-gaussians"),
      opacityMean: finiteAttr("data-spark-object-mask-opacity-mean"),
      minOpacity: finiteAttr("data-spark-object-mask-min-opacity"),
    };
  });
}

function validateStats({ assetId, variant, stats }) {
  const failures = [];
  if (stats.renderer !== "spark-splat") failures.push(`renderer=${stats.renderer}`);
  if (stats.objectFilter !== SPARK_OBJECT_FILTER) failures.push(`objectFilter=${stats.objectFilter}`);
  if (stats.filterStatus !== "ready") failures.push(`status=${stats.filterStatus}`);
  if (stats.appKind !== "commercial") failures.push(`kind=${stats.appKind}`);
  if (stats.appColorRole !== "source-color") failures.push(`colorRole=${stats.appColorRole}`);
  if (stats.appBoundary !== "hard-object-mask-no-reoptimize") failures.push(`boundary=${stats.appBoundary}`);
  if (stats.appResult !== "hard-mask-no-inpaint") failures.push(`result=${stats.appResult}`);
  if (stats.visibleGaussians <= 0) failures.push(`visible=${stats.visibleGaussians}`);
  if (stats.baseGaussians <= stats.visibleGaussians) {
    failures.push(`base/visible=${stats.baseGaussians}/${stats.visibleGaussians}`);
  }
  if (stats.removedObjects !== 1) failures.push(`removed=${stats.removedObjects}`);
  if (stats.colorMode !== "original") failures.push(`colorMode=${stats.colorMode}`);
  if (
    stats.objectMaskMode !== SPARK_OBJECT_MASK_MODE ||
    !stats.objectMaskSize ||
    stats.objectMaskVisible !== stats.visibleGaussians ||
    stats.objectMaskHidden !== stats.baseGaussians - stats.visibleGaussians ||
    stats.objectMaskUpdates <= 0
  ) {
    failures.push(
      `objectMask=${stats.objectMaskMode}:${stats.objectMaskSize}:${stats.objectMaskVisible}/${stats.objectMaskHidden}:${stats.objectMaskUpdates}`,
    );
  }
  if (!variant.enabled) {
    if (stats.featherMode !== "off") failures.push(`featherMode=${stats.featherMode}`);
    if (stats.featheredGaussians !== 0) failures.push(`feathered=${stats.featheredGaussians}`);
    if (Math.abs(stats.opacityMean - 1) > 0.00001 || Math.abs(stats.minOpacity - 1) > 0.00001) {
      failures.push(`opacity=${stats.opacityMean}/${stats.minOpacity}`);
    }
  } else {
    if (stats.featherMode !== SPARK_FEATHER_MODE) failures.push(`featherMode=${stats.featherMode}`);
    if (stats.featheredGaussians <= 0) failures.push(`feathered=${stats.featheredGaussians}`);
    if (Math.abs(stats.featherOpacity - variant.opacity) > 0.00001) {
      failures.push(`featherOpacity=${stats.featherOpacity}/${variant.opacity}`);
    }
    if (variant.radius !== null && Math.abs(stats.featherRadius - variant.radius) > 0.00001) {
      failures.push(`featherRadius=${stats.featherRadius}/${variant.radius}`);
    }
    if (stats.opacityMean >= 1 || stats.minOpacity >= 1) {
      failures.push(`opacity=${stats.opacityMean}/${stats.minOpacity}`);
    }
  }
  if (failures.length > 0) {
    summary.failures.push(`${assetId}/${variant.id}: ${failures.join(" ")}`);
  }
  return failures;
}

function buildComparisons(rows, variants) {
  const baselineId = variants.find((variant) => !variant.enabled)?.id ?? variants[0]?.id;
  const byAsset = new Map();
  for (const row of rows) {
    const list = byAsset.get(row.assetId) ?? [];
    list.push(row);
    byAsset.set(row.assetId, list);
  }
  const comparisons = [];
  for (const [assetId, assetRows] of byAsset) {
    const baseline = assetRows.find((row) => row.variantId === baselineId);
    for (const row of assetRows) {
      if (!baseline || row.variantId === baseline.variantId) continue;
      comparisons.push({
        assetId,
        baselineVariant: baseline.variantId,
        variantId: row.variantId,
        passed: row.passed && baseline.passed,
        featheredGaussians: row.feather.featheredGaussians,
        opacityMeanDelta: roundMetric(row.feather.opacityMean - baseline.feather.opacityMean),
        minOpacityDelta: roundMetric(row.feather.minOpacity - baseline.feather.minOpacity),
        coverageRatioDelta: roundMetric(row.visual.delta.coverageRatio - baseline.visual.delta.coverageRatio),
        lumaDeltaChange: roundMetric(row.visual.delta.lumaDelta - baseline.visual.delta.lumaDelta),
        chromaDeltaChange: roundMetric(row.visual.delta.chromaDelta - baseline.visual.delta.chromaDelta),
      });
    }
  }
  return comparisons;
}

function parseVariants(value) {
  const parsed = value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [id, opacityOrMode, radius] = entry.split(":");
      if (!id || !opacityOrMode) {
        throw new Error(`invalid variant ${JSON.stringify(entry)}; expected id:off or id:opacity[:radius]`);
      }
      if (opacityOrMode === "off" || opacityOrMode === "hard") {
        return { id, enabled: false, opacity: null, radius: null };
      }
      const opacity = Number(opacityOrMode);
      if (!Number.isFinite(opacity) || opacity <= 0 || opacity >= 1) {
        throw new Error(`invalid feather opacity for variant ${id}: ${opacityOrMode}`);
      }
      const radiusValue = radius === undefined || radius === "" ? null : Number(radius);
      if (radiusValue !== null && (!Number.isFinite(radiusValue) || radiusValue <= 0)) {
        throw new Error(`invalid feather radius for variant ${id}: ${radius}`);
      }
      return { id, enabled: true, opacity, radius: radiusValue };
    });
  if (parsed.length === 0) throw new Error("at least one variant is required");
  return parsed;
}

function selectAssets(value) {
  if (!value || value === true) throw new Error("--assets requires a comma-separated asset id list");
  const requested = String(value)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  const byId = new Map(ASSET_LIBRARY.map((asset) => [asset.id, asset]));
  return requested.map((id) => {
    const asset = byId.get(id);
    if (!asset) throw new Error(`unknown asset id: ${id}`);
    if (!asset.localPath || !asset.splatPath || !asset.fileName) {
      throw new Error(`asset is not a local renderable Gaussian sample: ${id}`);
    }
    return asset;
  });
}

function urlForVariant(baseUrl, variant, control) {
  const parsed = new URL(baseUrl);
  parsed.searchParams.delete("spark-object-mask-feather");
  parsed.searchParams.delete("spark-object-mask-feather-opacity");
  parsed.searchParams.delete("spark-object-mask-feather-radius");
  if (variant.enabled && control !== "ui") {
    parsed.searchParams.set("spark-object-mask-feather", "on");
    parsed.searchParams.set("spark-object-mask-feather-opacity", String(variant.opacity));
    if (variant.radius !== null) {
      parsed.searchParams.set("spark-object-mask-feather-radius", String(variant.radius));
    }
  }
  return parsed.toString();
}

function writeReport(outputDirPath, payload) {
  writeFileSync(path.join(outputDirPath, "summary.json"), `${JSON.stringify(payload, null, 2)}\n`);
  writeFileSync(path.join(outputDirPath, "summary.md"), renderMarkdown(payload));
}

function renderMarkdown(payload) {
  const lines = [
    "# Spark Mask Feather Sweep",
    "",
    `- Status: ${payload.passed ? "passed" : "failed"}`,
    `- Mode: ${payload.mode}`,
    `- Generated: ${payload.generatedAt}`,
    `- Assets: ${payload.assets.join(", ")}`,
    `- Variants: ${payload.variants.map((variant) => variant.id).join(", ")}`,
    `- Control: ${payload.control}`,
    `- Visual stats: ${payload.includeVisualStats ? "enabled" : "disabled"}`,
    "",
    "## Rows",
    "",
    "| Asset | Variant | Route | Feather mode | Soft Gaussians | Radius | Opacity | Mean opacity | Min opacity | Coverage ratio | Luma delta | Chroma delta | Screenshot |",
    "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ];
  for (const row of payload.rows) {
    lines.push(
      `| ${escapeMarkdown(row.assetId)} | ${escapeMarkdown(row.variantId)} | ${escapeMarkdown(row.route.id)} | ${escapeMarkdown(row.feather.mode)} | ${row.feather.featheredGaussians} | ${formatNumber(row.feather.radius)} | ${formatNumber(row.feather.opacity)} | ${formatNumber(row.feather.opacityMean)} | ${formatNumber(row.feather.minOpacity)} | ${formatNumber(row.visual.delta.coverageRatio)} | ${formatNumber(row.visual.delta.lumaDelta)} | ${formatNumber(row.visual.delta.chromaDelta)} | ${escapeMarkdown(row.screenshotPath)} |`,
    );
  }

  lines.push("", "## Comparisons", "");
  if (payload.comparisons.length === 0) {
    lines.push("No comparisons available.");
  } else {
    lines.push(
      "| Asset | Baseline | Variant | Soft Gaussians | Mean opacity delta | Min opacity delta | Coverage ratio delta | Luma delta change | Chroma delta change |",
      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    );
    for (const comparison of payload.comparisons) {
      lines.push(
        `| ${escapeMarkdown(comparison.assetId)} | ${escapeMarkdown(comparison.baselineVariant)} | ${escapeMarkdown(comparison.variantId)} | ${comparison.featheredGaussians} | ${formatNumber(comparison.opacityMeanDelta)} | ${formatNumber(comparison.minOpacityDelta)} | ${formatNumber(comparison.coverageRatioDelta)} | ${formatNumber(comparison.lumaDeltaChange)} | ${formatNumber(comparison.chromaDeltaChange)} |`,
      );
    }
  }
  if (payload.failures.length > 0) {
    lines.push("", "## Failures", "", ...payload.failures.map((failure) => `- ${escapeMarkdown(failure)}`));
  }
  return `${lines.join("\n")}\n`;
}

function printSummary(payload) {
  for (const row of payload.rows) {
    console.log(
      `spark_mask_feather_row asset=${JSON.stringify(row.assetId)} variant=${JSON.stringify(row.variantId)} ` +
        `control=${JSON.stringify(row.control)} ` +
        `route=${JSON.stringify(row.route.id)}:${JSON.stringify(row.route.boundary)}:${JSON.stringify(row.route.result)} ` +
        `objectMask=${JSON.stringify(row.objectMask.mode)}:${JSON.stringify(row.objectMask.size)}:${row.objectMask.visibleGaussians}/${row.objectMask.hiddenGaussians}:${row.objectMask.updates} ` +
        `feather=${JSON.stringify(row.feather.mode)}:${row.feather.featheredGaussians}:${row.feather.radius}:${row.feather.opacity}:${row.feather.opacityMean}/${row.feather.minOpacity} ` +
        `visual=${row.visual.delta.coverageRatio}/${row.visual.delta.lumaDelta}/${row.visual.delta.chromaDelta} ` +
        `screenshot=${JSON.stringify(row.screenshotPath)}`,
    );
  }
  for (const failure of payload.failures) {
    console.error(`spark_mask_feather_sweep_failure=${JSON.stringify(failure)}`);
  }
  console.log(
    `spark_mask_feather_sweep=${payload.passed ? "passed" : "failed"} mode=${JSON.stringify(payload.mode)} rows=${payload.rows.length} comparisons=${payload.comparisons.length} report=${JSON.stringify(path.join(payload.outputDir, "summary.md"))}`,
  );
}

async function runCommand({ label, command }) {
  const startedAt = performance.now();
  const result = await new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), {
      cwd: process.cwd(),
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      process.stdout.write(text);
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      process.stderr.write(text);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ code, stdout, stderr });
      } else {
        reject(new Error(`${label} failed with exit code ${code}`));
      }
    });
  });
  summary.commands.push({
    label,
    command,
    durationMs: Math.round(performance.now() - startedAt),
  });
  return result;
}

function startPreviewServer(port) {
  const child = spawn(
    "npm",
    ["run", "preview", "--", "--host", "127.0.0.1", "--port", String(port), "--strictPort"],
    { detached: true, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

function stopPreviewServer(child) {
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

async function expectNoFrameworkOverlay(page) {
  const bodyText = await page.locator("body").innerText();
  const overlaySignals = ["Vite Error", "Internal server error", "Failed to resolve import"];
  for (const signal of overlaySignals) {
    if (bodyText.includes(signal)) {
      throw new Error(`framework overlay detected: ${signal}`);
    }
  }
}

function launchOptions(options = {}) {
  const executablePath = firstExisting([
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ]);
  const launch = {
    args: ["--no-sandbox"],
    headless: !options.headed,
  };
  if (executablePath) launch.executablePath = executablePath;
  return launch;
}

function firstExisting(paths) {
  return paths.find((candidate) => existsSync(candidate));
}

async function closeBrowserWithTimeout(browser, timeoutMs = 5000) {
  const closePromise = browser.close().then(
    () => true,
    () => true,
  );
  const closed = await Promise.race([
    closePromise,
    sleep(timeoutMs).then(() => false),
  ]);
  if (closed) return;

  const child = typeof browser.process === "function" ? browser.process() : null;
  child?.kill?.("SIGTERM");
  await Promise.race([
    closePromise,
    sleep(1000).then(() => false),
  ]);
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

function normalizeControl(value) {
  const controlValue = String(value ?? "url").toLowerCase();
  if (controlValue === "url" || controlValue === "ui") return controlValue;
  throw new Error(`unsupported control ${JSON.stringify(value)}; expected url or ui`);
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === false || value === undefined || value === null) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) return 0;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function safeFileName(value) {
  return String(value).replace(/[^a-zA-Z0-9._-]+/g, "-");
}

function formatNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(6) : "0.000000";
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
