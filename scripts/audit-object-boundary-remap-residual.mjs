import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

import {
  canvasVisualStats,
  compareVisualStats,
  validateCanvasVisualStats,
} from "./lib/visual-stats.mjs";

const MODE = "object-boundary-remap-browser-residual-v1";
const DEFAULT_ASSETS =
  "nerf-lego-alpha-closure-local,plush-semantic-closure-local,polyhaven-chair-commercial-demo-local";
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-object-boundary-remap-residual";
const DEFAULT_PORT = 5395;
const SPARK_OBJECT_FILTER = "spark-object-opacity-mask";
const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const previewDir = String(args.previewDir ?? args["preview-dir"] ?? path.join(outputDir, "preview"));
const assetIds = String(args.assets ?? DEFAULT_ASSETS);
const skipBuild = flagEnabled(args.skipBuild ?? args["skip-build"]);
const noServer = flagEnabled(args.noServer ?? args["no-server"]) || Boolean(args.url);
const headed = flagEnabled(args.headed ?? args.headful);
const includeVisualStats = !flagEnabled(args.skipVisualStats ?? args["skip-visual-stats"]);
const minSceneCount = positiveInteger(args.minSceneCount ?? args["min-scene-count"] ?? 2);
const targetCount = positiveInteger(args.targetCount ?? args["target-count"] ?? 1);
const maxRemapSamples = positiveIntegerOrNull(args.maxRemapSamples ?? args["max-remap-samples"]);
const maxAfterCoverageDelta = nonNegativeNumber(
  args.maxAfterCoverageDelta ?? args["max-after-coverage-delta"] ?? 0.08,
);
const maxAfterLumaDelta = nonNegativeNumber(
  args.maxAfterLumaDelta ?? args["max-after-luma-delta"] ?? 0.08,
);
const maxAfterChromaDelta = nonNegativeNumber(
  args.maxAfterChromaDelta ?? args["max-after-chroma-delta"] ?? 0.08,
);

const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  url: baseUrl,
  outputDir,
  previewDir,
  assets: assetIds.split(",").map((entry) => entry.trim()).filter(Boolean),
  includeVisualStats,
  thresholds: {
    minSceneCount,
    targetCount,
    maxAfterCoverageDelta,
    maxAfterLumaDelta,
    maxAfterChromaDelta,
  },
  preview: null,
  skipped: [],
  targetPlans: [],
  rows: [],
  comparisons: [],
  recommendations: [],
  promotion: null,
  commands: [],
  failures: [],
  passed: false,
};

mkdirSync(outputDir, { recursive: true });
mkdirSync(previewDir, { recursive: true });

let server = null;
try {
  summary.preview = await generatePreviewPlys();
  const previewResults = (summary.preview.results ?? []).filter(
    (result) => result.outputPly && existsSync(result.outputPly),
  );
  summary.skipped = summary.preview.skipped ?? [];
  if (previewResults.length === 0) {
    throw new Error("no remap preview PLY was generated");
  }
  if (previewResults.length < minSceneCount) {
    throw new Error(`not enough remap preview scenes: ${previewResults.length}/${minSceneCount}`);
  }
  const targetPlans = buildTargetPlans(previewResults, targetCount);
  summary.targetPlans = targetPlans.map((plan) => ({
    assetId: plan.preview.assetId,
    targetObjectIds: plan.targetObjectIds,
  }));
  const targetCaseCount = targetPlans.reduce(
    (total, plan) => total + plan.targetObjectIds.length,
    0,
  );
  if (targetCaseCount <= 0) {
    throw new Error("no remap target cases were selected");
  }

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

  const result = await runBrowserResidualGate({
    baseUrl: withPackedObjectSource(baseUrl),
    outputDir,
    targetPlans,
    headed,
    includeVisualStats,
  });
  summary.rows = result.rows;
  summary.comparisons = result.comparisons;
  summary.recommendations = buildRecommendations(summary.comparisons);
  summary.promotion = buildPromotionSummary({
    comparisons: summary.comparisons,
    recommendations: summary.recommendations,
    sceneCount: previewResults.length,
    targetCaseCount,
  });
  summary.passed =
    summary.failures.length === 0 &&
    summary.rows.length === targetCaseCount * 2 &&
    summary.rows.every((row) => row.passed) &&
    summary.comparisons.every((comparison) => comparison.passed);
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

async function generatePreviewPlys() {
  const command = {
    label: "Generate remap preview PLY",
    command: [
      "node",
      "scripts/export-object-boundary-remap-preview.mjs",
      "--assets",
      assetIds,
      "--output-dir",
      previewDir,
    ],
  };
  if (maxRemapSamples !== null) {
    command.command.push("--max-remap-samples", String(maxRemapSamples));
  }
  await runCommand(command);
  const summaryPath = path.join(previewDir, "summary.json");
  if (!existsSync(summaryPath)) {
    throw new Error(`remap preview summary is missing: ${summaryPath}`);
  }
  return JSON.parse(readFileSync(summaryPath, "utf-8"));
}

async function runBrowserResidualGate({ baseUrl, outputDir, targetPlans, headed, includeVisualStats }) {
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
    for (const plan of targetPlans) {
      const { preview, targetObjectIds } = plan;
      for (const targetObjectId of targetObjectIds) {
        console.log(
          `object_boundary_remap_residual_asset_start asset=${JSON.stringify(preview.assetId)} target=${JSON.stringify(targetObjectId)}`,
        );
        rows.push(
          await auditPlyVariant({
            page,
            url: baseUrl,
            outputDir,
            preview,
            variant: "original",
            plyPath: preview.sourcePly,
            targetObjectId,
            includeVisualStats,
          }),
        );
        rows.push(
          await auditPlyVariant({
            page,
            url: baseUrl,
            outputDir,
            preview,
            variant: "remap-preview",
            plyPath: preview.outputPly,
            targetObjectId,
            includeVisualStats,
          }),
        );
      }
    }

    const relevantIssues = consoleIssues.filter(
      (issue) =>
        !issue.includes("THREE.WebGLRenderer") &&
        !issue.includes("GPU stall due to ReadPixels") &&
        !issue.includes("No available adapters.") &&
        !issue.includes("Worker terminate") &&
        !issue.includes("Missing rot_0 property") &&
        !issue.includes("Worker error: Missing rot_0 property") &&
        !issue.includes("Missing f_dc_0 property") &&
        !issue.includes("Worker error: Missing f_dc_0 property"),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }
    return {
      rows,
      comparisons: buildComparisons(rows),
    };
  } finally {
    await closeBrowserWithTimeout(browser);
  }
}

async function auditPlyVariant({
  page,
  url,
  outputDir,
  preview,
  variant,
  plyPath,
  targetObjectId,
  includeVisualStats,
}) {
  if (!existsSync(plyPath)) {
    throw new Error(`${preview.assetId}/${variant} missing PLY: ${plyPath}`);
  }
  await uploadPly(page, url, plyPath);
  await enterEditMode(page);
  await waitForSparkEditReady(page);
  await page.waitForTimeout(750);
  const beforeStats = includeVisualStats
    ? await readCanvasStats(page, `${preview.assetId}/${variant}`, "before delete", ".viewport canvas")
    : null;
  await selectObjectAndDelete(page, targetObjectId);
  await waitForSparkDeleteReady(page);
  await page.waitForTimeout(750);
  const afterStats = includeVisualStats
    ? await readCanvasStats(page, `${preview.assetId}/${variant}`, "after delete", ".viewport canvas")
    : null;
  const stats = await readRouteStats(page);
  const failures = validateRouteStats({ assetId: preview.assetId, variant, stats, targetObjectId });
  const screenshotPath = path.join(
    outputDir,
    `${safeFileName(preview.assetId)}-target-${safeFileName(targetObjectId)}-${safeFileName(variant)}.png`,
  );
  await page.screenshot({ path: screenshotPath, fullPage: false });
  const deleteDelta =
    beforeStats && afterStats
      ? compareVisualStats(beforeStats, afterStats)
      : { coverageRatio: 0, lumaDelta: 0, chromaDelta: 0 };
  return {
    assetId: preview.assetId,
    assetName: preview.name,
    variant,
    plyPath,
    targetObjectId,
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
    visual: {
      before: beforeStats,
      after: afterStats,
      deleteDelta,
    },
    screenshotPath,
  };
}

async function uploadPly(page, url, plyPath) {
  await page.goto(url, { waitUntil: "networkidle" });
  const title = await page.title();
  if (title !== "ObjGauss 查看器") {
    throw new Error(`unexpected page title: ${title}`);
  }
  await expectNoFrameworkOverlay(page);
  await page.getByText("素材库").first().waitFor({ timeout: 15000 });
  await page.locator('input[type="file"]').setInputFiles(plyPath);
  await page.waitForFunction(
    (fileName) => document.body.innerText.includes(fileName),
    path.basename(plyPath),
    { timeout: 30000 },
  );
  await page.waitForFunction(() => {
    const app = document.querySelector(".appShell");
    return app?.getAttribute("data-renderer-route-kind") === "commercial";
  }, undefined, { timeout: 30000 });
  await page.waitForTimeout(750);
}

async function enterEditMode(page) {
  await page.locator(".modeTabs").getByRole("button", { name: "对象编辑" }).click();
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    return Boolean(viewport?.getAttribute("data-renderer"));
  }, undefined, { timeout: 30000 });
}

async function waitForSparkEditReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    return (
      viewport?.getAttribute("data-renderer") === "spark-splat" &&
      viewport?.getAttribute("data-spark-filter-status") === "ready" &&
      viewport?.getAttribute("data-spark-mask-source") === "ply-packed"
    );
  }, undefined, { timeout: 120000 });
}

async function selectObjectAndDelete(page, targetObjectId) {
  const selected = await page.evaluate((target) => {
    const rows = [...document.querySelectorAll(".objectRow")];
    const row = rows.find(
      (candidate) => candidate.querySelector(".idCell")?.textContent?.trim() === String(target),
    );
    const button = row?.querySelector(".objectSelectButton");
    button?.click();
    return row?.querySelector(".idCell")?.textContent?.trim() ?? "";
  }, targetObjectId);
  if (!selected) {
    throw new Error(`target object ${targetObjectId} is missing from the object list`);
  }
  await page.waitForFunction(
    (expected) => {
      const status = document.querySelector(".statusBar")?.textContent ?? "";
      return status.includes(`所选：${expected}`);
    },
    selected,
    { timeout: 15000 },
  );
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
      viewport?.getAttribute("data-spark-mask-source") === "ply-packed" &&
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
    };
  });
}

function validateRouteStats({ assetId, variant, stats, targetObjectId }) {
  const failures = [];
  if (stats.renderer !== "spark-splat") failures.push(`renderer=${stats.renderer}`);
  if (stats.objectFilter !== SPARK_OBJECT_FILTER) failures.push(`objectFilter=${stats.objectFilter}`);
  if (stats.filterStatus !== "ready") failures.push(`status=${stats.filterStatus}`);
  if (stats.maskSource !== "ply-packed") failures.push(`maskSource=${stats.maskSource}`);
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
  if (failures.length > 0) {
    summary.failures.push(
      `${assetId}/${variant}/object=${targetObjectId}: ${failures.join(" ")}`,
    );
  }
  return failures;
}

function buildComparisons(rows) {
  const byCase = new Map();
  for (const row of rows) {
    const key = `${row.assetId}::${row.targetObjectId}`;
    const caseRows = byCase.get(key) ?? [];
    caseRows.push(row);
    byCase.set(key, caseRows);
  }
  const comparisons = [];
  for (const caseRows of byCase.values()) {
    const original = caseRows.find((row) => row.variant === "original");
    const remap = caseRows.find((row) => row.variant === "remap-preview");
    if (!original || !remap) continue;
    const beforeDelta =
      original.visual.before && remap.visual.before
        ? compareVisualStats(original.visual.before, remap.visual.before)
        : { coverageRatio: 1, lumaDelta: 0, chromaDelta: 0 };
    const afterDelta =
      original.visual.after && remap.visual.after
        ? compareVisualStats(original.visual.after, remap.visual.after)
        : { coverageRatio: 1, lumaDelta: 0, chromaDelta: 0 };
    const hiddenGaussianDelta = remap.objectMask.hiddenGaussians - original.objectMask.hiddenGaussians;
    const deletedCoverageDelta =
      remap.visual.deleteDelta.coverageRatio - original.visual.deleteDelta.coverageRatio;
    const passed =
      original.passed &&
      remap.passed &&
      Math.abs(afterDelta.coverageRatio - 1) <= maxAfterCoverageDelta &&
      afterDelta.lumaDelta <= maxAfterLumaDelta &&
      afterDelta.chromaDelta <= maxAfterChromaDelta;
    comparisons.push({
      assetId: original.assetId,
      targetObjectId: original.targetObjectId,
      passed,
      confidence: includeVisualStats ? "browser-visual-stats" : "telemetry-only",
      hiddenGaussianDelta,
      hiddenGaussianDeltaShare: roundMetric(
        hiddenGaussianDelta / Math.max(original.spark.baseGaussians, 1),
      ),
      beforeDelta,
      afterDelta,
      deleteDeltaChange: {
        coverageRatio: roundMetric(deletedCoverageDelta),
        lumaDelta: roundMetric(remap.visual.deleteDelta.lumaDelta - original.visual.deleteDelta.lumaDelta),
        chromaDelta: roundMetric(
          remap.visual.deleteDelta.chromaDelta - original.visual.deleteDelta.chromaDelta,
        ),
      },
      recommendation: "review-only",
    });
  }
  return comparisons;
}

function buildRecommendations(comparisons) {
  return comparisons.map((comparison) => {
    const promotionCandidate =
      includeVisualStats &&
      comparison.passed &&
      comparison.hiddenGaussianDelta < 0 &&
      comparison.afterDelta.coverageRatio >= 1 &&
      comparison.afterDelta.lumaDelta <= maxAfterLumaDelta * 0.5 &&
      comparison.afterDelta.chromaDelta <= maxAfterChromaDelta * 0.5;
    return {
      assetId: comparison.assetId,
      targetObjectId: comparison.targetObjectId,
      confidence: comparison.confidence,
      promotionCandidate,
      recommendation: promotionCandidate
        ? "candidate-for-manual-visual-review"
        : "browser-evidence-only",
      reason: promotionCandidate
        ? "remap hides fewer Gaussians and stays within strict residual thresholds"
        : "route is validated, but residual evidence is not strong enough for default promotion",
    };
  });
}

function buildPromotionSummary({ comparisons, recommendations, sceneCount, targetCaseCount }) {
  const passedTargets = comparisons.filter((comparison) => comparison.passed).length;
  const promotableTargets = recommendations.filter(
    (recommendation) => recommendation.promotionCandidate,
  ).length;
  const maxAfterCoverageDistance = maxOrZero(
    comparisons.map((comparison) => Math.abs(comparison.afterDelta.coverageRatio - 1)),
  );
  const maxAfterLumaDelta = maxOrZero(
    comparisons.map((comparison) => comparison.afterDelta.lumaDelta),
  );
  const maxAfterChromaDelta = maxOrZero(
    comparisons.map((comparison) => comparison.afterDelta.chromaDelta),
  );
  const meanHiddenDeltaShare = meanOrZero(
    comparisons.map((comparison) => comparison.hiddenGaussianDeltaShare),
  );
  const promotionCandidate =
    includeVisualStats &&
    sceneCount >= minSceneCount &&
    passedTargets === targetCaseCount &&
    promotableTargets === targetCaseCount;
  return {
    sceneCount,
    minSceneCount,
    targetCaseCount,
    targetCount,
    passedTargets,
    promotableTargets,
    confidence: includeVisualStats ? "browser-visual-stats" : "telemetry-only",
    maxAfterCoverageDistance,
    maxAfterLumaDelta,
    maxAfterChromaDelta,
    meanHiddenDeltaShare,
    promotionCandidate,
    recommendation: promotionCandidate
      ? "candidate-for-cleaned-ply-review"
      : "do-not-promote-default-hard-mask",
    reason: promotionCandidate
      ? "all target cases preserve browser residual and hide fewer target Gaussians"
      : "top-N target evidence is insufficient for replacing public samples",
  };
}

function buildTargetPlans(previewResults, requestedTargetCount) {
  return previewResults.map((preview) => ({
    preview,
    targetObjectIds: chooseTargetObjectIds(preview, requestedTargetCount),
  }));
}

function chooseTargetObjectIds(preview, requestedTargetCount) {
  const ids = [];
  for (const pair of preview.remapPairs ?? []) {
    if (!Number.isFinite(pair?.fromObject)) continue;
    if (!ids.includes(pair.fromObject)) ids.push(pair.fromObject);
    if (ids.length >= requestedTargetCount) return ids;
  }
  for (const objectRow of preview.byObject ?? []) {
    if (!Number.isFinite(objectRow?.objectId)) continue;
    if (!ids.includes(objectRow.objectId)) ids.push(objectRow.objectId);
    if (ids.length >= requestedTargetCount) return ids;
  }
  return ids.length > 0 ? ids : [0];
}

function withPackedObjectSource(baseUrlValue) {
  const parsed = new URL(baseUrlValue);
  parsed.searchParams.set("spark-object-source", "packed");
  parsed.searchParams.set("spark-ply-source", "off");
  parsed.searchParams.set("spark-reconstruct-probe", "1");
  return parsed.toString();
}

function writeReport(outputDirPath, payload) {
  writeFileSync(path.join(outputDirPath, "summary.json"), `${JSON.stringify(payload, null, 2)}\n`);
  writeFileSync(path.join(outputDirPath, "summary.md"), renderMarkdown(payload));
}

function renderMarkdown(payload) {
  const lines = [
    "# Object Boundary Remap Browser Residual Gate",
    "",
    `- Status: ${payload.passed ? "passed" : "failed"}`,
    `- Mode: ${payload.mode}`,
    `- Generated: ${payload.generatedAt}`,
    `- Assets: ${payload.assets.join(", ")}`,
    `- Skipped assets: ${payload.skipped.map((asset) => asset.assetId).join(", ") || "none"}`,
    `- Target count: ${payload.thresholds.targetCount}`,
    `- URL: ${payload.url}`,
    `- Preview dir: ${payload.previewDir}`,
    `- Visual stats: ${payload.includeVisualStats ? "enabled" : "disabled"}`,
    "",
    "This gate compares the original object-aware PLY against a sampled `object_id` remap preview PLY in the browser. It forces the same PLY-packed Spark object-mask route for both files and deletes the selected top-N remap-candidate target objects. Passing means the browser route is valid and the remap preview stays within residual thresholds; it does not automatically promote the remap into public samples.",
    "",
    "## Rows",
    "",
    "| Asset | Variant | Target object | Route | Mask source | Hidden | Visible | Delete coverage | Delete luma | Delete chroma | Screenshot |",
    "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
  ];
  for (const row of payload.rows) {
    lines.push(
      `| ${escapeMarkdown(row.assetId)} | ${escapeMarkdown(row.variant)} | ${row.targetObjectId} | ${escapeMarkdown(row.route.id)} | ${escapeMarkdown(row.spark.maskSource)} | ${row.objectMask.hiddenGaussians} | ${row.objectMask.visibleGaussians} | ${formatNumber(row.visual.deleteDelta.coverageRatio)} | ${formatNumber(row.visual.deleteDelta.lumaDelta)} | ${formatNumber(row.visual.deleteDelta.chromaDelta)} | ${escapeMarkdown(row.screenshotPath)} |`,
    );
  }

  lines.push("", "## Comparisons", "");
  if (payload.comparisons.length === 0) {
    lines.push("No comparisons available.");
  } else {
    lines.push(
      "| Asset | Target object | Pass | Hidden delta | Hidden share | After coverage ratio | After luma | After chroma | Delete coverage change | Delete luma change | Delete chroma change |",
      "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    );
    for (const comparison of payload.comparisons) {
      lines.push(
        `| ${escapeMarkdown(comparison.assetId)} | ${comparison.targetObjectId} | ${comparison.passed ? "yes" : "no"} | ${comparison.hiddenGaussianDelta} | ${formatNumber(comparison.hiddenGaussianDeltaShare)} | ${formatNumber(comparison.afterDelta.coverageRatio)} | ${formatNumber(comparison.afterDelta.lumaDelta)} | ${formatNumber(comparison.afterDelta.chromaDelta)} | ${formatNumber(comparison.deleteDeltaChange.coverageRatio)} | ${formatNumber(comparison.deleteDeltaChange.lumaDelta)} | ${formatNumber(comparison.deleteDeltaChange.chromaDelta)} |`,
      );
    }
  }

  lines.push("", "## Promotion Summary", "");
  if (!payload.promotion) {
    lines.push("No promotion summary available.");
  } else {
    lines.push(
      "| Scenes | Target cases | Passed targets | Promotable targets | Recommendation | Promotion candidate | Max coverage distance | Max luma | Max chroma | Mean hidden share | Reason |",
      "| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
      `| ${payload.promotion.sceneCount}/${payload.promotion.minSceneCount} | ${payload.promotion.targetCaseCount} | ${payload.promotion.passedTargets} | ${payload.promotion.promotableTargets} | ${escapeMarkdown(payload.promotion.recommendation)} | ${payload.promotion.promotionCandidate ? "yes" : "no"} | ${formatNumber(payload.promotion.maxAfterCoverageDistance)} | ${formatNumber(payload.promotion.maxAfterLumaDelta)} | ${formatNumber(payload.promotion.maxAfterChromaDelta)} | ${formatNumber(payload.promotion.meanHiddenDeltaShare)} | ${escapeMarkdown(payload.promotion.reason)} |`,
    );
  }

  lines.push("", "## Recommendations", "");
  if (payload.recommendations.length === 0) {
    lines.push("No recommendations available.");
  } else {
    lines.push(
      "| Asset | Target object | Recommendation | Promotion candidate | Reason |",
      "| --- | ---: | --- | --- | --- |",
    );
    for (const recommendation of payload.recommendations) {
      lines.push(
        `| ${escapeMarkdown(recommendation.assetId)} | ${recommendation.targetObjectId} | ${escapeMarkdown(recommendation.recommendation)} | ${recommendation.promotionCandidate ? "yes" : "no"} | ${escapeMarkdown(recommendation.reason)} |`,
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
      `object_boundary_remap_residual_row asset=${JSON.stringify(row.assetId)} variant=${JSON.stringify(row.variant)} ` +
        `target=${JSON.stringify(row.targetObjectId)} route=${JSON.stringify(row.route.id)}:${JSON.stringify(row.route.boundary)}:${JSON.stringify(row.route.result)} ` +
        `spark=${JSON.stringify(row.spark.maskSource)}:${JSON.stringify(row.spark.reconstructSource)}:${row.spark.visibleGaussians}/${row.spark.baseGaussians} ` +
        `objectMask=${JSON.stringify(row.objectMask.mode)}:${JSON.stringify(row.objectMask.size)}:${row.objectMask.visibleGaussians}/${row.objectMask.hiddenGaussians}:${row.objectMask.updates} ` +
        `deleteVisual=${row.visual.deleteDelta.coverageRatio}/${row.visual.deleteDelta.lumaDelta}/${row.visual.deleteDelta.chromaDelta} ` +
        `screenshot=${JSON.stringify(row.screenshotPath)}`,
    );
  }
  for (const comparison of payload.comparisons) {
    console.log(
      `object_boundary_remap_residual_comparison asset=${JSON.stringify(comparison.assetId)} target=${JSON.stringify(comparison.targetObjectId)} ` +
        `passed=${comparison.passed} hiddenDelta=${comparison.hiddenGaussianDelta} ` +
        `after=${comparison.afterDelta.coverageRatio}/${comparison.afterDelta.lumaDelta}/${comparison.afterDelta.chromaDelta} ` +
        `deleteChange=${comparison.deleteDeltaChange.coverageRatio}/${comparison.deleteDeltaChange.lumaDelta}/${comparison.deleteDeltaChange.chromaDelta}`,
    );
  }
  for (const recommendation of payload.recommendations) {
    console.log(
      `object_boundary_remap_residual_recommendation asset=${JSON.stringify(recommendation.assetId)} target=${JSON.stringify(recommendation.targetObjectId)} ` +
        `recommendation=${JSON.stringify(recommendation.recommendation)} promotion=${recommendation.promotionCandidate}`,
    );
  }
  if (payload.promotion) {
    console.log(
      `object_boundary_remap_residual_promotion recommendation=${JSON.stringify(payload.promotion.recommendation)} ` +
        `promotion=${payload.promotion.promotionCandidate} scenes=${payload.promotion.sceneCount}/${payload.promotion.minSceneCount} ` +
        `targets=${payload.promotion.targetCaseCount} passed=${payload.promotion.passedTargets} promotable=${payload.promotion.promotableTargets} ` +
        `max=${payload.promotion.maxAfterCoverageDistance}/${payload.promotion.maxAfterLumaDelta}/${payload.promotion.maxAfterChromaDelta}`,
    );
  }
  for (const failure of payload.failures) {
    console.error(`object_boundary_remap_residual_failure=${JSON.stringify(failure)}`);
  }
  console.log(
    `object_boundary_remap_residual=${payload.passed ? "passed" : "failed"} mode=${JSON.stringify(payload.mode)} rows=${payload.rows.length} comparisons=${payload.comparisons.length} report=${JSON.stringify(path.join(payload.outputDir, "summary.md"))}`,
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

function startPreviewServer(portValue) {
  const child = spawn(
    "npm",
    ["run", "preview", "--", "--host", "127.0.0.1", "--port", String(portValue), "--strictPort"],
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

function flagEnabled(value) {
  if (value === true) return true;
  if (value === false || value === undefined || value === null) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function nonNegativeNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) {
    throw new Error(`expected non-negative number, received ${JSON.stringify(value)}`);
  }
  return numeric;
}

function positiveInteger(value) {
  const numeric = Number(value);
  if (!Number.isInteger(numeric) || numeric <= 0) {
    throw new Error(`expected positive integer, received ${JSON.stringify(value)}`);
  }
  return numeric;
}

function positiveIntegerOrNull(value) {
  if (value === undefined || value === null || value === false || value === "") return null;
  return positiveInteger(value);
}

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) return 0;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function meanOrZero(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return 0;
  return roundMetric(finite.reduce((sum, value) => sum + value, 0) / finite.length);
}

function maxOrZero(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return 0;
  return roundMetric(Math.max(...finite));
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
