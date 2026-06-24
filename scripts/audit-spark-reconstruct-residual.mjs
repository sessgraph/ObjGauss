import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";
import { ASSET_LIBRARY } from "../src/assetLibrary.js";
import {
  canvasVisualStats,
  compareVisualStats,
  validateCanvasVisualStats,
} from "./lib/visual-stats.mjs";

const DEFAULT_PORT = 5296;
const DEFAULT_ASSETS = ["nerf-lego-alpha-closure-local"];
const RESIDUAL_MODE = "spark-reconstruct-residual-v1";
const SPARK_RECONSTRUCT_PROBE_PARAM = "spark-reconstruct-probe";
const DEFAULT_THRESHOLDS = {
  minCoverageRatio: 0.5,
  maxCoverageRatio: 2,
  maxLumaDelta: 0.08,
  maxChromaDelta: 0.08,
};
const SPARK_RECONSTRUCT_SOURCE = "packed-extract-v1";
const SPARK_RECONSTRUCT_SH_SOURCE = "packed-sh-extract-v1";
const SPARK_FULL_SOURCE_FILTERS = ["none", "spark-ply-source", "spark-ply-sh-source"];

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const assetIds = parseAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const outputDir = optionalString(args.outputDir ?? args["output-dir"]);
const shouldStartServer = !(args.url || args.noServer || args["no-server"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const headed = flagEnabled(args.headed ?? args.headful);
const thresholds = {
  minCoverageRatio:
    optionalFiniteNumber(args.minCoverageRatio ?? args["min-coverage-ratio"]) ??
    DEFAULT_THRESHOLDS.minCoverageRatio,
  maxCoverageRatio:
    optionalFiniteNumber(args.maxCoverageRatio ?? args["max-coverage-ratio"]) ??
    DEFAULT_THRESHOLDS.maxCoverageRatio,
  maxLumaDelta:
    optionalFiniteNumber(args.maxLumaDelta ?? args["max-luma-delta"]) ??
    DEFAULT_THRESHOLDS.maxLumaDelta,
  maxChromaDelta:
    optionalFiniteNumber(args.maxChromaDelta ?? args["max-chroma-delta"]) ??
    DEFAULT_THRESHOLDS.maxChromaDelta,
};

let server = null;

try {
  if (shouldStartServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before Spark reconstruct residual audit");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  const results = [];
  for (const assetId of assetIds) {
    const asset = localGaussianAsset(assetId);
    const result = await auditAsset({ asset, baseUrl, headed, thresholds });
    results.push(result);
    for (const check of result.gateChecks) {
      console.log(
        `spark_reconstruct_residual_gate_check=${check.passed ? "passed" : "failed"} ` +
          `asset=${JSON.stringify(asset.id)} metric=${JSON.stringify(check.metric)} ` +
          `actual=${check.actual} expected=${check.operator}${check.expected}`,
      );
    }
    console.log(
      `spark_reconstruct_residual_asset=${result.passed ? "passed" : "failed"} ` +
        `asset=${JSON.stringify(asset.id)} mode=${JSON.stringify(RESIDUAL_MODE)} ` +
        `full=${result.full.coverage}/${result.full.lumaMean}/${result.full.chromaMean} ` +
        `reconstruct=${result.reconstruct.coverage}/${result.reconstruct.lumaMean}/${result.reconstruct.chromaMean} ` +
        `coverageRatio=${result.residual.coverageRatio} ` +
        `lumaDelta=${result.residual.lumaDelta} ` +
        `chromaDelta=${result.residual.chromaDelta} ` +
        `fullSource=${JSON.stringify(result.fullObjectFilter)}:${JSON.stringify(result.fullReconstructSource)}:${result.fullShRestSourceGaussians}:${result.fullShRestPreservedGaussians}:${result.fullShRestPreserved}:${result.fullShRestCoefficientCount}:${result.fullShDegree} ` +
        `objectFilter=${JSON.stringify(result.objectFilter)} ` +
        `reconstructSource=${JSON.stringify(result.reconstructSource)} ` +
        `visibleGaussians=${result.visibleGaussians} ` +
        `filteredGaussians=${result.filteredGaussians} ` +
        `colorSourceGaussians=${result.colorSourceGaussians} ` +
        `packed=${result.packedBaseGaussians}/${result.packedVisibleIndices}:${result.packedBaseBuildMs}/${result.packedExtractMs} ` +
        `shRest=${result.shRestSourceGaussians}:${result.shRestPreservedGaussians}:${result.shRestPreserved}:${result.shRestCoefficientCount}:${result.shDegree} ` +
        `screenshot=${JSON.stringify(result.screenshotPath)}`,
    );
  }

  const summary = {
    mode: RESIDUAL_MODE,
    url: baseUrl,
    assets: assetIds,
    thresholds,
    passed: results.every((result) => result.passed),
    results,
  };
  if (outputDir) {
    writeReportFiles(outputDir, summary);
    console.log(
      `spark_reconstruct_residual_report=written outputDir=${JSON.stringify(outputDir)} ` +
        `summaryJson=${JSON.stringify(`${outputDir}/summary.json`)} ` +
        `summaryMd=${JSON.stringify(`${outputDir}/summary.md`)}`,
    );
  }
  console.log(
    `spark_reconstruct_residual=${summary.passed ? "passed" : "failed"} ` +
      `assets=${JSON.stringify(assetIds)} thresholds=${JSON.stringify(thresholds)} url=${baseUrl}`,
  );
  if (!summary.passed && !allowFailures) {
    process.exitCode = 1;
  }
} finally {
  if (server) stopPreviewServer(server);
}

async function auditAsset({ asset, baseUrl, headed, thresholds: gateThresholds }) {
  const browser = await chromium.launch(launchOptions({ headed }));
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const auditUrl = urlWithSparkReconstructProbe(baseUrl);
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  try {
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
    await waitForSparkViewport(page, SPARK_FULL_SOURCE_FILTERS, 60000);
    await page.waitForTimeout(1000);
    const screenshotOptions = { timeoutMs: 60000, usePageClip: true };
    const fullStats = await canvasVisualStats(page, ".splatViewport canvas", screenshotOptions);
    validateCanvasVisualStats(asset.id, "full Spark", fullStats);
    const fullViewport = page.locator(".viewport").first();
    const fullObjectFilter = await fullViewport.getAttribute("data-object-filter");
    const fullReconstructSource = await fullViewport.getAttribute("data-spark-reconstruct-source");
    const fullVisibleGaussians = numericValue(
      await fullViewport.getAttribute("data-spark-visible-gaussians") ?? "0",
    );
    const fullPackedBaseGaussians = numericValue(
      await fullViewport.getAttribute("data-spark-packed-base-gaussians") ?? "0",
    );
    const fullPackedVisibleIndices = numericValue(
      await fullViewport.getAttribute("data-spark-packed-visible-indices") ?? "0",
    );
    const fullShRestSourceGaussians = numericValue(
      await fullViewport.getAttribute("data-spark-sh-rest-source-gaussians") ?? "0",
    );
    const fullShRestPreservedGaussians = numericValue(
      await fullViewport.getAttribute("data-spark-sh-rest-preserved-gaussians") ?? "0",
    );
    const fullShRestPreserved = await fullViewport.getAttribute("data-spark-sh-rest-preserved");
    const fullShRestCoefficientCount = numericValue(
      await fullViewport.getAttribute("data-spark-sh-rest-coefficients") ?? "0",
    );
    const fullShDegree = numericValue(await fullViewport.getAttribute("data-spark-sh-degree") ?? "0");
    if (
      fullObjectFilter === "spark-ply-sh-source" &&
      (fullReconstructSource !== SPARK_RECONSTRUCT_SH_SOURCE ||
        fullVisibleGaussians <= 0 ||
        fullPackedBaseGaussians !== fullVisibleGaussians ||
        fullPackedVisibleIndices !== fullVisibleGaussians ||
        fullShRestSourceGaussians <= 0 ||
        fullShRestPreservedGaussians !== fullShRestSourceGaussians ||
        fullShRestPreserved !== "true" ||
        fullShRestCoefficientCount <= 0 ||
        fullShDegree <= 0)
    ) {
      throw new Error(
        `${asset.id} Spark PLY SH full source contract failed: filter=${fullObjectFilter} route=${fullReconstructSource} visible=${fullVisibleGaussians} packed=${fullPackedBaseGaussians}/${fullPackedVisibleIndices} shRest=${fullShRestSourceGaussians}:${fullShRestPreservedGaussians}:${fullShRestPreserved}:${fullShRestCoefficientCount}:${fullShDegree}`,
      );
    }

    await page.locator(".modeTabs").getByRole("button", { name: "对象编辑" }).click();
    await waitForSparkViewport(page, "spark-ply-reconstruct");
    await page.waitForTimeout(1000);
    const reconstructStats = await canvasVisualStats(page, ".viewport canvas", screenshotOptions);
    validateCanvasVisualStats(asset.id, "PLY reconstructed Spark", reconstructStats);

    const viewport = page.locator(".viewport").first();
    const objectFilter = await viewport.getAttribute("data-object-filter");
    const visibleGaussians = numericValue(await viewport.getAttribute("data-spark-visible-gaussians") ?? "0");
    const filteredGaussians = numericValue(await viewport.getAttribute("data-spark-filtered-gaussians") ?? "0");
    const colorSourceGaussians = numericValue(await viewport.getAttribute("data-spark-color-source-gaussians") ?? "0");
    const colorObjectGaussians = numericValue(await viewport.getAttribute("data-spark-color-object-gaussians") ?? "0");
    const reconstructSource = await viewport.getAttribute("data-spark-reconstruct-source");
    const packedBaseGaussians = numericValue(await viewport.getAttribute("data-spark-packed-base-gaussians") ?? "0");
    const packedVisibleIndices = numericValue(await viewport.getAttribute("data-spark-packed-visible-indices") ?? "0");
    const packedBaseBuildMs = finiteNumericValue(await viewport.getAttribute("data-spark-packed-base-build-ms") ?? "0");
    const packedExtractMs = finiteNumericValue(await viewport.getAttribute("data-spark-packed-extract-ms") ?? "0");
    const shRestSourceGaussians = numericValue(await viewport.getAttribute("data-spark-sh-rest-source-gaussians") ?? "0");
    const shRestPreservedGaussians = numericValue(
      await viewport.getAttribute("data-spark-sh-rest-preserved-gaussians") ?? "0",
    );
    const shRestPreserved = await viewport.getAttribute("data-spark-sh-rest-preserved");
    const shRestCoefficientCount = numericValue(
      await viewport.getAttribute("data-spark-sh-rest-coefficients") ?? "0",
    );
    const shDegree = numericValue(await viewport.getAttribute("data-spark-sh-degree") ?? "0");
    const expectedSparkRoute =
      shRestSourceGaussians > 0 ? SPARK_RECONSTRUCT_SH_SOURCE : SPARK_RECONSTRUCT_SOURCE;
    const expectedShPreserved = shRestSourceGaussians > 0 ? "true" : "false";
    if (
      objectFilter !== "spark-ply-reconstruct" ||
      reconstructSource !== expectedSparkRoute ||
      visibleGaussians <= 0 ||
      filteredGaussians !== 0 ||
      colorSourceGaussians !== visibleGaussians ||
      colorObjectGaussians !== 0 ||
      packedBaseGaussians !== visibleGaussians ||
      packedVisibleIndices !== visibleGaussians ||
      !Number.isFinite(packedBaseBuildMs) ||
      !Number.isFinite(packedExtractMs) ||
      shRestPreserved !== expectedShPreserved ||
      (shRestSourceGaussians > 0 &&
        (shRestPreservedGaussians !== shRestSourceGaussians ||
          shRestCoefficientCount <= 0 ||
          shDegree <= 0))
    ) {
      throw new Error(
        `${asset.id} Spark PLY reconstruct contract failed: filter=${objectFilter} route=${reconstructSource}/${expectedSparkRoute} visible=${visibleGaussians} filtered=${filteredGaussians} color=${colorSourceGaussians}/${colorObjectGaussians} packed=${packedBaseGaussians}/${packedVisibleIndices}:${packedBaseBuildMs}/${packedExtractMs} shRest=${shRestSourceGaussians}:${shRestPreservedGaussians}:${shRestPreserved}:${shRestCoefficientCount}:${shDegree}`,
      );
    }

    const residual = compareVisualStats(fullStats, reconstructStats);
    const gateChecks = evaluateGate(residual, gateThresholds);
    const screenshotPath = `/tmp/objgauss-spark-reconstruct-${asset.id}.png`;
    await page.screenshot({ path: screenshotPath, fullPage: false });
    const relevantIssues = consoleIssues.filter(
      (issue) =>
        !issue.includes("THREE.WebGLRenderer") &&
        !issue.includes("GPU stall due to ReadPixels") &&
        !issue.includes("No available adapters."),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }

    return {
      assetId: asset.id,
      assetName: asset.name,
      passed: gateChecks.every((check) => check.passed),
      gateChecks,
      full: fullStats,
      reconstruct: reconstructStats,
      residual,
      fullObjectFilter,
      fullReconstructSource,
      fullVisibleGaussians,
      fullPackedBaseGaussians,
      fullPackedVisibleIndices,
      fullShRestSourceGaussians,
      fullShRestPreservedGaussians,
      fullShRestPreserved,
      fullShRestCoefficientCount,
      fullShDegree,
      objectFilter,
      reconstructSource,
      visibleGaussians,
      filteredGaussians,
      colorSourceGaussians,
      colorObjectGaussians,
      packedBaseGaussians,
      packedVisibleIndices,
      packedBaseBuildMs,
      packedExtractMs,
      shRestSourceGaussians,
      shRestPreservedGaussians,
      shRestPreserved,
      shRestCoefficientCount,
      shDegree,
      screenshotPath,
    };
  } finally {
    await browser.close();
  }
}

function evaluateGate(residual, thresholds) {
  return [
    {
      metric: "coverageRatio",
      actual: residual.coverageRatio,
      operator: ">=",
      expected: thresholds.minCoverageRatio,
      passed: residual.coverageRatio >= thresholds.minCoverageRatio,
    },
    {
      metric: "coverageRatio",
      actual: residual.coverageRatio,
      operator: "<=",
      expected: thresholds.maxCoverageRatio,
      passed: residual.coverageRatio <= thresholds.maxCoverageRatio,
    },
    {
      metric: "lumaDelta",
      actual: residual.lumaDelta,
      operator: "<=",
      expected: thresholds.maxLumaDelta,
      passed: residual.lumaDelta <= thresholds.maxLumaDelta,
    },
    {
      metric: "chromaDelta",
      actual: residual.chromaDelta,
      operator: "<=",
      expected: thresholds.maxChromaDelta,
      passed: residual.chromaDelta <= thresholds.maxChromaDelta,
    },
  ];
}

function localGaussianAsset(assetId) {
  const asset = ASSET_LIBRARY.find((candidate) => candidate.id === assetId);
  if (!asset) throw new Error(`unknown asset id: ${assetId}`);
  if (!asset.localPath || !asset.splatPath) {
    throw new Error(`asset is not a local Gaussian viewer sample: ${assetId}`);
  }
  return asset;
}

function writeReportFiles(outputDir, summary) {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(`${outputDir}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(`${outputDir}/summary.md`, markdownReport(summary));
}

function markdownReport(summary) {
  const lines = [
    "# Spark Reconstruct Residual",
    "",
    `Mode: \`${summary.mode}\``,
    "",
    "| Asset | Gate | Coverage Ratio | Luma Delta | Chroma Delta | Full Source | Object Filter | Route | Visible | Extract ms | SH Rest |",
    "| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |",
  ];
  for (const result of summary.results) {
    lines.push(
      `| ${result.assetId} | ${result.passed ? "pass" : "fail"} | ${result.residual.coverageRatio} | ${result.residual.lumaDelta} | ${result.residual.chromaDelta} | ${result.fullObjectFilter}:${result.fullReconstructSource} | ${result.objectFilter} | ${result.reconstructSource} | ${result.visibleGaussians} | ${result.packedExtractMs} | ${result.shRestSourceGaussians}/${result.shRestPreservedGaussians}/${result.shRestPreserved}/${result.shRestCoefficientCount}/${result.shDegree} |`,
    );
  }
  lines.push(
    "",
    "Thresholds:",
    "",
    `- minCoverageRatio: ${summary.thresholds.minCoverageRatio}`,
    `- maxCoverageRatio: ${summary.thresholds.maxCoverageRatio}`,
    `- maxLumaDelta: ${summary.thresholds.maxLumaDelta}`,
    `- maxChromaDelta: ${summary.thresholds.maxChromaDelta}`,
    "",
    "Interpretation: this is a smoke residual gate for the Spark PLY reconstruction path. No-SH samples compare compact `.splat` full view against PLY reconstruction; SH-heavy samples can use a Spark PLY SH full-view source so full and reconstructed paths share the same SH-capable source representation.",
  );
  return `${lines.join("\n")}\n`;
}

function urlWithSparkReconstructProbe(url) {
  const parsed = new URL(url);
  parsed.searchParams.set(SPARK_RECONSTRUCT_PROBE_PARAM, "1");
  return parsed.toString();
}

async function waitForSparkViewport(page, objectFilter, timeoutMs = 15000) {
  const expectedFilters = Array.isArray(objectFilter) ? objectFilter : [objectFilter];
  await page.waitForFunction(
    (expectedObjectFilters) => {
      const viewport = document.querySelector(".viewport");
      return (
        viewport?.getAttribute("data-renderer") === "spark-splat" &&
        expectedObjectFilters.includes(viewport.getAttribute("data-object-filter")) &&
        viewport.getAttribute("data-spark-filter-status") === "ready"
      );
    },
    expectedFilters,
    { timeout: timeoutMs },
  );
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
  if (executablePath) {
    launch.executablePath = executablePath;
  }
  return launch;
}

function firstExisting(paths) {
  return paths.find((path) => existsSync(path));
}

function startPreviewServer(port) {
  const child = spawn(
    "npm",
    ["run", "preview", "--", "--port", String(port), "--strictPort"],
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

function parseAssets(value) {
  const parsed = String(value ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  return parsed.length > 0 ? parsed : DEFAULT_ASSETS;
}

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const text = String(value).trim();
  return text ? text : undefined;
}

function optionalFiniteNumber(value) {
  if (value === undefined || value === null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === false || value === undefined || value === null) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function numericValue(value) {
  return Number(String(value).replace(/[^\d]/g, ""));
}

function finiteNumericValue(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : Number.NaN;
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
