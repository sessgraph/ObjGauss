import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5315;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-spark-pick-report";
const REPORT_MODE = "spark-screen-space-pick-report-v1";
const SPARK_OBJECT_FILTER = "spark-object-opacity-mask";
const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";
const SPARK_PICK_MODE = "screen-space-object-pick-v1";
const DEFAULT_MIN_HITS = 1;
const DEFAULT_MIN_HIT_RATE = 0.2;
const DEFAULT_MAX_AMBIGUITY_RATE = 0.5;
const DEFAULT_CLICK_POINTS = [
  [0.5, 0.5],
  [0.45, 0.48],
  [0.55, 0.48],
  [0.4, 0.55],
  [0.6, 0.55],
  [0.5, 0.4],
  [0.35, 0.48],
  [0.65, 0.48],
  [0.42, 0.42],
  [0.58, 0.42],
  [0.42, 0.62],
  [0.58, 0.62],
  [0.5, 0.68],
  [0.32, 0.58],
  [0.68, 0.58],
];

const KNOWN_ASSETS = [
  {
    id: "nerf-lego-alpha-closure-local",
    name: "NeRF Lego 闭环代理样例",
    fileName: "lego_alpha_v1_objects.ply",
  },
  {
    id: "nerf-lego-trained-output-local",
    name: "NeRF Lego 训练输出样例",
    fileName: "nerf_lego_trained_objects.ply",
  },
  {
    id: "plush-semantic-closure-local",
    name: "Plush 2D 语义 Mask 闭环样例",
    fileName: "plush_semantic_objects.ply",
  },
];

const DEFAULT_ASSET_IDS = [
  "nerf-lego-alpha-closure-local",
];

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = args.url ?? `http://127.0.0.1:${port}/`;
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const minHits = optionalPositiveInteger(args.minHits ?? args["min-hits"]) ?? DEFAULT_MIN_HITS;
const minHitRate = optionalFiniteNumber(args.minHitRate ?? args["min-hit-rate"]) ?? DEFAULT_MIN_HIT_RATE;
const maxAmbiguityRate =
  optionalFiniteNumber(args.maxAmbiguityRate ?? args["max-ambiguity-rate"]) ??
  DEFAULT_MAX_AMBIGUITY_RATE;
const maxClicks = optionalPositiveInteger(args.maxClicks ?? args["max-clicks"]) ?? DEFAULT_CLICK_POINTS.length;
const assets = selectAssets(args);
const server = args.url || args.noServer ? null : startDevServer(port);

try {
  if (server) await ensureServerStarted(server);
  await mkdir(outputDir, { recursive: true });
  await waitForApp(baseUrl, server);
  const summary = await runPickReport({
    url: baseUrl,
    assets,
    outputDir,
    minHits,
    minHitRate,
    maxAmbiguityRate,
    maxClicks,
  });
  const summaryJson = `${outputDir}/summary.json`;
  const summaryMd = `${outputDir}/summary.md`;
  await writeFile(summaryJson, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  await writeFile(summaryMd, renderMarkdown(summary), "utf8");

  for (const result of summary.assets) {
    console.log(
      `spark_pick_report_asset=passed asset=${JSON.stringify(result.assetId)} ` +
        `clicks=${result.clicks} hits=${result.hits} misses=${result.misses} ` +
        `hitRate=${result.hitRate} ambiguousHits=${result.ambiguousHits} ` +
        `ambiguityRate=${result.ambiguityRate} markerHits=${result.markerHits}/${result.hits} ` +
        `pickStrategy=${JSON.stringify(result.pickStrategy)} scoreMargin=${result.meanScoreMargin}/${result.minScoreMargin} ` +
        `distinctHitObjects=${JSON.stringify(result.distinctHitObjects)} ` +
        `maskSource=${JSON.stringify(result.maskSource)} route=${JSON.stringify(result.route)} ` +
        `visible=${result.visibleGaussians}/${result.baseGaussians} screenshot=${result.screenshotPath}`,
    );
  }
  console.log(
    `spark_pick_report=passed assets=${JSON.stringify(summary.assets.map((asset) => asset.assetId))} ` +
      `summaryJson=${JSON.stringify(summaryJson)} summaryMd=${JSON.stringify(summaryMd)} ` +
      `minHits=${minHits} minHitRate=${minHitRate} maxAmbiguityRate=${maxAmbiguityRate} maxClicks=${maxClicks}`,
  );
} finally {
  if (server) stopDevServer(server);
}

async function runPickReport({
  url,
  assets,
  outputDir,
  minHits,
  minHitRate,
  maxAmbiguityRate,
  maxClicks,
}) {
  const browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  try {
    const results = [];
    for (const asset of assets) {
      console.log(`spark_pick_report_asset_start asset=${JSON.stringify(asset.id)}`);
      await loadAsset(page, url, asset);
      await enterSparkDeletePreview(page);
      const deleteStats = await readSparkDeleteStats(page);
      validateSparkDeleteStats(asset.id, deleteStats);
      const clickResults = await runPickClicks(page, asset, { maxClicks });
      const result = summarizePickResults({
        asset,
        deleteStats,
        clickResults,
        minHits,
        minHitRate,
        maxAmbiguityRate,
      });
      validatePickSummary(result, { minHits, minHitRate, maxAmbiguityRate });
      console.log(
        `spark_pick_report_asset_progress asset=${JSON.stringify(result.assetId)} ` +
          `clicks=${result.clicks} hits=${result.hits} hitRate=${result.hitRate} ` +
          `ambiguousHits=${result.ambiguousHits} ambiguityRate=${result.ambiguityRate}`,
      );

      const screenshotPath = `${outputDir}/${asset.id}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      results.push({
        ...result,
        screenshotPath,
      });
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
    return {
      mode: REPORT_MODE,
      generatedAt: new Date().toISOString(),
      url,
      gates: {
        minHits,
        minHitRate,
        maxAmbiguityRate,
        maxClicks,
        ambiguityIsReportedOnly: false,
      },
      assets: results,
    };
  } finally {
    await closeBrowserWithTimeout(browser);
  }
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
  }, undefined, { timeout: 90000 });
}

async function enterSparkDeletePreview(page) {
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
  await waitForSparkDeleteReady(page);
}

async function waitForSparkDeleteReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    return (
      viewport?.getAttribute("data-renderer") === "spark-splat" &&
      viewport?.getAttribute("data-object-filter") === "spark-object-opacity-mask" &&
      viewport?.getAttribute("data-spark-filter-status") === "ready"
    );
  }, undefined, { timeout: 120000 });
}

async function readSparkDeleteStats(page) {
  return page.locator(".viewport").first().evaluate((viewport) => {
    const numberAttr = (name) => {
      const parsed = Number(viewport.getAttribute(name) ?? "0");
      return Number.isFinite(parsed) ? parsed : 0;
    };
    return {
      renderer: viewport.getAttribute("data-renderer"),
      objectFilter: viewport.getAttribute("data-object-filter"),
      filterStatus: viewport.getAttribute("data-spark-filter-status"),
      filterMode: viewport.getAttribute("data-spark-filter-mode"),
      maskSource: viewport.getAttribute("data-spark-mask-source"),
      route: viewport.getAttribute("data-spark-reconstruct-source"),
      visibleGaussians: numberAttr("data-spark-visible-gaussians"),
      removedObjects: numberAttr("data-spark-removed-objects"),
      colorMode: viewport.getAttribute("data-spark-color-mode"),
      colorSourceGaussians: numberAttr("data-spark-color-source-gaussians"),
      colorObjectGaussians: numberAttr("data-spark-color-object-gaussians"),
      baseGaussians: numberAttr("data-spark-packed-base-gaussians"),
      visibleIndices: numberAttr("data-spark-packed-visible-indices"),
      objectMaskMode: viewport.getAttribute("data-spark-object-mask-mode"),
      objectMaskVisible: numberAttr("data-spark-object-mask-visible-gaussians"),
      objectMaskHidden: numberAttr("data-spark-object-mask-hidden-gaussians"),
      meshMode: viewport.getAttribute("data-spark-mesh-update-mode"),
      selectionMode: viewport.getAttribute("data-spark-selection-mode"),
    };
  });
}

function validateSparkDeleteStats(assetId, stats) {
  const failures = [];
  if (stats.renderer !== "spark-splat") failures.push(`renderer=${stats.renderer}`);
  if (stats.objectFilter !== SPARK_OBJECT_FILTER) failures.push(`objectFilter=${stats.objectFilter}`);
  if (stats.filterStatus !== "ready") failures.push(`status=${stats.filterStatus}`);
  if (stats.visibleGaussians <= 0) failures.push(`visible=${stats.visibleGaussians}`);
  if (stats.removedObjects !== 1) failures.push(`removed=${stats.removedObjects}`);
  if (stats.colorMode !== "original") failures.push(`colorMode=${stats.colorMode}`);
  if (stats.colorSourceGaussians <= 0 || stats.colorObjectGaussians !== 0) {
    failures.push(`color=${stats.colorSourceGaussians}/${stats.colorObjectGaussians}`);
  }
  if (stats.baseGaussians <= stats.visibleGaussians) {
    failures.push(`base/visible=${stats.baseGaussians}/${stats.visibleGaussians}`);
  }
  if (stats.visibleIndices !== stats.visibleGaussians) {
    failures.push(`indices=${stats.visibleIndices}/${stats.visibleGaussians}`);
  }
  if (
    stats.objectMaskMode !== SPARK_OBJECT_MASK_MODE ||
    stats.objectMaskVisible !== stats.visibleGaussians ||
    stats.objectMaskHidden !== stats.baseGaussians - stats.visibleGaussians
  ) {
    failures.push(
      `objectMask=${stats.objectMaskMode}:${stats.objectMaskVisible}/${stats.objectMaskHidden}`,
    );
  }
  if (stats.selectionMode !== SPARK_PICK_MODE) failures.push(`selection=${stats.selectionMode}`);
  if (failures.length > 0) {
    throw new Error(`${assetId} Spark delete preview contract failed: ${failures.join(" ")}`);
  }
}

async function runPickClicks(page, asset, { maxClicks }) {
  const canvas = page.locator(".viewport canvas").first();
  const box = await canvas.boundingBox();
  if (!box) throw new Error(`${asset.id} Spark canvas is missing`);

  const results = [];
  for (const [index, point] of DEFAULT_CLICK_POINTS.slice(0, maxClicks).entries()) {
    const [xRatio, yRatio] = point;
    const selectedBefore = await selectedObjectValue(page);
    const x = Math.round(box.x + box.width * xRatio);
    const y = Math.round(box.y + box.height * yRatio);
    await page.mouse.click(x, y);
    await page.waitForTimeout(220);
    const selectedAfter = await selectedObjectValue(page);
    const pick = await readSparkPickStats(page);
    results.push({
      index,
      xRatio,
      yRatio,
      x,
      y,
      selectedBefore,
      selectedAfter,
      ...pick,
      validHit: validPickHit(pick, selectedAfter),
    });
  }
  return results;
}

async function selectedObjectValue(page) {
  const status = await page.locator(".statusBar").innerText();
  const match = status.match(/所选：([^\n]+)/);
  return match?.[1] ?? "无";
}

async function readSparkPickStats(page) {
  return page.locator(".viewport").first().evaluate((viewport) => {
    const numericAttr = (name) => {
      const numeric = Number(viewport.getAttribute(name) ?? "0");
      return Number.isFinite(numeric) ? numeric : 0;
    };
    return {
      mode: viewport.getAttribute("data-spark-selection-mode") ?? "",
      status: viewport.getAttribute("data-spark-pick-status") ?? "",
      strategy: viewport.getAttribute("data-spark-pick-strategy") ?? "",
      object: viewport.getAttribute("data-spark-pick-object") ?? "",
      distancePx: numericAttr("data-spark-pick-distance-px"),
      candidateObjects: numericAttr("data-spark-pick-candidate-objects"),
      ambiguous: viewport.getAttribute("data-spark-pick-ambiguous") ?? "",
      radiusPx: numericAttr("data-spark-pick-radius-px"),
      score: numericAttr("data-spark-pick-score"),
      scoreMargin: numericAttr("data-spark-pick-score-margin"),
      secondObject: viewport.getAttribute("data-spark-pick-second-object") ?? "",
      secondScore: numericAttr("data-spark-pick-second-score"),
      markerVisible: viewport.getAttribute("data-spark-selected-marker-visible") ?? "",
    };
  });
}

function validPickHit(pick, selectedAfter) {
  return (
    pick.mode === SPARK_PICK_MODE &&
    pick.status === "hit" &&
    pick.object !== "" &&
    String(pick.object) === String(selectedAfter) &&
    pick.distancePx <= pick.radiusPx &&
    pick.candidateObjects > 0 &&
    pick.markerVisible === "true"
  );
}

function summarizePickResults({
  asset,
  deleteStats,
  clickResults,
  minHits,
  minHitRate,
  maxAmbiguityRate,
}) {
  const hits = clickResults.filter((click) => click.status === "hit");
  const validHits = clickResults.filter((click) => click.validHit);
  const invalidHits = hits.filter((click) => !click.validHit);
  const ambiguousHits = hits.filter((click) => click.ambiguous === "true");
  const markerHits = hits.filter((click) => click.markerVisible === "true");
  const distinctHitObjects = [...new Set(hits.map((click) => click.object).filter(Boolean))].sort(
    naturalSort,
  );
  const clicks = clickResults.length;
  const hitRate = ratio(hits.length, clicks);
  const ambiguityRate = ratio(ambiguousHits.length, Math.max(1, hits.length));
  const markerRate = ratio(markerHits.length, Math.max(1, hits.length));
  return {
    assetId: asset.id,
    assetName: asset.name,
    mode: REPORT_MODE,
    clicks,
    hits: hits.length,
    validHits: validHits.length,
    invalidHits: invalidHits.length,
    misses: clicks - hits.length,
    hitRate,
    ambiguityRate,
    ambiguousHits: ambiguousHits.length,
    markerHits: markerHits.length,
    markerRate,
    pickStrategy: firstNonEmpty(hits.map((click) => click.strategy)),
    distinctHitObjects,
    maxCandidateObjects: Math.max(0, ...clickResults.map((click) => click.candidateObjects)),
    meanHitDistancePx: roundMetric(mean(hits.map((click) => click.distancePx))),
    minHitDistancePx: roundMetric(Math.min(...hits.map((click) => click.distancePx))),
    maxHitDistancePx: roundMetric(Math.max(0, ...hits.map((click) => click.distancePx))),
    meanScoreMargin: roundMetric(mean(hits.map((click) => click.scoreMargin))),
    minScoreMargin: roundMetric(Math.min(...hits.map((click) => click.scoreMargin))),
    meanPickScore: roundMetric(mean(hits.map((click) => click.score))),
    gate: {
      minHits,
      minHitRate,
      maxAmbiguityRate,
      passed:
        hits.length >= minHits &&
        hitRate >= minHitRate &&
        ambiguityRate <= maxAmbiguityRate &&
        invalidHits.length === 0 &&
        markerHits.length === hits.length,
    },
    renderer: deleteStats.renderer,
    objectFilter: deleteStats.objectFilter,
    filterMode: deleteStats.filterMode,
    maskSource: deleteStats.maskSource,
    route: deleteStats.route,
    baseGaussians: deleteStats.baseGaussians,
    visibleGaussians: deleteStats.visibleGaussians,
    removedObjects: deleteStats.removedObjects,
    objectMaskMode: deleteStats.objectMaskMode,
    clicksDetail: clickResults,
  };
}

function validatePickSummary(result, { minHits, minHitRate, maxAmbiguityRate }) {
  const failures = [];
  if (result.hits < minHits) failures.push(`hits=${result.hits}<${minHits}`);
  if (result.hitRate < minHitRate) failures.push(`hitRate=${result.hitRate}<${minHitRate}`);
  if (result.ambiguityRate > maxAmbiguityRate) {
    failures.push(`ambiguityRate=${result.ambiguityRate}>${maxAmbiguityRate}`);
  }
  if (result.invalidHits > 0) failures.push(`invalidHits=${result.invalidHits}`);
  if (result.markerHits !== result.hits) {
    failures.push(`markerHits=${result.markerHits}/${result.hits}`);
  }
  if (failures.length > 0) {
    throw new Error(`${result.assetId} Spark pick report gate failed: ${failures.join(" ")}`);
  }
}

function renderMarkdown(summary) {
  const lines = [
    "# Spark Screen-Space Pick Report",
    "",
    `- Mode: \`${summary.mode}\``,
    `- URL: ${summary.url}`,
    `- Generated: ${summary.generatedAt}`,
    `- Gate: hits >= ${summary.gates.minHits}, hit rate >= ${summary.gates.minHitRate}, ambiguity rate <= ${summary.gates.maxAmbiguityRate}, max clicks = ${summary.gates.maxClicks}`,
    "- Ambiguity is now gated as a regression signal.",
    "",
    "| Asset | Route | Strategy | Clicks | Hits | Hit Rate | Ambiguous Hits | Ambiguity Rate | Mean Margin | Marker Hits | Objects |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ];
  for (const asset of summary.assets) {
    lines.push(
      `| ${asset.assetId} | ${asset.maskSource}/${asset.route} | ${asset.pickStrategy} | ${asset.clicks} | ${asset.hits} | ${asset.hitRate} | ${asset.ambiguousHits} | ${asset.ambiguityRate} | ${asset.meanScoreMargin} | ${asset.markerHits}/${asset.hits} | ${asset.distinctHitObjects.join(", ")} |`,
    );
  }
  lines.push("", "## Interpretation", "");
  lines.push(
    "This report measures Spark canvas selection after an object has been deleted with `spark-object-opacity-mask`. A hit is valid only when the DOM pick object matches the selected object, the pick distance is inside the advertised radius, at least one candidate object exists, and the marker is visible.",
  );
  lines.push(
    "High ambiguity means several object ids are close to the click in screen space. The report gates ambiguity rate as a regression signal, but this is still a screen-space CPU pick over object-aware PLY metadata rather than a Spark-internal raycast.",
  );
  lines.push("", "## Click Details", "");
  for (const asset of summary.assets) {
    lines.push(`### ${asset.assetId}`, "");
    lines.push("| # | x | y | Status | Object | Selected | Distance | Candidates | Score | Margin | Second | Ambiguous | Marker | Valid |");
    lines.push("| ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |");
    for (const click of asset.clicksDetail) {
      lines.push(
        `| ${click.index} | ${click.xRatio} | ${click.yRatio} | ${click.status} | ${click.object} | ${click.selectedAfter} | ${click.distancePx} | ${click.candidateObjects} | ${click.score} | ${click.scoreMargin} | ${click.secondObject} | ${click.ambiguous} | ${click.markerVisible} | ${click.validHit} |`,
      );
    }
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
}

function launchOptions() {
  const executablePath = firstExisting([
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ]);
  const launch = {
    args: ["--no-sandbox"],
    headless: true,
  };
  if (executablePath) launch.executablePath = executablePath;
  return launch;
}

function firstExisting(paths) {
  return paths.find((path) => existsSync(path));
}

function startDevServer(port) {
  const child = spawn(
    "npm",
    ["run", "preview", "--", "--host", "127.0.0.1", "--port", String(port), "--strictPort"],
    { detached: true, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

async function ensureServerStarted(child) {
  await sleep(750);
  if (child.exitCode !== null) {
    throw new Error(`preview server exited before audit started: exitCode=${child.exitCode}`);
  }
}

function stopDevServer(child) {
  if (!child.pid) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    child.kill("SIGTERM");
  }
}

async function waitForApp(url, child = null) {
  const deadline = Date.now() + 30000;
  let lastError = null;
  while (Date.now() < deadline) {
    if (child?.exitCode !== null) {
      throw new Error(`preview server exited before app became ready: exitCode=${child.exitCode}`);
    }
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

function selectAssets(parsedArgs) {
  const assetList = parsedArgs.assets;
  const requested = assetList
    ? String(assetList)
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean)
    : DEFAULT_ASSET_IDS;
  if (requested.length === 0 || assetList === true) {
    throw new Error("--assets requires a comma-separated asset id list");
  }
  const byId = new Map(KNOWN_ASSETS.map((asset) => [asset.id, asset]));
  const unknown = requested.filter((id) => !byId.has(id));
  if (unknown.length > 0) {
    throw new Error(`unknown asset id(s): ${unknown.join(",")}`);
  }
  return requested.map((id) => byId.get(id));
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

function optionalPositiveInteger(value) {
  if (value === undefined || value === null || value === true) return null;
  const numeric = Number(value);
  if (!Number.isInteger(numeric) || numeric <= 0) {
    throw new Error(`expected positive integer, got ${value}`);
  }
  return numeric;
}

function optionalFiniteNumber(value) {
  if (value === undefined || value === null || value === true) return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`expected finite number, got ${value}`);
  }
  return numeric;
}

function ratio(numerator, denominator) {
  if (!Number.isFinite(denominator) || denominator <= 0) return 0;
  return roundMetric(numerator / denominator);
}

function mean(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return 0;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function firstNonEmpty(values) {
  return values.find((value) => value !== undefined && value !== null && String(value) !== "") ?? "";
}

function roundMetric(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Number(numeric.toFixed(6)) : 0;
}

function naturalSort(a, b) {
  const numericA = Number(a);
  const numericB = Number(b);
  if (Number.isFinite(numericA) && Number.isFinite(numericB)) {
    return numericA - numericB;
  }
  return String(a).localeCompare(String(b));
}
