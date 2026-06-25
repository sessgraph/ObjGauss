import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5395;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-spark-native-pick-feasibility";
const REPORT_MODE = "spark-native-pick-feasibility-v1";
const SPARK_OBJECT_FILTER = "spark-object-opacity-mask";
const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";
const SPARK_PICK_MODE = "screen-space-object-pick-v1";

const KNOWN_ASSETS = [
  {
    id: "nerf-lego-alpha-closure-local",
    name: "NeRF Lego 闭环代理样例",
    fileName: "lego_alpha_v1_objects.ply",
  },
  {
    id: "plush-semantic-closure-local",
    name: "Plush 2D 语义 Mask 闭环样例",
    fileName: "plush_semantic_objects.ply",
  },
  {
    id: "nerf-lego-trained-output-local",
    name: "NeRF Lego 训练输出样例",
    fileName: "nerf_lego_trained_objects.ply",
  },
];

const DEFAULT_ASSET_IDS = ["nerf-lego-alpha-closure-local"];

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const assets = selectAssets(args);
const auditUrl = withNativePickProbe(baseUrl);
const server = args.url || args.noServer || args["no-server"] ? null : startPreviewServer(port);

try {
  if (server) await ensureServerStarted(server);
  await mkdir(outputDir, { recursive: true });
  await waitForApp(baseUrl, server);
  const summary = await runFeasibilityReport({ url: auditUrl, assets, outputDir });
  const summaryJson = `${outputDir}/summary.json`;
  const summaryMd = `${outputDir}/summary.md`;
  await writeFile(summaryJson, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  await writeFile(summaryMd, renderMarkdown(summary), "utf8");

  for (const result of summary.assets) {
    console.log(
      `spark_native_pick_feasibility_asset=passed asset=${JSON.stringify(result.assetId)} ` +
        `raycast=${result.raycastFunction}:${result.raycastable}:${result.sampleStatus}:${result.sampleHits} ` +
        `keys=${JSON.stringify(result.intersectionKeys)} index=${result.returnsSplatIndex} ` +
        `objectId=${result.returnsObjectId} filterAware=${result.objectFilterAware} ` +
        `metadata=${JSON.stringify(result.objectMetadata)} recommendation=${JSON.stringify(result.recommendation)} ` +
        `blocker=${JSON.stringify(result.blocker)} screenshot=${JSON.stringify(result.screenshotPath)}`,
    );
  }
  console.log(
    `spark_native_pick_feasibility=passed assets=${JSON.stringify(summary.assets.map((asset) => asset.assetId))} ` +
      `url=${JSON.stringify(summary.url)} summaryJson=${JSON.stringify(summaryJson)} summaryMd=${JSON.stringify(summaryMd)}`,
  );
} finally {
  if (server) stopPreviewServer(server);
}

async function runFeasibilityReport({ url, assets, outputDir }) {
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
      console.log(`spark_native_pick_feasibility_asset_start asset=${JSON.stringify(asset.id)}`);
      await loadAsset(page, url, asset);
      await enterSparkDeletePreview(page);
      const stats = await readFeasibilityStats(page);
      validateFeasibilityStats(asset.id, stats);

      const screenshotPath = `${outputDir}/${asset.id}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      results.push({
        assetId: asset.id,
        assetName: asset.name,
        ...stats,
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
      fixedPort: DEFAULT_PORT,
      assets: results,
      conclusion: summarizeConclusion(results),
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
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    return (
      viewport?.getAttribute("data-renderer") === "spark-splat" &&
      viewport?.getAttribute("data-object-filter") === "spark-object-opacity-mask" &&
      viewport?.getAttribute("data-spark-filter-status") === "ready" &&
      viewport?.getAttribute("data-spark-native-pick-probe-enabled") === "true" &&
      viewport?.getAttribute("data-spark-native-pick-sample-status") !== "idle"
    );
  }, undefined, { timeout: 120000 });
}

async function readFeasibilityStats(page) {
  return page.locator(".viewport").first().evaluate((viewport) => {
    const numberAttr = (name) => {
      const parsed = Number(viewport.getAttribute(name) ?? "0");
      return Number.isFinite(parsed) ? parsed : 0;
    };
    const booleanAttr = (name) => viewport.getAttribute(name) === "true";
    return {
      renderer: viewport.getAttribute("data-renderer"),
      objectFilter: viewport.getAttribute("data-object-filter"),
      filterStatus: viewport.getAttribute("data-spark-filter-status"),
      objectMaskMode: viewport.getAttribute("data-spark-object-mask-mode"),
      selectionMode: viewport.getAttribute("data-spark-selection-mode"),
      visibleGaussians: numberAttr("data-spark-visible-gaussians"),
      baseGaussians: numberAttr("data-spark-packed-base-gaussians"),
      removedObjects: numberAttr("data-spark-removed-objects"),
      probeMode: viewport.getAttribute("data-spark-native-pick-probe-mode"),
      probeEnabled: booleanAttr("data-spark-native-pick-probe-enabled"),
      raycastFunction: booleanAttr("data-spark-native-pick-raycast-function"),
      raycastable: booleanAttr("data-spark-native-pick-raycastable"),
      sampleStatus: viewport.getAttribute("data-spark-native-pick-sample-status"),
      sampleHits: numberAttr("data-spark-native-pick-sample-hits"),
      intersectionKeys: viewport.getAttribute("data-spark-native-pick-intersection-keys") ?? "",
      returnsSplatIndex: booleanAttr("data-spark-native-pick-returns-splat-index"),
      returnsObjectId: booleanAttr("data-spark-native-pick-returns-object-id"),
      objectFilterAware: booleanAttr("data-spark-native-pick-object-filter-aware"),
      sourceType: viewport.getAttribute("data-spark-native-pick-source-type"),
      sourceSplats: numberAttr("data-spark-native-pick-source-splats"),
      sourceMethods: viewport.getAttribute("data-spark-native-pick-source-methods") ?? "",
      objectMetadata: viewport.getAttribute("data-spark-native-pick-object-metadata"),
      recommendation: viewport.getAttribute("data-spark-native-pick-recommendation"),
      blocker: viewport.getAttribute("data-spark-native-pick-blocker"),
    };
  });
}

function validateFeasibilityStats(assetId, stats) {
  const failures = [];
  if (stats.renderer !== "spark-splat") failures.push(`renderer=${stats.renderer}`);
  if (stats.objectFilter !== SPARK_OBJECT_FILTER) failures.push(`objectFilter=${stats.objectFilter}`);
  if (stats.filterStatus !== "ready") failures.push(`status=${stats.filterStatus}`);
  if (stats.objectMaskMode !== SPARK_OBJECT_MASK_MODE) failures.push(`mask=${stats.objectMaskMode}`);
  if (stats.selectionMode !== SPARK_PICK_MODE) failures.push(`selection=${stats.selectionMode}`);
  if (stats.visibleGaussians <= 0 || stats.baseGaussians <= stats.visibleGaussians) {
    failures.push(`visible/base=${stats.visibleGaussians}/${stats.baseGaussians}`);
  }
  if (stats.removedObjects !== 1) failures.push(`removed=${stats.removedObjects}`);
  if (stats.probeMode !== REPORT_MODE) failures.push(`probeMode=${stats.probeMode}`);
  if (!stats.probeEnabled) failures.push("probeEnabled=false");
  if (!stats.raycastFunction) failures.push("raycastFunction=false");
  if (!stats.raycastable) failures.push("raycastable=false");
  if (stats.sampleStatus !== "hit") failures.push(`sampleStatus=${stats.sampleStatus}`);
  if (stats.sampleHits <= 0) failures.push(`sampleHits=${stats.sampleHits}`);
  if (!stats.intersectionKeys) failures.push("intersectionKeys=empty");
  if (stats.recommendation === "candidate-native-raycast-object-pick" && stats.blocker !== "none") {
    failures.push(`candidateBlocker=${stats.blocker}`);
  }
  if (stats.recommendation === "keep-screen-space-hover-confirm" && stats.blocker === "none") {
    failures.push("keepRecommendationWithoutBlocker");
  }
  if (!["candidate-native-raycast-object-pick", "keep-screen-space-hover-confirm"].includes(stats.recommendation)) {
    failures.push(`recommendation=${stats.recommendation}`);
  }
  if (failures.length > 0) {
    throw new Error(`${assetId} Spark native pick feasibility failed: ${failures.join(" ")}`);
  }
}

function summarizeConclusion(results) {
  const candidates = results.filter(
    (result) => result.recommendation === "candidate-native-raycast-object-pick",
  );
  if (candidates.length === results.length && results.length > 0) {
    return {
      recommendation: "candidate-native-raycast-object-pick",
      reason: "All assets expose raycast hits with index-aligned object metadata.",
    };
  }
  const blockers = [...new Set(results.map((result) => result.blocker).filter(Boolean))].sort();
  return {
    recommendation: "keep-screen-space-hover-confirm",
    reason:
      "Spark raycast is available, but current intersections do not expose enough object-level metadata for a safe product migration.",
    blockers,
  };
}

function renderMarkdown(summary) {
  const lines = [
    "# Spark Native Pick Feasibility",
    "",
    `- Mode: \`${summary.mode}\``,
    `- URL: ${summary.url}`,
    `- Fixed port: ${summary.fixedPort}`,
    `- Generated: ${summary.generatedAt}`,
    `- Recommendation: \`${summary.conclusion.recommendation}\``,
    `- Reason: ${summary.conclusion.reason}`,
    "",
    "| Asset | Raycast | Sample | Hits | Keys | Splat Index | Object ID | Filter Aware | Metadata | Recommendation | Blocker |",
    "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
  ];
  for (const asset of summary.assets) {
    lines.push(
      `| ${asset.assetId} | ${asset.raycastFunction}/${asset.raycastable} | ${asset.sampleStatus} | ${asset.sampleHits} | ${escapeMarkdown(asset.intersectionKeys)} | ${asset.returnsSplatIndex} | ${asset.returnsObjectId} | ${asset.objectFilterAware} | ${escapeMarkdown(asset.objectMetadata)} | ${escapeMarkdown(asset.recommendation)} | ${escapeMarkdown(asset.blocker)} |`,
    );
  }
  lines.push("", "## Interpretation", "");
  lines.push(
    "Spark `SplatMesh.raycast` is useful as a depth hit probe, but the current intersection payload must expose a splat index or object id before ObjGauss can map a renderer-native hit back to object-aware PLY metadata.",
  );
  lines.push(
    "Until that is true, the product path should keep `hover-confirm-v1` screen-space object picking. It is explicit, auditable, and respects ObjGauss visible / removed / isolated object state.",
  );
  return `${lines.join("\n")}\n`;
}

function withNativePickProbe(url) {
  const parsed = new URL(url);
  parsed.searchParams.set("spark-native-pick-probe", "1");
  return parsed.toString();
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

async function ensureServerStarted(child) {
  await sleep(750);
  if (child.exitCode !== null) {
    throw new Error(`preview server exited before audit started: exitCode=${child.exitCode}`);
  }
}

function stopPreviewServer(child) {
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

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
}
