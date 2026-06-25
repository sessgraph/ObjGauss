import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5395;
const SPARK_NATIVE_SOURCE = "native-splat-source-v1";
const SPARK_MASK_SOURCE = "native-splat";
const SPARK_OBJECT_FILTER = "spark-object-opacity-mask";
const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";
const SPARK_DISPLAY_CACHE_DISABLED = "disabled-by-native-mask-v1";
const SPARK_MESH_UPDATE_MODE = "persistent-splatmesh-v1";

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
];

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = args.url ?? `http://127.0.0.1:${port}/`;
const assets = selectAssets(args);
const auditUrl = flagEnabled(args.forceNativeParam ?? args["force-native-param"])
  ? withNativeMask(baseUrl)
  : baseUrl;
const server = args.url || args.noServer ? null : startDevServer(port);

try {
  await waitForApp(baseUrl);
  const results = await runNativeMaskGate({ url: auditUrl, assets });
  for (const result of results) {
    console.log(
      `native_mask_gate_asset=passed asset=${JSON.stringify(result.assetId)} ` +
        `source=${JSON.stringify(result.maskSource)} route=${JSON.stringify(result.route)} ` +
        `visible=${result.visibleGaussians}/${result.baseGaussians} ` +
        `objectMask=${JSON.stringify(result.objectMaskMode)}:${JSON.stringify(result.objectMaskSize)}:${result.objectMaskVisible}/${result.objectMaskHidden}:${result.objectMaskUpdates} ` +
        `mesh=${JSON.stringify(result.meshMode)}:${result.meshId}:${JSON.stringify(result.meshReused)}:${result.meshUpdates} ` +
        `visual=${JSON.stringify(result.visualMode)}:${result.visualCoverageDelta}/${result.visualLumaDelta}/${result.visualChromaDelta}:${result.visualRestoreCoverageDelta}/${result.visualRestoreLumaDelta}/${result.visualRestoreChromaDelta} ` +
        `screenshot=${result.screenshotPath}`,
    );
  }
  console.log(
    `native_mask_gate=passed assets=${JSON.stringify(results.map((result) => result.assetId))} url=${baseUrl} forceNativeParam=${auditUrl !== baseUrl}`,
  );
} finally {
  if (server) stopDevServer(server);
}

async function runNativeMaskGate({ url, assets }) {
  const browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const results = [];
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  try {
    for (const asset of assets) {
      console.log(`native_mask_gate_asset_start asset=${JSON.stringify(asset.id)}`);
      await loadAsset(page, url, asset);
      await enterEditModeAndDeleteObject(page);
      await waitForNativeMaskReady(page);

      const stats = await readNativeMaskStats(page);
      validateNativeMaskStats(asset.id, stats);

      const screenshotPath = `/tmp/objgauss-native-mask-gate-${asset.id}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      results.push({
        assetId: asset.id,
        ...stats,
        ...skippedVisualDelta(),
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

async function waitForNativeMaskReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    return (
      viewport?.getAttribute("data-renderer") === "spark-splat" &&
      viewport?.getAttribute("data-object-filter") === "spark-object-opacity-mask" &&
      viewport?.getAttribute("data-spark-filter-status") === "ready"
    );
  }, undefined, { timeout: 120000 });
}

async function readNativeMaskStats(page) {
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
      extractMs: numberAttr("data-spark-packed-extract-ms"),
      displayCacheMode: viewport.getAttribute("data-spark-display-cache-mode"),
      displayCacheHit: viewport.getAttribute("data-spark-display-cache-hit"),
      displayCacheSize: numberAttr("data-spark-display-cache-size"),
      objectMaskMode: viewport.getAttribute("data-spark-object-mask-mode"),
      objectMaskSize: viewport.getAttribute("data-spark-object-mask-size"),
      objectMaskVisible: numberAttr("data-spark-object-mask-visible-gaussians"),
      objectMaskHidden: numberAttr("data-spark-object-mask-hidden-gaussians"),
      objectMaskUpdates: numberAttr("data-spark-object-mask-updates"),
      meshMode: viewport.getAttribute("data-spark-mesh-update-mode"),
      meshId: numberAttr("data-spark-mesh-id"),
      meshReused: viewport.getAttribute("data-spark-mesh-reused"),
      meshUpdates: numberAttr("data-spark-mesh-updates"),
    };
  });
}

function validateNativeMaskStats(assetId, stats) {
  const failures = [];
  if (stats.renderer !== "spark-splat") failures.push(`renderer=${stats.renderer}`);
  if (stats.objectFilter !== SPARK_OBJECT_FILTER) failures.push(`objectFilter=${stats.objectFilter}`);
  if (stats.filterStatus !== "ready") failures.push(`status=${stats.filterStatus}`);
  if (stats.filterMode !== "native-splat-mask") failures.push(`mode=${stats.filterMode}`);
  if (stats.maskSource !== SPARK_MASK_SOURCE) failures.push(`source=${stats.maskSource}`);
  if (stats.route !== SPARK_NATIVE_SOURCE) failures.push(`route=${stats.route}`);
  if (stats.visibleGaussians <= 0) failures.push(`visible=${stats.visibleGaussians}`);
  if (stats.removedObjects !== 1) failures.push(`removed=${stats.removedObjects}`);
  if (stats.colorMode !== "original") failures.push(`colorMode=${stats.colorMode}`);
  if (stats.colorSourceGaussians <= 0 || stats.colorObjectGaussians !== 0) {
    failures.push(`color=${stats.colorSourceGaussians}/${stats.colorObjectGaussians}`);
  }
  if (stats.baseGaussians <= stats.visibleGaussians) {
    failures.push(`base/visible=${stats.baseGaussians}/${stats.visibleGaussians}`);
  }
  if (stats.visibleIndices !== stats.visibleGaussians || stats.extractMs !== 0) {
    failures.push(`indices/extract=${stats.visibleIndices}/${stats.extractMs}`);
  }
  if (
    stats.displayCacheMode !== SPARK_DISPLAY_CACHE_DISABLED ||
    stats.displayCacheHit !== "false" ||
    stats.displayCacheSize !== 0
  ) {
    failures.push(
      `cache=${stats.displayCacheMode}:${stats.displayCacheHit}:${stats.displayCacheSize}`,
    );
  }
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
  if (
    stats.meshMode !== SPARK_MESH_UPDATE_MODE ||
    stats.meshId <= 0 ||
    stats.meshUpdates <= 0
  ) {
    failures.push(`mesh=${stats.meshMode}:${stats.meshId}:${stats.meshUpdates}`);
  }
  if (failures.length > 0) {
    throw new Error(`${assetId} native mask contract failed: ${failures.join(" ")}`);
  }
}

function skippedVisualDelta() {
  return {
    visualMode: "skipped-contract-gate-v1",
    visualCoverageDelta: 0,
    visualLumaDelta: 0,
    visualChromaDelta: 0,
    visualRestoreCoverageDelta: 0,
    visualRestoreLumaDelta: 0,
    visualRestoreChromaDelta: 0,
  };
}

function withNativeMask(url) {
  const parsed = new URL(url);
  parsed.searchParams.set("spark-native-mask", "on");
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

function flagEnabled(value) {
  if (value === true) return true;
  if (value === false || value === undefined || value === null) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
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
  if (!assetList) return KNOWN_ASSETS;
  if (assetList === true) {
    throw new Error("--assets requires a comma-separated asset id list");
  }
  const requested = String(assetList)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
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
