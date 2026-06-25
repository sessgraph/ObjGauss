import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5395;
const SPARK_OBJECT_FILTER = "spark-object-opacity-mask";
const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";
const SPARK_DISPLAY_CACHE_DISABLED = "disabled-by-native-mask-v1";
const SPARK_MESH_UPDATE_MODE = "persistent-splatmesh-v1";
const EXPECTED_INITIAL_ROUTE = "spark-ply-sh-source";
const EXPECTED_DELETE_ROUTE = "spark-packed-sh-mask";
const EXPECTED_MASK_SOURCE = "ply-packed";
const EXPECTED_RECONSTRUCT_SOURCE = "packed-sh-extract-v1";

const KNOWN_ASSETS = [
  {
    id: "nerf-lego-trained-output-local",
    name: "NeRF Lego 训练输出样例",
    fileName: "nerf_lego_trained_objects.ply",
  },
  {
    id: "polyhaven-chair-commercial-demo-local",
    name: "Poly Haven Chair 商用展示样例",
    fileName: "polyhaven_chair_demo_objects.ply",
  },
];

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = args.url ?? `http://127.0.0.1:${port}/`;
const assets = selectAssets(args);
const server = args.url || args.noServer ? null : startPreviewServer(port);

try {
  await waitForApp(baseUrl);
  const results = await runRouteAudit({ url: baseUrl, assets });
  for (const result of results) {
    console.log(
      `spark_trained_route_asset=passed asset=${JSON.stringify(result.assetId)} ` +
        `initial=${JSON.stringify(result.initialRoute)}:${JSON.stringify(result.initialKind)}:${JSON.stringify(result.initialBoundary)} ` +
        `delete=${JSON.stringify(result.deleteRoute)}:${JSON.stringify(result.deleteKind)}:${JSON.stringify(result.deleteBoundary)} ` +
        `spark=${JSON.stringify(result.maskSource)}:${JSON.stringify(result.reconstructSource)} ` +
        `visible=${result.visibleGaussians}/${result.baseGaussians} ` +
        `objectMask=${JSON.stringify(result.objectMaskMode)}:${JSON.stringify(result.objectMaskSize)}:${result.objectMaskVisible}/${result.objectMaskHidden}:${result.objectMaskUpdates} ` +
        `shRest=${result.shRestSource}:${result.shRestPreserved}:${JSON.stringify(result.shRestPreservedFlag)}:${result.shRestCoefficients}:${result.shDegree} ` +
        `screenshot=${result.screenshotPath}`,
    );
  }
  console.log(
    `spark_trained_route=passed assets=${JSON.stringify(results.map((result) => result.assetId))} url=${baseUrl}`,
  );
} finally {
  if (server) stopServer(server);
}

async function runRouteAudit({ url, assets }) {
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
      console.log(`spark_trained_route_asset_start asset=${JSON.stringify(asset.id)}`);
      await loadAsset(page, url, asset);
      const initialStats = await readRouteStats(page);
      validateInitialRoute(asset.id, initialStats);

      await enterEditModeAndDeleteObject(page);
      await waitForSparkDeleteReady(page);
      const deleteStats = await readRouteStats(page);
      validateDeleteRoute(asset.id, deleteStats);

      const screenshotPath = `/tmp/objgauss-spark-trained-route-${asset.id}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      results.push({
        assetId: asset.id,
        ...initialResultFields(initialStats),
        ...deleteResultFields(deleteStats),
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
    const app = document.querySelector(".appShell");
    return (
      viewport?.getAttribute("data-renderer") === "spark-splat" &&
      viewport?.getAttribute("data-spark-filter-status") === "ready" &&
      app?.getAttribute("data-renderer-route-kind") === "commercial"
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
      appQuality: appAttr("data-preview-quality"),
      renderer: viewport.getAttribute("data-renderer"),
      objectFilter: viewport.getAttribute("data-object-filter"),
      filterStatus: viewport.getAttribute("data-spark-filter-status"),
      filterMode: viewport.getAttribute("data-spark-filter-mode"),
      maskSource: viewport.getAttribute("data-spark-mask-source"),
      reconstructSource: viewport.getAttribute("data-spark-reconstruct-source"),
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
      meshUpdates: numberAttr("data-spark-mesh-updates"),
      shRestSource: numberAttr("data-spark-sh-rest-source-gaussians"),
      shRestPreserved: numberAttr("data-spark-sh-rest-preserved-gaussians"),
      shRestPreservedFlag: viewport.getAttribute("data-spark-sh-rest-preserved"),
      shRestCoefficients: numberAttr("data-spark-sh-rest-coefficients"),
      shDegree: numberAttr("data-spark-sh-degree"),
    };
  });
}

function validateInitialRoute(assetId, stats) {
  const failures = [];
  if (stats.appRoute !== EXPECTED_INITIAL_ROUTE) failures.push(`route=${stats.appRoute}`);
  if (stats.appKind !== "commercial") failures.push(`kind=${stats.appKind}`);
  if (stats.appColorRole !== "source-color") failures.push(`colorRole=${stats.appColorRole}`);
  if (stats.appBoundary !== "source-splat") failures.push(`boundary=${stats.appBoundary}`);
  if (stats.appQuality !== "ply-sh-source") failures.push(`quality=${stats.appQuality}`);
  if (stats.renderer !== "spark-splat") failures.push(`renderer=${stats.renderer}`);
  if (stats.objectFilter !== "spark-ply-sh-source") failures.push(`objectFilter=${stats.objectFilter}`);
  if (stats.filterStatus !== "ready") failures.push(`status=${stats.filterStatus}`);
  if (stats.reconstructSource !== EXPECTED_RECONSTRUCT_SOURCE) {
    failures.push(`source=${stats.reconstructSource}`);
  }
  validateShRest(stats, failures);
  if (failures.length > 0) {
    throw new Error(`${assetId} trained initial SH route failed: ${failures.join(" ")}`);
  }
}

function validateDeleteRoute(assetId, stats) {
  const failures = [];
  if (stats.appRoute !== EXPECTED_DELETE_ROUTE) failures.push(`route=${stats.appRoute}`);
  if (stats.appKind !== "commercial") failures.push(`kind=${stats.appKind}`);
  if (stats.appColorRole !== "source-color") failures.push(`colorRole=${stats.appColorRole}`);
  if (stats.appBoundary !== "hard-object-mask-no-reoptimize") {
    failures.push(`boundary=${stats.appBoundary}`);
  }
  if (stats.appQuality !== "packed-sh-mask") failures.push(`quality=${stats.appQuality}`);
  if (stats.renderer !== "spark-splat") failures.push(`renderer=${stats.renderer}`);
  if (stats.objectFilter !== SPARK_OBJECT_FILTER) failures.push(`objectFilter=${stats.objectFilter}`);
  if (stats.filterStatus !== "ready") failures.push(`status=${stats.filterStatus}`);
  if (stats.filterMode !== "ply-reconstruct") failures.push(`mode=${stats.filterMode}`);
  if (stats.maskSource !== EXPECTED_MASK_SOURCE) failures.push(`maskSource=${stats.maskSource}`);
  if (stats.reconstructSource !== EXPECTED_RECONSTRUCT_SOURCE) {
    failures.push(`reconstruct=${stats.reconstructSource}`);
  }
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
  validateShRest(stats, failures);
  if (failures.length > 0) {
    throw new Error(`${assetId} trained delete SH route failed: ${failures.join(" ")}`);
  }
}

function validateShRest(stats, failures) {
  if (stats.shRestSource <= 0) failures.push(`shSource=${stats.shRestSource}`);
  if (stats.shRestPreserved !== stats.shRestSource) {
    failures.push(`shPreserved=${stats.shRestPreserved}/${stats.shRestSource}`);
  }
  if (stats.shRestPreservedFlag !== "true") failures.push(`shFlag=${stats.shRestPreservedFlag}`);
  if (stats.shRestCoefficients <= 0) failures.push(`shCoeffs=${stats.shRestCoefficients}`);
  if (stats.shDegree !== 3) failures.push(`shDegree=${stats.shDegree}`);
}

function initialResultFields(stats) {
  return {
    initialRoute: stats.appRoute,
    initialKind: stats.appKind,
    initialBoundary: stats.appBoundary,
  };
}

function deleteResultFields(stats) {
  return {
    deleteRoute: stats.appRoute,
    deleteKind: stats.appKind,
    deleteBoundary: stats.appBoundary,
    maskSource: stats.maskSource,
    reconstructSource: stats.reconstructSource,
    visibleGaussians: stats.visibleGaussians,
    baseGaussians: stats.baseGaussians,
    objectMaskMode: stats.objectMaskMode,
    objectMaskSize: stats.objectMaskSize,
    objectMaskVisible: stats.objectMaskVisible,
    objectMaskHidden: stats.objectMaskHidden,
    objectMaskUpdates: stats.objectMaskUpdates,
    shRestSource: stats.shRestSource,
    shRestPreserved: stats.shRestPreserved,
    shRestPreservedFlag: stats.shRestPreservedFlag,
    shRestCoefficients: stats.shRestCoefficients,
    shDegree: stats.shDegree,
  };
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
    ["run", "preview", "--", "--port", String(port), "--strictPort"],
    { detached: true, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

function stopServer(child) {
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
