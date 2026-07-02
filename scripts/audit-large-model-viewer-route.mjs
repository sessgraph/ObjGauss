import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

import { MODEL_ARTIFACT_MANIFEST_SCHEMA } from "../src/modelArtifactManifest.js";

const DEFAULT_PORT = 5395;
const ASSET = {
  id: "nerf-lego-trained-near1m-random1300k-local",
  name: "NeRF Lego near-1M 训练输出样例",
  splatPath: "/samples/nerf_lego_trained_near1m_random1300k.splat",
  objectPlyPath: "/samples/nerf_lego_trained_near1m_random1300k_objects.ply",
};

const TINY_OBJECT_PLY = `ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property int object_id
end_header
-0.1 0 0 255 80 80 0.8 0.018 0.018 0.018 1 0 0 0 0
0.1 0 0 255 120 80 0.8 0.018 0.018 0.018 1 0 0 0 0
0 0.1 0 80 180 255 0.8 0.018 0.018 0.018 1 0 0 0 1
0 -0.1 0 80 220 255 0.8 0.018 0.018 0.018 1 0 0 0 1
`;

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const serverMode = normalizeServerMode(args.serverMode ?? args["server-mode"] ?? "dev");
const server = args.url || args.noServer ? null : startServer(port, serverMode);

try {
  await waitForApp(baseUrl);
  const summary = await runAudit(baseUrl);
  console.log(
    `large_model_viewer_route=passed asset=${JSON.stringify(ASSET.id)} ` +
      `filter=${JSON.stringify(summary.filter)} cards=${summary.filteredCardCount} ` +
      `quickSplatRequests=${summary.quickSplatRequests} quickObjectPlyRequests=${summary.quickObjectPlyRequests} ` +
      `editObjectPlyRequests=${summary.editObjectPlyRequests} directObjectPlyRequests=${summary.directObjectPlyRequests} ` +
      `quickArtifactRole=${summary.quickArtifactRole} diagnosticArtifactRole=${summary.diagnosticArtifactRole} ` +
      `screenshot=${summary.screenshotPath}`,
  );
} finally {
  if (server) stopServer(server);
}

async function runAudit(url) {
  const browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const requests = [];
  let objectPlyRequestCount = 0;
  const consoleIssues = [];

  page.on("request", (request) => {
    const pathname = requestPathname(request.url());
    if (pathname.includes("nerf_lego_trained_near1m_random1300k")) {
      requests.push({ url: request.url(), pathname, resourceType: request.resourceType() });
    }
  });
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  await page.route(`**${ASSET.objectPlyPath}`, async (route) => {
    objectPlyRequestCount += 1;
    await route.fulfill({
      status: 200,
      contentType: "application/octet-stream",
      body: TINY_OBJECT_PLY,
    });
  });

  try {
    const filteredUrl = withAssetFilter(url, "trained");
    await page.goto(filteredUrl, { waitUntil: "networkidle" });
    await expectReadyPage(page);
    await page.waitForFunction(() => {
      const app = document.querySelector(".appShell");
      return app?.getAttribute("data-asset-filter") === "trained";
    }, undefined, { timeout: 15000 });
    const filteredCardCount = await page.locator("article.assetCard").count();
    const nearCard = page.locator(`article.assetCard[data-asset-id="${ASSET.id}"]`).first();
    await nearCard.waitFor({ timeout: 15000 });
    const nearPolicy = await nearCard.getAttribute("data-object-ply-policy");
    if (nearPolicy !== "on-demand") {
      throw new Error(`near-1M card did not expose on-demand PLY policy: ${nearPolicy}`);
    }
    const manifestSchema = await nearCard.getAttribute("data-model-manifest-schema");
    const quickRole = await nearCard.getAttribute("data-model-artifact-quick-role");
    const quickBrowserReady = await nearCard.getAttribute("data-model-artifact-quick-browser-ready");
    const objectEditRole = await nearCard.getAttribute("data-model-artifact-object-edit-role");
    const diagnosticRole = await nearCard.getAttribute("data-model-artifact-diagnostic-role");
    const diagnosticBrowserReady = await nearCard.getAttribute("data-model-artifact-diagnostic-browser-ready");
    if (manifestSchema !== MODEL_ARTIFACT_MANIFEST_SCHEMA) {
      throw new Error(`near-1M card did not expose model artifact manifest schema: ${manifestSchema}`);
    }
    if (quickRole !== "quick_splat" || quickBrowserReady !== "true") {
      throw new Error(`near-1M quick artifact is not browser-ready quick_splat: ${quickRole}/${quickBrowserReady}`);
    }
    if (objectEditRole) {
      throw new Error(`near-1M full PLY should not be exposed as browser object_edit: ${objectEditRole}`);
    }
    if (diagnosticRole !== "diagnostic_full" || diagnosticBrowserReady !== "false") {
      throw new Error(`near-1M diagnostic artifact route mismatch: ${diagnosticRole}/${diagnosticBrowserReady}`);
    }

    await nearCard.getByRole("button", { name: "快速查看" }).click();
    await page.waitForFunction(({ assetId, schema, splatPath, objectPlyPath }) => {
      const app = document.querySelector(".appShell");
      return (
        app?.getAttribute("data-active-asset-id") === assetId &&
        app?.getAttribute("data-object-ply-load-state") === "deferred" &&
        app?.getAttribute("data-object-ply-load-mode") === "quick-view" &&
        app?.getAttribute("data-model-manifest-schema") === schema &&
        app?.getAttribute("data-model-artifact-active-role") === "quick_splat" &&
        app?.getAttribute("data-model-artifact-active-tier") === "browser_quick" &&
        app?.getAttribute("data-model-artifact-active-browser-ready") === "true" &&
        app?.getAttribute("data-model-artifact-active-path") === splatPath &&
        app?.getAttribute("data-model-artifact-diagnostic-path") === objectPlyPath
      );
    }, {
      assetId: ASSET.id,
      schema: MODEL_ARTIFACT_MANIFEST_SCHEMA,
      splatPath: ASSET.splatPath,
      objectPlyPath: ASSET.objectPlyPath,
    }, { timeout: 15000 });
    await page.waitForFunction((splatPath) => {
      const app = document.querySelector(".appShell");
      return app?.getAttribute("data-splat-path") === splatPath;
    }, ASSET.splatPath, { timeout: 15000 });
    await waitForRequestCount(requests, ASSET.splatPath, 1);
    const quickObjectPlyRequests = objectPlyRequestCount;
    if (quickObjectPlyRequests !== 0) {
      throw new Error(`quick view requested object PLY ${quickObjectPlyRequests} time(s)`);
    }

    await page.locator(".modeTabs").getByRole("button", { name: "对象编辑" }).click();
    await page.waitForFunction(({ assetId, objectPlyPath }) => {
      const app = document.querySelector(".appShell");
      return (
        app?.getAttribute("data-active-asset-id") === assetId &&
        app?.getAttribute("data-object-ply-load-state") === "loaded" &&
        app?.getAttribute("data-object-ply-load-mode") === "object-ply" &&
        app?.getAttribute("data-model-artifact-active-role") === "diagnostic_full" &&
        app?.getAttribute("data-model-artifact-active-tier") === "diagnostic" &&
        app?.getAttribute("data-model-artifact-active-browser-ready") === "false" &&
        app?.getAttribute("data-model-artifact-active-path") === objectPlyPath
      );
    }, { assetId: ASSET.id, objectPlyPath: ASSET.objectPlyPath }, { timeout: 15000 });
    const editObjectPlyRequests = objectPlyRequestCount - quickObjectPlyRequests;
    if (editObjectPlyRequests !== 1) {
      throw new Error(`object edit requested object PLY ${editObjectPlyRequests} time(s)`);
    }

    await page.goto(filteredUrl, { waitUntil: "networkidle" });
    await expectReadyPage(page);
    const directCard = page.locator(`article.assetCard[data-asset-id="${ASSET.id}"]`).first();
    await directCard.getByRole("button", { name: "加载对象 PLY" }).click();
    await page.waitForFunction(({ assetId, objectPlyPath }) => {
      const app = document.querySelector(".appShell");
      return (
        app?.getAttribute("data-active-asset-id") === assetId &&
        app?.getAttribute("data-object-ply-load-state") === "loaded" &&
        app?.getAttribute("data-object-ply-load-mode") === "object-ply" &&
        app?.getAttribute("data-model-artifact-active-role") === "diagnostic_full" &&
        app?.getAttribute("data-model-artifact-active-browser-ready") === "false" &&
        app?.getAttribute("data-model-artifact-active-path") === objectPlyPath
      );
    }, { assetId: ASSET.id, objectPlyPath: ASSET.objectPlyPath }, { timeout: 15000 });
    const directObjectPlyRequests = objectPlyRequestCount - quickObjectPlyRequests - editObjectPlyRequests;
    if (directObjectPlyRequests !== 1) {
      throw new Error(`direct load requested object PLY ${directObjectPlyRequests} time(s)`);
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

    const screenshotPath = "/tmp/objgauss-large-model-viewer-route.png";
    await page.screenshot({ path: screenshotPath, fullPage: false });
    return {
      filter: "trained",
      filteredCardCount,
      quickSplatRequests: requestCount(requests, ASSET.splatPath),
      quickObjectPlyRequests,
      editObjectPlyRequests,
      directObjectPlyRequests,
      quickArtifactRole: "quick_splat",
      diagnosticArtifactRole: "diagnostic_full",
      screenshotPath,
    };
  } finally {
    await closeBrowserWithTimeout(browser);
  }
}

async function expectReadyPage(page) {
  const title = await page.title();
  if (title !== "ObjGauss 查看器") {
    throw new Error(`unexpected page title: ${title}`);
  }
  await page.getByText("素材库").first().waitFor({ timeout: 15000 });
  await expectNoFrameworkOverlay(page);
}

function withAssetFilter(url, filter) {
  const parsed = new URL(url);
  parsed.searchParams.set("asset-filter", filter);
  return parsed.toString();
}

function requestPathname(url) {
  try {
    return new URL(url).pathname;
  } catch {
    return "";
  }
}

async function waitForRequestCount(requests, pathname, minCount) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    if (requestCount(requests, pathname) >= minCount) return;
    await sleep(100);
  }
  throw new Error(`request not observed: ${pathname}`);
}

function requestCount(requests, pathname) {
  return requests.filter((request) => request.pathname === pathname).length;
}

function launchOptions() {
  const executablePath = firstExisting([
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
  ]);
  return executablePath ? { executablePath } : {};
}

function startServer(port, mode) {
  const script = mode === "preview" ? "preview" : "dev";
  const child = spawn(
    "npm",
    ["run", script, "--", "--port", String(port), "--strictPort"],
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
  const text = await page.locator("body").innerText({ timeout: 15000 });
  const forbidden = ["Failed to compile", "Internal server error", "Vite Error"];
  const found = forbidden.find((entry) => text.includes(entry));
  if (found) {
    throw new Error(`framework overlay visible: ${found}`);
  }
}

async function closeBrowserWithTimeout(browser, timeoutMs = 5000) {
  await Promise.race([
    browser.close(),
    sleep(timeoutMs).then(() => undefined),
  ]);
}

function normalizeServerMode(value) {
  const mode = String(value ?? "dev").trim().toLowerCase();
  if (mode === "dev" || mode === "preview") return mode;
  throw new Error(`unknown --server-mode: ${value}`);
}

function firstExisting(paths) {
  return paths.find((entry) => entry && existsSync(entry));
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
