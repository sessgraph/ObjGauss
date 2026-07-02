import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5395;
const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const server = args.url || args.noServer ? null : startServer(port);

try {
  await waitForApp(baseUrl);
  const summary = await auditWorld(baseUrl);
  console.log(
    [
      "world_viewer=passed",
      `models=${summary.modelCount}`,
      `objects=${summary.objectCount}`,
      `draggableObjects=${summary.draggableObjectCount}`,
      `selectedModel=${JSON.stringify(summary.selectedModelId)}`,
      `selectedObject=${JSON.stringify(summary.selectedObjectId)}`,
      `debugOs=${summary.debugOs}`,
      `assignmentSlots=${summary.assignmentSlots}`,
      `selectedGaussian=${JSON.stringify(summary.selectedGaussian)}`,
      `sidebars=${summary.sidebars}`,
      `screenshot=${summary.screenshotPath}`,
    ].join(" "),
  );
} finally {
  if (server) stopServer(server);
}

async function auditWorld(url) {
  const browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    const app = page.locator(".worldShell");
    await app.waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const world = window.__OBJGAUSS_WORLD__;
      return world?.draggableObjectCount > world?.modelCount && world?.objectSelections?.length > world?.modelCount;
    }, undefined, {
      timeout: 15000,
    });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      return (
        shell?.getAttribute("data-sidebars") === "none" &&
        shell?.getAttribute("data-frosted-ui") === "enabled" &&
        shell?.getAttribute("data-app-mode") === "vr-three-world" &&
        shell?.getAttribute("data-debug-os") === "object-state"
      );
    }, undefined, { timeout: 15000 });

    const sidebars = await page.locator("aside, .leftRail, .rightRail").count();
    if (sidebars !== 0) {
      throw new Error(`sidebars should not exist in VR world shell, found ${sidebars}`);
    }
    const canvas = page.locator("canvas[data-three-world-canvas='true']");
    await canvas.waitFor({ timeout: 15000 });
    await page.locator(".glassHud.floatingInspector").waitFor({ timeout: 15000 });
    await page.locator("[data-object-debug-panel='true']").waitFor({ timeout: 15000 });
    await page.locator("[data-assignment-heatmap='true']").waitFor({ timeout: 15000 });
    const pills = await page.locator(".modelPill").count();
    if (pills < 5) {
      throw new Error(`expected at least 5 model pills, found ${pills}`);
    }
    await page.locator(".modelPill").nth(2).click();
    await page.waitForFunction(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      return world?.selectedModelId === shell?.getAttribute("data-selected-model");
    }, undefined, { timeout: 15000 });
    const objectSelection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const selection = world?.objectSelections?.find(
        (entry) => entry.modelId === world.selectedModelId,
      ) ?? world?.objectSelections?.[0];
      return {
        ok: world?.selectObjectForAudit?.(selection?.selectionId) ?? false,
        selectionId: selection?.selectionId ?? null,
      };
    });
    if (!objectSelection.ok) {
      throw new Error("expected audit handle to select a per-object render target");
    }
    await page.waitForFunction((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      return (
        world?.selectedId === selectionId &&
        shell?.getAttribute("data-selected-target") === selectionId &&
        shell?.getAttribute("data-selected-object") !== ""
      );
    }, objectSelection.selectionId, { timeout: 15000 });
    const gaussianSelection = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        protocol: world?.debugProtocol ?? null,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, objectSelection.selectionId);
    if (!gaussianSelection.ok) {
      throw new Error("expected audit handle to select a Gaussian assignment probe");
    }
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      return (
        shell?.getAttribute("data-selected-gaussian") !== "" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) > 0
      );
    }, undefined, { timeout: 15000 });
    if (gaussianSelection.protocol !== "object-state-debug-os-v1") {
      throw new Error(`unexpected debug protocol: ${gaussianSelection.protocol}`);
    }
    if (gaussianSelection.assignmentSource !== "derived_from_object_id") {
      throw new Error(`unexpected assignment source: ${gaussianSelection.assignmentSource}`);
    }
    const visibilityToggle = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const selection = world?.objectSelections?.[0];
      const before = world?.visibleObjectCount ?? null;
      const visibleAfterToggle = world?.toggleObjectVisibilityForAudit?.(selection?.selectionId) ?? null;
      const after = window.__OBJGAUSS_WORLD__?.visibleObjectCount ?? null;
      return { before, after, visibleAfterToggle };
    });
    if (!(visibilityToggle.after < visibilityToggle.before)) {
      throw new Error(`expected object visibility toggle to reduce visible count: ${JSON.stringify(visibilityToggle)}`);
    }

    const relevantIssues = consoleIssues.filter(
      (issue) =>
        !issue.includes("THREE.WebGLRenderer") &&
        !issue.includes("Multiple instances of Three.js") &&
        !issue.includes("GPU stall due to ReadPixels") &&
        !issue.includes("WebGL warning"),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }

    const screenshotPath = "/tmp/objgauss-world-viewer.png";
    await page.screenshot({ path: screenshotPath, fullPage: false });
    const world = await page.evaluate(() => {
      const handle = window.__OBJGAUSS_WORLD__;
      return {
        modelCount: handle.modelCount,
        objectCount: handle.objectCount,
        draggableObjectCount: handle.draggableObjectCount,
        selectedId: handle.selectedId,
        selectedModelId: handle.selectedModelId,
        selectedObjectId: handle.selectedObjectId,
        debugMode: handle.debugMode,
        debugProtocol: handle.debugProtocol,
      };
    });
    const assignmentSlots = await page
      .locator("[data-assignment-heatmap='true']")
      .evaluate((node) => Number(node.getAttribute("data-assignment-slots") ?? 0));
    const selectedGaussian = await page
      .locator(".worldShell")
      .evaluate((node) => node.getAttribute("data-selected-gaussian"));
    return {
      modelCount: world.modelCount,
      objectCount: world.objectCount,
      draggableObjectCount: world.draggableObjectCount,
      selectedModelId: world.selectedModelId,
      selectedObjectId: world.selectedObjectId,
      debugOs: world.debugProtocol,
      assignmentSlots,
      selectedGaussian,
      sidebars,
      screenshotPath,
    };
  } finally {
    await closeBrowserWithTimeout(browser);
  }
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

function startServer(port) {
  const child = spawn(
    "npm",
    ["run", "dev", "--", "--port", String(port), "--strictPort"],
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

async function closeBrowserWithTimeout(browser, timeoutMs = 5000) {
  await Promise.race([
    browser.close(),
    sleep(timeoutMs).then(() => undefined),
  ]);
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
