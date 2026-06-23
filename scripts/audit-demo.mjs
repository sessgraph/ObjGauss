import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

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

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = args.url ?? `http://127.0.0.1:${port}/`;
const assets = args.asset ? KNOWN_ASSETS.filter((asset) => asset.id === args.asset) : DEFAULT_ASSETS;

if (assets.length === 0) {
  throw new Error(`unknown asset id: ${args.asset}`);
}

const server = args.url || args.noServer ? null : startDevServer(port);
try {
  await waitForApp(baseUrl);
  const results = await runAudit(baseUrl, assets);
  for (const result of results) {
    console.log(
        `asset=${result.assetId} title=${JSON.stringify(result.title)} ` +
        `splatPixels=${result.splatPixels} editRenderer=${JSON.stringify(result.editRenderer)} ` +
        `objectFilter=${JSON.stringify(result.objectFilter)} ` +
        `editPixels=${result.editPixels} ` +
        `canvasSelectedObject=${result.canvasSelectedObject} ` +
        `visibleAfterIsolate=${result.visibleAfterIsolate} ` +
        `visibleAfterDelete=${result.visibleAfterDelete} ` +
        `renderModeAfterDelete=${JSON.stringify(result.renderModeAfterDelete)} ` +
        `deletedObjects=${result.deletedObjects} screenshot=${result.screenshotPath}`,
    );
  }
  console.log(`browser_audit=passed assets=${results.length} url=${baseUrl}`);
} finally {
  if (server) {
    stopDevServer(server);
  }
}

async function runAudit(url, assetsToCheck) {
  const browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
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
      await page.goto(url, { waitUntil: "networkidle" });
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

      await page.getByLabel("渲染模式").selectOption("clustered");
      const editRenderer = await labeledValue(page, "渲染器");
      if (editRenderer !== "Gaussian OIT 编辑") {
        throw new Error(`${asset.id} did not enter Gaussian OIT edit renderer: ${editRenderer}`);
      }
      const objectFilter = await page.locator(".viewport").first().getAttribute("data-object-filter");
      if (objectFilter !== "gpu-object-state-texture") {
        throw new Error(`${asset.id} did not expose GPU object-state filtering: ${objectFilter}`);
      }
      const editPixels = await waitForNonBackgroundPixels(page);
      if (editPixels <= 0) {
        throw new Error(`${asset.id} point-edit canvas appears blank: ${editPixels}`);
      }
      const canvasSelectedObject = await selectObjectFromCanvas(page, asset.id);
      await page.getByRole("button", { name: "只看所选" }).click();
      await page.waitForTimeout(300);
      const visibleAfterIsolate = await labeledValue(page, "可见");
      await page.getByRole("button", { name: "预览删除" }).click();
      await page.waitForTimeout(300);
      const deletedObjects = await labeledValue(page, "已删除对象");
      const visibleAfterDelete = await labeledValue(page, "可见");
      const renderModeAfterDelete = await labeledValue(page, "模式");
      if (deletedObjects !== "1") {
        throw new Error(`${asset.id} delete preview did not update: ${deletedObjects}`);
      }
      if (numericValue(visibleAfterDelete) <= 0) {
        throw new Error(`${asset.id} delete preview did not show remaining scene`);
      }
      if (renderModeAfterDelete !== "自身颜色") {
        throw new Error(`${asset.id} delete preview did not restore original colors`);
      }
      const screenshotPath = `/tmp/objgauss-audit-${asset.id}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      results.push({
        assetId: asset.id,
        title,
        splatPixels,
        editPixels,
        editRenderer,
        objectFilter,
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
        !issue.includes("GPU stall due to ReadPixels"),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }
    return results;
  } finally {
    await browser.close();
  }
}

function launchOptions() {
  const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ?? firstExisting([
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ]);
  return executablePath ? { executablePath, args: ["--no-sandbox"] } : {};
}

function firstExisting(paths) {
  return paths.find((path) => existsSync(path));
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
