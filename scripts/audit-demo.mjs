import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5180;
const DEFAULT_ASSETS = [
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

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = args.url ?? `http://127.0.0.1:${port}/`;
const assets = args.asset ? DEFAULT_ASSETS.filter((asset) => asset.id === args.asset) : DEFAULT_ASSETS;

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
        `splatPixels=${result.splatPixels} editPixels=${result.editPixels} ` +
        `visibleAfterIsolate=${result.visibleAfterIsolate} ` +
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
      const splatPixels = await nonBackgroundPixels(page);
      if (splatPixels <= 0) {
        throw new Error(`${asset.id} splat canvas appears blank: ${splatPixels}`);
      }

      await page.getByLabel("渲染模式").selectOption("clustered");
      await page.waitForTimeout(700);
      const editPixels = await nonBackgroundPixels(page);
      if (editPixels <= 0) {
        throw new Error(`${asset.id} point-edit canvas appears blank: ${editPixels}`);
      }
      await page.locator(".objectRow").first().click();
      await page.getByRole("button", { name: "只看所选" }).click();
      await page.waitForTimeout(300);
      const visibleAfterIsolate = await labeledValue(page, "可见");
      await page.getByRole("button", { name: "预览删除" }).click();
      await page.waitForTimeout(300);
      const deletedObjects = await labeledValue(page, "已删除对象");
      if (deletedObjects !== "1") {
        throw new Error(`${asset.id} delete preview did not update: ${deletedObjects}`);
      }
      const screenshotPath = `/tmp/objgauss-audit-${asset.id}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      results.push({
        assetId: asset.id,
        title,
        splatPixels,
        editPixels,
        visibleAfterIsolate,
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
  return page.locator("canvas").first().evaluate((canvas) => {
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) return -1;
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
    return count;
  });
}

async function labeledValue(page, label) {
  const row = page.locator(".metric, .stateRow").filter({ hasText: label }).first();
  return row.locator("strong").innerText();
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
