import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5395;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-webgpu-frame-pacing";
const DEFAULT_ASSETS = ["nerf-lego-alpha-closure-local", "plush-semantic-closure-local"];
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
    id: "plush-v1-closure-local",
    name: "ObjGauss v1 闭环样例",
    fileName: "plush_v1_objects.ply",
  },
  {
    id: "nerf-lego-trained-output-local",
    name: "NeRF Lego 训练输出样例",
    fileName: "nerf_lego_trained_objects.ply",
  },
];
const MODE = "webgpu-frame-pacing-smoke-v1";
const LONG_FRAME_MS = 50;

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const assets = selectAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const headed = !flagEnabled(args.headless);
const shouldStartServer = !(args.url || args.noServer || args["no-server"]);
const browserChannel = optionalString(args.browserChannel ?? args["browser-channel"]);
const executablePath = optionalString(args.executablePath ?? args["executable-path"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const frameCount = positiveFiniteNumber(args.frameCount ?? args["frame-count"], 75);
const maxMeanFrameMs = positiveFiniteNumber(args.maxMeanFrameMs ?? args["max-mean-frame-ms"], 60);
const maxP95FrameMs = positiveFiniteNumber(args.maxP95FrameMs ?? args["max-p95-frame-ms"], 120);
const maxLongFrameRatio = positiveFiniteNumber(
  args.maxLongFrameRatio ?? args["max-long-frame-ratio"],
  0.35,
);
const minApproxFps = positiveFiniteNumber(args.minApproxFps ?? args["min-approx-fps"], 12);
const minLargeSceneGaussians = positiveFiniteNumber(
  args.minLargeSceneGaussians ?? args["min-large-scene-gaussians"],
  250000,
);

let server = null;
let browser = null;

try {
  if (shouldStartServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before frame pacing audit");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  mkdirSync(outputDir, { recursive: true });
  browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  const rows = [];
  for (const asset of assets) {
    console.log(`webgpu_frame_pacing_asset_start asset=${JSON.stringify(asset.id)}`);
    const row = await auditAsset(page, asset);
    rows.push(row);
    console.log(
      `webgpu_frame_pacing_asset=${row.passed ? "passed" : "failed"} ` +
        `asset=${JSON.stringify(row.asset)} gaussians=${row.packedGaussians} ` +
        `phases=${row.phases
          .map((phase) => `${phase.phase}:${phase.meanFrameMs}/${phase.p95FrameMs}/${phase.approxFps}`)
          .join(",")} screenshot=${JSON.stringify(row.screenshotPath)}`,
    );
    if (!row.passed && !allowFailures) break;
  }

  const checks = buildSuiteChecks(rows);
  const passed = rows.length === assets.length && rows.every((row) => row.passed) && checks.every((check) => check.passed);
  const summary = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    outputDir,
    url: baseUrl,
    assets: assets.map((asset) => asset.id),
    webGpuFlags,
    headed,
    budgets: {
      frameCount,
      longFrameMs: LONG_FRAME_MS,
      maxMeanFrameMs,
      maxP95FrameMs,
      maxLongFrameRatio,
      minApproxFps,
      minLargeSceneGaussians,
    },
    passed,
    aggregate: summarizeRows(rows),
    consoleIssues,
    checks,
    rows,
    interpretation:
      "This is a headed browser rAF frame-pacing smoke for current real scenes. It is not a sustained renderer FPS benchmark and not a 1M browser runtime proof.",
  };
  writeReport(summary);
  console.log(
    `webgpu_frame_pacing=${passed ? "passed" : "failed"} scenes=${rows.length}/${assets.length} ` +
      `largestGaussians=${summary.aggregate.largestGaussians} minApproxFps=${summary.aggregate.minApproxFps} ` +
      `maxMeanFrameMs=${summary.aggregate.maxMeanFrameMs} maxP95FrameMs=${summary.aggregate.maxP95FrameMs} ` +
      `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
  );
  if (!passed && !allowFailures) process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  if (server) stopPreviewServer(server);
}

async function auditAsset(page, asset) {
  const url = urlWithProbe(baseUrl);
  await page.goto(url, { waitUntil: "networkidle" });
  await expectReadyPage(page);
  await loadAsset(page, asset);
  await page.locator(".modeTabs").getByRole("button", { name: "对象编辑" }).click();
  await waitForEditViewportReady(page);
  await page.getByLabel("渲染模式").selectOption("clustered");
  await waitForWebGpuViewport(page);

  const initialTelemetry = await readWebGpuTelemetry(page);
  const idle = await sampleFramePacing(page, "idle");

  const selectedObject = await selectObjectFromCanvas(page, asset.id);
  const isolateStarted = performance.now();
  await page.getByRole("button", { name: "只看所选" }).click();
  await waitForWebGpuStorageUpdate(page, {
    previousChecksum: initialTelemetry.storageChecksum,
    allowFullUpload: false,
  });
  const isolateWallMs = roundMetric(performance.now() - isolateStarted);
  const isolateTelemetry = await readWebGpuTelemetry(page);
  const afterIsolate = await sampleFramePacing(page, "after-isolate");

  const deleteStarted = performance.now();
  await page.getByRole("button", { name: "预览删除" }).click();
  await waitForWebGpuStorageUpdate(page, {
    previousChecksum: isolateTelemetry.storageChecksum,
    allowFullUpload: true,
  });
  const deleteWallMs = roundMetric(performance.now() - deleteStarted);
  const deleteTelemetry = await readWebGpuTelemetry(page);
  const afterDelete = await sampleFramePacing(page, "after-delete");
  const screenshotPath = path.join(outputDir, `${asset.id}-webgpu-frame-pacing.png`);
  await page.screenshot({ path: screenshotPath, fullPage: false });

  const phases = [idle, afterIsolate, afterDelete];
  const checks = [
    check("renderer", deleteTelemetry.renderer, "webgpu-tile", deleteTelemetry.renderer === "webgpu-tile"),
    check(
      "object-filter",
      deleteTelemetry.objectFilter,
      "gpu-object-state-buffer",
      deleteTelemetry.objectFilter === "gpu-object-state-buffer",
    ),
    check("first-frame", initialTelemetry.firstFrameStatus, "rendered", initialTelemetry.firstFrameStatus === "rendered"),
    check("selected-object", selectedObject, "object id", isObjectId(selectedObject)),
    check("packed-gaussians", initialTelemetry.packedGaussians, ">0", initialTelemetry.packedGaussians > 0),
    check("tile-references", initialTelemetry.tileReferences, ">0", initialTelemetry.tileReferences > 0),
    check("tile-overflow", initialTelemetry.tileOverflowCount, 0, initialTelemetry.tileOverflowCount === 0),
    check("isolate-update-mode", isolateTelemetry.storageUpdateMode, "object-state-only", isolateTelemetry.storageUpdateMode === "object-state-only"),
    check("delete-update-mode", deleteTelemetry.storageUpdateMode, "object-state-only or full-upload", ["object-state-only", "full-upload"].includes(deleteTelemetry.storageUpdateMode)),
    ...phases.flatMap((phase) => phaseChecks(phase)),
  ];
  return {
    asset: asset.id,
    passed: checks.every((entry) => entry.passed),
    checks,
    selectedObject,
    packedGaussians: initialTelemetry.packedGaussians,
    tileReferences: initialTelemetry.tileReferences,
    firstFramePixels: initialTelemetry.firstFramePixels,
    initial: initialTelemetry,
    isolate: { ...isolateTelemetry, wallMs: isolateWallMs },
    delete: { ...deleteTelemetry, wallMs: deleteWallMs },
    phases,
    screenshotPath,
  };
}

async function expectReadyPage(page) {
  const title = await page.title();
  if (title !== "ObjGauss 查看器") {
    throw new Error(`unexpected page title: ${title}`);
  }
  await page.getByText("素材库").first().waitFor({ timeout: 15000 });
  const bodyText = await page.locator("body").innerText();
  for (const signal of ["Vite Error", "Internal server error", "Failed to resolve import"]) {
    if (bodyText.includes(signal)) throw new Error(`framework overlay detected: ${signal}`);
  }
}

async function loadAsset(page, asset) {
  const card = page.locator("article.assetCard").filter({ hasText: asset.name }).first();
  await card.getByRole("button", { name: "加载" }).click();
  await page.waitForFunction(
    (fileName) => document.body.innerText.includes(fileName),
    asset.fileName,
    { timeout: 15000 },
  );
  await page.waitForTimeout(1500);
}

async function waitForEditViewportReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    const renderer = viewport?.getAttribute("data-renderer");
    const webGpuStatus = viewport?.getAttribute("data-webgpu-status");
    return (renderer === "gaussian-oit" || renderer === "webgpu-tile") && webGpuStatus !== "pending";
  }, undefined, { timeout: 15000 });
}

async function waitForWebGpuViewport(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    return (
      viewport?.getAttribute("data-renderer") === "webgpu-tile" &&
      viewport?.getAttribute("data-object-filter") === "gpu-object-state-buffer" &&
      viewport?.getAttribute("data-webgpu-first-frame-status") === "rendered" &&
      ["done", "submitted"].includes(viewport?.getAttribute("data-webgpu-queue-status") ?? "")
    );
  }, undefined, { timeout: 20000 });
  await page.waitForTimeout(250);
}

async function waitForWebGpuStorageUpdate(page, { previousChecksum, allowFullUpload }, timeoutMs = 20000) {
  await page.waitForFunction(
    ({ previous, allowFull }) => {
      const viewport = document.querySelector(".viewport");
      if (viewport?.getAttribute("data-renderer") !== "webgpu-tile") return false;
      const checksum = viewport.getAttribute("data-webgpu-storage-checksum") ?? "";
      const status = viewport.getAttribute("data-webgpu-storage-status");
      const mode = viewport.getAttribute("data-webgpu-storage-update-mode");
      const updateMs = Number(viewport.getAttribute("data-webgpu-storage-update-ms") ?? "NaN");
      const submitMs = Number(viewport.getAttribute("data-webgpu-frame-submit-ms") ?? "NaN");
      const validMode =
        (status === "object-state-updated" && mode === "object-state-only") ||
        (allowFull && status === "uploaded" && mode === "full-upload");
      return (
        validMode &&
        /^[0-9a-f]{8}$/.test(checksum) &&
        checksum !== previous &&
        Number.isFinite(updateMs) &&
        updateMs >= 0 &&
        Number.isFinite(submitMs) &&
        submitMs >= 0
      );
    },
    { previous: previousChecksum, allowFull: allowFullUpload },
    { timeout: timeoutMs },
  );
  await page.waitForTimeout(200);
}

async function readWebGpuTelemetry(page) {
  return page.locator(".viewport").first().evaluate((viewport) => {
    const numberAttr = (name) => {
      const parsed = Number(viewport.getAttribute(name) ?? "0");
      return Number.isFinite(parsed) ? parsed : 0;
    };
    return {
      renderer: viewport.getAttribute("data-renderer") ?? "",
      objectFilter: viewport.getAttribute("data-object-filter") ?? "",
      firstFrameStatus: viewport.getAttribute("data-webgpu-first-frame-status") ?? "",
      firstFramePixels: numberAttr("data-webgpu-first-frame-pixels"),
      packedGaussians: numberAttr("data-webgpu-packed-gaussians"),
      tileReferences: numberAttr("data-webgpu-tile-reference-count"),
      tileOverflowCount: numberAttr("data-webgpu-tile-overflow-count"),
      storageStatus: viewport.getAttribute("data-webgpu-storage-status") ?? "",
      storageUpdateMode: viewport.getAttribute("data-webgpu-storage-update-mode") ?? "",
      storageChecksum: viewport.getAttribute("data-webgpu-storage-checksum") ?? "",
      storageUpdateMs: numberAttr("data-webgpu-storage-update-ms"),
      frameSubmitMs: numberAttr("data-webgpu-frame-submit-ms"),
      queueDoneMs: numberAttr("data-webgpu-queue-done-ms"),
      queueStatus: viewport.getAttribute("data-webgpu-queue-status") ?? "",
      deviceLostStatus: viewport.getAttribute("data-webgpu-device-lost-status") ?? "",
    };
  });
}

async function sampleFramePacing(page, phase) {
  const metrics = await page.evaluate(
    ({ samples, longFrameMs }) =>
      new Promise((resolve) => {
        const intervals = [];
        let previous = null;
        const tick = (timestamp) => {
          if (previous !== null) intervals.push(timestamp - previous);
          previous = timestamp;
          if (intervals.length >= samples) {
            const sorted = [...intervals].sort((a, b) => a - b);
            const sum = intervals.reduce((total, value) => total + value, 0);
            const mean = sum / intervals.length;
            const percentile = (ratio) => {
              const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1));
              return sorted[index];
            };
            const longFrames = intervals.filter((value) => value > longFrameMs).length;
            resolve({
              samples: intervals.length,
              meanFrameMs: mean,
              p50FrameMs: percentile(0.5),
              p95FrameMs: percentile(0.95),
              maxFrameMs: sorted[sorted.length - 1] ?? 0,
              longFrameCount: longFrames,
              longFrameRatio: longFrames / intervals.length,
              approxFps: mean > 0 ? 1000 / mean : 0,
            });
            return;
          }
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }),
    { samples: Math.round(frameCount), longFrameMs: LONG_FRAME_MS },
  );
  return {
    phase,
    samples: metrics.samples,
    meanFrameMs: roundMetric(metrics.meanFrameMs),
    p50FrameMs: roundMetric(metrics.p50FrameMs),
    p95FrameMs: roundMetric(metrics.p95FrameMs),
    maxFrameMs: roundMetric(metrics.maxFrameMs),
    longFrameCount: metrics.longFrameCount,
    longFrameRatio: roundMetric(metrics.longFrameRatio),
    approxFps: roundMetric(metrics.approxFps),
  };
}

function phaseChecks(phase) {
  return [
    check(`${phase.phase}:samples`, phase.samples, `>= ${frameCount}`, phase.samples >= frameCount),
    check(`${phase.phase}:mean-frame-ms`, phase.meanFrameMs, `<= ${maxMeanFrameMs}`, phase.meanFrameMs <= maxMeanFrameMs),
    check(`${phase.phase}:p95-frame-ms`, phase.p95FrameMs, `<= ${maxP95FrameMs}`, phase.p95FrameMs <= maxP95FrameMs),
    check(`${phase.phase}:long-frame-ratio`, phase.longFrameRatio, `<= ${maxLongFrameRatio}`, phase.longFrameRatio <= maxLongFrameRatio),
    check(`${phase.phase}:approx-fps`, phase.approxFps, `>= ${minApproxFps}`, phase.approxFps >= minApproxFps),
  ];
}

async function selectObjectFromCanvas(page, assetId, previousSelected = "无") {
  const canvas = page.locator(".viewport canvas").first();
  const box = await canvas.boundingBox();
  if (!box) throw new Error(`${assetId} WebGPU canvas is missing`);
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
    const x = box.x + box.width * xRatio;
    const y = box.y + box.height * yRatio;
    await page.mouse.move(x, y);
    await page.waitForTimeout(180);
    await page.mouse.click(x, y);
    await page.waitForTimeout(250);
    const selected = await selectedObjectValue(page);
    if (selected !== "无" && selected !== previousSelected) return selected;
  }
  throw new Error(`${assetId} canvas selection did not choose an object`);
}

async function selectedObjectValue(page) {
  const status = await page.locator(".statusBar").innerText();
  const match = status.match(/所选：([^\n]+)/);
  return match?.[1] ?? "无";
}

function buildSuiteChecks(rows) {
  const largeRows = rows.filter((row) => row.packedGaussians >= minLargeSceneGaussians);
  return [
    check("asset-count", rows.length, assets.length, rows.length === assets.length),
    check(
      "large-scene-coverage",
      largeRows.map((row) => `${row.asset}:${row.packedGaussians}`).join(","),
      `>= ${minLargeSceneGaussians}`,
      largeRows.length > 0,
    ),
    ...rows.flatMap((row) => row.checks.map((entry) => ({ asset: row.asset, ...entry }))),
  ];
}

function summarizeRows(rows) {
  const phases = rows.flatMap((row) => row.phases);
  return {
    scenes: rows.length,
    largestGaussians: maxNumeric(rows.map((row) => row.packedGaussians)),
    maxTileReferences: maxNumeric(rows.map((row) => row.tileReferences)),
    maxMeanFrameMs: maxNumeric(phases.map((phase) => phase.meanFrameMs)),
    maxP95FrameMs: maxNumeric(phases.map((phase) => phase.p95FrameMs)),
    maxLongFrameRatio: maxNumeric(phases.map((phase) => phase.longFrameRatio)),
    minApproxFps: minNumeric(phases.map((phase) => phase.approxFps)),
  };
}

function writeReport(summary) {
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary), "utf8");
}

function renderMarkdown(summary) {
  const lines = [
    "# WebGPU Frame Pacing Smoke",
    "",
    `- Mode: \`${summary.mode}\``,
    `- Status: \`${summary.passed ? "passed" : "failed"}\``,
    `- Generated: \`${summary.generatedAt}\``,
    `- URL: \`${summary.url}\``,
    `- Headed: \`${summary.headed ? "true" : "false"}\``,
    "",
    summary.interpretation,
    "",
    "| Asset | Passed | Gaussians | Phase | Mean ms | P95 ms | Long frame ratio | Approx FPS | Screenshot |",
    "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
  ];
  for (const row of summary.rows) {
    for (const phase of row.phases) {
      lines.push(
        `| ${escapeMarkdown(row.asset)} | ${row.passed ? "yes" : "no"} | ${row.packedGaussians} | ${phase.phase} | ${phase.meanFrameMs} | ${phase.p95FrameMs} | ${phase.longFrameRatio} | ${phase.approxFps} | ${escapeMarkdown(row.screenshotPath)} |`,
      );
    }
  }
  const failed = summary.checks.filter((entry) => !entry.passed);
  lines.push("", "## Failed Checks", "");
  if (failed.length === 0) {
    lines.push("None.");
  } else {
    lines.push("| Asset | Check | Actual | Expected |", "| --- | --- | --- | --- |");
    for (const entry of failed) {
      lines.push(
        `| ${escapeMarkdown(entry.asset ?? "suite")} | ${escapeMarkdown(entry.name)} | ${escapeMarkdown(entry.actual)} | ${escapeMarkdown(entry.expected)} |`,
      );
    }
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function urlWithProbe(url) {
  const parsed = new URL(url);
  parsed.searchParams.set("webgpu-probe", "full");
  parsed.searchParams.set("spark-filtered-edit", "off");
  return parsed.toString();
}

function launchOptions() {
  const launch = {
    headless: !headed,
    args: ["--no-sandbox", ...webGpuLaunchArgs(webGpuFlags)],
  };
  if (browserChannel) launch.channel = browserChannel;
  const browserPath =
    executablePath ||
    firstExisting([
      "/usr/bin/google-chrome",
      "/usr/bin/google-chrome-stable",
      "/usr/bin/chromium",
      "/usr/bin/chromium-browser",
    ]);
  if (!browserChannel && browserPath) launch.executablePath = browserPath;
  return launch;
}

function webGpuLaunchArgs(mode) {
  if (mode === "none" || mode === "false" || mode === "") return [];
  if (mode === "unsafe") return ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"];
  if (mode === "vulkan") {
    return [
      "--enable-unsafe-webgpu",
      "--ignore-gpu-blocklist",
      "--enable-features=Vulkan,DefaultANGLEVulkan,WebGPUDeveloperFeatures",
      "--use-angle=vulkan",
    ];
  }
  throw new Error(`unsupported WebGPU launch flags mode: ${mode}`);
}

function startPreviewServer(portToUse) {
  const child = spawn(
    "npm",
    ["run", "preview", "--", "--port", String(portToUse), "--strictPort"],
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

function selectAssets(value) {
  const requested = String(value ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (requested.length === 0) throw new Error("at least one asset is required");
  const byId = new Map(KNOWN_ASSETS.map((asset) => [asset.id, asset]));
  const unknown = requested.filter((id) => !byId.has(id));
  if (unknown.length > 0) throw new Error(`unknown asset id(s): ${unknown.join(",")}`);
  return requested.map((id) => byId.get(id));
}

function check(name, actual, expected, passed) {
  return { name, actual, expected, passed: Boolean(passed) };
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const entry = argv[index];
    if (!entry.startsWith("--")) continue;
    const key = entry.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return "";
  const text = String(value).trim();
  return text || "";
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function positiveFiniteNumber(value, fallback) {
  if (value === undefined || value === null || value === true || value === false) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function roundMetric(value) {
  return Number(Number(value).toFixed(3));
}

function maxNumeric(values) {
  const numeric = values.filter((value) => Number.isFinite(value));
  return numeric.length > 0 ? Math.max(...numeric) : 0;
}

function minNumeric(values) {
  const numeric = values.filter((value) => Number.isFinite(value));
  return numeric.length > 0 ? Math.min(...numeric) : 0;
}

function firstExisting(paths) {
  return paths.find((entry) => existsSync(entry));
}

function isObjectId(value) {
  const parsed = Number(String(value ?? ""));
  return Number.isInteger(parsed) && parsed >= 0;
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
