import { spawn } from "node:child_process";
import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5395;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-webgpu-synthetic-1m-runtime";
const DEFAULT_GAUSSIANS = 1_000_000;
const DEFAULT_OBJECTS = 256;
const MODE = "webgpu-synthetic-upload-1m-runtime-v1";
const LONG_FRAME_MS = 50;

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const gaussianCount = positiveInteger(args.gaussians ?? args.count, DEFAULT_GAUSSIANS);
const objectCount = positiveInteger(args.objects ?? args["object-count"], DEFAULT_OBJECTS);
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const headed = !flagEnabled(args.headless);
const shouldStartServer = !(args.url || args.noServer || args["no-server"]);
const browserChannel = optionalString(args.browserChannel ?? args["browser-channel"]);
const executablePath = optionalString(args.executablePath ?? args["executable-path"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const frameCount = positiveFiniteNumber(args.frameCount ?? args["frame-count"], 45);
const maxMeanFrameMs = positiveFiniteNumber(args.maxMeanFrameMs ?? args["max-mean-frame-ms"], 100);
const maxP95FrameMs = positiveFiniteNumber(args.maxP95FrameMs ?? args["max-p95-frame-ms"], 220);
const maxLongFrameRatio = positiveFiniteNumber(
  args.maxLongFrameRatio ?? args["max-long-frame-ratio"],
  0.65,
);
const minApproxFps = positiveFiniteNumber(args.minApproxFps ?? args["min-approx-fps"], 8);
const maxInitialQueueDoneMs = positiveFiniteNumber(
  args.maxInitialQueueDoneMs ?? args["max-initial-queue-done-ms"],
  12_000,
);
const maxObjectStateUpdateMs = positiveFiniteNumber(
  args.maxObjectStateUpdateMs ?? args["max-object-state-update-ms"],
  1_500,
);
const maxFullUploadUpdateMs = positiveFiniteNumber(
  args.maxFullUploadUpdateMs ?? args["max-full-upload-update-ms"],
  4_000,
);

let server = null;
let browser = null;

try {
  mkdirSync(outputDir, { recursive: true });
  const plyPath = path.join(outputDir, `synthetic_${gaussianCount}_objects_${objectCount}.ply`);
  const generated = generateSyntheticPly({ outputPath: plyPath, gaussianCount, objectCount });

  if (shouldStartServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before synthetic 1M runtime audit");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  const row = await auditSyntheticUpload(page, generated);
  const checks = buildSuiteChecks(row);
  const passed = row.passed && checks.every((checkEntry) => checkEntry.passed);
  const summary = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    outputDir,
    url: baseUrl,
    webGpuFlags,
    headed,
    generated,
    budgets: {
      frameCount,
      longFrameMs: LONG_FRAME_MS,
      maxMeanFrameMs,
      maxP95FrameMs,
      maxLongFrameRatio,
      minApproxFps,
      maxInitialQueueDoneMs,
      maxObjectStateUpdateMs,
      maxFullUploadUpdateMs,
    },
    passed,
    proof: {
      browserRuntime1m:
        row.packedGaussians >= 1_000_000 && row.initial.firstFrameStatus === "rendered"
          ? "proven-synthetic-upload"
          : "not-proven",
      realTrainedScene1m: "not-proven",
      sustainedFpsSla: "not-proven",
    },
    aggregate: summarizeRow(row),
    consoleIssues,
    checks,
    row,
    interpretation:
      `This audit uploads a synthetic ${generated.gaussianCount} Gaussian PLY through the real UI and exercises the headed browser WebGPU Tile C-path. Only runs at or above 1M Gaussians count as synthetic 1M browser-runtime proof. This is not evidence for a trained 1M scene, paper-quality image fidelity, or sustained FPS SLA.`,
  };
  writeReport(summary);
  console.log(
    `webgpu_synthetic_1m_runtime=${passed ? "passed" : "failed"} ` +
      `gaussians=${row.packedGaussians} objects=${generated.objectCount} ` +
      `tileReferences=${row.tileReferences} minApproxFps=${summary.aggregate.minApproxFps} ` +
      `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
  );
  if (!passed && !allowFailures) process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  if (server) stopPreviewServer(server);
}

async function auditSyntheticUpload(page, generated) {
  const url = urlWithProbe(baseUrl);
  await page.goto(url, { waitUntil: "networkidle" });
  await expectReadyPage(page);

  const uploadStarted = performance.now();
  await page.locator('input[type="file"]').setInputFiles(generated.path);
  await page.waitForFunction(
    (fileName) =>
      document.body.innerText.includes(fileName) &&
      document.body.innerText.includes("状态：就绪"),
    generated.fileName,
    { timeout: 180_000 },
  );
  const uploadWallMs = roundMetric(performance.now() - uploadStarted);

  await page.locator(".modeTabs").getByRole("button", { name: "对象编辑" }).click();
  await waitForEditViewportReady(page);
  await page.getByLabel("渲染模式").selectOption("clustered");
  await waitForWebGpuViewport(page, 120_000);

  const initialTelemetry = await readWebGpuTelemetry(page);
  const idle = await sampleFramePacing(page, "idle");

  const selectedObject = await selectObjectFromCanvas(page);
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

  const screenshotPath = path.join(outputDir, "synthetic-1m-webgpu-runtime.png");
  await page.screenshot({ path: screenshotPath, fullPage: false });

  const phases = [idle, afterIsolate, afterDelete];
  const checks = [
    check("uploaded-file-present", generated.path, "exists", existsSync(generated.path)),
    check("renderer", deleteTelemetry.renderer, "webgpu-tile", deleteTelemetry.renderer === "webgpu-tile"),
    check(
      "object-filter",
      deleteTelemetry.objectFilter,
      "gpu-object-state-buffer",
      deleteTelemetry.objectFilter === "gpu-object-state-buffer",
    ),
    check(
      "first-frame-status",
      initialTelemetry.firstFrameStatus,
      "rendered",
      initialTelemetry.firstFrameStatus === "rendered",
    ),
    check("first-frame-pixels", initialTelemetry.firstFramePixels, ">0", initialTelemetry.firstFramePixels > 0),
    check(
      "packed-gaussians",
      initialTelemetry.packedGaussians,
      `>= ${generated.gaussianCount}`,
      initialTelemetry.packedGaussians >= generated.gaussianCount,
    ),
    check("tile-references", initialTelemetry.tileReferences, ">0", initialTelemetry.tileReferences > 0),
    check("tile-overflow", initialTelemetry.tileOverflowCount, 0, initialTelemetry.tileOverflowCount === 0),
    check("selected-object", selectedObject, "object id", isObjectId(selectedObject)),
    check(
      "initial-queue-done-ms",
      initialTelemetry.queueDoneMs,
      `<= ${maxInitialQueueDoneMs}`,
      initialTelemetry.queueDoneMs <= maxInitialQueueDoneMs,
    ),
    check(
      "isolate-update-mode",
      isolateTelemetry.storageUpdateMode,
      "object-state-only",
      isolateTelemetry.storageUpdateMode === "object-state-only",
    ),
    check(
      "isolate-update-ms",
      isolateTelemetry.storageUpdateMs,
      `<= ${maxObjectStateUpdateMs}`,
      isolateTelemetry.storageUpdateMs <= maxObjectStateUpdateMs,
    ),
    check(
      "delete-update-mode",
      deleteTelemetry.storageUpdateMode,
      "object-state-only or full-upload",
      ["object-state-only", "full-upload"].includes(deleteTelemetry.storageUpdateMode),
    ),
    check(
      "delete-update-ms",
      deleteTelemetry.storageUpdateMs,
      `<= ${deleteTelemetry.storageUpdateMode === "full-upload" ? maxFullUploadUpdateMs : maxObjectStateUpdateMs}`,
      deleteTelemetry.storageUpdateMs <=
        (deleteTelemetry.storageUpdateMode === "full-upload"
          ? maxFullUploadUpdateMs
          : maxObjectStateUpdateMs),
    ),
    ...phases.flatMap((phase) => phaseChecks(phase)),
  ];

  return {
    asset: "synthetic-upload-1m",
    passed: checks.every((entry) => entry.passed),
    checks,
    uploadWallMs,
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

function generateSyntheticPly({ outputPath, gaussianCount, objectCount }) {
  const header = [
    "ply",
    "format binary_little_endian 1.0",
    "comment ObjGauss synthetic browser runtime audit asset; generated under /tmp and not committed",
    `element vertex ${gaussianCount}`,
    "property float x",
    "property float y",
    "property float z",
    "property float scale_0",
    "property float scale_1",
    "property float scale_2",
    "property float opacity",
    "property float rot_0",
    "property float rot_1",
    "property float rot_2",
    "property float rot_3",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
    "property int object_id",
    "end_header",
    "",
  ].join("\n");
  const rowStride = 51;
  const body = Buffer.allocUnsafe(gaussianCount * rowStride);
  const side = Math.ceil(Math.sqrt(objectCount));
  const perObject = Math.ceil(gaussianCount / objectCount);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  for (let index = 0; index < gaussianCount; index += 1) {
    const objectId = Math.min(objectCount - 1, Math.floor(index / perObject));
    const objectLocalIndex = index - objectId * perObject;
    const column = objectId % side;
    const row = Math.floor(objectId / side);
    const centerX = coordinateFromGrid(column, side, 3.4);
    const centerY = coordinateFromGrid(row, side, 2.5);
    const centerZ = ((objectId % 11) - 5) * 0.035;
    const angle = objectLocalIndex * goldenAngle;
    const radial = Math.sqrt((objectLocalIndex + 0.5) / perObject);
    const radius = 0.055 + 0.02 * ((objectId % 5) / 4);
    const ripple = Math.sin(objectLocalIndex * 0.017 + objectId) * 0.012;
    const x = centerX + Math.cos(angle) * radial * radius;
    const y = centerY + Math.sin(angle) * radial * radius;
    const z = centerZ + ripple;
    const scale = 0.003 + 0.001 * ((objectId % 7) / 6);
    const opacity = 0.55;
    const [red, green, blue] = objectPalette(objectId);

    let offset = index * rowStride;
    body.writeFloatLE(x, offset);
    offset += 4;
    body.writeFloatLE(y, offset);
    offset += 4;
    body.writeFloatLE(z, offset);
    offset += 4;
    body.writeFloatLE(scale, offset);
    offset += 4;
    body.writeFloatLE(scale * 1.2, offset);
    offset += 4;
    body.writeFloatLE(scale * 0.85, offset);
    offset += 4;
    body.writeFloatLE(opacity, offset);
    offset += 4;
    body.writeFloatLE(1, offset);
    offset += 4;
    body.writeFloatLE(0, offset);
    offset += 4;
    body.writeFloatLE(0, offset);
    offset += 4;
    body.writeFloatLE(0, offset);
    offset += 4;
    body.writeUInt8(red, offset);
    offset += 1;
    body.writeUInt8(green, offset);
    offset += 1;
    body.writeUInt8(blue, offset);
    offset += 1;
    body.writeInt32LE(objectId, offset);
  }

  writeFileSync(outputPath, Buffer.concat([Buffer.from(header, "ascii"), body]));
  const stats = statSync(outputPath);
  return {
    path: outputPath,
    fileName: path.basename(outputPath),
    gaussianCount,
    objectCount,
    rowStride,
    byteSize: stats.size,
  };
}

function coordinateFromGrid(index, side, span) {
  if (side <= 1) return 0;
  return (index / (side - 1) - 0.5) * span;
}

function objectPalette(objectId) {
  const hue = (objectId * 47) % 360;
  const saturation = 0.62 + 0.18 * ((objectId % 3) / 2);
  const lightness = 0.46 + 0.14 * ((objectId % 5) / 4);
  const [r, g, b] = hslToRgb(hue / 360, saturation, lightness);
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function hslToRgb(h, s, l) {
  if (s === 0) return [l, l, l];
  const hue2rgb = (p, q, t) => {
    let value = t;
    if (value < 0) value += 1;
    if (value > 1) value -= 1;
    if (value < 1 / 6) return p + (q - p) * 6 * value;
    if (value < 1 / 2) return q;
    if (value < 2 / 3) return p + (q - p) * (2 / 3 - value) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)];
}

async function expectReadyPage(page) {
  const title = await page.title();
  if (title !== "ObjGauss 查看器") {
    throw new Error(`unexpected page title: ${title}`);
  }
  await page.getByText("素材库").first().waitFor({ timeout: 15_000 });
  const bodyText = await page.locator("body").innerText();
  for (const signal of ["Vite Error", "Internal server error", "Failed to resolve import"]) {
    if (bodyText.includes(signal)) throw new Error(`framework overlay detected: ${signal}`);
  }
}

async function waitForEditViewportReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    const renderer = viewport?.getAttribute("data-renderer");
    const webGpuStatus = viewport?.getAttribute("data-webgpu-status");
    return (renderer === "gaussian-oit" || renderer === "webgpu-tile") && webGpuStatus !== "pending";
  }, undefined, { timeout: 120_000 });
}

async function waitForWebGpuViewport(page, timeoutMs) {
  await page.waitForFunction(
    () => {
      const viewport = document.querySelector(".viewport");
      return (
        viewport?.getAttribute("data-renderer") === "webgpu-tile" &&
        viewport?.getAttribute("data-object-filter") === "gpu-object-state-buffer" &&
        viewport?.getAttribute("data-webgpu-first-frame-status") === "rendered" &&
        ["done", "submitted"].includes(viewport?.getAttribute("data-webgpu-queue-status") ?? "")
      );
    },
    undefined,
    { timeout: timeoutMs },
  );
  await page.waitForTimeout(250);
}

async function waitForWebGpuStorageUpdate(page, { previousChecksum, allowFullUpload }, timeoutMs = 60_000) {
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

async function selectObjectFromCanvas(page, previousSelected = "无") {
  const canvas = page.locator(".viewport canvas").first();
  const box = await canvas.boundingBox();
  if (!box) throw new Error("WebGPU canvas is missing");
  const clickPoints = [
    [0.5, 0.5],
    [0.45, 0.48],
    [0.55, 0.48],
    [0.4, 0.55],
    [0.6, 0.55],
    [0.5, 0.4],
    [0.35, 0.48],
    [0.65, 0.48],
    [0.5, 0.62],
    [0.5, 0.35],
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
  throw new Error("canvas selection did not choose an object");
}

async function selectedObjectValue(page) {
  const status = await page.locator(".statusBar").innerText();
  const match = status.match(/所选：([^\n]+)/);
  return match?.[1] ?? "无";
}

function buildSuiteChecks(row) {
  return [
    check("gaussian-count-request", gaussianCount, ">= 1000000", gaussianCount >= 1_000_000),
    check("object-count-request", objectCount, ">= 2", objectCount >= 2),
    ...row.checks.map((entry) => ({ asset: row.asset, ...entry })),
  ];
}

function summarizeRow(row) {
  const phases = row.phases ?? [];
  return {
    scenes: 1,
    uploadedGaussians: row.packedGaussians,
    maxTileReferences: row.tileReferences,
    maxMeanFrameMs: maxNumeric(phases.map((phase) => phase.meanFrameMs)),
    maxP95FrameMs: maxNumeric(phases.map((phase) => phase.p95FrameMs)),
    maxLongFrameRatio: maxNumeric(phases.map((phase) => phase.longFrameRatio)),
    minApproxFps: minNumeric(phases.map((phase) => phase.approxFps)),
    uploadWallMs: row.uploadWallMs,
    initialQueueDoneMs: row.initial.queueDoneMs,
    isolateUpdateMs: row.isolate.storageUpdateMs,
    deleteUpdateMs: row.delete.storageUpdateMs,
  };
}

function writeReport(summary) {
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary), "utf8");
}

function renderMarkdown(summary) {
  const row = summary.row;
  const lines = [
    "# WebGPU Synthetic 1M Runtime Audit",
    "",
    `- Mode: \`${summary.mode}\``,
    `- Status: \`${summary.passed ? "passed" : "failed"}\``,
    `- Generated: \`${summary.generatedAt}\``,
    `- URL: \`${summary.url}\``,
    `- Headed: \`${summary.headed ? "true" : "false"}\``,
    `- Generated PLY: \`${summary.generated.path}\``,
    `- PLY bytes: \`${summary.generated.byteSize}\``,
    `- Proof: browserRuntime1m=\`${summary.proof.browserRuntime1m}\`, realTrainedScene1m=\`${summary.proof.realTrainedScene1m}\`, sustainedFpsSla=\`${summary.proof.sustainedFpsSla}\``,
    "",
    summary.interpretation,
    "",
    "| Phase | Mean ms | P95 ms | Long frame ratio | Approx FPS |",
    "| --- | ---: | ---: | ---: | ---: |",
  ];
  for (const phase of row.phases) {
    lines.push(
      `| ${escapeMarkdown(phase.phase)} | ${phase.meanFrameMs} | ${phase.p95FrameMs} | ${phase.longFrameRatio} | ${phase.approxFps} |`,
    );
  }
  lines.push(
    "",
    "## Runtime",
    "",
    `- Uploaded gaussians: \`${row.packedGaussians}\``,
    `- Tile references: \`${row.tileReferences}\``,
    `- First-frame pixels: \`${row.firstFramePixels}\``,
    `- Upload wall ms: \`${row.uploadWallMs}\``,
    `- Initial queue done ms: \`${row.initial.queueDoneMs}\``,
    `- Isolate update: \`${row.isolate.storageUpdateMode}:${row.isolate.storageUpdateMs}ms\``,
    `- Delete update: \`${row.delete.storageUpdateMode}:${row.delete.storageUpdateMs}ms\``,
    `- Screenshot: \`${row.screenshotPath}\``,
    "",
    "## Failed Checks",
    "",
  );
  const failed = summary.checks.filter((entry) => !entry.passed);
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
  parsed.searchParams.set("uploaded-ply-splat-source", "off");
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
  const deadline = Date.now() + 30_000;
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

function positiveInteger(value, fallback) {
  const parsed = positiveFiniteNumber(value, fallback);
  return Math.max(1, Math.round(parsed));
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
