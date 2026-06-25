import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const DEFAULT_PORT = 5395;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-webgpu-presentation-performance";
const DEFAULT_ASSETS = [
  "nerf-lego-alpha-closure-local",
  "plush-semantic-closure-local",
];
const MODE = "webgpu-presentation-performance-smoke-v1";

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const assets = parseAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const headed = !flagEnabled(args.headless);
const shouldStartServer = !(args.url || args.noServer || args["no-server"]);
const browserChannel = optionalString(args.browserChannel ?? args["browser-channel"]);
const executablePath = optionalString(args.executablePath ?? args["executable-path"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const maxFullUploadUpdateMs = positiveFiniteNumber(
  args.maxFullUploadUpdateMs ?? args["max-full-upload-update-ms"],
  500,
);
const maxSubmitMs = positiveFiniteNumber(args.maxSubmitMs ?? args["max-submit-ms"], 25);
const maxQueueDoneMs = positiveFiniteNumber(
  args.maxQueueDoneMs ?? args["max-queue-done-ms"],
  2500,
);
const minLargeSceneGaussians = positiveFiniteNumber(
  args.minLargeSceneGaussians ?? args["min-large-scene-gaussians"],
  250000,
);

let server = null;

try {
  if (assets.length === 0) {
    throw new Error("at least one asset is required for WebGPU presentation performance audit");
  }
  if (shouldStartServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before presentation performance audit");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  mkdirSync(outputDir, { recursive: true });
  const rows = [];
  for (const asset of assets) {
    console.log(`webgpu_presentation_performance_asset_start asset=${JSON.stringify(asset)}`);
    const result = await runPresentationProbe(asset);
    process.stdout.write(result.output);
    process.stderr.write(result.errorOutput);
    const parsed = parseAuditOutput(result.output);
    const row = buildRow({ asset, exitCode: result.exitCode, parsed });
    rows.push(row);
    for (const check of row.checks) {
      console.log(
        `webgpu_presentation_performance_check=${check.passed ? "passed" : "failed"} ` +
          `asset=${JSON.stringify(asset)} check=${JSON.stringify(check.name)} ` +
          `actual=${JSON.stringify(check.actual)} expected=${JSON.stringify(check.expected)}`,
      );
    }
    console.log(
      `webgpu_presentation_performance_asset=${row.passed ? "passed" : "failed"} ` +
        `asset=${JSON.stringify(asset)} firstFrame=${JSON.stringify(row.firstFrameStatus)}:${row.firstFramePixels} ` +
        `queue=${JSON.stringify(row.queueStatus)}:${JSON.stringify(row.queueReason)} ` +
        `deviceLost=${JSON.stringify(row.deviceLostStatus)}:${JSON.stringify(row.deviceLostReason)} ` +
        `pixel=${JSON.stringify(row.pixelStatus)}:${JSON.stringify(row.pixelSource)}:${row.pixelWorkgroups} ` +
        `storageTiming=${JSON.stringify(row.storageTiming.mode)}:${row.storageTiming.updateMs}/${row.storageTiming.submitMs}/${row.storageTiming.queueDoneMs} ` +
        `packedGaussians=${row.packedGaussians} tileReferences=${row.tileReferences} screenshot=${JSON.stringify(row.screenshotPath)}`,
    );
    if (!row.passed && !allowFailures) {
      process.exitCode = 1;
      break;
    }
  }

  const checks = buildSuiteChecks(rows);
  const passed =
    rows.length === assets.length &&
    rows.every((row) => row.passed) &&
    checks.every((check) => check.passed);
  const summary = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    outputDir,
    url: baseUrl,
    assets,
    webGpuFlags,
    headed,
    budgets: {
      maxFullUploadUpdateMs,
      maxSubmitMs,
      maxQueueDoneMs,
      minLargeSceneGaussians,
    },
    passed,
    aggregate: summarizeRows(rows),
    checks,
    rows,
  };
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary), "utf8");
  console.log(
    `webgpu_presentation_performance=${passed ? "passed" : "failed"} ` +
      `assets=${JSON.stringify(assets)} scenes=${rows.length}/${assets.length} ` +
      `largestGaussians=${summary.aggregate.largestGaussians} ` +
      `maxUpdateMs=${summary.aggregate.maxUpdateMs} maxQueueDoneMs=${summary.aggregate.maxQueueDoneMs} ` +
      `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
  );
  if (!passed && !allowFailures) {
    process.exitCode = 1;
  }
} finally {
  if (server) stopPreviewServer(server);
}

async function runPresentationProbe(asset) {
  const commandArgs = [
    "scripts/audit-demo.mjs",
    "--require-webgpu",
    "--webgpu-flags",
    webGpuFlags,
    "--asset",
    asset,
    "--url",
    baseUrl,
    "--no-server",
    "--skip-visual-residual",
    "--webgpu-presentation-only",
  ];
  if (headed) commandArgs.push("--headed");
  if (browserChannel) commandArgs.push("--browser-channel", browserChannel);
  if (executablePath) commandArgs.push("--executable-path", executablePath);
  return runProcess(process.execPath, commandArgs);
}

function parseAuditOutput(output) {
  const line = output
    .split("\n")
    .find((entry) => entry.startsWith("asset=") && entry.includes('runtimeProbe="full"'));
  if (!line) {
    return { parsed: false, rawLine: "" };
  }
  const firstFrame = line.match(/firstFrame="([^"]+)":([0-9]+)/);
  const viewport = line.match(/webgpuViewport=([0-9]+)x([0-9]+):([0-9]+):"([^"]+)":"([^"]+)":([0-9]+)/);
  const queue = line.match(/queue="([^"]+)":"([^"]+)"/);
  const deviceLost = line.match(/deviceLost="([^"]+)":"([^"]+)"/);
  const deviceError = line.match(/deviceError="([^"]*)":"([^"]*)"/);
  const pixel = line.match(/pixel="([^"]*)":"([^"]*)":([0-9]+)/);
  const storageTiming = parseStorageTiming(line, "storageTiming");
  return {
    parsed: true,
    rawLine: line,
    runtimeProbe: jsonStringCapture(line, /runtimeProbe=("[^"]+")/),
    firstFrameStatus: firstFrame?.[1] ?? "",
    firstFramePixels: numberCapture(firstFrame?.[2]),
    viewportWidth: numberCapture(viewport?.[1]),
    viewportHeight: numberCapture(viewport?.[2]),
    pixelCount: numberCapture(viewport?.[3]),
    viewportAspectMode: viewport?.[4] ?? "",
    viewportQuality: viewport?.[5] ?? "",
    viewportPixelBudget: numberCapture(viewport?.[6]),
    queueStatus: queue?.[1] ?? "",
    queueReason: queue?.[2] ?? "",
    deviceLostStatus: deviceLost?.[1] ?? "",
    deviceLostReason: deviceLost?.[2] ?? "",
    deviceErrorStatus: deviceError?.[1] ?? "",
    deviceErrorType: deviceError?.[2] ?? "",
    pixelStatus: pixel?.[1] ?? "",
    pixelSource: pixel?.[2] ?? "",
    pixelWorkgroups: numberCapture(pixel?.[3]),
    storageTiming,
    packedGaussians: numberMatch(line, /packedGaussians=([0-9]+)/),
    tileReferences: numberMatch(line, /tileReferences=([0-9]+)/),
    tileOverflowCount: numberMatch(line, /tileOverflowCount=([0-9]+)/),
    screenshotPath: stringMatch(line, /screenshot=([^\s]+)/),
  };
}

function buildRow({ asset, exitCode, parsed }) {
  const storageTiming = parsed.storageTiming ?? {
    status: "",
    mode: "",
    updateMs: null,
    submitMs: null,
    queueDoneMs: null,
    objectStateBytes: 0,
  };
  const checks = [
    check("exit-code", exitCode, 0, exitCode === 0),
    check("parsed-audit-line", parsed.parsed, true, parsed.parsed === true),
    check("runtime-probe", parsed.runtimeProbe, "full", parsed.runtimeProbe === "full"),
    check("first-frame-status", parsed.firstFrameStatus, "rendered", parsed.firstFrameStatus === "rendered"),
    check("first-frame-pixels", parsed.firstFramePixels, ">0", parsed.firstFramePixels > 0),
    check("queue-status", parsed.queueStatus, "done", parsed.queueStatus === "done"),
    check("device-lost-status", parsed.deviceLostStatus, "active", parsed.deviceLostStatus === "active"),
    check("device-error-status", parsed.deviceErrorStatus, "none", parsed.deviceErrorStatus === "none"),
    check("pixel-status", parsed.pixelStatus, "dispatched", parsed.pixelStatus === "dispatched"),
    check("pixel-source", parsed.pixelSource, "webgpu-compute-*", parsed.pixelSource?.startsWith("webgpu-compute-")),
    check("pixel-workgroups", parsed.pixelWorkgroups, ">0", parsed.pixelWorkgroups > 0),
    check(
      "storage-mode",
      `${storageTiming.status}/${storageTiming.mode}`,
      "uploaded/full-upload or object-state-updated/object-state-only",
      (storageTiming.status === "uploaded" && storageTiming.mode === "full-upload") ||
        (storageTiming.status === "object-state-updated" && storageTiming.mode === "object-state-only"),
    ),
    check("storage-update-ms", storageTiming.updateMs, `<= ${maxFullUploadUpdateMs}`, isNonNegativeNumber(storageTiming.updateMs) && storageTiming.updateMs <= maxFullUploadUpdateMs),
    check("submit-ms", storageTiming.submitMs, `<= ${maxSubmitMs}`, isNonNegativeNumber(storageTiming.submitMs) && storageTiming.submitMs <= maxSubmitMs),
    check("queue-done-ms", storageTiming.queueDoneMs, `<= ${maxQueueDoneMs}`, isNonNegativeNumber(storageTiming.queueDoneMs) && storageTiming.queueDoneMs <= maxQueueDoneMs),
    check("packed-gaussians", parsed.packedGaussians, ">0", parsed.packedGaussians > 0),
    check("tile-references", parsed.tileReferences, ">0", parsed.tileReferences > 0),
    check("tile-overflow", parsed.tileOverflowCount, 0, parsed.tileOverflowCount === 0),
    check("screenshot", parsed.screenshotPath, "/tmp path exists", parsed.screenshotPath?.startsWith("/tmp/") && existsSync(parsed.screenshotPath)),
  ];
  return {
    asset,
    passed: checks.every((entry) => entry.passed),
    checks,
    ...parsed,
    storageTiming,
  };
}

function buildSuiteChecks(rows) {
  const largeScenes = rows.filter((row) => row.packedGaussians >= minLargeSceneGaussians);
  return [
    check("asset-count", rows.length, assets.length, rows.length === assets.length),
    check(
      "large-scene-coverage",
      largeScenes.map((row) => `${row.asset}:${row.packedGaussians}`).join(","),
      `>= ${minLargeSceneGaussians}`,
      largeScenes.length > 0,
    ),
  ];
}

function summarizeRows(rows) {
  return {
    scenes: rows.length,
    largestGaussians: maxNumeric(rows.map((row) => row.packedGaussians)),
    maxTileReferences: maxNumeric(rows.map((row) => row.tileReferences)),
    maxUpdateMs: maxNumeric(rows.map((row) => row.storageTiming.updateMs)),
    maxSubmitMs: maxNumeric(rows.map((row) => row.storageTiming.submitMs)),
    maxQueueDoneMs: maxNumeric(rows.map((row) => row.storageTiming.queueDoneMs)),
  };
}

function renderMarkdown(summary) {
  const lines = [
    "# WebGPU Presentation Performance Smoke",
    "",
    `Mode: \`${summary.mode}\``,
    `Generated: \`${summary.generatedAt}\``,
    `Status: \`${summary.passed ? "passed" : "failed"}\``,
    `URL: \`${summary.url}\``,
    `Headed: \`${summary.headed ? "true" : "false"}\``,
    "",
    "This gate proves the full WebGPU canvas presentation path can render and report timing in a real browser. It is not an FPS benchmark or a 1M interactive SLA.",
    "",
    "| Asset | Passed | Frame pixels | Gaussians | Tile refs | Storage timing | Queue | Screenshot |",
    "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
  ];
  for (const row of summary.rows) {
    lines.push(
      `| ${escapeMarkdown(row.asset)} | ${row.passed ? "yes" : "no"} | ${row.firstFramePixels} | ${row.packedGaussians} | ${row.tileReferences} | ${escapeMarkdown(formatStorageTiming(row.storageTiming))} | ${escapeMarkdown(`${row.queueStatus}/${row.deviceLostStatus}`)} | ${escapeMarkdown(row.screenshotPath)} |`,
    );
  }
  const failed = [
    ...summary.checks,
    ...summary.rows.flatMap((row) =>
      row.checks.map((entry) => ({ asset: row.asset, ...entry })),
    ),
  ].filter((entry) => !entry.passed);
  lines.push("", "## Failed Checks", "");
  if (failed.length === 0) {
    lines.push("None.");
  } else {
    lines.push("| Asset | Check | Actual | Expected |", "| --- | --- | --- | --- |");
    for (const entry of failed) {
      lines.push(
        `| ${escapeMarkdown(entry.asset ?? "suite")} | ${escapeMarkdown(entry.name)} | ${escapeMarkdown(String(entry.actual))} | ${escapeMarkdown(String(entry.expected))} |`,
      );
    }
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function parseStorageTiming(line, token) {
  const match = line.match(new RegExp(`${token}="([^"]*)":"([^"]*)":([^:\\s]+):([^:\\s]+):([^:\\s]+):([0-9]+)`));
  if (!match) {
    return { status: "", mode: "", updateMs: null, submitMs: null, queueDoneMs: null, objectStateBytes: 0 };
  }
  return {
    status: match[1],
    mode: match[2],
    updateMs: finiteNumberCapture(match[3]),
    submitMs: finiteNumberCapture(match[4]),
    queueDoneMs: finiteNumberCapture(match[5]),
    objectStateBytes: numberCapture(match[6]),
  };
}

function runProcess(command, commandArgs) {
  return new Promise((resolve) => {
    const child = spawn(command, commandArgs, { stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    let errorOutput = "";
    child.stdout.on("data", (chunk) => {
      output += chunk;
    });
    child.stderr.on("data", (chunk) => {
      errorOutput += chunk;
    });
    child.on("close", (exitCode) => {
      resolve({ exitCode, output, errorOutput });
    });
    child.on("error", (error) => {
      resolve({ exitCode: 1, output, errorOutput: `${errorOutput}\n${error.message}` });
    });
  });
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
    await sleep(500);
  }
  throw new Error(`app did not become ready at ${url}: ${lastError?.message ?? "timeout"}`);
}

function check(name, actual, expected, passed) {
  return { name, actual, expected, passed: Boolean(passed) };
}

function parseAssets(value) {
  return String(value ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
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

function jsonStringCapture(source, pattern) {
  const match = source.match(pattern);
  if (!match) return "";
  try {
    return JSON.parse(match[1]);
  } catch {
    return "";
  }
}

function numberMatch(source, pattern) {
  return numberCapture(source.match(pattern)?.[1]);
}

function stringMatch(source, pattern) {
  return source.match(pattern)?.[1] ?? "";
}

function numberCapture(value) {
  const parsed = Number(String(value ?? "").replaceAll(",", ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

function finiteNumberCapture(value) {
  const parsed = Number(String(value ?? "").replaceAll(",", ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function isNonNegativeNumber(value) {
  return Number.isFinite(value) && value >= 0;
}

function maxNumeric(values) {
  const numeric = values.filter((value) => Number.isFinite(value));
  if (numeric.length === 0) return 0;
  return Math.max(...numeric);
}

function formatStorageTiming(timing) {
  return `${timing.mode}:${timing.updateMs}/${timing.submitMs}/${timing.queueDoneMs}ms`;
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
