import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const DEFAULT_PORT = 5321;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-webgpu-offscreen-readback";
const DEFAULT_ASSETS = [
  "nerf-lego-alpha-closure-local",
  "plush-semantic-closure-local",
];
const MODE = "webgpu-offscreen-readback-suite-v1";
const PROBE = "offscreen-readback";

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const assets = parseAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const headed = flagEnabled(args.headed ?? args.headful);
const shouldStartServer = !(args.url || args.noServer || args["no-server"]);
const webGpuViewportSize = optionalPositiveInteger(
  args.webGpuViewportSize ?? args["webgpu-viewport-size"],
);

let server = null;

try {
  if (assets.length === 0) {
    throw new Error("at least one asset is required for offscreen readback suite");
  }

  if (shouldStartServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before offscreen readback suite");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  mkdirSync(outputDir, { recursive: true });
  const results = [];
  for (const asset of assets) {
    console.log(`webgpu_offscreen_readback_asset_start asset=${JSON.stringify(asset)}`);
    const result = await runOffscreenReadback({
      asset,
      baseUrl,
      webGpuFlags,
      headed,
      webGpuViewportSize,
    });
    process.stdout.write(result.output);
    process.stderr.write(result.errorOutput);
    const parsed = parseAuditOutput(result.output);
    const checks = evaluateResult({ asset, exitCode: result.exitCode, parsed });
    const passed = checks.every((check) => check.passed);
    const row = {
      asset,
      passed,
      exitCode: result.exitCode,
      checks,
      ...parsed,
    };
    results.push(row);
    for (const check of checks) {
      console.log(
        `webgpu_offscreen_readback_check=${check.passed ? "passed" : "failed"} ` +
          `asset=${JSON.stringify(asset)} check=${JSON.stringify(check.name)} ` +
          `actual=${JSON.stringify(check.actual)} expected=${JSON.stringify(check.expected)}`,
      );
    }
    console.log(
      `webgpu_offscreen_readback_asset=${passed ? "passed" : "failed"} ` +
        `asset=${JSON.stringify(asset)} firstFrame=${JSON.stringify(row.firstFrameStatus)}:${row.firstFramePixels ?? 0} ` +
        `queue=${JSON.stringify(row.queueStatus)}:${JSON.stringify(row.queueReason)} ` +
        `deviceLost=${JSON.stringify(row.deviceLostStatus)}:${JSON.stringify(row.deviceLostReason)} ` +
        `pixel=${JSON.stringify(row.pixelStatus)}:${JSON.stringify(row.pixelSource)}:${row.pixelWorkgroups ?? 0} ` +
        `readback=${JSON.stringify(row.readbackStatus)}:${JSON.stringify(row.readbackSource)}:${JSON.stringify(row.readbackChecksum)}:${row.readbackByteSize ?? 0}:${row.readbackFiniteFloats ?? 0}/${row.readbackFloatCount ?? 0}:${row.readbackNonzeroFloats ?? 0} ` +
        `packedGaussians=${row.packedGaussians ?? 0} tileReferences=${row.tileReferences ?? 0}`,
    );
    if (!passed && !allowFailures) {
      process.exitCode = 1;
      break;
    }
  }

  const summary = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    url: baseUrl,
    assets,
    webGpuFlags,
    headed,
    webGpuViewportSize: webGpuViewportSize ?? null,
    passed: results.length === assets.length && results.every((result) => result.passed),
    results,
  };
  const summaryJson = `${outputDir}/summary.json`;
  const summaryMd = `${outputDir}/summary.md`;
  writeFileSync(summaryJson, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  writeFileSync(summaryMd, renderMarkdown(summary), "utf8");
  console.log(
    `webgpu_offscreen_readback_report=written outputDir=${JSON.stringify(outputDir)} ` +
      `summaryJson=${JSON.stringify(summaryJson)} summaryMd=${JSON.stringify(summaryMd)}`,
  );
  console.log(
    `webgpu_offscreen_readback=${summary.passed ? "passed" : "failed"} ` +
      `assets=${JSON.stringify(assets)} scenes=${results.length}/${assets.length} ` +
      `webGpuFlags=${JSON.stringify(webGpuFlags)} headed=${headed} url=${baseUrl}`,
  );
  if (!summary.passed && !allowFailures) {
    process.exitCode = 1;
  }
} finally {
  if (server) stopPreviewServer(server);
}

async function runOffscreenReadback({
  asset,
  baseUrl,
  webGpuFlags,
  headed,
  webGpuViewportSize,
}) {
  const commandArgs = [
    "scripts/audit-demo.mjs",
    "--require-webgpu",
    "--webgpu-flags",
    webGpuFlags,
    "--webgpu-probe",
    PROBE,
    "--asset",
    asset,
    "--url",
    baseUrl,
    "--no-server",
  ];
  if (headed) commandArgs.push("--headed");
  if (webGpuViewportSize) {
    commandArgs.push("--webgpu-viewport-size", String(webGpuViewportSize));
  }
  const result = await runProcess(process.execPath, commandArgs);
  return {
    asset,
    ...result,
  };
}

function parseAuditOutput(output) {
  const line = output
    .split("\n")
    .find((entry) => entry.startsWith("asset=") && entry.includes(`runtimeProbe="${PROBE}"`));
  if (!line) {
    return { parsed: false, rawLine: "" };
  }
  const firstFrame = line.match(/firstFrame="([^"]+)":([0-9]+)/);
  const viewport = line.match(/webgpuViewport=([0-9]+)x([0-9]+):([0-9]+):"([^"]+)":"([^"]+)":([0-9]+)/);
  const queue = line.match(/queue="([^"]+)":"([^"]+)"/);
  const deviceLost = line.match(/deviceLost="([^"]+)":"([^"]+)"/);
  const deviceError = line.match(/deviceError="([^"]*)":"([^"]*)"/);
  const pixel = line.match(/pixel="([^"]*)":"([^"]*)":([0-9]+)/);
  const readback = line.match(/readback="([^"]*)":"([^"]*)":"([^"]*)":([0-9]+):([0-9]+)\/([0-9]+):([0-9]+)/);
  const storage = line.match(/storage="([^"]*)":"([^"]*)"/);
  const storageLimit = line.match(/storageLimit="([^"]*)":"([^"]*)":"([^"]*)":([0-9]+):([0-9]+)\/([0-9]+)/);
  const activeTiles = line.match(/activeTiles=([0-9]+)\/([0-9]+)/);

  return {
    parsed: true,
    rawLine: line,
    assetId: stringMatch(line, /^asset=([^\s]+)/),
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
    readbackStatus: readback?.[1] ?? "",
    readbackSource: readback?.[2] ?? "",
    readbackChecksum: readback?.[3] ?? "",
    readbackByteSize: numberCapture(readback?.[4]),
    readbackFiniteFloats: numberCapture(readback?.[5]),
    readbackFloatCount: numberCapture(readback?.[6]),
    readbackNonzeroFloats: numberCapture(readback?.[7]),
    storageStatus: storage?.[1] ?? "",
    storageChecksum: storage?.[2] ?? "",
    storageLimitGate: storageLimit?.[1] ?? "",
    storageLimitBlocker: storageLimit?.[2] ?? "",
    storageEstimatedMaxBufferKey: storageLimit?.[3] ?? "",
    storageEstimatedMaxBufferByteSize: numberCapture(storageLimit?.[4]),
    storageRequiredBuffersPerStage: numberCapture(storageLimit?.[5]),
    storageMaxBuffersPerStage: numberCapture(storageLimit?.[6]),
    packedGaussians: numberMatch(line, /packedGaussians=([0-9]+)/),
    binnedGaussians: numberMatch(line, /binnedGaussians=([0-9]+)/),
    activeTileCount: numberCapture(activeTiles?.[1]),
    tileCount: numberCapture(activeTiles?.[2]),
    tileReferences: numberMatch(line, /tileReferences=([0-9]+)/),
    maxTileOccupancy: numberMatch(line, /maxTileOccupancy=([0-9]+)/),
    tileOverflowCount: numberMatch(line, /tileOverflowCount=([0-9]+)/),
    objectFilter: jsonStringCapture(line, /objectFilter=("[^"]+")/),
    objectFilterTarget: jsonStringCapture(line, /objectFilterTarget=("[^"]+")/),
    screenshotPath: stringMatch(line, /screenshot=([^\s]+)/),
  };
}

function evaluateResult({ asset, exitCode, parsed }) {
  const checks = [
    check("exit-code", exitCode, 0, exitCode === 0),
    check("parsed-audit-line", parsed.parsed, true, parsed.parsed === true),
    check("asset-id", parsed.assetId, asset, parsed.assetId === asset),
    check("runtime-probe", parsed.runtimeProbe, PROBE, parsed.runtimeProbe === PROBE),
    check("first-frame-status", parsed.firstFrameStatus, "readback", parsed.firstFrameStatus === "readback"),
    check("first-frame-pixels", parsed.firstFramePixels, ">0", parsed.firstFramePixels > 0),
    check("queue-status", parsed.queueStatus, "done", parsed.queueStatus === "done"),
    check("device-lost-status", parsed.deviceLostStatus, "active", parsed.deviceLostStatus === "active"),
    check("device-error-status", parsed.deviceErrorStatus, "none", parsed.deviceErrorStatus === "none"),
    check("pixel-status", parsed.pixelStatus, "dispatched", parsed.pixelStatus === "dispatched"),
    check("pixel-source", parsed.pixelSource, "webgpu-compute-*", parsed.pixelSource?.startsWith("webgpu-compute-")),
    check("pixel-workgroups", parsed.pixelWorkgroups, ">0", parsed.pixelWorkgroups > 0),
    check("readback-status", parsed.readbackStatus, "mapped", parsed.readbackStatus === "mapped"),
    check("readback-source", parsed.readbackSource, parsed.pixelSource, parsed.readbackSource === parsed.pixelSource),
    check("readback-checksum", parsed.readbackChecksum, "8 hex chars", /^[0-9a-f]{8}$/.test(parsed.readbackChecksum ?? "")),
    check("readback-byte-size", parsed.readbackByteSize, ">0", parsed.readbackByteSize > 0),
    check(
      "readback-finite-floats",
      `${parsed.readbackFiniteFloats}/${parsed.readbackFloatCount}`,
      "finite == total > 0",
      parsed.readbackFloatCount > 0 && parsed.readbackFiniteFloats === parsed.readbackFloatCount,
    ),
    check("readback-nonzero-floats", parsed.readbackNonzeroFloats, ">0", parsed.readbackNonzeroFloats > 0),
    check("storage-status", parsed.storageStatus, "uploaded", parsed.storageStatus === "uploaded"),
    check("storage-limit", parsed.storageLimitGate, "pass", parsed.storageLimitGate === "pass"),
    check("object-filter", parsed.objectFilter, "gpu-object-state-buffer", parsed.objectFilter === "gpu-object-state-buffer"),
    check(
      "object-filter-target",
      parsed.objectFilterTarget,
      "gpu-object-state-buffer",
      parsed.objectFilterTarget === "gpu-object-state-buffer",
    ),
  ];
  return checks;
}

function check(name, actual, expected, passed) {
  return { name, actual, expected, passed: Boolean(passed) };
}

function renderMarkdown(summary) {
  const lines = [
    "# WebGPU Offscreen Readback Suite",
    "",
    `Mode: \`${summary.mode}\``,
    `Generated: \`${summary.generatedAt}\``,
    `URL: \`${summary.url}\``,
    `WebGPU flags: \`${summary.webGpuFlags}\``,
    "",
    "| Asset | Passed | Frame pixels | Readback checksum | Readback bytes | Finite floats | Nonzero floats | Queue | Device | Packed | Tile refs |",
    "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
  ];
  for (const result of summary.results) {
    lines.push(
      `| ${escapeMarkdown(result.asset)} | ${result.passed ? "yes" : "no"} | ${formatValue(result.firstFramePixels)} | ${escapeMarkdown(result.readbackChecksum || "")} | ${formatValue(result.readbackByteSize)} | ${formatValue(result.readbackFiniteFloats)}/${formatValue(result.readbackFloatCount)} | ${formatValue(result.readbackNonzeroFloats)} | ${escapeMarkdown(result.queueStatus || "")} | ${escapeMarkdown(result.deviceLostStatus || "")} | ${formatValue(result.packedGaussians)} | ${formatValue(result.tileReferences)} |`,
    );
  }
  lines.push("", "## Failed Checks", "");
  const failedChecks = summary.results.flatMap((result) =>
    result.checks
      .filter((entry) => !entry.passed)
      .map((entry) => ({ asset: result.asset, ...entry })),
  );
  if (failedChecks.length === 0) {
    lines.push("None.");
  } else {
    lines.push("| Asset | Check | Actual | Expected |", "| --- | --- | --- | --- |");
    for (const entry of failedChecks) {
      lines.push(
        `| ${escapeMarkdown(entry.asset)} | ${escapeMarkdown(entry.name)} | ${escapeMarkdown(String(entry.actual))} | ${escapeMarkdown(String(entry.expected))} |`,
      );
    }
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
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

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function optionalPositiveInteger(value) {
  if (value === undefined || value === null || value === true || value === false) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return Math.round(parsed);
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
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "";
  return String(value);
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
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

function runProcess(command, commandArgs) {
  return new Promise((resolve) => {
    const child = spawn(command, commandArgs, {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    let errorOutput = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      errorOutput += chunk.toString();
    });
    child.on("close", (exitCode) => {
      resolve({
        exitCode: exitCode ?? 1,
        output,
        errorOutput,
      });
    });
  });
}
