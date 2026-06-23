import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const DEFAULT_PORT = 5230;
const DEFAULT_ASSET = "nerf-lego-alpha-closure-local";
const DEFAULT_PROBES = ["clear-only", "texture-display-only", "full"];

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const asset = String(args.asset ?? DEFAULT_ASSET);
const probes = String(args.probes ?? DEFAULT_PROBES.join(","))
  .split(",")
  .map((probe) => probe.trim())
  .filter(Boolean);
const webGpuFlags = String(args.webgpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const headed = !flagEnabled(args.headless);
const allowDeviceLostProbes = flagEnabled(
  args.allowDeviceLostProbes ?? args["allow-device-lost-probes"],
);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const browserChannel = optionalString(args.browserChannel ?? args["browser-channel"]);
const executablePath = optionalString(args.executablePath ?? args["executable-path"]);
const slowMo = optionalString(args.slowMo ?? args["slow-mo"]);
const webGpuViewportSize = optionalPositiveInteger(
  args.webGpuViewportSize ?? args["webgpu-viewport-size"],
);
const webGpuFootprintScale = optionalFiniteNumber(
  args.webGpuFootprintScale ?? args["webgpu-footprint-scale"],
);
const webGpuCovarianceMaxAnisotropy = optionalFiniteNumber(
  args.webGpuCovarianceMaxAnisotropy ?? args["webgpu-covariance-max-anisotropy"],
);
const webGpuDepthBins = optionalFiniteNumber(
  args.webGpuDepthBins ?? args["webgpu-depth-bins"],
);
const shouldStartServer = !(args.url || args.noServer || args["no-server"]);
let server = null;

try {
  if (shouldStartServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before desktop WebGPU audit");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  const results = [];
  for (const probe of probes) {
    const result = await runProbe({
      asset,
      baseUrl,
      probe,
      webGpuFlags,
      headed,
      allowDeviceLost: allowDeviceLostProbes && probe !== "full",
      browserChannel,
      executablePath,
      slowMo,
      webGpuViewportSize,
      webGpuFootprintScale,
      webGpuCovarianceMaxAnisotropy,
      webGpuDepthBins,
    });
    results.push(result);
    process.stdout.write(result.output);
    process.stderr.write(result.errorOutput);
    console.log(
      `webgpu_desktop_probe=${result.passed ? "passed" : "failed"} ` +
        `probe=${JSON.stringify(probe)} exitCode=${result.exitCode} ` +
        `classification=${JSON.stringify(classifyProbe(result))}`,
    );
  }

  const failed = results.filter((result) => !result.passed);
  const classification = classifySuite(results);
  console.log(
    `webgpu_desktop_audit=${failed.length === 0 ? "passed" : "failed"} ` +
      `asset=${JSON.stringify(asset)} url=${baseUrl} headed=${headed} ` +
      `webGpuFlags=${JSON.stringify(webGpuFlags)} probes=${JSON.stringify(probes)} ` +
      `webGpuViewportSize=${webGpuViewportSize ?? "default"} ` +
      `webGpuFootprintScale=${webGpuFootprintScale ?? "default"} ` +
      `webGpuCovarianceMaxAnisotropy=${webGpuCovarianceMaxAnisotropy ?? "default"} ` +
      `webGpuDepthBins=${webGpuDepthBins ?? "default"} ` +
      `classification=${JSON.stringify(classification)}`,
  );
  if (failed.length > 0 && !allowFailures) {
    process.exitCode = 1;
  }
} finally {
  if (server) stopPreviewServer(server);
}

async function runProbe({
  asset,
  baseUrl,
  probe,
  webGpuFlags,
  headed,
  allowDeviceLost,
  browserChannel,
  executablePath,
  slowMo,
  webGpuViewportSize,
  webGpuFootprintScale,
  webGpuCovarianceMaxAnisotropy,
  webGpuDepthBins,
}) {
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
  ];
  if (probe !== "full") {
    commandArgs.push("--webgpu-probe", probe);
  }
  if (headed) commandArgs.push("--headed");
  if (allowDeviceLost) commandArgs.push("--allow-webgpu-device-lost");
  if (browserChannel) commandArgs.push("--browser-channel", browserChannel);
  if (executablePath) commandArgs.push("--executable-path", executablePath);
  if (slowMo) commandArgs.push("--slow-mo", slowMo);
  if (webGpuViewportSize) {
    commandArgs.push("--webgpu-viewport-size", String(webGpuViewportSize));
  }
  if (Number.isFinite(webGpuFootprintScale)) {
    commandArgs.push("--webgpu-footprint-scale", String(webGpuFootprintScale));
  }
  if (Number.isFinite(webGpuCovarianceMaxAnisotropy)) {
    commandArgs.push("--webgpu-covariance-max-anisotropy", String(webGpuCovarianceMaxAnisotropy));
  }
  if (Number.isFinite(webGpuDepthBins)) {
    commandArgs.push("--webgpu-depth-bins", String(webGpuDepthBins));
  }

  const result = await runProcess(process.execPath, commandArgs);
  return {
    probe,
    ...result,
    passed: result.exitCode === 0,
  };
}

function classifyProbe(result) {
  const output = `${result.output}\n${result.errorOutput}`;
  if (result.passed && result.probe === "full") return "desktop-webgpu-full-runtime-pass";
  if (result.passed) return "desktop-webgpu-probe-pass";
  if (output.includes("webgpu-adapter-unavailable") || output.includes("No available adapters.")) {
    return "webgpu-adapter-unavailable";
  }
  if (
    output.includes("webgpu-device-lost-destroyed") ||
    output.includes("A valid external Instance reference no longer exists")
  ) {
    return "webgpu-presentation-or-backend-loss";
  }
  return "webgpu-runtime-audit-failed";
}

function classifySuite(results) {
  if (results.every((result) => result.passed)) {
    return "desktop-webgpu-runtime-passed";
  }
  const failed = results.filter((result) => !result.passed);
  const failedClassifications = new Set(failed.map(classifyProbe));
  if (failedClassifications.has("webgpu-adapter-unavailable")) {
    return "desktop-webgpu-unavailable";
  }
  if (failedClassifications.has("webgpu-presentation-or-backend-loss")) {
    return "desktop-webgpu-presentation-backend-loss";
  }
  return "desktop-webgpu-runtime-failed";
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
    const child = spawn(command, commandArgs, { stdio: ["ignore", "pipe", "pipe"] });
    let output = "";
    let errorOutput = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      errorOutput += chunk.toString();
    });
    child.on("close", (exitCode) => {
      resolve({ exitCode, output, errorOutput });
    });
  });
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

function flagEnabled(value) {
  if (value === true) return true;
  if (value === false || value === undefined || value === null) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const text = String(value).trim();
  return text ? text : undefined;
}

function optionalPositiveInteger(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  return Math.round(parsed);
}

function optionalFiniteNumber(value) {
  if (value === undefined || value === null || value === true || value === false || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
