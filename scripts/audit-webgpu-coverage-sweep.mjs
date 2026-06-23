import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const DEFAULT_PORT = 5265;
const DEFAULT_ASSET = "nerf-lego-alpha-closure-local";
const DEFAULT_VARIANTS = [
  { id: "baseline", footprintScale: 2.2, maxAnisotropy: 4 },
  { id: "compact", footprintScale: 1.9, maxAnisotropy: 3 },
  { id: "tight", footprintScale: 1.7, maxAnisotropy: 2.5 },
];

const args = parseArgs(process.argv.slice(2));
const asset = String(args.asset ?? DEFAULT_ASSET);
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const variants = parseVariants(args.variants ?? args.variant);
const shouldStartServer = !(args.url || args.noServer || args["no-server"]);
const webGpuFlags = String(args.webgpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const headed = !flagEnabled(args.headless);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const webGpuViewportSize = optionalPositiveInteger(
  args.webGpuViewportSize ?? args["webgpu-viewport-size"],
);
let server = null;

try {
  if (shouldStartServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before coverage sweep");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  const results = [];
  let stoppedEarly = false;
  for (const variant of variants) {
    const result = await runVariant({ variant, asset, baseUrl, webGpuFlags, headed, webGpuViewportSize });
    process.stdout.write(result.output);
    process.stderr.write(result.errorOutput);
    const metrics = parseVariantMetrics(result.output);
    results.push({ ...variant, ...metrics, passed: result.exitCode === 0 });
    console.log(
      `webgpu_coverage_sweep_variant=${JSON.stringify(variant.id)} ` +
        `passed=${result.exitCode === 0} ` +
        `footprint=${variant.footprintScale} maxAnisotropy=${variant.maxAnisotropy} ` +
        `coverageRatio=${metrics.coverageRatio ?? "unknown"} ` +
        `lumaDelta=${metrics.lumaDelta ?? "unknown"} ` +
        `chromaDelta=${metrics.chromaDelta ?? "unknown"}`,
    );
    if (result.exitCode !== 0 && !allowFailures) {
      process.exitCode = 1;
      stoppedEarly = true;
      break;
    }
  }

  const passed = results.filter((result) => result.passed && Number.isFinite(result.coverageRatio));
  const bestCoverage = passed
    .slice()
    .sort((left, right) => left.coverageRatio - right.coverageRatio)[0];
  console.log(
    `webgpu_coverage_sweep=${!stoppedEarly && passed.length === variants.length ? "passed" : "partial"} ` +
      `asset=${JSON.stringify(asset)} variants=${variants.length} ` +
      `bestCoverage=${JSON.stringify(bestCoverage?.id ?? "none")}:${bestCoverage?.coverageRatio ?? "unknown"} ` +
      `url=${baseUrl}`,
  );
} finally {
  if (server) stopPreviewServer(server);
}

async function runVariant({ variant, asset, baseUrl, webGpuFlags, headed, webGpuViewportSize }) {
  const commandArgs = [
    "scripts/audit-webgpu-desktop.mjs",
    "--asset",
    asset,
    "--url",
    baseUrl,
    "--no-server",
    "--probes",
    "full",
    "--webgpu-flags",
    webGpuFlags,
    "--webgpu-footprint-scale",
    String(variant.footprintScale),
    "--webgpu-covariance-max-anisotropy",
    String(variant.maxAnisotropy),
  ];
  if (!headed) commandArgs.push("--headless");
  if (webGpuViewportSize) {
    commandArgs.push("--webgpu-viewport-size", String(webGpuViewportSize));
  }
  return runProcess(process.execPath, commandArgs);
}

function parseVariantMetrics(output) {
  const residual = output.match(
    /visualResidual="spark-edit-visual-residual-v1":([0-9.]+)\/([0-9.]+):([0-9.]+):([0-9.]+):([0-9.]+)/,
  );
  const pixelCoverage = output.match(
    /pixelCoverage="([^"]+)":"([^"]+)":([0-9.]+):([0-9.]+)/,
  );
  const screenCovariance = output.match(
    /screenCovariance="([^"]+)":[0-9]+\/[0-9]+\/[0-9]+:([0-9.]+):([0-9.]+)/,
  );
  return {
    sparkCoverage: residual ? Number(residual[1]) : null,
    editCoverage: residual ? Number(residual[2]) : null,
    coverageRatio: residual ? Number(residual[3]) : null,
    lumaDelta: residual ? Number(residual[4]) : null,
    chromaDelta: residual ? Number(residual[5]) : null,
    coverageMode: pixelCoverage?.[1] ?? "",
    tuningMode: pixelCoverage?.[2] ?? "",
    coverageWeightFloor: pixelCoverage ? Number(pixelCoverage[3]) : null,
    footprintScale: pixelCoverage ? Number(pixelCoverage[4]) : null,
    maxAnisotropy: screenCovariance ? Number(screenCovariance[2]) : null,
    sigmaMean: screenCovariance ? Number(screenCovariance[3]) : null,
  };
}

function parseVariants(value) {
  if (!value) return DEFAULT_VARIANTS;
  return String(value)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const [id, footprintScale, maxAnisotropy] = entry.split(":");
      return {
        id,
        footprintScale: Number(footprintScale),
        maxAnisotropy: Number(maxAnisotropy),
      };
    })
    .filter((variant) =>
      variant.id &&
      Number.isFinite(variant.footprintScale) &&
      Number.isFinite(variant.maxAnisotropy),
    );
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

function optionalPositiveInteger(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  return Math.round(parsed);
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === false || value === undefined || value === null) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
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
