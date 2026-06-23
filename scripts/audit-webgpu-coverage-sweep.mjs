import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const DEFAULT_PORT = 5265;
const DEFAULT_ASSETS = ["nerf-lego-alpha-closure-local"];
const SCORE_WEIGHTS = {
  coverage: 0.35,
  luma: 0.25,
  chroma: 0.25,
  tileReferences: 0.15,
};
const DEFAULT_VARIANTS = [
  { id: "baseline", footprintScale: 2.2, maxAnisotropy: 4 },
  { id: "compact", footprintScale: 1.9, maxAnisotropy: 3 },
  { id: "tight", footprintScale: 1.7, maxAnisotropy: 2.5 },
];

const args = parseArgs(process.argv.slice(2));
const assets = parseAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const variants = parseVariants(args.variants ?? args.variant);
if (variants.length === 0) {
  throw new Error("coverage sweep requires at least one variant formatted as id:footprintScale:maxAnisotropy");
}
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
  sweep:
  for (const asset of assets) {
    for (const variant of variants) {
      const result = await runVariant({ variant, asset, baseUrl, webGpuFlags, headed, webGpuViewportSize });
      process.stdout.write(result.output);
      process.stderr.write(result.errorOutput);
      const metrics = parseVariantMetrics(result.output);
      results.push({ asset, ...variant, ...metrics, passed: result.exitCode === 0 });
      console.log(
        `webgpu_coverage_sweep_variant=${JSON.stringify(variant.id)} ` +
          `asset=${JSON.stringify(asset)} passed=${result.exitCode === 0} ` +
          `footprint=${variant.footprintScale} maxAnisotropy=${variant.maxAnisotropy} ` +
          `coverageRatio=${metrics.coverageRatio ?? "unknown"} ` +
          `lumaDelta=${metrics.lumaDelta ?? "unknown"} ` +
          `chromaDelta=${metrics.chromaDelta ?? "unknown"} ` +
          `tileReferences=${metrics.tileReferenceCount ?? "unknown"}`,
      );
      if (result.exitCode !== 0 && !allowFailures) {
        process.exitCode = 1;
        stoppedEarly = true;
        break sweep;
      }
    }
  }

  const rankedResults = rankResults(results);
  for (const result of rankedResults) {
    console.log(
      `webgpu_coverage_sweep_rank asset=${JSON.stringify(result.asset)} ` +
        `variant=${JSON.stringify(result.id)} passed=${result.passed} ` +
        `paretoScore=${result.paretoScore ?? "unknown"} ` +
        `dominated=${result.paretoDominated ?? "unknown"} ` +
        `coverageNorm=${result.coverageNorm ?? "unknown"} ` +
        `lumaNorm=${result.lumaNorm ?? "unknown"} ` +
        `chromaNorm=${result.chromaNorm ?? "unknown"} ` +
        `tileReferenceNorm=${result.tileReferenceNorm ?? "unknown"}`,
    );
  }

  const sceneSummaries = summarizeScenes(rankedResults, variants.length);
  for (const summary of sceneSummaries) {
    console.log(
      `webgpu_coverage_sweep_scene=${summary.complete ? "passed" : "partial"} ` +
        `asset=${JSON.stringify(summary.asset)} variants=${summary.resultCount}/${variants.length} ` +
        `bestPareto=${JSON.stringify(summary.bestPareto?.id ?? "none")}:${summary.bestPareto?.paretoScore ?? "unknown"} ` +
        `bestCoverage=${JSON.stringify(summary.bestCoverage?.id ?? "none")}:${summary.bestCoverage?.coverageRatio ?? "unknown"} ` +
        `bestLuma=${JSON.stringify(summary.bestLuma?.id ?? "none")}:${summary.bestLuma?.lumaDelta ?? "unknown"} ` +
        `bestChroma=${JSON.stringify(summary.bestChroma?.id ?? "none")}:${summary.bestChroma?.chromaDelta ?? "unknown"} ` +
        `lowestCost=${JSON.stringify(summary.lowestCost?.id ?? "none")}:${summary.lowestCost?.tileReferenceCount ?? "unknown"}`,
    );
  }

  const variantSummaries = summarizeVariants(rankedResults, assets.length);
  for (const summary of variantSummaries) {
    console.log(
      `webgpu_coverage_sweep_variant_summary=${summary.complete ? "passed" : "partial"} ` +
        `variant=${JSON.stringify(summary.id)} scenes=${summary.sceneCount}/${assets.length} ` +
        `meanParetoScore=${summary.meanParetoScore ?? "unknown"} ` +
        `meanCoverageNorm=${summary.meanCoverageNorm ?? "unknown"} ` +
        `meanLumaNorm=${summary.meanLumaNorm ?? "unknown"} ` +
        `meanChromaNorm=${summary.meanChromaNorm ?? "unknown"} ` +
        `meanTileReferenceNorm=${summary.meanTileReferenceNorm ?? "unknown"}`,
    );
  }

  const passed = rankedResults.filter((result) => isScorable(result));
  const bestPareto = passed
    .slice()
    .sort((left, right) => left.paretoScore - right.paretoScore)[0];
  const bestCoverage = passed
    .slice()
    .sort((left, right) => left.coverageRatio - right.coverageRatio)[0];
  const bestMeanPareto = variantSummaries
    .filter((summary) => Number.isFinite(summary.meanParetoScore))
    .slice()
    .sort((left, right) => left.meanParetoScore - right.meanParetoScore)[0];
  console.log(
    `webgpu_coverage_sweep=${!stoppedEarly && sceneSummaries.every((summary) => summary.complete) ? "passed" : "partial"} ` +
      `assets=${JSON.stringify(assets)} scenes=${assets.length} variants=${variants.length} ` +
      `scoreWeights=${JSON.stringify(SCORE_WEIGHTS)} ` +
      `bestMeanParetoVariant=${JSON.stringify(bestMeanPareto?.id ?? "none")}:${bestMeanPareto?.meanParetoScore ?? "unknown"} ` +
      `bestPareto=${JSON.stringify(bestPareto?.asset ?? "none")}/${JSON.stringify(bestPareto?.id ?? "none")}:${bestPareto?.paretoScore ?? "unknown"} ` +
      `bestCoverage=${JSON.stringify(bestCoverage?.asset ?? "none")}/${JSON.stringify(bestCoverage?.id ?? "none")}:${bestCoverage?.coverageRatio ?? "unknown"} ` +
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
  const tileReferences = output.match(/tileReferences=([0-9]+)/);
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
    tileReferenceCount: tileReferences ? Number(tileReferences[1]) : null,
  };
}

function parseAssets(value) {
  const parsed = String(value ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  return parsed.length > 0 ? parsed : DEFAULT_ASSETS;
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

function rankResults(results) {
  const byAsset = new Map();
  for (const result of results) {
    if (!byAsset.has(result.asset)) byAsset.set(result.asset, []);
    byAsset.get(result.asset).push(result);
  }
  const ranked = [];
  for (const assetResults of byAsset.values()) {
    const scorable = assetResults.filter((result) =>
      result.passed &&
      Number.isFinite(result.coverageRatio) &&
      Number.isFinite(result.lumaDelta) &&
      Number.isFinite(result.chromaDelta) &&
      Number.isFinite(result.tileReferenceCount),
    );
    const baseline = scorable.find((result) => result.id === "baseline") ?? scorable[0];
    const scored = assetResults.map((result) => scoreResult(result, baseline));
    for (const result of scored) {
      ranked.push({
        ...result,
        paretoDominated: isScorable(result) ? isParetoDominated(result, scored) : null,
      });
    }
  }
  return ranked;
}

function scoreResult(result, baseline) {
  if (!baseline || !result.passed) {
    return {
      ...result,
      coverageNorm: null,
      lumaNorm: null,
      chromaNorm: null,
      tileReferenceNorm: null,
      paretoScore: null,
    };
  }
  const coverageNorm = normalizedRatio(result.coverageRatio, baseline.coverageRatio);
  const lumaNorm = normalizedRatio(result.lumaDelta, baseline.lumaDelta);
  const chromaNorm = normalizedRatio(result.chromaDelta, baseline.chromaDelta);
  const tileReferenceNorm = normalizedRatio(result.tileReferenceCount, baseline.tileReferenceCount);
  const paretoScore = roundMetric(
    coverageNorm * SCORE_WEIGHTS.coverage +
      lumaNorm * SCORE_WEIGHTS.luma +
      chromaNorm * SCORE_WEIGHTS.chroma +
      tileReferenceNorm * SCORE_WEIGHTS.tileReferences,
  );
  return {
    ...result,
    coverageNorm,
    lumaNorm,
    chromaNorm,
    tileReferenceNorm,
    paretoScore,
  };
}

function normalizedRatio(value, baseline) {
  if (!Number.isFinite(value) || !Number.isFinite(baseline)) return null;
  return roundMetric(value / Math.max(Math.abs(baseline), 0.000001));
}

function isScorable(result) {
  return (
    result.passed &&
    Number.isFinite(result.coverageRatio) &&
    Number.isFinite(result.lumaDelta) &&
    Number.isFinite(result.chromaDelta) &&
    Number.isFinite(result.tileReferenceCount) &&
    Number.isFinite(result.paretoScore)
  );
}

function isParetoDominated(result, candidates) {
  const comparable = candidates.filter((candidate) => candidate.asset === result.asset && isScorable(candidate));
  return comparable.some((candidate) => {
    if (candidate.id === result.id) return false;
    const metrics = ["coverageRatio", "lumaDelta", "chromaDelta", "tileReferenceCount"];
    const noWorse = metrics.every((metric) => candidate[metric] <= result[metric] + 0.000001);
    const better = metrics.some((metric) => candidate[metric] < result[metric] - 0.000001);
    return noWorse && better;
  });
}

function summarizeScenes(results, variantCount) {
  const byAsset = new Map();
  for (const result of results) {
    if (!byAsset.has(result.asset)) byAsset.set(result.asset, []);
    byAsset.get(result.asset).push(result);
  }
  return Array.from(byAsset.entries()).map(([asset, assetResults]) => {
    const scorable = assetResults.filter(isScorable);
    return {
      asset,
      resultCount: assetResults.length,
      complete: assetResults.length === variantCount && assetResults.every((result) => result.passed),
      bestPareto: scorable.slice().sort((left, right) => left.paretoScore - right.paretoScore)[0],
      bestCoverage: scorable.slice().sort((left, right) => left.coverageRatio - right.coverageRatio)[0],
      bestLuma: scorable.slice().sort((left, right) => left.lumaDelta - right.lumaDelta)[0],
      bestChroma: scorable.slice().sort((left, right) => left.chromaDelta - right.chromaDelta)[0],
      lowestCost: scorable.slice().sort((left, right) => left.tileReferenceCount - right.tileReferenceCount)[0],
    };
  });
}

function summarizeVariants(results, sceneCount) {
  const byVariant = new Map();
  for (const result of results.filter(isScorable)) {
    if (!byVariant.has(result.id)) byVariant.set(result.id, []);
    byVariant.get(result.id).push(result);
  }
  return Array.from(byVariant.entries()).map(([id, variantResults]) => ({
    id,
    sceneCount: variantResults.length,
    complete: variantResults.length === sceneCount,
    meanParetoScore: meanMetric(variantResults, "paretoScore"),
    meanCoverageNorm: meanMetric(variantResults, "coverageNorm"),
    meanLumaNorm: meanMetric(variantResults, "lumaNorm"),
    meanChromaNorm: meanMetric(variantResults, "chromaNorm"),
    meanTileReferenceNorm: meanMetric(variantResults, "tileReferenceNorm"),
  }));
}

function meanMetric(results, key) {
  const values = results.map((result) => result[key]).filter(Number.isFinite);
  if (values.length === 0) return null;
  return roundMetric(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function roundMetric(value) {
  if (!Number.isFinite(value)) return null;
  return Number(value.toFixed(6));
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
