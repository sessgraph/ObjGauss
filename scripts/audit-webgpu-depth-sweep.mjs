import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { normalizeWebGpuPixelDepthBinCount } from "../src/webgpuDepthTuning.js";

const DEFAULT_PORT = 5395;
const DEFAULT_ASSETS = ["nerf-lego-alpha-closure-local"];
const DEFAULT_BINS = [4, 8, 12, 16];
const DEFAULT_FOOTPRINT_SCALE = 2.2;
const DEFAULT_MAX_ANISOTROPY = 4;
const BASELINE_BINS = 8;
const SCORE_WEIGHTS = {
  coverage: 0.35,
  luma: 0.25,
  chroma: 0.25,
  tileReferences: 0.15,
};

const args = parseArgs(process.argv.slice(2));
const assets = parseAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const bins = parseBins(args.bins ?? args["depth-bins"] ?? DEFAULT_BINS.join(","));
if (bins.length === 0) {
  throw new Error("depth sweep requires at least one bin count in the 4-16 range");
}
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const shouldStartServer = !(args.url || args.noServer || args["no-server"]);
const webGpuFlags = String(args.webgpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const headed = !flagEnabled(args.headless);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const outputDir = optionalString(args.outputDir ?? args["output-dir"]);
const webGpuViewportSize = optionalPositiveInteger(
  args.webGpuViewportSize ?? args["webgpu-viewport-size"],
);
const footprintScale =
  optionalFiniteNumber(args.webGpuFootprintScale ?? args["webgpu-footprint-scale"]) ??
  DEFAULT_FOOTPRINT_SCALE;
const maxAnisotropy =
  optionalFiniteNumber(args.webGpuCovarianceMaxAnisotropy ?? args["webgpu-covariance-max-anisotropy"]) ??
  DEFAULT_MAX_ANISOTROPY;
const webGpuDepthAlphaMode = optionalString(
  args.webGpuDepthAlphaMode ?? args["webgpu-depth-alpha-mode"],
);
const webGpuCameraMode = optionalString(args.webGpuCameraMode ?? args["webgpu-camera-mode"]);
let server = null;

try {
  if (shouldStartServer) {
    if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; run `npm run build` before depth sweep");
    }
    server = startPreviewServer(port);
    await waitForApp(baseUrl);
  }

  const results = [];
  let stoppedEarly = false;
  sweep:
  for (const asset of assets) {
    for (const depthBins of bins) {
      const result = await runDepthVariant({
        asset,
        baseUrl,
        depthBins,
        webGpuFlags,
        headed,
        webGpuViewportSize,
        footprintScale,
        maxAnisotropy,
        webGpuDepthAlphaMode,
        webGpuCameraMode,
      });
      process.stdout.write(result.output);
      process.stderr.write(result.errorOutput);
      const metrics = parseVariantMetrics(result.output);
      results.push({
        asset,
        id: `bins-${depthBins}`,
        requestedDepthBins: depthBins,
        ...metrics,
        passed: result.exitCode === 0,
      });
      console.log(
        `webgpu_depth_sweep_variant=${JSON.stringify(`bins-${depthBins}`)} ` +
          `asset=${JSON.stringify(asset)} passed=${result.exitCode === 0} ` +
          `requestedBins=${depthBins} actualBins=${metrics.depthBinCount ?? "unknown"} ` +
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
      `webgpu_depth_sweep_rank asset=${JSON.stringify(result.asset)} ` +
        `variant=${JSON.stringify(result.id)} passed=${result.passed} ` +
        `paretoScore=${result.paretoScore ?? "unknown"} ` +
        `dominated=${result.paretoDominated ?? "unknown"} ` +
        `coverageNorm=${result.coverageNorm ?? "unknown"} ` +
        `lumaNorm=${result.lumaNorm ?? "unknown"} ` +
        `chromaNorm=${result.chromaNorm ?? "unknown"} ` +
        `tileReferenceNorm=${result.tileReferenceNorm ?? "unknown"}`,
    );
  }

  const sceneSummaries = summarizeScenes(rankedResults, bins.length);
  for (const summary of sceneSummaries) {
    console.log(
      `webgpu_depth_sweep_scene=${summary.complete ? "passed" : "partial"} ` +
        `asset=${JSON.stringify(summary.asset)} variants=${summary.resultCount}/${bins.length} ` +
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
      `webgpu_depth_sweep_variant_summary=${summary.complete ? "passed" : "partial"} ` +
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
  const suitePassed = !stoppedEarly && sceneSummaries.every((summary) => summary.complete);
  const summary = buildSummary({
    suitePassed,
    assets,
    bins,
    baseUrl,
    footprintScale,
    maxAnisotropy,
    rankedResults,
    sceneSummaries,
    variantSummaries,
    bestPareto,
    bestCoverage,
    bestMeanPareto,
  });
  if (outputDir) {
    writeReportFiles(outputDir, summary);
    console.log(
      `webgpu_depth_sweep_report=written outputDir=${JSON.stringify(outputDir)} ` +
        `summaryJson=${JSON.stringify(`${outputDir}/summary.json`)} ` +
        `summaryMd=${JSON.stringify(`${outputDir}/summary.md`)}`,
    );
  }
  const overallStatus = summary.passed ? "passed" : suitePassed ? "failed" : "partial";
  console.log(
    `webgpu_depth_sweep=${overallStatus} ` +
      `assets=${JSON.stringify(assets)} scenes=${assets.length} bins=${JSON.stringify(bins)} ` +
      `footprintScale=${footprintScale} maxAnisotropy=${maxAnisotropy} ` +
      `webGpuDepthAlphaMode=${JSON.stringify(webGpuDepthAlphaMode ?? "default")} ` +
      `webGpuCameraMode=${JSON.stringify(webGpuCameraMode ?? "default")} ` +
      `scoreWeights=${JSON.stringify(SCORE_WEIGHTS)} ` +
      `bestMeanParetoVariant=${JSON.stringify(bestMeanPareto?.id ?? "none")}:${bestMeanPareto?.meanParetoScore ?? "unknown"} ` +
      `bestPareto=${JSON.stringify(bestPareto?.asset ?? "none")}/${JSON.stringify(bestPareto?.id ?? "none")}:${bestPareto?.paretoScore ?? "unknown"} ` +
      `bestCoverage=${JSON.stringify(bestCoverage?.asset ?? "none")}/${JSON.stringify(bestCoverage?.id ?? "none")}:${bestCoverage?.coverageRatio ?? "unknown"} ` +
      `url=${baseUrl}`,
  );
} finally {
  if (server) stopPreviewServer(server);
}

async function runDepthVariant({
  asset,
  baseUrl,
  depthBins,
  webGpuFlags,
  headed,
  webGpuViewportSize,
  footprintScale,
  maxAnisotropy,
  webGpuDepthAlphaMode,
  webGpuCameraMode,
}) {
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
    String(footprintScale),
    "--webgpu-covariance-max-anisotropy",
    String(maxAnisotropy),
    "--webgpu-depth-bins",
    String(depthBins),
  ];
  if (!headed) commandArgs.push("--headless");
  if (webGpuViewportSize) {
    commandArgs.push("--webgpu-viewport-size", String(webGpuViewportSize));
  }
  if (webGpuDepthAlphaMode) {
    commandArgs.push("--webgpu-depth-alpha-mode", webGpuDepthAlphaMode);
  }
  if (webGpuCameraMode) {
    commandArgs.push("--webgpu-camera-mode", webGpuCameraMode);
  }
  return runProcess(process.execPath, commandArgs);
}

function parseVariantMetrics(output) {
  const residual = output.match(
    /visualResidual="spark-edit-visual-residual-v1":([0-9.]+)\/([0-9.]+):([0-9.]+):([0-9.]+):([0-9.]+)/,
  );
  const pixelDepth = output.match(
    /pixelDepthSort="([^"]+)":"([^"]+)":"([^"]+)":([0-9.]+)\/([0-9.]+):([0-9]+)/,
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
    depthSortMode: pixelDepth?.[1] ?? "",
    depthTuningMode: pixelDepth?.[2] ?? "",
    depthAlphaMode: pixelDepth?.[3] ?? "",
    depthGateStrength: pixelDepth ? Number(pixelDepth[4]) : null,
    depthGateFloor: pixelDepth ? Number(pixelDepth[5]) : null,
    depthBinCount: pixelDepth ? Number(pixelDepth[6]) : null,
    coverageMode: pixelCoverage?.[1] ?? "",
    coverageTuningMode: pixelCoverage?.[2] ?? "",
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

function parseBins(value) {
  const uniqueBins = new Set();
  for (const entry of String(value ?? "")
    .split(",")
    .map((candidate) => candidate.trim())
    .filter(Boolean)) {
    uniqueBins.add(normalizeWebGpuPixelDepthBinCount(entry));
  }
  return Array.from(uniqueBins).sort((left, right) => left - right);
}

function rankResults(results) {
  const byAsset = new Map();
  for (const result of results) {
    if (!byAsset.has(result.asset)) byAsset.set(result.asset, []);
    byAsset.get(result.asset).push(result);
  }
  const ranked = [];
  for (const assetResults of byAsset.values()) {
    const scorable = assetResults.filter(isResultComplete);
    const baseline =
      scorable.find((result) => result.depthBinCount === BASELINE_BINS) ??
      scorable.find((result) => result.requestedDepthBins === BASELINE_BINS) ??
      scorable[0];
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
  if (!baseline || !isResultComplete(result)) {
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

function isResultComplete(result) {
  return (
    result.passed &&
    Number.isFinite(result.coverageRatio) &&
    Number.isFinite(result.lumaDelta) &&
    Number.isFinite(result.chromaDelta) &&
    Number.isFinite(result.tileReferenceCount) &&
    Number.isFinite(result.depthBinCount) &&
    result.depthBinCount === result.requestedDepthBins
  );
}

function isScorable(result) {
  return isResultComplete(result) && Number.isFinite(result.paretoScore);
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

function buildSummary({
  suitePassed,
  assets,
  bins,
  baseUrl,
  footprintScale,
  maxAnisotropy,
  rankedResults,
  sceneSummaries,
  variantSummaries,
  bestPareto,
  bestCoverage,
  bestMeanPareto,
}) {
  return {
    mode: "webgpu-depth-bin-sweep-v1",
    generatedAt: new Date().toISOString(),
    passed: suitePassed,
    suitePassed,
    url: baseUrl,
    assets,
    bins,
    baselineBins: BASELINE_BINS,
    footprintScale,
    maxAnisotropy,
    scoreWeights: SCORE_WEIGHTS,
    bestMeanParetoVariant: bestMeanPareto
      ? { id: bestMeanPareto.id, meanParetoScore: bestMeanPareto.meanParetoScore }
      : null,
    bestPareto: bestPareto
      ? { asset: bestPareto.asset, id: bestPareto.id, paretoScore: bestPareto.paretoScore }
      : null,
    bestCoverage: bestCoverage
      ? { asset: bestCoverage.asset, id: bestCoverage.id, coverageRatio: bestCoverage.coverageRatio }
      : null,
    sceneSummaries: sceneSummaries.map(summarizeSceneForReport),
    variantSummaries,
    results: rankedResults,
  };
}

function summarizeSceneForReport(summary) {
  return {
    asset: summary.asset,
    resultCount: summary.resultCount,
    complete: summary.complete,
    bestPareto: summarizeResult(summary.bestPareto, ["id", "paretoScore"]),
    bestCoverage: summarizeResult(summary.bestCoverage, ["id", "coverageRatio"]),
    bestLuma: summarizeResult(summary.bestLuma, ["id", "lumaDelta"]),
    bestChroma: summarizeResult(summary.bestChroma, ["id", "chromaDelta"]),
    lowestCost: summarizeResult(summary.lowestCost, ["id", "tileReferenceCount"]),
  };
}

function summarizeResult(result, keys) {
  if (!result) return null;
  return Object.fromEntries(keys.map((key) => [key, result[key]]));
}

function writeReportFiles(directory, summary) {
  mkdirSync(directory, { recursive: true });
  writeFileSync(`${directory}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`, "utf-8");
  writeFileSync(`${directory}/summary.md`, renderMarkdown(summary), "utf-8");
}

function renderMarkdown(summary) {
  const lines = [];
  lines.push("# WebGPU Depth-Bin Sweep");
  lines.push("");
  lines.push(`- Status: ${summary.passed ? "passed" : "partial"}`);
  lines.push(`- URL: ${summary.url}`);
  lines.push(`- Assets: ${summary.assets.join(", ")}`);
  lines.push(`- Depth bins: ${summary.bins.join(", ")}; baseline=${summary.baselineBins}`);
  lines.push(`- Fixed coverage tuning: footprint=${summary.footprintScale}, maxAnisotropy=${summary.maxAnisotropy}`);
  lines.push(`- Weights: coverage=${SCORE_WEIGHTS.coverage}, luma=${SCORE_WEIGHTS.luma}, chroma=${SCORE_WEIGHTS.chroma}, tileReferences=${SCORE_WEIGHTS.tileReferences}`);
  lines.push(`- Best mean Pareto variant: ${summary.bestMeanParetoVariant?.id ?? "none"} (${summary.bestMeanParetoVariant?.meanParetoScore ?? "unknown"})`);
  lines.push("");
  lines.push("## Variant Summary");
  lines.push("");
  lines.push("| Variant | Scenes | Mean score | Coverage norm | Luma norm | Chroma norm | Tile refs norm |");
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const row of summary.variantSummaries) {
    lines.push(
      `| ${escapeMarkdown(row.id)} | ${row.sceneCount} | ${formatReportValue(row.meanParetoScore)} | ${formatReportValue(row.meanCoverageNorm)} | ${formatReportValue(row.meanLumaNorm)} | ${formatReportValue(row.meanChromaNorm)} | ${formatReportValue(row.meanTileReferenceNorm)} |`,
    );
  }
  lines.push("");
  lines.push("## Scene Summary");
  lines.push("");
  lines.push("| Asset | Complete | Best Pareto | Best coverage | Best luma | Best chroma | Lowest cost |");
  lines.push("| --- | --- | --- | --- | --- | --- | --- |");
  for (const row of summary.sceneSummaries) {
    lines.push(
      `| ${escapeMarkdown(row.asset)} | ${row.complete ? "yes" : "no"} | ${formatWinner(row.bestPareto, "paretoScore")} | ${formatWinner(row.bestCoverage, "coverageRatio")} | ${formatWinner(row.bestLuma, "lumaDelta")} | ${formatWinner(row.bestChroma, "chromaDelta")} | ${formatWinner(row.lowestCost, "tileReferenceCount")} |`,
    );
  }
  lines.push("");
  lines.push("## Rows");
  lines.push("");
  lines.push("| Asset | Variant | Bins | Passed | Score | Coverage ratio | Luma delta | Chroma delta | Tile refs |");
  lines.push("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |");
  for (const row of summary.results) {
    lines.push(
      `| ${escapeMarkdown(row.asset)} | ${escapeMarkdown(row.id)} | ${row.depthBinCount ?? "unknown"} | ${row.passed ? "yes" : "no"} | ${formatReportValue(row.paretoScore)} | ${formatReportValue(row.coverageRatio)} | ${formatReportValue(row.lumaDelta)} | ${formatReportValue(row.chromaDelta)} | ${formatReportValue(row.tileReferenceCount)} |`,
    );
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function formatWinner(value, metric) {
  if (!value) return "none";
  return `${value.id}:${formatReportValue(value[metric])}`;
}

function formatReportValue(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "unknown";
  return String(value);
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
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

function optionalFiniteNumber(value) {
  if (value === undefined || value === null || value === true || value === false || value === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const text = String(value).trim();
  return text ? text : undefined;
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
