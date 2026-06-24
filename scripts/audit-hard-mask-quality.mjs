import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const MODE = "hard-mask-quality-chain-v1";
const DEFAULT_BOUNDARY_SUMMARY = "/tmp/objgauss-object-mask-boundary/summary.json";
const DEFAULT_ROUTE_SUMMARIES = [
  "/tmp/objgauss-spark-commercial-route/summary.json",
  "/tmp/objgauss-spark-commercial-route-availability/summary.json",
  "/tmp/objgauss-spark-commercial-route-report/summary.json",
];
const DEFAULT_RESIDUAL_SUMMARIES = [
  "/tmp/objgauss-spark-reconstruct-residual-multiscene/summary.json",
  "/tmp/objgauss-spark-reconstruct-residual/summary.json",
  "/tmp/objgauss-spark-reconstruct-residual-plush/summary.json",
  "/tmp/objgauss-spark-reconstruct-residual-trained/summary.json",
];

const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-hard-mask-quality");
const boundarySummaryPath = String(
  args.boundarySummary ?? args["boundary-summary"] ?? DEFAULT_BOUNDARY_SUMMARY,
);
const routeSummaryPaths = pathList(args.routeSummary ?? args["route-summary"], DEFAULT_ROUTE_SUMMARIES);
const residualSummaryPaths = pathList(
  args.residualSummary ?? args["residual-summary"],
  DEFAULT_RESIDUAL_SUMMARIES,
);
const requireRoute = flagEnabled(args.requireRoute ?? args["require-route"]);
const requireResidual = flagEnabled(args.requireResidual ?? args["require-residual"]);

if (!existsSync(boundarySummaryPath)) {
  throw new Error(
    `missing boundary summary: ${boundarySummaryPath}; run npm run audit:object-mask-boundary first`,
  );
}

const boundarySummary = readJson(boundarySummaryPath);
const routeSummary = firstExistingJson(routeSummaryPaths);
const residualSummaries = residualSummaryPaths.filter(existsSync).map(readJson);
const routeByAsset = buildRouteIndex(routeSummary?.payload);
const residualByAsset = buildResidualIndex(residualSummaries);
const rows = (boundarySummary.results ?? []).map((boundary) =>
  qualityRow({
    boundary,
    route: routeByAsset.get(boundary.assetId) ?? null,
    residual: residualByAsset.get(boundary.assetId) ?? null,
  }),
);

const missingRouteRows = rows.filter((row) => row.routeStatus === "missing");
const missingResidualRows = rows.filter((row) => row.residualStatus === "missing");
const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  inputs: {
    boundarySummary: boundarySummaryPath,
    routeSummary: routeSummary?.path ?? "",
    residualSummaries: residualSummaries.map((summaryItem) => summaryItem.__path),
  },
  requirements: {
    requireRoute,
    requireResidual,
  },
  passed:
    boundarySummary.passed !== false &&
    (!requireRoute || missingRouteRows.length === 0) &&
    (!requireResidual || missingResidualRows.length === 0),
  rows,
  missing: {
    routeAssets: missingRouteRows.map((row) => row.assetId),
    residualAssets: missingResidualRows.map((row) => row.assetId),
  },
};

writeReport(outputDir, summary);
for (const row of rows) {
  console.log(
    `hard_mask_quality_asset=${row.passed ? "passed" : "partial"} ` +
      `asset=${JSON.stringify(row.assetId)} evidence=${JSON.stringify(row.evidenceLevel)} ` +
      `route=${JSON.stringify(row.routeStatus)}:${JSON.stringify(row.routeKind)} ` +
      `deleted=${row.deletedObjectId ?? "unknown"}:${row.deletedGaussians ?? 0} ` +
      `gap=${formatNumber(row.boundary.hardMaskGapScore)} ` +
      `unique=${formatNumber(row.boundary.uniqueCoverageLossRatio)} ` +
      `shared=${formatNumber(row.boundary.sharedBoundaryCoverageRatio)} ` +
      `neighbor=${formatNumber(row.boundary.neighborBoundaryRatio)} ` +
      `residual=${JSON.stringify(row.residualStatus)}:${formatNumber(row.residual.coverageRatio)}:${formatNumber(row.residual.lumaDelta)}:${formatNumber(row.residual.chromaDelta)} ` +
      `interpretation=${JSON.stringify(row.interpretation)}`,
  );
}
console.log(
  `hard_mask_quality=${summary.passed ? "passed" : "partial"} ` +
    `mode=${JSON.stringify(MODE)} rows=${rows.length} ` +
    `routeSummary=${JSON.stringify(summary.inputs.routeSummary || "missing")} ` +
    `residualSummaries=${summary.inputs.residualSummaries.length} ` +
    `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
);

if (!summary.passed) {
  process.exitCode = 1;
}

function qualityRow({ boundary, route, residual }) {
  const routeHidden = route?.hiddenGaussians ?? null;
  const routeDeletedObject = inferDeletedObject(boundary, routeHidden);
  const boundaryObject = routeDeletedObject ?? boundary.worstObject ?? {};
  const residualStatus = residual ? (residual.passed ? "passed" : "failed") : "missing";
  const routeStatus = route ? "passed" : "missing";
  const evidenceLevel = evidence(routeStatus, residualStatus);
  const interpretation = interpret({ boundaryObject, residual, route });
  return {
    assetId: boundary.assetId,
    name: boundary.name,
    passed: routeStatus !== "missing" && residualStatus !== "missing" ? residualStatus === "passed" : false,
    evidenceLevel,
    routeStatus,
    routeKind: route?.kind ?? "",
    routeSource: route?.source ?? "",
    routeBoundary: route?.boundary ?? "",
    deletedObjectId: routeDeletedObject?.objectId ?? null,
    deletedGaussians: routeHidden,
    baseGaussians: route?.baseGaussians ?? boundary.gaussianCount,
    visibleGaussians: route?.visibleGaussians ?? null,
    boundary: {
      selectedObjectId: boundaryObject.objectId ?? boundary.worstObject?.objectId ?? null,
      worstObjectId: boundary.worstObject?.objectId ?? null,
      hardMaskGapScore: roundMetric(boundaryObject.hardMaskGapScore ?? 0),
      uniqueCoverageLossRatio: roundMetric(boundaryObject.uniqueCoverageLossRatio ?? 0),
      deletedSubsetCoverageRatio: roundMetric(boundaryObject.deletedSubsetCoverageRatio ?? 0),
      sharedBoundaryCoverageRatio: roundMetric(boundaryObject.sharedBoundaryCoverageRatio ?? 0),
      neighborBoundaryRatio: roundMetric(boundaryObject.neighborBoundaryRatio ?? 0),
      neighborMixedRatio: roundMetric(boundaryObject.neighborMixedRatio ?? 0),
    },
    residualStatus,
    residual: {
      coverageRatio: roundMetric(residual?.residual?.coverageRatio ?? 0),
      lumaDelta: roundMetric(residual?.residual?.lumaDelta ?? 0),
      chromaDelta: roundMetric(residual?.residual?.chromaDelta ?? 0),
      source: residual?.source ?? "",
      screenshotPath: residual?.screenshotPath ?? "",
    },
    interpretation,
  };
}

function inferDeletedObject(boundary, hiddenGaussians) {
  if (!Number.isFinite(hiddenGaussians) || hiddenGaussians <= 0) return null;
  const candidates = boundary.perObject ?? [];
  let best = null;
  for (const row of candidates) {
    const delta = Math.abs(Number(row.gaussianCount ?? 0) - hiddenGaussians);
    if (!best || delta < best.delta) best = { row, delta };
  }
  if (!best) return null;
  const tolerance = Math.max(2, Math.ceil(hiddenGaussians * 0.002));
  return best.delta <= tolerance ? best.row : null;
}

function interpret({ boundaryObject, residual, route }) {
  const uniqueLoss = Number(boundaryObject.uniqueCoverageLossRatio ?? 0);
  const sharedBoundary = Number(boundaryObject.sharedBoundaryCoverageRatio ?? 0);
  const neighborBoundary = Number(boundaryObject.neighborBoundaryRatio ?? 0);
  const coverageRatio = Number(residual?.residual?.coverageRatio ?? 0);
  const lumaDelta = Number(residual?.residual?.lumaDelta ?? 0);
  if (!route) return "missing-route-evidence";
  if (!residual) return "missing-browser-residual";
  if (coverageRatio > 2 || lumaDelta > 0.08) return "browser-residual-dominant";
  if (uniqueLoss >= 0.08) return "coverage-hole-risk";
  if (sharedBoundary >= 0.8 || neighborBoundary >= 0.5) return "boundary-mixing-dominant";
  return "low-hard-mask-risk";
}

function evidence(routeStatus, residualStatus) {
  if (routeStatus !== "missing" && residualStatus !== "missing") return "boundary+route+residual";
  if (routeStatus !== "missing") return "boundary+route";
  if (residualStatus !== "missing") return "boundary+residual";
  return "boundary-only";
}

function buildRouteIndex(routeSummary) {
  const map = new Map();
  if (!routeSummary?.routes) return map;
  for (const route of routeSummary.routes.native ?? []) {
    const parsed = parseObjectMask(route.objectMask);
    map.set(route.asset, {
      kind: "native",
      source: route.source,
      boundary: route.boundary,
      visibleGaussians: Number(route.visibleGaussians),
      baseGaussians: Number(route.baseGaussians),
      hiddenGaussians: parsed.hiddenGaussians ?? Number(route.baseGaussians) - Number(route.visibleGaussians),
      objectMask: route.objectMask,
      screenshot: route.screenshot,
    });
  }
  for (const route of routeSummary.routes.trained ?? []) {
    const parsed = parseObjectMask(route.objectMask);
    map.set(route.asset, {
      kind: "trained",
      source: route.spark?.join("/") ?? "",
      boundary: route.delete?.join("/") ?? "",
      visibleGaussians: Number(route.visibleGaussians),
      baseGaussians: Number(route.baseGaussians),
      hiddenGaussians: parsed.hiddenGaussians ?? Number(route.baseGaussians) - Number(route.visibleGaussians),
      objectMask: route.objectMask,
      screenshot: route.screenshot,
    });
  }
  return map;
}

function buildResidualIndex(summaries) {
  const map = new Map();
  for (const summary of summaries) {
    for (const result of summary.results ?? []) {
      map.set(result.assetId, {
        ...result,
        source: summary.__path,
      });
    }
  }
  return map;
}

function parseObjectMask(value) {
  const match = String(value ?? "").match(/:(\d+)\/(\d+):(\d+)$/);
  if (!match) return {};
  return {
    visibleGaussians: Number(match[1]),
    hiddenGaussians: Number(match[2]),
    updates: Number(match[3]),
  };
}

function firstExistingJson(paths) {
  for (const summaryPath of paths) {
    if (!existsSync(summaryPath)) continue;
    return { path: summaryPath, payload: readJson(summaryPath) };
  }
  return null;
}

function readJson(summaryPath) {
  const payload = JSON.parse(readFileSync(summaryPath, "utf8"));
  Object.defineProperty(payload, "__path", {
    enumerable: false,
    value: summaryPath,
  });
  return payload;
}

function writeReport(outputDirPath, summary) {
  mkdirSync(outputDirPath, { recursive: true });
  writeFileSync(path.join(outputDirPath, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(path.join(outputDirPath, "summary.md"), renderMarkdown(summary));
}

function renderMarkdown(summary) {
  const lines = [
    "# Hard Mask Quality Chain",
    "",
    `- Status: ${summary.passed ? "passed" : "partial"}`,
    `- Mode: ${summary.mode}`,
    `- Generated: ${summary.generatedAt}`,
    `- Boundary summary: ${summary.inputs.boundarySummary}`,
    `- Spark route summary: ${summary.inputs.routeSummary || "missing"}`,
    `- Residual summaries: ${summary.inputs.residualSummaries.length}`,
    "",
    "This report joins the PLY-level hard-mask boundary diagnostic with Spark route evidence and browser visual residual artifacts. It explains whether source/original grain is more likely from hard object boundaries, coverage holes, or the Spark reconstruction / browser path.",
    "",
    "| Asset | Evidence | Route | Deleted object | Gap | Unique loss | Shared boundary | Neighbor boundary | Coverage ratio | Luma delta | Chroma delta | Interpretation |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ];
  for (const row of summary.rows) {
    lines.push(
      `| ${escapeMarkdown(row.assetId)} | ${escapeMarkdown(row.evidenceLevel)} | ${escapeMarkdown(row.routeKind || row.routeStatus)} | ${row.deletedObjectId ?? ""} | ${formatNumber(row.boundary.hardMaskGapScore)} | ${formatNumber(row.boundary.uniqueCoverageLossRatio)} | ${formatNumber(row.boundary.sharedBoundaryCoverageRatio)} | ${formatNumber(row.boundary.neighborBoundaryRatio)} | ${formatNumber(row.residual.coverageRatio)} | ${formatNumber(row.residual.lumaDelta)} | ${formatNumber(row.residual.chromaDelta)} | ${escapeMarkdown(row.interpretation)} |`,
    );
  }
  if (summary.missing.routeAssets.length > 0 || summary.missing.residualAssets.length > 0) {
    lines.push(
      "",
      "## Missing Evidence",
      "",
      `- Missing route assets: ${summary.missing.routeAssets.join(", ") || "none"}`,
      `- Missing residual assets: ${summary.missing.residualAssets.join(", ") || "none"}`,
    );
  }
  lines.push(
    "",
    "Interpretation notes:",
    "",
    "- `boundary-mixing-dominant`: browser route is available and PLY coverage holes are low, but shared / neighbor object boundaries are high.",
    "- `coverage-hole-risk`: the deleted object owns enough unique projected coverage that hard masking can create visible holes.",
    "- `browser-residual-dominant`: Spark reconstruction residual is large enough that source mismatch may dominate the visual issue.",
    "",
  );
  return lines.join("\n");
}

function pathList(value, defaults) {
  if (value === undefined || value === null || value === false) return defaults;
  if (value === true) throw new Error("summary path flag requires a comma-separated path list");
  return String(value)
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

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(Number(value))) return 0;
  const scale = 10 ** digits;
  return Math.round(Number(value) * scale) / scale;
}

function formatNumber(value) {
  return roundMetric(value).toFixed(6);
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
}
