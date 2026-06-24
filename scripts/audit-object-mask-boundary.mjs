import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { ASSET_LIBRARY } from "../src/assetLibrary.js";
import { parsePly } from "../src/ply.js";

const MODE = "object-mask-boundary-diagnostic-v1";
const CLEANUP_MODE = "object-boundary-cleanup-candidate-v1";
const DEFAULT_ASSETS = [
  "nerf-lego-alpha-closure-local",
  "plush-semantic-closure-local",
  "nerf-lego-trained-output-local",
];
const PROJECTIONS = [
  { id: "xy", axes: [0, 1] },
  { id: "xz", axes: [0, 2] },
  { id: "yz", axes: [1, 2] },
];

const args = parseArgs(process.argv.slice(2));
const outputDir = String(
  args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-object-mask-boundary",
);
const gridSize = positiveInteger(args.gridSize ?? args["grid-size"] ?? 160);
const maxRadiusCells = positiveInteger(args.maxRadiusCells ?? args["max-radius-cells"] ?? 2);
const footprintScale = positiveNumber(args.footprintScale ?? args["footprint-scale"] ?? 1.0);
const neighborRadiusRatio = positiveNumber(
  args.neighborRadiusRatio ?? args["neighbor-radius-ratio"] ?? 0.015,
);
const maxNeighborSamples = positiveInteger(
  args.maxNeighborSamples ?? args["max-neighbor-samples"] ?? 50_000,
);
const maxCellChecks = positiveInteger(args.maxCellChecks ?? args["max-cell-checks"] ?? 128);
const cleanupNeighborThreshold = unitNumber(
  args.cleanupNeighborThreshold ?? args["cleanup-neighbor-threshold"] ?? 0.65,
);
const cleanupDominantThreshold = unitNumber(
  args.cleanupDominantThreshold ?? args["cleanup-dominant-threshold"] ?? 0.55,
);
const cleanupMinNeighbors = positiveInteger(
  args.cleanupMinNeighbors ?? args["cleanup-min-neighbors"] ?? 3,
);
const failOnMissing = flagEnabled(args.failOnMissing ?? args["fail-on-missing"]);
const assetIds = assetIdList(args.assets ?? args.asset);

const assets = selectAssets(assetIds);
const results = [];
const skipped = [];
for (const asset of assets) {
  const plyPath = publicPath(asset.localPath);
  if (!existsSync(plyPath)) {
    const entry = { assetId: asset.id, plyPath, reason: "missing object-aware PLY" };
    skipped.push(entry);
    if (failOnMissing) {
      results.push({
        assetId: asset.id,
        name: asset.name,
        plyPath,
        passed: false,
        failures: [entry.reason],
      });
    }
    continue;
  }
  const result = auditAsset(asset, {
    plyPath,
    gridSize,
    maxRadiusCells,
    footprintScale,
    neighborRadiusRatio,
    maxNeighborSamples,
    maxCellChecks,
    cleanupNeighborThreshold,
    cleanupDominantThreshold,
    cleanupMinNeighbors,
  });
  results.push(result);
  console.log(
    `object_mask_boundary_asset=${result.passed ? "passed" : "failed"} ` +
      `asset=${JSON.stringify(result.assetId)} gaussians=${result.gaussianCount} ` +
      `objects=${result.objectCount} worstObject=${result.worstObject.objectId} ` +
      `hardMaskGap=${formatNumber(result.worstObject.hardMaskGapScore)} ` +
      `uniqueLoss=${formatNumber(result.worstObject.uniqueCoverageLossRatio)} ` +
      `deletedCoverage=${formatNumber(result.worstObject.deletedSubsetCoverageRatio)} ` +
      `sharedBoundary=${formatNumber(result.worstObject.sharedBoundaryCoverageRatio)} ` +
      `neighborBoundary=${formatNumber(result.worstObject.neighborBoundaryRatio)} ` +
      `cleanupCandidates=${result.cleanup?.candidateGaussianEstimate ?? 0} ` +
      `cleanupTopObject=${result.cleanup?.topObject?.objectId ?? ""}`,
  );
}

const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  options: {
    gridSize,
    maxRadiusCells,
    footprintScale,
    neighborRadiusRatio,
    maxNeighborSamples,
    maxCellChecks,
    cleanupNeighborThreshold,
    cleanupDominantThreshold,
    cleanupMinNeighbors,
    failOnMissing,
  },
  assets: assets.map((asset) => asset.id),
  skipped,
  passed: results.length > 0 && results.every((result) => result.passed),
  results,
};

writeReport(outputDir, summary);
console.log(
  `object_mask_boundary=${summary.passed ? "passed" : "failed"} ` +
    `mode=${JSON.stringify(MODE)} assets=${results.length} skipped=${skipped.length} ` +
    `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
);

if (!summary.passed) {
  process.exitCode = 1;
}

function auditAsset(asset, options) {
  const parsed = parsePly(readArrayBuffer(options.plyPath));
  const points = parsed.points;
  const failures = [];
  if (points.length === 0) failures.push("PLY contains no Gaussian points");
  const objectIds = [...new Set(points.map((point) => point.objectId))].sort((left, right) => left - right);
  if (objectIds.length < 2) failures.push("diagnostic requires at least 2 object_id values");

  const bounds = sceneBounds(points);
  const objectStats = baseObjectStats(points, objectIds);
  const projections = PROJECTIONS.map((projection) =>
    projectionCoverage(points, objectIds, bounds, projection, options),
  );
  const neighbors = neighborBoundaryStats(points, objectIds, bounds, options);
  const perObject = objectIds.map((objectId) =>
    objectDiagnostic({
      objectId,
      totalGaussians: points.length,
      base: objectStats.get(objectId),
      projections,
      neighbor: neighbors.objects.get(objectId),
    }),
  );
  const worstObject =
    perObject.slice().sort((left, right) => right.hardMaskGapScore - left.hardMaskGapScore)[0] ??
    emptyWorstObject();
  const meanHardMaskGapScore = mean(perObject.map((item) => item.hardMaskGapScore));
  const meanUniqueCoverageLossRatio = mean(perObject.map((item) => item.uniqueCoverageLossRatio));
  const meanNeighborBoundaryRatio = mean(perObject.map((item) => item.neighborBoundaryRatio));
  const cleanup = cleanupSummary(perObject);

  return {
    assetId: asset.id,
    name: asset.name,
    plyPath: options.plyPath,
    passed: failures.length === 0,
    failures,
    gaussianCount: points.length,
    objectCount: objectIds.length,
    objectIds,
    bounds,
    projectionIds: PROJECTIONS.map((projection) => projection.id),
    neighborRadius: neighbors.radius,
    neighborSampleCount: neighbors.sampleCount,
    meanHardMaskGapScore,
    meanUniqueCoverageLossRatio,
    meanNeighborBoundaryRatio,
    cleanup,
    worstObject,
    perObject,
  };
}

function objectDiagnostic({ objectId, totalGaussians, base, projections, neighbor }) {
  const projectionRows = projections.map((projection) => projection.objects.get(objectId));
  const deletedSubsetCoverageRatio = mean(
    projectionRows.map((row) => row.deletedSubsetCoverageRatio),
  );
  const uniqueCoverageLossRatio = mean(
    projectionRows.map((row) => row.uniqueCoverageLossRatio),
  );
  const visibleAfterDeleteCoverageRatio = mean(
    projectionRows.map((row) => row.visibleAfterDeleteCoverageRatio),
  );
  const sharedBoundaryCoverageRatio = mean(
    projectionRows.map((row) => row.sharedBoundaryCoverageRatio),
  );
  const uniqueWithinDeletedCoverageRatio = mean(
    projectionRows.map((row) => row.uniqueWithinDeletedCoverageRatio),
  );
  const neighborBoundaryRatio = neighbor?.boundaryRatio ?? 0;
  const neighborMixedRatio = neighbor?.mixedNeighborRatio ?? 0;
  const hardMaskGapScore = clamp01(
    uniqueCoverageLossRatio * 0.45 +
      neighborBoundaryRatio * 0.3 +
      sharedBoundaryCoverageRatio * 0.25,
  );
  const cleanupCandidateRatio = neighbor?.cleanupCandidateRatio ?? 0;
  const cleanupCandidateSamples = neighbor?.cleanupCandidateSamples ?? 0;
  const cleanupCandidateGaussianEstimate = Math.round(cleanupCandidateRatio * (base?.count ?? 0));
  const cleanupPriorityScore = cleanupPriority({
    cleanupCandidateRatio,
    hardMaskGapScore,
    sharedBoundaryCoverageRatio,
    uniqueCoverageLossRatio,
  });
  const cleanupRecommendation = cleanupRecommendationForObject({
    cleanupCandidateRatio,
    cleanupCandidateGaussianEstimate,
    cleanupPriorityScore,
    sharedBoundaryCoverageRatio,
    uniqueCoverageLossRatio,
  });
  return {
    objectId,
    gaussianCount: base?.count ?? 0,
    gaussianShare: roundMetric((base?.count ?? 0) / Math.max(totalGaussians, 1)),
    opacityMean: roundMetric(base?.opacityMean ?? 0),
    scaleMean: roundMetric(base?.scaleMean ?? 0),
    deletedSubsetCoverageRatio,
    uniqueCoverageLossRatio,
    uniqueWithinDeletedCoverageRatio,
    visibleAfterDeleteCoverageRatio,
    sharedBoundaryCoverageRatio,
    neighborBoundaryRatio,
    neighborMixedRatio,
    neighborSamples: neighbor?.samples ?? 0,
    cleanupCandidateRatio,
    cleanupCandidateSamples,
    cleanupCandidateGaussianEstimate,
    cleanupDominantTargetObject: neighbor?.cleanupDominantTargetObject ?? null,
    cleanupDominantTargetSupport: neighbor?.cleanupDominantTargetSupport ?? 0,
    cleanupPriorityScore,
    cleanupRecommendation,
    hardMaskGapScore: roundMetric(hardMaskGapScore),
  };
}

function baseObjectStats(points, objectIds) {
  const stats = new Map(
    objectIds.map((objectId) => [
      objectId,
      {
        count: 0,
        opacitySum: 0,
        scaleSum: 0,
      },
    ]),
  );
  for (const point of points) {
    const row = stats.get(point.objectId);
    if (!row) continue;
    row.count += 1;
    row.opacitySum += Number(point.opacity ?? 1);
    row.scaleSum += largestScale(point);
  }
  for (const row of stats.values()) {
    row.opacityMean = row.count > 0 ? row.opacitySum / row.count : 0;
    row.scaleMean = row.count > 0 ? row.scaleSum / row.count : 0;
  }
  return stats;
}

function projectionCoverage(points, objectIds, bounds, projection, options) {
  const cellCount = options.gridSize * options.gridSize;
  const full = new Uint8Array(cellCount);
  const objectGrids = new Map(objectIds.map((objectId) => [objectId, new Uint8Array(cellCount)]));
  const [axisA, axisB] = projection.axes;
  const spanA = Math.max(bounds.max[axisA] - bounds.min[axisA], 1e-6);
  const spanB = Math.max(bounds.max[axisB] - bounds.min[axisB], 1e-6);

  for (const point of points) {
    const centerA = coordinateToCell(pointCoord(point, axisA), bounds.min[axisA], spanA, options.gridSize);
    const centerB = coordinateToCell(pointCoord(point, axisB), bounds.min[axisB], spanB, options.gridSize);
    const radius = pointRadiusCells(point, { axisA, axisB, spanA, spanB }, options);
    const grid = objectGrids.get(point.objectId);
    if (!grid) continue;
    markDisc({ full, grid, centerA, centerB, radius, gridSize: options.gridSize });
  }

  const cellObjectCounts = new Uint8Array(cellCount);
  for (const grid of objectGrids.values()) {
    for (let index = 0; index < cellCount; index += 1) {
      if (grid[index]) cellObjectCounts[index] += 1;
    }
  }
  const fullCoverageCells = countCells(full);
  const objects = new Map();
  for (const objectId of objectIds) {
    const grid = objectGrids.get(objectId);
    let objectCoverageCells = 0;
    let uniqueCells = 0;
    let sharedCells = 0;
    for (let index = 0; index < cellCount; index += 1) {
      if (!grid[index]) continue;
      objectCoverageCells += 1;
      if (cellObjectCounts[index] === 1) uniqueCells += 1;
      else sharedCells += 1;
    }
    const remainingCoverageCells = Math.max(fullCoverageCells - uniqueCells, 0);
    objects.set(objectId, {
      projection: projection.id,
      fullCoverageCells,
      objectCoverageCells,
      uniqueCells,
      sharedCells,
      deletedSubsetCoverageRatio: ratio(objectCoverageCells, fullCoverageCells),
      uniqueCoverageLossRatio: ratio(uniqueCells, fullCoverageCells),
      uniqueWithinDeletedCoverageRatio: ratio(uniqueCells, objectCoverageCells),
      sharedBoundaryCoverageRatio: ratio(sharedCells, objectCoverageCells),
      visibleAfterDeleteCoverageRatio: ratio(remainingCoverageCells, fullCoverageCells),
    });
  }
  return { id: projection.id, fullCoverageCells, objects };
}

function neighborBoundaryStats(points, objectIds, bounds, options) {
  const diagonal = Math.hypot(
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  );
  const radius = Math.max(diagonal * options.neighborRadiusRatio, 1e-6);
  const cellSize = radius;
  const cells = new Map();
  for (let index = 0; index < points.length; index += 1) {
    const key = neighborCellKey(points[index], cellSize);
    let bucket = cells.get(key);
    if (!bucket) {
      bucket = [];
      cells.set(key, bucket);
    }
    bucket.push(index);
  }

  const perObject = new Map(
    objectIds.map((objectId) => [
      objectId,
      {
        samples: 0,
        boundarySamples: 0,
        cleanupCandidateSamples: 0,
        cleanupLowConfidenceSamples: 0,
        sameNeighbors: 0,
        differentNeighbors: 0,
        cleanupTargetCounts: new Map(),
      },
    ]),
  );
  const sampleStep = Math.max(1, Math.ceil(points.length / options.maxNeighborSamples));
  let sampleCount = 0;
  for (let index = 0; index < points.length; index += sampleStep) {
    sampleCount += 1;
    const point = points[index];
    const row = perObject.get(point.objectId);
    if (!row) continue;
    row.samples += 1;
    const nearby = nearbyIndices({ point, cells, cellSize, maxCellChecks: options.maxCellChecks });
    let hasDifferent = false;
    let sameNeighbors = 0;
    let differentNeighbors = 0;
    const differentByObject = new Map();
    for (const otherIndex of nearby) {
      if (otherIndex === index) continue;
      const other = points[otherIndex];
      if (distance(point, other) > radius) continue;
      if (other.objectId === point.objectId) {
        row.sameNeighbors += 1;
        sameNeighbors += 1;
      } else {
        row.differentNeighbors += 1;
        differentNeighbors += 1;
        differentByObject.set(other.objectId, (differentByObject.get(other.objectId) ?? 0) + 1);
        hasDifferent = true;
      }
    }
    if (hasDifferent) row.boundarySamples += 1;
    const totalNeighbors = sameNeighbors + differentNeighbors;
    if (totalNeighbors >= options.cleanupMinNeighbors && differentNeighbors > 0) {
      const mixedRatio = differentNeighbors / totalNeighbors;
      const dominant = dominantObjectCount(differentByObject);
      const dominantSupport = dominant.count / Math.max(differentNeighbors, 1);
      if (
        mixedRatio >= options.cleanupNeighborThreshold &&
        dominantSupport >= options.cleanupDominantThreshold
      ) {
        row.cleanupCandidateSamples += 1;
        row.cleanupTargetCounts.set(
          dominant.objectId,
          (row.cleanupTargetCounts.get(dominant.objectId) ?? 0) + 1,
        );
      } else if (mixedRatio >= options.cleanupNeighborThreshold) {
        row.cleanupLowConfidenceSamples += 1;
      }
    }
  }

  const objects = new Map();
  for (const [objectId, row] of perObject.entries()) {
    const totalNeighbors = row.sameNeighbors + row.differentNeighbors;
    const dominantTarget = dominantObjectCount(row.cleanupTargetCounts);
    objects.set(objectId, {
      objectId,
      samples: row.samples,
      boundarySamples: row.boundarySamples,
      boundaryRatio: roundMetric(ratio(row.boundarySamples, row.samples)),
      mixedNeighborRatio: roundMetric(ratio(row.differentNeighbors, totalNeighbors)),
      sameNeighbors: row.sameNeighbors,
      differentNeighbors: row.differentNeighbors,
      cleanupMode: CLEANUP_MODE,
      cleanupCandidateSamples: row.cleanupCandidateSamples,
      cleanupLowConfidenceSamples: row.cleanupLowConfidenceSamples,
      cleanupCandidateRatio: roundMetric(ratio(row.cleanupCandidateSamples, row.samples)),
      cleanupDominantTargetObject: dominantTarget.objectId,
      cleanupDominantTargetSupport: roundMetric(
        ratio(dominantTarget.count, row.cleanupCandidateSamples),
      ),
    });
  }
  return { radius: roundMetric(radius), sampleCount, objects };
}

function dominantObjectCount(counts) {
  let objectId = null;
  let count = 0;
  for (const [candidateObjectId, candidateCount] of counts.entries()) {
    if (candidateCount > count) {
      objectId = candidateObjectId;
      count = candidateCount;
    }
  }
  return { objectId, count };
}

function cleanupSummary(perObject) {
  const rows = perObject
    .filter((row) => row.cleanupCandidateGaussianEstimate > 0)
    .slice()
    .sort((left, right) => right.cleanupPriorityScore - left.cleanupPriorityScore);
  const topObject = rows[0] ?? null;
  const candidateGaussianEstimate = rows.reduce(
    (total, row) => total + row.cleanupCandidateGaussianEstimate,
    0,
  );
  return {
    mode: CLEANUP_MODE,
    objectCount: rows.length,
    candidateGaussianEstimate,
    candidateGaussianShare: roundMetric(
      candidateGaussianEstimate /
        Math.max(
          perObject.reduce((total, row) => total + row.gaussianCount, 0),
          1,
        ),
    ),
    meanCandidateRatio: mean(perObject.map((row) => row.cleanupCandidateRatio)),
    topObject: topObject
      ? {
          objectId: topObject.objectId,
          targetObject: topObject.cleanupDominantTargetObject,
          priorityScore: topObject.cleanupPriorityScore,
          candidateGaussianEstimate: topObject.cleanupCandidateGaussianEstimate,
          candidateRatio: topObject.cleanupCandidateRatio,
          recommendation: topObject.cleanupRecommendation,
        }
      : null,
  };
}

function cleanupPriority({
  cleanupCandidateRatio,
  hardMaskGapScore,
  sharedBoundaryCoverageRatio,
  uniqueCoverageLossRatio,
}) {
  return roundMetric(
    clamp01(
      cleanupCandidateRatio * 0.45 +
        hardMaskGapScore * 0.25 +
        sharedBoundaryCoverageRatio * 0.2 +
        Math.max(0, 0.08 - uniqueCoverageLossRatio) * 1.25 * 0.1,
    ),
  );
}

function cleanupRecommendationForObject({
  cleanupCandidateRatio,
  cleanupCandidateGaussianEstimate,
  cleanupPriorityScore,
  sharedBoundaryCoverageRatio,
  uniqueCoverageLossRatio,
}) {
  if (cleanupCandidateGaussianEstimate === 0) return "keep-hard-mask-no-remap-candidate";
  if (uniqueCoverageLossRatio >= 0.08) return "delete-hole-risk-review";
  if (
    cleanupPriorityScore >= 0.35 ||
    (cleanupCandidateRatio >= 0.08 && sharedBoundaryCoverageRatio >= 0.75)
  ) {
    return "boundary-remap-review";
  }
  return "low-priority-boundary-cleanup";
}

function nearbyIndices({ point, cells, cellSize, maxCellChecks }) {
  const base = neighborCell(point, cellSize);
  const indices = [];
  for (let dx = -1; dx <= 1; dx += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dz = -1; dz <= 1; dz += 1) {
        const bucket = cells.get(`${base[0] + dx},${base[1] + dy},${base[2] + dz}`);
        if (!bucket) continue;
        const step = Math.max(1, Math.ceil(bucket.length / maxCellChecks));
        for (let offset = 0; offset < bucket.length; offset += step) {
          indices.push(bucket[offset]);
        }
      }
    }
  }
  return indices;
}

function markDisc({ full, grid, centerA, centerB, radius, gridSize }) {
  for (let dy = -radius; dy <= radius; dy += 1) {
    const y = centerB + dy;
    if (y < 0 || y >= gridSize) continue;
    for (let dx = -radius; dx <= radius; dx += 1) {
      const x = centerA + dx;
      if (x < 0 || x >= gridSize) continue;
      if (dx * dx + dy * dy > radius * radius) continue;
      const index = y * gridSize + x;
      full[index] = 1;
      grid[index] = 1;
    }
  }
}

function pointRadiusCells(point, { axisA, axisB, spanA, spanB }, options) {
  const scaleA = point.scale3?.[axisA] ?? largestScale(point);
  const scaleB = point.scale3?.[axisB] ?? largestScale(point);
  const normalized = Math.max(scaleA / spanA, scaleB / spanB);
  return Math.max(
    0,
    Math.min(options.maxRadiusCells, Math.ceil(normalized * options.gridSize * options.footprintScale)),
  );
}

function sceneBounds(points) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    min[0] = Math.min(min[0], point.x);
    min[1] = Math.min(min[1], point.y);
    min[2] = Math.min(min[2], point.z);
    max[0] = Math.max(max[0], point.x);
    max[1] = Math.max(max[1], point.y);
    max[2] = Math.max(max[2], point.z);
  }
  return {
    min: min.map(roundMetric),
    max: max.map(roundMetric),
  };
}

function countCells(grid) {
  let count = 0;
  for (const value of grid) {
    if (value) count += 1;
  }
  return count;
}

function largestScale(point) {
  if (Array.isArray(point.scale3)) return Math.max(...point.scale3);
  if (Array.isArray(point.scale)) return Math.max(...point.scale);
  return 0;
}

function pointCoord(point, axis) {
  if (axis === 0) return point.x;
  if (axis === 1) return point.y;
  return point.z;
}

function coordinateToCell(value, minValue, span, gridSize) {
  const normalized = (value - minValue) / span;
  return Math.max(0, Math.min(gridSize - 1, Math.floor(normalized * (gridSize - 1))));
}

function neighborCell(point, cellSize) {
  return [
    Math.floor(point.x / cellSize),
    Math.floor(point.y / cellSize),
    Math.floor(point.z / cellSize),
  ];
}

function neighborCellKey(point, cellSize) {
  return neighborCell(point, cellSize).join(",");
}

function distance(left, right) {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function emptyWorstObject() {
  return {
    objectId: null,
    hardMaskGapScore: 0,
    uniqueCoverageLossRatio: 0,
    deletedSubsetCoverageRatio: 0,
    sharedBoundaryCoverageRatio: 0,
    neighborBoundaryRatio: 0,
  };
}

function selectAssets(ids) {
  const candidates = ASSET_LIBRARY.filter(
    (asset) =>
      asset.sourceType === "gaussian" &&
      typeof asset.localPath === "string" &&
      asset.localPath.endsWith(".ply"),
  );
  const selectedIds = ids ?? DEFAULT_ASSETS;
  const byId = new Map(candidates.map((asset) => [asset.id, asset]));
  return selectedIds.map((id) => {
    const asset = byId.get(id);
    if (!asset) throw new Error(`unknown local Gaussian asset: ${id}`);
    return asset;
  });
}

function assetIdList(value) {
  if (value === undefined || value === null || value === false) return null;
  if (value === true) throw new Error("--assets requires a comma-separated asset id list");
  return String(value)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function publicPath(value) {
  const normalized = String(value ?? "");
  if (!normalized.startsWith("/")) return normalized;
  return path.join("public", normalized.slice(1));
}

function readArrayBuffer(filePath) {
  const buffer = readFileSync(filePath);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

function writeReport(outputDirPath, summary) {
  mkdirSync(outputDirPath, { recursive: true });
  writeFileSync(path.join(outputDirPath, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(path.join(outputDirPath, "summary.md"), renderMarkdown(summary));
}

function renderMarkdown(summary) {
  const lines = [
    "# Object Mask Boundary Diagnostic",
    "",
    `- Status: ${summary.passed ? "passed" : "failed"}`,
    `- Mode: ${summary.mode}`,
    `- Generated: ${summary.generatedAt}`,
    `- Grid: ${summary.options.gridSize}x${summary.options.gridSize}`,
    `- Footprint scale: ${summary.options.footprintScale}`,
    `- Neighbor radius ratio: ${summary.options.neighborRadiusRatio}`,
    `- Cleanup mode: ${CLEANUP_MODE}`,
    `- Cleanup thresholds: neighbor=${summary.options.cleanupNeighborThreshold}, dominant=${summary.options.cleanupDominantThreshold}, minNeighbors=${summary.options.cleanupMinNeighbors}`,
    "",
    "This diagnostic estimates where hard `object_id` masking can create visible holes, sparse remnants, or boundary grain after delete/isolate. It is a deterministic PLY-level proxy, not a replacement for browser visual residual screenshots.",
    "",
    "The cleanup candidate layer is read-only. It samples local 3D neighborhoods and flags Gaussian subsets whose nearby support is dominated by another object id; those rows are candidates for assignment cleanup or boundary remap review, not an automatic label rewrite.",
    "",
    "## Assets",
    "",
    "| Asset | Gaussians | Objects | Worst object | Gap score | Unique loss | Deleted coverage | Shared boundary | Neighbor boundary | Cleanup estimate | Cleanup top object | Cleanup target |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ];
  for (const result of summary.results) {
    lines.push(
      `| ${escapeMarkdown(result.assetId)} | ${result.gaussianCount ?? 0} | ${result.objectCount ?? 0} | ${result.worstObject?.objectId ?? ""} | ${formatNumber(result.worstObject?.hardMaskGapScore)} | ${formatNumber(result.worstObject?.uniqueCoverageLossRatio)} | ${formatNumber(result.worstObject?.deletedSubsetCoverageRatio)} | ${formatNumber(result.worstObject?.sharedBoundaryCoverageRatio)} | ${formatNumber(result.worstObject?.neighborBoundaryRatio)} | ${result.cleanup?.candidateGaussianEstimate ?? 0} | ${result.cleanup?.topObject?.objectId ?? ""} | ${result.cleanup?.topObject?.targetObject ?? ""} |`,
    );
  }
  lines.push("", "## Per Object", "");
  for (const result of summary.results) {
    lines.push(`### ${result.assetId}`, "");
    if (result.failures?.length) {
      lines.push(...result.failures.map((failure) => `- Failure: ${failure}`), "");
      continue;
    }
    lines.push(
      "| Object | Gaussians | Share | Gap score | Unique loss | Deleted coverage | Visible after delete | Shared boundary | Neighbor boundary | Cleanup est. | Cleanup ratio | Target | Priority | Recommendation |",
      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    );
    for (const row of result.perObject ?? []) {
      lines.push(
        `| ${row.objectId} | ${row.gaussianCount} | ${formatNumber(row.gaussianShare)} | ${formatNumber(row.hardMaskGapScore)} | ${formatNumber(row.uniqueCoverageLossRatio)} | ${formatNumber(row.deletedSubsetCoverageRatio)} | ${formatNumber(row.visibleAfterDeleteCoverageRatio)} | ${formatNumber(row.sharedBoundaryCoverageRatio)} | ${formatNumber(row.neighborBoundaryRatio)} | ${row.cleanupCandidateGaussianEstimate ?? 0} | ${formatNumber(row.cleanupCandidateRatio)} | ${row.cleanupDominantTargetObject ?? ""} | ${formatNumber(row.cleanupPriorityScore)} | ${escapeMarkdown(row.cleanupRecommendation)} |`,
      );
    }
    lines.push("");
  }
  if (summary.skipped.length > 0) {
    lines.push("## Skipped", "", ...summary.skipped.map((item) => `- ${item.assetId}: ${item.reason} (${item.plyPath})`), "");
  }
  return lines.join("\n");
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

function positiveInteger(value) {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
}

function positiveNumber(value) {
  const parsed = Number.parseFloat(String(value));
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
}

function unitNumber(value) {
  const parsed = Number.parseFloat(String(value));
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.min(1, parsed));
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function ratio(numerator, denominator) {
  return denominator > 0 ? roundMetric(numerator / denominator) : 0;
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return 0;
  return roundMetric(finite.reduce((total, value) => total + value, 0) / finite.length);
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) return 0;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function formatNumber(value) {
  return roundMetric(Number(value ?? 0)).toFixed(6);
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
}
