import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { ASSET_LIBRARY } from "../src/assetLibrary.js";
import { parsePly } from "../src/ply.js";

const MODE = "object-boundary-remap-preview-v1";
const DEFAULT_ASSETS = ["nerf-lego-alpha-closure-local"];
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-object-boundary-remap-preview";
const TYPE_INFO = {
  char: { size: 1, getter: "getInt8", setter: "setInt8", integer: true },
  int8: { size: 1, getter: "getInt8", setter: "setInt8", integer: true },
  uchar: { size: 1, getter: "getUint8", setter: "setUint8", integer: true },
  uint8: { size: 1, getter: "getUint8", setter: "setUint8", integer: true },
  short: { size: 2, getter: "getInt16", setter: "setInt16", integer: true },
  int16: { size: 2, getter: "getInt16", setter: "setInt16", integer: true },
  ushort: { size: 2, getter: "getUint16", setter: "setUint16", integer: true },
  uint16: { size: 2, getter: "getUint16", setter: "setUint16", integer: true },
  int: { size: 4, getter: "getInt32", setter: "setInt32", integer: true },
  int32: { size: 4, getter: "getInt32", setter: "setInt32", integer: true },
  uint: { size: 4, getter: "getUint32", setter: "setUint32", integer: true },
  uint32: { size: 4, getter: "getUint32", setter: "setUint32", integer: true },
  float: { size: 4, getter: "getFloat32", setter: "setFloat32", integer: false },
  float32: { size: 4, getter: "getFloat32", setter: "setFloat32", integer: false },
  double: { size: 8, getter: "getFloat64", setter: "setFloat64", integer: false },
  float64: { size: 8, getter: "getFloat64", setter: "setFloat64", integer: false },
};

const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const assetIds = assetIdList(args.assets ?? args.asset) ?? DEFAULT_ASSETS;
const neighborRadiusRatio = positiveNumber(
  args.neighborRadiusRatio ?? args["neighbor-radius-ratio"] ?? 0.015,
);
const maxRemapSamples = positiveInteger(
  args.maxRemapSamples ?? args["max-remap-samples"] ?? 120_000,
);
const maxCellChecks = positiveInteger(args.maxCellChecks ?? args["max-cell-checks"] ?? 128);
const policyPath = optionalString(args.policy ?? args["policy-path"] ?? args.remapPolicy ?? args["remap-policy"]);
const explicitAllowTargets = parseAllowTargets(
  args.allowTarget ?? args["allow-target"] ?? args.allowTargets ?? args["allow-targets"],
);
const cleanupNeighborThreshold = unitNumber(
  args.cleanupNeighborThreshold ?? args["cleanup-neighbor-threshold"] ?? 0.65,
);
const cleanupDominantThreshold = unitNumber(
  args.cleanupDominantThreshold ?? args["cleanup-dominant-threshold"] ?? 0.55,
);
const cleanupMinNeighbors = positiveInteger(
  args.cleanupMinNeighbors ?? args["cleanup-min-neighbors"] ?? 3,
);
const writePreviewPly = !flagEnabled(args.dryRun ?? args["dry-run"]);
const decisionPolicy = loadDecisionPolicy({ policyPath, explicitAllowTargets });

const assets = selectAssets(assetIds);
const results = [];
const skipped = [];
mkdirSync(outputDir, { recursive: true });

for (const asset of assets) {
  const plyPath = publicPath(asset.localPath);
  if (!existsSync(plyPath)) {
    skipped.push({ assetId: asset.id, plyPath, reason: "missing object-aware PLY" });
    continue;
  }

  const input = readFileSync(plyPath);
  const arrayBuffer = input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength);
  const header = readPlyHeader(arrayBuffer);
  const objectIdLayout = vertexPropertyLayout(header, "object_id");
  const cloud = parsePly(arrayBuffer);
  const candidates = remapCandidates(cloud.points, {
    neighborRadiusRatio,
    maxRemapSamples,
    maxCellChecks,
    cleanupNeighborThreshold,
    cleanupDominantThreshold,
    cleanupMinNeighbors,
  });
  const policyGateResult = decisionPolicy
    ? applyDecisionPolicy({
        policy: decisionPolicy,
        assetId: asset.id,
        remaps: candidates.remaps,
      })
    : null;
  const policyGate = policyGateResult?.summary ?? null;
  const appliedRemaps = policyGateResult ? policyGateResult.remaps : candidates.remaps;
  const appliedRows = summarizeRemaps(appliedRemaps);
  const outputPly = path.join(outputDir, `${asset.id}.remap-preview.ply`);
  if (writePreviewPly) {
    const patched = patchObjectIds({
      input,
      header,
      objectIdLayout,
      remaps: appliedRemaps,
    });
    writeFileSync(outputPly, patched);
  }

  const result = {
    assetId: asset.id,
    name: asset.name,
    sourcePly: plyPath,
    outputPly: writePreviewPly ? outputPly : null,
    passed: true,
    mode: MODE,
    format: header.format,
    gaussianCount: cloud.points.length,
    objectCount: new Set(cloud.points.map((point) => point.objectId)).size,
    sampleStep: candidates.sampleStep,
    sampledGaussians: candidates.sampledGaussians,
    rawCandidateRemapGaussians: candidates.remaps.length,
    rawCandidateRemapShare: roundMetric(candidates.remaps.length / Math.max(cloud.points.length, 1)),
    rawEstimatedFullRemapGaussians: candidates.estimatedFullRemapGaussians,
    rawEstimatedFullRemapShare: roundMetric(
      candidates.estimatedFullRemapGaussians / Math.max(cloud.points.length, 1),
    ),
    remappedGaussians: appliedRemaps.length,
    remappedGaussianShare: roundMetric(appliedRemaps.length / Math.max(cloud.points.length, 1)),
    estimatedFullRemapGaussians: Math.min(cloud.points.length, appliedRemaps.length * candidates.sampleStep),
    estimatedFullRemapShare: roundMetric(
      Math.min(cloud.points.length, appliedRemaps.length * candidates.sampleStep) /
        Math.max(cloud.points.length, 1),
    ),
    lowConfidenceSamples: candidates.lowConfidenceSamples,
    policyGate,
    byObject: appliedRows.byObject,
    remapPairs: appliedRows.remapPairs,
    rawByObject: candidates.byObject,
    rawRemapPairs: candidates.remapPairs,
  };
  results.push(result);
  console.log(
    `object_boundary_remap_preview_asset=passed asset=${JSON.stringify(asset.id)} ` +
      `gaussians=${result.gaussianCount} sampled=${result.sampledGaussians} ` +
      `step=${result.sampleStep} raw=${result.rawCandidateRemapGaussians} remapped=${result.remappedGaussians} ` +
      `policyBlocked=${result.policyGate?.blockedRemaps ?? 0} ` +
      `estimated=${result.estimatedFullRemapGaussians} output=${JSON.stringify(result.outputPly)}`,
  );
}

const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  writePreviewPly,
  options: {
    neighborRadiusRatio,
    maxRemapSamples,
    maxCellChecks,
    cleanupNeighborThreshold,
    cleanupDominantThreshold,
    cleanupMinNeighbors,
  },
  decisionPolicy: decisionPolicy?.summary ?? null,
  assets: assetIds,
  skipped,
  passed: results.length > 0 && results.every((result) => result.passed),
  results,
};

writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary));
console.log(
  `object_boundary_remap_preview=${summary.passed ? "passed" : "failed"} ` +
    `assets=${results.length} skipped=${skipped.length} report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
);

if (!summary.passed) {
  process.exitCode = 1;
}

function remapCandidates(points, options) {
  const bounds = sceneBounds(points);
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

  const sampleStep = Math.max(1, Math.ceil(points.length / options.maxRemapSamples));
  const remaps = [];
  const objectRows = new Map();
  const pairRows = new Map();
  let sampledGaussians = 0;
  let lowConfidenceSamples = 0;

  for (let index = 0; index < points.length; index += sampleStep) {
    sampledGaussians += 1;
    const point = points[index];
    const nearby = nearbyIndices({ point, cells, cellSize, maxCellChecks: options.maxCellChecks });
    let sameNeighbors = 0;
    let differentNeighbors = 0;
    const differentByObject = new Map();
    for (const otherIndex of nearby) {
      if (otherIndex === index) continue;
      const other = points[otherIndex];
      if (distance(point, other) > radius) continue;
      if (other.objectId === point.objectId) {
        sameNeighbors += 1;
      } else {
        differentNeighbors += 1;
        differentByObject.set(other.objectId, (differentByObject.get(other.objectId) ?? 0) + 1);
      }
    }
    const totalNeighbors = sameNeighbors + differentNeighbors;
    if (totalNeighbors < options.cleanupMinNeighbors || differentNeighbors <= 0) continue;
    const mixedRatio = differentNeighbors / totalNeighbors;
    const dominant = dominantObjectCount(differentByObject);
    const dominantSupport = dominant.count / Math.max(differentNeighbors, 1);
    if (
      mixedRatio < options.cleanupNeighborThreshold ||
      dominantSupport < options.cleanupDominantThreshold ||
      dominant.objectId === null
    ) {
      if (mixedRatio >= options.cleanupNeighborThreshold) lowConfidenceSamples += 1;
      continue;
    }
    const remap = {
      index,
      fromObject: point.objectId,
      toObject: dominant.objectId,
      sameNeighbors,
      differentNeighbors,
      totalNeighbors,
      mixedRatio: roundMetric(mixedRatio),
      dominantSupport: roundMetric(dominantSupport),
    };
    remaps.push(remap);
    updateObjectRow(objectRows, remap);
    updatePairRow(pairRows, remap);
  }

  return {
    sampleStep,
    sampledGaussians,
    remaps,
    lowConfidenceSamples,
    estimatedFullRemapGaussians: Math.min(points.length, remaps.length * sampleStep),
    byObject: [...objectRows.values()].sort((left, right) => right.remapped - left.remapped),
    remapPairs: [...pairRows.values()].sort((left, right) => right.remapped - left.remapped),
  };
}

function updateObjectRow(rows, remap) {
  let row = rows.get(remap.fromObject);
  if (!row) {
    row = {
      objectId: remap.fromObject,
      remapped: 0,
      targetCounts: {},
      meanMixedRatio: 0,
      meanDominantSupport: 0,
    };
    rows.set(remap.fromObject, row);
  }
  row.remapped += 1;
  row.targetCounts[String(remap.toObject)] = (row.targetCounts[String(remap.toObject)] ?? 0) + 1;
  row.meanMixedRatio = runningMean(row.meanMixedRatio, row.remapped, remap.mixedRatio);
  row.meanDominantSupport = runningMean(
    row.meanDominantSupport,
    row.remapped,
    remap.dominantSupport,
  );
}

function updatePairRow(rows, remap) {
  const key = `${remap.fromObject}->${remap.toObject}`;
  let row = rows.get(key);
  if (!row) {
    row = {
      fromObject: remap.fromObject,
      toObject: remap.toObject,
      remapped: 0,
      meanMixedRatio: 0,
      meanDominantSupport: 0,
    };
    rows.set(key, row);
  }
  row.remapped += 1;
  row.meanMixedRatio = runningMean(row.meanMixedRatio, row.remapped, remap.mixedRatio);
  row.meanDominantSupport = runningMean(
    row.meanDominantSupport,
    row.remapped,
    remap.dominantSupport,
  );
}

function summarizeRemaps(remaps) {
  const objectRows = new Map();
  const pairRows = new Map();
  for (const remap of remaps) {
    updateObjectRow(objectRows, remap);
    updatePairRow(pairRows, remap);
  }
  return {
    byObject: [...objectRows.values()].sort((left, right) => right.remapped - left.remapped),
    remapPairs: [...pairRows.values()].sort((left, right) => right.remapped - left.remapped),
  };
}

function loadDecisionPolicy({ policyPath: inputPolicyPath, explicitAllowTargets: inputAllowTargets }) {
  if (!inputPolicyPath) {
    if (inputAllowTargets.length > 0) {
      throw new Error("--allow-target requires --policy");
    }
    return null;
  }
  if (!existsSync(inputPolicyPath)) {
    throw new Error(`decision policy is missing: ${inputPolicyPath}`);
  }
  const raw = JSON.parse(readFileSync(inputPolicyPath, "utf-8"));
  if (raw.mode !== "object-boundary-remap-decision-policy-v1") {
    throw new Error(`unsupported decision policy mode: ${raw.mode}`);
  }
  const targetsByKey = new Map();
  const allowlistCandidatesByAsset = new Map();
  for (const target of raw.targets ?? []) {
    const assetId = String(target.assetId ?? "");
    const targetObjectId = Number(target.targetObjectId);
    if (!assetId || !Number.isFinite(targetObjectId)) continue;
    targetsByKey.set(targetKey(assetId, targetObjectId), target);
    if (target.decision === "allowlist-candidate") {
      mapSetAdd(allowlistCandidatesByAsset, assetId, targetObjectId);
    }
  }
  const requestedByAsset = new Map();
  for (const target of inputAllowTargets) {
    mapSetAdd(requestedByAsset, target.assetId, target.targetObjectId);
  }
  return {
    path: inputPolicyPath,
    raw,
    targetsByKey,
    allowlistCandidatesByAsset,
    requestedByAsset,
    summary: {
      mode: raw.mode,
      path: inputPolicyPath,
      recommendation: raw.recommendation ?? "",
      defaultAction: raw.defaultAction ?? "",
      sourceApplyMode: raw.applyMode ?? "",
      applyMode: "explicit-target-allowlist-v1",
      explicitAllowTargets: inputAllowTargets,
      counts: raw.counts ?? {},
    },
  };
}

function applyDecisionPolicy({ policy, assetId, remaps }) {
  const requested = policy.requestedByAsset.get(assetId) ?? new Set();
  const candidateTargets = policy.allowlistCandidatesByAsset.get(assetId) ?? new Set();
  const allowedTargets = new Set(
    [...requested].filter((targetObjectId) => candidateTargets.has(targetObjectId)),
  );
  const appliedRemaps = [];
  const blockedByDecision = {};
  const blockedTargetObjectIds = new Set();
  const appliedTargetObjectIds = new Set();
  for (const remap of remaps) {
    if (allowedTargets.has(remap.fromObject)) {
      appliedRemaps.push(remap);
      appliedTargetObjectIds.add(remap.fromObject);
      continue;
    }
    blockedTargetObjectIds.add(remap.fromObject);
    const decision = policy.targetsByKey.get(targetKey(assetId, remap.fromObject))?.decision ?? "not-in-policy";
    const reason =
      !candidateTargets.has(remap.fromObject)
        ? decision
        : requested.has(remap.fromObject)
          ? "not-allowed-by-policy"
          : "missing-explicit-allow-target";
    blockedByDecision[reason] = (blockedByDecision[reason] ?? 0) + 1;
  }
  return {
    remaps: appliedRemaps,
    summary: {
      mode: "explicit-target-allowlist-v1",
      policyPath: policy.path,
      defaultAction: policy.raw.defaultAction ?? "keep-hard-mask",
      policyRecommendation: policy.raw.recommendation ?? "",
      requestedTargetObjectIds: [...requested].sort(numberSort),
      policyAllowlistCandidateTargetObjectIds: [...candidateTargets].sort(numberSort),
      allowedTargetObjectIds: [...allowedTargets].sort(numberSort),
      appliedTargetObjectIds: [...appliedTargetObjectIds].sort(numberSort),
      blockedTargetObjectIds: [...blockedTargetObjectIds].sort(numberSort),
      rawRemaps: remaps.length,
      appliedRemaps: appliedRemaps.length,
      blockedRemaps: remaps.length - appliedRemaps.length,
      blockedByDecision,
    },
  };
}

function mapSetAdd(map, key, value) {
  let set = map.get(key);
  if (!set) {
    set = new Set();
    map.set(key, set);
  }
  set.add(value);
}

function targetKey(assetId, targetObjectId) {
  return `${assetId}::${targetObjectId}`;
}

function patchObjectIds({ input, header, objectIdLayout, remaps }) {
  if (!objectIdLayout.info.integer) {
    throw new Error(`object_id property must be integer, got ${objectIdLayout.property.type}`);
  }
  if (header.format === "ascii") {
    return patchAsciiObjectIds({ input, header, objectIdLayout, remaps });
  }
  if (header.format === "binary_little_endian" || header.format === "binary_big_endian") {
    return patchBinaryObjectIds({ input, header, objectIdLayout, remaps });
  }
  throw new Error(`unsupported PLY format: ${header.format}`);
}

function patchBinaryObjectIds({ input, header, objectIdLayout, remaps }) {
  const output = Buffer.from(input);
  const view = new DataView(output.buffer, output.byteOffset, output.byteLength);
  const littleEndian = header.format === "binary_little_endian";
  for (const remap of remaps) {
    const offset = header.headerEnd + remap.index * header.vertexStride + objectIdLayout.offset;
    view[objectIdLayout.info.setter](offset, remap.toObject, littleEndian);
  }
  return output;
}

function patchAsciiObjectIds({ input, header, objectIdLayout, remaps }) {
  const headerText = input.subarray(0, header.headerEnd).toString("utf8");
  const body = input.subarray(header.headerEnd).toString("utf8");
  const lines = body.split(/\r?\n/);
  for (const remap of remaps) {
    const line = lines[remap.index];
    if (!line) continue;
    const values = line.trim().split(/\s+/);
    values[objectIdLayout.propertyIndex] = String(remap.toObject);
    lines[remap.index] = values.join(" ");
  }
  return Buffer.from(`${headerText}${lines.join("\n")}`, "utf8");
}

function readPlyHeader(buffer) {
  const bytes = new Uint8Array(buffer);
  let end = -1;
  for (let index = 0; index < bytes.length - 10; index += 1) {
    if (
      bytes[index] === 101 &&
      bytes[index + 1] === 110 &&
      bytes[index + 2] === 100 &&
      bytes[index + 3] === 95 &&
      bytes[index + 4] === 104 &&
      bytes[index + 5] === 101 &&
      bytes[index + 6] === 97 &&
      bytes[index + 7] === 100 &&
      bytes[index + 8] === 101 &&
      bytes[index + 9] === 114
    ) {
      let cursor = index + 10;
      if (bytes[cursor] === 13) cursor += 1;
      if (bytes[cursor] === 10) cursor += 1;
      end = cursor;
      break;
    }
  }
  if (end < 0) throw new Error("missing PLY end_header");
  const text = new TextDecoder("ascii").decode(buffer.slice(0, end));
  const lines = text.split(/\r?\n/);
  if (lines[0] !== "ply") throw new Error("not a PLY file");

  let format = null;
  let currentElement = null;
  let vertexCount = 0;
  const properties = [];
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (!parts[0]) continue;
    if (parts[0] === "format") {
      format = parts[1];
    } else if (parts[0] === "element") {
      currentElement = parts[1];
      if (currentElement === "vertex") vertexCount = Number(parts[2]);
    } else if (parts[0] === "property" && currentElement === "vertex") {
      if (parts[1] === "list") throw new Error("list vertex properties are not supported");
      const info = TYPE_INFO[parts[1]];
      if (!info) throw new Error(`unsupported PLY property type: ${parts[1]}`);
      properties.push({ type: parts[1], name: parts[2], info });
    }
  }
  if (!format) throw new Error("PLY format is missing");
  const vertexStride = properties.reduce((total, property) => total + property.info.size, 0);
  return {
    format,
    vertexCount,
    properties,
    headerEnd: end,
    vertexStride,
  };
}

function vertexPropertyLayout(header, name) {
  let offset = 0;
  for (let index = 0; index < header.properties.length; index += 1) {
    const property = header.properties[index];
    if (property.name === name) {
      return { property, propertyIndex: index, offset, info: property.info };
    }
    offset += property.info.size;
  }
  throw new Error(`PLY is missing required vertex property: ${name}`);
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
  return { min, max };
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

function distance(left, right) {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function selectAssets(ids) {
  const candidates = ASSET_LIBRARY.filter(
    (asset) =>
      asset.sourceType === "gaussian" &&
      typeof asset.localPath === "string" &&
      asset.localPath.endsWith(".ply"),
  );
  const byId = new Map(candidates.map((asset) => [asset.id, asset]));
  return ids.map((id) => {
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

function parseAllowTargets(value) {
  if (value === undefined || value === null || value === false) return [];
  if (value === true) throw new Error("--allow-target requires asset_id:object_id entries");
  return String(value)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => {
      const separator = entry.lastIndexOf(":");
      if (separator <= 0 || separator === entry.length - 1) {
        throw new Error(`invalid --allow-target entry: ${entry}`);
      }
      const assetId = entry.slice(0, separator);
      const targetObjectId = Number(entry.slice(separator + 1));
      if (!assetId || !Number.isFinite(targetObjectId)) {
        throw new Error(`invalid --allow-target entry: ${entry}`);
      }
      return { assetId, targetObjectId };
    });
}

function publicPath(value) {
  const normalized = String(value ?? "");
  if (!normalized.startsWith("/")) return normalized;
  return path.join("public", normalized.slice(1));
}

function renderMarkdown(summary) {
  const lines = [
    "# Object Boundary Remap Preview",
    "",
    `- Status: ${summary.passed ? "passed" : "failed"}`,
    `- Mode: ${summary.mode}`,
    `- Generated: ${summary.generatedAt}`,
    `- Preview PLY written: ${summary.writePreviewPly ? "yes" : "no"}`,
    `- Decision policy: ${summary.decisionPolicy ? summary.decisionPolicy.path : "none"}`,
    "",
    "This is an experimental, sampled remap preview. It preserves the source PLY bytes and only patches `object_id` for sampled Gaussians whose local 3D neighborhood is dominated by another object id. It is not a promoted cleanup policy and must be followed by browser residual checks before any default route change.",
    "",
  ];
  if (summary.decisionPolicy) {
    lines.push(
      "When a decision policy is provided, remaps are applied only when the target is both a policy `allowlist-candidate` and explicitly listed with `--allow-target asset_id:object_id`. All other candidate remaps keep the original hard-mask assignment.",
      "",
      "## Decision Policy Gate",
      "",
      `- Policy mode: ${summary.decisionPolicy.mode}`,
      `- Apply mode: ${summary.decisionPolicy.applyMode}`,
      `- Recommendation: ${summary.decisionPolicy.recommendation}`,
      `- Default action: ${summary.decisionPolicy.defaultAction}`,
      `- Explicit allow targets: ${summary.decisionPolicy.explicitAllowTargets
        .map((target) => `${target.assetId}:${target.targetObjectId}`)
        .join(", ") || "none"}`,
      "",
    );
  }
  lines.push(
    "## Assets",
    "",
    "| Asset | Gaussians | Sampled | Step | Raw candidates | Applied remaps | Policy blocked | Estimated full remap | Share | Output |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  );
  for (const result of summary.results) {
    lines.push(
      `| ${escapeMarkdown(result.assetId)} | ${result.gaussianCount} | ${result.sampledGaussians} | ${result.sampleStep} | ${result.rawCandidateRemapGaussians ?? result.remappedGaussians} | ${result.remappedGaussians} | ${result.policyGate?.blockedRemaps ?? 0} | ${result.estimatedFullRemapGaussians} | ${formatNumber(result.estimatedFullRemapShare)} | ${escapeMarkdown(result.outputPly ?? "")} |`,
    );
  }
  if (summary.decisionPolicy) {
    lines.push("", "## Policy Gate By Asset", "");
    lines.push(
      "| Asset | Requested targets | Policy candidates | Applied targets | Blocked targets | Raw remaps | Applied remaps | Blocked remaps |",
      "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    );
    for (const result of summary.results) {
      const gate = result.policyGate;
      lines.push(
        `| ${escapeMarkdown(result.assetId)} | ${formatIdList(gate?.requestedTargetObjectIds)} | ${formatIdList(gate?.policyAllowlistCandidateTargetObjectIds)} | ${formatIdList(gate?.appliedTargetObjectIds)} | ${formatIdList(gate?.blockedTargetObjectIds)} | ${gate?.rawRemaps ?? 0} | ${gate?.appliedRemaps ?? 0} | ${gate?.blockedRemaps ?? 0} |`,
      );
    }
  }
  lines.push("", "## Remap Pairs", "");
  for (const result of summary.results) {
    lines.push(`### ${result.assetId}`, "");
    lines.push(
      "| From | To | Sampled remaps | Mean mixed ratio | Mean target support |",
      "| ---: | ---: | ---: | ---: | ---: |",
    );
    for (const row of result.remapPairs ?? []) {
      lines.push(
        `| ${row.fromObject} | ${row.toObject} | ${row.remapped} | ${formatNumber(row.meanMixedRatio)} | ${formatNumber(row.meanDominantSupport)} |`,
      );
    }
    lines.push("");
  }
  if (summary.skipped.length > 0) {
    lines.push("## Skipped", "");
    for (const item of summary.skipped) {
      lines.push(`- ${item.assetId}: ${item.reason} (${item.plyPath})`);
    }
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
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

function optionalString(value) {
  if (value === undefined || value === null || value === false) return null;
  if (value === true) throw new Error("expected string value");
  const text = String(value).trim();
  return text ? text : null;
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function roundMetric(value, digits = 6) {
  if (!Number.isFinite(value)) return 0;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function formatNumber(value) {
  return roundMetric(Number(value ?? 0)).toFixed(6);
}

function formatIdList(value) {
  const values = Array.isArray(value) ? value : [];
  return values.length > 0 ? values.join(", ") : "none";
}

function runningMean(currentMean, countAfterInsert, nextValue) {
  return roundMetric(currentMean + (nextValue - currentMean) / Math.max(countAfterInsert, 1));
}

function numberSort(left, right) {
  return left - right;
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
}
