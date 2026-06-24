import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { ASSET_LIBRARY } from "../src/assetLibrary.js";

const MODE = "splat-index-mapping-v1";
const DEFAULT_POSITION_TOLERANCE = 1e-6;
const ROUND_POSITION_DECIMALS = 6;

const TYPE_INFO = {
  char: { size: 1, getter: "getInt8" },
  int8: { size: 1, getter: "getInt8" },
  uchar: { size: 1, getter: "getUint8" },
  uint8: { size: 1, getter: "getUint8" },
  short: { size: 2, getter: "getInt16" },
  int16: { size: 2, getter: "getInt16" },
  ushort: { size: 2, getter: "getUint16" },
  uint16: { size: 2, getter: "getUint16" },
  int: { size: 4, getter: "getInt32" },
  int32: { size: 4, getter: "getInt32" },
  uint: { size: 4, getter: "getUint32" },
  uint32: { size: 4, getter: "getUint32" },
  float: { size: 4, getter: "getFloat32" },
  float32: { size: 4, getter: "getFloat32" },
  double: { size: 8, getter: "getFloat64" },
  float64: { size: 8, getter: "getFloat64" },
};

const args = parseArgs(process.argv.slice(2));
const tolerance =
  optionalFiniteNumber(args.tolerance ?? args["position-tolerance"]) ??
  DEFAULT_POSITION_TOLERANCE;
const assetIds = optionalString(args.assets ?? args.asset)
  ? String(args.assets ?? args.asset)
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean)
  : null;
const outputDir = optionalString(args.outputDir ?? args["output-dir"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);

const assets = selectedAssets(assetIds);
const results = assets.map((asset) => auditAsset(asset, { tolerance }));
const summary = {
  mode: MODE,
  tolerance,
  assets: assets.map((asset) => asset.id),
  passed: results.every((result) => result.passed),
  results,
};

for (const result of results) {
  console.log(
    `splat_index_mapping_asset=${result.passed ? "passed" : "failed"} ` +
      `asset=${JSON.stringify(result.assetId)} ` +
      `splat=${JSON.stringify(result.splatPath)} ply=${JSON.stringify(result.plyPath)} ` +
      `count=${result.splatCount}/${result.plyCount} ` +
      `indexMatches=${result.positionIndexMatches}/${result.compareCount} ` +
      `maxPositionDelta=${formatNumber(result.maxPositionDelta)} ` +
      `meanPositionDelta=${formatNumber(result.meanPositionDelta)} ` +
      `positionMultisetCoverage=${formatNumber(result.positionMultisetCoverage)} ` +
      `duplicatePositionKeys=${result.duplicatePositionKeys} ` +
      `objects=${result.objectCount} ` +
      `scaleIndexMatches=${result.scaleIndexMatches}/${result.scaleCompareCount} ` +
      `maxScaleDelta=${formatNumber(result.maxScaleDelta)} ` +
      `feasibility=${JSON.stringify(result.nativeMaskFeasibility)}`,
  );
  for (const mismatch of result.firstPositionMismatches) {
    console.log(
      `splat_index_mapping_mismatch asset=${JSON.stringify(result.assetId)} ` +
        `index=${mismatch.index} positionDelta=${formatNumber(mismatch.positionDelta)} ` +
        `splat=${JSON.stringify(mismatch.splat)} ply=${JSON.stringify(mismatch.ply)}`,
    );
  }
}

if (outputDir) {
  writeReport(outputDir, summary);
  console.log(
    `splat_index_mapping_report=written outputDir=${JSON.stringify(outputDir)} ` +
      `summaryJson=${JSON.stringify(path.join(outputDir, "summary.json"))} ` +
      `summaryMd=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
  );
}

console.log(
  `splat_index_mapping=${summary.passed ? "passed" : "failed"} ` +
    `mode=${JSON.stringify(MODE)} assets=${JSON.stringify(summary.assets)} ` +
    `tolerance=${tolerance}`,
);

if (!summary.passed && !allowFailures) {
  process.exitCode = 1;
}

function auditAsset(asset, { tolerance: positionTolerance }) {
  const splatPath = publicPath(asset.splatPath);
  const plyPath = publicPath(asset.localPath);
  if (!existsSync(splatPath)) {
    throw new Error(`${asset.id} missing splat file: ${splatPath}`);
  }
  if (!existsSync(plyPath)) {
    throw new Error(`${asset.id} missing object-aware PLY file: ${plyPath}`);
  }

  const splat = parseSplat(readFileSync(splatPath));
  const ply = parsePlyRaw(readFileSync(plyPath));
  const compareCount = Math.min(splat.count, ply.count);
  let positionIndexMatches = 0;
  let maxPositionDelta = 0;
  let positionDeltaSum = 0;
  const firstPositionMismatches = [];

  for (let index = 0; index < compareCount; index += 1) {
    const positionDelta = vectorDelta(splat.positions, ply.positions, index);
    positionDeltaSum += positionDelta;
    maxPositionDelta = Math.max(maxPositionDelta, positionDelta);
    if (positionDelta <= positionTolerance) {
      positionIndexMatches += 1;
    } else if (firstPositionMismatches.length < 5) {
      firstPositionMismatches.push({
        index,
        positionDelta,
        splat: vectorAt(splat.positions, index),
        ply: vectorAt(ply.positions, index),
      });
    }
  }

  let scaleIndexMatches = 0;
  let scaleCompareCount = 0;
  let maxScaleDelta = 0;
  if (splat.scales && ply.scales) {
    scaleCompareCount = compareCount;
    for (let index = 0; index < compareCount; index += 1) {
      const scaleDelta = vectorDelta(splat.scales, ply.scales, index);
      maxScaleDelta = Math.max(maxScaleDelta, scaleDelta);
      if (scaleDelta <= positionTolerance) {
        scaleIndexMatches += 1;
      }
    }
  }

  const positionSet = positionMultisetCoverage(splat.positions, ply.positions, compareCount);
  const objectCount = ply.objectIds ? new Set(ply.objectIds).size : 0;
  const countsMatch = splat.count === ply.count;
  const indexPreserved = countsMatch && positionIndexMatches === compareCount;
  const samePositionMultiset =
    countsMatch && positionSet.commonRoundedPositions === compareCount;
  const nativeMaskFeasibility = indexPreserved
    ? "index-preserved-public-sample"
    : samePositionMultiset
      ? "same-position-set-but-not-index-preserved"
      : "not-index-mapped";

  return {
    assetId: asset.id,
    name: asset.name,
    splatPath,
    plyPath,
    splatCount: splat.count,
    plyCount: ply.count,
    compareCount,
    positionTolerance,
    positionIndexMatches,
    positionIndexMatchRatio: ratio(positionIndexMatches, compareCount),
    maxPositionDelta,
    meanPositionDelta: compareCount > 0 ? positionDeltaSum / compareCount : 0,
    firstPositionMismatches,
    positionMultisetCoverage: ratio(positionSet.commonRoundedPositions, compareCount),
    duplicatePositionKeys: Math.max(
      positionSet.splatDuplicateKeys,
      positionSet.plyDuplicateKeys,
    ),
    splatDuplicatePositionKeys: positionSet.splatDuplicateKeys,
    plyDuplicatePositionKeys: positionSet.plyDuplicateKeys,
    objectCount,
    objectIdRange: ply.objectIds ? idRange(ply.objectIds) : null,
    scaleIndexMatches,
    scaleCompareCount,
    maxScaleDelta,
    nativeMaskFeasibility,
    passed: indexPreserved,
  };
}

function selectedAssets(ids) {
  const localGaussianAssets = ASSET_LIBRARY.filter(
    (asset) =>
      asset.sourceType === "gaussian" &&
      typeof asset.localPath === "string" &&
      typeof asset.splatPath === "string" &&
      asset.localPath.endsWith(".ply") &&
      asset.splatPath.endsWith(".splat"),
  );
  if (!ids) return localGaussianAssets;
  const byId = new Map(localGaussianAssets.map((asset) => [asset.id, asset]));
  return ids.map((id) => {
    const asset = byId.get(id);
    if (!asset) {
      throw new Error(`unknown local Gaussian asset for splat index audit: ${id}`);
    }
    return asset;
  });
}

function publicPath(value) {
  const normalized = String(value ?? "");
  if (!normalized.startsWith("/")) return normalized;
  return path.join("public", normalized.slice(1));
}

function parseSplat(buffer) {
  if (buffer.byteLength % 32 !== 0) {
    throw new Error(`unsupported .splat byte length ${buffer.byteLength}; not divisible by 32`);
  }
  const view = dataView(buffer);
  const count = buffer.byteLength / 32;
  const positions = new Float64Array(count * 3);
  const scales = new Float64Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const base = index * 32;
    positions[index * 3] = view.getFloat32(base, true);
    positions[index * 3 + 1] = view.getFloat32(base + 4, true);
    positions[index * 3 + 2] = view.getFloat32(base + 8, true);
    scales[index * 3] = view.getFloat32(base + 12, true);
    scales[index * 3 + 1] = view.getFloat32(base + 16, true);
    scales[index * 3 + 2] = view.getFloat32(base + 20, true);
  }
  return { count, positions, scales };
}

function parsePlyRaw(buffer) {
  const header = readPlyHeader(buffer);
  const positions = new Float64Array(header.vertexCount * 3);
  const scalePropertyIndices = ["scale_0", "scale_1", "scale_2"].map((name) =>
    header.properties.findIndex((property) => property.name === name),
  );
  const hasScaleProperties = scalePropertyIndices.every((index) => index >= 0);
  const scales = hasScaleProperties ? new Float64Array(header.vertexCount * 3) : null;
  const objectIdIndex = header.properties.findIndex((property) => property.name === "object_id");
  const objectIds = objectIdIndex >= 0 ? new Int32Array(header.vertexCount) : null;
  const xIndex = propertyIndex(header, "x");
  const yIndex = propertyIndex(header, "y");
  const zIndex = propertyIndex(header, "z");

  if (header.format === "ascii") {
    const text = new TextDecoder("utf-8").decode(arrayBuffer(buffer, header.headerEnd));
    const lines = text.trim().split(/\r?\n/);
    for (let row = 0; row < header.vertexCount; row += 1) {
      const values = (lines[row] ?? "").trim().split(/\s+/).map(Number);
      positions[row * 3] = numberOrZero(values[xIndex]);
      positions[row * 3 + 1] = numberOrZero(values[yIndex]);
      positions[row * 3 + 2] = numberOrZero(values[zIndex]);
      if (scales) {
        scales[row * 3] = numberOrZero(values[scalePropertyIndices[0]]);
        scales[row * 3 + 1] = numberOrZero(values[scalePropertyIndices[1]]);
        scales[row * 3 + 2] = numberOrZero(values[scalePropertyIndices[2]]);
      }
      if (objectIds) objectIds[row] = Math.trunc(numberOrZero(values[objectIdIndex]));
    }
    return { count: header.vertexCount, positions, scales, objectIds };
  }

  const view = dataView(buffer);
  const littleEndian = header.format === "binary_little_endian";
  let offset = header.headerEnd;
  for (let row = 0; row < header.vertexCount; row += 1) {
    for (let column = 0; column < header.properties.length; column += 1) {
      const property = header.properties[column];
      const info = TYPE_INFO[property.type];
      const value = view[info.getter](offset, littleEndian);
      if (column === xIndex) positions[row * 3] = value;
      if (column === yIndex) positions[row * 3 + 1] = value;
      if (column === zIndex) positions[row * 3 + 2] = value;
      if (scales) {
        const scaleColumn = scalePropertyIndices.indexOf(column);
        if (scaleColumn >= 0) scales[row * 3 + scaleColumn] = value;
      }
      if (objectIds && column === objectIdIndex) objectIds[row] = Math.trunc(value);
      offset += info.size;
    }
  }
  return { count: header.vertexCount, positions, scales, objectIds };
}

function readPlyHeader(buffer) {
  const bytes = new Uint8Array(arrayBuffer(buffer));
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
  if (end < 0) throw new Error("PLY end_header not found");

  const text = new TextDecoder("ascii").decode(arrayBuffer(buffer, 0, end));
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
      if (parts[1] === "list") {
        throw new Error("PLY list vertex properties are not supported by this audit");
      }
      if (!TYPE_INFO[parts[1]]) throw new Error(`unsupported PLY property type: ${parts[1]}`);
      properties.push({ type: parts[1], name: parts[2] });
    }
  }
  if (!["ascii", "binary_little_endian", "binary_big_endian"].includes(format)) {
    throw new Error(`unsupported PLY format: ${format}`);
  }
  return { format, vertexCount, properties, headerEnd: end };
}

function propertyIndex(header, name) {
  const index = header.properties.findIndex((property) => property.name === name);
  if (index < 0) throw new Error(`PLY vertex data is missing required property: ${name}`);
  return index;
}

function positionMultisetCoverage(left, right, count) {
  const leftMap = roundedPositionMap(left, count);
  const rightMap = roundedPositionMap(right, count);
  let commonRoundedPositions = 0;
  for (const [key, leftCount] of leftMap.entries()) {
    commonRoundedPositions += Math.min(leftCount, rightMap.get(key) ?? 0);
  }
  return {
    commonRoundedPositions,
    splatDuplicateKeys: duplicateKeyCount(leftMap),
    plyDuplicateKeys: duplicateKeyCount(rightMap),
  };
}

function roundedPositionMap(positions, count) {
  const map = new Map();
  for (let index = 0; index < count; index += 1) {
    const key = vectorAt(positions, index)
      .map((value) => value.toFixed(ROUND_POSITION_DECIMALS))
      .join(",");
    map.set(key, (map.get(key) ?? 0) + 1);
  }
  return map;
}

function duplicateKeyCount(map) {
  let duplicates = 0;
  for (const count of map.values()) {
    if (count > 1) duplicates += 1;
  }
  return duplicates;
}

function vectorDelta(left, right, index) {
  const offset = index * 3;
  return Math.hypot(
    left[offset] - right[offset],
    left[offset + 1] - right[offset + 1],
    left[offset + 2] - right[offset + 2],
  );
}

function vectorAt(values, index) {
  const offset = index * 3;
  return [values[offset], values[offset + 1], values[offset + 2]];
}

function idRange(values) {
  if (!values.length) return null;
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return { min, max };
}

function arrayBuffer(buffer, start = 0, end = buffer.byteLength) {
  return buffer.buffer.slice(buffer.byteOffset + start, buffer.byteOffset + end);
}

function dataView(buffer) {
  return new DataView(arrayBuffer(buffer));
}

function numberOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function ratio(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : 0;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "null";
  if (Math.abs(value) >= 0.001) return Number(value.toFixed(6));
  return Number(value.toExponential(6));
}

function writeReport(outputDirPath, summary) {
  mkdirSync(outputDirPath, { recursive: true });
  writeFileSync(
    path.join(outputDirPath, "summary.json"),
    JSON.stringify(summary, null, 2) + "\n",
    "utf-8",
  );
  const lines = [
    "# Splat Index Mapping Audit",
    "",
    `Mode: \`${MODE}\``,
    "",
    `Passed: **${summary.passed ? "yes" : "no"}**`,
    "",
    "| Asset | Passed | Count | Index Matches | Max Position Delta | Objects | Feasibility |",
    "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ...summary.results.map(
      (result) =>
        `| ${escapeMarkdown(result.assetId)} | ${result.passed ? "yes" : "no"} | ` +
        `${result.splatCount}/${result.plyCount} | ` +
        `${result.positionIndexMatches}/${result.compareCount} | ` +
        `${formatNumber(result.maxPositionDelta)} | ${result.objectCount} | ` +
        `${escapeMarkdown(result.nativeMaskFeasibility)} |`,
    ),
    "",
    "Interpretation: this audit checks whether public compact `.splat` rows and object-aware PLY vertices preserve the same Gaussian index order. Passing samples are evidence that an external object mask can be keyed by Gaussian index for these generated assets; they do not prove arbitrary third-party `.splat` files carry object ids internally.",
    "",
  ];
  writeFileSync(path.join(outputDirPath, "summary.md"), lines.join("\n"), "utf-8");
}

function escapeMarkdown(value) {
  return String(value).replaceAll("|", "\\|");
}

function parseArgs(rawArgs) {
  const parsed = {};
  for (let index = 0; index < rawArgs.length; index += 1) {
    const raw = rawArgs[index];
    if (!raw.startsWith("--")) continue;
    const withoutPrefix = raw.slice(2);
    if (withoutPrefix.includes("=")) {
      const [key, ...rest] = withoutPrefix.split("=");
      parsed[key] = rest.join("=");
      continue;
    }
    const next = rawArgs[index + 1];
    if (next && !next.startsWith("--")) {
      parsed[withoutPrefix] = next;
      index += 1;
    } else {
      parsed[withoutPrefix] = true;
    }
  }
  return parsed;
}

function optionalString(value) {
  if (value === undefined || value === null || value === false) return null;
  const text = String(value).trim();
  return text ? text : null;
}

function optionalFiniteNumber(value) {
  if (value === undefined || value === null || value === false || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function flagEnabled(value) {
  if (value === undefined || value === null || value === false) return false;
  if (value === true) return true;
  return !["0", "false", "no", "off"].includes(String(value).toLowerCase());
}
