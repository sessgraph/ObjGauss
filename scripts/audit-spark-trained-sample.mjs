import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

import { ASSET_LIBRARY } from "../src/assetLibrary.js";

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
const assetId = String(args.asset ?? args["asset-id"] ?? "nerf-lego-trained-output-local");
const publicDir = String(args.publicDir ?? args["public-dir"] ?? "public");
const outputDir = String(
  args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-spark-trained-sample",
);
const minGaussians = positiveInteger(args.minGaussians ?? args["min-gaussians"] ?? 100_000);
const minObjects = positiveInteger(args.minObjects ?? args["min-objects"] ?? 2);
const minShRestCoefficients = positiveInteger(
  args.minShRestCoefficients ?? args["min-sh-rest-coefficients"] ?? 45,
);
const minShDegree = positiveInteger(args.minShDegree ?? args["min-sh-degree"] ?? 3);

const summary = auditSample({
  assetId,
  publicDir,
  minGaussians,
  minObjects,
  minShRestCoefficients,
  minShDegree,
});

writeReport(outputDir, summary);

if (!summary.passed) {
  console.error(`spark_trained_sample=failed report=${outputDir}/summary.md`);
  for (const failure of summary.failures) console.error(`failure=${failure}`);
  process.exit(1);
}

console.log(
  `spark_trained_sample=passed asset=${JSON.stringify(summary.asset.id)} ` +
    `ply=${JSON.stringify(summary.files.objectPly.path)} splat=${JSON.stringify(summary.files.splat.path)} ` +
    `gaussians=${summary.ply.vertexCount} objects=${summary.ply.objectCount} ` +
    `shRest=${summary.ply.shRestCount}:${summary.ply.shDegree} ` +
    `report=${outputDir}/summary.md`,
);

function auditSample({
  assetId,
  publicDir,
  minGaussians,
  minObjects,
  minShRestCoefficients,
  minShDegree,
}) {
  const failures = [];
  const asset = ASSET_LIBRARY.find((entry) => entry.id === assetId);
  if (!asset) {
    return {
      passed: false,
      generatedAt: new Date().toISOString(),
      asset: { id: assetId },
      thresholds: { minGaussians, minObjects, minShRestCoefficients, minShDegree },
      files: {},
      ply: {},
      failures: [`asset not registered in src/assetLibrary.js: ${assetId}`],
      prepare: prepareCommands(),
    };
  }

  const objectPly = publicAssetPath(publicDir, asset.localPath);
  const splat = publicAssetPath(publicDir, asset.splatPath);
  if (!objectPly || !existsSync(objectPly)) {
    failures.push(`missing public object PLY: ${objectPly || asset.localPath || "unset"}`);
  }
  if (!splat || !existsSync(splat)) {
    failures.push(`missing public splat: ${splat || asset.splatPath || "unset"}`);
  }

  let ply = {
    vertexCount: 0,
    format: "",
    properties: [],
    shRestCount: 0,
    shDegree: 0,
    objectCount: 0,
    objectIdCounts: [],
  };
  if (objectPly && existsSync(objectPly)) {
    ply = inspectObjectPly(objectPly);
    const propertyNames = new Set(ply.properties);
    for (const name of requiredProperties()) {
      if (!propertyNames.has(name)) failures.push(`PLY missing property: ${name}`);
    }
    if (ply.vertexCount < minGaussians) {
      failures.push(`PLY vertex count ${ply.vertexCount} < min ${minGaussians}`);
    }
    if (ply.objectCount < minObjects) {
      failures.push(`PLY object count ${ply.objectCount} < min ${minObjects}`);
    }
    if (ply.shRestCount < minShRestCoefficients) {
      failures.push(`PLY f_rest count ${ply.shRestCount} < min ${minShRestCoefficients}`);
    }
    if (ply.shDegree < minShDegree) {
      failures.push(`PLY inferred SH degree ${ply.shDegree} < min ${minShDegree}`);
    }
  }

  return {
    passed: failures.length === 0,
    generatedAt: new Date().toISOString(),
    asset: {
      id: asset.id,
      name: asset.name,
      status: asset.status,
      localPath: asset.localPath,
      splatPath: asset.splatPath,
      license: asset.license,
    },
    thresholds: { minGaussians, minObjects, minShRestCoefficients, minShDegree },
    files: {
      objectPly: fileStatus(objectPly),
      splat: fileStatus(splat),
    },
    ply,
    failures,
    prepare: prepareCommands(),
  };
}

function inspectObjectPly(plyPath) {
  const buffer = readFileSync(plyPath);
  const header = readPlyHeader(buffer);
  const properties = header.properties.map((property) => property.name);
  const objectIdIndex = properties.indexOf("object_id");
  const objectIdCounts = objectIdIndex >= 0 ? countObjectIds(buffer, header, objectIdIndex) : [];
  const shRestCount = properties.filter((name) => /^f_rest_\d+$/.test(name)).length;
  return {
    format: header.format,
    vertexCount: header.vertexCount,
    properties,
    objectCount: objectIdCounts.length,
    objectIdCounts,
    shRestCount,
    shDegree: inferShDegree(shRestCount),
  };
}

function countObjectIds(buffer, header, objectIdIndex) {
  const counts = new Map();
  if (header.format === "ascii") {
    const text = new TextDecoder("utf-8").decode(arrayBuffer(buffer, header.headerEnd));
    const lines = text.trim().split(/\r?\n/);
    for (let row = 0; row < header.vertexCount; row += 1) {
      const value = Number.parseInt((lines[row] ?? "").trim().split(/\s+/)[objectIdIndex], 10);
      if (Number.isFinite(value)) counts.set(value, (counts.get(value) ?? 0) + 1);
    }
    return sortedCounts(counts);
  }

  const view = dataView(buffer);
  const littleEndian = header.format === "binary_little_endian";
  let offset = header.headerEnd;
  for (let row = 0; row < header.vertexCount; row += 1) {
    for (let column = 0; column < header.properties.length; column += 1) {
      const property = header.properties[column];
      const info = TYPE_INFO[property.type];
      const value = view[info.getter](offset, littleEndian);
      if (column === objectIdIndex) {
        const objectId = Math.trunc(value);
        counts.set(objectId, (counts.get(objectId) ?? 0) + 1);
      }
      offset += info.size;
    }
  }
  return sortedCounts(counts);
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

function requiredProperties() {
  return [
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "object_id",
  ];
}

function inferShDegree(shRestCount) {
  for (let degree = 0; degree <= 6; degree += 1) {
    if (3 * ((degree + 1) ** 2 - 1) === shRestCount) return degree;
  }
  return 0;
}

function fileStatus(filePath) {
  if (!filePath) return { path: "", exists: false, byteSize: 0 };
  const exists = existsSync(filePath);
  return {
    path: filePath,
    exists,
    byteSize: exists ? statSync(filePath).size : 0,
  };
}

function writeReport(outputDirPath, summary) {
  mkdirSync(outputDirPath, { recursive: true });
  writeFileSync(`${outputDirPath}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(`${outputDirPath}/summary.md`, renderMarkdown(summary));
}

function renderMarkdown(summary) {
  const objectCounts = summary.ply.objectIdCounts ?? [];
  const lines = [
    "# Spark Trained Sample Availability",
    "",
    `- Status: ${summary.passed ? "passed" : "failed"}`,
    `- Asset: ${summary.asset.id}`,
    `- Generated: ${summary.generatedAt}`,
    "",
    "## Files",
    "",
    "| File | Exists | Bytes |",
    "| --- | ---: | ---: |",
    `| ${escapeMarkdown(summary.files.objectPly?.path ?? "")} | ${yesNo(summary.files.objectPly?.exists)} | ${summary.files.objectPly?.byteSize ?? 0} |`,
    `| ${escapeMarkdown(summary.files.splat?.path ?? "")} | ${yesNo(summary.files.splat?.exists)} | ${summary.files.splat?.byteSize ?? 0} |`,
    "",
    "## PLY Contract",
    "",
    `- Format: ${summary.ply.format ?? ""}`,
    `- Gaussians: ${summary.ply.vertexCount ?? 0}`,
    `- Objects: ${summary.ply.objectCount ?? 0}`,
    `- Object counts: ${objectCounts.map((entry) => `${entry.object_id}:${entry.count}`).join(", ")}`,
    `- f_rest coefficients: ${summary.ply.shRestCount ?? 0}`,
    `- inferred SH degree: ${summary.ply.shDegree ?? 0}`,
    "",
    "## Thresholds",
    "",
    `- min gaussians: ${summary.thresholds.minGaussians}`,
    `- min objects: ${summary.thresholds.minObjects}`,
    `- min f_rest coefficients: ${summary.thresholds.minShRestCoefficients}`,
    `- min SH degree: ${summary.thresholds.minShDegree}`,
    "",
  ];
  if (summary.failures.length > 0) {
    lines.push("## Failures", "", ...summary.failures.map((failure) => `- ${failure}`), "");
  }
  lines.push("## Prepare", "", ...summary.prepare.map((command) => `- \`${command}\``), "");
  return lines.join("\n");
}

function prepareCommands() {
  return [
    "npm run benchmark:splatfacto:balanced -- --status",
    "npm run benchmark:splatfacto:balanced -- --run --skip-sam --publish",
    "npm run audit:spark-trained-route",
  ];
}

function publicAssetPath(publicDir, assetPath) {
  if (!assetPath) return "";
  const normalized = String(assetPath).replace(/^\/+/, "");
  const relative = normalized.startsWith("samples/") ? normalized : `samples/${normalized}`;
  return path.join(publicDir, relative);
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

function sortedCounts(counts) {
  return [...counts.entries()]
    .sort(([left], [right]) => left - right)
    .map(([object_id, count]) => ({ object_id, count }));
}

function arrayBuffer(buffer, start = 0, end = buffer.byteLength) {
  return buffer.buffer.slice(buffer.byteOffset + start, buffer.byteOffset + end);
}

function dataView(buffer) {
  return new DataView(arrayBuffer(buffer));
}

function yesNo(value) {
  return value ? "yes" : "no";
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
}
