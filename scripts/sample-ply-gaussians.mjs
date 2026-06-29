import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readSync,
  statSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";

const TYPE_INFO = {
  char: { size: 1, reader: (buffer, offset) => buffer.readInt8(offset) },
  int8: { size: 1, reader: (buffer, offset) => buffer.readInt8(offset) },
  uchar: { size: 1, reader: (buffer, offset) => buffer.readUInt8(offset) },
  uint8: { size: 1, reader: (buffer, offset) => buffer.readUInt8(offset) },
  short: { size: 2, reader: (buffer, offset, little) => little ? buffer.readInt16LE(offset) : buffer.readInt16BE(offset) },
  int16: { size: 2, reader: (buffer, offset, little) => little ? buffer.readInt16LE(offset) : buffer.readInt16BE(offset) },
  ushort: { size: 2, reader: (buffer, offset, little) => little ? buffer.readUInt16LE(offset) : buffer.readUInt16BE(offset) },
  uint16: { size: 2, reader: (buffer, offset, little) => little ? buffer.readUInt16LE(offset) : buffer.readUInt16BE(offset) },
  int: { size: 4, reader: (buffer, offset, little) => little ? buffer.readInt32LE(offset) : buffer.readInt32BE(offset) },
  int32: { size: 4, reader: (buffer, offset, little) => little ? buffer.readInt32LE(offset) : buffer.readInt32BE(offset) },
  uint: { size: 4, reader: (buffer, offset, little) => little ? buffer.readUInt32LE(offset) : buffer.readUInt32BE(offset) },
  uint32: { size: 4, reader: (buffer, offset, little) => little ? buffer.readUInt32LE(offset) : buffer.readUInt32BE(offset) },
  float: { size: 4, reader: (buffer, offset, little) => little ? buffer.readFloatLE(offset) : buffer.readFloatBE(offset) },
  float32: { size: 4, reader: (buffer, offset, little) => little ? buffer.readFloatLE(offset) : buffer.readFloatBE(offset) },
  double: { size: 8, reader: (buffer, offset, little) => little ? buffer.readDoubleLE(offset) : buffer.readDoubleBE(offset) },
  float64: { size: 8, reader: (buffer, offset, little) => little ? buffer.readDoubleLE(offset) : buffer.readDoubleBE(offset) },
};

const DEFAULT_CHUNK_ROWS = 65536;
const args = parseArgs(process.argv.slice(2));
const inputPath = requiredString(args.input ?? args["input-ply"] ?? args.ply, "--input");
const outputPath = requiredString(args.output ?? args["output-ply"], "--output");
const targetGaussians = positiveInteger(
  args.targetGaussians ?? args["target-gaussians"] ?? args.count,
  "--target-gaussians",
);
const manifestPath = String(
  args.manifestOutput ??
    args["manifest-output"] ??
    outputPath.replace(/\.ply$/i, ".sample-manifest.json"),
);
const chunkRows = positiveInteger(args.chunkRows ?? args["chunk-rows"] ?? DEFAULT_CHUNK_ROWS, "--chunk-rows");
const overwrite = flagEnabled(args.overwrite);
const label = optionalString(args.label);

const summary = samplePly({
  inputPath,
  outputPath,
  manifestPath,
  targetGaussians,
  chunkRows,
  overwrite,
  label,
});

console.log(
  `sample_ply=passed source=${JSON.stringify(summary.source.path)} ` +
    `output=${JSON.stringify(summary.output.path)} targetGaussians=${summary.output.vertexCount} ` +
    `rowStride=${summary.layout.rowStride} sha256=${summary.output.sha256} ` +
    `manifest=${JSON.stringify(summary.manifestPath)}`,
);

function samplePly({ inputPath, outputPath, manifestPath, targetGaussians, chunkRows, overwrite, label }) {
  const source = path.resolve(inputPath);
  const output = path.resolve(outputPath);
  const manifest = path.resolve(manifestPath);
  if (!existsSync(source)) throw new Error(`input PLY does not exist: ${source}`);
  if (!overwrite && existsSync(output)) throw new Error(`output PLY already exists; pass --overwrite: ${output}`);
  if (!overwrite && existsSync(manifest)) throw new Error(`manifest already exists; pass --overwrite: ${manifest}`);

  const sourceStats = statSync(source);
  const header = readPlyHeader(source);
  if (header.format === "ascii") {
    throw new Error("ASCII PLY sampling is not supported; use binary_little_endian or binary_big_endian");
  }
  if (!["binary_little_endian", "binary_big_endian"].includes(header.format)) {
    throw new Error(`unsupported PLY format: ${header.format}`);
  }
  if (header.extraElements.length > 0) {
    throw new Error(`only vertex-only PLY files are supported; extra elements: ${header.extraElements.join(",")}`);
  }
  if (targetGaussians > header.vertexCount) {
    throw new Error(`target gaussians ${targetGaussians} exceeds source vertex count ${header.vertexCount}`);
  }

  const rowStride = header.properties.reduce((total, property) => {
    const info = TYPE_INFO[property.type];
    if (!info) throw new Error(`unsupported PLY property type: ${property.type}`);
    return total + info.size;
  }, 0);
  const expectedBytes = header.headerEnd + header.vertexCount * rowStride;
  if (sourceStats.size < expectedBytes) {
    throw new Error(`PLY body is shorter than expected: ${sourceStats.size} < ${expectedBytes}`);
  }

  mkdirSync(path.dirname(output), { recursive: true });
  mkdirSync(path.dirname(manifest), { recursive: true });
  const outputHeader = rewriteVertexCount(header.text, header.vertexCount, targetGaussians);
  const outputHash = createHash("sha256");
  const objectLayout = propertyLayout(header.properties, "object_id");
  const objectCounts = new Map();
  const littleEndian = header.format === "binary_little_endian";
  const inputFd = openSync(source, "r");
  const outputFd = openSync(output, "w");
  let writtenRows = 0;

  try {
    const headerBytes = Buffer.from(outputHeader, "ascii");
    writeSync(outputFd, headerBytes);
    outputHash.update(headerBytes);

    const rowsPerChunk = Math.max(1, chunkRows);
    const inputBuffer = Buffer.allocUnsafe(rowsPerChunk * rowStride);
    const outputBuffer = Buffer.allocUnsafe(Math.min(rowsPerChunk, targetGaussians) * rowStride);
    let outputOffset = 0;
    let nextSampleOrdinal = 0;
    let nextSampleIndex = sampleIndex(nextSampleOrdinal, header.vertexCount, targetGaussians);

    for (let chunkStart = 0; chunkStart < header.vertexCount && nextSampleOrdinal < targetGaussians; chunkStart += rowsPerChunk) {
      const currentRows = Math.min(rowsPerChunk, header.vertexCount - chunkStart);
      const currentBytes = currentRows * rowStride;
      const bytesRead = readSync(inputFd, inputBuffer, 0, currentBytes, header.headerEnd + chunkStart * rowStride);
      if (bytesRead !== currentBytes) {
        throw new Error(`short read at source row ${chunkStart}: ${bytesRead}/${currentBytes}`);
      }
      const chunkEnd = chunkStart + currentRows;
      while (nextSampleOrdinal < targetGaussians && nextSampleIndex < chunkEnd) {
        const sourceOffset = (nextSampleIndex - chunkStart) * rowStride;
        inputBuffer.copy(outputBuffer, outputOffset, sourceOffset, sourceOffset + rowStride);
        if (objectLayout) {
          const objectId = Math.trunc(
            objectLayout.info.reader(inputBuffer, sourceOffset + objectLayout.offset, littleEndian),
          );
          objectCounts.set(objectId, (objectCounts.get(objectId) ?? 0) + 1);
        }
        outputOffset += rowStride;
        writtenRows += 1;
        if (outputOffset === outputBuffer.length) {
          writeChunk(outputFd, outputHash, outputBuffer, outputOffset);
          outputOffset = 0;
        }
        nextSampleOrdinal += 1;
        nextSampleIndex = sampleIndex(nextSampleOrdinal, header.vertexCount, targetGaussians);
      }
    }
    if (outputOffset > 0) writeChunk(outputFd, outputHash, outputBuffer, outputOffset);
  } finally {
    closeSync(inputFd);
    closeSync(outputFd);
  }

  if (writtenRows !== targetGaussians) {
    throw new Error(`sampled ${writtenRows} rows, expected ${targetGaussians}`);
  }

  const outputStats = statSync(output);
  const report = {
    schema: "objgauss-sampled-ply-manifest-v1",
    generatedAt: new Date().toISOString(),
    label: label || null,
    source: {
      path: source,
      vertexCount: header.vertexCount,
      byteSize: sourceStats.size,
    },
    output: {
      path: output,
      vertexCount: targetGaussians,
      byteSize: outputStats.size,
      sha256: outputHash.digest("hex"),
    },
    layout: {
      format: header.format,
      rowStride,
      properties: header.properties,
      headerEnd: header.headerEnd,
      outputHeaderBytes: Buffer.byteLength(outputHeader, "ascii"),
    },
    samplePolicy: {
      mode: "deterministic-uniform-midpoint-index-v1",
      formula: "floor((ordinal + 0.5) * sourceVertexCount / targetVertexCount)",
      targetGaussians,
      chunkRows,
    },
    objectCounts: [...objectCounts.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([objectId, count]) => ({ object_id: objectId, count })),
  };
  writeFileSync(manifest, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  return { ...report, manifestPath: manifest };
}

function readPlyHeader(filePath) {
  const fd = openSync(filePath, "r");
  try {
    const buffer = Buffer.alloc(1024 * 1024);
    const byteCount = readSync(fd, buffer, 0, buffer.length, 0);
    const headerEnd = findHeaderEnd(buffer, byteCount);
    if (headerEnd < 0) throw new Error("PLY header is larger than 1 MiB or missing end_header");
    const text = buffer.subarray(0, headerEnd).toString("ascii");
    const lines = text.split(/\r?\n/);
    if (lines[0] !== "ply") throw new Error("not a PLY file");
    let format = "";
    let currentElement = "";
    let vertexCount = 0;
    const properties = [];
    const extraElements = [];
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (!parts[0]) continue;
      if (parts[0] === "format") {
        format = parts[1] ?? "";
      } else if (parts[0] === "element") {
        currentElement = parts[1] ?? "";
        if (currentElement === "vertex") {
          vertexCount = Number(parts[2] ?? 0);
        } else if (currentElement) {
          extraElements.push(currentElement);
        }
      } else if (parts[0] === "property" && currentElement === "vertex") {
        if (parts[1] === "list") throw new Error("list vertex properties are not supported");
        properties.push({ type: parts[1], name: parts[2] });
      }
    }
    if (!Number.isInteger(vertexCount) || vertexCount <= 0) {
      throw new Error("PLY header is missing a positive vertex count");
    }
    return { text, format, vertexCount, properties, extraElements, headerEnd };
  } finally {
    closeSync(fd);
  }
}

function findHeaderEnd(buffer, byteCount) {
  for (let index = 0; index < byteCount - 10; index += 1) {
    if (buffer.subarray(index, index + 10).toString("ascii") !== "end_header") continue;
    let cursor = index + 10;
    if (buffer[cursor] === 13) cursor += 1;
    if (buffer[cursor] === 10) cursor += 1;
    return cursor;
  }
  return -1;
}

function rewriteVertexCount(headerText, sourceCount, targetCount) {
  const pattern = new RegExp(`^element\\s+vertex\\s+${sourceCount}\\s*$`, "m");
  if (!pattern.test(headerText)) throw new Error("could not rewrite PLY vertex count");
  return headerText.replace(pattern, `element vertex ${targetCount}`);
}

function propertyLayout(properties, name) {
  let offset = 0;
  for (const property of properties) {
    const info = TYPE_INFO[property.type];
    if (!info) throw new Error(`unsupported PLY property type: ${property.type}`);
    if (property.name === name) return { property, info, offset };
    offset += info.size;
  }
  return null;
}

function sampleIndex(ordinal, sourceCount, targetCount) {
  if (ordinal >= targetCount) return sourceCount;
  return Math.floor(((ordinal + 0.5) * sourceCount) / targetCount);
}

function writeChunk(fd, hash, buffer, byteCount) {
  const chunk = buffer.subarray(0, byteCount);
  writeSync(fd, chunk);
  hash.update(chunk);
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

function requiredString(value, label) {
  const text = optionalString(value);
  if (!text) throw new Error(`${label} is required`);
  return text;
}

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return "";
  return String(value).trim();
}

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${label} must be a positive integer`);
  return parsed;
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}
