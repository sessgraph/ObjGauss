import { colorForObject } from "./palette.js";

const SH_C0 = 0.28209479177387814;

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

export async function parsePlyFile(file) {
  const buffer = await file.arrayBuffer();
  const cloud = parsePly(buffer);
  return {
    name: file.name,
    points: cloud.points,
    shRestCoefficients: cloud.shRestCoefficients,
    shRestCoefficientCount: cloud.shRestCoefficientCount,
    shDegree: cloud.shDegree,
    splatSource: {
      fileBytes: buffer.slice(0),
      fileName: file.name,
    },
  };
}

export function parsePly(buffer) {
  const header = readHeader(buffer);
  if (!header.properties.some((property) => property.name === "x")) {
    throw new Error("PLY 缺少 x/y/z 顶点属性");
  }

  const cloud =
    header.format === "ascii"
      ? parseAscii(buffer, header)
      : parseBinary(buffer, header);

  return cloud;
}

function readHeader(buffer) {
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
  if (end < 0) {
    throw new Error("没有找到 PLY end_header");
  }

  const text = new TextDecoder("ascii").decode(buffer.slice(0, end));
  const lines = text.split(/\r?\n/);
  if (lines[0] !== "ply") {
    throw new Error("不是有效 PLY 文件");
  }

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
      if (currentElement === "vertex") {
        vertexCount = Number(parts[2]);
      }
    } else if (parts[0] === "property" && currentElement === "vertex") {
      if (parts[1] === "list") {
        throw new Error("查看器只支持标量 vertex PLY，不支持 list 属性");
      }
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  if (!["ascii", "binary_little_endian", "binary_big_endian"].includes(format)) {
    throw new Error(`不支持的 PLY 格式: ${format}`);
  }

  return {
    format,
    vertexCount,
    properties,
    headerEnd: end,
  };
}

function parseAscii(buffer, header) {
  const text = new TextDecoder("utf-8").decode(buffer.slice(header.headerEnd));
  const lines = text.trim().split(/\r?\n/);
  const vertexCount = Math.min(header.vertexCount, lines.length);
  const shRest = shRestPropertyLayout(header.properties);
  const shRestCoefficients =
    shRest.coefficientCount > 0
      ? new Float32Array(vertexCount * shRest.coefficientCount)
      : null;
  const points = [];

  for (let row = 0; row < vertexCount; row += 1) {
    const values = lines[row].trim().split(/\s+/);
    const vertex = {};
    header.properties.forEach((property, index) => {
      vertex[property.name] = Number(values[index]);
    });
    packShRestCoefficients({ vertex, row, shRest, shRestCoefficients });
    points.push(vertexToPoint(vertex, shRest.coefficientCount));
  }
  return {
    points,
    shRestCoefficients,
    shRestCoefficientCount: shRest.coefficientCount,
    shDegree: shDegreeFromRestCoefficientCount(shRest.coefficientCount),
  };
}

function parseBinary(buffer, header) {
  const view = new DataView(buffer);
  const littleEndian = header.format === "binary_little_endian";
  const shRest = shRestPropertyLayout(header.properties);
  const shRestCoefficients =
    shRest.coefficientCount > 0
      ? new Float32Array(header.vertexCount * shRest.coefficientCount)
      : null;
  const stride = header.properties.reduce((total, property) => {
    const info = TYPE_INFO[property.type];
    if (!info) throw new Error(`不支持的 PLY 属性类型: ${property.type}`);
    return total + info.size;
  }, 0);

  const points = [];
  let offset = header.headerEnd;
  for (let row = 0; row < header.vertexCount; row += 1) {
    const vertex = {};
    for (const property of header.properties) {
      const info = TYPE_INFO[property.type];
      vertex[property.name] = view[info.getter](offset, littleEndian);
      offset += info.size;
    }
    packShRestCoefficients({ vertex, row, shRest, shRestCoefficients });
    points.push(vertexToPoint(vertex, shRest.coefficientCount));
  }

  if (offset > header.headerEnd + stride * header.vertexCount) {
    throw new Error("PLY binary 解析越界");
  }
  return {
    points,
    shRestCoefficients,
    shRestCoefficientCount: shRest.coefficientCount,
    shDegree: shDegreeFromRestCoefficientCount(shRest.coefficientCount),
  };
}

function vertexToPoint(vertex, shRestCoefficientCount = null) {
  const objectId =
    vertex.object_id !== undefined
      ? Math.trunc(vertex.object_id)
      : vertex.label !== undefined
        ? Math.trunc(vertex.label)
        : 0;

  const original = originalColor(vertex);
  const scale3 = gaussianScale3(vertex);
  const rotationQuaternion = gaussianRotationQuaternion(vertex);
  const shRest = shRestMetadata(vertex, shRestCoefficientCount);
  return {
    x: Number(vertex.x ?? 0),
    y: Number(vertex.y ?? 0),
    z: Number(vertex.z ?? 0),
    opacity: opacityValue(vertex.opacity),
    scale: gaussianScale(scale3),
    scale3,
    rotation: gaussianRotation(rotationQuaternion),
    rotationQuaternion,
    objectId,
    color: original.rgb,
    colorSource: original.source,
    shDc: original.shDc,
    shRestCoefficientCount: shRest.coefficientCount,
    shDegree: shRest.degree,
    objectColor: colorForObject(objectId),
  };
}

function originalColor(vertex) {
  const shDc = shDcCoefficients(vertex);
  if (
    vertex.red !== undefined &&
    vertex.green !== undefined &&
    vertex.blue !== undefined
  ) {
    return {
      rgb: [
        normalizeRgb(vertex.red),
        normalizeRgb(vertex.green),
        normalizeRgb(vertex.blue),
      ],
      source: "rgb",
      shDc,
    };
  }

  if (shDc) {
    return {
      rgb: [shToRgb(shDc[0]), shToRgb(shDc[1]), shToRgb(shDc[2])],
      source: "sh-dc",
      shDc,
    };
  }

  return { rgb: [198, 207, 217], source: "fallback", shDc: null };
}

function shDcCoefficients(vertex) {
  if (
    vertex.f_dc_0 === undefined ||
    vertex.f_dc_1 === undefined ||
    vertex.f_dc_2 === undefined
  ) {
    return null;
  }
  const shDc = [Number(vertex.f_dc_0), Number(vertex.f_dc_1), Number(vertex.f_dc_2)];
  return shDc.every(Number.isFinite) ? shDc : null;
}

function normalizeRgb(value) {
  if (value <= 1) {
    return Math.round(clamp(value, 0, 1) * 255);
  }
  return Math.round(clamp(value, 0, 255));
}

function shToRgb(value) {
  return Math.round(clamp(value * SH_C0 + 0.5, 0, 1) * 255);
}

function shRestMetadata(vertex, shRestCoefficientCount = null) {
  if (Number.isFinite(shRestCoefficientCount) && shRestCoefficientCount >= 0) {
    const coefficientCount = Math.trunc(shRestCoefficientCount);
    return {
      coefficientCount,
      degree: shDegreeFromRestCoefficientCount(coefficientCount),
    };
  }
  let coefficientCount = 0;
  for (const [name, value] of Object.entries(vertex)) {
    if (!name.startsWith("f_rest_")) continue;
    const coefficientIndex = Number(name.slice(7));
    if (!Number.isInteger(coefficientIndex) || coefficientIndex < 0) continue;
    if (Number.isFinite(Number(value))) {
      coefficientCount += 1;
    }
  }
  return {
    coefficientCount,
    degree: shDegreeFromRestCoefficientCount(coefficientCount),
  };
}

function shRestPropertyLayout(properties) {
  const restProperties = properties
    .map((property) => ({
      ...property,
      restIndex: shRestPropertyIndex(property.name),
    }))
    .filter((property) => property.restIndex !== null)
    .sort((left, right) => left.restIndex - right.restIndex);

  return {
    properties: restProperties,
    coefficientCount: restProperties.length,
  };
}

function shRestPropertyIndex(name) {
  if (!name.startsWith("f_rest_")) return null;
  const index = Number(name.slice(7));
  if (!Number.isInteger(index) || index < 0) return null;
  return index;
}

function packShRestCoefficients({ vertex, row, shRest, shRestCoefficients }) {
  if (!shRestCoefficients || shRest.coefficientCount <= 0) return;
  const rowOffset = row * shRest.coefficientCount;
  shRest.properties.forEach((property, column) => {
    const value = Number(vertex[property.name] ?? 0);
    shRestCoefficients[rowOffset + column] = Number.isFinite(value) ? value : 0;
  });
}

function shDegreeFromRestCoefficientCount(count) {
  if (count >= 45) return 3;
  if (count >= 24) return 2;
  if (count >= 9) return 1;
  return 0;
}

function opacityValue(value) {
  if (value === undefined) return 1;
  if (value >= 0 && value <= 1) return value;
  return 1 / (1 + Math.exp(-clamp(value, -80, 80)));
}

function gaussianScale(scale3) {
  const values = [...scale3].sort((left, right) => right - left);
  return [values[0] ?? 0.018, values[1] ?? values[0] ?? 0.018];
}

function gaussianScale3(vertex) {
  const fallback = 0.018;
  return [vertex.scale_0, vertex.scale_1, vertex.scale_2].map((value) =>
    scaleValue(value, fallback),
  );
}

function scaleValue(value, fallback) {
  if (value === undefined || !Number.isFinite(Number(value))) return fallback;
  const numeric = Number(value);
  const scale = numeric < 0 ? Math.exp(clamp(numeric, -16, 4)) : numeric;
  return clamp(scale, 0.0006, 0.35);
}

function gaussianRotation(rotationQuaternion) {
  if (!rotationQuaternion) return 0;
  const [nw, nx, ny, nz] = rotationQuaternion;
  return Math.atan2(2 * (nw * nz + nx * ny), 1 - 2 * (ny * ny + nz * nz));
}

function gaussianRotationQuaternion(vertex) {
  if (
    vertex.rot_0 === undefined ||
    vertex.rot_1 === undefined ||
    vertex.rot_2 === undefined ||
    vertex.rot_3 === undefined
  ) {
    return null;
  }

  const raw = [vertex.rot_0, vertex.rot_1, vertex.rot_2, vertex.rot_3].map(Number);
  const components = raw.every((value) => Number.isFinite(value) && value >= -1 && value <= 1)
    ? raw
    : raw.map(quaternionByte);
  const [w, x, y, z] = components;
  const length = Math.hypot(w, x, y, z);
  if (!Number.isFinite(length) || length <= 0.0001) return null;

  return [w / length, x / length, y / length, z / length];
}

function quaternionByte(value) {
  return clamp(Number(value ?? 128) / 255, 0, 1) * 2 - 1;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
