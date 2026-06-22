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
  };
}

export function parsePly(buffer) {
  const header = readHeader(buffer);
  if (!header.properties.some((property) => property.name === "x")) {
    throw new Error("PLY 缺少 x/y/z 顶点属性");
  }

  const points =
    header.format === "ascii"
      ? parseAscii(buffer, header)
      : parseBinary(buffer, header);

  return { points };
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
  const points = [];

  for (let row = 0; row < Math.min(header.vertexCount, lines.length); row += 1) {
    const values = lines[row].trim().split(/\s+/);
    const vertex = {};
    header.properties.forEach((property, index) => {
      vertex[property.name] = Number(values[index]);
    });
    points.push(vertexToPoint(vertex));
  }
  return points;
}

function parseBinary(buffer, header) {
  const view = new DataView(buffer);
  const littleEndian = header.format === "binary_little_endian";
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
    points.push(vertexToPoint(vertex));
  }

  if (offset > header.headerEnd + stride * header.vertexCount) {
    throw new Error("PLY binary 解析越界");
  }
  return points;
}

function vertexToPoint(vertex) {
  const objectId =
    vertex.object_id !== undefined
      ? Math.trunc(vertex.object_id)
      : vertex.label !== undefined
        ? Math.trunc(vertex.label)
        : 0;

  const color = originalColor(vertex);
  return {
    x: Number(vertex.x ?? 0),
    y: Number(vertex.y ?? 0),
    z: Number(vertex.z ?? 0),
    opacity: opacityValue(vertex.opacity),
    objectId,
    color,
    objectColor: colorForObject(objectId),
  };
}

function originalColor(vertex) {
  if (
    vertex.red !== undefined &&
    vertex.green !== undefined &&
    vertex.blue !== undefined
  ) {
    return [
      normalizeRgb(vertex.red),
      normalizeRgb(vertex.green),
      normalizeRgb(vertex.blue),
    ];
  }

  if (
    vertex.f_dc_0 !== undefined &&
    vertex.f_dc_1 !== undefined &&
    vertex.f_dc_2 !== undefined
  ) {
    return [
      shToRgb(vertex.f_dc_0),
      shToRgb(vertex.f_dc_1),
      shToRgb(vertex.f_dc_2),
    ];
  }

  return [198, 207, 217];
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

function opacityValue(value) {
  if (value === undefined) return 1;
  if (value >= 0 && value <= 1) return value;
  return 1 / (1 + Math.exp(-clamp(value, -80, 80)));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
