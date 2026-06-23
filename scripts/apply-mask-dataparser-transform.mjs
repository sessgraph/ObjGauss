import { dirname } from "node:path";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const options = parseArgs(args);
const input = options._[0];
if (!input) {
  throw new Error("usage: node scripts/apply-mask-dataparser-transform.mjs <mask-manifest> --dataparser-transform <path> [--output <path>]");
}
if (!options.dataparserTransform) {
  throw new Error("--dataparser-transform is required");
}

const output = options.output ?? input;
const manifest = readJson(input);
const dataparser = readJson(options.dataparserTransform);
const transform = dataparserTransformMatrix(dataparser);
for (const frame of manifest.frames ?? []) {
  if (!Array.isArray(frame.transform_matrix)) {
    throw new Error("mask frame is missing transform_matrix");
  }
  frame.transform_matrix = multiply4(transform, frame.transform_matrix);
}
manifest.dataparser_transform = {
  source: options.dataparserTransform,
  scale: dataparser.scale ?? 1.0,
};

mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(manifest, null, 2)}\n`, "utf-8");
console.log(`manifest=${output}`);
console.log(`frames=${manifest.frames?.length ?? 0}`);
console.log(`dataparser_transform=${options.dataparserTransform}`);

function dataparserTransformMatrix(payload) {
  const raw = payload.transform;
  if (!Array.isArray(raw) || raw.length !== 3) {
    throw new Error("dataparser transform must be a 3x4 matrix");
  }
  const scale = Number(payload.scale ?? 1.0);
  const matrix = [
    [...raw[0].map(Number)],
    [...raw[1].map(Number)],
    [...raw[2].map(Number)],
    [0, 0, 0, 1],
  ];
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      matrix[row][column] *= scale;
    }
  }
  return matrix;
}

function multiply4(left, right) {
  if (!Array.isArray(right) || right.length !== 4) {
    throw new Error("frame transform_matrix must be 4x4");
  }
  return left.map((row) =>
    right[0].map((_, column) =>
      row.reduce((sum, value, index) => sum + value * Number(right[index][column]), 0),
    ),
  );
}

function parseArgs(values) {
  const parsed = { _: [] };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value.startsWith("--")) {
      const key = value.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      const next = values[index + 1];
      if (!next || next.startsWith("--")) {
        throw new Error(`${value} requires a value`);
      }
      parsed[key] = next;
      index += 1;
    } else {
      parsed._.push(value);
    }
  }
  return parsed;
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf-8"));
}
