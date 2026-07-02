import {
  decodeQuantizedOgcChunk,
  decodeQuantizedOgcPayload,
  QUANTIZED_OGC_PAYLOAD_SCHEMA,
  QUANTIZED_OGC_RECORD_BYTE_SIZE,
  QUANTIZED_OGC_RECORD_FORMAT,
  validateQuantizedOgcIndex,
} from "../src/ogcDecoder.js";
import {
  browserReadyArtifact,
  MODEL_ARTIFACT_MANIFEST_SCHEMA,
  resolveModelArtifactRoutes,
} from "../src/modelArtifactManifest.js";

const payload = fixturePayload();
const index = fixtureIndex();

validateQuantizedOgcIndex(index);

const decoded = decodeQuantizedOgcPayload(payload, index);
assertEqual(decoded.points.length, 4, "full decode point count");
assertEqual(decoded.metadata.decodedChunks, 2, "decoded chunk count");
assertEqual(decoded.objects.length, 2, "object metadata count");
assertEqual(decoded.lod.schema, "objgauss-object-aware-lod-v1", "top-level LOD schema");

assertPoint(decoded.points[0], {
  x: 0,
  y: 0,
  z: 0,
  color: [10, 20, 30],
  opacity: 128 / 255,
  objectId: 0,
  chunkId: 0,
});
assertPoint(decoded.points[1], {
  x: 1,
  y: 0,
  z: 0,
  color: [40, 50, 60],
  opacity: 255 / 255,
  objectId: 0,
  chunkId: 0,
});
assertPoint(decoded.points[2], {
  x: 10,
  y: 5,
  z: -1,
  color: [70, 80, 90],
  opacity: 64 / 255,
  objectId: 1,
  chunkId: 1,
});
assertPoint(decoded.points[3], {
  x: 10,
  y: 6,
  z: -1,
  color: [100, 110, 120],
  opacity: 32 / 255,
  objectId: 1,
  chunkId: 1,
});

const lodChunk = decodeQuantizedOgcChunk(payload, index, 0, { lodLevel: 1 });
assertEqual(lodChunk.points.length, 1, "LOD chunk point count");
assertEqual(lodChunk.points[0].lodLevel, 1, "LOD point marker");
assertEqual(lodChunk.chunk.lod.schema, "objgauss-object-aware-lod-v1", "chunk LOD metadata preserved");

const routes = resolveModelArtifactRoutes(fixtureManifest());
assertEqual(routes.compressedChunked.role, "compressed_chunked", "compressed chunked route role");
assertEqual(browserReadyArtifact({ modelArtifactRoutes: routes }, "compressed_chunked").path, "/samples/fixture.ogc", "browserReadyArtifact compressed path");

console.log(
  [
    "ogc_decoder_contract=passed",
    `points=${decoded.points.length}`,
    `chunks=${decoded.chunks.length}`,
    `objects=${decoded.objects.length}`,
    `lodLevelPoints=${lodChunk.points.length}`,
    `compressedRoute=${routes.compressedChunked.role}`,
  ].join(" "),
);

function fixturePayload() {
  const buffer = new ArrayBuffer(4 * QUANTIZED_OGC_RECORD_BYTE_SIZE);
  const view = new DataView(buffer);
  writeRecord(view, 0, { x: 0, y: 0, z: 0, red: 10, green: 20, blue: 30, opacity: 128 });
  writeRecord(view, 1, { x: 65535, y: 0, z: 0, red: 40, green: 50, blue: 60, opacity: 255 });
  writeRecord(view, 2, { x: 0, y: 0, z: 0, red: 70, green: 80, blue: 90, opacity: 64 });
  writeRecord(view, 3, { x: 0, y: 65535, z: 0, red: 100, green: 110, blue: 120, opacity: 32 });
  return buffer;
}

function writeRecord(view, row, record) {
  const offset = row * QUANTIZED_OGC_RECORD_BYTE_SIZE;
  view.setUint16(offset, record.x, true);
  view.setUint16(offset + 2, record.y, true);
  view.setUint16(offset + 4, record.z, true);
  view.setUint8(offset + 6, record.red);
  view.setUint8(offset + 7, record.green);
  view.setUint8(offset + 8, record.blue);
  view.setUint8(offset + 9, record.opacity);
}

function fixtureIndex() {
  return {
    schema: "objgauss-chunk-index-v1",
    sort_key: "object_id+morton_xyz",
    gaussian_count: 4,
    object_count: 2,
    payload: {
      schema: QUANTIZED_OGC_PAYLOAD_SCHEMA,
      path: "/samples/fixture.ogc",
      format: ".ogc",
      record_format: QUANTIZED_OGC_RECORD_FORMAT,
      record_byte_size: QUANTIZED_OGC_RECORD_BYTE_SIZE,
      byte_size: 4 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
      sha256: "0".repeat(64),
    },
    lod: {
      schema: "objgauss-object-aware-lod-v1",
      levels: [
        { level: 0, ratio: 1.0, gaussian_count: 4 },
        { level: 1, ratio: 0.5, gaussian_count: 2 },
      ],
    },
    objects: [
      { object_id: 0, gaussian_count: 2, chunk_ids: [0] },
      { object_id: 1, gaussian_count: 2, chunk_ids: [1] },
    ],
    chunks: [
      {
        chunk_id: 0,
        object_id: 0,
        gaussian_count: 2,
        record_count: 2,
        record_format: QUANTIZED_OGC_RECORD_FORMAT,
        byte_offset: 0,
        byte_length: 2 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
        aabb_min: [0, 0, 0],
        aabb_max: [1, 0, 0],
        lod: {
          schema: "objgauss-object-aware-lod-v1",
          levels: [
            {
              level: 0,
              ratio: 1.0,
              record_count: 2,
              byte_offset: 0,
              byte_length: 2 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
            },
            {
              level: 1,
              ratio: 0.5,
              record_count: 1,
              byte_offset: 0,
              byte_length: QUANTIZED_OGC_RECORD_BYTE_SIZE,
            },
          ],
        },
      },
      {
        chunk_id: 1,
        object_id: 1,
        gaussian_count: 2,
        record_count: 2,
        record_format: QUANTIZED_OGC_RECORD_FORMAT,
        byte_offset: 2 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
        byte_length: 2 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
        aabb_min: [10, 5, -1],
        aabb_max: [10, 6, -1],
        lod: {
          schema: "objgauss-object-aware-lod-v1",
          levels: [
            {
              level: 0,
              ratio: 1.0,
              record_count: 2,
              byte_offset: 2 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
              byte_length: 2 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
            },
            {
              level: 1,
              ratio: 0.5,
              record_count: 1,
              byte_offset: 2 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
              byte_length: QUANTIZED_OGC_RECORD_BYTE_SIZE,
            },
          ],
        },
      },
    ],
  };
}

function fixtureManifest() {
  return {
    schema: MODEL_ARTIFACT_MANIFEST_SCHEMA,
    manifest_id: "fixture-model-artifacts",
    asset_id: "fixture",
    name: "Fixture",
    artifacts: [
      {
        role: "quick_splat",
        path: "/samples/fixture.splat",
        format: ".splat",
        delivery_tier: "browser_quick",
        browser_ready: true,
      },
      {
        role: "compressed_chunked",
        path: "/samples/fixture.ogc",
        format: ".ogc",
        delivery_tier: "browser_edit",
        browser_ready: true,
        chunk_index: { path: "/samples/fixture.index.json" },
      },
    ],
  };
}

function assertPoint(actual, expected) {
  assertClose(actual.x, expected.x, "x");
  assertClose(actual.y, expected.y, "y");
  assertClose(actual.z, expected.z, "z");
  assertClose(actual.opacity, expected.opacity, "opacity");
  assertEqual(actual.objectId, expected.objectId, "objectId");
  assertEqual(actual.chunkId, expected.chunkId, "chunkId");
  assertEqual(JSON.stringify(actual.color), JSON.stringify(expected.color), "color");
  if (!Array.isArray(actual.objectColor) || actual.objectColor.length !== 3) {
    throw new Error("decoded point did not include objectColor");
  }
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertClose(actual, expected, label, epsilon = 1e-6) {
  if (Math.abs(actual - expected) > epsilon) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}
