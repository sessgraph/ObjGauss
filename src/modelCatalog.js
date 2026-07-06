import {
  QUANTIZED_OGC_PAYLOAD_SCHEMA,
  QUANTIZED_OGC_RECORD_BYTE_SIZE,
  QUANTIZED_OGC_RECORD_FORMAT,
} from "./ogcDecoder.js";
import { MODEL_ARTIFACT_MANIFEST_SCHEMA } from "./modelArtifactManifest.js";

const OGC_DEBUG_PAYLOAD_BASE64 =
  "AIAAAACA7GNY5uAuUEYgTv+VYM0gywhSEKTuzXfX8FVwlJi30FtwyMivsLOwNvi0U9IAgP//AID/6Z7rAIAAAACAJ8LL4YA+mDoIUnHc4s1Qw2hCgLtTod/SOEpYmFDDTNSsxpi3mLeYOpjYgtcAgP//AIDJ9pbo";

const OGC_DEBUG_INDEX = {
  schema: "objgauss-chunk-index-v1",
  sort_key: "object_id+morton_xyz",
  gaussian_count: 12,
  object_count: 2,
  chunk_size_target: 6,
  payload: {
    schema: QUANTIZED_OGC_PAYLOAD_SCHEMA,
    path: "inline://ogc-debug.ogc",
    format: ".ogc",
    record_format: QUANTIZED_OGC_RECORD_FORMAT,
    record_byte_size: QUANTIZED_OGC_RECORD_BYTE_SIZE,
    byte_size: 120,
    sha256: "673f6782c9c9e4d50318e0dcb41276b2167084a1da8894e820e0e8487ca7ecd9",
  },
  compression: {
    codec: "objgauss-ogc-prototype",
    layout: "object-aware-chunked-local-quantized",
    quantization: {
      schema: "objgauss-local-quantization-v1",
      policy: "chunk-aabb-uint16-rgb8-opacity8-v0",
      status: "actual_payload_prototype",
    },
  },
  lod: {
    schema: "objgauss-object-aware-lod-v1",
    levels: [
      { level: 0, ratio: 1.0, gaussian_count: 12 },
      { level: 1, ratio: 0.5, gaussian_count: 6 },
    ],
  },
  object_id_coverage: {
    field: "object_id",
    mode: "complete",
    has_object_ids: true,
    object_count: 2,
  },
  objects: [
    { object_id: 0, gaussian_count: 6, chunk_ids: [0] },
    { object_id: 1, gaussian_count: 6, chunk_ids: [1] },
  ],
  chunks: [
    {
      chunk_id: 0,
      object_id: 0,
      gaussian_count: 6,
      record_count: 6,
      record_format: QUANTIZED_OGC_RECORD_FORMAT,
      byte_offset: 0,
      byte_length: 6 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
      aabb_min: [-0.6, 0, -0.35],
      aabb_max: [0.6, 1.2, 0.35],
      lod: {
        schema: "objgauss-object-aware-lod-v1",
        levels: [
          {
            level: 0,
            ratio: 1.0,
            record_count: 6,
            byte_offset: 0,
            byte_length: 6 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
          },
          {
            level: 1,
            ratio: 0.5,
            record_count: 3,
            byte_offset: 0,
            byte_length: 3 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
          },
        ],
      },
    },
    {
      chunk_id: 1,
      object_id: 1,
      gaussian_count: 6,
      record_count: 6,
      record_format: QUANTIZED_OGC_RECORD_FORMAT,
      byte_offset: 6 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
      byte_length: 6 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
      aabb_min: [1.4, 0, -0.45],
      aabb_max: [2.4, 1.3, 0.45],
      lod: {
        schema: "objgauss-object-aware-lod-v1",
        levels: [
          {
            level: 0,
            ratio: 1.0,
            record_count: 6,
            byte_offset: 6 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
            byte_length: 6 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
          },
          {
            level: 1,
            ratio: 0.5,
            record_count: 3,
            byte_offset: 6 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
            byte_length: 3 * QUANTIZED_OGC_RECORD_BYTE_SIZE,
          },
        ],
      },
    },
  ],
};

const DEFAULT_DISPLAY_SLOT = "lego-object-segmentation-preview";

const RAW_MODEL_CATALOG = [
  {
    id: "real-sample-v2-sample-aware-lego",
    name: "真实样例 V2 样本自适应 Lego 预览",
    label: "真实样例 V2",
    sourcePath: "/samples/objgauss-real-sample-v2-sample-aware-lego.ply",
    loadMode: "eager",
    kind: "object-aware-ply",
    kindLabel: "对象感知 PLY",
    stage: "sample-aware-preview",
    displaySlot: DEFAULT_DISPLAY_SLOT,
    displayOrder: 0,
    updatedAt: "2026-07-06",
    description: "本机生成的样本自适应强化策略预览 PLY，用于查看当前阶段对象分割效果；缺少该文件时自动回退到 Lego alpha。",
    objectCount: 4,
    galleryPosition: [-2.35, 0, -3.58],
    accent: "#35d0c8",
    displayScale: 2.46,
    pointSize: 0.036,
    maxDisplayPoints: 50000,
    optionalLocalPreview: true,
    fallbackModelId: "lego-alpha",
    compression: {
      layout: "real-sample-v2-sample-aware-preview-ply",
      status: "local-generated-preview",
      chunkRoot: "/models/real-sample-v2-sample-aware-lego/objects/",
    },
  },
  {
    id: "plush",
    name: "Plush object scene",
    label: "Plush",
    sourcePath: "/samples/plush_objects.ply",
    loadMode: "eager",
    kind: "object-aware-ply",
    stage: "processed",
    objectCount: 3,
    galleryPosition: [-6.2, 0, -1.35],
    accent: "#ff6b7a",
    displayScale: 2.42,
    pointSize: 0.034,
    minObjectDisplaySpan: 1.22,
    maxObjectDisplayBoost: 5.2,
    maxDisplayPoints: 34000,
    compression: {
      layout: "per-object-corepoint-chunks",
      status: "planned",
      chunkRoot: "/models/plush/objects/",
    },
  },
  {
    id: "plush-v1",
    name: "ObjGauss v1 closure",
    label: "Plush v1",
    sourcePath: "/samples/plush_v1_objects.ply",
    loadMode: "eager",
    kind: "object-aware-ply",
    stage: "processed",
    objectCount: 4,
    galleryPosition: [-3.15, 0, 1.55],
    accent: "#f7b267",
    displayScale: 2.34,
    pointSize: 0.034,
    minObjectDisplaySpan: 1.08,
    maxObjectDisplayBoost: 4.6,
    maxDisplayPoints: 34000,
    compression: {
      layout: "per-object-corepoint-chunks",
      status: "planned",
      chunkRoot: "/models/plush-v1/objects/",
    },
  },
  {
    id: "lego-alpha",
    name: "Lego alpha closure",
    label: "Lego alpha",
    sourcePath: "/samples/lego_alpha_v1_objects.ply",
    loadMode: "eager",
    kind: "object-aware-ply",
    stage: "processed",
    displaySlot: DEFAULT_DISPLAY_SLOT,
    displayOrder: 0,
    updatedAt: "2026-06-22",
    objectCount: 5,
    galleryPosition: [0.05, 0, -1.25],
    accent: "#27c2cb",
    displayScale: 2.46,
    pointSize: 0.035,
    minObjectDisplaySpan: 1.08,
    maxObjectDisplayBoost: 4.8,
    maxDisplayPoints: 26000,
    compression: {
      layout: "per-object-corepoint-chunks",
      status: "planned",
      chunkRoot: "/models/lego-alpha/objects/",
    },
  },
  {
    id: "polyhaven-chair",
    name: "Poly Haven chair",
    label: "Chair",
    sourcePath: "/samples/polyhaven_chair_demo_objects.ply",
    loadMode: "eager",
    kind: "object-aware-ply",
    stage: "processed",
    objectCount: 1,
    galleryPosition: [3.25, 0, 1.42],
    accent: "#88d498",
    displayScale: 2.14,
    pointSize: 0.033,
    maxDisplayPoints: 50000,
    compression: {
      layout: "per-object-corepoint-chunks",
      status: "planned",
      chunkRoot: "/models/polyhaven-chair/objects/",
    },
  },
  {
    id: "near1m-lego",
    name: "Near-1M Lego candidate",
    label: "Near-1M",
    sourcePath: "/samples/nerf_lego_trained_near1m_random1300k_objects.ply",
    loadMode: "compressed-placeholder",
    kind: "diagnostic-ply",
    stage: "diagnostic",
    objectCount: 5,
    galleryPosition: [6.35, 0, -1.42],
    accent: "#8e6cff",
    displayScale: 2.28,
    pointSize: 0.042,
    placeholderPointsPerObject: 820,
    maxDisplayPoints: 45000,
    compression: {
      layout: "per-object-corepoint-chunks",
      status: "prototype",
      chunkRoot: "/models/near1m-lego/objects/",
    },
  },
  {
    id: "ogc-debug",
    name: "OGC chunk debug fixture",
    label: "OGC chunks",
    loadMode: "ogc-chunked",
    kind: "compressed-chunked-ogc",
    stage: "browser-delivery-fixture",
    objectCount: 2,
    galleryPosition: [0.18, 0, 3.25],
    accent: "#2fd3a6",
    displayScale: 1.88,
    pointSize: 0.058,
    maxDisplayPoints: 1200,
    ogc: {
      lodLevel: 0,
    },
    compression: {
      layout: "object-aware-quantized-ogc-chunks",
      status: "browser-ready",
      chunkRoot: "/models/ogc-debug/objects/",
    },
    modelArtifactManifest: {
      schema: MODEL_ARTIFACT_MANIFEST_SCHEMA,
      manifest_id: "ogc-debug-model-artifacts",
      asset_id: "ogc-debug",
      name: "OGC chunk debug fixture",
      stage: "development",
      source: {
        type: "object_aware_gaussian_codec_fixture",
        input: "inline quantized OGC records",
      },
      license: "fixture",
      counts: {
        gaussians: OGC_DEBUG_INDEX.gaussian_count,
        objects: OGC_DEBUG_INDEX.object_count,
      },
      artifacts: [
        {
          role: "compressed_chunked",
          path: "inline://ogc-debug.ogc",
          format: ".ogc",
          delivery_tier: "browser_edit",
          browser_ready: true,
          gaussian_count: OGC_DEBUG_INDEX.gaussian_count,
          object_count: OGC_DEBUG_INDEX.object_count,
          byte_size: OGC_DEBUG_INDEX.payload.byte_size,
          sha256: OGC_DEBUG_INDEX.payload.sha256,
          chunk_index: {
            schema: OGC_DEBUG_INDEX.schema,
            path: "inline://ogc-debug.index.json",
            chunk_count: OGC_DEBUG_INDEX.chunks.length,
            sort_key: OGC_DEBUG_INDEX.sort_key,
            chunk_size_target: OGC_DEBUG_INDEX.chunk_size_target,
          },
          compression: OGC_DEBUG_INDEX.compression,
          lod: OGC_DEBUG_INDEX.lod,
          object_id_coverage: OGC_DEBUG_INDEX.object_id_coverage,
          inlineIndex: OGC_DEBUG_INDEX,
          payloadBase64: OGC_DEBUG_PAYLOAD_BASE64,
        },
      ],
      quality_evidence: [
        {
          kind: "browser_decoder_contract",
          status: "fixture",
          decoded_gaussians: OGC_DEBUG_INDEX.gaussian_count,
        },
      ],
      limitations: ["Tiny inline fixture for browser OGC delivery wiring; not a trained scene."],
      created_from: {
        source: "OGC-BROWSER-STREAMING-001",
      },
    },
  },
  {
    id: "trainable-mvp-debug",
    name: "Trainable MVP artifact",
    label: "Trainable MVP",
    loadMode: "trainable-artifact",
    kind: "trainable-kernel-model-artifact",
    stage: "debug-fixture",
    objectCount: 2,
    galleryPosition: [-2.65, 0, 3.62],
    accent: "#f3df5d",
    displayScale: 1.92,
    pointSize: 0.082,
    maxDisplayPoints: 128,
    trainableArtifactPath: "/models/trainable-mvp-debug/model-artifact.json",
    compression: {
      layout: "trainable-kernel-artifact-json",
      status: "debug-fixture",
      chunkRoot: "/models/trainable-mvp-debug/objects/",
    },
  },
];

export const MODEL_CATALOG = sortModelCatalog(RAW_MODEL_CATALOG);

export function catalogSummary(models = MODEL_CATALOG) {
  return {
    modelCount: models.length,
    compressedReadyCount: models.filter((model) =>
      ["prototype", "browser-ready"].includes(model.compression?.status),
    ).length,
    processedCount: models.filter((model) => model.stage === "processed").length,
  };
}

export function modelCatalogFromSearch(search = "") {
  const params = new URLSearchParams(String(search ?? "").replace(/^\?/, ""));
  const trainableArtifactPath = trainableArtifactPathFromParams(params);
  const plyPath = plyPathFromParams(params);
  const modelArtifactManifest = modelArtifactManifestFromParams(params);
  const ogcManifest = ogcManifestArtifactFromParams(params);
  const ogcArtifact = ogcArtifactFromParams(params);
  if (!trainableArtifactPath && !plyPath && !modelArtifactManifest && !ogcManifest && !ogcArtifact) return MODEL_CATALOG;
  return sortModelCatalog([
    ...MODEL_CATALOG,
    ...(trainableArtifactPath ? [trainableUrlArtifactModel(trainableArtifactPath)] : []),
    ...(plyPath ? [plyUrlArtifactModel(plyPath)] : []),
    ...(modelArtifactManifest ? [modelArtifactManifestBundleModel(modelArtifactManifest)] : []),
    ...(ogcManifest ? [ogcManifestUrlArtifactModel(ogcManifest)] : []),
    ...(ogcArtifact ? [ogcUrlArtifactModel(ogcArtifact)] : []),
  ]);
}

export function defaultModelIdForCatalog(models = MODEL_CATALOG) {
  return models.find((model) => model.id === "trainable-url-artifact")?.id
    ?? models.find((model) => model.id === "ply-url-artifact")?.id
    ?? models.find((model) => model.id === "model-artifact-manifest")?.id
    ?? models.find((model) => model.id === "ogc-manifest-artifact")?.id
    ?? models.find((model) => model.id === "ogc-url-artifact")?.id
    ?? latestModelForDisplaySlot(models, DEFAULT_DISPLAY_SLOT)?.id
    ?? models[0]?.id
    ?? "";
}

function sortModelCatalog(models = []) {
  return models
    .map((model, index) => ({ index, model }))
    .sort((left, right) => {
      const orderDelta = modelDisplayOrder(left.model) - modelDisplayOrder(right.model);
      if (orderDelta !== 0) return orderDelta;
      if (left.model.displaySlot && left.model.displaySlot === right.model.displaySlot) {
        const updatedDelta = modelUpdatedTime(right.model) - modelUpdatedTime(left.model);
        if (updatedDelta !== 0) return updatedDelta;
      }
      return left.index - right.index;
    })
    .map(({ model }) => model);
}

function latestModelForDisplaySlot(models = [], displaySlot = "") {
  return sortModelCatalog(models.filter((model) => model.displaySlot === displaySlot))[0] ?? null;
}

function modelDisplayOrder(model) {
  const value = Number(model?.displayOrder);
  return Number.isFinite(value) ? value : Number.MAX_SAFE_INTEGER;
}

function modelUpdatedTime(model) {
  const value = Date.parse(model?.updatedAt ?? "");
  return Number.isFinite(value) ? value : 0;
}

function trainableArtifactPathFromParams(params) {
  return sameOriginPathParam(params, ["trainableArtifact", "trainable-artifact"], ".json");
}

function plyPathFromParams(params) {
  return sameOriginPathParam(params, ["ply", "debugPly", "debug-ply", "objectPly", "object-ply"], ".ply");
}

function modelArtifactManifestFromParams(params) {
  const manifestPath = sameOriginPathParam(
    params,
    ["modelArtifactManifest", "model-artifact-manifest", "artifactManifest", "artifact-manifest"],
    ".json",
  );
  if (!manifestPath) return null;
  return {
    manifestPath,
    lodLevel: ogcLodLevelFromParams(params),
    chunkIds: ogcChunkIdsFromParams(params),
  };
}

function ogcManifestArtifactFromParams(params) {
  const manifestPath = sameOriginPathParam(
    params,
    ["ogcManifest", "ogc-manifest"],
    ".json",
  );
  if (!manifestPath) return null;
  return {
    manifestPath,
    lodLevel: ogcLodLevelFromParams(params),
    chunkIds: ogcChunkIdsFromParams(params),
  };
}

function ogcArtifactFromParams(params) {
  const indexPath = sameOriginPathParam(params, ["ogcIndex", "ogc-index"], ".json");
  const payloadPath = sameOriginPathParam(params, ["ogcPayload", "ogc-payload"], ".ogc");
  if (!indexPath || !payloadPath) return null;
  return {
    indexPath,
    payloadPath,
    lodLevel: ogcLodLevelFromParams(params),
    chunkIds: ogcChunkIdsFromParams(params),
  };
}

function trainableUrlArtifactModel(artifactPath) {
  return {
    id: "trainable-url-artifact",
    name: "URL trainable artifact",
    label: "URL Artifact",
    loadMode: "trainable-artifact",
    kind: "trainable-kernel-model-artifact",
    stage: "url-debug-artifact",
    objectCount: 0,
    galleryPosition: [-5.55, 0, 4.52],
    accent: "#9eeaf2",
    displayScale: 1.92,
    pointSize: 0.082,
    maxDisplayPoints: 256,
    trainableArtifactPath: artifactPath,
    compression: {
      layout: "trainable-kernel-artifact-json",
      status: "url-debug-artifact",
      chunkRoot: "/models/url-trainable-artifact/objects/",
    },
  };
}

function plyUrlArtifactModel(plyPath) {
  return {
    id: "ply-url-artifact",
    name: "URL object-aware PLY",
    label: "URL PLY",
    sourcePath: plyPath,
    loadMode: "eager",
    kind: "object-aware-ply",
    stage: "url-debug-artifact",
    objectCount: 0,
    galleryPosition: [-4.25, 0, 4.24],
    accent: "#6ee7f8",
    displayScale: 2.46,
    pointSize: 0.035,
    maxDisplayPoints: 50000,
    compression: {
      layout: "url-object-aware-ply",
      status: "url-debug-artifact",
      chunkRoot: "/models/url-ply-artifact/objects/",
    },
  };
}

function modelArtifactManifestBundleModel({ manifestPath, lodLevel, chunkIds }) {
  return {
    id: "model-artifact-manifest",
    name: "Algorithm model manifest",
    label: "Model Manifest",
    loadMode: "model-artifact-manifest",
    kind: "algorithm-model-artifact-manifest",
    stage: "algorithm-handoff-artifact",
    objectCount: 0,
    galleryPosition: [-0.9, 0, 5.18],
    accent: "#d7f45a",
    displayScale: 1.62,
    pointSize: 0.068,
    maxDisplayPoints: 2400,
    manifestPath,
    ogc: {
      lodLevel,
      chunkIds,
    },
    compression: {
      layout: "model-artifact-manifest-handoff",
      status: "url-debug-artifact",
      chunkRoot: "/models/model-artifact-manifest/objects/",
    },
  };
}

function ogcManifestUrlArtifactModel({ manifestPath, lodLevel, chunkIds }) {
  return {
    id: "ogc-manifest-artifact",
    name: "URL OGC manifest",
    label: "OGC Manifest",
    loadMode: "ogc-manifest",
    kind: "compressed-chunked-ogc-manifest",
    stage: "url-browser-delivery-artifact",
    objectCount: 0,
    galleryPosition: [4.18, 0, 4.36],
    accent: "#58f2c2",
    displayScale: 1.72,
    pointSize: 0.07,
    maxDisplayPoints: 2400,
    ogc: {
      manifestPath,
      lodLevel,
      chunkIds,
    },
    compression: {
      layout: "object-aware-quantized-ogc-manifest",
      status: "url-debug-artifact",
      chunkRoot: "/models/url-ogc-manifest/objects/",
    },
  };
}

function ogcUrlArtifactModel({ indexPath, payloadPath, lodLevel, chunkIds }) {
  return {
    id: "ogc-url-artifact",
    name: "URL OGC artifact",
    label: "URL OGC",
    loadMode: "ogc-chunked",
    kind: "compressed-chunked-ogc",
    stage: "url-browser-delivery-artifact",
    objectCount: 0,
    galleryPosition: [2.9, 0, 4.48],
    accent: "#6ee7f8",
    displayScale: 1.72,
    pointSize: 0.07,
    maxDisplayPoints: 2400,
    ogc: {
      lodLevel,
      chunkIds,
      indexPath,
      payloadPath,
    },
    compression: {
      layout: "object-aware-quantized-ogc-chunks",
      status: "url-debug-artifact",
      chunkRoot: "/models/url-ogc-artifact/objects/",
    },
    modelArtifactManifest: {
      schema: MODEL_ARTIFACT_MANIFEST_SCHEMA,
      manifest_id: "ogc-url-artifact-model-artifacts",
      asset_id: "ogc-url-artifact",
      name: "URL OGC artifact",
      stage: "development",
      source: {
        type: "url_injected_compressed_chunked_artifact",
        index: indexPath,
        payload: payloadPath,
      },
      license: "local-debug-artifact",
      counts: {
        gaussians: 0,
        objects: 0,
      },
      artifacts: [
        {
          role: "compressed_chunked",
          path: payloadPath,
          format: ".ogc",
          delivery_tier: "browser_edit",
          browser_ready: true,
          chunk_index: {
            path: indexPath,
          },
          compression: {
            codec: "objgauss-ogc-prototype",
            layout: "object-aware-chunked-local-quantized",
            status: "url-debug-artifact",
          },
        },
      ],
      quality_evidence: [
        {
          kind: "url-ogc-browser-route",
          status: "requires-runtime-decode",
        },
      ],
      limitations: [
        "Runtime URL artifact for local browser delivery debugging; not a published model.",
      ],
      created_from: {
        source: "OGC-URL-ARTIFACT-001",
      },
    },
  };
}

function sameOriginPathParam(params, names, extension) {
  const rawPath = names.map((name) => params.get(name)).find(Boolean);
  if (!rawPath) return "";
  const path = rawPath.trim();
  if (!path || path.length > 240) return "";
  if (!path.startsWith("/") || path.startsWith("//")) return "";
  if (path.includes("\n") || path.includes("\r")) return "";
  if (path.includes("?") || path.includes("#")) return "";
  if (!path.endsWith(extension)) return "";
  return path;
}

function ogcLodLevelFromParams(params) {
  const rawValue = params.get("ogcLod") ?? params.get("ogc-lod");
  if (rawValue === null || rawValue === "") return 0;
  const level = Number(rawValue);
  return Number.isInteger(level) && level >= 0 && level <= 16 ? level : 0;
}

function ogcChunkIdsFromParams(params) {
  const rawValue = params.get("ogcChunks") ?? params.get("ogc-chunks");
  if (!rawValue) return undefined;
  const ids = rawValue
    .split(",")
    .map((entry) => Number(entry.trim()))
    .filter((value) => Number.isInteger(value) && value >= 0 && value <= 1_000_000);
  return ids.length ? [...new Set(ids)] : undefined;
}
