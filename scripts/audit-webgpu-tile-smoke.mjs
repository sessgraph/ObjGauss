import assert from "node:assert/strict";

import { parsePly } from "../src/ply.js";
import { createSampleScene } from "../src/sampleScene.js";
import {
  buildPackedShExtra,
  extractPackedShExtra,
  shDcRgb01,
  SPARK_PACKED_EXTRACT_ROUTE,
  SPARK_PACKED_SH_EXTRACT_ROUTE,
} from "../src/sparkPackedSh.js";
import {
  createWebGpuAccumulationMeta,
  createWebGpuComputeMeta,
  createWebGpuPixelResolveMeta,
  createWebGpuPixelResolveShader,
  webGpuAccumulationWorkgroups,
  webGpuComputeWorkgroups,
  webGpuPixelResolveWorkgroups,
  WEBGPU_PIXEL_RESOLVE_SHADER,
  WEBGPU_PIXEL_RESOLVE_SOURCE,
  WEBGPU_PIXEL_RESOLVE_WORKGROUP_SIZE,
  WEBGPU_TILE_ACCUMULATION_SHADER,
  WEBGPU_TILE_ACCUMULATION_SOURCE,
  WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE,
  WEBGPU_TILE_COMPUTE_SHADER,
  WEBGPU_TILE_COMPUTE_SOURCE,
  WEBGPU_TILE_COMPUTE_WORKGROUP_SIZE,
} from "../src/webgpuTileComputeShader.js";
import {
  buildWebGpuTileSmoke,
  WEBGPU_CAMERA_MODE_EDIT_FIXED,
  WEBGPU_CAMERA_MODE_SPARK_FRAME,
  WEBGPU_CAMERA_TUNING_MODE,
  WEBGPU_COLOR_MODE_SH_VIEW,
  WEBGPU_COLOR_MODE_SOURCE,
  WEBGPU_COLOR_TUNING_MODE,
  WEBGPU_COVERAGE_TUNING_MODE,
  WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED,
  WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K,
  WEBGPU_DEPTH_SORT_TUNING_MODE,
  WEBGPU_OBJECT_STATE_LAYOUT_VERSION,
  WEBGPU_OBJECT_STATE_STRIDE_UINT32,
  WEBGPU_PIXEL_COVERAGE_MODE,
  WEBGPU_PIXEL_DEPTH_SORT_MODE,
  WEBGPU_PIXEL_DEPTH_SORT_MODE_FRONT_TOP_K,
  WEBGPU_TILE_COLOR_FIDELITY_MODE,
  WEBGPU_TILE_DEPTH_WEIGHT_MODE,
  WEBGPU_TILE_ENTRY_LAYOUT_COMPACT,
  WEBGPU_TILE_ENTRY_LAYOUT_FIXED,
  WEBGPU_TILE_PROJECTION_MODE,
  WEBGPU_TILE_SCREEN_COVARIANCE_MODE,
  WEBGPU_TILE_SPARK_FRAME_PROJECTION_MODE,
  WEBGPU_TILE_SMOKE_LAYOUT_VERSION,
} from "../src/webgpuTileSmoke.js";
import {
  createWebGpuTileStorageBuffers,
  describeWebGpuTileStorage,
  estimateWebGpuTileRuntimeStorage,
  WEBGPU_TILE_STORAGE_LAYOUT_VERSION,
} from "../src/webgpuTileStorage.js";
import {
  createWebGpuResolveMeta,
  createWebGpuTileResolveShader,
  normalizeWebGpuAlphaPresentationTuning,
  WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR,
  WEBGPU_TILE_ALPHA_PRESENTATION_MODE,
  WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE,
  WEBGPU_TILE_RESOLVE_FILTER,
  WEBGPU_TILE_RESOLVE_SHADER,
  WEBGPU_TILE_RESOLVE_SOURCE,
} from "../src/webgpuTileResolveShader.js";
import {
  WEBGPU_FLOAT_TEXTURE_COPY_RESOLVE_SOURCE,
  WEBGPU_FLOAT_TEXTURE_LOAD_RESOLVE_SHADER,
  WEBGPU_SAMPLED_TEXTURE_RESOLVE_SHADER,
  WEBGPU_SAMPLED_TEXTURE_RESOLVE_SOURCE,
} from "../src/webgpuTextureResolveShader.js";
import {
  normalizeWebGpuRuntimeProbe,
  WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY,
  WEBGPU_RUNTIME_PROBE_CLEAR_ONLY,
  WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY,
  WEBGPU_RUNTIME_PROBE_FULL,
  WEBGPU_RUNTIME_PROBE_MODES,
  WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK,
  WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY,
  WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY,
  WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY,
  WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY,
  WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY,
  WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT,
  WEBGPU_RUNTIME_PROBE_TINY_VIEWPORT_SIZE,
} from "../src/webgpuRuntimeProbe.js";
import { editRendererContract } from "../src/webgpuCapability.js";

const scene = createSampleScene();
const allObjectIds = new Set(scene.points.map((point) => point.objectId));
const firstObjectId = Math.min(...allObjectIds);
const parsedShRestPly = parsePly(
  new TextEncoder().encode(
    [
      "ply",
      "format ascii 1.0",
      "element vertex 1",
      "property float x",
      "property float y",
      "property float z",
      "property float f_dc_0",
      "property float f_dc_1",
      "property float f_dc_2",
      ...Array.from({ length: 9 }, (_, index) => `property float f_rest_${index}`),
      "end_header",
      `0 0 0 0.1 0.2 0.3 ${Array.from({ length: 9 }, (_, index) => index * 0.01).join(" ")}`,
    ].join("\n"),
  ).buffer,
);
assert.equal(parsedShRestPly.points[0].colorSource, "sh-dc");
assert.deepEqual(parsedShRestPly.points[0].shDc, [0.1, 0.2, 0.3]);
assert.equal(parsedShRestPly.points[0].shRestCoefficientCount, 9);
assert.equal(parsedShRestPly.points[0].shDegree, 1);
assert.equal(parsedShRestPly.shRestCoefficientCount, 9);
assert.equal(parsedShRestPly.shDegree, 1);
assert.equal(parsedShRestPly.shRestCoefficients.length, 9);
assert.ok(Math.abs(parsedShRestPly.shRestCoefficients[8] - 0.08) < 0.000001);

const sparkShPoints = [
  {
    shDc: [0.1, 0.2, 0.3],
    shRestCoefficientCount: 45,
    shDegree: 3,
  },
  {
    shDc: [0.2, 0.1, 0],
    shRestCoefficientCount: 0,
    shDegree: 0,
  },
];
const sparkShRest = new Float32Array(sparkShPoints.length * 45);
for (let index = 0; index < 45; index += 1) {
  sparkShRest[index] = index % 2 === 0 ? 0.12 : -0.08;
}
const sparkSh = buildPackedShExtra({
  points: sparkShPoints,
  shRestCoefficients: sparkShRest,
  shRestCoefficientCount: 45,
});
assert.equal(sparkSh.route, SPARK_PACKED_SH_EXTRACT_ROUTE);
assert.equal(sparkSh.sourceGaussians, 1);
assert.equal(sparkSh.preservedGaussians, 1);
assert.equal(sparkSh.preserved, true);
assert.equal(sparkSh.coefficientCount, 45);
assert.equal(sparkSh.degree, 3);
assert.equal(sparkSh.extra.sh1.length, sparkShPoints.length * 2);
assert.equal(sparkSh.extra.sh2.length, sparkShPoints.length * 4);
assert.equal(sparkSh.extra.sh3.length, sparkShPoints.length * 4);
assert.ok(Array.from(sparkSh.extra.sh1).some((value) => value !== 0));
assert.ok(Array.from(sparkSh.extra.sh2).some((value) => value !== 0));
assert.ok(Array.from(sparkSh.extra.sh3).some((value) => value !== 0));
const extractedSparkSh = extractPackedShExtra(sparkSh.extra, new Uint32Array([0]));
assert.deepEqual(
  Array.from(extractedSparkSh.sh1),
  Array.from(sparkSh.extra.sh1.subarray(0, 2)),
);
assert.deepEqual(
  Array.from(extractedSparkSh.sh2),
  Array.from(sparkSh.extra.sh2.subarray(0, 4)),
);
assert.deepEqual(
  Array.from(extractedSparkSh.sh3),
  Array.from(sparkSh.extra.sh3.subarray(0, 4)),
);
const sparkNoSh = buildPackedShExtra({
  points: sparkShPoints,
  shRestCoefficients: null,
  shRestCoefficientCount: 0,
});
assert.equal(sparkNoSh.route, SPARK_PACKED_EXTRACT_ROUTE);
assert.equal(sparkNoSh.preserved, false);
assert.deepEqual(shDcRgb01(sparkShPoints[0]).map((value) => Number(value.toFixed(6))), [
  0.528209,
  0.556419,
  0.584628,
]);

assert.deepEqual(WEBGPU_RUNTIME_PROBE_MODES, [
  WEBGPU_RUNTIME_PROBE_FULL,
  WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY,
  WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY,
  WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY,
  WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY,
  WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK,
  WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY,
  WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT,
  WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY,
  WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY,
  WEBGPU_RUNTIME_PROBE_CLEAR_ONLY,
]);
assert.equal(normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_FULL), WEBGPU_RUNTIME_PROBE_FULL);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY),
  WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY),
  WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY),
  WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY),
  WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK),
  WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY),
  WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT),
  WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY),
  WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY),
  WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY,
);
assert.equal(
  normalizeWebGpuRuntimeProbe(WEBGPU_RUNTIME_PROBE_CLEAR_ONLY),
  WEBGPU_RUNTIME_PROBE_CLEAR_ONLY,
);
assert.equal(WEBGPU_RUNTIME_PROBE_TINY_VIEWPORT_SIZE, 32);
assert.equal(normalizeWebGpuRuntimeProbe("invalid"), WEBGPU_RUNTIME_PROBE_FULL);

const base = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  includeTileEntries: true,
  includePixelOutput: true,
  maxEntriesPerTile: 64,
});

assert.equal(base.layoutVersion, WEBGPU_TILE_SMOKE_LAYOUT_VERSION);
assert.equal(base.packedGaussians, scene.points.length);
assert.ok(Number.isFinite(base.boundsMinX));
assert.ok(Number.isFinite(base.boundsMinZ));
assert.ok(base.boundsSpanX > 0);
assert.ok(base.boundsSpanZ > 0);
assert.equal(base.buffers.positionRadius.length, scene.points.length * 4);
assert.equal(base.buffers.colorOpacity.length, scene.points.length * 4);
assert.equal(base.buffers.scaleRotation.length, scene.points.length * 4);
assert.equal(base.buffers.objectIndices.length, scene.points.length);
assert.equal(base.buffers.objectState.length, allObjectIds.size * WEBGPU_OBJECT_STATE_STRIDE_UINT32);
assert.equal(base.buffers.objectIds.length, allObjectIds.size);
assert.equal(base.buffers.tileCounts.length, base.tileCount);
assert.equal(base.buffers.tileOffsets.length, base.tileCount);
assert.equal(base.buffers.tileAccumulation.length, base.tileCount * 4);
assert.equal(base.buffers.tileResolvedRgba.length, base.tileCount * 4);
assert.equal(base.buffers.pixelResolvedRgba.length, base.pixelCount * 4);
assert.equal(base.buffers.tileEntries.length, base.tileEntryCapacity);
assert.equal(base.buffers.tileEntries.length, base.tileReferenceCount);
assert.ok(base.visibleGaussians > 0);
assert.ok(base.binnedGaussians > 0);
assert.ok(base.activeTileCount > 0);
assert.ok(base.tileReferenceCount >= base.binnedGaussians);
assert.equal(base.tileEntryLayout, WEBGPU_TILE_ENTRY_LAYOUT_COMPACT);
assert.equal(base.tileEntryOffsetCount, base.tileCount);
assert.equal(base.tileCapacityMode, WEBGPU_TILE_ENTRY_LAYOUT_COMPACT);
assert.equal(base.tileCapacityStatus, "ok");
assert.equal(base.tileCapacityGate, "pass");
assert.equal(base.tileOverflowCount, 0);
assert.equal(base.tileOverflowTileCount, 0);
assert.equal(base.tileOverflowRatio, 0);
assert.equal(base.tileOverflowMaxExcess, 0);
assert.equal(base.tileEntryStoredCount, base.tileReferenceCount);
assert.equal(base.tileEntryCapacity, base.tileReferenceCount);
assert.equal(base.tileEntryUtilization, 1);
assert.equal(base.resolveVersion, "webgpu-tile-resolve-v1");
assert.equal(base.resolveMode, "tile-2x2-covariance-weighted-oit");
assert.equal(base.pixelOutputMode, "viewport-storage-rgba-direct-gaussian");
assert.equal(base.pixelOutputIncluded, true);
assert.equal(base.pixelReferenceIncluded, true);
assert.equal(base.pixelCount, base.viewportWidth * base.viewportHeight);
assert.equal(base.boundsFitMode, "aspect-fit-padding");
assert.equal(base.boundsPaddingRatio, 0.08);
assert.ok(Math.abs(base.boundsWorldAspect - base.boundsViewportAspect) < 0.02);
assert.equal(base.projectionMode, WEBGPU_TILE_PROJECTION_MODE);
assert.equal(base.projectionCameraTuningMode, WEBGPU_CAMERA_TUNING_MODE);
assert.equal(base.projectionCameraMode, WEBGPU_CAMERA_MODE_EDIT_FIXED);
assert.equal(base.projectionCameraFovDegrees, 52);
assert.deepEqual(base.projectionCameraPosition, [3.6, 2.8, 3.4]);
assert.deepEqual(base.projectionCameraTarget, [0, 0, 0.25]);
assert.ok(base.projectionCameraDistance > 5);
assert.equal(base.projectionCameraFrameMaxDim, 0);
assert.equal(base.depthWeightMode, WEBGPU_TILE_DEPTH_WEIGHT_MODE);
assert.equal(base.pixelDepthSortMode, WEBGPU_PIXEL_DEPTH_SORT_MODE);
assert.equal(base.pixelDepthTuningMode, WEBGPU_DEPTH_SORT_TUNING_MODE);
assert.equal(base.pixelDepthAlphaMode, WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED);
assert.equal(base.pixelDepthGateStrength, 12);
assert.equal(base.pixelDepthGateFloor, 0.06);
assert.equal(base.pixelDepthBinCount, 8);
assert.equal(base.pixelCoverageMode, WEBGPU_PIXEL_COVERAGE_MODE);
assert.equal(base.pixelCoverageTuningMode, WEBGPU_COVERAGE_TUNING_MODE);
assert.equal(base.pixelCoverageWeightFloor, 0.004);
assert.equal(base.pixelCoverageFootprintScale, 2.2);
assert.ok(Number.isFinite(base.projectionDepthMin));
assert.ok(Number.isFinite(base.projectionDepthMax));
assert.ok(base.projectionDepthMax > base.projectionDepthMin);
assert.ok(base.projectionDepthSpan > 0);
assert.equal(base.colorFidelityMode, WEBGPU_TILE_COLOR_FIDELITY_MODE);
assert.equal(base.colorTuningMode, WEBGPU_COLOR_TUNING_MODE);
assert.equal(base.colorMode, WEBGPU_COLOR_MODE_SOURCE);
assert.equal(base.colorSourceRgbGaussians, base.packedGaussians);
assert.equal(base.colorSourceShDcGaussians, 0);
assert.equal(base.colorSourceFallbackGaussians, 0);
assert.equal(base.colorSourceObjectGaussians, 0);
assert.equal(base.colorShRestGaussians, 0);
assert.equal(base.colorShRestCoefficientMax, 0);
assert.equal(base.colorShDegreeMax, 0);
assert.equal(base.colorShViewGaussians, 0);
assert.ok(base.colorOpacityMean > 0);
assert.ok(base.colorOpacityMean <= 1);
assert.equal(base.screenCovarianceMode, WEBGPU_TILE_SCREEN_COVARIANCE_MODE);
assert.equal(base.screenCovarianceGaussians, base.packedGaussians);
assert.equal(base.screenCovarianceFallbackGaussians, 0);
assert.ok(base.screenCovarianceClampedGaussians >= 0);
assert.equal(base.screenCovarianceMaxAnisotropy, 4);
assert.ok(base.screenCovarianceSigmaMean > 0);

const shRestScenePoints = scene.points.map((point, index) =>
  index === 0
    ? {
        ...point,
        colorSource: "sh-dc",
        shDc: [0.1, 0.2, 0.3],
        color: [135, 142, 149],
        shRestCoefficientCount: 45,
        shDegree: 3,
      }
    : point,
);
const shRestCoefficients = new Float32Array(scene.points.length * 45);
for (let index = 0; index < 45; index += 1) {
  shRestCoefficients[index] = index % 3 === 0 ? 0.08 : index % 3 === 1 ? -0.04 : 0.02;
}
const shRestMetadata = buildWebGpuTileSmoke({
  points: shRestScenePoints,
  shRestCoefficients,
  shRestCoefficientCount: 45,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
});
assert.equal(shRestMetadata.colorMode, WEBGPU_COLOR_MODE_SOURCE);
assert.equal(shRestMetadata.colorSourceRgbGaussians, base.packedGaussians - 1);
assert.equal(shRestMetadata.colorSourceShDcGaussians, 1);
assert.equal(shRestMetadata.colorShRestGaussians, 1);
assert.equal(shRestMetadata.colorShRestCoefficientMax, 45);
assert.equal(shRestMetadata.colorShDegreeMax, 3);
assert.equal(shRestMetadata.colorShViewGaussians, 0);

const shViewMetadata = buildWebGpuTileSmoke({
  points: shRestScenePoints,
  shRestCoefficients,
  shRestCoefficientCount: 45,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  colorTuning: { colorMode: WEBGPU_COLOR_MODE_SH_VIEW },
});
assert.equal(shViewMetadata.colorMode, WEBGPU_COLOR_MODE_SH_VIEW);
assert.equal(shViewMetadata.colorShViewGaussians, 1);
assert.notDeepEqual(
  Array.from(shViewMetadata.buffers.colorOpacity.slice(0, 3)),
  Array.from(shRestMetadata.buffers.colorOpacity.slice(0, 3)),
);

const sparkFrameCamera = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  includeTileEntries: true,
  includePixelOutput: true,
  maxEntriesPerTile: 64,
  cameraTuning: { cameraMode: WEBGPU_CAMERA_MODE_SPARK_FRAME },
});
assert.equal(sparkFrameCamera.projectionMode, WEBGPU_TILE_SPARK_FRAME_PROJECTION_MODE);
assert.equal(sparkFrameCamera.projectionCameraTuningMode, WEBGPU_CAMERA_TUNING_MODE);
assert.equal(sparkFrameCamera.projectionCameraMode, WEBGPU_CAMERA_MODE_SPARK_FRAME);
assert.equal(sparkFrameCamera.projectionCameraFovDegrees, 58);
assert.ok(Array.isArray(sparkFrameCamera.projectionCameraPosition));
assert.ok(Array.isArray(sparkFrameCamera.projectionCameraTarget));
assert.equal(sparkFrameCamera.projectionCameraPosition.length, 3);
assert.equal(sparkFrameCamera.projectionCameraTarget.length, 3);
assert.ok(sparkFrameCamera.projectionCameraDistance > 0);
assert.ok(sparkFrameCamera.projectionCameraFrameMaxDim > 0);
assert.notDeepEqual(sparkFrameCamera.projectionCameraPosition, base.projectionCameraPosition);

assert.ok(base.resolvedTileCount > 0);
assert.ok(base.pixelResolvedCount > 0);
assert.ok(base.pixelResolvedCount > base.resolvedTileCount);
assert.ok(base.resolveWeightSum > 0);
assert.ok(base.resolveAlphaMean > 0);
assert.ok(base.resolveLumaMean > 0);
assert.match(base.resolveChecksum, /^[0-9a-f]{8}$/);
assert.match(base.pixelResolveChecksum, /^[0-9a-f]{8}$/);

const tunedCoverage = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  includeTileEntries: true,
  includePixelOutput: true,
  maxEntriesPerTile: 64,
  coverageTuning: {
    footprintScale: 1.7,
    maxAnisotropy: 2.5,
  },
});
assert.equal(tunedCoverage.pixelCoverageTuningMode, WEBGPU_COVERAGE_TUNING_MODE);
assert.equal(tunedCoverage.pixelCoverageFootprintScale, 1.7);
assert.equal(tunedCoverage.screenCovarianceMaxAnisotropy, 2.5);
assert.notEqual(tunedCoverage.tileReferenceCount, base.tileReferenceCount);
assert.notEqual(tunedCoverage.screenCovarianceSigmaMean, base.screenCovarianceSigmaMean);

const tunedDepthSort = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  includeTileEntries: true,
  includePixelOutput: true,
  maxEntriesPerTile: 64,
  depthSortTuning: {
    depthBins: 12,
  },
});
assert.equal(tunedDepthSort.pixelDepthTuningMode, WEBGPU_DEPTH_SORT_TUNING_MODE);
assert.equal(tunedDepthSort.pixelDepthBinCount, 12);
assert.notEqual(tunedDepthSort.pixelResolveChecksum, base.pixelResolveChecksum);

const frontTopKDepthSort = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  includeTileEntries: true,
  includePixelOutput: true,
  maxEntriesPerTile: 64,
  depthSortTuning: {
    depthAlphaMode: WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K,
    depthBins: 12,
  },
});
assert.equal(frontTopKDepthSort.pixelDepthSortMode, WEBGPU_PIXEL_DEPTH_SORT_MODE_FRONT_TOP_K);
assert.equal(frontTopKDepthSort.pixelDepthAlphaMode, WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K);
assert.equal(frontTopKDepthSort.pixelDepthBinCount, 12);
assert.ok(frontTopKDepthSort.pixelResolvedCount > 0);
assert.notEqual(frontTopKDepthSort.pixelResolveChecksum, base.pixelResolveChecksum);

const wideViewport = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  selectedId: null,
  renderMode: "original",
  pointSize: 0.018,
  viewportWidth: 384,
  viewportHeight: 192,
  includeTileEntries: true,
  includePixelOutput: false,
});
assert.equal(wideViewport.boundsFitMode, "aspect-fit-padding");
assert.ok(Math.abs(wideViewport.boundsViewportAspect - 2) < 0.001);
assert.ok(Math.abs(wideViewport.boundsWorldAspect - wideViewport.boundsViewportAspect) < 0.02);
assert.equal(base.objectStateLayoutVersion, WEBGPU_OBJECT_STATE_LAYOUT_VERSION);
assert.equal(base.objectStateStrideUint32, WEBGPU_OBJECT_STATE_STRIDE_UINT32);
assert.equal(base.objectStateVisibleObjects, allObjectIds.size);
assert.equal(base.objectStateHiddenObjects, 0);
assert.equal(base.objectStateRemovedObjects, 0);
assert.equal(base.objectStateSelectedObjects, 0);
assert.equal(base.objectStateIsolatedObjects, 0);
assert.match(base.objectStateChecksum, /^[0-9a-f]{8}$/);

const storage = describeWebGpuTileStorage(base);
const storageEstimate = estimateWebGpuTileRuntimeStorage(base);
assert.equal(storage.layoutVersion, WEBGPU_TILE_STORAGE_LAYOUT_VERSION);
assert.equal(storage.bufferCount, 11);
assert.ok(storage.totalByteLength > 0);
assert.match(storage.checksum, /^[0-9a-f]{8}$/);
assert.equal(storage.tileEntriesIncluded, true);
assert.equal(storage.tileOffsetsIncluded, true);
assert.equal(storage.pixelOutputIncluded, true);
assert.deepEqual(
  storage.descriptors.map((descriptor) => descriptor.key),
  [
    "positionRadius",
    "colorOpacity",
    "scaleRotation",
    "objectIndices",
    "objectState",
    "tileCounts",
    "tileOffsets",
    "tileAccumulation",
    "tileResolvedRgba",
    "pixelResolvedRgba",
    "tileEntries",
  ],
);
assert.equal(
  storage.totalByteLength,
  storage.descriptors.reduce((total, descriptor) => total + descriptor.allocatedByteLength, 0),
);
assert.equal(storageEstimate.layoutVersion, WEBGPU_TILE_STORAGE_LAYOUT_VERSION);
assert.equal(storageEstimate.bufferCount, 11);
assert.equal(storageEstimate.totalByteLength, storage.totalByteLength);
assert.equal(storageEstimate.maxBufferKey, "pixelResolvedRgba");
assert.equal(
  storageEstimate.maxBufferByteLength,
  storage.descriptors.find((descriptor) => descriptor.key === "pixelResolvedRgba").allocatedByteLength,
);
assert.equal(storageEstimate.tileEntriesIncluded, true);
assert.equal(storageEstimate.tileOffsetsIncluded, true);
assert.equal(storageEstimate.pixelOutputIncluded, true);
assert.deepEqual(
  storageEstimate.descriptors.map((descriptor) => descriptor.key),
  storage.descriptors.map((descriptor) => descriptor.key),
);

const fakeDevice = createFakeDevice();
const storageBundle = createWebGpuTileStorageBuffers(fakeDevice, base);
assert.equal(storageBundle.layoutVersion, WEBGPU_TILE_STORAGE_LAYOUT_VERSION);
assert.equal(storageBundle.bufferCount, storage.bufferCount);
assert.equal(storageBundle.totalByteLength, storage.totalByteLength);
assert.equal(storageBundle.checksum, storage.checksum);
assert.equal(storageBundle.buffers.length, storage.bufferCount);
assert.equal(
  storageBundle.getBuffer("tileResolvedRgba").byteLength,
  base.buffers.tileResolvedRgba.byteLength,
);
assert.equal(
  storageBundle.getBuffer("tileOffsets").byteLength,
  base.buffers.tileOffsets.byteLength,
);
assert.equal(
  storageBundle.getBuffer("pixelResolvedRgba").byteLength,
  base.buffers.pixelResolvedRgba.byteLength,
);
assert.equal(storageBundle.getBuffer("objectState").byteLength, base.buffers.objectState.byteLength);
assert.equal(fakeDevice.created.length, storage.bufferCount);
assert.equal(fakeDevice.writes.length, storage.bufferCount);
assert.ok(fakeDevice.created.every((buffer) => buffer.descriptor.usage > 0));
assert.ok(
  fakeDevice.writes.every((write, index) => write.byteLength === storage.descriptors[index].byteLength),
);
storageBundle.destroy();
assert.ok(fakeDevice.created.every((buffer) => buffer.destroyed));

const resolveMeta = createWebGpuResolveMeta(base);
assert.deepEqual([...resolveMeta], [base.viewportWidth, base.viewportHeight, 0, 0]);
assert.equal(resolveMeta.byteLength, 16);
assert.equal(WEBGPU_TILE_RESOLVE_SOURCE, "webgpu-pixel-storage-resolve-v1");
assert.equal(WEBGPU_TILE_RESOLVE_FILTER, "bilinear-storage");
assert.equal(WEBGPU_TILE_ALPHA_PRESENTATION_MODE, "alpha-edge-gated-presentation-v1");
assert.equal(WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE, "runtime-alpha-presentation-tuning-v1");
assert.equal(WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR, 0.035);
assert.deepEqual(normalizeWebGpuAlphaPresentationTuning(), {
  mode: WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE,
  alphaPresentationFloor: WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR,
});
assert.equal(
  normalizeWebGpuAlphaPresentationTuning({ alphaPresentationFloor: 0.075 }).alphaPresentationFloor,
  0.075,
);
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /var<storage,\s*read>\s+pixelResolvedRgba/);
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /var<uniform>\s+resolveMeta/);
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /fn\s+samplePixel/);
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /fract\(pixelPosition\)/);
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /mix\(/);
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /ALPHA_PRESENTATION_FLOOR/);
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /alpha\s*<\s*ALPHA_PRESENTATION_FLOOR/);
assert.match(
  createWebGpuTileResolveShader({ alphaPresentationFloor: 0.075 }),
  /ALPHA_PRESENTATION_FLOOR = 0\.075000/,
);
assert.ok(!WEBGPU_TILE_RESOLVE_SHADER.includes("textureSample"));
assert.equal(WEBGPU_SAMPLED_TEXTURE_RESOLVE_SOURCE, "webgpu-sampled-texture-resolve-v1");
assert.match(WEBGPU_SAMPLED_TEXTURE_RESOLVE_SHADER, /texture_2d<f32>/);
assert.match(WEBGPU_SAMPLED_TEXTURE_RESOLVE_SHADER, /var\s+sourceSampler:\s*sampler/);
assert.match(WEBGPU_SAMPLED_TEXTURE_RESOLVE_SHADER, /textureSampleLevel/);
assert.ok(!WEBGPU_SAMPLED_TEXTURE_RESOLVE_SHADER.includes("pixelResolvedRgba"));
assert.equal(WEBGPU_FLOAT_TEXTURE_COPY_RESOLVE_SOURCE, "webgpu-buffer-copy-texture-resolve-v1");
assert.match(WEBGPU_FLOAT_TEXTURE_LOAD_RESOLVE_SHADER, /texture_2d<f32>/);
assert.match(WEBGPU_FLOAT_TEXTURE_LOAD_RESOLVE_SHADER, /var<uniform>\s+resolveMeta/);
assert.match(WEBGPU_FLOAT_TEXTURE_LOAD_RESOLVE_SHADER, /textureLoad/);
assert.ok(!WEBGPU_FLOAT_TEXTURE_LOAD_RESOLVE_SHADER.includes("pixelResolvedRgba"));

const computeMeta = createWebGpuComputeMeta(base);
assert.deepEqual([...computeMeta], [base.tileCount, 0, 0, 0]);
assert.equal(computeMeta.byteLength, 16);
assert.equal(WEBGPU_TILE_COMPUTE_SOURCE, "webgpu-compute-resolve-v1");
assert.equal(WEBGPU_TILE_COMPUTE_WORKGROUP_SIZE, 64);
assert.equal(webGpuComputeWorkgroups(base), Math.ceil(base.tileCount / WEBGPU_TILE_COMPUTE_WORKGROUP_SIZE));
assert.match(WEBGPU_TILE_COMPUTE_SHADER, /@compute\s+@workgroup_size\(64\)/);
assert.match(WEBGPU_TILE_COMPUTE_SHADER, /var<storage,\s*read>\s+tileAccumulation/);
assert.match(WEBGPU_TILE_COMPUTE_SHADER, /var<storage,\s*read_write>\s+tileResolvedRgba/);
assert.match(WEBGPU_TILE_COMPUTE_SHADER, /tileResolvedRgba\[tileIndex\]/);

const pixelMeta = createWebGpuPixelResolveMeta(base);
assert.deepEqual(
  [...pixelMeta.slice(0, 6)],
  [
    base.pixelCount,
    base.viewportWidth,
    base.viewportHeight,
    base.tileSize,
    base.tileColumns,
    base.maxEntriesPerTile,
  ],
);
assert.equal(pixelMeta[6], Math.fround(base.boundsMinX));
assert.equal(pixelMeta[7], Math.fround(base.boundsMinZ));
assert.equal(pixelMeta[8], Math.fround(base.boundsSpanX));
assert.equal(pixelMeta[9], Math.fround(base.boundsSpanZ));
assert.equal(pixelMeta[10], Math.fround(base.projectionDepthMin));
assert.equal(pixelMeta[11], Math.fround(base.projectionDepthSpan));
assert.equal(pixelMeta.byteLength, 48);
assert.equal(WEBGPU_PIXEL_RESOLVE_SOURCE, "webgpu-compute-depth-binned-alpha-composite-v1");
assert.equal(WEBGPU_PIXEL_RESOLVE_WORKGROUP_SIZE, 64);
assert.equal(
  webGpuPixelResolveWorkgroups(base),
  Math.ceil(base.pixelCount / WEBGPU_PIXEL_RESOLVE_WORKGROUP_SIZE),
);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /@compute\s+@workgroup_size\(64\)/);
assert.ok(!WEBGPU_PIXEL_RESOLVE_SHADER.includes("tileResolvedRgba"));
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var<storage,\s*read>\s+positionRadius/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var<storage,\s*read>\s+colorOpacity/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var<storage,\s*read>\s+objectState/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var<storage,\s*read>\s+tileEntries/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var<storage,\s*read>\s+scaleRotation/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var<storage,\s*read>\s+tileOffsets/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var<storage,\s*read_write>\s+pixelResolvedRgba/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /pixelResolvedRgba\[pixelIndex\]/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /colorOpacity\[gaussianIndex\]/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /let\s+screen\s*=\s*centerRadius\.xy/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /PIXEL_DEPTH_BIN_COUNT/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /const PIXEL_DEPTH_BIN_COUNT = 8;/);
assert.match(createWebGpuPixelResolveShader({ pixelDepthBinCount: 12 }), /const PIXEL_DEPTH_BIN_COUNT = 12;/);
assert.match(
  createWebGpuPixelResolveShader({ pixelDepthAlphaMode: WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K }),
  /const USE_FRONT_TOP_K_ALPHA = true;/,
);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var\s+binAccumulation/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var\s+topDepth/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /insertDepth\s*<\s*topDepth/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /normalizedDepth\s*\*\s*f32\(PIXEL_DEPTH_BIN_COUNT\)/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /candidateCount\s*==\s*0u/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /var\s+outputRgbPremultiplied/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /let\s+visibility\s*=\s*1\.0\s*-\s*outputAlpha/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /PIXEL_COVERAGE_WEIGHT_FLOOR/);
assert.match(WEBGPU_PIXEL_RESOLVE_SHADER, /weight\s*<=\s*PIXEL_COVERAGE_WEIGHT_FLOOR/);

const accumulationMeta = createWebGpuAccumulationMeta(base);
assert.equal(accumulationMeta.byteLength, 48);
assert.deepEqual(
  [...accumulationMeta.slice(0, 6)],
  [
    base.tileCount,
    base.maxEntriesPerTile,
    base.tileColumns,
    base.tileSize,
    base.viewportWidth,
    base.viewportHeight,
  ],
);
assert.equal(accumulationMeta[10], Math.fround(base.projectionDepthMin));
assert.equal(accumulationMeta[11], Math.fround(base.projectionDepthSpan));
assert.equal(WEBGPU_TILE_ACCUMULATION_SOURCE, "webgpu-compute-covariance-accumulation-v1");
assert.equal(WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE, 64);
assert.equal(
  webGpuAccumulationWorkgroups(base),
  Math.ceil(base.tileCount / WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE),
);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /@compute\s+@workgroup_size\(64\)/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /var<storage,\s*read>\s+positionRadius/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /var<storage,\s*read>\s+colorOpacity/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /var<storage,\s*read>\s+scaleRotation/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /var<storage,\s*read>\s+objectState/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /var<storage,\s*read>\s+tileEntries/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /var<storage,\s*read>\s+tileOffsets/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /var<storage,\s*read_write>\s+tileAccumulation/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /sampleIndex\s*<\s*4u/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /gaussianScale\.xy/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /let\s+screen\s*=\s*centerRadius\.xy/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /let\s+frontWeight\s*=\s*clamp/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /RESOLVE_ALPHA_GAIN\s*\*\s*frontWeight\s*\*\s*SAMPLE_WEIGHT/);
assert.match(WEBGPU_TILE_ACCUMULATION_SHADER, /tileAccumulation\[tileIndex\]\s*=\s*accumulation/);

const roomy = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  tileEntryLayout: WEBGPU_TILE_ENTRY_LAYOUT_FIXED,
  maxEntriesPerTile: 100000,
});

assert.equal(roomy.tileCapacityMode, WEBGPU_TILE_ENTRY_LAYOUT_FIXED);
assert.equal(roomy.tileCapacityStatus, "ok");
assert.equal(roomy.tileCapacityGate, "pass");
assert.equal(roomy.tileOverflowCount, 0);
assert.equal(roomy.tileOverflowTileCount, 0);
assert.equal(roomy.tileOverflowRatio, 0);
assert.equal(roomy.tileOverflowMaxExcess, 0);
assert.equal(roomy.tileEntryStoredCount, roomy.tileReferenceCount);
const readyCapability = {
  status: "available",
  reason: "webgpu-device-ready",
  label: "可用",
  maxBufferSize: 4 * 1024 * 1024 * 1024,
  maxStorageBufferBindingSize: 4 * 1024 * 1024 * 1024,
};
const roomyContract = editRendererContract(readyCapability, roomy);
assert.equal(roomyContract.rendererId, "webgpu-tile");
assert.equal(roomyContract.objectFilter, "gpu-object-state-buffer");
assert.equal(roomyContract.targetGate, "pass");
assert.equal(roomyContract.targetGateReason, "webgpu-tile-first-frame-ready");
assert.equal(roomyContract.fallbackReason, "");

const fixedOverflow = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  tileEntryLayout: WEBGPU_TILE_ENTRY_LAYOUT_FIXED,
  includeTileEntries: true,
  maxEntriesPerTile: 64,
});
assert.equal(fixedOverflow.tileCapacityMode, WEBGPU_TILE_ENTRY_LAYOUT_FIXED);
assert.equal(fixedOverflow.tileCapacityStatus, "overflow");
assert.equal(fixedOverflow.tileCapacityGate, "blocked");
assert.ok(fixedOverflow.tileOverflowCount > 0);
assert.ok(fixedOverflow.tileOverflowTileCount > 0);
assert.ok(fixedOverflow.tileOverflowRatio > 0);
assert.ok(fixedOverflow.tileOverflowMaxExcess > 0);
assert.equal(fixedOverflow.tileEntryStoredCount, fixedOverflow.tileReferenceCount - fixedOverflow.tileOverflowCount);
assert.equal(fixedOverflow.tileEntryCapacity, fixedOverflow.tileCount * fixedOverflow.maxEntriesPerTile);

const compactContract = editRendererContract(readyCapability, base);
assert.equal(compactContract.rendererId, "webgpu-tile");
assert.equal(compactContract.targetGate, "pass");
assert.equal(compactContract.storageLimitGate, "pass");
assert.equal(compactContract.storageLimitReason, "webgpu-storage-buffer-limits-pass");
assert.equal(compactContract.storageLimitBlocker, "");
assert.equal(compactContract.storageEstimatedBufferCount, storageEstimate.bufferCount);
assert.equal(compactContract.storageEstimatedByteSize, storageEstimate.totalByteLength);
assert.equal(compactContract.storageEstimatedMaxBufferByteSize, storageEstimate.maxBufferByteLength);
assert.equal(compactContract.storageEstimatedMaxBufferKey, storageEstimate.maxBufferKey);
assert.equal(compactContract.fallbackReason, "");

const limitedCapability = {
  status: "available",
  reason: "webgpu-device-ready",
  label: "可用",
  maxBufferSize: storageEstimate.maxBufferByteLength - 4,
  maxStorageBufferBindingSize: storageEstimate.maxBufferByteLength - 4,
};
const limitedContract = editRendererContract(limitedCapability, base);
assert.equal(limitedContract.rendererId, "gaussian-oit");
assert.equal(limitedContract.targetGate, "blocked");
assert.equal(limitedContract.targetGateReason, "webgpu-storage-buffer-too-large");
assert.equal(limitedContract.targetGateBlocker, "webgpu-buffer-limit");
assert.equal(limitedContract.storageLimitGate, "blocked");
assert.equal(limitedContract.storageLimitReason, "webgpu-storage-buffer-too-large");
assert.equal(limitedContract.storageLimitBlocker, "webgpu-buffer-limit");
assert.equal(limitedContract.fallbackReason, "webgpu-buffer-limit");

const overflowContract = editRendererContract(readyCapability, fixedOverflow);
assert.equal(overflowContract.rendererId, "gaussian-oit");
assert.equal(overflowContract.targetGate, "blocked");
assert.equal(overflowContract.targetGateBlocker, "tile-overflow");
assert.equal(overflowContract.fallbackReason, "webgpu-tile-overflow");

const isolated = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: firstObjectId,
  selectedId: firstObjectId,
  renderMode: "clustered",
  pointSize: 0.018,
});

assert.ok(isolated.visibleGaussians < base.visibleGaussians);
assert.ok(isolated.binnedGaussians < base.binnedGaussians);
assert.ok(isolated.tileReferenceCount <= base.tileReferenceCount);
assert.ok(isolated.resolvedTileCount <= base.resolvedTileCount);
assert.notEqual(isolated.resolveChecksum, base.resolveChecksum);
assert.equal(isolated.objectStateVisibleObjects, 1);
assert.equal(isolated.objectStateHiddenObjects, allObjectIds.size - 1);
assert.equal(isolated.objectStateSelectedObjects, 1);
assert.equal(isolated.objectStateIsolatedObjects, 1);
assert.notEqual(isolated.objectStateChecksum, base.objectStateChecksum);
const isolatedStorage = describeWebGpuTileStorage(isolated);
assert.notEqual(isolatedStorage.checksum, storage.checksum);

const removed = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set([firstObjectId]),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
});

assert.ok(removed.visibleGaussians < base.visibleGaussians);
assert.equal(removed.packedGaussians, base.packedGaussians);
assert.notEqual(removed.resolveChecksum, base.resolveChecksum);
assert.equal(removed.objectStateVisibleObjects, allObjectIds.size - 1);
assert.equal(removed.objectStateRemovedObjects, 1);
assert.notEqual(removed.objectStateChecksum, base.objectStateChecksum);
const removedStorage = describeWebGpuTileStorage(removed);
assert.notEqual(removedStorage.checksum, storage.checksum);

console.log(
  `webgpu_tile_smoke=passed packed=${base.packedGaussians} ` +
    `objects=${base.objectCount} tiles=${base.activeTileCount}/${base.tileCount} ` +
    `refs=${base.tileReferenceCount} resolved=${base.resolvedTileCount} ` +
    `checksum=${base.resolveChecksum} objectState=${base.objectStateChecksum} ` +
    `boundsFit=${base.boundsFitMode}:${base.boundsWorldAspect.toFixed(3)}/${base.boundsViewportAspect.toFixed(3)} ` +
    `projection=${base.projectionMode}:${base.projectionCameraFovDegrees} ` +
    `depthWeight=${base.depthWeightMode}:${base.projectionDepthSpan.toFixed(3)} ` +
    `pixelDepthSort=${base.pixelDepthSortMode}:${base.pixelDepthTuningMode}:${base.pixelDepthAlphaMode}:${base.pixelDepthGateStrength}/${base.pixelDepthGateFloor}:${base.pixelDepthBinCount} ` +
    `pixelCoverage=${base.pixelCoverageMode}:${base.pixelCoverageTuningMode}:${base.pixelCoverageWeightFloor}:${base.pixelCoverageFootprintScale} ` +
    `colorTuning=${base.colorTuningMode}:${base.colorMode}:${base.colorShViewGaussians} ` +
    `colorFidelity=${base.colorFidelityMode}:${base.colorSourceRgbGaussians}/${base.colorSourceShDcGaussians}/${base.colorSourceFallbackGaussians}/${base.colorSourceObjectGaussians}:${base.colorOpacityMean.toFixed(3)} ` +
    `shRest=${base.colorShRestGaussians}/${base.colorShRestCoefficientMax}/${base.colorShDegreeMax} ` +
    `screenCovariance=${base.screenCovarianceMode}:${base.screenCovarianceGaussians}/${base.screenCovarianceFallbackGaussians}/${base.screenCovarianceClampedGaussians}:${base.screenCovarianceMaxAnisotropy}:${base.screenCovarianceSigmaMean.toFixed(3)} ` +
    `overflow=${base.tileOverflowCount} overflowTiles=${base.tileOverflowTileCount} ` +
    `capacity=${base.tileCapacityGate} storage=${storage.checksum}:${storage.bufferCount} ` +
    `accumulation=${WEBGPU_TILE_ACCUMULATION_SOURCE}:${webGpuAccumulationWorkgroups(base)} ` +
    `compute=${WEBGPU_TILE_COMPUTE_SOURCE}:${webGpuComputeWorkgroups(base)} ` +
    `pixel=${WEBGPU_PIXEL_RESOLVE_SOURCE}:${webGpuPixelResolveWorkgroups(base)} ` +
    `resolveSource=${WEBGPU_TILE_RESOLVE_SOURCE}:${WEBGPU_TILE_RESOLVE_FILTER}:${WEBGPU_TILE_ALPHA_PRESENTATION_MODE}:${WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR}`,
);

function createFakeDevice() {
  const created = [];
  const writes = [];
  return {
    created,
    writes,
    createBuffer(descriptor) {
      const buffer = {
        descriptor,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      created.push(buffer);
      return buffer;
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        writes.push({
          label: buffer.descriptor.label,
          offset,
          byteLength: data.byteLength,
        });
      },
    },
  };
}
