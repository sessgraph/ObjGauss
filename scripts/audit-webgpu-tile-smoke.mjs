import assert from "node:assert/strict";

import { createSampleScene } from "../src/sampleScene.js";
import {
  createWebGpuAccumulationMeta,
  createWebGpuComputeMeta,
  createWebGpuPixelResolveMeta,
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
  WEBGPU_OBJECT_STATE_LAYOUT_VERSION,
  WEBGPU_OBJECT_STATE_STRIDE_UINT32,
  WEBGPU_TILE_ENTRY_LAYOUT_COMPACT,
  WEBGPU_TILE_ENTRY_LAYOUT_FIXED,
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
  WEBGPU_TILE_RESOLVE_SHADER,
  WEBGPU_TILE_RESOLVE_SOURCE,
} from "../src/webgpuTileResolveShader.js";
import {
  normalizeWebGpuRuntimeProbe,
  WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY,
  WEBGPU_RUNTIME_PROBE_FULL,
  WEBGPU_RUNTIME_PROBE_MODES,
  WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY,
  WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY,
} from "../src/webgpuRuntimeProbe.js";
import { editRendererContract } from "../src/webgpuCapability.js";

const scene = createSampleScene();
const allObjectIds = new Set(scene.points.map((point) => point.objectId));
const firstObjectId = Math.min(...allObjectIds);

assert.deepEqual(WEBGPU_RUNTIME_PROBE_MODES, [
  WEBGPU_RUNTIME_PROBE_FULL,
  WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY,
  WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY,
  WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY,
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
assert.ok(base.resolvedTileCount > 0);
assert.ok(base.pixelResolvedCount > 0);
assert.ok(base.pixelResolvedCount > base.resolvedTileCount);
assert.ok(base.resolveWeightSum > 0);
assert.ok(base.resolveAlphaMean > 0);
assert.ok(base.resolveLumaMean > 0);
assert.match(base.resolveChecksum, /^[0-9a-f]{8}$/);
assert.match(base.pixelResolveChecksum, /^[0-9a-f]{8}$/);
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
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /var<storage,\s*read>\s+pixelResolvedRgba/);
assert.match(WEBGPU_TILE_RESOLVE_SHADER, /var<uniform>\s+resolveMeta/);
assert.ok(!WEBGPU_TILE_RESOLVE_SHADER.includes("textureSample"));

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
assert.ok(Math.abs(pixelMeta[6] - base.boundsMinX) < 0.000001);
assert.ok(Math.abs(pixelMeta[7] - base.boundsMinZ) < 0.000001);
assert.ok(Math.abs(pixelMeta[8] - base.boundsSpanX) < 0.000001);
assert.ok(Math.abs(pixelMeta[9] - base.boundsSpanZ) < 0.000001);
assert.equal(pixelMeta[10], 0);
assert.equal(pixelMeta[11], 0);
assert.equal(pixelMeta.byteLength, 48);
assert.equal(WEBGPU_PIXEL_RESOLVE_SOURCE, "webgpu-compute-pixel-accumulation-v1");
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
    `overflow=${base.tileOverflowCount} overflowTiles=${base.tileOverflowTileCount} ` +
    `capacity=${base.tileCapacityGate} storage=${storage.checksum}:${storage.bufferCount} ` +
    `accumulation=${WEBGPU_TILE_ACCUMULATION_SOURCE}:${webGpuAccumulationWorkgroups(base)} ` +
    `compute=${WEBGPU_TILE_COMPUTE_SOURCE}:${webGpuComputeWorkgroups(base)} ` +
    `pixel=${WEBGPU_PIXEL_RESOLVE_SOURCE}:${webGpuPixelResolveWorkgroups(base)} ` +
    `resolveSource=${WEBGPU_TILE_RESOLVE_SOURCE}`,
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
