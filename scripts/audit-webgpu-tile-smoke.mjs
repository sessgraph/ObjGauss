import assert from "node:assert/strict";

import { createSampleScene } from "../src/sampleScene.js";
import {
  buildWebGpuTileSmoke,
  WEBGPU_OBJECT_STATE_LAYOUT_VERSION,
  WEBGPU_OBJECT_STATE_STRIDE_UINT32,
  WEBGPU_TILE_SMOKE_LAYOUT_VERSION,
} from "../src/webgpuTileSmoke.js";
import { editRendererContract } from "../src/webgpuCapability.js";

const scene = createSampleScene();
const allObjectIds = new Set(scene.points.map((point) => point.objectId));
const firstObjectId = Math.min(...allObjectIds);

const base = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  includeTileEntries: true,
  maxEntriesPerTile: 64,
});

assert.equal(base.layoutVersion, WEBGPU_TILE_SMOKE_LAYOUT_VERSION);
assert.equal(base.packedGaussians, scene.points.length);
assert.equal(base.buffers.positionRadius.length, scene.points.length * 4);
assert.equal(base.buffers.colorOpacity.length, scene.points.length * 4);
assert.equal(base.buffers.scaleRotation.length, scene.points.length * 4);
assert.equal(base.buffers.objectIndices.length, scene.points.length);
assert.equal(base.buffers.objectState.length, allObjectIds.size * WEBGPU_OBJECT_STATE_STRIDE_UINT32);
assert.equal(base.buffers.objectIds.length, allObjectIds.size);
assert.equal(base.buffers.tileCounts.length, base.tileCount);
assert.equal(base.buffers.tileAccumulation.length, base.tileCount * 4);
assert.equal(base.buffers.tileResolvedRgba.length, base.tileCount * 4);
assert.equal(base.buffers.tileEntries.length, base.tileEntryCapacity);
assert.ok(base.visibleGaussians > 0);
assert.ok(base.binnedGaussians > 0);
assert.ok(base.activeTileCount > 0);
assert.ok(base.tileReferenceCount >= base.binnedGaussians);
assert.equal(base.tileCapacityMode, "fixed-cap-smoke");
assert.equal(base.tileCapacityStatus, "overflow");
assert.equal(base.tileCapacityGate, "blocked");
assert.ok(base.tileOverflowCount > 0);
assert.ok(base.tileOverflowTileCount > 0);
assert.ok(base.tileOverflowRatio > 0);
assert.ok(base.tileOverflowMaxExcess > 0);
assert.equal(base.tileEntryStoredCount, base.tileReferenceCount - base.tileOverflowCount);
assert.equal(base.tileEntryCapacity, base.tileCount * base.maxEntriesPerTile);
assert.ok(base.tileEntryUtilization > 0 && base.tileEntryUtilization <= 1);
assert.equal(base.resolveVersion, "webgpu-tile-resolve-v1");
assert.equal(base.resolveMode, "tile-center-weighted-oit");
assert.ok(base.resolvedTileCount > 0);
assert.ok(base.resolveWeightSum > 0);
assert.ok(base.resolveAlphaMean > 0);
assert.ok(base.resolveLumaMean > 0);
assert.match(base.resolveChecksum, /^[0-9a-f]{8}$/);
assert.equal(base.objectStateLayoutVersion, WEBGPU_OBJECT_STATE_LAYOUT_VERSION);
assert.equal(base.objectStateStrideUint32, WEBGPU_OBJECT_STATE_STRIDE_UINT32);
assert.equal(base.objectStateVisibleObjects, allObjectIds.size);
assert.equal(base.objectStateHiddenObjects, 0);
assert.equal(base.objectStateRemovedObjects, 0);
assert.equal(base.objectStateSelectedObjects, 0);
assert.equal(base.objectStateIsolatedObjects, 0);
assert.match(base.objectStateChecksum, /^[0-9a-f]{8}$/);

const roomy = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: null,
  renderMode: "original",
  pointSize: 0.018,
  maxEntriesPerTile: 100000,
});

assert.equal(roomy.tileCapacityMode, "fixed-cap-smoke");
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
};
const roomyContract = editRendererContract(readyCapability, roomy);
assert.equal(roomyContract.rendererId, "webgpu-tile");
assert.equal(roomyContract.objectFilter, "gpu-object-state-buffer");
assert.equal(roomyContract.targetGate, "pass");
assert.equal(roomyContract.targetGateReason, "webgpu-tile-first-frame-ready");
assert.equal(roomyContract.fallbackReason, "");

const overflowContract = editRendererContract(readyCapability, base);
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

console.log(
  `webgpu_tile_smoke=passed packed=${base.packedGaussians} ` +
    `objects=${base.objectCount} tiles=${base.activeTileCount}/${base.tileCount} ` +
    `refs=${base.tileReferenceCount} resolved=${base.resolvedTileCount} ` +
    `checksum=${base.resolveChecksum} objectState=${base.objectStateChecksum} ` +
    `overflow=${base.tileOverflowCount} overflowTiles=${base.tileOverflowTileCount} ` +
    `capacity=${base.tileCapacityGate}`,
);
