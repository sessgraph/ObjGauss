import assert from "node:assert/strict";

import { createSampleScene } from "../src/sampleScene.js";
import { buildWebGpuTileSmoke, WEBGPU_TILE_SMOKE_LAYOUT_VERSION } from "../src/webgpuTileSmoke.js";

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
assert.equal(base.buffers.objectState.length, allObjectIds.size);
assert.equal(base.buffers.tileCounts.length, base.tileCount);
assert.equal(base.buffers.tileEntries.length, base.tileEntryCapacity);
assert.ok(base.visibleGaussians > 0);
assert.ok(base.binnedGaussians > 0);
assert.ok(base.activeTileCount > 0);
assert.ok(base.tileReferenceCount >= base.binnedGaussians);

const isolated = buildWebGpuTileSmoke({
  points: scene.points,
  visibleIds: allObjectIds,
  removedIds: new Set(),
  isolatedId: firstObjectId,
  renderMode: "clustered",
  pointSize: 0.018,
});

assert.ok(isolated.visibleGaussians < base.visibleGaussians);
assert.ok(isolated.binnedGaussians < base.binnedGaussians);
assert.ok(isolated.tileReferenceCount <= base.tileReferenceCount);

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

console.log(
  `webgpu_tile_smoke=passed packed=${base.packedGaussians} ` +
    `objects=${base.objectCount} tiles=${base.activeTileCount}/${base.tileCount} ` +
    `refs=${base.tileReferenceCount} overflow=${base.tileOverflowCount}`,
);
