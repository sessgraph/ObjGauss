import assert from "node:assert/strict";
import test from "node:test";
import {
  cameraDepth,
  sortSplatIndices,
} from "../viewer/splat-sort.mjs";
import {
  centerPositionsForRendering,
  lookAtMatrix,
  perspectiveMatrix,
} from "../viewer/splat-renderer.mjs";

const IDENTITY_VIEW = Float32Array.from([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

test("sorts stable far-to-near for premultiplied alpha", () => {
  const positions = Float32Array.from([
    0, 0, -1,
    0, 0, -3,
    0, 0, -2,
    1, 0, -3,
  ]);
  assert.deepEqual([...sortSplatIndices(positions, IDENTITY_VIEW)], [1, 3, 2, 0]);
});

test("a reversed camera flips the red/blue overlap order", () => {
  const positions = Float32Array.from([
    0, 0, -2, // red is farther in the identity view
    0, 0, -1, // blue is nearer in the identity view
  ]);
  assert.deepEqual([...sortSplatIndices(positions, IDENTITY_VIEW)], [0, 1]);
  const reverseZ = Float32Array.from([
    -1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1,
  ]);
  assert.deepEqual([...sortSplatIndices(positions, reverseZ)], [1, 0]);
});

test("equal-depth input remains stable and indices form a permutation", () => {
  const positions = Float32Array.from([
    -1, 0, -2,
    0, 0, -2,
    1, 0, -2,
  ]);
  assert.deepEqual([...sortSplatIndices(positions, IDENTITY_VIEW)], [0, 1, 2]);
});

test("lookAt and perspective matrices produce finite camera depth", () => {
  const view = lookAtMatrix([0, 0, 5], [0, 0, 0]);
  const projection = perspectiveMatrix(Math.PI / 4, 16 / 9, 0.1, 100);
  assert.ok([...view, ...projection].every(Number.isFinite));
  assert.ok(cameraDepth(0, 0, 0, view) > 0);
  assert.throws(() => perspectiveMatrix(Math.PI / 4, 0, 0.1, 100), /positive frustum/);
  assert.throws(() => lookAtMatrix([0, 0, 0], [0, 0, 0]), /distinct/);
});

test("recenters large absolute positions before camera and depth math", () => {
  const single = centerPositionsForRendering(
    Float32Array.from([1_000_000, 1_000_000, 1_000_000]),
    Float32Array.from([1_000_000, 1_000_000, 1_000_000]),
  );
  assert.deepEqual([...single], [0, 0, 0]);
  const singleView = lookAtMatrix([0, 0, 0.0303], [0, 0, 0]);
  assert.ok(cameraDepth(single[0], single[1], single[2], singleView) > 0.01);

  const positions = Float32Array.from([
    1_000_000, 1_000_000, 1_000_000,
    1_000_001, 1_000_000, 999_999,
  ]);
  const centered = centerPositionsForRendering(
    positions,
    Float32Array.from([1_000_000.5, 1_000_000, 999_999.5]),
  );
  assert.deepEqual([...centered], [-0.5, 0, 0.5, 0.5, 0, -0.5]);
  assert.ok(cameraDepth(centered[3], centered[4], centered[5], singleView) > 0);
});

test("rejects malformed sort inputs", () => {
  assert.throws(() => sortSplatIndices(new Float32Array(), IDENTITY_VIEW), /non-empty/);
  assert.throws(() => sortSplatIndices(Float32Array.of(1, 2), IDENTITY_VIEW), /xyz triples/);
  assert.throws(() => sortSplatIndices(Float32Array.of(1, 2, 3), [1, 2]), /16 values/);
});
