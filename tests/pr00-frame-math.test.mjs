import assert from "node:assert/strict";
import test from "node:test";
import {
  FrameMathError,
  invertRigidTransform,
  multiplyTransforms,
  projectWorldPoint,
  rigidTransformFromQuaternionTranslation,
  transformPoint,
  unprojectPixel,
} from "../src/pr00/frame-math.mjs";
import { createSyntheticAudit } from "../src/pr00/synthetic-audit.mjs";

function close(actual, expected, tolerance = 1e-9) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) <= tolerance, `${value} != ${expected[index]}`));
}

test("T_AB · p_B = p_A composes and inverts rigid transforms", () => {
  const T_WO = rigidTransformFromQuaternionTranslation([1, 0, 0, 0], [2, 3, 4]);
  const T_OC = rigidTransformFromQuaternionTranslation([Math.SQRT1_2, 0, 0, Math.SQRT1_2], [1, 0, 0]);
  const T_WC = multiplyTransforms(T_WO, T_OC);
  close(transformPoint(T_WC, [0, 0, 0]), transformPoint(T_WO, [1, 0, 0]));
  close(transformPoint(invertRigidTransform(T_WC), transformPoint(T_WC, [0.3, -0.2, 1.1])), [0.3, -0.2, 1.1]);
  const T_OW = invertRigidTransform(T_WO);
  close(multiplyTransforms(T_OW, T_WC), T_OC);
});

test("projection and unprojection round-trip in Robotics/OpenCV coordinates", () => {
  const { episode } = createSyntheticAudit();
  for (const auditPoint of episode.audit.primary_points) {
    const observation = episode.observations.find((candidate) => candidate.observation_id === auditPoint.observation_id);
    const T_WC = observation.T_WC.value.matrix_row_major;
    const K = observation.K.value.matrix_row_major;
    const imageSizePx = observation.K.value.image_size_px;
    const projected = projectWorldPoint({ T_WC, K, imageSizePx, pointW: auditPoint.point_W_m });
    close(projected.pixel, auditPoint.expected_pixel, 1e-9);
    close(unprojectPixel({ T_WC, K, imageSizePx, pixel: projected.pixel, depthM: projected.depth_m }), auditPoint.point_W_m, 1e-9);
  }
});

test("projection rejects behind-camera, out-of-bounds, and singular inputs", () => {
  const { episode } = createSyntheticAudit();
  const observation = episode.observations[0];
  const T_WC = observation.T_WC.value.matrix_row_major;
  const K = observation.K.value.matrix_row_major;
  const imageSizePx = observation.K.value.image_size_px;
  const behindW = transformPoint(T_WC, [0, 0, -1]);
  const outsideW = transformPoint(T_WC, [100, 0, 1]);
  assert.throws(
    () => projectWorldPoint({ T_WC, K, imageSizePx, pointW: behindW }),
    (error) => error instanceof FrameMathError && error.code === "BEHIND_CAMERA",
  );
  assert.throws(
    () => projectWorldPoint({ T_WC, K, imageSizePx, pointW: outsideW }),
    (error) => error instanceof FrameMathError && error.code === "OUT_OF_BOUNDS",
  );
  assert.throws(
    () => projectWorldPoint({ T_WC, K: new Array(9).fill(0), imageSizePx, pointW: auditPoint }),
    (error) => error instanceof FrameMathError && error.code === "SINGULAR_INTRINSICS",
  );
});

const auditPoint = [0, 0, 1];

test("non-rigid, row/column-mixed, and left-handed matrices fail closed", () => {
  const rowColumnMixed = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1];
  const leftHanded = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  assert.throws(() => invertRigidTransform(rowColumnMixed), /must end with/);
  assert.throws(() => invertRigidTransform(leftHanded), /determinant must equal \+1/);
});
