import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import {
  MAX_ABS_POSITION,
  MAX_SPLAT_BYTES,
  MAX_SPLAT_DISPLAY_MULTIPLIER,
  MAX_SPLAT_SCALE,
  SPLAT_FORMAT,
  SPLAT_RECORD_BYTES,
  assertSplatByteLength,
  covarianceFromScaleQuaternion,
  decodeQuaternionBytes,
  parseSplatV1,
} from "../viewer/splat-format.mjs";

const EXPECTED_LEGOBRICK_BYTES = 3_297_920;
const EXPECTED_LEGOBRICK_RECORDS = 103_060;
const EXPECTED_LEGOBRICK_SHA256 = "d5131a664a12a8764da70552c85f567d276313110f63f1efd48424845917899e";

function record({
  position = [1.25, -2.5, 3.75],
  scale = [0.5, 1.5, 2.5],
  color = [12, 34, 56, 200],
  quaternion = [255, 128, 128, 128],
} = {}) {
  const buffer = new ArrayBuffer(SPLAT_RECORD_BYTES);
  const view = new DataView(buffer);
  position.forEach((value, index) => view.setFloat32(index * 4, value, true));
  scale.forEach((value, index) => view.setFloat32(12 + index * 4, value, true));
  color.forEach((value, index) => view.setUint8(24 + index, value));
  quaternion.forEach((value, index) => view.setUint8(28 + index, value));
  return new Uint8Array(buffer);
}

function almostEqual(actual, expected, epsilon = 1e-5) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) <= epsilon, `${value} != ${expected[index]} at ${index}`);
  });
}

test("parses the strict 32-byte little-endian v1 layout", () => {
  const source = record();
  const padded = new Uint8Array(source.byteLength + 9);
  padded.set(source, 5);
  const parsed = parseSplatV1(padded.subarray(5, 5 + source.byteLength));

  assert.equal(parsed.format, SPLAT_FORMAT);
  assert.equal(parsed.count, 1);
  assert.equal(parsed.visibleCount, 1);
  almostEqual(parsed.positions, [1.25, -2.5, 3.75]);
  almostEqual(parsed.scales, [0.5, 1.5, 2.5]);
  assert.deepEqual([...parsed.colors], [12, 34, 56, 200]);
  almostEqual(parsed.quaternions, [1, 0, 0, 0]);
  almostEqual(parsed.bounds.min, [1.25, -2.5, 3.75]);
  almostEqual(parsed.bounds.max, [1.25, -2.5, 3.75]);
  assert.equal(parsed.bounds.radius, 0);
});

test("normalizes decoded wxyz quaternion bytes", () => {
  const quaternion = decodeQuaternionBytes(192, 192, 128, 128);
  almostEqual(quaternion, [Math.SQRT1_2, Math.SQRT1_2, 0, 0]);
});

test("builds identity and 90-degree rotated covariances", () => {
  almostEqual(covarianceFromScaleQuaternion([2, 3, 4], [1, 0, 0, 0]), [4, 0, 0, 9, 0, 16]);
  almostEqual(
    covarianceFromScaleQuaternion([2, 3, 4], [Math.SQRT1_2, 0, 0, Math.SQRT1_2]),
    [9, 0, 0, 4, 0, 16],
  );
});

test("rejects invalid byte lengths before parsing", () => {
  assert.throws(() => parseSplatV1(new ArrayBuffer(0)), /at least one/);
  assert.throws(() => parseSplatV1(new ArrayBuffer(31)), /multiple of 32/);
  assert.throws(() => assertSplatByteLength(MAX_SPLAT_BYTES + SPLAT_RECORD_BYTES), /hard limit/);
  assert.throws(() => parseSplatV1("not bytes"), /ArrayBuffer/);
});

test("rejects non-finite positions and non-positive or non-finite scales", () => {
  for (const badPosition of [NaN, Infinity, -Infinity]) {
    assert.throws(() => parseSplatV1(record({ position: [badPosition, 0, 0] })), /position\[0\] must be finite/);
  }
  assert.throws(
    () => parseSplatV1(record({ position: [3e38, 0, 0] })),
    /position\[0\].*GPU-safe absolute limit/,
  );
  for (const badScale of [0, -1, NaN, Infinity, -Infinity]) {
    assert.throws(() => parseSplatV1(record({ scale: [badScale, 1, 1] })), /scale\[0\].*greater than zero/);
  }
  assert.throws(() => parseSplatV1(record({ scale: [1e-30, 1, 1] })), /float32 covariance/);
  assert.throws(() => parseSplatV1(record({ scale: [3e38, 1, 1] })), /at most 10/);
  assert.equal(MAX_SPLAT_DISPLAY_MULTIPLIER, 32);
  assert.equal(MAX_SPLAT_SCALE, 10);
  assert.equal(MAX_ABS_POSITION, 1_000_000);
  assert.throws(
    () => parseSplatV1(record({ scale: [1.5e19, 1.5e19, 1.5e19] })),
    /at most 10/,
  );
  assert.throws(
    () => parseSplatV1(record({
      scale: [8.152386001310843e17, 1, 1],
      quaternion: [242, 128, 86, 170],
    })),
    /at most 10/,
  );
  assert.throws(() => parseSplatV1(record({ scale: [10.01, 1, 1] })), /at most 10/);
});

test("rejects zero-norm quaternions and fully transparent files", () => {
  assert.throws(() => parseSplatV1(record({ quaternion: [128, 128, 128, 128] })), /invalid quaternion.*non-zero/);
  assert.throws(() => parseSplatV1(record({ color: [12, 34, 56, 0] })), /alpha greater than zero/);
});

test("rejects malformed covariance inputs", () => {
  assert.throws(() => covarianceFromScaleQuaternion([1, 0, 1], [1, 0, 0, 0]), /greater than zero/);
  assert.throws(() => covarianceFromScaleQuaternion([1, 1, 1], [0, 0, 0, 0]), /non-zero/);
  assert.throws(() => covarianceFromScaleQuaternion([1, 1, 1], [NaN, 0, 0, 1]), /finite/);
  assert.throws(() => decodeQuaternionBytes(256, 128, 128, 128), /\[0, 255\]/);
});

test("the ignored legobrick fixture has the pinned bytes, records, and SHA-256", () => {
  const path = "data/local-preview/legobrick-1267e213/legobrick.splat";
  assert.ok(existsSync(path), `missing fixture: ${path}`);

  const bytes = readFileSync(path);
  assert.equal(bytes.byteLength, EXPECTED_LEGOBRICK_BYTES);
  assert.equal(bytes.byteLength / SPLAT_RECORD_BYTES, EXPECTED_LEGOBRICK_RECORDS);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), EXPECTED_LEGOBRICK_SHA256);

  const parsed = parseSplatV1(bytes);
  assert.equal(parsed.count, EXPECTED_LEGOBRICK_RECORDS);
  assert.equal(parsed.visibleCount, EXPECTED_LEGOBRICK_RECORDS);
});
