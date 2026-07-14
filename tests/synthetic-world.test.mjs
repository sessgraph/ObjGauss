import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { parseSplatV1 } from "../viewer/splat-format.mjs";
import {
  SYNTHETIC_WORLD_SEED,
  createSyntheticWorldSplat,
} from "../viewer/synthetic-world.mjs";

const EXPECTED_BYTES = 272_736;
const EXPECTED_COUNT = 8_523;
const EXPECTED_SHA256 = "4782f6ed4816aee54618bb4d1fcbce8df67e65301e23a89c155985084f51cfe6";

test("synthetic Gaussian world is deterministic and frozen", () => {
  const first = createSyntheticWorldSplat();
  const second = createSyntheticWorldSplat();
  assert.equal(SYNTHETIC_WORLD_SEED, 0x4f_47_57_31);
  assert.equal(first.byteLength, EXPECTED_BYTES);
  assert.deepEqual(new Uint8Array(first), new Uint8Array(second));
  assert.equal(
    createHash("sha256").update(new Uint8Array(first)).digest("hex"),
    EXPECTED_SHA256,
  );
});

test("synthetic world parses as an environment-scale Gaussian scene", () => {
  const parsed = parseSplatV1(createSyntheticWorldSplat());
  assert.equal(parsed.count, EXPECTED_COUNT);
  assert.equal(parsed.visibleCount, EXPECTED_COUNT);
  assert.ok(parsed.bounds.max[0] - parsed.bounds.min[0] > 28);
  assert.ok(parsed.bounds.max[2] - parsed.bounds.min[2] > 28);
  assert.ok(parsed.bounds.max[1] - parsed.bounds.min[1] > 9);
});

test("synthetic world contains anisotropic scales and translucent records", () => {
  const parsed = parseSplatV1(createSyntheticWorldSplat());
  let anisotropic = 0;
  let translucent = 0;
  for (let index = 0; index < parsed.count; index += 1) {
    const scaleOffset = index * 3;
    if (parsed.scales[scaleOffset] !== parsed.scales[scaleOffset + 1]
      || parsed.scales[scaleOffset + 1] !== parsed.scales[scaleOffset + 2]) {
      anisotropic += 1;
    }
    if (parsed.colors[index * 4 + 3] < 255) {
      translucent += 1;
    }
  }
  assert.ok(anisotropic > parsed.count * 0.8);
  assert.ok(translucent > parsed.count * 0.8);
});
