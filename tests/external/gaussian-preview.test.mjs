import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { SPLAT_RECORD_BYTES, parseSplatV1 } from "../../viewer/splat-format.mjs";

const PREVIEW_PATH = "data/local-preview/legobrick-1267e213/legobrick.splat";
const EXPECTED_BYTES = 3_297_920;
const EXPECTED_RECORDS = 103_060;
const EXPECTED_SHA256 = "d5131a664a12a8764da70552c85f567d276313110f63f1efd48424845917899e";

test("the downloaded ignored legobrick preview has the pinned bytes, records, and SHA-256", () => {
  assert.ok(
    existsSync(PREVIEW_PATH),
    `missing external preview: ${PREVIEW_PATH}; run bash scripts/fetch-gaussian-preview.sh first`,
  );

  const bytes = readFileSync(PREVIEW_PATH);
  assert.equal(bytes.byteLength, EXPECTED_BYTES);
  assert.equal(bytes.byteLength / SPLAT_RECORD_BYTES, EXPECTED_RECORDS);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), EXPECTED_SHA256);

  const parsed = parseSplatV1(bytes);
  assert.equal(parsed.count, EXPECTED_RECORDS);
  assert.equal(parsed.visibleCount, EXPECTED_RECORDS);
});
