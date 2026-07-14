import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { BRANCH_ORDER, recordAt, worldToCanvas } from "../viewer/pr01/render.mjs";

test("delivery viewer freezes exactly five sibling branches", () => {
  assert.deepEqual(BRANCH_ORDER, ["hold", "push-neg-x-weak", "push-pos-x-weak", "push-pos-x-strong", "push-pos-y-weak"]);
});

test("shared timeline selects the latest non-future record", () => {
  const records = [{ episode_time_s: 0 }, { episode_time_s: 0.1 }, { episode_time_s: 1.1 }];
  assert.equal(recordAt(records, 0.09), records[0]);
  assert.equal(recordAt(records, 0.1), records[1]);
  assert.equal(recordAt(records, 5), records[2]);
});

test("world-to-canvas preserves a shared top-view coordinate convention", () => {
  const canvas = { width: 480, height: 340 };
  const origin = worldToCanvas(canvas, [0, 0, 0]);
  const positive = worldToCanvas(canvas, [0.1, 0.1, 0]);
  assert.deepEqual(origin, [240, 170]);
  assert.ok(positive[0] > origin[0]);
  assert.ok(positive[1] < origin[1]);
});

test("viewer is fail-closed on audit and artifact checksums without RGB, WebGL or CDN", async () => {
  const [html, app, render] = await Promise.all([
    readFile("viewer/pr01/index.html", "utf8"),
    readFile("viewer/pr01/app.mjs", "utf8"),
    readFile("viewer/pr01/render.mjs", "utf8"),
  ]);
  assert.match(html, /id="branches"/);
  assert.match(html, /id="timeline"/);
  assert.match(app, /report\.verdict\.status !== "supported"/);
  assert.match(app, /checksum mismatch/);
  assert.match(render, /point\.normal_W/);
  assert.match(render, /point\.impulse_W_N_s/);
  assert.match(render, /time \+ 1e-9 < contacts\.records\[0\]\.episode_time_s/);
  assert.doesNotMatch(`${html}\n${app}\n${render}`, /WebGL|cdn\.|https?:\/\/|rgb_uri|rgb-card/i);
});

test("delivery and acceptance fail closed on a dirty checkout", async () => {
  const [accept, build, verify, workflow] = await Promise.all([
    readFile("scripts/accept-pr01", "utf8"),
    readFile("scripts/build-pr01-delivery.mjs", "utf8"),
    readFile("scripts/verify-pr01-delivery.mjs", "utf8"),
    readFile(".github/workflows/pr01-delivery.yml", "utf8"),
  ]);
  for (const source of [accept, build, verify]) {
    assert.match(source, /status/);
    assert.match(source, /--porcelain=v1/);
    assert.match(source, /--untracked-files=all/);
    assert.match(source, /clean checkout/);
  }
  assert.match(verify, /successful attempt ledger count differs/);
  assert.match(verify, /standalone failed attempt ledger count differs/);
  assert.match(accept, /\.\/scripts\/check-pr01b-runtime/);
  assert.match(accept, /\.\/scripts\/check-pr01e-cohort/);
  assert.match(workflow, /github\.event\.pull_request\.head\.sha/);
  assert.match(workflow, /github\.sha/);
});
