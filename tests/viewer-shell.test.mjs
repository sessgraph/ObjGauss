import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const html = readFileSync("viewer/index.html", "utf8");
const app = readFileSync("viewer/app.mjs", "utf8");
const renderer = readFileSync("viewer/splat-renderer.mjs", "utf8");
const syntheticWorld = readFileSync("viewer/synthetic-world.mjs", "utf8");
const pr00View = readFileSync("viewer/pr00-view.mjs", "utf8");
const browserConsumer = readFileSync("src/pr00/browser-consumer.mjs", "utf8");

test("viewer exposes an actual WebGL canvas and local .splat input", () => {
  assert.match(html, /id="splat-canvas"/);
  assert.match(html, /accept="\.splat/);
  assert.doesNotMatch(html, /\.ipynb/);
  assert.match(app, /legobrick\.splat/);
});

test("renderer projects covariance and uses Gaussian alpha, not point sprites", () => {
  assert.match(renderer, /covarianceCamera/);
  assert.match(renderer, /exp\(-0\.5 \* radiusSquared\)/);
  assert.match(renderer, /drawArraysInstanced/);
  assert.doesNotMatch(renderer, /gl\.POINTS/);
  assert.match(renderer, /30_000/);
  assert.match(renderer, /fail-closed/);
  assert.match(renderer, /firstFrame/);
  assert.match(renderer, /gl\.getError\(\)/);
  assert.match(renderer, /MAX_SPLAT_DISPLAY_MULTIPLIER/);
  assert.match(renderer, /MAX_RENDER_DIMENSION = 4_096/);
  assert.match(renderer, /discriminantScale/);
  assert.match(renderer, /any\(isinf\(clip\)\)/);
  assert.match(renderer, /centerPositionsForRendering/);
});

test("world presentation keeps Stage-0 rendering separate from PR-00 contract evidence", () => {
  assert.match(html, /class="world-stage"/);
  assert.match(syntheticWorld, /createSyntheticWorldSplat/);
  assert.match(app, /loadSyntheticWorld/);
  assert.doesNotMatch(renderer, /gl\.LINES/);
  assert.match(html, /id="load-contract"/);
  assert.match(html, /id="contract-workbench"/);
  assert.match(html, /Stage-0 render and PR-00 contract evidence remain separate/);
  assert.match(pr00View, /loadPr00Contract/);
  assert.match(pr00View, /drawCamera/);
  assert.match(pr00View, /drawTrajectory/);
  assert.match(pr00View, /drawAxes/);
  assert.match(pr00View, /mode.*contract/);
  assert.match(html, /WASD/);
  assert.match(renderer, /this\.camera\.yaw = Math\.PI/);
  assert.match(renderer, /this\.boundsRadius \* \(narrowScreen \? 0\.42 : 0\.5\)/);
});

test("browser consumer validates the unique schema, checksums, resources, and report before display", () => {
  assert.match(browserConsumer, /createEpisodeValidator/);
  assert.match(browserConsumer, /schema SHA-256/);
  assert.match(browserConsumer, /episode SHA-256/);
  assert.match(browserConsumer, /machine verdict/);
  assert.match(pr00View, /BLOCKED/);
  assert.match(html, /仅支持 synthetic contract、坐标链与独立重投影门/);
  assert.match(html, /不支持真实数据、Gaussian 重建、世界模型、动力学或规划价值声明/);
});

test("page keeps rendering claims separate from research claims", () => {
  assert.doesNotMatch(html, />3D Gaussian 已在浏览器中实际渲染/);
  assert.match(html, /id="render-claim-title">渲染验证尚未完成/);
  assert.match(app, /kind === "blocked"/);
  assert.match(app, /3D Gaussian 已在浏览器中实际渲染/);
  assert.match(html, /这不是训练好的 ObjGauss 结果/);
  assert.match(html, /id="asset-provenance"/);
  assert.match(html, /id="license-status"/);
  assert.match(app, /synthetic-gaussian-world/);
  assert.match(app, /point-derived-splat/);
});
