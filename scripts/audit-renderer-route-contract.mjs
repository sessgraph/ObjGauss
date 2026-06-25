import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-renderer-route-contract");
const generatedAt = new Date().toISOString();

const packageJson = JSON.parse(readFile("package.json"));
const packageScripts = packageJson.scripts ?? {};

const checks = [
  {
    id: "B01",
    phase: "B-webgl-gaussian-oit",
    target: "src/PointCloudViewport.jsx",
    statement: "Object edit fallback is implemented as the current WebGL Gaussian OIT viewport.",
    pass: () => fileExists("src/PointCloudViewport.jsx"),
    evidence: () => ["PointCloudViewport.jsx exists"],
  },
  {
    id: "B02",
    phase: "B-webgl-gaussian-oit",
    target: "src/PointCloudViewport.jsx",
    statement: "The fallback no longer uses PointsMaterial; it uses explicit shader materials.",
    pass: () =>
      contains("src/PointCloudViewport.jsx", "new THREE.ShaderMaterial") &&
      !contains("src/PointCloudViewport.jsx", "PointsMaterial"),
    evidence: () => [
      "new THREE.ShaderMaterial",
      contains("src/PointCloudViewport.jsx", "PointsMaterial") ? "PointsMaterial present" : "PointsMaterial absent",
    ],
  },
  {
    id: "B03",
    phase: "B-webgl-gaussian-oit",
    target: "src/PointCloudViewport.jsx",
    statement: "The edit shader renders screen-space elliptical Gaussian kernels from PLY scale/rotation/opacity.",
    pass: () =>
      containsAll("src/PointCloudViewport.jsx", [
        "attribute vec2 gaussianScale",
        "attribute float gaussianOpacity",
        "attribute float gaussianRotation",
        "gl_PointCoord",
        "exp(-0.5 * d)",
        "uKernelCutoff",
      ]),
    evidence: () => [
      "gaussianScale",
      "gaussianOpacity",
      "gaussianRotation",
      "gl_PointCoord",
      "exp(-0.5 * d)",
    ],
  },
  {
    id: "B04",
    phase: "B-webgl-gaussian-oit",
    target: "src/PointCloudViewport.jsx",
    statement: "The fallback path uses half-float accumulation plus fullscreen weighted OIT resolve.",
    pass: () =>
      containsAll("src/PointCloudViewport.jsx", [
        "THREE.WebGLRenderTarget",
        "THREE.HalfFloatType",
        "uAccumulation",
        "createResolvePass",
        "renderWeightedOitFrame",
        "THREE.CustomBlending",
        "THREE.OneFactor",
      ]),
    evidence: () => [
      "WebGLRenderTarget",
      "HalfFloatType",
      "uAccumulation",
      "renderWeightedOitFrame",
      "CustomBlending/OneFactor",
    ],
  },
  {
    id: "B05",
    phase: "B-webgl-gaussian-oit",
    target: "src/PointCloudViewport.jsx",
    statement: "Object hide/isolate/delete state is GPU-fed through a dense object-state texture.",
    pass: () =>
      containsAll("src/PointCloudViewport.jsx", [
        "gaussianObjectIndex",
        "uObjectState",
        "new THREE.DataTexture",
        "updateObjectStateTexture",
      ]) &&
      containsAll("src/webgpuCapability.js", [
        'GAUSSIAN_OIT_RENDERER_ID = "gaussian-oit"',
        'GAUSSIAN_OIT_OBJECT_FILTER = "gpu-object-state-texture"',
      ]),
    evidence: () => [
      "gaussianObjectIndex",
      "uObjectState DataTexture",
      "GAUSSIAN_OIT_OBJECT_FILTER=gpu-object-state-texture",
    ],
  },
  {
    id: "C01",
    phase: "C-webgpu-tile",
    target: "docs/adr/0005-webgpu-tile-renderer.md",
    statement: "The terminal renderer architecture is captured in the WebGPU tile renderer ADR.",
    pass: () =>
      containsAll("docs/adr/0005-webgpu-tile-renderer.md", [
        "WebGPU tile-based renderer",
        "storage buffers",
        "tile_counts",
        "tile_entries",
        "object_state",
        "Per-tile accumulation",
        "Gaussian OIT",
      ]),
    evidence: () => [
      "WebGPU tile-based renderer",
      "storage buffers",
      "tile_counts/tile_entries/object_state",
      "per-tile accumulation",
    ],
  },
  {
    id: "C02",
    phase: "C-webgpu-tile",
    target: "src/WebGpuTileViewport.jsx",
    statement: "The WebGPU runtime creates device-backed compute/render pipelines and submits work.",
    pass: () =>
      containsAll("src/WebGpuTileViewport.jsx", [
        "navigator.gpu.requestAdapter",
        "requestWebGpuTileDevice",
        "createComputePipeline",
        "createBindGroup",
        "dispatchWorkgroups",
        "device.queue.submit",
      ]),
    evidence: () => [
      "requestAdapter/requestDevice",
      "createComputePipeline",
      "dispatchWorkgroups",
      "queue.submit",
    ],
  },
  {
    id: "C03",
    phase: "C-webgpu-tile",
    target: "src/webgpuTileStorage.js",
    statement: "WebGPU storage layout contains Gaussian, object-state, tile-list, accumulation, and pixel-output buffers.",
    pass: () =>
      containsAll("src/webgpuTileStorage.js", [
        "WEBGPU_TILE_STORAGE_LAYOUT_VERSION",
        "positionRadius",
        "colorOpacity",
        "scaleRotation",
        "objectIndices",
        "objectState",
        "tileCounts",
        "tileOffsets",
        "tileAccumulation",
        "pixelResolvedRgba",
        "tileEntries",
        "canReuseWebGpuTileStorageBuffers",
        "updateWebGpuTileObjectStateBuffer",
      ]),
    evidence: () => [
      "positionRadius/colorOpacity/scaleRotation",
      "objectIndices/objectState",
      "tileCounts/tileOffsets/tileEntries",
      "objectState-only update helper",
      "tileAccumulation/pixelResolvedRgba",
    ],
  },
  {
    id: "C04",
    phase: "C-webgpu-tile",
    target: "src/webgpuTileComputeShader.js",
    statement: "WebGPU shader code performs object-aware tile accumulation and per-pixel Gaussian resolve.",
    pass: () =>
      containsAll("src/webgpuTileComputeShader.js", [
        "WEBGPU_TILE_ACCUMULATION_SHADER",
        "fn accumulationMain",
        "fn pixelResolveMain",
        "var<storage, read> objectState",
        "var<storage, read> tileEntries",
        "var<storage, read> tileOffsets",
        "var<storage, read> scaleRotation",
        "exp(-0.5 * d)",
      ]),
    evidence: () => [
      "accumulationMain",
      "pixelResolveMain",
      "objectState/tileEntries/tileOffsets",
      "scaleRotation + exp(-0.5*d)",
    ],
  },
  {
    id: "C05",
    phase: "C-webgpu-tile",
    target: "src/webgpuTileSmoke.js",
    statement: "The CPU smoke contract packs compact tile lists, object state, camera covariance, and pixel resolve telemetry.",
    pass: () =>
      containsAll("src/webgpuTileSmoke.js", [
        "WEBGPU_TILE_SMOKE_LAYOUT_VERSION",
        "WEBGPU_TILE_ENTRY_LAYOUT_COMPACT",
        "WEBGPU_TILE_LIST_MODE_OBJECT_STATE",
        "WEBGPU_OBJECT_STATE_LAYOUT_VERSION",
        "WEBGPU_TILE_SCREEN_COVARIANCE_MODE",
        "WEBGPU_PIXEL_DEPTH_SORT_MODE",
        "buildWebGpuTileSmoke",
        "tileOffsets",
        "tileEntries",
        "pixelResolveChecksum",
      ]),
    evidence: () => [
      "webgpu-tile-smoke-v1",
      "compact-offset-list",
      "object-state-filtered",
      "object-state-v1",
      "camera-jacobian covariance",
      "pixelResolveChecksum",
    ],
  },
  {
    id: "C06",
    phase: "C-webgpu-tile",
    target: "src/WebGpuTileViewport.jsx",
    statement: "The WebGPU viewport exposes DOM telemetry for renderer, object-state buffer, tile storage, and runtime passes.",
    pass: () =>
      containsAll("src/WebGpuTileViewport.jsx", [
        "data-renderer",
        "data-object-filter",
        "gpu-object-state-buffer",
        "data-webgpu-storage-layout",
        "data-webgpu-storage-update-mode",
        "data-webgpu-storage-update-ms",
        "data-webgpu-storage-object-state-byte-size",
        "data-webgpu-frame-submit-ms",
        "data-webgpu-queue-done-ms",
        "data-webgpu-tile-list-mode",
        "data-webgpu-object-state-layout",
        "data-webgpu-accumulation-source",
        "data-webgpu-pixel-source",
        "data-webgpu-readback-status",
      ]),
    evidence: () => [
      "data-renderer",
      "data-object-filter=gpu-object-state-buffer",
      "storage-update-mode telemetry",
      "storage/submit/queue timing telemetry",
      "tile-list-mode telemetry",
      "storage/object-state/pass/readback telemetry",
    ],
  },
  {
    id: "BR01",
    phase: "bridge-route-contract",
    target: "src/App.jsx",
    statement: "The UI keeps Spark source viewing, WebGPU C-path diagnostics, and Gaussian OIT fallback as explicit routes.",
    pass: () =>
      containsAll("src/App.jsx", [
        "SplatViewport",
        "WebGpuTileViewport",
        "PointCloudViewport",
        "rendererRouteContract",
        "spark-original-view",
        "webgpu-c-path-diagnostic",
        "gaussian-oit-fallback",
      ]),
    evidence: () => [
      "SplatViewport/WebGpuTileViewport/PointCloudViewport",
      "spark-original-view",
      "webgpu-c-path-diagnostic",
      "gaussian-oit-fallback",
    ],
  },
  {
    id: "BR02",
    phase: "bridge-route-contract",
    target: "scripts/audit-demo.mjs",
    statement: "Browser audit checks both WebGPU and fallback renderer telemetry plus object-filter contracts.",
    pass: () =>
      containsAll("scripts/audit-demo.mjs", [
        'data-renderer"',
        '"gaussian-oit"',
        '"webgpu-tile"',
        "data-object-filter",
        "gpu-object-state-buffer",
        "gpu-object-state-texture",
        "data-webgpu-tile-size",
        "object-state-only",
        "full-upload",
        "waitForWebGpuStorageUpdate",
        "storageTimingAfterDelete",
        "webgpu-presentation-only",
      ]),
    evidence: () => [
      "gaussian-oit/webgpu-tile accepted",
      "object-filter contract",
      "object-state-only / full-upload fallback browser audit",
      "object-state timing browser audit output",
      "tile telemetry contract",
    ],
  },
  {
    id: "BR03",
    phase: "bridge-route-contract",
    target: "package.json",
    statement: "Renderer audits and acceptance commands are registered as npm scripts.",
    pass: () =>
      hasScripts([
        "audit:demo",
        "audit:webgpu-tile-smoke",
        "audit:webgpu-scale-budget",
        "audit:webgpu-edit-cost-budget",
        "audit:webgpu-offscreen-readback",
        "audit:webgpu-runtime-performance",
        "audit:webgpu-presentation-performance",
        "audit:webgpu-presentation-transition",
        "audit:webgpu-frame-pacing",
        "audit:webgpu-synthetic-1m-runtime",
        "audit:webgpu-cpath-readiness",
        "audit:spark-native-mask-gate",
        "audit:spark-native-pick-feasibility",
        "acceptance:renderer-ci",
        "acceptance:webgpu-headless",
      ]) &&
      contains("scripts/acceptance-renderer-profile.mjs", "audit:webgpu-edit-cost-budget") &&
      contains("scripts/acceptance-renderer-profile.mjs", "audit:webgpu-presentation-performance") &&
      contains("scripts/acceptance-renderer-profile.mjs", "audit:webgpu-presentation-transition") &&
      contains("scripts/audit-webgpu-cpath-readiness.mjs", "audit:webgpu-synthetic-1m-runtime"),
    evidence: () => [
      "audit:demo",
      "audit:webgpu-tile-smoke",
      "audit:webgpu-scale-budget",
      "audit:webgpu-edit-cost-budget",
      "audit:webgpu-offscreen-readback",
      "audit:webgpu-runtime-performance",
      "audit:webgpu-presentation-performance",
      "audit:webgpu-presentation-transition",
      "audit:webgpu-frame-pacing",
      "audit:webgpu-synthetic-1m-runtime",
      "audit:webgpu-cpath-readiness",
      "acceptance renderer CI includes edit cost budget",
      "acceptance renderer product includes presentation performance gate",
      "acceptance renderer product includes presentation object transition gate",
      "C-path readiness includes synthetic 1M runtime gate",
      "acceptance:renderer-ci",
      "acceptance:webgpu-headless",
    ],
  },
  {
    id: "BR04",
    phase: "bridge-route-contract",
    target: "package.json + scripts/*",
    statement: "Local dev, preview, browser audit, and acceptance defaults are pinned to fixed port 5395 instead of rotating ports.",
    pass: () =>
      contains("package.json", '"dev": "vite --host 127.0.0.1 --port 5395 --strictPort"') &&
      contains("package.json", '"preview": "vite preview --host 127.0.0.1 --port 5395 --strictPort"') &&
      contains("scripts/audit-demo.mjs", "const DEFAULT_PORT = 5395") &&
      contains("scripts/audit-webgpu-synthetic-1m-runtime.mjs", "const DEFAULT_PORT = 5395") &&
      contains("scripts/audit-spark-native-mask-gate.mjs", "const DEFAULT_PORT = 5395") &&
      contains("scripts/acceptance-renderer-profile.mjs", '?? "5395"') &&
      contains("scripts/acceptance-demo.mjs", '|| "5395"'),
    evidence: () => [
      "npm run dev/preview default to 5395 --strictPort",
      "audit-demo DEFAULT_PORT=5395",
      "spark-native-mask-gate DEFAULT_PORT=5395",
      "acceptance renderer/demo defaults=5395",
    ],
  },
  {
    id: "BR05",
    phase: "bridge-route-contract",
    target: "src/SplatViewport.jsx",
    statement: "Spark remains the commercial source-render route while native object picking is still blocked by missing object metadata.",
    pass: () =>
      containsAll("src/SplatViewport.jsx", [
        "@sparkjsdev/spark",
        "SPARK_SELECTION_MODE",
        "screen-space-object-pick-v1",
        "hover-confirm-v1",
        "spark-native-pick-feasibility-v1",
      ]) &&
      contains("package.json", "audit:spark-native-pick-feasibility"),
    evidence: () => [
      "Spark route",
      "screen-space-object-pick-v1",
      "hover-confirm-v1",
      "native-pick feasibility audit",
    ],
  },
];

const results = checks.map((check) => {
  let passed = false;
  let error = "";
  try {
    passed = Boolean(check.pass());
  } catch (caught) {
    error = caught?.message ?? String(caught);
  }
  return {
    id: check.id,
    phase: check.phase,
    target: check.target,
    statement: check.statement,
    status: passed ? "passed" : "failed",
    evidence: safeEvidence(check),
    ...(error ? { error } : {}),
  };
});

const summary = {
  status: results.every((result) => result.status === "passed") ? "passed" : "failed",
  generatedAt,
  outputDir,
  checks: results.length,
  passed: results.filter((result) => result.status === "passed").length,
  failed: results.filter((result) => result.status === "failed").length,
  phases: summarizePhases(results),
  results,
};

writeReport(summary);

console.log(
  [
    `renderer_route_contract=${summary.status}`,
    `checks=${summary.checks}`,
    `passed=${summary.passed}`,
    `failed=${summary.failed}`,
    `phases=${Object.entries(summary.phases)
      .map(([phase, counts]) => `${phase}:${counts.passed}/${counts.total}`)
      .join(",")}`,
    `outputDir=${JSON.stringify(outputDir)}`,
  ].join(" "),
);

if (summary.status !== "passed") {
  process.exitCode = 1;
}

function fileExists(relativePath) {
  return existsSync(resolvePath(relativePath));
}

function readFile(relativePath) {
  const fullPath = resolvePath(relativePath);
  if (!existsSync(fullPath)) {
    throw new Error(`missing file: ${relativePath}`);
  }
  return readFileSync(fullPath, "utf8");
}

function contains(relativePath, needle) {
  return readFile(relativePath).includes(needle);
}

function containsAll(relativePath, needles) {
  const text = readFile(relativePath);
  return needles.every((needle) => text.includes(needle));
}

function hasScripts(scriptNames) {
  return scriptNames.every((name) => typeof packageScripts[name] === "string");
}

function safeEvidence(check) {
  try {
    return check.evidence();
  } catch (caught) {
    return [`evidence error: ${caught?.message ?? String(caught)}`];
  }
}

function summarizePhases(resultsToSummarize) {
  const phases = {};
  for (const result of resultsToSummarize) {
    if (!phases[result.phase]) {
      phases[result.phase] = { total: 0, passed: 0, failed: 0 };
    }
    phases[result.phase].total += 1;
    if (result.status === "passed") phases[result.phase].passed += 1;
    else phases[result.phase].failed += 1;
  }
  return phases;
}

function writeReport(summaryToWrite) {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(
    path.join(outputDir, "summary.json"),
    `${JSON.stringify(summaryToWrite, null, 2)}\n`,
  );
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summaryToWrite));
}

function renderMarkdown(summaryToRender) {
  return [
    "# Renderer Route Contract Audit",
    "",
    `- Status: ${summaryToRender.status}`,
    `- Generated: ${summaryToRender.generatedAt}`,
    `- Checks: ${summaryToRender.passed}/${summaryToRender.checks} passed`,
    `- Output: ${summaryToRender.outputDir}`,
    "",
    "## Phase Summary",
    "",
    "| Phase | Passed | Failed | Total |",
    "| --- | ---: | ---: | ---: |",
    ...Object.entries(summaryToRender.phases).map(
      ([phase, counts]) =>
        `| ${escapeMarkdown(phase)} | ${counts.passed} | ${counts.failed} | ${counts.total} |`,
    ),
    "",
    "## Checks",
    "",
    "| ID | Phase | Status | Target | Statement | Evidence |",
    "| --- | --- | --- | --- | --- | --- |",
    ...summaryToRender.results.map(
      (result) =>
        `| ${result.id} | ${escapeMarkdown(result.phase)} | ${result.status} | \`${escapeMarkdown(result.target)}\` | ${escapeMarkdown(result.statement)} | ${escapeMarkdown(result.evidence.join("; "))} |`,
    ),
    "",
  ].join("\n");
}

function escapeMarkdown(value) {
  return String(value).replaceAll("|", "\\|").replaceAll("\n", " ");
}

function resolvePath(relativePath) {
  return path.join(ROOT, relativePath);
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const entry = argv[index];
    if (!entry.startsWith("--")) continue;
    const key = entry.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}
