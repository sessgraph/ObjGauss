import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const DEFAULT_PORT = 5395;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-webgpu-runtime-performance";
const DEFAULT_ASSETS = [
  "nerf-lego-alpha-closure-local",
  "plush-semantic-closure-local",
];
const MODE = "webgpu-runtime-performance-smoke-v1";

const args = parseArgs(process.argv.slice(2));
const port = String(args.port ?? DEFAULT_PORT);
const baseUrl = optionalString(args.url);
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const assets = parseAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const headed = flagEnabled(args.headed ?? args.headful);
const inputSummary = optionalString(args.inputSummary ?? args["input-summary"]);
const skipRun = flagEnabled(args.skipRun ?? args["skip-run"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const maxObjectStateUpdateMs = positiveFiniteNumber(
  args.maxObjectStateUpdateMs ?? args["max-object-state-update-ms"],
  300,
);
const maxFullUploadUpdateMs = positiveFiniteNumber(
  args.maxFullUploadUpdateMs ?? args["max-full-upload-update-ms"],
  500,
);
const maxSubmitMs = positiveFiniteNumber(args.maxSubmitMs ?? args["max-submit-ms"], 25);
const maxQueueDoneMs = positiveFiniteNumber(
  args.maxQueueDoneMs ?? args["max-queue-done-ms"],
  2500,
);
const maxObjectStateBytes = positiveFiniteNumber(
  args.maxObjectStateBytes ?? args["max-object-state-bytes"],
  1024 * 1024,
);
const minLargeSceneGaussians = positiveFiniteNumber(
  args.minLargeSceneGaussians ?? args["min-large-scene-gaussians"],
  250000,
);

if (assets.length === 0) {
  throw new Error("at least one asset is required for WebGPU runtime performance audit");
}

mkdirSync(outputDir, { recursive: true });
const offscreenDir = path.join(outputDir, "offscreen-readback");
const offscreenSummaryPath = inputSummary || path.join(offscreenDir, "summary.json");

if (!inputSummary && !skipRun) {
  if (!existsSync("dist/index.html")) {
    throw new Error("dist/index.html is missing; run `npm run build` before runtime performance audit");
  }
  await run([
    "npm",
    "run",
    "audit:webgpu-offscreen-readback",
    "--",
    "--port",
    port,
    "--output-dir",
    offscreenDir,
    "--assets",
    assets.join(","),
    "--webgpu-flags",
    webGpuFlags,
    ...(headed ? ["--headed"] : []),
    ...(baseUrl ? ["--url", baseUrl] : []),
  ]);
}

const offscreenSummary = readJson(offscreenSummaryPath);
const rows = (offscreenSummary.results ?? []).map((result) => buildPerformanceRow(result));
const checks = buildChecks(rows, offscreenSummary);
const passed = checks.every((check) => check.passed);
const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  sourceSummary: offscreenSummaryPath,
  assets,
  url: offscreenSummary.url ?? baseUrl ?? `http://127.0.0.1:${port}/`,
  webGpuFlags,
  headed,
  budgets: {
    maxObjectStateUpdateMs,
    maxFullUploadUpdateMs,
    maxSubmitMs,
    maxQueueDoneMs,
    maxObjectStateBytes,
    minLargeSceneGaussians,
  },
  passed,
  aggregate: summarizeRows(rows),
  checks,
  rows,
};

writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary), "utf8");

for (const row of rows) {
  console.log(
    `webgpu_runtime_performance_asset=${row.passed ? "passed" : "failed"} ` +
      `asset=${JSON.stringify(row.asset)} gaussians=${row.packedGaussians} tileReferences=${row.tileReferences} ` +
      `initial=${formatTiming(row.initial)} isolate=${formatTiming(row.isolate)} delete=${formatTiming(row.delete)} ` +
      `maxUpdateMs=${row.maxUpdateMs} maxQueueDoneMs=${row.maxQueueDoneMs}`,
  );
}

console.log(
  `webgpu_runtime_performance=${passed ? "passed" : "failed"} ` +
    `assets=${JSON.stringify(assets)} scenes=${rows.length}/${assets.length} ` +
    `maxUpdateMs=${summary.aggregate.maxUpdateMs} maxQueueDoneMs=${summary.aggregate.maxQueueDoneMs} ` +
    `largestGaussians=${summary.aggregate.largestGaussians} ` +
    `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
);

if (!passed && !allowFailures) {
  process.exitCode = 1;
}

function buildPerformanceRow(result) {
  const initial = {
    label: "initial",
    status: result.storageTimingStatus,
    mode: result.storageTimingMode,
    updateMs: numericOrNull(result.storageTimingUpdateMs),
    submitMs: numericOrNull(result.storageTimingSubmitMs),
    queueDoneMs: numericOrNull(result.storageTimingQueueDoneMs),
    objectStateBytes: numericOrNull(result.storageTimingObjectStateByteSize),
  };
  const isolate = {
    label: "isolate",
    status: result.storageTimingAfterIsolateStatus,
    mode: result.storageTimingAfterIsolateMode,
    updateMs: numericOrNull(result.storageTimingAfterIsolateUpdateMs),
    submitMs: numericOrNull(result.storageTimingAfterIsolateSubmitMs),
    queueDoneMs: numericOrNull(result.storageTimingAfterIsolateQueueDoneMs),
    objectStateBytes: numericOrNull(result.storageTimingAfterIsolateObjectStateByteSize),
  };
  const deleteTiming = {
    label: "delete",
    status: result.storageTimingAfterDeleteStatus,
    mode: result.storageTimingAfterDeleteMode,
    updateMs: numericOrNull(result.storageTimingAfterDeleteUpdateMs),
    submitMs: numericOrNull(result.storageTimingAfterDeleteSubmitMs),
    queueDoneMs: numericOrNull(result.storageTimingAfterDeleteQueueDoneMs),
    objectStateBytes: numericOrNull(result.storageTimingAfterDeleteObjectStateByteSize),
  };
  const timings = [initial, isolate, deleteTiming];
  const row = {
    asset: result.asset,
    passed: Boolean(result.passed),
    packedGaussians: numericOrZero(result.packedGaussians),
    tileReferences: numericOrZero(result.tileReferences),
    firstFramePixels: numericOrZero(result.firstFramePixels),
    readbackChecksum: result.readbackChecksum ?? "",
    readbackAfterIsolateChecksum: result.readbackAfterIsolateChecksum ?? "",
    readbackAfterDeleteChecksum: result.readbackAfterDeleteChecksum ?? "",
    objectStateChecksum: result.objectStateChecksum ?? "",
    objectStateAfterIsolateChecksum: result.objectStateAfterIsolateChecksum ?? "",
    objectStateAfterDeleteChecksum: result.objectStateAfterDeleteChecksum ?? "",
    initial,
    isolate,
    delete: deleteTiming,
    maxUpdateMs: maxNumeric(timings.map((entry) => entry.updateMs)),
    maxSubmitMs: maxNumeric(timings.map((entry) => entry.submitMs)),
    maxQueueDoneMs: maxNumeric(timings.map((entry) => entry.queueDoneMs)),
  };
  row.checks = buildRowChecks(row);
  row.passed = row.passed && row.checks.every((check) => check.passed);
  return row;
}

function buildRowChecks(row) {
  return [
    check(`${row.asset}:offscreen-suite-passed`, row.passed, true, row.passed),
    check(`${row.asset}:packed-gaussians`, row.packedGaussians, ">0", row.packedGaussians > 0),
    check(`${row.asset}:tile-references`, row.tileReferences, ">0", row.tileReferences > 0),
    checkTiming(row, row.initial, {
      allowed: [
        { status: "uploaded", mode: "full-upload" },
        { status: "object-state-updated", mode: "object-state-only" },
      ],
    }),
    checkTiming(row, row.isolate, {
      allowed: [{ status: "object-state-updated", mode: "object-state-only" }],
    }),
    checkTiming(row, row.delete, {
      allowed: [
        { status: "object-state-updated", mode: "object-state-only" },
        { status: "uploaded", mode: "full-upload" },
      ],
      queueDoneMayBePending: true,
    }),
  ].flat();
}

function checkTiming(row, timing, { allowed, queueDoneMayBePending = false }) {
  const timingBudget =
    timing.mode === "full-upload" ? maxFullUploadUpdateMs : maxObjectStateUpdateMs;
  const expected = allowed.map((entry) => `${entry.status}/${entry.mode}`).join(" or ");
  return [
    check(
      `${row.asset}:${timing.label}:mode`,
      `${timing.status}/${timing.mode}`,
      expected,
      allowed.some((entry) => entry.status === timing.status && entry.mode === timing.mode),
    ),
    check(
      `${row.asset}:${timing.label}:update-ms`,
      timing.updateMs,
      `<= ${timingBudget}`,
      isNonNegativeNumber(timing.updateMs) && timing.updateMs <= timingBudget,
    ),
    check(
      `${row.asset}:${timing.label}:submit-ms`,
      timing.submitMs,
      `<= ${maxSubmitMs}`,
      isNonNegativeNumber(timing.submitMs) && timing.submitMs <= maxSubmitMs,
    ),
    check(
      `${row.asset}:${timing.label}:queue-done-ms`,
      timing.queueDoneMs,
      queueDoneMayBePending ? `0 or <= ${maxQueueDoneMs}` : `<= ${maxQueueDoneMs}`,
      isNonNegativeNumber(timing.queueDoneMs) &&
        (queueDoneMayBePending
          ? timing.queueDoneMs === 0 || timing.queueDoneMs <= maxQueueDoneMs
          : timing.queueDoneMs <= maxQueueDoneMs),
    ),
    check(
      `${row.asset}:${timing.label}:object-state-bytes`,
      timing.objectStateBytes,
      `1..${maxObjectStateBytes}`,
      isNonNegativeNumber(timing.objectStateBytes) &&
        timing.objectStateBytes > 0 &&
        timing.objectStateBytes <= maxObjectStateBytes,
    ),
  ];
}

function buildChecks(rows, offscreenSummary) {
  const largeScenes = rows.filter((row) => row.packedGaussians >= minLargeSceneGaussians);
  return [
    check("source-summary-passed", offscreenSummary.passed, true, offscreenSummary.passed === true),
    check("asset-count", rows.length, assets.length, rows.length === assets.length),
    check(
      "large-scene-coverage",
      largeScenes.map((row) => `${row.asset}:${row.packedGaussians}`).join(","),
      `>= ${minLargeSceneGaussians}`,
      largeScenes.length > 0,
    ),
    ...rows.flatMap((row) => row.checks),
  ];
}

function summarizeRows(rows) {
  return {
    scenes: rows.length,
    largestGaussians: maxNumeric(rows.map((row) => row.packedGaussians)),
    maxTileReferences: maxNumeric(rows.map((row) => row.tileReferences)),
    maxUpdateMs: maxNumeric(rows.map((row) => row.maxUpdateMs)),
    maxSubmitMs: maxNumeric(rows.map((row) => row.maxSubmitMs)),
    maxQueueDoneMs: maxNumeric(rows.map((row) => row.maxQueueDoneMs)),
  };
}

function renderMarkdown(summary) {
  const lines = [
    "# WebGPU Runtime Performance Smoke",
    "",
    `Mode: \`${summary.mode}\``,
    `Generated: \`${summary.generatedAt}\``,
    `Status: \`${summary.passed ? "passed" : "failed"}\``,
    `Source: \`${summary.sourceSummary}\``,
    "",
    "This is a browser runtime timing envelope for the WebGPU C-path object edit transition. It is not an FPS benchmark or a 1M interactive SLA.",
    "",
    "## Budgets",
    "",
    "| Budget | Value |",
    "| --- | ---: |",
    `| Object-state update | ${summary.budgets.maxObjectStateUpdateMs} ms |`,
    `| Full-upload update | ${summary.budgets.maxFullUploadUpdateMs} ms |`,
    `| Queue submit | ${summary.budgets.maxSubmitMs} ms |`,
    `| Queue done | ${summary.budgets.maxQueueDoneMs} ms |`,
    `| Object-state bytes | ${summary.budgets.maxObjectStateBytes} |`,
    "",
    "## Results",
    "",
    "| Asset | Passed | Gaussians | Tile refs | Initial | Isolate | Delete | Max update | Max queue done |",
    "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
  ];
  for (const row of summary.rows) {
    lines.push(
      `| ${escapeMarkdown(row.asset)} | ${row.passed ? "yes" : "no"} | ${row.packedGaussians} | ${row.tileReferences} | ${escapeMarkdown(formatTiming(row.initial))} | ${escapeMarkdown(formatTiming(row.isolate))} | ${escapeMarkdown(formatTiming(row.delete))} | ${row.maxUpdateMs} | ${row.maxQueueDoneMs} |`,
    );
  }
  const failed = summary.checks.filter((entry) => !entry.passed);
  lines.push("", "## Failed Checks", "");
  if (failed.length === 0) {
    lines.push("None.");
  } else {
    lines.push("| Check | Actual | Expected |", "| --- | --- | --- |");
    for (const entry of failed) {
      lines.push(
        `| ${escapeMarkdown(entry.name)} | ${escapeMarkdown(String(entry.actual))} | ${escapeMarkdown(String(entry.expected))} |`,
      );
    }
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function check(name, actual, expected, passed) {
  return { name, actual, expected, passed: Boolean(passed) };
}

function readJson(filePath) {
  if (!existsSync(filePath)) {
    throw new Error(`missing summary: ${filePath}`);
  }
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function run(command) {
  return new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), { stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command.join(" ")} exited with ${code}`));
    });
  });
}

function parseAssets(value) {
  return String(value ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
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

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return "";
  const text = String(value).trim();
  return text || "";
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function positiveFiniteNumber(value, fallback) {
  if (value === undefined || value === null || value === true || value === false) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function numericOrNull(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function numericOrZero(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function isNonNegativeNumber(value) {
  return Number.isFinite(value) && value >= 0;
}

function maxNumeric(values) {
  const numeric = values.filter((value) => Number.isFinite(value));
  if (numeric.length === 0) return 0;
  return Math.max(...numeric);
}

function formatTiming(timing) {
  return `${timing.mode}:${timing.updateMs}/${timing.submitMs}/${timing.queueDoneMs}ms`;
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
