import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const DEFAULT_PORT = 5395;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-webgpu-sustained-frame-pacing";
const DEFAULT_FRAME_COUNT = 120;
const MODE = "webgpu-sustained-frame-pacing-baseline-v1";

const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const port = String(args.port ?? DEFAULT_PORT);
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const skipBuild = flagEnabled(args.skipBuild ?? args["skip-build"]);
const skipRun = flagEnabled(args.skipRun ?? args["skip-run"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const realFrameCount = positiveInteger(
  args.realFrameCount ?? args["real-frame-count"] ?? args.frameCount ?? args["frame-count"],
  DEFAULT_FRAME_COUNT,
);
const syntheticFrameCount = positiveInteger(
  args.syntheticFrameCount ??
    args["synthetic-frame-count"] ??
    args.frameCount ??
    args["frame-count"],
  DEFAULT_FRAME_COUNT,
);
const minRealApproxFps = positiveFiniteNumber(
  args.minRealApproxFps ?? args["min-real-approx-fps"],
  10,
);
const minSyntheticApproxFps = positiveFiniteNumber(
  args.minSyntheticApproxFps ?? args["min-synthetic-approx-fps"],
  8,
);
const maxRealMeanFrameMs = positiveFiniteNumber(
  args.maxRealMeanFrameMs ?? args["max-real-mean-frame-ms"],
  120,
);
const maxSyntheticMeanFrameMs = positiveFiniteNumber(
  args.maxSyntheticMeanFrameMs ?? args["max-synthetic-mean-frame-ms"],
  150,
);
const maxP95FrameMs = positiveFiniteNumber(args.maxP95FrameMs ?? args["max-p95-frame-ms"], 220);
const maxLongFrameRatio = positiveFiniteNumber(
  args.maxLongFrameRatio ?? args["max-long-frame-ratio"],
  0.25,
);

const realDir = path.join(outputDir, "real-scenes-frame-pacing");
const syntheticDir = path.join(outputDir, "synthetic-1m-frame-pacing");
const realSummaryPath = String(
  args.realSummary ?? args["real-summary"] ?? path.join(realDir, "summary.json"),
);
const syntheticSummaryPath = String(
  args.syntheticSummary ?? args["synthetic-summary"] ?? path.join(syntheticDir, "summary.json"),
);

mkdirSync(outputDir, { recursive: true });
const steps = [];

try {
  if (!skipRun) {
    if (!skipBuild) {
      await runStep("Build viewer", ["npm", "run", "build"]);
    } else if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; remove --skip-build or run `npm run build`");
    }
    await runStep("Current real-scene sustained frame pacing", [
      "npm",
      "run",
      "audit:webgpu-frame-pacing",
      "--",
      "--port",
      port,
      "--webgpu-flags",
      webGpuFlags,
      "--output-dir",
      realDir,
      "--frame-count",
      String(realFrameCount),
      "--max-mean-frame-ms",
      String(maxRealMeanFrameMs),
      "--max-p95-frame-ms",
      String(maxP95FrameMs),
      "--max-long-frame-ratio",
      String(maxLongFrameRatio),
      "--min-approx-fps",
      String(minRealApproxFps),
      ...(allowFailures ? ["--allow-failures"] : []),
    ]);
    await runStep("Synthetic 1M sustained frame pacing", [
      "npm",
      "run",
      "audit:webgpu-synthetic-1m-runtime",
      "--",
      "--port",
      port,
      "--webgpu-flags",
      webGpuFlags,
      "--output-dir",
      syntheticDir,
      "--frame-count",
      String(syntheticFrameCount),
      "--max-mean-frame-ms",
      String(maxSyntheticMeanFrameMs),
      "--max-p95-frame-ms",
      String(maxP95FrameMs),
      "--max-long-frame-ratio",
      String(maxLongFrameRatio),
      "--min-approx-fps",
      String(minSyntheticApproxFps),
      ...(allowFailures ? ["--allow-failures"] : []),
    ]);
  }

  const realSummary = readJsonOrNull(realSummaryPath);
  const syntheticSummary = readJsonOrNull(syntheticSummaryPath);
  const evidence = buildEvidence({ realSummary, syntheticSummary });
  const checks = buildChecks(evidence);
  const passed = checks.every((entry) => entry.passed);
  const summary = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    outputDir,
    port,
    webGpuFlags,
    skipBuild,
    skipRun,
    thresholds: {
      realFrameCount,
      syntheticFrameCount,
      minRealApproxFps,
      minSyntheticApproxFps,
      maxRealMeanFrameMs,
      maxSyntheticMeanFrameMs,
      maxP95FrameMs,
      maxLongFrameRatio,
    },
    sourceSummaries: {
      realScenes: realSummaryPath,
      synthetic1m: syntheticSummaryPath,
    },
    passed,
    status: passed ? "passed" : "failed",
    fpsBaseline:
      passed && evidence.realScenes.status === "passed" && evidence.synthetic1m.status === "passed"
        ? "baseline-passed"
        : "not-proven",
    steps,
    evidence,
    checks,
    remainingGaps: [
      {
        id: "real-trained-browser-runtime-1m",
        status: "not-proven",
        nextEvidence:
          "Run a trained or captured real scene near 1M Gaussians through the same sustained frame pacing gate.",
      },
      {
        id: "production-fps-sla",
        status: "not-proven",
        nextEvidence:
          "Promote this baseline into a production SLA only after threshold review on real trained 1M scenes and target hardware.",
      },
    ],
    interpretation:
      "This is a longer fixed-port headed browser frame-pacing baseline for current real scenes plus synthetic 1M upload/runtime. It is stronger than the short smoke gates, but it is not a production FPS SLA and does not prove trained 1M scene quality.",
  };
  writeReport(summary);
  console.log(
    `webgpu_sustained_frame_pacing=${summary.status} fpsBaseline=${summary.fpsBaseline} ` +
      `realMinApproxFps=${summary.evidence.realScenes.minApproxFps} ` +
      `syntheticMinApproxFps=${summary.evidence.synthetic1m.minApproxFps} ` +
      `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
  );
  if (!passed && !allowFailures) process.exitCode = 1;
} catch (error) {
  const summary = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    outputDir,
    port,
    webGpuFlags,
    status: "failed",
    passed: false,
    error: error?.message ?? String(error),
    steps,
  };
  writeReport(summary);
  throw error;
}

async function runStep(label, command) {
  console.log(`\n=== ${label} ===`);
  console.log(command.join(" "));
  const startedAt = Date.now();
  const result = await runProcess(command[0], command.slice(1));
  const durationMs = Date.now() - startedAt;
  const step = {
    label,
    command: command.join(" "),
    exitCode: result.exitCode,
    durationMs,
  };
  steps.push(step);
  if (result.exitCode !== 0 && !allowFailures) {
    const error = new Error(`${label} failed with exit code ${result.exitCode}`);
    error.result = step;
    throw error;
  }
}

function buildEvidence({ realSummary, syntheticSummary }) {
  const realAggregate = realSummary?.aggregate ?? {};
  const syntheticAggregate = syntheticSummary?.aggregate ?? {};
  const syntheticRow = syntheticSummary?.row ?? {};
  return {
    realScenes: {
      status:
        realSummary?.passed === true &&
        numeric(realAggregate.minApproxFps) >= minRealApproxFps &&
        numeric(realAggregate.maxMeanFrameMs) <= maxRealMeanFrameMs &&
        numeric(realAggregate.maxP95FrameMs) <= maxP95FrameMs &&
        numeric(realAggregate.maxLongFrameRatio) <= maxLongFrameRatio
          ? "passed"
          : "failed",
      mode: realSummary?.mode ?? "",
      scenes: numeric(realAggregate.scenes),
      largestGaussians: numeric(realAggregate.largestGaussians),
      maxTileReferences: numeric(realAggregate.maxTileReferences),
      minApproxFps: numeric(realAggregate.minApproxFps),
      maxMeanFrameMs: numeric(realAggregate.maxMeanFrameMs),
      maxP95FrameMs: numeric(realAggregate.maxP95FrameMs),
      maxLongFrameRatio: numeric(realAggregate.maxLongFrameRatio),
      frameCount: realFrameCount,
      summaryPath: realSummaryPath,
      interpretation:
        "Current real scenes use longer rAF sampling on Lego proxy and Plush semantic while staying on WebGPU Tile.",
    },
    synthetic1m: {
      status:
        syntheticSummary?.passed === true &&
        syntheticSummary?.proof?.browserRuntime1m === "proven-synthetic-upload" &&
        numeric(syntheticRow.packedGaussians) >= 1_000_000 &&
        numeric(syntheticAggregate.minApproxFps) >= minSyntheticApproxFps &&
        numeric(syntheticAggregate.maxMeanFrameMs) <= maxSyntheticMeanFrameMs &&
        numeric(syntheticAggregate.maxP95FrameMs) <= maxP95FrameMs &&
        numeric(syntheticAggregate.maxLongFrameRatio) <= maxLongFrameRatio
          ? "passed"
          : "failed",
      mode: syntheticSummary?.mode ?? "",
      proof: syntheticSummary?.proof?.browserRuntime1m ?? "not-proven",
      uploadedGaussians: numeric(syntheticRow.packedGaussians),
      tileReferences: numeric(syntheticRow.tileReferences),
      minApproxFps: numeric(syntheticAggregate.minApproxFps),
      maxMeanFrameMs: numeric(syntheticAggregate.maxMeanFrameMs),
      maxP95FrameMs: numeric(syntheticAggregate.maxP95FrameMs),
      maxLongFrameRatio: numeric(syntheticAggregate.maxLongFrameRatio),
      uploadWallMs: numeric(syntheticAggregate.uploadWallMs),
      isolateUpdateMs: numeric(syntheticAggregate.isolateUpdateMs),
      deleteUpdateMs: numeric(syntheticAggregate.deleteUpdateMs),
      frameCount: syntheticFrameCount,
      screenshotPath: syntheticRow.screenshotPath ?? "",
      summaryPath: syntheticSummaryPath,
      interpretation:
        "Synthetic 1M upload/runtime uses longer rAF sampling through real UI upload and WebGPU Tile object edits.",
    },
  };
}

function buildChecks(evidence) {
  return [
    check("real-scenes-summary", Boolean(evidence.realScenes.mode), "present", Boolean(evidence.realScenes.mode)),
    check("real-scenes-status", evidence.realScenes.status, "passed", evidence.realScenes.status === "passed"),
    check(
      "real-scenes-min-approx-fps",
      evidence.realScenes.minApproxFps,
      `>= ${minRealApproxFps}`,
      evidence.realScenes.minApproxFps >= minRealApproxFps,
    ),
    check(
      "real-scenes-max-mean-frame-ms",
      evidence.realScenes.maxMeanFrameMs,
      `<= ${maxRealMeanFrameMs}`,
      evidence.realScenes.maxMeanFrameMs <= maxRealMeanFrameMs,
    ),
    check(
      "synthetic-1m-summary",
      Boolean(evidence.synthetic1m.mode),
      "present",
      Boolean(evidence.synthetic1m.mode),
    ),
    check(
      "synthetic-1m-proof",
      evidence.synthetic1m.proof,
      "proven-synthetic-upload",
      evidence.synthetic1m.proof === "proven-synthetic-upload",
    ),
    check(
      "synthetic-1m-count",
      evidence.synthetic1m.uploadedGaussians,
      ">= 1000000",
      evidence.synthetic1m.uploadedGaussians >= 1_000_000,
    ),
    check("synthetic-1m-status", evidence.synthetic1m.status, "passed", evidence.synthetic1m.status === "passed"),
    check(
      "synthetic-1m-min-approx-fps",
      evidence.synthetic1m.minApproxFps,
      `>= ${minSyntheticApproxFps}`,
      evidence.synthetic1m.minApproxFps >= minSyntheticApproxFps,
    ),
    check(
      "synthetic-1m-max-mean-frame-ms",
      evidence.synthetic1m.maxMeanFrameMs,
      `<= ${maxSyntheticMeanFrameMs}`,
      evidence.synthetic1m.maxMeanFrameMs <= maxSyntheticMeanFrameMs,
    ),
  ];
}

function writeReport(summary) {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary), "utf8");
}

function renderMarkdown(summary) {
  const failed = (summary.checks ?? []).filter((entry) => !entry.passed);
  const rows = Object.entries(summary.evidence ?? {}).map(([id, evidence]) => ({
    id,
    ...evidence,
  }));
  return [
    "# WebGPU Sustained Frame Pacing Baseline",
    "",
    `- Mode: \`${summary.mode}\``,
    `- Status: \`${summary.status}\``,
    `- FPS baseline: \`${summary.fpsBaseline ?? "not-proven"}\``,
    `- Generated: \`${summary.generatedAt}\``,
    `- Fixed port: \`${summary.port}\``,
    "",
    summary.interpretation ?? "",
    "",
    "## Evidence",
    "",
    "| Evidence | Status | Gaussians | Min approx FPS | Max mean ms | Max P95 ms | Long frame ratio | Summary |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ...rows.map(
      (row) =>
        `| ${escapeMarkdown(row.id)} | ${escapeMarkdown(row.status)} | ${row.largestGaussians || row.uploadedGaussians || 0} | ${row.minApproxFps} | ${row.maxMeanFrameMs} | ${row.maxP95FrameMs} | ${row.maxLongFrameRatio} | ${escapeMarkdown(row.summaryPath)} |`,
    ),
    "",
    "## Checks",
    "",
    failed.length === 0
      ? "All required sustained baseline checks passed."
      : "| Check | Actual | Expected |\n| --- | --- | --- |\n" +
          failed
            .map(
              (entry) =>
                `| ${escapeMarkdown(entry.name)} | ${escapeMarkdown(entry.actual)} | ${escapeMarkdown(entry.expected)} |`,
            )
            .join("\n"),
    "",
    "## Remaining Gaps",
    "",
    "| Gap | Status | Next evidence |",
    "| --- | --- | --- |",
    ...(summary.remainingGaps ?? []).map(
      (gap) =>
        `| ${escapeMarkdown(gap.id)} | ${escapeMarkdown(gap.status)} | ${escapeMarkdown(gap.nextEvidence)} |`,
    ),
    "",
    "## Steps",
    "",
    "| Step | Exit | Duration ms | Command |",
    "| --- | ---: | ---: | --- |",
    ...(summary.steps ?? []).map(
      (step) =>
        `| ${escapeMarkdown(step.label)} | ${step.exitCode ?? ""} | ${step.durationMs ?? 0} | \`${escapeMarkdown(step.command)}\` |`,
    ),
    "",
  ].join("\n");
}

function runProcess(command, commandArgs) {
  return new Promise((resolve) => {
    const child = spawn(command, commandArgs, { stdio: ["ignore", "inherit", "inherit"] });
    child.on("close", (exitCode) => resolve({ exitCode }));
    child.on("error", (error) => {
      console.error(error.message);
      resolve({ exitCode: 1 });
    });
  });
}

function readJsonOrNull(filePath) {
  if (!existsSync(filePath)) return null;
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function check(name, actual, expected, passed) {
  return { name, actual, expected, passed: Boolean(passed) };
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

function positiveInteger(value, fallback) {
  return Math.max(1, Math.round(positiveFiniteNumber(value, fallback)));
}

function numeric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
