import { spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";

const DEFAULT_PORT = 5395;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-webgpu-cpath-production-sla";
const DEFAULT_ASSETS = ["nerf-lego-alpha-closure-local", "plush-semantic-closure-local"];
const MODE = "webgpu-cpath-production-sla-v1";
const MIN_PRODUCTION_TRAINED_GAUSSIANS = 1_000_000;

const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const readinessDir = path.join(outputDir, "readiness");
const port = String(args.port ?? DEFAULT_PORT);
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const assets = parseAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const trainedPlyPath = optionalString(
  args.trainedPly ?? args["trained-ply"] ?? args.inputPly ?? args["input-ply"],
);
const targetHardware = optionalString(
  args.fpsSlaTargetHardware ?? args["fps-sla-target-hardware"] ?? args.targetHardware ?? args["target-hardware"],
);
const trainedMinGaussians = Math.max(
  MIN_PRODUCTION_TRAINED_GAUSSIANS,
  positiveInteger(
    args.trainedMinGaussians ??
      args["trained-min-gaussians"] ??
      args.minGaussians ??
      args["min-gaussians"] ??
      args.fpsSlaMinTrainedGaussians ??
      args["fps-sla-min-trained-gaussians"],
    MIN_PRODUCTION_TRAINED_GAUSSIANS,
  ),
);
const minTrainedApproxFps = positiveFiniteNumber(
  args.fpsSlaMinTrainedApproxFps ??
    args["fps-sla-min-trained-approx-fps"] ??
    args.minTrainedApproxFps ??
    args["min-trained-approx-fps"],
  24,
);
const sustainedFrameCount = positiveInteger(
  args.sustainedFrameCount ?? args["sustained-frame-count"] ?? args.frameCount ?? args["frame-count"],
  120,
);
const sustainedMinRealApproxFps = positiveFiniteNumber(
  args.sustainedMinRealApproxFps ?? args["sustained-min-real-approx-fps"],
  10,
);
const sustainedMinSyntheticApproxFps = positiveFiniteNumber(
  args.sustainedMinSyntheticApproxFps ?? args["sustained-min-synthetic-approx-fps"],
  8,
);
const sustainedMaxRealMeanFrameMs = positiveFiniteNumber(
  args.sustainedMaxRealMeanFrameMs ?? args["sustained-max-real-mean-frame-ms"],
  120,
);
const sustainedMaxSyntheticMeanFrameMs = positiveFiniteNumber(
  args.sustainedMaxSyntheticMeanFrameMs ?? args["sustained-max-synthetic-mean-frame-ms"],
  150,
);
const sustainedMaxP95FrameMs = positiveFiniteNumber(
  args.sustainedMaxP95FrameMs ?? args["sustained-max-p95-frame-ms"],
  220,
);
const sustainedMaxLongFrameRatio = positiveFiniteNumber(
  args.sustainedMaxLongFrameRatio ?? args["sustained-max-long-frame-ratio"],
  0.25,
);
const dryRun = flagEnabled(args.dryRun ?? args["dry-run"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);

const forbiddenFlags = [
  "skip-run",
  "skipRun",
  "skip-synthetic-1m-runtime",
  "skipSynthetic1mRuntime",
  "scale-summary",
  "scaleSummary",
  "edit-cost-summary",
  "editCostSummary",
  "transition-summary",
  "transitionSummary",
  "synthetic-runtime-summary",
  "syntheticRuntimeSummary",
  "trained-ply-runtime-summary",
  "trainedPlyRuntimeSummary",
  "sustained-frame-pacing-summary",
  "sustainedFramePacingSummary",
];
const usedForbiddenFlags = forbiddenFlags.filter((key) =>
  Object.prototype.hasOwnProperty.call(args, key),
);

mkdirSync(outputDir, { recursive: true });

const preflight = buildPreflight();
const readinessCommand = buildReadinessCommand();
let readinessResult = null;
let readinessSummary = null;

if (preflight.passed && !dryRun) {
  readinessResult = await runProcess(readinessCommand[0], readinessCommand.slice(1));
  const readinessSummaryPath = path.join(readinessDir, "summary.json");
  if (existsSync(readinessSummaryPath)) {
    readinessSummary = readJson(readinessSummaryPath);
  }
}

const checks = buildChecks({ preflight, readinessResult, readinessSummary });
const passed = checks.every((entry) => entry.passed);
const status = dryRun && preflight.passed ? "dry-run" : passed ? "passed" : "failed";
const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  port,
  webGpuFlags,
  assets,
  dryRun,
  thresholds: {
    trainedMinGaussians,
    minTrainedApproxFps,
    sustainedFrameCount,
    sustainedMinRealApproxFps,
    sustainedMinSyntheticApproxFps,
    sustainedMaxRealMeanFrameMs,
    sustainedMaxSyntheticMeanFrameMs,
    sustainedMaxP95FrameMs,
    sustainedMaxLongFrameRatio,
  },
  input: {
    trainedPly: trainedPlyPath || "not-provided",
    targetHardware: targetHardware || "not-provided",
  },
  preflight,
  readinessCommand: readinessCommand.join(" "),
  readinessExitCode: readinessResult?.exitCode ?? "not-run",
  readinessSummaryPath: path.join(readinessDir, "summary.json"),
  readiness: summarizeReadiness(readinessSummary),
  passed,
  status,
  checks,
  remainingGaps: buildRemainingGaps({ preflight, readinessSummary }),
  interpretation:
    "This strict production SLA gate is the terminal WebGPU C-path proof wrapper. It requires a real trained near-1M object-aware PLY, a reviewed target hardware label, real browser runtime proof, sustained trained PLY evidence, and reviewed FPS threshold pass. Synthetic 1M and reused summaries are intentionally not accepted as substitutes.",
};

writeReport(summary);
console.log(
  `webgpu_cpath_production_sla=${summary.status} ` +
    `preflight=${preflight.passed ? "passed" : "failed"} ` +
    `trainedGaussians=${preflight.trainedPlyGaussians} ` +
    `target=${JSON.stringify(summary.input.targetHardware)} ` +
    `readiness=${summary.readiness.status} ` +
    `fpsSla=${summary.readiness.fpsSla} ` +
    `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
);

if (!passed && !dryRun && !allowFailures) {
  process.exitCode = 1;
}
if (!preflight.passed && !allowFailures) {
  process.exitCode = 1;
}

function buildPreflight() {
  const checks = [
    check("no-summary-shortcuts", usedForbiddenFlags.join(",") || "none", "none", usedForbiddenFlags.length === 0),
    check("trained-ply-provided", trainedPlyPath || "not-provided", "path", Boolean(trainedPlyPath)),
    check("target-hardware-provided", targetHardware || "not-provided", "non-empty", Boolean(targetHardware)),
  ];
  let trainedPlyGaussians = 0;
  let trainedPlyError = "";
  if (trainedPlyPath) {
    if (!existsSync(trainedPlyPath)) {
      trainedPlyError = "trained PLY path does not exist";
    } else {
      try {
        trainedPlyGaussians = readPlyVertexCount(trainedPlyPath);
      } catch (error) {
        trainedPlyError = error?.message ?? String(error);
      }
    }
  }
  checks.push(
    check(
      "trained-ply-gaussians",
      trainedPlyGaussians,
      `>= ${trainedMinGaussians}`,
      trainedPlyGaussians >= trainedMinGaussians,
    ),
  );
  if (trainedPlyError) {
    checks.push(check("trained-ply-readable", trainedPlyError, "readable PLY header", false));
  }
  return {
    trainedPlyGaussians,
    trainedPlyError,
    forbiddenFlags: usedForbiddenFlags,
    passed: checks.every((entry) => entry.passed),
    checks,
  };
}

function buildReadinessCommand() {
  return [
    "npm",
    "run",
    "audit:webgpu-cpath-readiness",
    "--",
    "--trained-ply",
    trainedPlyPath || "__missing_trained_ply__",
    "--trained-min-gaussians",
    String(trainedMinGaussians),
    "--include-sustained-frame-pacing",
    "--fps-sla-reviewed",
    "--fps-sla-target-hardware",
    targetHardware || "__missing_target_hardware__",
    "--fps-sla-min-trained-gaussians",
    String(trainedMinGaussians),
    "--fps-sla-min-trained-approx-fps",
    String(minTrainedApproxFps),
    "--sustained-frame-count",
    String(sustainedFrameCount),
    "--sustained-min-real-approx-fps",
    String(sustainedMinRealApproxFps),
    "--sustained-min-synthetic-approx-fps",
    String(sustainedMinSyntheticApproxFps),
    "--sustained-max-real-mean-frame-ms",
    String(sustainedMaxRealMeanFrameMs),
    "--sustained-max-synthetic-mean-frame-ms",
    String(sustainedMaxSyntheticMeanFrameMs),
    "--sustained-max-p95-frame-ms",
    String(sustainedMaxP95FrameMs),
    "--sustained-max-long-frame-ratio",
    String(sustainedMaxLongFrameRatio),
    "--assets",
    assets.join(","),
    "--port",
    port,
    "--webgpu-flags",
    webGpuFlags,
    "--output-dir",
    readinessDir,
    ...(allowFailures ? ["--allow-failures"] : []),
  ];
}

function buildChecks({ preflight, readinessResult, readinessSummary }) {
  const readinessEvidence = readinessSummary?.evidence ?? {};
  return [
    ...preflight.checks,
    check("dry-run", dryRun, false, dryRun === false),
    check("readiness-exit-code", readinessResult?.exitCode ?? "not-run", 0, readinessResult?.exitCode === 0),
    check("readiness-status", readinessSummary?.status ?? "not-run", "passed", readinessSummary?.status === "passed"),
    check(
      "trained-ply-runtime",
      readinessEvidence.trainedPlyRuntime?.status ?? "not-run",
      "passed",
      readinessEvidence.trainedPlyRuntime?.status === "passed",
    ),
    check(
      "real-trained-browser-runtime-1m",
      readinessEvidence.realTrainedBrowserRuntime1m?.status ?? "not-run",
      "passed",
      readinessEvidence.realTrainedBrowserRuntime1m?.status === "passed",
    ),
    check(
      "sustained-frame-pacing",
      readinessEvidence.sustainedFramePacing?.status ?? "not-run",
      "passed",
      readinessEvidence.sustainedFramePacing?.status === "passed",
    ),
    check(
      "sustained-trained-ply",
      readinessEvidence.sustainedFramePacing?.trainedPlyStatus ?? "not-run",
      "passed",
      readinessEvidence.sustainedFramePacing?.trainedPlyStatus === "passed",
    ),
    check(
      "reviewed-fps-sla",
      readinessEvidence.fpsSla?.status ?? "not-run",
      "passed",
      readinessEvidence.fpsSla?.status === "passed",
    ),
  ];
}

function summarizeReadiness(readinessSummary) {
  const evidence = readinessSummary?.evidence ?? {};
  return {
    status: readinessSummary?.status ?? "not-run",
    trainedPlyRuntime: evidence.trainedPlyRuntime?.status ?? "not-run",
    trainedPlyGaussians: evidence.trainedPlyRuntime?.gaussians ?? 0,
    realTrainedBrowserRuntime1m: evidence.realTrainedBrowserRuntime1m?.status ?? "not-run",
    sustainedFramePacing: evidence.sustainedFramePacing?.status ?? "not-run",
    sustainedTrainedPly: evidence.sustainedFramePacing?.trainedPlyStatus ?? "not-run",
    sustainedTrainedMinApproxFps: evidence.sustainedFramePacing?.trainedPlyMinApproxFps ?? 0,
    fpsSla: evidence.fpsSla?.status ?? "not-run",
    fpsSlaBlockers: evidence.fpsSla?.blockers ?? [],
  };
}

function buildRemainingGaps({ preflight, readinessSummary }) {
  const readiness = summarizeReadiness(readinessSummary);
  const gaps = [];
  if (!preflight.passed) {
    gaps.push({
      id: "production-sla-preflight",
      status: "blocked",
      nextEvidence:
        "Provide a real trained object-aware PLY with at least 1,000,000 Gaussians and a target hardware label; do not use summary shortcuts.",
    });
  }
  if (dryRun) {
    gaps.push({
      id: "production-sla-browser-run",
      status: "not-run",
      nextEvidence:
        "Run without --dry-run to collect the real browser C-path readiness, sustained trained PLY row, and reviewed FPS SLA evidence.",
    });
  }
  if (readiness.fpsSla !== "passed") {
    gaps.push({
      id: "production-fps-sla",
      status: readiness.fpsSla,
      blockers: readiness.fpsSlaBlockers,
      nextEvidence:
        "Run the strict gate with a real trained near-1M PLY on the reviewed target hardware until fpsSla passes.",
    });
  }
  return gaps;
}

function writeReport(summary) {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary));
}

function renderMarkdown(summary) {
  const lines = [
    "# WebGPU C-Path Production SLA Gate",
    "",
    `- Status: ${summary.status}`,
    `- Mode: ${summary.mode}`,
    `- Trained PLY: ${summary.input.trainedPly}`,
    `- Target hardware: ${summary.input.targetHardware}`,
    `- Trained Gaussians: ${summary.preflight.trainedPlyGaussians}`,
    `- Min trained Gaussians: ${summary.thresholds.trainedMinGaussians}`,
    `- Min trained approx FPS: ${summary.thresholds.minTrainedApproxFps}`,
    `- Dry run: ${summary.dryRun}`,
    "",
    "## Checks",
    "",
    "| Check | Actual | Expected | Status |",
    "| --- | ---: | ---: | --- |",
    ...summary.checks.map(
      (entry) =>
        `| ${escapeMarkdown(entry.id)} | ${escapeMarkdown(entry.actual)} | ${escapeMarkdown(entry.expected)} | ${
          entry.passed ? "passed" : "failed"
        } |`,
    ),
    "",
    "## Readiness",
    "",
    `- Status: ${summary.readiness.status}`,
    `- Trained PLY runtime: ${summary.readiness.trainedPlyRuntime}`,
    `- Real trained browser runtime 1M: ${summary.readiness.realTrainedBrowserRuntime1m}`,
    `- Sustained frame pacing: ${summary.readiness.sustainedFramePacing}`,
    `- Sustained trained PLY: ${summary.readiness.sustainedTrainedPly}`,
    `- Sustained trained min approx FPS: ${summary.readiness.sustainedTrainedMinApproxFps}`,
    `- FPS SLA: ${summary.readiness.fpsSla}`,
    "",
    "## Command",
    "",
    "```bash",
    summary.readinessCommand,
    "```",
    "",
  ];
  if (summary.remainingGaps.length > 0) {
    lines.push("## Remaining Gaps", "");
    for (const gap of summary.remainingGaps) {
      lines.push(`- ${gap.id}: ${gap.nextEvidence}`);
    }
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
}

function readPlyVertexCount(filePath) {
  const stats = statSync(filePath);
  if (!stats.isFile()) {
    throw new Error("trained PLY path is not a file");
  }
  const fd = openSync(filePath, "r");
  try {
    const buffer = Buffer.alloc(Math.min(131072, stats.size));
    const bytesRead = readSync(fd, buffer, 0, buffer.length, 0);
    const header = buffer.subarray(0, bytesRead).toString("utf8");
    if (!header.includes("end_header")) {
      throw new Error("PLY header was not found in the first 128 KiB");
    }
    const match = header.match(/^element\s+vertex\s+(\d+)\s*$/m);
    if (!match) {
      throw new Error("PLY vertex count is missing");
    }
    return Number(match[1]);
  } finally {
    closeSync(fd);
  }
}

async function runProcess(command, argsForCommand) {
  return new Promise((resolve) => {
    const child = spawn(command, argsForCommand, {
      stdio: "inherit",
      env: process.env,
    });
    child.on("close", (exitCode) => resolve({ exitCode: exitCode ?? 1 }));
    child.on("error", () => resolve({ exitCode: 1 }));
  });
}

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function check(id, actual, expected, passed) {
  return { id, actual, expected, passed: Boolean(passed) };
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

function positiveInteger(value, fallback) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function positiveFiniteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
