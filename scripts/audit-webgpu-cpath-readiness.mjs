import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const DEFAULT_PORT = 5395;
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-webgpu-cpath-readiness";
const DEFAULT_ASSETS = ["nerf-lego-alpha-closure-local", "plush-semantic-closure-local"];
const MODE = "webgpu-cpath-readiness-v1";

const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const port = String(args.port ?? DEFAULT_PORT);
const assets = parseAssets(args.assets ?? args.asset ?? DEFAULT_ASSETS.join(","));
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const skipBuild = flagEnabled(args.skipBuild ?? args["skip-build"]);
const skipRun = flagEnabled(args.skipRun ?? args["skip-run"]);
const allowFailures = flagEnabled(args.allowFailures ?? args["allow-failures"]);
const minLargeSceneGaussians = positiveFiniteNumber(
  args.minLargeSceneGaussians ?? args["min-large-scene-gaussians"],
  250000,
);

const scaleDir = path.join(outputDir, "scale-budget");
const editCostDir = path.join(outputDir, "edit-cost-budget");
const transitionDir = path.join(outputDir, "presentation-transition");
const scaleSummaryPath = String(
  args.scaleSummary ?? args["scale-summary"] ?? path.join(scaleDir, "summary.json"),
);
const editCostSummaryPath = String(
  args.editCostSummary ?? args["edit-cost-summary"] ?? path.join(editCostDir, "summary.json"),
);
const transitionSummaryPath = String(
  args.transitionSummary ??
    args["transition-summary"] ??
    path.join(transitionDir, "summary.json"),
);

if (assets.length === 0) {
  throw new Error("at least one asset is required for WebGPU C-path readiness audit");
}

mkdirSync(outputDir, { recursive: true });
const steps = [];

try {
  if (!skipRun) {
    if (!skipBuild) {
      await runStep("Build viewer", ["npm", "run", "build"]);
    } else if (!existsSync("dist/index.html")) {
      throw new Error("dist/index.html is missing; remove --skip-build or run `npm run build`");
    }
    await runStep("WebGPU scale budget", [
      "npm",
      "run",
      "audit:webgpu-scale-budget",
      "--",
      "--output-dir",
      scaleDir,
    ]);
    await runStep("WebGPU edit cost budget", [
      "npm",
      "run",
      "audit:webgpu-edit-cost-budget",
      "--",
      "--output-dir",
      editCostDir,
    ]);
    await runStep("WebGPU presentation object transition", [
      "npm",
      "run",
      "audit:webgpu-presentation-transition",
      "--",
      "--port",
      port,
      "--assets",
      assets.join(","),
      "--webgpu-flags",
      webGpuFlags,
      "--output-dir",
      transitionDir,
    ]);
  }

  const scaleSummary = readJson(scaleSummaryPath);
  const editCostSummary = readJson(editCostSummaryPath);
  const transitionSummary = readJson(transitionSummaryPath);
  const evidence = buildEvidence({ scaleSummary, editCostSummary, transitionSummary });
  const checks = buildChecks(evidence);
  const gaps = buildGaps(evidence);
  const passed = checks.every((check) => check.passed);
  const summary = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    outputDir,
    port,
    assets,
    webGpuFlags,
    skipBuild,
    skipRun,
    thresholds: {
      minLargeSceneGaussians,
    },
    sourceSummaries: {
      scaleBudget: scaleSummaryPath,
      editCostBudget: editCostSummaryPath,
      presentationTransition: transitionSummaryPath,
    },
    passed,
    status: passed ? "passed" : "failed",
    steps,
    evidence,
    checks,
    gaps,
  };
  writeReport(summary);
  printSummary(summary);
  if (!passed && !allowFailures) {
    process.exitCode = 1;
  }
} catch (error) {
  const summary = {
    mode: MODE,
    generatedAt: new Date().toISOString(),
    outputDir,
    port,
    assets,
    webGpuFlags,
    skipBuild,
    skipRun,
    status: "failed",
    passed: false,
    error: error?.message ?? String(error),
    steps,
    evidence: {},
    checks: [],
    gaps: [
      {
        id: "readiness-audit-incomplete",
        status: "blocked",
        reason: "The readiness audit could not collect all required evidence.",
      },
    ],
  };
  writeReport(summary);
  printSummary(summary);
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
  if (result.exitCode !== 0) {
    const error = new Error(`${label} failed with exit code ${result.exitCode}`);
    error.result = step;
    throw error;
  }
}

function buildEvidence({ scaleSummary, editCostSummary, transitionSummary }) {
  const scale1m = findRow(scaleSummary.rows, "c-path-1m-budget");
  const edit1m = findRow(editCostSummary.rows, "c-path-1m-budget");
  const transitionRows = Array.isArray(transitionSummary.rows) ? transitionSummary.rows : [];
  const transitionCpathRows = transitionRows.filter(
    (row) =>
      row.postDeleteRendererId === "webgpu-tile" &&
      row.postDeleteObjectFilter === "gpu-object-state-buffer",
  );
  const headedLargeRows = transitionRows.filter(
    (row) => numeric(row.packedGaussians) >= minLargeSceneGaussians,
  );
  return {
    scaleBudget1m: {
      status: scaleSummary.status === "passed" && scale1m?.status === "passed" ? "passed" : "failed",
      scope: "synthetic 1M storage budget",
      profile: scale1m?.id ?? "",
      gaussians: numeric(scale1m?.gaussians),
      maxBufferMiB: numeric(scale1m?.maxBufferMiB),
      totalMiB: numeric(scale1m?.totalMiB),
      targetGate: scale1m?.targetGate ?? "",
      objectFilter: scale1m?.objectFilter ?? "",
      sourceMode: scaleSummary.mode ?? "",
      interpretation:
        "Storage layout and desktop-style binding budget fit the 1M C-path profile; this is not browser FPS proof.",
    },
    editCost1m: {
      status: editCostSummary.status === "passed" && edit1m?.status === "passed" ? "passed" : "failed",
      scope: "synthetic 1M object edit cost budget",
      profile: edit1m?.id ?? "",
      gaussians: numeric(edit1m?.gaussians),
      fullUploadMiB: numeric(edit1m?.fullUploadMiB),
      objectStateUpdateKiB: numeric(edit1m?.objectStateUpdateKiB),
      objectStateUploadShare: numeric(edit1m?.objectStateUploadShare),
      pixelCandidateChecksG: numeric(edit1m?.pixelCandidateChecksG),
      editUpdateMode: edit1m?.editUpdateMode ?? "",
      sourceMode: editCostSummary.mode ?? "",
      interpretation:
        "Compatible edits stay on objectState-only small-buffer update shape; this is still a budget envelope, not FPS.",
    },
    headedBrowserTransition: {
      status:
        transitionSummary.passed === true &&
        transitionRows.length === assets.length &&
        transitionCpathRows.length === transitionRows.length &&
        headedLargeRows.length > 0
          ? "passed"
          : "failed",
      scope: "headed browser full-canvas object transition",
      mode: transitionSummary.mode ?? "",
      url: transitionSummary.url ?? "",
      headed: transitionSummary.headed === true,
      scenes: transitionRows.length,
      expectedScenes: assets.length,
      largestGaussians: numeric(transitionSummary.aggregate?.largestGaussians),
      maxTileReferences: numeric(transitionSummary.aggregate?.maxTileReferences),
      maxUpdateMs: numeric(transitionSummary.aggregate?.maxUpdateMs),
      maxQueueDoneMs: numeric(transitionSummary.aggregate?.maxQueueDoneMs),
      cpathRows: transitionCpathRows.length,
      largeRows: headedLargeRows.map((row) => ({
        asset: row.asset,
        gaussians: numeric(row.packedGaussians),
        tileReferences: numeric(row.tileReferences),
        screenshotPath: row.screenshotPath ?? "",
      })),
      interpretation:
        "Real headed browser evidence covers current small and 281k-class scenes staying on WebGPU Tile through select/isolate/delete.",
    },
    browserRuntime1m: {
      status: "not-proven",
      scope: "real 1M headed browser runtime / FPS",
      interpretation:
        "The 1M rows are static budget envelopes. The largest current headed browser transition evidence is the local large scene, not a true 1M interactive FPS run.",
    },
    fpsSla: {
      status: "not-proven",
      scope: "interactive FPS SLA",
      interpretation:
        "Timing smoke records update, submit, and queue-done envelopes; it does not sample frame pacing or sustained FPS.",
    },
  };
}

function buildChecks(evidence) {
  return [
    check(
      "scale-budget-1m-passed",
      evidence.scaleBudget1m.status,
      "passed",
      evidence.scaleBudget1m.status === "passed",
    ),
    check(
      "scale-budget-1m-webgpu-filter",
      evidence.scaleBudget1m.objectFilter,
      "gpu-object-state-buffer",
      evidence.scaleBudget1m.objectFilter === "gpu-object-state-buffer",
    ),
    check(
      "edit-cost-1m-passed",
      evidence.editCost1m.status,
      "passed",
      evidence.editCost1m.status === "passed",
    ),
    check(
      "edit-cost-1m-object-state-only",
      evidence.editCost1m.editUpdateMode,
      "object-state-only",
      evidence.editCost1m.editUpdateMode === "object-state-only",
    ),
    check(
      "headed-transition-passed",
      evidence.headedBrowserTransition.status,
      "passed",
      evidence.headedBrowserTransition.status === "passed",
    ),
    check(
      "headed-transition-large-scene",
      evidence.headedBrowserTransition.largestGaussians,
      `>= ${minLargeSceneGaussians}`,
      evidence.headedBrowserTransition.largestGaussians >= minLargeSceneGaussians,
    ),
    check(
      "headed-transition-cpath-scenes",
      evidence.headedBrowserTransition.cpathRows,
      evidence.headedBrowserTransition.scenes,
      evidence.headedBrowserTransition.scenes > 0 &&
        evidence.headedBrowserTransition.cpathRows === evidence.headedBrowserTransition.scenes,
    ),
  ];
}

function buildGaps(evidence) {
  return [
    {
      id: "browser-runtime-1m",
      status: evidence.browserRuntime1m.status,
      reason: evidence.browserRuntime1m.interpretation,
      nextEvidence:
        "Run a real browser scene near 1M Gaussians with the same select/isolate/delete transition and frame pacing instrumentation.",
    },
    {
      id: "fps-sla",
      status: evidence.fpsSla.status,
      reason: evidence.fpsSla.interpretation,
      nextEvidence:
        "Add sustained frame pacing / FPS sampling for camera idle and object edit transitions.",
    },
  ];
}

function writeReport(summary) {
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary), "utf8");
}

function renderMarkdown(summary) {
  const evidenceRows = Object.entries(summary.evidence ?? {}).map(([id, item]) => ({
    id,
    status: item.status,
    scope: item.scope,
    keyResult: keyResult(id, item),
    interpretation: item.interpretation,
  }));
  const failed = (summary.checks ?? []).filter((entry) => !entry.passed);
  return [
    "# WebGPU C-path Readiness",
    "",
    `- Mode: \`${summary.mode}\``,
    `- Status: \`${summary.status}\``,
    `- Generated: \`${summary.generatedAt}\``,
    `- Fixed port: \`${summary.port}\``,
    "",
    "This report combines synthetic 1M C-path budgets with the current headed browser object-transition evidence. Passing means the architecture evidence is internally consistent; it does not mean 1M browser FPS is proven.",
    "",
    "## Evidence",
    "",
    "| Evidence | Status | Scope | Key result | Interpretation |",
    "| --- | --- | --- | --- | --- |",
    ...evidenceRows.map(
      (row) =>
        `| ${escapeMarkdown(row.id)} | ${escapeMarkdown(row.status)} | ${escapeMarkdown(row.scope)} | ${escapeMarkdown(row.keyResult)} | ${escapeMarkdown(row.interpretation)} |`,
    ),
    "",
    "## Checks",
    "",
    failed.length === 0
      ? "All required readiness checks passed."
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
    ...(summary.gaps ?? []).map(
      (gap) =>
        `| ${escapeMarkdown(gap.id)} | ${escapeMarkdown(gap.status)} | ${escapeMarkdown(gap.nextEvidence ?? gap.reason)} |`,
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

function keyResult(id, item) {
  if (id === "scaleBudget1m") {
    return `${item.gaussians} Gaussians, max ${item.maxBufferMiB} MiB, total ${item.totalMiB} MiB`;
  }
  if (id === "editCost1m") {
    return `${item.objectStateUpdateKiB} KiB edit update, ${item.fullUploadMiB} MiB full upload, ${item.pixelCandidateChecksG}G candidates`;
  }
  if (id === "headedBrowserTransition") {
    return `${item.scenes}/${item.expectedScenes} scenes, largest ${item.largestGaussians}, max queue ${item.maxQueueDoneMs} ms`;
  }
  return item.status;
}

function printSummary(summary) {
  console.log(
    [
      `webgpu_cpath_readiness=${summary.status}`,
      `scale1m=${summary.evidence?.scaleBudget1m?.status ?? "missing"}`,
      `edit1m=${summary.evidence?.editCost1m?.status ?? "missing"}`,
      `headedTransition=${summary.evidence?.headedBrowserTransition?.status ?? "missing"}`,
      `largestHeadedGaussians=${summary.evidence?.headedBrowserTransition?.largestGaussians ?? 0}`,
      `browserRuntime1m=${summary.evidence?.browserRuntime1m?.status ?? "not-proven"}`,
      `fpsSla=${summary.evidence?.fpsSla?.status ?? "not-proven"}`,
      `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
    ].join(" "),
  );
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

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function findRow(rows, id) {
  return Array.isArray(rows) ? rows.find((row) => row.id === id) : null;
}

function check(name, actual, expected, passed) {
  return { name, actual, expected, passed: Boolean(passed) };
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

function numeric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
