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
const syntheticRuntimeDir = path.join(outputDir, "synthetic-1m-runtime");
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
const syntheticRuntimeSummaryPath = String(
  args.syntheticRuntimeSummary ??
    args["synthetic-runtime-summary"] ??
    path.join(syntheticRuntimeDir, "summary.json"),
);
const skipSynthetic1mRuntime = flagEnabled(
  args.skipSynthetic1mRuntime ?? args["skip-synthetic-1m-runtime"],
);
const trainedPlyPath = optionalString(
  args.trainedPly ?? args["trained-ply"] ?? args.inputPly ?? args["input-ply"],
);
const trainedPlyRuntimeEnabled = Boolean(trainedPlyPath);
const trainedPlyMinGaussians = positiveFiniteNumber(
  args.trainedMinGaussians ?? args["trained-min-gaussians"] ?? args.minGaussians ?? args["min-gaussians"],
  1_000_000,
);
const trainedPlyRuntimeDir = path.join(outputDir, "trained-ply-runtime");
const trainedPlyRuntimeSummaryPath = String(
  args.trainedPlyRuntimeSummary ??
    args["trained-ply-runtime-summary"] ??
    path.join(trainedPlyRuntimeDir, "summary.json"),
);
const sustainedFramePacingDir = path.join(outputDir, "sustained-frame-pacing");
const sustainedFramePacingSummaryPath = String(
  args.sustainedFramePacingSummary ??
    args["sustained-frame-pacing-summary"] ??
    path.join(sustainedFramePacingDir, "summary.json"),
);
const includeSustainedFramePacing =
  flagEnabled(args.includeSustainedFramePacing ?? args["include-sustained-frame-pacing"]) ||
  Boolean(optionalString(args.sustainedFramePacingSummary ?? args["sustained-frame-pacing-summary"]));
const sustainedFrameCount = positiveFiniteNumber(
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
    if (!skipSynthetic1mRuntime) {
      await runStep("WebGPU synthetic 1M browser runtime", [
        "npm",
        "run",
        "audit:webgpu-synthetic-1m-runtime",
        "--",
        "--port",
        port,
        "--webgpu-flags",
        webGpuFlags,
        "--output-dir",
        syntheticRuntimeDir,
      ]);
    }
    if (trainedPlyRuntimeEnabled) {
      await runStep("WebGPU trained PLY browser runtime", [
        "npm",
        "run",
        "audit:webgpu-ply-runtime",
        "--",
        "--input-ply",
        trainedPlyPath,
        "--scene-kind",
        "trained",
        "--min-gaussians",
        String(Math.round(trainedPlyMinGaussians)),
        "--port",
        port,
        "--webgpu-flags",
        webGpuFlags,
        "--output-dir",
        trainedPlyRuntimeDir,
      ]);
    }
    if (includeSustainedFramePacing) {
      await runStep("WebGPU sustained frame pacing baseline", [
        "npm",
        "run",
        "audit:webgpu-sustained-frame-pacing",
        "--",
        "--port",
        port,
        "--webgpu-flags",
        webGpuFlags,
        "--output-dir",
        sustainedFramePacingDir,
        "--skip-build",
        "--frame-count",
        String(Math.round(sustainedFrameCount)),
        "--min-real-approx-fps",
        String(sustainedMinRealApproxFps),
        "--min-synthetic-approx-fps",
        String(sustainedMinSyntheticApproxFps),
        "--max-real-mean-frame-ms",
        String(sustainedMaxRealMeanFrameMs),
        "--max-synthetic-mean-frame-ms",
        String(sustainedMaxSyntheticMeanFrameMs),
        "--max-p95-frame-ms",
        String(sustainedMaxP95FrameMs),
        "--max-long-frame-ratio",
        String(sustainedMaxLongFrameRatio),
        ...(trainedPlyRuntimeEnabled
          ? [
              "--trained-ply",
              trainedPlyPath,
              "--trained-min-gaussians",
              String(Math.round(trainedPlyMinGaussians)),
            ]
          : []),
      ]);
    }
  }

  const scaleSummary = readJson(scaleSummaryPath);
  const editCostSummary = readJson(editCostSummaryPath);
  const transitionSummary = readJson(transitionSummaryPath);
  const syntheticRuntimeSummary = skipSynthetic1mRuntime
    ? null
    : readJson(syntheticRuntimeSummaryPath);
  const trainedPlyRuntimeSummary = trainedPlyRuntimeEnabled
    ? readJson(trainedPlyRuntimeSummaryPath)
    : null;
  const sustainedFramePacingSummary = includeSustainedFramePacing
    ? readJson(sustainedFramePacingSummaryPath)
    : null;
  const evidence = buildEvidence({
    scaleSummary,
    editCostSummary,
    transitionSummary,
    syntheticRuntimeSummary,
    trainedPlyRuntimeSummary,
    sustainedFramePacingSummary,
  });
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
      trainedPlyMinGaussians,
      sustainedFrameCount,
      sustainedMinRealApproxFps,
      sustainedMinSyntheticApproxFps,
      sustainedMaxRealMeanFrameMs,
      sustainedMaxSyntheticMeanFrameMs,
      sustainedMaxP95FrameMs,
      sustainedMaxLongFrameRatio,
    },
    sourceSummaries: {
      scaleBudget: scaleSummaryPath,
      editCostBudget: editCostSummaryPath,
      presentationTransition: transitionSummaryPath,
      syntheticRuntime1m: skipSynthetic1mRuntime ? "skipped" : syntheticRuntimeSummaryPath,
      trainedPlyRuntime: trainedPlyRuntimeEnabled ? trainedPlyRuntimeSummaryPath : "not-provided",
      sustainedFramePacing: includeSustainedFramePacing ? sustainedFramePacingSummaryPath : "not-provided",
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

function buildEvidence({
  scaleSummary,
  editCostSummary,
  transitionSummary,
  syntheticRuntimeSummary,
  trainedPlyRuntimeSummary,
  sustainedFramePacingSummary,
}) {
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
  const syntheticRuntimeRow = syntheticRuntimeSummary?.row ?? {};
  const syntheticRuntimeProof = syntheticRuntimeSummary?.proof?.browserRuntime1m ?? "not-run";
  const trainedPlyRow = trainedPlyRuntimeSummary?.row ?? {};
  const trainedPlyProof = trainedPlyRuntimeSummary?.proof?.plyRuntime ?? "not-run";
  const trainedPlySceneProof = trainedPlyRuntimeSummary?.proof?.realTrainedScene1m ?? "not-proven";
  const trainedPlyGaussians = numeric(trainedPlyRow.packedGaussians);
  const sustainedEvidence = sustainedFramePacingSummary?.evidence ?? {};
  const sustainedReal = sustainedEvidence.realScenes ?? {};
  const sustainedSynthetic = sustainedEvidence.synthetic1m ?? {};
  const sustainedTrained = sustainedEvidence.trainedPly ?? {};
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
      status:
        syntheticRuntimeSummary?.passed === true &&
        syntheticRuntimeProof === "proven-synthetic-upload" &&
        numeric(syntheticRuntimeRow.packedGaussians) >= 1_000_000
          ? "passed"
          : skipSynthetic1mRuntime
            ? "skipped"
            : "failed",
      scope: "synthetic 1M headed browser upload/runtime",
      mode: syntheticRuntimeSummary?.mode ?? "",
      proof: syntheticRuntimeProof,
      gaussians: numeric(syntheticRuntimeRow.packedGaussians),
      tileReferences: numeric(syntheticRuntimeRow.tileReferences),
      minApproxFps: numeric(syntheticRuntimeSummary?.aggregate?.minApproxFps),
      uploadWallMs: numeric(syntheticRuntimeSummary?.aggregate?.uploadWallMs),
      isolateUpdateMs: numeric(syntheticRuntimeSummary?.aggregate?.isolateUpdateMs),
      deleteUpdateMs: numeric(syntheticRuntimeSummary?.aggregate?.deleteUpdateMs),
      screenshotPath: syntheticRuntimeRow.screenshotPath ?? "",
      interpretation: skipSynthetic1mRuntime
        ? "Synthetic 1M browser runtime was skipped in this readiness run."
        : "A synthetic 1M binary PLY is uploaded through the real UI and exercised through WebGPU Tile select/isolate/delete. This proves browser runtime shape for synthetic 1M, not trained-scene quality.",
    },
    trainedPlyRuntime: {
      status:
        !trainedPlyRuntimeEnabled
          ? "not-provided"
          : trainedPlyRuntimeSummary?.passed === true &&
              trainedPlyProof === "proven-ply-upload" &&
              trainedPlyGaussians >= trainedPlyMinGaussians
            ? "passed"
            : "failed",
      scope: "real/trained PLY headed browser upload/runtime",
      mode: trainedPlyRuntimeSummary?.mode ?? "",
      proof: trainedPlyProof,
      sceneProof: trainedPlySceneProof,
      gaussians: trainedPlyGaussians,
      minGaussians: trainedPlyMinGaussians,
      tileReferences: numeric(trainedPlyRow.tileReferences),
      minApproxFps: numeric(trainedPlyRuntimeSummary?.aggregate?.minApproxFps),
      uploadWallMs: numeric(trainedPlyRuntimeSummary?.aggregate?.uploadWallMs),
      isolateUpdateMs: numeric(trainedPlyRuntimeSummary?.aggregate?.isolateUpdateMs),
      deleteUpdateMs: numeric(trainedPlyRuntimeSummary?.aggregate?.deleteUpdateMs),
      screenshotPath: trainedPlyRow.screenshotPath ?? "",
      interpretation: trainedPlyRuntimeEnabled
        ? "A caller-provided trained/object-aware PLY was uploaded through the real UI and exercised through WebGPU Tile select/isolate/delete."
        : "No trained PLY was provided for this readiness run; use --trained-ply with --trained-min-gaussians to collect this evidence.",
    },
    sustainedFramePacing: {
      status:
        !includeSustainedFramePacing
          ? "not-provided"
          : sustainedFramePacingSummary?.passed === true &&
              sustainedFramePacingSummary?.fpsBaseline === "baseline-passed" &&
              sustainedReal.status === "passed" &&
              sustainedSynthetic.status === "passed"
            ? "passed"
            : "failed",
      scope: "current real scenes plus synthetic 1M sustained rAF baseline",
      mode: sustainedFramePacingSummary?.mode ?? "",
      fpsBaseline: sustainedFramePacingSummary?.fpsBaseline ?? "not-proven",
      realLargestGaussians: numeric(sustainedReal.largestGaussians),
      realMinApproxFps: numeric(sustainedReal.minApproxFps),
      realMaxMeanFrameMs: numeric(sustainedReal.maxMeanFrameMs),
      realMaxP95FrameMs: numeric(sustainedReal.maxP95FrameMs),
      realMaxLongFrameRatio: numeric(sustainedReal.maxLongFrameRatio),
      syntheticGaussians: numeric(sustainedSynthetic.uploadedGaussians),
      syntheticMinApproxFps: numeric(sustainedSynthetic.minApproxFps),
      syntheticMaxMeanFrameMs: numeric(sustainedSynthetic.maxMeanFrameMs),
      syntheticMaxP95FrameMs: numeric(sustainedSynthetic.maxP95FrameMs),
      syntheticMaxLongFrameRatio: numeric(sustainedSynthetic.maxLongFrameRatio),
      trainedPlyStatus: sustainedTrained.status ?? "not-provided",
      trainedPlyGaussians: numeric(sustainedTrained.uploadedGaussians),
      trainedPlyMinApproxFps: numeric(sustainedTrained.minApproxFps),
      trainedPlyMaxMeanFrameMs: numeric(sustainedTrained.maxMeanFrameMs),
      trainedPlyMaxP95FrameMs: numeric(sustainedTrained.maxP95FrameMs),
      trainedPlyMaxLongFrameRatio: numeric(sustainedTrained.maxLongFrameRatio),
      trainedPlyProof: sustainedTrained.proof ?? "not-proven",
      trainedPlySceneProof: sustainedTrained.sceneProof ?? "not-proven",
      interpretation: includeSustainedFramePacing
        ? "Longer rAF sampling baseline was collected for current real scenes, synthetic 1M upload/runtime, and optional trained PLY upload/runtime. This is baseline evidence, not production FPS SLA."
        : "No sustained frame-pacing baseline was included in this readiness run; use --include-sustained-frame-pacing to collect it.",
    },
    realTrainedBrowserRuntime1m: {
      status:
        trainedPlySceneProof === "proven-trained-ply-upload" && trainedPlyGaussians >= 1_000_000
          ? "passed"
          : "not-proven",
      scope: "real trained 1M headed browser runtime",
      interpretation:
        trainedPlyGaussians > 0
          ? `Current trained PLY runtime evidence covers ${trainedPlyGaussians} Gaussians; near-1M trained scene proof still requires >= 1000000.`
          : "The largest current real headed transition evidence is still the local large scene, not a trained scene near 1M Gaussians.",
    },
    fpsSla: {
      status: "not-proven",
      scope: "interactive FPS SLA",
      interpretation:
        evidenceStatusText(includeSustainedFramePacing, sustainedFramePacingSummary?.fpsBaseline) +
        " Production FPS SLA still requires reviewed thresholds on target hardware and real trained 1M scenes.",
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
    check(
      "synthetic-browser-runtime-1m",
      evidence.browserRuntime1m.status,
      skipSynthetic1mRuntime ? "skipped" : "passed",
      skipSynthetic1mRuntime
        ? evidence.browserRuntime1m.status === "skipped"
        : evidence.browserRuntime1m.status === "passed",
    ),
    ...(trainedPlyRuntimeEnabled
      ? [
          check(
            "trained-ply-runtime",
            evidence.trainedPlyRuntime.status,
            "passed",
            evidence.trainedPlyRuntime.status === "passed",
          ),
          check(
            "trained-ply-runtime-min-gaussians",
            evidence.trainedPlyRuntime.gaussians,
            `>= ${trainedPlyMinGaussians}`,
            evidence.trainedPlyRuntime.gaussians >= trainedPlyMinGaussians,
          ),
        ]
      : []),
    ...(includeSustainedFramePacing
      ? [
          check(
            "sustained-frame-pacing-baseline",
            evidence.sustainedFramePacing.status,
            "passed",
            evidence.sustainedFramePacing.status === "passed",
          ),
          check(
            "sustained-real-scenes-min-fps",
            evidence.sustainedFramePacing.realMinApproxFps,
            `>= ${sustainedMinRealApproxFps}`,
            evidence.sustainedFramePacing.realMinApproxFps >= sustainedMinRealApproxFps,
          ),
          check(
            "sustained-synthetic-1m-min-fps",
            evidence.sustainedFramePacing.syntheticMinApproxFps,
            `>= ${sustainedMinSyntheticApproxFps}`,
            evidence.sustainedFramePacing.syntheticMinApproxFps >= sustainedMinSyntheticApproxFps,
          ),
          ...(trainedPlyRuntimeEnabled
            ? [
                check(
                  "sustained-trained-ply",
                  evidence.sustainedFramePacing.trainedPlyStatus,
                  "passed",
                  evidence.sustainedFramePacing.trainedPlyStatus === "passed",
                ),
              ]
            : []),
        ]
      : []),
  ];
}

function buildGaps(evidence) {
  return [
    {
      id: "real-trained-browser-runtime-1m",
      status: evidence.realTrainedBrowserRuntime1m.status,
      reason: evidence.realTrainedBrowserRuntime1m.interpretation,
      nextEvidence:
        "Run a trained or captured real scene near 1M Gaussians with the same select/isolate/delete transition and frame pacing instrumentation.",
    },
    {
      id: "fps-sla",
      status: evidence.fpsSla.status,
      reason: evidence.fpsSla.interpretation,
      nextEvidence:
        includeSustainedFramePacing
          ? "Promote the collected sustained baseline into a production SLA only after threshold review on target hardware and real trained 1M scenes."
          : "Include the sustained frame-pacing baseline, then promote it into a production SLA only after threshold review on target hardware and real trained 1M scenes.",
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
    "This report combines synthetic 1M C-path budgets, synthetic 1M browser upload/runtime proof, current headed real-scene browser object-transition evidence, optional trained PLY runtime evidence, and optional sustained frame-pacing baseline evidence. Passing means the architecture evidence is internally consistent; it does not mean trained 1M scene quality or production FPS SLA is proven unless the trained PLY evidence explicitly reaches 1M and FPS thresholds have been reviewed on target hardware.",
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
  if (id === "browserRuntime1m") {
    return `${item.gaussians} uploaded Gaussians, ${item.tileReferences} tile refs, min approx FPS ${item.minApproxFps}`;
  }
  if (id === "trainedPlyRuntime") {
    return `${item.gaussians} uploaded Gaussians, min required ${item.minGaussians}, proof ${item.proof}`;
  }
  if (id === "sustainedFramePacing") {
    const trained =
      item.trainedPlyStatus && item.trainedPlyStatus !== "not-provided"
        ? `, trained min FPS ${item.trainedPlyMinApproxFps}`
        : "";
    return `real min FPS ${item.realMinApproxFps}, synthetic min FPS ${item.syntheticMinApproxFps}${trained}, baseline ${item.fpsBaseline}`;
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
      `browserRuntime1mProof=${summary.evidence?.browserRuntime1m?.proof ?? "not-proven"}`,
      `trainedPlyRuntime=${summary.evidence?.trainedPlyRuntime?.status ?? "not-provided"}`,
      `trainedPlyGaussians=${summary.evidence?.trainedPlyRuntime?.gaussians ?? 0}`,
      `sustainedFramePacing=${summary.evidence?.sustainedFramePacing?.status ?? "not-provided"}`,
      `sustainedRealMinApproxFps=${summary.evidence?.sustainedFramePacing?.realMinApproxFps ?? 0}`,
      `sustainedSyntheticMinApproxFps=${summary.evidence?.sustainedFramePacing?.syntheticMinApproxFps ?? 0}`,
      `sustainedTrainedPly=${summary.evidence?.sustainedFramePacing?.trainedPlyStatus ?? "not-provided"}`,
      `sustainedTrainedMinApproxFps=${summary.evidence?.sustainedFramePacing?.trainedPlyMinApproxFps ?? 0}`,
      `realTrainedBrowserRuntime1m=${summary.evidence?.realTrainedBrowserRuntime1m?.status ?? "not-proven"}`,
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

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return "";
  const text = String(value).trim();
  return text || "";
}

function evidenceStatusText(enabled, fpsBaseline) {
  if (!enabled) return "Sustained frame-pacing baseline was not included in this readiness run.";
  if (fpsBaseline === "baseline-passed") return "Sustained frame-pacing baseline passed for current real scenes plus synthetic 1M.";
  return "Sustained frame-pacing baseline is present but not passed.";
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
