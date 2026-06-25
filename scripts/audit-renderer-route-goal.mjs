import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const MODE = "renderer-route-goal-audit-v1";
const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-renderer-route-goal");
const requireProductionReady = flagEnabled(
  args.requireProductionReady ?? args["require-production-ready"],
);
const requireCpathReadiness = flagEnabled(
  args.requireCpathReadiness ?? args["require-cpath-readiness"],
);
const providedCpathReadinessSummaryPath = optionalString(
  args.cpathReadinessSummary ?? args["cpath-readiness-summary"],
);
const cpathReadinessOutputDir = childDir("webgpu-cpath-readiness");
const cpathReadinessSummaryPath =
  providedCpathReadinessSummaryPath || path.join(cpathReadinessOutputDir, "summary.json");
const includeCpathReadiness =
  flagEnabled(args.includeCpathReadiness ?? args["include-cpath-readiness"]) ||
  Boolean(providedCpathReadinessSummaryPath);

mkdirSync(outputDir, { recursive: true });

const steps = [
  {
    id: "route-contract",
    label: "B -> C renderer route contract",
    command: [
      "npm",
      "run",
      "audit:renderer-route-contract",
      "--",
      "--output-dir",
      childDir("route-contract"),
    ],
    summaryPath: path.join(childDir("route-contract"), "summary.json"),
  },
  {
    id: "webgpu-scale-budget",
    label: "C-path 100k-1M storage budget",
    command: [
      "npm",
      "run",
      "audit:webgpu-scale-budget",
      "--",
      "--output-dir",
      childDir("webgpu-scale-budget"),
    ],
    summaryPath: path.join(childDir("webgpu-scale-budget"), "summary.json"),
  },
  {
    id: "webgpu-edit-cost-budget",
    label: "C-path object edit cost budget",
    command: [
      "npm",
      "run",
      "audit:webgpu-edit-cost-budget",
      "--",
      "--output-dir",
      childDir("webgpu-edit-cost-budget"),
    ],
    summaryPath: path.join(childDir("webgpu-edit-cost-budget"), "summary.json"),
  },
  {
    id: "near1m-production-gap",
    label: "Near-1M terminal production proof",
    command: [
      "npm",
      "run",
      "audit:near1m-production-gap",
      "--",
      "--output-dir",
      childDir("near1m-production-gap"),
      ...(requireProductionReady ? ["--require-ready"] : []),
    ],
    summaryPath: path.join(childDir("near1m-production-gap"), "summary.json"),
    allowIncomplete: !requireProductionReady,
  },
  ...(includeCpathReadiness && !providedCpathReadinessSummaryPath
    ? [
        {
          id: "webgpu-cpath-readiness",
          label: "WebGPU C-path runtime readiness",
          command: [
            "npm",
            "run",
            "audit:webgpu-cpath-readiness",
            "--",
            "--output-dir",
            cpathReadinessOutputDir,
          ],
          summaryPath: cpathReadinessSummaryPath,
        },
      ]
    : []),
];

const stepResults = appendExternalCpathReadinessResult(steps.map(runStep));
const evidence = buildEvidence(stepResults);
const foundationalPassed = evidence
  .filter((item) => item.required !== false && item.scope === "renderer-foundation")
  .every((item) => item.status === "passed");
const requiredRuntimePassed = evidence
  .filter((item) => item.required !== false && item.scope === "renderer-runtime")
  .every((item) => item.status === "passed");
const productionReady = evidence
  .filter((item) => item.required !== false && item.scope === "near1m-production")
  .every((item) => item.status === "passed");
const status = getGoalStatus({ foundationalPassed, requiredRuntimePassed, productionReady });
const missingEvidence = evidence.filter((item) => item.required !== false && item.status !== "passed");
const passed =
  status === "ready" ||
  (status === "incomplete" && missingEvidence.every((item) => missingEvidenceAllowed(item)));

const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  requireProductionReady,
  includeCpathReadiness,
  requireCpathReadiness,
  cpathReadinessSummaryPath,
  status,
  passed,
  nextAction: nextActionFor({ requiredRuntimePassed, productionReady }),
  steps: stepResults,
  evidence,
  missingEvidence,
};

writeReport(summary);
printSummary(summary);

if (!summary.passed) {
  process.exitCode = 1;
}

function runStep(step) {
  const startedAt = performance.now();
  const result = spawnSync(step.command[0], step.command.slice(1), {
    encoding: "utf8",
    env: process.env,
  });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  const exitCode = result.status ?? 1;
  const summary = readJsonIfPresent(step.summaryPath);
  const status = childStatus({ step, exitCode, summary });
  return {
    id: step.id,
    label: step.label,
    command: formatCommand(step.command),
    exitCode,
    durationMs: Math.round(performance.now() - startedAt),
    status,
    summaryPath: step.summaryPath,
    summary,
  };
}

function childStatus({ step, exitCode, summary }) {
  if (!summary) return "failed";
  if (summary.status === "incomplete") return "incomplete";
  if (exitCode === 0) return summary.status ?? "passed";
  if (step.allowIncomplete && summary.status === "incomplete") return "incomplete";
  return "failed";
}

function buildEvidence(results) {
  const route = byId(results, "route-contract");
  const scale = byId(results, "webgpu-scale-budget");
  const edit = byId(results, "webgpu-edit-cost-budget");
  const near1m = byId(results, "near1m-production-gap");
  const cpath = byId(results, "webgpu-cpath-readiness");
  const phases = route?.summary?.phases ?? {};
  const gap = near1m?.summary?.goalGap ?? {};
  const cpathRuntime = buildCpathRuntimeEvidence(cpath);
  return [
    {
      id: "phase-b-shader-gaussian-oit",
      scope: "renderer-foundation",
      required: true,
      label: "Phase B WebGL Gaussian shader + weighted OIT fallback",
      status: phaseStatus(phases["B-webgl-gaussian-oit"]),
      evidence: phaseEvidence(phases["B-webgl-gaussian-oit"]),
      source: route?.summaryPath,
    },
    {
      id: "phase-c-webgpu-tile",
      scope: "renderer-foundation",
      required: true,
      label: "Phase C WebGPU tile renderer architecture",
      status: phaseStatus(phases["C-webgpu-tile"]),
      evidence: phaseEvidence(phases["C-webgpu-tile"]),
      source: route?.summaryPath,
    },
    {
      id: "bridge-route-contract",
      scope: "renderer-foundation",
      required: true,
      label: "Explicit Spark / WebGPU / Gaussian OIT route boundary",
      status: phaseStatus(phases["bridge-route-contract"]),
      evidence: phaseEvidence(phases["bridge-route-contract"]),
      source: route?.summaryPath,
    },
    {
      id: "webgpu-1m-scale-budget",
      scope: "renderer-foundation",
      required: true,
      label: "C-path storage budget covers 100k-1M Gaussians",
      status: scale?.summary?.status === "passed" ? "passed" : "failed",
      evidence: summarizeBudgetRows(scale?.summary?.rows),
      source: scale?.summaryPath,
    },
    {
      id: "webgpu-object-edit-cost-budget",
      scope: "renderer-foundation",
      required: true,
      label: "C-path object-state-only edit update budget",
      status: edit?.summary?.status === "passed" ? "passed" : "failed",
      evidence: summarizeBudgetRows(edit?.summary?.rows),
      source: edit?.summaryPath,
    },
    ...cpathRuntime,
    {
      id: "near1m-production-proof",
      scope: "near1m-production",
      required: true,
      label: "Real trained near-1M object-aware PLY + production SLA",
      status: gap.status === "ready" ? "passed" : "incomplete",
      evidence:
        gap.status === "ready"
          ? `ready; completed=${gap.completedEvidenceCount ?? "unknown"}`
          : `missing=${gap.missingEvidenceCount ?? "unknown"}; next=${gap.nextAction ?? "unknown"}`,
      source: near1m?.summaryPath,
    },
  ];
}

function buildCpathRuntimeEvidence(step) {
  const summary = step?.summary;
  const source = step?.summaryPath ?? cpathReadinessSummaryPath;
  const cpathStatus = cpathSummaryStatus(summary);
  const runtime = summary?.evidence ?? {};
  const headed = runtime.headedBrowserTransition ?? {};
  const synthetic = runtime.browserRuntime1m ?? {};
  const trained = runtime.trainedPlyRuntime ?? {};
  const realTrained1m = runtime.realTrainedBrowserRuntime1m ?? {};
  const sustained = runtime.sustainedFramePacing ?? {};
  const fpsSla = runtime.fpsSla ?? {};
  return [
    {
      id: "webgpu-cpath-readiness",
      scope: "renderer-runtime",
      required: requireCpathReadiness,
      label: "WebGPU C-path readiness aggregate",
      status: cpathStatus,
      evidence: cpathSummaryEvidence(summary),
      source,
    },
    {
      id: "headed-browser-object-transition",
      scope: "renderer-runtime",
      required: requireCpathReadiness,
      label: "Headed browser WebGPU object transition",
      status: nestedStatus(headed.status, cpathStatus),
      evidence:
        headed.status === "passed"
          ? `${headed.scenes ?? "unknown"}/${headed.expectedScenes ?? "unknown"} scenes; largest=${
              headed.largestGaussians ?? "unknown"
            }`
          : cpathMissingEvidence("headed object transition"),
      source,
    },
    {
      id: "synthetic-1m-browser-runtime",
      scope: "renderer-runtime",
      required: requireCpathReadiness,
      label: "Synthetic 1M WebGPU browser upload/runtime",
      status: nestedStatus(synthetic.status, cpathStatus),
      evidence:
        synthetic.status === "passed"
          ? `${synthetic.proof ?? "proof"}; gaussians=${synthetic.gaussians ?? "unknown"}; minApproxFps=${
              synthetic.minApproxFps ?? "unknown"
            }`
          : cpathMissingEvidence("synthetic 1M browser runtime"),
      source,
    },
    {
      id: "trained-ply-runtime",
      scope: "renderer-runtime",
      required: false,
      label: "Caller-provided trained PLY runtime",
      status: nestedStatus(trained.status, cpathStatus),
      evidence:
        trained.status === "passed"
          ? `${trained.proof ?? "proof"}; gaussians=${trained.gaussians ?? "unknown"}; minRequired=${
              trained.minGaussians ?? "unknown"
            }`
          : trained.interpretation ?? cpathMissingEvidence("trained PLY runtime"),
      source,
    },
    {
      id: "real-trained-browser-runtime-1m",
      scope: "near1m-production",
      required: false,
      label: "Real trained 1M WebGPU browser runtime",
      status: nestedStatus(realTrained1m.status, cpathStatus),
      evidence:
        realTrained1m.status === "passed"
          ? "real trained >=1M browser runtime proven"
          : realTrained1m.interpretation ?? cpathMissingEvidence("real trained 1M runtime"),
      source,
    },
    {
      id: "sustained-frame-pacing-baseline",
      scope: "renderer-runtime",
      required: false,
      label: "Sustained frame-pacing baseline",
      status: nestedStatus(sustained.status, cpathStatus),
      evidence:
        sustained.status === "passed"
          ? `realMinFps=${sustained.realMinApproxFps ?? "unknown"}; syntheticMinFps=${
              sustained.syntheticMinApproxFps ?? "unknown"
            }; trainedMinFps=${sustained.trainedPlyMinApproxFps ?? "unknown"}`
          : sustained.interpretation ?? cpathMissingEvidence("sustained frame pacing"),
      source,
    },
    {
      id: "fps-sla",
      scope: "near1m-production",
      required: false,
      label: "Reviewed production FPS SLA",
      status: nestedStatus(fpsSla.status, cpathStatus),
      evidence:
        fpsSla.status === "passed"
          ? `target=${fpsSla.targetHardware ?? "unknown"}; trainedMinFps=${
              fpsSla.sustainedTrainedMinApproxFps ?? "unknown"
            }`
          : fpsSla.interpretation ?? cpathMissingEvidence("production FPS SLA"),
      source,
    },
  ];
}

function getGoalStatus({ foundationalPassed, requiredRuntimePassed, productionReady }) {
  if (!foundationalPassed) return "failed";
  if (!requiredRuntimePassed) return "incomplete";
  return productionReady ? "ready" : "incomplete";
}

function nextActionFor({ requiredRuntimePassed, productionReady }) {
  if (!requiredRuntimePassed) return "run-webgpu-cpath-readiness";
  return productionReady ? "none" : "start-background-long-run";
}

function missingEvidenceAllowed(item) {
  if (item.id === "near1m-production-proof" && !requireProductionReady) return true;
  if (item.scope === "renderer-runtime" && !requireCpathReadiness) return true;
  return false;
}

function phaseStatus(phase) {
  if (!phase) return "failed";
  return phase.failed === 0 && phase.passed === phase.total ? "passed" : "failed";
}

function phaseEvidence(phase) {
  if (!phase) return "missing phase summary";
  return `${phase.passed}/${phase.total} checks passed`;
}

function summarizeBudgetRows(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return "missing rows";
  return rows.map((row) => `${row.id}:${row.status}:${row.gaussians}`).join(", ");
}

function writeReport(summary) {
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary));
}

function renderMarkdown(summary) {
  const lines = [
    "# Renderer Route Goal Audit",
    "",
    `- Status: ${summary.status}`,
    `- Passed: ${summary.passed}`,
    `- Require production ready: ${summary.requireProductionReady}`,
    `- Include C-path readiness: ${summary.includeCpathReadiness}`,
    `- Require C-path readiness: ${summary.requireCpathReadiness}`,
    `- Next action: ${summary.nextAction}`,
    `- Output: ${summary.outputDir}`,
    "",
    "## Evidence",
    "",
    "| ID | Required | Status | Scope | Evidence | Source |",
    "| --- | --- | --- | --- | --- | --- |",
    ...summary.evidence.map(
      (item) =>
        `| ${escapeMarkdown(item.id)} | ${item.required === false ? "no" : "yes"} | ${item.status} | ${escapeMarkdown(item.scope)} | ${escapeMarkdown(
          item.evidence,
        )} | \`${escapeMarkdown(item.source ?? "")}\` |`,
    ),
    "",
    "## Steps",
    "",
    "| Step | Status | Exit | Duration ms | Command |",
    "| --- | --- | ---: | ---: | --- |",
    ...summary.steps.map(
      (step) =>
        `| ${escapeMarkdown(step.label)} | ${step.status} | ${step.exitCode} | ${step.durationMs} | \`${escapeMarkdown(
          step.command,
        )}\` |`,
    ),
    "",
  ];
  if (summary.missingEvidence.length > 0) {
    lines.push("## Missing Evidence", "");
    for (const item of summary.missingEvidence) {
      lines.push(`- ${item.id}: ${item.evidence}`);
    }
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
}

function printSummary(summary) {
  console.log(
    [
      `renderer_route_goal=${summary.status}`,
      `passed=${summary.passed}`,
      `require_production_ready=${summary.requireProductionReady}`,
      `include_cpath_readiness=${summary.includeCpathReadiness}`,
      `require_cpath_readiness=${summary.requireCpathReadiness}`,
      `missing=${summary.missingEvidence.length}`,
      `next_action=${summary.nextAction}`,
      `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
    ].join(" "),
  );
  for (const item of summary.missingEvidence) {
    console.log(
      `renderer_route_goal_missing=${item.id} status=${item.status} evidence=${JSON.stringify(
        item.evidence,
      )}`,
    );
  }
}

function appendExternalCpathReadinessResult(results) {
  if (!includeCpathReadiness || !providedCpathReadinessSummaryPath) return results;
  const summary = readJsonIfPresent(cpathReadinessSummaryPath);
  return [
    ...results,
    {
      id: "webgpu-cpath-readiness",
      label: "WebGPU C-path runtime readiness",
      command: `external summary: ${cpathReadinessSummaryPath}`,
      exitCode: summary ? 0 : null,
      durationMs: 0,
      status: summary ? summary.status ?? "provided" : "not-collected",
      summaryPath: cpathReadinessSummaryPath,
      summary,
    },
  ];
}

function byId(results, id) {
  return results.find((result) => result.id === id);
}

function readJsonIfPresent(filePath) {
  if (!existsSync(filePath)) return null;
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function cpathSummaryStatus(summary) {
  if (!summary) return "not-collected";
  if (summary.status === "passed") return "passed";
  if (summary.status === "failed") return "failed";
  return summary.status ?? "unknown";
}

function cpathSummaryEvidence(summary) {
  if (!summary) {
    return "not collected; pass --cpath-readiness-summary <summary.json> or --include-cpath-readiness";
  }
  return `status=${summary.status ?? "unknown"}; browserRuntime1m=${
    summary.evidence?.browserRuntime1m?.status ?? "unknown"
  }; realTrained1m=${summary.evidence?.realTrainedBrowserRuntime1m?.status ?? "unknown"}; fpsSla=${
    summary.evidence?.fpsSla?.status ?? "unknown"
  }`;
}

function nestedStatus(status, parentStatus) {
  if (status) return status;
  if (parentStatus === "not-collected") return "not-collected";
  return "unknown";
}

function cpathMissingEvidence(label) {
  if (!includeCpathReadiness) {
    return `not collected in this route-goal run; provide a C-path readiness summary to report ${label}`;
  }
  return `${label} not proven in the provided C-path readiness summary`;
}

function childDir(name) {
  return path.join(outputDir, name);
}

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return "";
  const text = String(value).trim();
  return text || "";
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

function formatCommand(values) {
  return `$ ${values.map(quote).join(" ")}`;
}

function quote(value) {
  const text = String(value);
  return /^[A-Za-z0-9_./:=@%+,-]+$/.test(text) ? text : JSON.stringify(text);
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}
