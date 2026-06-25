import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";

const MODE = "near1m-production-gap-audit-v1";
const args = parseArgs(process.argv.slice(2));
const outputDir = String(args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-near1m-production-gap");
const candidateStatusPath = String(
  args.candidateStatusJson ?? args["candidate-status-json"] ?? path.join(outputDir, "candidate-status.json"),
);
const requireReady = flagEnabled(args.requireReady ?? args["require-ready"]);
const includeGpuPreflight = flagEnabled(args.includeGpuPreflight ?? args["include-gpu-preflight"]);

mkdirSync(outputDir, { recursive: true });

const candidateCommand = buildCandidateStatusCommand();
const result = spawnSync(candidateCommand[0], candidateCommand.slice(1), {
  encoding: "utf8",
  env: process.env,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);

const candidateStatus = readJsonIfPresent(candidateStatusPath);
const goalGap = candidateStatus?.goalGap ?? null;
const auditStatus = getAuditStatus({ result, candidateStatus, goalGap });
const passed = auditStatus === "ready" || (auditStatus === "incomplete" && !requireReady);
const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  requireReady,
  includeGpuPreflight,
  status: auditStatus,
  passed,
  command: formatCommand(candidateCommand),
  candidateExitCode: result.status ?? 1,
  candidateStatusPath,
  candidateStatus: candidateStatus?.status ?? "missing",
  launchReadiness: candidateStatus?.launchReadiness?.status ?? "missing",
  readiness: candidateStatus?.readiness ?? null,
  goalGap,
};

writeReport(summary);
printSummary(summary);

if (!summary.passed) {
  process.exitCode = 1;
}

function buildCandidateStatusCommand() {
  return [
    "npm",
    "run",
    "train:splatfacto:near1m-candidate",
    "--",
    "--status",
    "--status-json",
    candidateStatusPath,
    ...optionalPair("--target-hardware", args.targetHardware ?? args["target-hardware"]),
    ...optionalPair("--gpu-memory-reserve-gb", args.gpuMemoryReserveGb ?? args["gpu-memory-reserve-gb"]),
    ...optionalPair("--port", args.port),
    ...optionalPair("--sam-checkpoint", args.samCheckpoint ?? args["sam-checkpoint"]),
    ...optionalPair("--min-exported-gaussians", args.minExportedGaussians ?? args["min-exported-gaussians"]),
    ...optionalPair("--dataset", args.dataset),
    ...optionalPair("--output-root", args.outputRoot ?? args["output-root"]),
    ...optionalPair("--experiment", args.experiment),
    ...optionalPair("--timestamp", args.timestamp),
    ...optionalPair("--export-dir", args.exportDir ?? args["export-dir"]),
    ...optionalPair("--candidate-output-dir", args.candidateOutputDir ?? args["candidate-output-dir"]),
    ...optionalPair("--sla-output-dir", args.slaOutputDir ?? args["sla-output-dir"]),
    ...(includeGpuPreflight ? [] : ["--skip-gpu-preflight"]),
  ];
}

function getAuditStatus({ result, candidateStatus, goalGap }) {
  if ((result.status ?? 1) !== 0) return "failed";
  if (!candidateStatus || !goalGap) return "failed";
  return goalGap.status === "ready" ? "ready" : "incomplete";
}

function writeReport(summary) {
  writeFileSync(path.join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(path.join(outputDir, "summary.md"), renderMarkdown(summary));
}

function printSummary(summary) {
  const gap = summary.goalGap ?? {};
  console.log(
    `near1m_production_gap=${summary.status} next_action=${gap.nextAction ?? "unknown"} ` +
      `missing_evidence=${gap.missingEvidenceCount ?? "unknown"} ` +
      `completed_evidence=${gap.completedEvidenceCount ?? "unknown"} ` +
      `require_ready=${summary.requireReady} report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
  );
  const missing = Array.isArray(gap.missingEvidence) ? gap.missingEvidence : [];
  for (const [index, item] of missing.entries()) {
    const pathText = item.path ? ` path=${JSON.stringify(item.path)}` : "";
    const countText = Number.isFinite(item.count) ? ` count=${item.count}` : "";
    const minText = Number.isFinite(item.minGaussians) ? ` min_gaussians=${item.minGaussians}` : "";
    const statusText = item.status ? ` evidence_status=${JSON.stringify(item.status)}` : "";
    console.log(
      `near1m_production_blocker_${index + 1}=${item.kind} label=${JSON.stringify(
        item.label,
      )}${pathText}${countText}${minText}${statusText} next=${JSON.stringify(item.nextEvidence)}`,
    );
  }
}

function renderMarkdown(summary) {
  const gap = summary.goalGap ?? {};
  const lines = [
    "# Near-1M Production Gap Audit",
    "",
    `- Status: ${summary.status}`,
    `- Passed: ${summary.passed}`,
    `- Require ready: ${summary.requireReady}`,
    `- Candidate status: ${summary.candidateStatus}`,
    `- Launch readiness: ${summary.launchReadiness}`,
    `- Next action: ${gap.nextAction ?? "unknown"}`,
    `- Completed evidence: ${gap.completedEvidenceCount ?? "unknown"}`,
    `- Missing evidence: ${gap.missingEvidenceCount ?? "unknown"}`,
    "",
    "## Target",
    "",
    gap.target ?? "WebGPU C-path production proof with a real trained object-aware near-1M PLY.",
    "",
    "## Command",
    "",
    "```bash",
    stripPrompt(summary.command),
    "```",
    "",
  ];
  const missing = Array.isArray(gap.missingEvidence) ? gap.missingEvidence : [];
  if (missing.length > 0) {
    lines.push("## Missing Evidence", "");
    for (const [index, item] of missing.entries()) {
      lines.push(`### ${index + 1}. ${item.label ?? "Unknown evidence"}`, "");
      lines.push(`- Kind: \`${item.kind ?? "unknown"}\``);
      if (item.path) lines.push(`- Path: \`${item.path}\``);
      if (Number.isFinite(item.count) || Number.isFinite(item.minGaussians)) {
        lines.push(`- Count: \`${item.count ?? "unknown"} / ${item.minGaussians ?? "unknown"}\``);
      }
      if (item.status) lines.push(`- Evidence status: \`${item.status}\``);
      if (item.nextEvidence) {
        lines.push("- Next evidence:", "", "```bash", stripPrompt(item.nextEvidence), "```", "");
      }
    }
  } else {
    lines.push("## Missing Evidence", "", "No missing terminal evidence reported.", "");
  }
  return `${lines.join("\n")}\n`;
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith("--")) {
      throw new Error(`unknown argument: ${value}`);
    }
    const key = value.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    const next = values[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}

function readJsonIfPresent(filePath) {
  if (!existsSync(filePath)) return null;
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function optionalPair(flag, value) {
  return value === undefined || value === null || value === "" || value === true || value === false
    ? []
    : [flag, String(value)];
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

function stripPrompt(command) {
  return String(command ?? "").replace(/^\$ /, "");
}
