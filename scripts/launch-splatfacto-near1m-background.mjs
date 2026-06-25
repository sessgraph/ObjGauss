import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import path from "node:path";

const args = parseArgs(process.argv.slice(2));
const mode = args.run ? "run" : args.status ? "status" : args.stop ? "stop" : args.preflight ? "preflight" : "dry-run";

const outputDir = args.outputDir ?? "/tmp/objgauss-splatfacto-near1m-background";
const manifestPath = args.manifest ?? path.join(outputDir, "launcher.json");
const statusPath = args.statusJson ?? args.statusJsonOutput ?? path.join(outputDir, "status.json");
const candidateStatusPath = args.candidateStatusJson ?? path.join(outputDir, "near1m-candidate-status.json");
const logPath = args.logPath ?? path.join(outputDir, "near1m-run.log");
const targetHardware = args.targetHardware ?? "local-rtx5060ti";
const gpuMemoryReserveGb = args.gpuMemoryReserveGb ?? "1";
const port = args.port ?? "5395";
const confirmLongRun = Boolean(args.confirmLongRun);
const confirmStop = Boolean(args.confirmStop);
const allowExisting = Boolean(args.allowExisting);
const stopSignal = args.signal ?? "SIGTERM";

const commonCandidateOptions = [
  "--target-hardware",
  targetHardware,
  "--gpu-memory-reserve-gb",
  gpuMemoryReserveGb,
  "--port",
  port,
  "--status-json",
  candidateStatusPath,
  ...optionalPair("--sam-checkpoint", args.samCheckpoint),
  ...optionalPair("--iterations", args.iterations),
  ...optionalPair("--steps-per-save", args.stepsPerSave),
  ...optionalPair("--camera-res-scale-factor", args.cameraResScaleFactor),
  ...optionalPair("--max-jobs", args.maxJobs),
  ...optionalPair("--gpu-index", args.gpuIndex),
  ...optionalPair("--min-exported-gaussians", args.minExportedGaussians),
  ...(args.skipPull ? ["--skip-pull"] : []),
  ...(args.skipGpuPreflight ? ["--skip-gpu-preflight"] : []),
  ...(args.allowSlaFailures ? ["--allow-sla-failures"] : []),
];

const command = [
  "npm",
  "run",
  "train:splatfacto:near1m-candidate",
  "--",
  "--run",
  "--confirm-long-run",
  ...commonCandidateOptions,
];

const candidateStatusCommand = [
  "npm",
  "run",
  "train:splatfacto:near1m-candidate",
  "--",
  "--status",
  ...commonCandidateOptions,
];

if (mode === "status") {
  const report = buildStatusReport();
  printStatus(report);
  writeJson(statusPath, report);
  process.exit(0);
}

if (mode === "dry-run") {
  const report = buildDryRunReport();
  printDryRun(report);
  writeJson(statusPath, report);
  process.exit(0);
}

if (mode === "preflight") {
  const report = runPreflight();
  printPreflight(report);
  writeJson(statusPath, report);
  process.exit(report.status === "ready" ? 0 : 2);
}

if (mode === "stop") {
  if (!confirmStop) {
    console.error("near1m_background_stop=failed reason=\"stopping background near-1M training requires --confirm-stop\"");
    process.exit(2);
  }
  const report = stopBackgroundRun();
  printStop(report);
  writeJson(statusPath, report);
  process.exit(report.status === "stop-failed" ? 2 : 0);
}

if (!confirmLongRun) {
  console.error(
    "near1m_background_guard=failed reason=\"background near-1M training requires --confirm-long-run\"",
  );
  process.exit(2);
}

const existing = readManifestIfPresent();
if (existing?.pid && isPidAlive(existing.pid) && !allowExisting) {
  console.error(
    `near1m_background_guard=failed reason=${JSON.stringify(
      `existing launcher pid ${existing.pid} is still running; pass --allow-existing only for diagnostics`,
    )}`,
  );
  process.exit(2);
}

mkdirSync(outputDir, { recursive: true });
const logFd = openSync(logPath, "a");
try {
  const child = spawn(command[0], command.slice(1), {
    cwd: process.cwd(),
    detached: true,
    env: process.env,
    stdio: ["ignore", logFd, logFd],
  });
  child.unref();
  const manifest = {
    schema: "objgauss-near1m-background-launch-v1",
    status: "started",
    startedAt: new Date().toISOString(),
    pid: child.pid,
    processGroupId: child.pid,
    command,
    commandText: formatCommand(command),
    cwd: process.cwd(),
    outputDir,
    logPath,
    statusPath,
    candidateStatusPath,
    targetHardware,
    gpuMemoryReserveGb: Number.parseFloat(gpuMemoryReserveGb),
  };
  writeJson(manifestPath, manifest);
  writeJson(statusPath, {
    schema: "objgauss-near1m-background-status-v1",
    status: child.pid ? "running" : "unknown",
    pid: child.pid,
    manifestPath,
    logPath,
    candidateStatusPath,
    commandText: manifest.commandText,
    startedAt: manifest.startedAt,
  });
  console.log(`near1m_background=started pid=${child.pid} log=${logPath} manifest=${manifestPath}`);
} finally {
  closeSync(logFd);
}

function buildDryRunReport() {
  return {
    schema: "objgauss-near1m-background-dry-run-v1",
    mode,
    generatedAt: new Date().toISOString(),
    status: "not-started",
    command,
    commandText: formatCommand(command),
    outputDir,
    manifestPath,
    statusPath,
    candidateStatusPath,
    logPath,
    targetHardware,
    gpuMemoryReserveGb: Number.parseFloat(gpuMemoryReserveGb),
  };
}

function buildStatusReport() {
  const manifest = readManifestIfPresent();
  const pid = manifest?.pid;
  const running = pid ? isPidAlive(pid) : false;
  const logStats = existsSync(logPath) ? statSync(logPath) : null;
  const candidateStatus = readJsonIfPresent(candidateStatusPath);
  const report = {
    schema: "objgauss-near1m-background-status-v1",
    mode,
    generatedAt: new Date().toISOString(),
    status: running ? "running" : manifest ? "not-running" : "not-started",
    pid: pid ?? null,
    manifestPath,
    logPath,
    candidateStatusPath,
    logBytes: logStats?.size ?? 0,
    tail: existsSync(logPath) ? tailFile(logPath, 40) : [],
    candidateStatus,
    candidateSummary: summarizeCandidateStatus(candidateStatus),
    manifest,
  };
  return report;
}

function runPreflight() {
  mkdirSync(outputDir, { recursive: true });
  const result = spawnSync(candidateStatusCommand[0], candidateStatusCommand.slice(1), {
    cwd: process.cwd(),
    env: process.env,
    stdio: "inherit",
  });
  const candidateStatus = readJsonIfPresent(candidateStatusPath);
  const candidateSummary = summarizeCandidateStatus(candidateStatus);
  const commandPassed = result.status === 0 && !result.error;
  const status = commandPassed ? (candidateSummary?.launchReadiness === "ready" ? "ready" : "not-ready") : "failed";
  return {
    schema: "objgauss-near1m-background-preflight-v1",
    mode,
    generatedAt: new Date().toISOString(),
    status,
    exitCode: result.status ?? null,
    error: result.error?.message ?? null,
    command: candidateStatusCommand,
    commandText: formatCommand(candidateStatusCommand),
    outputDir,
    manifestPath,
    statusPath,
    candidateStatusPath,
    logPath,
    targetHardware,
    gpuMemoryReserveGb: Number.parseFloat(gpuMemoryReserveGb),
    candidateStatus,
    candidateSummary,
  };
}

function stopBackgroundRun() {
  const manifest = readManifestIfPresent();
  const pid = manifest?.pid;
  if (!pid) {
    return {
      schema: "objgauss-near1m-background-stop-v1",
      status: "not-started",
      stoppedAt: new Date().toISOString(),
      pid: null,
      signal: stopSignal,
      manifestPath,
      logPath,
      candidateStatusPath,
      reason: "launcher manifest is missing or has no pid",
    };
  }
  const running = isPidAlive(pid);
  if (!running) {
    return {
      schema: "objgauss-near1m-background-stop-v1",
      status: "not-running",
      stoppedAt: new Date().toISOString(),
      pid,
      signal: stopSignal,
      manifestPath,
      logPath,
      candidateStatusPath,
      reason: "recorded pid is not running",
      manifest,
    };
  }
  try {
    // The launcher starts the child detached, so the child PID is also the
    // process group id. Signal the group so nested training subprocesses exit.
    process.kill(-Number(pid), stopSignal);
    return {
      schema: "objgauss-near1m-background-stop-v1",
      status: "stop-sent",
      stoppedAt: new Date().toISOString(),
      pid,
      processGroupId: pid,
      signal: stopSignal,
      manifestPath,
      logPath,
      candidateStatusPath,
      manifest,
    };
  } catch (error) {
    return {
      schema: "objgauss-near1m-background-stop-v1",
      status: error?.code === "ESRCH" ? "not-running" : "stop-failed",
      stoppedAt: new Date().toISOString(),
      pid,
      processGroupId: pid,
      signal: stopSignal,
      manifestPath,
      logPath,
      candidateStatusPath,
      reason: error?.message ?? String(error),
      manifest,
    };
  }
}

function printDryRun(report) {
  console.log(`near1m_background=dry-run`);
  console.log(`command=${report.commandText}`);
  console.log(`log=${report.logPath}`);
  console.log(`manifest=${report.manifestPath}`);
  console.log(`status_json=${report.statusPath}`);
  console.log(`candidate_status_json=${report.candidateStatusPath}`);
}

function printStatus(report) {
  console.log(
    `near1m_background=${report.status} pid=${report.pid ?? "none"} log=${report.logPath} log_bytes=${report.logBytes}`,
  );
  if (report.candidateSummary) {
    console.log(
      `near1m_candidate_status=${report.candidateSummary.status} launch_readiness=${report.candidateSummary.launchReadiness} launch_missing=${report.candidateSummary.launchMissing} missing=${report.candidateSummary.missing} exported_gaussians=${report.candidateSummary.exportedGaussians} object_gaussians=${report.candidateSummary.objectGaussians} production_sla=${report.candidateSummary.productionSla} blockers=${report.candidateSummary.blockers} last_exit=${report.candidateSummary.lastExitStatus} last_failure=${report.candidateSummary.lastFailureKind}`,
    );
  } else {
    console.log(`near1m_candidate_status=missing path=${report.candidateStatusPath}`);
  }
  if (report.tail.length > 0) {
    console.log("log_tail_begin");
    for (const line of report.tail) console.log(line);
    console.log("log_tail_end");
  }
}

function printPreflight(report) {
  console.log(
    `near1m_background_preflight=${report.status} exit_code=${report.exitCode ?? "unknown"} candidate_status_json=${report.candidateStatusPath}`,
  );
  if (report.candidateSummary) {
    console.log(
      `near1m_launch_readiness=${report.candidateSummary.launchReadiness} launch_missing=${report.candidateSummary.launchMissing} gpu_preflight=${report.candidateSummary.gpuMemoryPreflight} candidate_status=${report.candidateSummary.status} missing=${report.candidateSummary.missing} blockers=${report.candidateSummary.blockers}`,
    );
  } else {
    console.log(`near1m_candidate_status=missing path=${report.candidateStatusPath}`);
  }
}

function printStop(report) {
  console.log(
    `near1m_background_stop=${report.status} pid=${report.pid ?? "none"} signal=${report.signal} reason=${JSON.stringify(report.reason ?? "")}`,
  );
}

function readManifestIfPresent() {
  return readJsonIfPresent(manifestPath);
}

function readJsonIfPresent(filePath) {
  if (!existsSync(filePath)) return null;
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function summarizeCandidateStatus(report) {
  if (!report) return null;
  return {
    schema: report.schema ?? "unknown",
    status: report.status ?? "unknown",
    missing: report.missing ?? "unknown",
    exportedGaussians: report.readiness?.exportedGaussians ?? "unknown",
    objectGaussians: report.readiness?.objectGaussians ?? "unknown",
    productionSla: report.readiness?.productionSla ?? "unknown",
    gpuMemoryPreflight: report.readiness?.gpuMemoryPreflight ?? "unknown",
    launchReadiness: report.launchReadiness?.status ?? "unknown",
    launchMissing: report.launchReadiness?.missing ?? "unknown",
    blockers: Array.isArray(report.blockers) ? report.blockers.length : "unknown",
    lastExitStatus: report.lastExit?.status ?? "unknown",
    lastExitCode: report.lastExit?.code ?? "unknown",
    lastFailureKind: report.lastFailure?.kind ?? "none",
    lastFailureReason: report.lastFailure?.reason ?? "",
  };
}

function isPidAlive(pid) {
  if (!Number.isFinite(Number(pid)) || Number(pid) <= 0) return false;
  try {
    process.kill(Number(pid), 0);
    return true;
  } catch {
    return false;
  }
}

function tailFile(filePath, maxLines) {
  const text = readFileSync(filePath, "utf8");
  return text.trimEnd().split(/\r?\n/).slice(-maxLines);
}

function writeJson(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function optionalPair(flag, value) {
  return value === undefined || value === null || value === "" ? [] : [flag, String(value)];
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--run") parsed.run = true;
    else if (value === "--dry-run") parsed.run = false;
    else if (value === "--status") parsed.status = true;
    else if (value === "--stop") parsed.stop = true;
    else if (value === "--preflight") parsed.preflight = true;
    else if (value === "--confirm-long-run") parsed.confirmLongRun = true;
    else if (value === "--confirm-stop") parsed.confirmStop = true;
    else if (value === "--allow-existing") parsed.allowExisting = true;
    else if (value === "--skip-pull") parsed.skipPull = true;
    else if (value === "--skip-gpu-preflight") parsed.skipGpuPreflight = true;
    else if (value === "--allow-sla-failures") parsed.allowSlaFailures = true;
    else if (value.startsWith("--")) {
      const key = value.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      const next = values[index + 1];
      if (!next || next.startsWith("--")) {
        throw new Error(`${value} requires a value`);
      }
      parsed[key] = next;
      index += 1;
    } else {
      throw new Error(`unknown argument: ${value}`);
    }
  }
  return parsed;
}

function formatCommand(values) {
  return `$ ${values.map(quote).join(" ")}`;
}

function quote(value) {
  const text = String(value);
  return /^[A-Za-z0-9_./:=@%+,-]+$/.test(text) ? text : JSON.stringify(text);
}
