import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readlinkSync,
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
const handoffPath = args.handoffMd ?? path.join(outputDir, "handoff.md");
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
  ...optionalPair("--background-slot", args.backgroundSlot),
  ...optionalPair("--background-weight", args.backgroundWeight),
  ...optionalPair("--object-min-confidence", args.objectMinConfidence),
  ...optionalPair("--unknown-object-id", args.unknownObjectId),
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

const backgroundOptions = [
  "--target-hardware",
  targetHardware,
  "--gpu-memory-reserve-gb",
  gpuMemoryReserveGb,
  "--output-dir",
  outputDir,
  "--port",
  port,
  ...optionalPair("--status-json", args.statusJson ?? args.statusJsonOutput),
  ...optionalPair("--candidate-status-json", args.candidateStatusJson),
  ...optionalPair("--manifest", args.manifest),
  ...optionalPair("--log-path", args.logPath),
  ...optionalPair("--handoff-md", args.handoffMd),
  ...optionalPair("--sam-checkpoint", args.samCheckpoint),
  ...optionalPair("--iterations", args.iterations),
  ...optionalPair("--steps-per-save", args.stepsPerSave),
  ...optionalPair("--camera-res-scale-factor", args.cameraResScaleFactor),
  ...optionalPair("--max-jobs", args.maxJobs),
  ...optionalPair("--gpu-index", args.gpuIndex),
  ...optionalPair("--min-exported-gaussians", args.minExportedGaussians),
  ...optionalPair("--background-slot", args.backgroundSlot),
  ...optionalPair("--background-weight", args.backgroundWeight),
  ...optionalPair("--object-min-confidence", args.objectMinConfidence),
  ...optionalPair("--unknown-object-id", args.unknownObjectId),
  ...(args.skipPull ? ["--skip-pull"] : []),
  ...(args.skipGpuPreflight ? ["--skip-gpu-preflight"] : []),
  ...(args.allowSlaFailures ? ["--allow-sla-failures"] : []),
];

const backgroundStartCommand = [
  "npm",
  "run",
  "train:splatfacto:near1m-background",
  "--",
  "--run",
  "--confirm-long-run",
  ...backgroundOptions,
];

const backgroundPreflightCommand = [
  "npm",
  "run",
  "train:splatfacto:near1m-background",
  "--",
  "--preflight",
  ...backgroundOptions,
];

const backgroundStatusCommand = [
  "npm",
  "run",
  "train:splatfacto:near1m-background",
  "--",
  "--status",
  "--output-dir",
  outputDir,
  "--candidate-status-json",
  candidateStatusPath,
];

const backgroundStopCommand = [
  "npm",
  "run",
  "train:splatfacto:near1m-background",
  "--",
  "--stop",
  "--confirm-stop",
  "--output-dir",
  outputDir,
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
  writeReport(report);
  process.exit(0);
}

if (mode === "dry-run") {
  const report = buildDryRunReport();
  printDryRun(report);
  writeReport(report);
  process.exit(0);
}

if (mode === "preflight") {
  const report = runPreflight();
  printPreflight(report);
  writeReport(report);
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
  process.exit(report.status === "stop-failed" || report.status === "stop-refused" ? 2 : 0);
}

if (!confirmLongRun) {
  console.error(
    "near1m_background_guard=failed reason=\"background near-1M training requires --confirm-long-run\"",
  );
  process.exit(2);
}

const existing = readManifestIfPresent();
const existingProcess = inspectBackgroundProcess(existing);
if (existingProcess.blocksNewRun && !allowExisting) {
  console.error(
    `near1m_background_guard=failed reason=${JSON.stringify(
      `existing launcher pid ${existing.pid} is ${existingProcess.status}; pass --allow-existing only for diagnostics`,
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
    handoffPath,
  };
  manifest.launchProcessIdentity = readProcIdentity(child.pid);
  writeJson(manifestPath, manifest);
  const candidateStatus = readJsonIfPresent(candidateStatusPath);
  const processIdentity = inspectBackgroundProcess(manifest);
  const launchStatus = {
    schema: "objgauss-near1m-background-status-v1",
    mode,
    generatedAt: new Date().toISOString(),
    status: processIdentity.running ? "running" : "unknown",
    pid: child.pid,
    process: processIdentity,
    manifestPath,
    logPath,
    handoffPath,
    candidateStatusPath,
    logBytes: existsSync(logPath) ? statSync(logPath).size : 0,
    tail: existsSync(logPath) ? tailFile(logPath, 40) : [],
    candidateStatus,
    candidateSummary: summarizeCandidateStatus(candidateStatus),
    manifest,
    commandText: manifest.commandText,
    startedAt: manifest.startedAt,
  };
  launchStatus.handoff = buildHandoff({
    candidateStatus,
    candidateSummary: launchStatus.candidateSummary,
    running: processIdentity.running,
  });
  writeReport(launchStatus);
  console.log(
    `near1m_background=started pid=${child.pid} log=${logPath} manifest=${manifestPath} handoff_md=${handoffPath}`,
  );
  printHandoff(launchStatus.handoff);
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
    handoffPath,
    targetHardware,
    gpuMemoryReserveGb: Number.parseFloat(gpuMemoryReserveGb),
    handoff: buildHandoff({ candidateStatus: null, running: false }),
  };
}

function buildStatusReport() {
  const manifest = readManifestIfPresent();
  const pid = manifest?.pid;
  const processIdentity = inspectBackgroundProcess(manifest);
  const running = processIdentity.running;
  const logStats = existsSync(logPath) ? statSync(logPath) : null;
  const candidateStatus = readJsonIfPresent(candidateStatusPath);
  const status = processIdentity.status === "stale-pid"
    ? "stale-pid"
    : running
      ? "running"
      : manifest
        ? "not-running"
        : "not-started";
  const report = {
    schema: "objgauss-near1m-background-status-v1",
    mode,
    generatedAt: new Date().toISOString(),
    status,
    pid: pid ?? null,
    process: processIdentity,
    manifestPath,
    logPath,
    handoffPath,
    candidateStatusPath,
    logBytes: logStats?.size ?? 0,
    tail: existsSync(logPath) ? tailFile(logPath, 40) : [],
    candidateStatus,
    candidateSummary: summarizeCandidateStatus(candidateStatus),
    manifest,
  };
  report.handoff = buildHandoff({
    candidateStatus,
    candidateSummary: report.candidateSummary,
    running,
  });
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
    handoffPath,
    targetHardware,
    gpuMemoryReserveGb: Number.parseFloat(gpuMemoryReserveGb),
    candidateStatus,
    candidateSummary,
    handoff: buildHandoff({
      candidateStatus,
      candidateSummary,
      running: false,
    }),
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
  const processIdentity = inspectBackgroundProcess(manifest);
  if (!processIdentity.running) {
    return {
      schema: "objgauss-near1m-background-stop-v1",
      status: processIdentity.status === "stale-pid" ? "stale-pid" : "not-running",
      stoppedAt: new Date().toISOString(),
      pid,
      signal: stopSignal,
      manifestPath,
      logPath,
      candidateStatusPath,
      reason: processIdentity.reason ?? "recorded pid is not running",
      process: processIdentity,
      manifest,
    };
  }
  if (!processIdentity.verified) {
    return {
      schema: "objgauss-near1m-background-stop-v1",
      status: "stop-refused",
      stoppedAt: new Date().toISOString(),
      pid,
      signal: stopSignal,
      manifestPath,
      logPath,
      candidateStatusPath,
      reason: processIdentity.reason ?? "recorded pid is alive but process identity is not verified",
      process: processIdentity,
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
      process: processIdentity,
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
      process: processIdentity,
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
  console.log(`handoff_md=${report.handoffPath}`);
  printHandoff(report.handoff);
}

function printStatus(report) {
  console.log(
    `near1m_background=${report.status} pid=${report.pid ?? "none"} log=${report.logPath} log_bytes=${report.logBytes} handoff_md=${report.handoffPath}`,
  );
  if (report.process) {
    console.log(
      `near1m_process_identity=${report.process.status} verified=${report.process.verified} blocks_new_run=${report.process.blocksNewRun} match_method=${report.process.matchMethod ?? "none"} reason=${JSON.stringify(report.process.reason ?? "")}`,
    );
  }
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
  printHandoff(report.handoff);
}

function printPreflight(report) {
  console.log(
    `near1m_background_preflight=${report.status} exit_code=${report.exitCode ?? "unknown"} candidate_status_json=${report.candidateStatusPath} handoff_md=${report.handoffPath}`,
  );
  if (report.candidateSummary) {
    console.log(
      `near1m_launch_readiness=${report.candidateSummary.launchReadiness} launch_missing=${report.candidateSummary.launchMissing} gpu_preflight=${report.candidateSummary.gpuMemoryPreflight} candidate_status=${report.candidateSummary.status} missing=${report.candidateSummary.missing} blockers=${report.candidateSummary.blockers}`,
    );
  } else {
    console.log(`near1m_candidate_status=missing path=${report.candidateStatusPath}`);
  }
  printHandoff(report.handoff);
}

function printStop(report) {
  console.log(
    `near1m_background_stop=${report.status} pid=${report.pid ?? "none"} signal=${report.signal} reason=${JSON.stringify(report.reason ?? "")}`,
  );
  if (report.process) {
    console.log(
      `near1m_process_identity=${report.process.status} verified=${report.process.verified} match_method=${report.process.matchMethod ?? "none"} reason=${JSON.stringify(report.process.reason ?? "")}`,
    );
  }
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

function buildHandoff({ candidateStatus, candidateSummary, running }) {
  const launchReadiness = candidateSummary?.launchReadiness ?? "unknown";
  const finalStatus = candidateSummary?.status ?? "unknown";
  const remainingEvidence = summarizeRemainingEvidence(candidateStatus, candidateSummary);
  const canStart = !running && launchReadiness === "ready";
  const finalComplete = finalStatus === "ready" && remainingEvidence.length === 0;
  let nextAction = "refresh-preflight";
  let nextCommand = backgroundPreflightCommand;

  if (running) {
    nextAction = "monitor-background";
    nextCommand = backgroundStatusCommand;
  } else if (finalComplete) {
    nextAction = "production-sla-ready";
    nextCommand = backgroundStatusCommand;
  } else if (canStart) {
    nextAction = "start-background-long-run";
    nextCommand = backgroundStartCommand;
  } else if (candidateSummary) {
    nextAction = "fix-launch-readiness";
    nextCommand = backgroundPreflightCommand;
  }

  return {
    schema: "objgauss-near1m-background-handoff-v1",
    nextAction,
    canStartLongRun: canStart,
    finalCandidateStatus: finalStatus,
    launchReadiness,
    remainingEvidence,
    commands: {
      next: formatCommand(nextCommand),
      preflight: formatCommand(backgroundPreflightCommand),
      startBackground: formatCommand(backgroundStartCommand),
      status: formatCommand(backgroundStatusCommand),
      stop: formatCommand(backgroundStopCommand),
    },
    safety: {
      startsTraining: nextAction === "start-background-long-run",
      requiresExplicitConfirmation: true,
      confirmationFlag: "--confirm-long-run",
      note: "preflight/status/dry-run do not start Splatfacto; start-background-long-run does.",
    },
  };
}

function summarizeRemainingEvidence(candidateStatus, candidateSummary) {
  if (!candidateStatus) {
    return [
      {
        kind: "missing-candidate-status",
        label: "candidate status report",
        nextEvidence: "run background preflight to refresh near1m-candidate-status.json",
      },
    ];
  }
  if (candidateSummary?.status === "ready") return [];
  const blockers = Array.isArray(candidateStatus.blockers) ? candidateStatus.blockers : [];
  if (blockers.length > 0) {
    return blockers.map((blocker) => ({
      kind: blocker.kind ?? "blocker",
      label: blocker.label ?? "unknown",
      path: blocker.path ?? null,
      count: blocker.count ?? null,
      minGaussians: blocker.minGaussians ?? null,
      nextEvidence: blocker.prepare ?? "rerun near-1M candidate pipeline and production SLA",
    }));
  }
  return [
    {
      kind: "incomplete-candidate",
      label: "near-1M production SLA evidence",
      nextEvidence: "produce exported/object-aware PLY >= min threshold and pass production SLA",
    },
  ];
}

function printHandoff(handoff) {
  if (!handoff) return;
  console.log(
    `near1m_next_action=${handoff.nextAction} can_start_long_run=${handoff.canStartLongRun} final_candidate_status=${handoff.finalCandidateStatus} remaining_evidence=${handoff.remainingEvidence.length}`,
  );
  console.log(`near1m_next_command=${handoff.commands.next}`);
  if (handoff.remainingEvidence.length > 0) {
    for (const [index, item] of handoff.remainingEvidence.entries()) {
      const countText = item.count === null || item.count === undefined ? "" : ` count=${item.count}`;
      const minText =
        item.minGaussians === null || item.minGaussians === undefined ? "" : ` min_gaussians=${item.minGaussians}`;
      const pathText = item.path ? ` path=${JSON.stringify(item.path)}` : "";
      console.log(
        `near1m_remaining_evidence_${index + 1}=${item.kind} label=${JSON.stringify(
          item.label,
        )}${pathText}${countText}${minText} next=${JSON.stringify(item.nextEvidence)}`,
      );
    }
  }
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

function inspectBackgroundProcess(manifest) {
  const pid = manifest?.pid;
  if (!Number.isFinite(Number(pid)) || Number(pid) <= 0) {
    return {
      pid: pid ?? null,
      status: "missing-pid",
      running: false,
      verified: false,
      blocksNewRun: false,
      reason: "launcher manifest is missing or has no valid pid",
    };
  }
  if (!isPidAlive(pid)) {
    return {
      pid: Number(pid),
      status: "not-running",
      running: false,
      verified: false,
      blocksNewRun: false,
      reason: "recorded pid is not running",
    };
  }

  const identity = readProcIdentity(pid);
  const match = matchRecordedProcess(manifest, identity);
  if (match.status === "matched") {
    return {
      pid: Number(pid),
      status: "running",
      running: true,
      verified: true,
      blocksNewRun: true,
      reason: match.reason ?? "recorded pid matches near-1M launcher process identity",
      matchMethod: match.method ?? "unknown",
      identity,
    };
  }
  if (match.status === "mismatch") {
    return {
      pid: Number(pid),
      status: "stale-pid",
      running: false,
      verified: false,
      blocksNewRun: false,
      reason: match.reason,
      missingIdentityTokens: match.missingTokens,
      identity,
    };
  }
  return {
    pid: Number(pid),
    status: "running-unverified",
    running: true,
    verified: false,
    blocksNewRun: true,
    reason: match.reason,
    identity,
  };
}

function readProcIdentity(pid) {
  const procDir = `/proc/${Number(pid)}`;
  const identity = {
    pid: Number(pid),
    procAvailable: existsSync(procDir),
    argv: [],
    commandLine: null,
    cwd: null,
    processGroupId: null,
    startTimeTicks: null,
    readErrors: [],
  };
  try {
    const raw = readFileSync(path.join(procDir, "cmdline"));
    identity.argv = raw.toString("utf8").split("\0").filter(Boolean);
    identity.commandLine = identity.argv.join(" ");
  } catch (error) {
    identity.readErrors.push(`cmdline:${error?.code ?? error?.message ?? String(error)}`);
  }
  try {
    identity.cwd = readlinkSync(path.join(procDir, "cwd"));
  } catch (error) {
    identity.readErrors.push(`cwd:${error?.code ?? error?.message ?? String(error)}`);
  }
  try {
    const stat = parseProcStat(readFileSync(path.join(procDir, "stat"), "utf8"));
    identity.processGroupId = stat.processGroupId;
    identity.startTimeTicks = stat.startTimeTicks;
  } catch (error) {
    identity.readErrors.push(`stat:${error?.code ?? error?.message ?? String(error)}`);
  }
  return identity;
}

function parseProcStat(text) {
  const closeParen = text.lastIndexOf(")");
  if (closeParen < 0) throw new Error("invalid /proc stat format");
  const fieldsAfterCommand = text.slice(closeParen + 2).trim().split(/\s+/);
  return {
    processGroupId: Number.parseInt(fieldsAfterCommand[2], 10),
    startTimeTicks: Number.parseInt(fieldsAfterCommand[19], 10),
  };
}

function matchRecordedProcess(manifest, identity) {
  const launchIdentityMatch = matchLaunchProcessIdentity(manifest, identity);
  if (launchIdentityMatch.status === "matched") {
    return launchIdentityMatch;
  }

  if (!identity?.commandLine) {
    return {
      status: "unverified",
      reason: "process command line is unavailable; refusing to treat pid as verified near-1M training",
    };
  }

  const commandLine = identity.commandLine;
  const missingTokens = [];
  for (const token of ["train:splatfacto:near1m-candidate", "--confirm-long-run"]) {
    if (!commandLine.includes(token)) missingTokens.push(token);
  }
  if (manifest?.candidateStatusPath && !commandLine.includes(String(manifest.candidateStatusPath))) {
    missingTokens.push("candidateStatusPath");
  }
  if (manifest?.cwd && identity.cwd && path.resolve(identity.cwd) !== path.resolve(manifest.cwd)) {
    missingTokens.push("cwd");
  }
  if (missingTokens.length > 0) {
    return {
      status: "mismatch",
      missingTokens,
      reason: `recorded pid is alive but does not match near-1M launcher identity (${missingTokens.join(", ")})`,
    };
  }

  return {
    status: "matched",
    method: "command-line",
    reason: "recorded pid command line matches near-1M launcher command identity",
  };
}

function matchLaunchProcessIdentity(manifest, identity) {
  const recorded = manifest?.launchProcessIdentity;
  if (!recorded || !identity) return { status: "unavailable" };
  if (
    Number.isFinite(Number(recorded.startTimeTicks)) &&
    Number.isFinite(Number(identity.startTimeTicks)) &&
    Number(recorded.startTimeTicks) === Number(identity.startTimeTicks)
  ) {
    if (
      Number.isFinite(Number(recorded.processGroupId)) &&
      Number.isFinite(Number(identity.processGroupId)) &&
      Number(recorded.processGroupId) !== Number(identity.processGroupId)
    ) {
      return {
        status: "mismatch",
        method: "launch-process-start-time",
        reason: "recorded pid start time matches, but process group changed",
        missingTokens: ["processGroupId"],
      };
    }
    if (recorded.cwd && identity.cwd && path.resolve(recorded.cwd) !== path.resolve(identity.cwd)) {
      return {
        status: "mismatch",
        method: "launch-process-start-time",
        reason: "recorded pid start time matches, but cwd changed",
        missingTokens: ["cwd"],
      };
    }
    return {
      status: "matched",
      method: "launch-process-start-time",
      reason: "recorded pid start time matches the process captured at background launch",
    };
  }
  return { status: "unavailable" };
}

function tailFile(filePath, maxLines) {
  const text = readFileSync(filePath, "utf8");
  return text.trimEnd().split(/\r?\n/).slice(-maxLines);
}

function writeJson(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function writeReport(report) {
  writeJson(statusPath, report);
  writeHandoffMarkdown(handoffPath, report);
}

function writeHandoffMarkdown(filePath, report) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, renderHandoffMarkdown(report));
}

function renderHandoffMarkdown(report) {
  const handoff = report.handoff;
  const lines = [
    "# ObjGauss Near-1M Background Handoff",
    "",
    `- Generated: ${report.generatedAt ?? new Date().toISOString()}`,
    `- Mode: ${report.mode ?? mode}`,
    `- Background status: ${report.status ?? "unknown"}`,
    `- Target hardware: ${targetHardware}`,
    `- GPU reserve: ${gpuMemoryReserveGb} GB`,
    `- Status JSON: \`${statusPath}\``,
    `- Candidate status JSON: \`${candidateStatusPath}\``,
    `- Log path: \`${logPath}\``,
    ...(report.process
      ? [
          `- Process identity: \`${report.process.status}\``,
          `- Process verified: \`${report.process.verified}\``,
          `- Process match method: \`${report.process.matchMethod ?? "none"}\``,
          `- Process reason: ${report.process.reason ?? "unknown"}`,
        ]
      : []),
    "",
    "## Next Action",
    "",
    `- Action: \`${handoff?.nextAction ?? "unknown"}\``,
    `- Can start long run: \`${handoff?.canStartLongRun ?? false}\``,
    `- Final candidate status: \`${handoff?.finalCandidateStatus ?? "unknown"}\``,
    `- Launch readiness: \`${handoff?.launchReadiness ?? "unknown"}\``,
    `- Starts training: \`${handoff?.safety?.startsTraining ?? false}\``,
    "",
    "```bash",
    stripCommandPrompt(handoff?.commands?.next ?? ""),
    "```",
    "",
    "## Remaining Evidence",
    "",
  ];

  const remainingEvidence = Array.isArray(handoff?.remainingEvidence) ? handoff.remainingEvidence : [];
  if (remainingEvidence.length === 0) {
    lines.push("No remaining evidence blockers are reported by the current candidate status.");
  } else {
    for (const [index, item] of remainingEvidence.entries()) {
      lines.push(`### ${index + 1}. ${item.label ?? "Unknown evidence"}`, "");
      lines.push(`- Kind: \`${item.kind ?? "unknown"}\``);
      if (item.path) lines.push(`- Path: \`${item.path}\``);
      const count = formatCountCell(item);
      if (count) lines.push(`- Count: \`${count}\``);
      appendNextEvidenceMarkdown(lines, item.nextEvidence);
    }
  }

  lines.push(
    "",
    "## Safety",
    "",
    "This handoff file is not production SLA proof. `start-background-long-run` only means the launch inputs and GPU reserve gate are ready. Final completion still requires a real exported PLY and object-aware PLY at the configured near-1M scale gate plus a passing production SLA summary.",
    "",
  );
  return `${lines.join("\n")}\n`;
}

function stripCommandPrompt(commandText) {
  return String(commandText ?? "").replace(/^\$ /, "");
}

function appendNextEvidenceMarkdown(lines, nextEvidence) {
  const text = String(nextEvidence ?? "").trim();
  if (!text) return;
  if (isShellCommand(text)) {
    lines.push("- Next evidence:", "", "```bash", stripCommandPrompt(text), "```", "");
  } else {
    lines.push(`- Next evidence: ${text}`, "");
  }
}

function isShellCommand(text) {
  const command = stripCommandPrompt(text);
  return /^(npm|uv|node|python|python3|git|SAM_CHECKPOINT=)\b/.test(command);
}

function formatCountCell(item) {
  if (item?.count === null || item?.count === undefined) return "";
  if (item?.minGaussians === null || item?.minGaussians === undefined) return String(item.count);
  return `${item.count} / ${item.minGaussians}`;
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
