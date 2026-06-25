import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";

const args = parseArgs(process.argv.slice(2));
const mode = args.run ? "run" : args.status ? "status" : "dry-run";

const outputDir = args.outputDir ?? "/tmp/objgauss-splatfacto-near1m-background";
const manifestPath = args.manifest ?? path.join(outputDir, "launcher.json");
const statusPath = args.statusJson ?? args.statusJsonOutput ?? path.join(outputDir, "status.json");
const logPath = args.logPath ?? path.join(outputDir, "near1m-run.log");
const targetHardware = args.targetHardware ?? "local-rtx5060ti";
const gpuMemoryReserveGb = args.gpuMemoryReserveGb ?? "1";
const port = args.port ?? "5395";
const confirmLongRun = Boolean(args.confirmLongRun);
const allowExisting = Boolean(args.allowExisting);

const command = [
  "npm",
  "run",
  "train:splatfacto:near1m-candidate",
  "--",
  "--run",
  "--confirm-long-run",
  "--target-hardware",
  targetHardware,
  "--gpu-memory-reserve-gb",
  gpuMemoryReserveGb,
  "--port",
  port,
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
    command,
    commandText: formatCommand(command),
    cwd: process.cwd(),
    outputDir,
    logPath,
    statusPath,
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
  const report = {
    schema: "objgauss-near1m-background-status-v1",
    mode,
    generatedAt: new Date().toISOString(),
    status: running ? "running" : manifest ? "not-running" : "not-started",
    pid: pid ?? null,
    manifestPath,
    logPath,
    logBytes: logStats?.size ?? 0,
    tail: existsSync(logPath) ? tailFile(logPath, 40) : [],
    manifest,
  };
  return report;
}

function printDryRun(report) {
  console.log(`near1m_background=dry-run`);
  console.log(`command=${report.commandText}`);
  console.log(`log=${report.logPath}`);
  console.log(`manifest=${report.manifestPath}`);
  console.log(`status_json=${report.statusPath}`);
}

function printStatus(report) {
  console.log(
    `near1m_background=${report.status} pid=${report.pid ?? "none"} log=${report.logPath} log_bytes=${report.logBytes}`,
  );
  if (report.tail.length > 0) {
    console.log("log_tail_begin");
    for (const line of report.tail) console.log(line);
    console.log("log_tail_end");
  }
}

function readManifestIfPresent() {
  if (!existsSync(manifestPath)) return null;
  try {
    return JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch {
    return null;
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
    else if (value === "--confirm-long-run") parsed.confirmLongRun = true;
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
