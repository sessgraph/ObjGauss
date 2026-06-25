import { closeSync, existsSync, mkdirSync, openSync, readSync, statSync, writeFileSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { dirname } from "node:path";

const args = process.argv.slice(2);
const options = parseArgs(args);
const mode = options.run
  ? "run"
  : options.status
    ? "status"
    : options.gpuPreflightOnly
      ? "gpu-preflight"
      : "dry-run";

const paths = {
  dataset: options.dataset ?? "outputs/assets/training/nerf-synthetic-lego",
  outputRoot: options.outputRoot ?? "outputs/training/nerf-lego-splatfacto-near1m",
  experiment: options.experiment ?? "lego-splatfacto-near1m",
  timestamp: options.timestamp ?? "near1m-cpu-cache-v1",
  exportDir:
    options.exportDir ?? "outputs/training/nerf-lego-splatfacto-near1m/export-near1m-cpu-cache-v1",
  objectFieldDir:
    options.objectFieldDir ??
    "outputs/training/nerf-lego-splatfacto-near1m/object-field-sam8f-balanced03-slots4",
  samManifest:
    options.samManifest ?? "outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json",
  samCheckpoint:
    options.samCheckpoint ?? process.env.SAM_CHECKPOINT ?? "/home/ljy/models/sam/sam_vit_b_01ec64.pth",
  candidateOutputDir:
    options.candidateOutputDir ??
    "outputs/assets/gaussians/nerf-lego-trained-near1m-sam8f-balanced03-slots4-candidate",
  benchmarkOutputDir:
    options.benchmarkOutputDir ?? "/tmp/objgauss-splatfacto-near1m-candidate-benchmark",
  slaOutputDir: options.slaOutputDir ?? "/tmp/objgauss-webgpu-cpath-production-sla-near1m-candidate",
};

const exportedPly = `${paths.exportDir}/splat.ply`;
const candidateObjectPly = `${paths.candidateOutputDir}/object_aware_gaussians.ply`;
const trainConfig = `${paths.outputRoot}/${paths.experiment}/splatfacto/${paths.timestamp}/config.yml`;
const checkpointStep = options.checkpointStep ?? formatCheckpointStep(options.iterations ?? "10000");
const checkpoint = `${paths.outputRoot}/${paths.experiment}/splatfacto/${paths.timestamp}/nerfstudio_models/step-${checkpointStep}.ckpt`;

const assetId = options.assetId ?? "nerf-lego-splatfacto-near1m-sam8f-balanced03-slots4-candidate";
const publicName = options.publicName ?? "nerf_lego_trained_near1m";
const label = options.label ?? "near1m-balanced";
const targetHardware = options.targetHardware ?? options.fpsSlaTargetHardware ?? "local-rtx5060ti";
const iterations = options.iterations ?? "10000";
const stepsPerSave = options.stepsPerSave ?? "1000";
const minExportedGaussians = positiveInteger(
  options.minExportedGaussians ?? options["min-exported-gaussians"] ?? options.minGaussians,
  1_000_000,
);
const dataParser = options.dataParser ?? "blender-data";
const vis = options.vis ?? "tensorboard";
const cacheImages = options.cacheImages ?? "cpu";
const cameraResScaleFactor = options.cameraResScaleFactor ?? "1.0";
const cudaHome =
  options.cudaHome ??
  process.env.CUDA_HOME ??
  (existsSync("/tmp/objgauss-cuda13") ? "/tmp/objgauss-cuda13" : undefined);
const maxJobs = options.maxJobs ?? process.env.MAX_JOBS ?? "2";
const device = options.device ?? "cuda";
const slots = options.slots ?? "4";
const samMaxFrames = options.samMaxFrames ?? "8";
const samMaxMasksPerFrame = options.samMaxMasksPerFrame ?? "4";
const samMinArea = options.samMinArea ?? "64";
const samMaxAreaFraction = options.samMaxAreaFraction ?? "0.3";
const samMaxImageSize = options.samMaxImageSize;
const objectIterations = options.objectIterations ?? "160";
const curveIterations = options.curveIterations ?? "80";
const evalEvery = options.evalEvery ?? "20";
const renderSize = options.renderSize ?? "96";
const learningRate = options.learningRate ?? "1.0";
const port = options.port ?? "5395";
const minTrainedApproxFps = options.fpsSlaMinTrainedApproxFps ?? options.minTrainedApproxFps ?? "24";
const gpuIndex = nonNegativeInteger(options.gpuIndex, 0);
const gpuMemoryReserveGb = nonNegativeNumber(options.gpuMemoryReserveGb, 1);

const skipTrain = Boolean(options.skipTrain);
const skipRegister = Boolean(options.skipRegister);
const skipSla = Boolean(options.skipSla);
const publish = Boolean(options.publish);
const allowSlaFailures = Boolean(options.allowSlaFailures);
const confirmLongRun = Boolean(options.confirmLongRun);
const skipGpuPreflight = Boolean(options.skipGpuPreflight);
const statusJsonPath = options.statusJson ?? options.statusJsonOutput;
let lastExit = null;
let lastFailure = null;

const steps = [
  {
    label: "Train/export near-1M Splatfacto candidate",
    skip: skipTrain,
    command: [
      "npm",
      "run",
      "train:splatfacto:smoke",
      "--",
      "--run",
      "--asset-id",
      options.sourceAssetId ?? "nerf-synthetic-lego",
      "--dataset",
      paths.dataset,
      "--output-root",
      paths.outputRoot,
      "--experiment",
      paths.experiment,
      "--timestamp",
      paths.timestamp,
      "--export-dir",
      paths.exportDir,
      "--object-field-dir",
      paths.objectFieldDir,
      "--sam-manifest",
      paths.samManifest,
      "--sam-checkpoint",
      paths.samCheckpoint,
      "--data-parser",
      dataParser,
      "--iterations",
      iterations,
      "--steps-per-save",
      stepsPerSave,
      "--vis",
      vis,
      "--cache-images",
      cacheImages,
      "--camera-res-scale-factor",
      cameraResScaleFactor,
      ...optionalPair("--cuda-home", cudaHome),
      "--max-jobs",
      maxJobs,
      "--device",
      device,
      "--sam-max-frames",
      samMaxFrames,
      "--sam-max-masks-per-frame",
      samMaxMasksPerFrame,
      "--sam-min-area",
      samMinArea,
      "--sam-max-area-fraction",
      samMaxAreaFraction,
      ...optionalPair("--sam-max-image-size", samMaxImageSize),
      "--slots",
      slots,
      "--object-iterations",
      objectIterations,
      "--skip-benchmark",
      ...(options.skipPull ? ["--skip-pull", "true"] : []),
    ],
  },
  {
    label: "Register balanced near-1M Object Field candidate",
    skip: skipRegister,
    beforeRun: () => assertPlyScale("exported near-1M PLY", exportedPly, minExportedGaussians),
    command: [
      "npm",
      "run",
      "benchmark:splatfacto:balanced",
      "--",
      "--run",
      "--skip-sam",
      "--input-ply",
      exportedPly,
      "--dataset",
      paths.dataset,
      "--sam-manifest",
      paths.samManifest,
      "--output-dir",
      paths.candidateOutputDir,
      "--benchmark-output-dir",
      paths.benchmarkOutputDir,
      "--asset-id",
      assetId,
      "--public-name",
      publicName,
      "--label",
      label,
      "--slots",
      slots,
      "--object-iterations",
      objectIterations,
      "--curve-iterations",
      curveIterations,
      "--eval-every",
      evalEvery,
      "--render-size",
      renderSize,
      "--learning-rate",
      learningRate,
      ...(publish ? ["--publish"] : []),
    ],
  },
  {
    label: "Run strict WebGPU C-path production SLA gate",
    skip: skipSla,
    beforeRun: () => assertPlyScale("candidate object-aware PLY", candidateObjectPly, minExportedGaussians),
    command: [
      "npm",
      "run",
      "audit:webgpu-cpath-production-sla",
      "--",
      "--trained-ply",
      candidateObjectPly,
      "--target-hardware",
      targetHardware,
      "--fps-sla-min-trained-approx-fps",
      minTrainedApproxFps,
      "--port",
      port,
      "--output-dir",
      paths.slaOutputDir,
      ...(allowSlaFailures ? ["--allow-failures"] : []),
    ],
  },
];

if (mode === "status") {
  const report = buildStatusReport();
  printStatus(report);
  if (statusJsonPath) {
    writeStatusJson(statusJsonPath, report);
  }
  process.exit(0);
}

if (mode === "gpu-preflight") {
  const report = buildGpuPreflightReport();
  printGpuPreflight(report.gpuMemoryPreflight);
  if (statusJsonPath) {
    writeStatusJson(statusJsonPath, report);
  }
  process.exit(report.gpuMemoryPreflight.status === "passed" || report.gpuMemoryPreflight.status === "skipped" ? 0 : 2);
}

console.log(`mode=${mode}`);
console.log(`dataset=${paths.dataset}`);
console.log(`exported_ply=${exportedPly}`);
console.log(`candidate_object_ply=${candidateObjectPly}`);
console.log(`sam_manifest=${paths.samManifest}`);
console.log(`target_hardware=${targetHardware}`);
console.log(`iterations=${iterations}`);
console.log(`camera_res_scale_factor=${cameraResScaleFactor}`);
console.log(`min_exported_gaussians=${minExportedGaussians}`);
console.log(`gpu_memory_reserve_gb=${gpuMemoryReserveGb}`);

if (mode === "run") {
  if (!skipTrain && !confirmLongRun) {
    const reason = "starting near-1M Splatfacto training requires --confirm-long-run";
    const hint = "use --skip-train when reusing an existing exported PLY";
    console.error(
      `near1m_long_run_guard=failed reason=${JSON.stringify(reason)}`,
    );
    console.error(`hint=${hint}`);
    exitWithStatusJson(2, {
      kind: "long-run-confirmation-required",
      phase: "preflight",
      reason,
      hint,
    });
  }
  const missing = collectMissingForRun();
  if (missing.length > 0) {
    printMissing(missing);
    exitWithStatusJson(2, {
      kind: "missing-inputs",
      phase: "preflight",
      reason: `${missing.length} required input(s) missing`,
      missing: missing.map((item) => ({
        label: item.label,
        path: item.path,
      })),
    });
  }
  if (!skipTrain) {
    try {
      assertGpuPreflight();
    } catch (error) {
      const reason = error?.message ?? String(error);
      const hint = "pass --skip-gpu-preflight only if you accept running without a 1GB reserve preflight";
      console.error(`near1m_gpu_preflight=failed reason=${JSON.stringify(reason)}`);
      console.error(`hint=${hint}`);
      exitWithStatusJson(2, {
        kind: "gpu-preflight-failed",
        phase: "preflight",
        reason,
        hint,
      });
    }
  }
}

try {
  for (const step of steps.filter((item) => !item.skip)) {
    console.log(`\n=== ${step.label} ===`);
    console.log(formatCommand(step.command));
    if (mode === "run") {
      if (step.beforeRun) {
        try {
          step.beforeRun();
        } catch (error) {
          const reason = error?.message ?? String(error);
          console.error(`near1m_scale_gate=failed reason=${JSON.stringify(reason)}`);
          exitWithStatusJson(2, {
            kind: "scale-gate-failed",
            phase: step.label,
            reason,
          });
        }
      }
      await run(step.command);
    }
  }
} catch (error) {
  const reason = error?.message ?? String(error);
  console.error(`near1m_step=failed reason=${JSON.stringify(reason)}`);
  exitWithStatusJson(2, {
    kind: "step-failed",
    phase: "run-step",
    reason,
  });
}

if (mode === "dry-run") {
  console.log("\ndry_run=passed");
} else {
  console.log("\ntrain_splatfacto_near1m_candidate=passed");
}
writeCurrentStatusJson({ code: 0 });

function buildStatusReport() {
  const exportedCount = existsSync(exportedPly) ? readPlyVertexCountOrZero(exportedPly) : 0;
  const objectCount = existsSync(candidateObjectPly) ? readPlyVertexCountOrZero(candidateObjectPly) : 0;
  const checks = [
    {
      label: "dataset transforms",
      path: `${paths.dataset}/transforms_train.json`,
      prepare: "uv run objgauss assets pull nerf-synthetic-lego",
    },
    {
      label: "SAM checkpoint",
      path: paths.samCheckpoint,
      prepare: "export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth",
    },
    {
      label: "balanced SAM manifest",
      path: paths.samManifest,
      prepare: formatCommand(steps[0].command),
    },
    {
      label: "Nerfstudio config",
      path: trainConfig,
      prepare: formatCommand(steps[0].command),
    },
    {
      label: "Nerfstudio checkpoint",
      path: checkpoint,
      prepare: formatCommand(steps[0].command),
    },
    {
      label: "exported near-1M PLY",
      path: exportedPly,
      count: exportedCount,
      prepare: formatCommand(steps[0].command),
    },
    {
      label: "candidate object-aware PLY",
      path: candidateObjectPly,
      count: objectCount,
      prepare: formatCommand(steps[1].command),
    },
    {
      label: "production SLA summary",
      path: `${paths.slaOutputDir}/summary.json`,
      prepare: formatCommand(steps[2].command),
    },
  ].map((check) => ({
    ...check,
    status: existsSync(check.path) ? "present" : "missing",
  }));
  const missingChecks = checks.filter((check) => check.status === "missing");
  const exportedReady = exportedCount >= minExportedGaussians;
  const objectReady = objectCount >= minExportedGaussians;
  const productionSlaReady = existsSync(`${paths.slaOutputDir}/summary.json`);
  const gpuPreflight = getGpuPreflight();
  const blockers = [];
  for (const check of missingChecks) {
    blockers.push({
      kind: "missing-file",
      label: check.label,
      path: check.path,
      prepare: check.prepare,
    });
  }
  if (exportedCount > 0 && !exportedReady) {
    blockers.push({
      kind: "under-scale-export",
      label: "exported near-1M PLY",
      path: exportedPly,
      count: exportedCount,
      minGaussians: minExportedGaussians,
    });
  }
  if (objectCount > 0 && !objectReady) {
    blockers.push({
      kind: "under-scale-object-ply",
      label: "candidate object-aware PLY",
      path: candidateObjectPly,
      count: objectCount,
      minGaussians: minExportedGaussians,
    });
  }
  return {
    schema: "objgauss-near1m-candidate-status-v1",
    generatedAt: new Date().toISOString(),
    mode,
    status: missingChecks.length === 0 && exportedReady && objectReady && productionSlaReady ? "ready" : "incomplete",
    missing: missingChecks.length,
    thresholds: {
      minExportedGaussians,
      minObjectGaussians: minExportedGaussians,
      minTrainedApproxFps: Number.parseFloat(minTrainedApproxFps),
      gpuMemoryReserveGb,
      gpuMemoryReserveMiB: Math.ceil(gpuMemoryReserveGb * 1024),
    },
    parameters: {
      targetHardware,
      iterations,
      stepsPerSave,
      cacheImages,
      cameraResScaleFactor,
      maxJobs,
      device,
      slots,
      samMaxFrames,
      samMaxMasksPerFrame,
      samMaxAreaFraction,
      objectIterations,
      curveIterations,
      evalEvery,
      renderSize,
      learningRate,
      port,
      gpuIndex,
    },
    flags: {
      skipTrain,
      skipRegister,
      skipSla,
      publish,
      allowSlaFailures,
      confirmLongRun,
      skipGpuPreflight,
    },
    paths: {
      dataset: paths.dataset,
      outputRoot: paths.outputRoot,
      experiment: paths.experiment,
      timestamp: paths.timestamp,
      exportDir: paths.exportDir,
      exportedPly,
      objectFieldDir: paths.objectFieldDir,
      samManifest: paths.samManifest,
      samCheckpoint: paths.samCheckpoint,
      candidateOutputDir: paths.candidateOutputDir,
      candidateObjectPly,
      benchmarkOutputDir: paths.benchmarkOutputDir,
      slaOutputDir: paths.slaOutputDir,
      productionSlaSummary: `${paths.slaOutputDir}/summary.json`,
      trainConfig,
      checkpoint,
    },
    checks,
    readiness: {
      exportedPly: exportedReady ? "ready" : "not-ready",
      exportedGaussians: exportedCount,
      candidateObjectPly: objectReady ? "ready" : "not-ready",
      objectGaussians: objectCount,
      productionSla: productionSlaReady ? "ready" : "not-ready",
      gpuMemoryPreflight: gpuPreflight.status,
    },
    gpuMemoryPreflight: gpuPreflight,
    lastExit,
    lastFailure,
    blockers,
    commands: {
      train: formatCommand(steps[0].command),
      register: formatCommand(steps[1].command),
      productionSla: formatCommand(steps[2].command),
      guardedRun: formatCommand([
        "npm",
        "run",
        "train:splatfacto:near1m-candidate",
        "--",
        "--run",
        "--confirm-long-run",
        "--target-hardware",
        targetHardware,
        "--gpu-memory-reserve-gb",
        String(gpuMemoryReserveGb),
      ]),
    },
  };
}

function buildGpuPreflightReport() {
  return {
    schema: "objgauss-near1m-gpu-preflight-v1",
    generatedAt: new Date().toISOString(),
    mode,
    thresholds: {
      gpuMemoryReserveGb,
      gpuMemoryReserveMiB: Math.ceil(gpuMemoryReserveGb * 1024),
    },
    parameters: {
      device,
      gpuIndex,
      targetHardware,
    },
    flags: {
      skipGpuPreflight,
    },
    gpuMemoryPreflight: getGpuPreflight(),
  };
}

function printStatus(report) {
  for (const check of report.checks) {
    const countText = Number.isFinite(check.count) ? ` count=${check.count}` : "";
    console.log(`check=${check.label} status=${check.status} path=${check.path}${countText}`);
    if (check.status !== "present") {
      console.log(`prepare=${check.prepare}`);
    }
  }
  console.log(
    `near1m_export=${report.readiness.exportedPly} exported_gaussians=${report.readiness.exportedGaussians} min_exported_gaussians=${minExportedGaussians}`,
  );
  console.log(
    `near1m_object_ply=${report.readiness.candidateObjectPly} object_gaussians=${report.readiness.objectGaussians} min_object_gaussians=${minExportedGaussians}`,
  );
  console.log(`production_sla=${report.readiness.productionSla}`);
  printGpuPreflight(report.gpuMemoryPreflight);
  console.log(`status=${report.status} missing=${report.missing}`);
}

function printGpuPreflight(gpu) {
  console.log(
    `gpu_preflight=${gpu.status} reserve_mib=${gpu.reserveMiB} free_mib=${gpu.freeMiB ?? "unknown"} device=${gpu.deviceIndex ?? gpuIndex} reason=${JSON.stringify(gpu.reason ?? "")}`,
  );
}

function writeStatusJson(filePath, report) {
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`status_json=${filePath}`);
}

function writeCurrentStatusJson({ code = 0, failure = null } = {}) {
  if (statusJsonPath) {
    lastExit = {
      status: code === 0 ? "passed" : "failed",
      code,
      mode,
      at: new Date().toISOString(),
    };
    lastFailure = failure
      ? {
          ...failure,
          at: lastExit.at,
        }
      : null;
    writeStatusJson(statusJsonPath, buildStatusReport());
  }
}

function exitWithStatusJson(code, failure = null) {
  writeCurrentStatusJson({ code, failure });
  process.exit(code);
}

function collectMissingForRun() {
  const required = [
    {
      label: "dataset transforms",
      path: `${paths.dataset}/transforms_train.json`,
      prepare: "uv run objgauss assets pull nerf-synthetic-lego",
    },
  ];
  if (!skipTrain) {
    required.push({
      label: "SAM checkpoint",
      path: paths.samCheckpoint,
      prepare: "export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth",
    });
  }
  if (skipTrain) {
    required.push({
      label: "exported near-1M PLY",
      path: exportedPly,
      prepare: formatCommand(steps[0].command),
    });
    required.push({
      label: "balanced SAM manifest",
      path: paths.samManifest,
      prepare: formatCommand(steps[0].command),
    });
  }
  if (skipRegister && !skipSla) {
    required.push({
      label: "candidate object-aware PLY",
      path: candidateObjectPly,
      prepare: formatCommand(steps[1].command),
    });
  }
  return required.filter((item) => !existsSync(item.path));
}

function printMissing(missing) {
  console.error("near-1M candidate inputs are missing:");
  for (const item of missing) {
    console.error(`missing=${item.label} path=${item.path}`);
    console.error(`prepare=${item.prepare}`);
  }
}

function assertGpuPreflight() {
  const result = getGpuPreflight();
  if (result.status !== "passed" && result.status !== "skipped") {
    throw new Error(result.reason ?? `GPU preflight status is ${result.status}`);
  }
  console.log(
    `gpu_preflight=${result.status} device=${result.deviceIndex ?? gpuIndex} reserve_mib=${result.reserveMiB} free_mib=${result.freeMiB ?? "unknown"}`,
  );
}

function getGpuPreflight() {
  const reserveMiB = Math.ceil(gpuMemoryReserveGb * 1024);
  if (skipGpuPreflight) {
    return {
      status: "skipped",
      reason: "--skip-gpu-preflight",
      reserveGb: gpuMemoryReserveGb,
      reserveMiB,
    };
  }
  if (device !== "cuda") {
    return {
      status: "skipped",
      reason: `device=${device}`,
      reserveGb: gpuMemoryReserveGb,
      reserveMiB,
    };
  }
  const command = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
  ];
  const result = spawnSync(command[0], command.slice(1), {
    encoding: "utf8",
  });
  if (result.error) {
    return {
      status: "unavailable",
      reason: result.error.message,
      command: formatCommand(command),
      reserveGb: gpuMemoryReserveGb,
      reserveMiB,
    };
  }
  if (result.status !== 0) {
    return {
      status: "unavailable",
      reason: (result.stderr || result.stdout || `nvidia-smi exited with code ${result.status}`).trim(),
      command: formatCommand(command),
      reserveGb: gpuMemoryReserveGb,
      reserveMiB,
    };
  }
  const rows = parseNvidiaSmiRows(result.stdout);
  const selected = rows.find((row) => row.index === gpuIndex);
  if (!selected) {
    return {
      status: "unavailable",
      reason: `nvidia-smi returned no row for GPU index ${gpuIndex}`,
      command: formatCommand(command),
      reserveGb: gpuMemoryReserveGb,
      reserveMiB,
    };
  }
  const freeAfterReserveMiB = selected.freeMiB - reserveMiB;
  return {
    status: freeAfterReserveMiB >= 0 ? "passed" : "failed",
    reason:
      freeAfterReserveMiB >= 0
        ? "free memory satisfies reserve"
        : `GPU ${selected.index} free memory ${selected.freeMiB} MiB is below reserve ${reserveMiB} MiB`,
    command: formatCommand(command),
    deviceIndex: selected.index,
    name: selected.name,
    totalMiB: selected.totalMiB,
    usedMiB: selected.usedMiB,
    freeMiB: selected.freeMiB,
    reserveGb: gpuMemoryReserveGb,
    reserveMiB,
    freeAfterReserveMiB,
  };
}

function parseNvidiaSmiRows(output) {
  return output
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split(",").map((part) => part.trim()))
    .filter((parts) => parts.length >= 5)
    .map(([indexText, name, totalText, usedText, freeText]) => ({
      index: nonNegativeInteger(indexText, 0),
      name,
      totalMiB: nonNegativeInteger(totalText, 0),
      usedMiB: nonNegativeInteger(usedText, 0),
      freeMiB: nonNegativeInteger(freeText, 0),
    }));
}

function assertPlyScale(label, filePath, minGaussians) {
  if (!existsSync(filePath)) {
    throw new Error(`${label} is missing: ${filePath}`);
  }
  const count = readPlyVertexCount(filePath);
  if (count < minGaussians) {
    throw new Error(`${label} has ${count} Gaussians; expected >= ${minGaussians}`);
  }
  console.log(`scale_check=${label} gaussians=${count} min=${minGaussians} status=passed`);
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
    else if (value === "--skip-train") parsed.skipTrain = true;
    else if (value === "--skip-register") parsed.skipRegister = true;
    else if (value === "--skip-sla") parsed.skipSla = true;
    else if (value === "--skip-pull") parsed.skipPull = true;
    else if (value === "--publish") parsed.publish = true;
    else if (value === "--allow-sla-failures") parsed.allowSlaFailures = true;
    else if (value === "--confirm-long-run") parsed.confirmLongRun = true;
    else if (value === "--skip-gpu-preflight") parsed.skipGpuPreflight = true;
    else if (value === "--gpu-preflight-only") parsed.gpuPreflightOnly = true;
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

function formatCommand(command) {
  return `$ ${command.map(quote).join(" ")}`;
}

function quote(value) {
  const text = String(value);
  return /^[A-Za-z0-9_./:=@%+,-]+$/.test(text) ? text : JSON.stringify(text);
}

function positiveInteger(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function nonNegativeInteger(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function nonNegativeNumber(value, fallback) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function formatCheckpointStep(iterationCount) {
  const parsed = Number.parseInt(iterationCount, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return "000009999";
  }
  return String(parsed - 1).padStart(9, "0");
}

function readPlyVertexCountOrZero(filePath) {
  try {
    return readPlyVertexCount(filePath);
  } catch {
    return 0;
  }
}

function readPlyVertexCount(filePath) {
  const stats = statSync(filePath);
  const fd = openSync(filePath, "r");
  try {
    const buffer = Buffer.alloc(Math.min(131072, stats.size));
    const bytesRead = readSync(fd, buffer, 0, buffer.length, 0);
    const header = buffer.subarray(0, bytesRead).toString("utf8");
    const match = header.match(/^element\s+vertex\s+(\d+)\s*$/m);
    if (!match) return 0;
    return Number.parseInt(match[1], 10);
  } finally {
    closeSync(fd);
  }
}

function run(command) {
  return new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), {
      env: process.env,
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command[0]} exited with code ${code}`));
      }
    });
  });
}
