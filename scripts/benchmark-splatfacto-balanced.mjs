import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const args = process.argv.slice(2);
const options = parseArgs(args);
const mode = options.run ? "run" : options.status ? "status" : "dry-run";

const paths = {
  dataset: options.dataset ?? "outputs/assets/training/nerf-synthetic-lego",
  inputPly:
    options.inputPly ??
    "outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply",
  samManifest:
    options.samManifest ?? "outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json",
  samCheckpoint:
    options.samCheckpoint ?? process.env.SAM_CHECKPOINT ?? "/home/ljy/models/sam/sam_vit_b_01ec64.pth",
  outputDir:
    options.outputDir ??
    "outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam8f-balanced03-slots4-benchmark",
  benchmarkOutputDir:
    options.benchmarkOutputDir ?? "/tmp/objgauss-splatfacto-balanced-benchmark",
};

const label = options.label ?? "safe-2000-balanced";
const slots = options.slots ?? "4";
const device = options.device ?? "cuda";
const samModelType = options.samModelType ?? "vit_b";
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
const assetId =
  options.assetId ?? "nerf-lego-splatfacto-safe-2000-sam8f-balanced03-slots4-benchmark";
const publicName = options.publicName ?? "nerf_lego_trained";
const dataparserTransform = options.dataparserTransform;

const trainingManifest = `${paths.outputDir}/training-output-manifest.json`;
const maskTrainingSummary = `${paths.outputDir}/mask-training-summary.json`;
const initialField = `${paths.outputDir}/object_field_initial.npz`;
const trainedField = `${paths.outputDir}/object_field_trained.npz`;
const objectPly = `${paths.outputDir}/object_aware_gaussians.ply`;
const emergenceJson = `${paths.benchmarkOutputDir}/emergence.json`;
const curveJson = `${paths.benchmarkOutputDir}/curve.json`;
const curveCsv = `${paths.benchmarkOutputDir}/curve.csv`;
const reportHtml = `${paths.benchmarkOutputDir}/report.html`;
const summaryJson = `${paths.benchmarkOutputDir}/summary.json`;

const skipSam = Boolean(options.skipSam);
const publish = Boolean(options.publish);

const steps = [
  {
    label: "Generate balanced SAM mask manifest",
    skip: skipSam,
    command: [
      "uv",
      "run",
      "--with",
      "torch",
      "--with",
      "torchvision",
      "--with",
      "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
      "objgauss",
      "masks",
      "from-nerf-sam",
      paths.dataset,
      "--output",
      paths.samManifest,
      "--checkpoint",
      paths.samCheckpoint,
      "--model-type",
      samModelType,
      "--device",
      device,
      "--split",
      "train",
      "--max-frames",
      samMaxFrames,
      "--max-masks-per-frame",
      samMaxMasksPerFrame,
      "--min-area",
      samMinArea,
      "--max-area-fraction",
      samMaxAreaFraction,
      ...optionalPair("--max-image-size", samMaxImageSize),
    ],
  },
  {
    label: "Apply dataparser transform to SAM mask manifest",
    skip: skipSam || !dataparserTransform,
    command: [
      "node",
      "scripts/apply-mask-dataparser-transform.mjs",
      paths.samManifest,
      "--dataparser-transform",
      dataparserTransform,
      "--output",
      paths.samManifest,
    ],
  },
  {
    label: "Register safe-2000 balanced Object Field",
    command: [
      "uv",
      "run",
      "objgauss",
      "training",
      "register-output",
      paths.inputPly,
      "--asset-id",
      assetId,
      "--output-dir",
      paths.outputDir,
      "--dataset",
      paths.dataset,
      "--masks",
      paths.samManifest,
      "--slots",
      slots,
      "--iterations",
      objectIterations,
      "--learning-rate",
      learningRate,
      ...(publish ? ["--public-name", publicName] : ["--no-public-copy"]),
    ],
  },
  {
    label: "Compute single-point emergence metrics",
    command: [
      "uv",
      "run",
      "objgauss",
      "object-field",
      "emergence",
      trainedField,
      "--cloud",
      paths.inputPly,
      "--reference",
      initialField,
      "--output",
      emergenceJson,
    ],
  },
  {
    label: "Compute emergence curve",
    command: [
      "uv",
      "run",
      "objgauss",
      "object-field",
      "emergence-curve",
      paths.inputPly,
      "--field",
      initialField,
      "--masks",
      paths.samManifest,
      "--output",
      curveJson,
      "--csv-output",
      curveCsv,
      "--iterations",
      curveIterations,
      "--learning-rate",
      learningRate,
      "--eval-every",
      evalEvery,
      "--render-size",
      renderSize,
    ],
  },
  {
    label: "Build benchmark report",
    command: [
      "uv",
      "run",
      "objgauss",
      "object-field",
      "emergence-report",
      curveJson,
      "--label",
      label,
      "--output",
      reportHtml,
      "--title",
      "ObjGauss Splatfacto Safe-2000 Balanced Benchmark",
    ],
  },
  {
    label: "Check object-aware PLY",
    command: ["uv", "run", "objgauss", "stats", objectPly],
    captureForSummary: true,
  },
];

if (mode === "status") {
  printStatus();
  process.exit(0);
}

console.log(`mode=${mode}`);
console.log(`dataset=${paths.dataset}`);
console.log(`input_ply=${paths.inputPly}`);
console.log(`sam_manifest=${paths.samManifest}`);
console.log(`sam_checkpoint=${paths.samCheckpoint}`);
console.log(`output_dir=${paths.outputDir}`);
console.log(`benchmark_output_dir=${paths.benchmarkOutputDir}`);
console.log(`publish=${publish ? "true" : "false"}`);

if (mode === "run") {
  const missing = collectMissingForRun();
  if (missing.length > 0) {
    printMissing(missing);
    process.exit(2);
  }
  mkdirSync(paths.outputDir, { recursive: true });
  mkdirSync(paths.benchmarkOutputDir, { recursive: true });
}

let statsOutput = "";
for (const step of steps.filter((item) => !item.skip)) {
  console.log(`\n=== ${step.label} ===`);
  console.log(formatCommand(step.command));
  if (mode === "run") {
    if (step.captureForSummary) {
      statsOutput = runCapture(step.command);
    } else {
      await run(step.command);
    }
  }
}

if (mode === "dry-run") {
  console.log("\ndry_run=passed");
} else {
  const summary = collectSummary(statsOutput);
  writeJson(summaryJson, summary);
  printSummary(summary);
  console.log(`\nsplatfacto_balanced_benchmark=${summary.passed ? "passed" : "failed"}`);
  if (!summary.passed) {
    process.exitCode = 2;
  }
}

function printStatus() {
  const checks = [
    {
      label: "dataset transforms",
      path: `${paths.dataset}/transforms_train.json`,
      prepare: "uv run objgauss assets pull nerf-synthetic-lego",
    },
    {
      label: "safe-2000 exported PLY",
      path: paths.inputPly,
      prepare: "see docs/training/splatfacto-smoke.md#train-003c-higher-quality-candidate",
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
      label: "registration manifest",
      path: trainingManifest,
      prepare: formatCommand(steps[1].command),
    },
    {
      label: "mask training summary",
      path: maskTrainingSummary,
      prepare: formatCommand(steps[1].command),
    },
    {
      label: "initial Object Field",
      path: initialField,
      prepare: formatCommand(steps[1].command),
    },
    {
      label: "trained Object Field",
      path: trainedField,
      prepare: formatCommand(steps[1].command),
    },
    {
      label: "object-aware PLY",
      path: objectPly,
      prepare: formatCommand(steps[1].command),
    },
    {
      label: "emergence metrics",
      path: emergenceJson,
      prepare: formatCommand(steps[2].command),
    },
    {
      label: "emergence curve",
      path: curveJson,
      prepare: formatCommand(steps[3].command),
    },
    {
      label: "benchmark report",
      path: reportHtml,
      prepare: formatCommand(steps[4].command),
    },
    {
      label: "benchmark summary",
      path: summaryJson,
      prepare: "npm run benchmark:splatfacto:balanced -- --run",
    },
  ];

  let missing = 0;
  for (const check of checks) {
    const ok = existsSync(check.path);
    if (!ok) {
      missing += 1;
    }
    console.log(`check=${check.label} status=${ok ? "present" : "missing"} path=${check.path}`);
    if (!ok) {
      console.log(`prepare=${check.prepare}`);
    }
  }
  console.log(`status=${missing === 0 ? "ready" : "incomplete"} missing=${missing}`);
}

function collectMissingForRun() {
  const required = [
    {
      label: "dataset transforms",
      path: `${paths.dataset}/transforms_train.json`,
      prepare: "uv run objgauss assets pull nerf-synthetic-lego",
    },
    {
      label: "safe-2000 exported PLY",
      path: paths.inputPly,
      prepare: "see docs/training/splatfacto-smoke.md#train-003c-higher-quality-candidate",
    },
  ];
  if (!skipSam && dataparserTransform) {
    required.push({
      label: "dataparser transform",
      path: dataparserTransform,
      prepare: "run the matching Nerfstudio Splatfacto training/export first",
    });
  }
  if (skipSam) {
    required.push({
      label: "balanced SAM manifest",
      path: paths.samManifest,
      prepare: formatCommand(steps[0].command),
    });
  } else {
    required.push({
      label: "SAM checkpoint",
      path: paths.samCheckpoint,
      prepare: "export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth",
    });
  }
  return required.filter((item) => !existsSync(item.path));
}

function optionalPair(flag, value) {
  return value === undefined || value === null ? [] : [flag, String(value)];
}

function printMissing(missing) {
  console.error("benchmark inputs are missing:");
  for (const item of missing) {
    console.error(`missing=${item.label} path=${item.path}`);
    console.error(`prepare=${item.prepare}`);
  }
}

function collectSummary(statsOutput) {
  const maskManifest = readJson(paths.samManifest);
  const trainingManifestData = readJson(trainingManifest);
  const emergence = readJson(emergenceJson);
  const curve = readJson(curveJson);
  const objectCounts = parseObjectCounts(statsOutput);
  const finalPoint = curve.points?.[curve.points.length - 1] ?? {};
  const maskStats = summarizeMaskManifest(maskManifest);
  const training = trainingManifestData.training ?? {};
  const checks = {
    frames_at_least_requested: maskStats.frames >= Number.parseInt(samMaxFrames, 10),
    masks_present: maskStats.masks > 0,
    object_slots_nonempty: objectCounts.length === Number.parseInt(slots, 10) && objectCounts.every((item) => item.count > 0),
    projection_loss_decreased: Number(training.final_loss) < Number(training.initial_loss),
    ari_recorded: typeof emergence.stability?.adjusted_rand_index === "number",
    oes_recorded: typeof emergence.object_emergence_score?.score === "number",
    render_occlusion_recorded:
      typeof finalPoint.render_occlusion_delta?.occlusion_effect_score === "number" ||
      typeof finalPoint.render_occlusion_delta?.mean_relative_delta_l1 === "number",
  };
  const passed = Object.values(checks).every(Boolean);
  return {
    kind: "splatfacto_safe_2000_balanced_benchmark",
    label,
    passed,
    generated_at: new Date().toISOString(),
    paths: {
      dataset: paths.dataset,
      input_ply: paths.inputPly,
      sam_manifest: paths.samManifest,
      output_dir: paths.outputDir,
      training_manifest: trainingManifest,
      object_ply: objectPly,
      emergence: emergenceJson,
      curve: curveJson,
      curve_csv: curveCsv,
      report: reportHtml,
      summary: summaryJson,
    },
    sam: {
      frames: maskStats.frames,
      masks: maskStats.masks,
      mask_pixels: maskStats.mask_pixels,
      slots: maskStats.slots,
      max_area_fraction: maskManifest.sam?.max_area_fraction ?? null,
      max_masks_per_frame: maskManifest.sam?.max_masks_per_frame ?? null,
    },
    registration: {
      gaussians: trainingManifestData.gaussian_count,
      slots: trainingManifestData.slots,
      supervised_gaussians: training.supervised_gaussians ?? null,
      initial_loss: training.initial_loss ?? null,
      final_loss: training.final_loss ?? null,
      iterations: training.iterations ?? null,
      changed_gaussians: trainingManifestData.object_field_delta?.changed_gaussians ?? null,
      changed_fraction: trainingManifestData.object_field_delta?.changed_fraction ?? null,
    },
    object_id_counts: objectCounts,
    emergence: {
      assignment_confidence: emergence.assignment?.assignment_confidence ?? null,
      mean_normalized_entropy: emergence.assignment?.mean_normalized_entropy ?? null,
      effective_slots: emergence.assignment?.effective_slots ?? null,
      stability_ari: emergence.stability?.adjusted_rand_index ?? null,
      matched_label_agreement: emergence.stability?.matched_label_agreement ?? null,
      spatial_compactness_score: emergence.spatial?.compactness_score ?? null,
      object_emergence_score: emergence.object_emergence_score?.score ?? null,
    },
    curve: {
      points: curve.points?.length ?? 0,
      initial_projection_loss: curve.points?.[0]?.projection_loss ?? null,
      final_projection_loss: finalPoint.projection_loss ?? null,
      final_assignment_confidence: finalPoint.assignment_confidence ?? null,
      final_ari_to_initial: finalPoint.ari_to_initial ?? null,
      final_spatial_compactness_score: finalPoint.spatial_compactness_score ?? null,
      final_mask_proxy_occlusion_mean_delta_loss:
        finalPoint.mask_proxy_occlusion_delta?.mean_delta_loss ?? null,
      final_render_occlusion_mean_delta_l1: finalPoint.render_occlusion_delta?.mean_delta_l1 ?? null,
      final_render_occlusion_mean_relative_delta_l1:
        finalPoint.render_occlusion_delta?.mean_relative_delta_l1 ?? null,
      final_render_occlusion_effect_score:
        finalPoint.render_occlusion_delta?.occlusion_effect_score ??
        finalPoint.render_occlusion_delta?.mean_relative_delta_l1 ??
        null,
      final_object_emergence_score: finalPoint.object_emergence_score?.score ?? null,
    },
    checks,
  };
}

function summarizeMaskManifest(manifest) {
  const frames = manifest.frames ?? [];
  let masks = 0;
  let maskPixels = 0;
  for (const frame of frames) {
    for (const mask of frame.masks ?? []) {
      masks += 1;
      maskPixels += Number(mask.area ?? 0);
    }
  }
  return {
    frames: frames.length,
    masks,
    mask_pixels: maskPixels,
    slots: manifest.slots?.length ?? 0,
  };
}

function parseObjectCounts(output) {
  const counts = [];
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^object_id=(\d+) count=(\d+)$/);
    if (match) {
      counts.push({ object_id: Number.parseInt(match[1], 10), count: Number.parseInt(match[2], 10) });
    }
  }
  return counts;
}

function printSummary(summary) {
  console.log(`summary=${summary.paths.summary}`);
  console.log(`frames=${summary.sam.frames}`);
  console.log(`masks=${summary.sam.masks}`);
  console.log(`mask_pixels=${summary.sam.mask_pixels}`);
  console.log(`object_id_counts=${summary.object_id_counts.map((item) => item.count).join("/")}`);
  console.log(`stability_ari=${formatNumber(summary.emergence.stability_ari)}`);
  console.log(`object_emergence_score=${formatNumber(summary.emergence.object_emergence_score)}`);
  console.log(
    `render_occlusion_effect_score=${formatNumber(summary.curve.final_render_occlusion_effect_score)}`,
  );
  console.log(`summary_status=${summary.passed ? "passed" : "failed"}`);
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--run") parsed.run = true;
    else if (value === "--dry-run") parsed.run = false;
    else if (value === "--status") parsed.status = true;
    else if (value === "--skip-sam") parsed.skipSam = true;
    else if (value === "--publish") parsed.publish = true;
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

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf-8"));
}

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function formatCommand(command) {
  return `$ ${command.map(quote).join(" ")}`;
}

function quote(value) {
  const text = String(value);
  return /^[A-Za-z0-9_./:=@%+,-]+$/.test(text) ? text : JSON.stringify(text);
}

function formatNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(6) : String(value);
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

function runCapture(command) {
  const result = spawnSync(command[0], command.slice(1), {
    env: process.env,
    encoding: "utf-8",
  });
  if (result.stdout) {
    process.stdout.write(result.stdout);
  }
  if (result.stderr) {
    process.stderr.write(result.stderr);
  }
  if (result.status !== 0) {
    throw new Error(`${command[0]} exited with code ${result.status}`);
  }
  return result.stdout ?? "";
}
