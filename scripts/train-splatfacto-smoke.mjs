import { existsSync } from "node:fs";
import { spawn } from "node:child_process";

const args = process.argv.slice(2);
const options = parseArgs(args);
const mode = options.run ? "run" : options.status ? "status" : "dry-run";

const paths = {
  dataset: options.dataset ?? "outputs/assets/training/nerf-synthetic-lego",
  outputRoot: options.outputRoot ?? "outputs/training/nerf-lego-splatfacto-smoke",
  experiment: options.experiment ?? "lego-splatfacto-smoke",
  timestamp: options.timestamp ?? "smoke-cuda",
  exportDir: options.exportDir ?? "outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda",
  objectFieldDir:
    options.objectFieldDir ?? "outputs/training/nerf-lego-splatfacto-smoke/object-field-sam",
  samManifest: options.samManifest ?? "outputs/masks/nerf-lego-sam/mask-manifest.json",
  samCheckpoint:
    options.samCheckpoint ?? process.env.SAM_CHECKPOINT ?? "/home/ljy/models/sam/sam_vit_b_01ec64.pth",
};

const iterations = options.iterations ?? "100";
const objectIterations = options.objectIterations ?? "80";
const device = options.device ?? "cuda";
const slots = options.slots ?? "8";
const samModelType = options.samModelType ?? "vit_b";
const samMaxFrames = options.samMaxFrames ?? "2";
const samMaxMasksPerFrame = options.samMaxMasksPerFrame ?? "8";
const samMinArea = options.samMinArea ?? "64";
const checkpointStep = options.checkpointStep ?? formatCheckpointStep(iterations);

const trainConfig = `${paths.outputRoot}/${paths.experiment}/splatfacto/${paths.timestamp}/config.yml`;
const checkpoint = `${paths.outputRoot}/${paths.experiment}/splatfacto/${paths.timestamp}/nerfstudio_models/step-${checkpointStep}.ckpt`;
const splatPly = `${paths.exportDir}/splat.ply`;
const initialField = `${paths.objectFieldDir}/object_field_initial.npz`;
const trainedField = `${paths.objectFieldDir}/object_field_sam.npz`;
const objectPly = `${paths.objectFieldDir}/lego_splatfacto_sam_objects.ply`;

const uvNerfstudioPrefix = [
  "uv",
  "run",
  "--with",
  "nerfstudio",
  "--with",
  "torch",
  "--with",
  "torchvision",
  "--with",
  "gsplat",
  "--with",
  "nvidia-cuda-nvcc==13.0.*",
  "--with",
  "nvidia-cuda-cccl==13.0.*",
  "--with",
  "nvidia-nvvm==13.0.*",
  "--with",
  "nvidia-cuda-crt==13.0.*",
];

const steps = [
  {
    label: "Pull NeRF Synthetic Lego dataset",
    command: ["uv", "run", "objgauss", "assets", "pull", "nerf-synthetic-lego"],
  },
  {
    label: "Train Splatfacto smoke",
    command: [
      ...uvNerfstudioPrefix,
      "ns-train",
      "splatfacto",
      "--max-num-iterations",
      iterations,
      "--output-dir",
      paths.outputRoot,
      "--experiment-name",
      paths.experiment,
      "--timestamp",
      paths.timestamp,
      "blender-data",
      "--data",
      paths.dataset,
    ],
  },
  {
    label: "Export Gaussian PLY",
    env: { TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD: "1" },
    command: [
      ...uvNerfstudioPrefix,
      "ns-export",
      "gaussian-splat",
      "--load-config",
      trainConfig,
      "--output-dir",
      paths.exportDir,
    ],
  },
  {
    label: "Generate SAM mask manifest",
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
    ],
  },
  {
    label: "Initialize Object Field",
    command: [
      "uv",
      "run",
      "objgauss",
      "object-field",
      "init",
      splatPly,
      "--output",
      initialField,
      "--slots",
      slots,
      "--ply-output",
      `${paths.objectFieldDir}/lego_splatfacto_sam_initial_objects.ply`,
      "--colorize",
    ],
  },
  {
    label: "Apply SAM mask voting",
    command: [
      "uv",
      "run",
      "objgauss",
      "object-field",
      "vote-masks",
      splatPly,
      "--field",
      initialField,
      "--masks",
      paths.samManifest,
      "--output",
      trainedField,
      "--ply-output",
      objectPly,
      "--iterations",
      objectIterations,
      "--learning-rate",
      "1.0",
      "--colorize",
      "--summary-output",
      `${paths.objectFieldDir}/sam-mask-training-summary.json`,
    ],
  },
  {
    label: "Check exported PLY",
    command: ["uv", "run", "objgauss", "stats", splatPly],
  },
  {
    label: "Run semantic acceptance benchmark",
    command: ["npm", "run", "acceptance:semantic"],
    skip: options.skipBenchmark,
  },
];

if (mode === "status") {
  printStatus();
  process.exit(0);
}

console.log(`mode=${mode}`);
console.log(`dataset=${paths.dataset}`);
console.log(`output_root=${paths.outputRoot}`);
console.log(`train_config=${trainConfig}`);
console.log(`exported_ply=${splatPly}`);
console.log(`sam_manifest=${paths.samManifest}`);
console.log(`sam_checkpoint=${paths.samCheckpoint}`);

for (const step of steps.filter((item) => !item.skip)) {
  console.log(`\n=== ${step.label} ===`);
  console.log(formatCommand(step.command, step.env));
  if (mode === "run") {
    await run(step.command, step.env);
  }
}

if (mode === "dry-run") {
  console.log("\ndry_run=passed");
} else {
  console.log("\ntrain_splatfacto_smoke=passed");
}

function printStatus() {
  const checks = [
    ["dataset transforms", `${paths.dataset}/transforms_train.json`],
    ["SAM checkpoint", paths.samCheckpoint],
    ["SAM manifest", paths.samManifest],
    ["Nerfstudio config", trainConfig],
    ["Nerfstudio checkpoint", checkpoint],
    ["exported PLY", splatPly],
    ["initial Object Field", initialField],
    ["trained Object Field", trainedField],
    ["object-aware PLY", objectPly],
  ];
  let missing = 0;
  for (const [label, path] of checks) {
    const ok = existsSync(path);
    if (!ok) {
      missing += 1;
    }
    console.log(`check=${label} status=${ok ? "present" : "missing"} path=${path}`);
  }
  console.log(`status=${missing === 0 ? "ready" : "incomplete"} missing=${missing}`);
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--run") parsed.run = true;
    else if (value === "--dry-run") parsed.run = false;
    else if (value === "--status") parsed.status = true;
    else if (value === "--skip-benchmark") parsed.skipBenchmark = true;
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

function formatCommand(command, env = {}) {
  const prefix = Object.entries(env).map(([key, value]) => `${key}=${quote(value)}`);
  return `$ ${[...prefix, ...command.map(quote)].join(" ")}`;
}

function quote(value) {
  const text = String(value);
  return /^[A-Za-z0-9_./:=@%+,-]+$/.test(text) ? text : JSON.stringify(text);
}

function formatCheckpointStep(iterationCount) {
  const parsed = Number.parseInt(iterationCount, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return "000000099";
  }
  return String(parsed - 1).padStart(9, "0");
}

function run(command, env = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), {
      env: { ...process.env, ...env },
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
