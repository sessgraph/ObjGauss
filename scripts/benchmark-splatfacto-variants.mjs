import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { spawn, spawnSync } from "node:child_process";

const args = process.argv.slice(2);
const options = parseArgs(args);
const mode = options.run ? "run" : options.status ? "status" : "dry-run";

const paths = {
  dataset: options.dataset ?? "outputs/assets/training/nerf-synthetic-lego",
  inputPly:
    options.inputPly ??
    "outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply",
  suiteOutputDir:
    options.suiteOutputDir ?? "/tmp/objgauss-splatfacto-safe-2000-variant-suite",
  samCheckpoint:
    options.samCheckpoint ?? process.env.SAM_CHECKPOINT ?? "/home/ljy/models/sam/sam_vit_b_01ec64.pth",
};

const common = {
  objectIterations: options.objectIterations ?? "160",
  curveIterations: options.curveIterations ?? "80",
  evalEvery: options.evalEvery ?? "20",
  renderSize: options.renderSize ?? "96",
  learningRate: options.learningRate ?? "1.0",
  device: options.device ?? "cuda",
};

const selectedVariantIds = toArray(options.variant);
const skipSam = Boolean(options.skipSam);
const variants = buildVariants().filter((variant) => {
  return selectedVariantIds.length === 0 || selectedVariantIds.includes(variant.id);
});

if (variants.length === 0) {
  console.error(`no benchmark variants selected: ${selectedVariantIds.join(",")}`);
  process.exit(2);
}

const summaryJson = `${paths.suiteOutputDir}/summary.json`;
const summaryCsv = `${paths.suiteOutputDir}/summary.csv`;
const summaryMd = `${paths.suiteOutputDir}/summary.md`;
const reportHtml = `${paths.suiteOutputDir}/report.html`;

if (mode === "status") {
  printStatus();
  process.exit(0);
}

console.log(`mode=${mode}`);
console.log(`dataset=${paths.dataset}`);
console.log(`input_ply=${paths.inputPly}`);
console.log(`suite_output_dir=${paths.suiteOutputDir}`);
console.log(`sam_checkpoint=${paths.samCheckpoint}`);
console.log(`variants=${variants.map((variant) => variant.id).join(",")}`);
console.log(`skip_sam=${skipSam ? "true" : "false"}`);

if (mode === "run") {
  const missing = collectMissingForRun();
  if (missing.length > 0) {
    printMissing(missing);
    process.exit(2);
  }
  mkdirSync(paths.suiteOutputDir, { recursive: true });
}

for (const variant of variants) {
  const command = buildVariantCommand(variant, mode === "run" ? "--run" : "--dry-run");
  console.log(`\n=== ${variant.label} ===`);
  console.log(formatCommand(command));
  if (mode === "run") {
    await run(command);
  }
}

if (mode === "dry-run") {
  console.log("\ndry_run=passed");
} else {
  writeSuiteArtifacts();
}

function buildVariants() {
  return [
    {
      id: "sam2f-slots8",
      label: "SAM 2f / 8 slots",
      slots: "8",
      samManifest: "outputs/masks/nerf-lego-sam/mask-manifest.json",
      outputDir: "outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam2f-slots8-benchmark",
      samMaxFrames: "2",
      samMaxMasksPerFrame: "8",
      samMaxAreaFraction: "1.0",
      maskPolicy: "2 frames, 8 largest SAM masks per frame",
    },
    {
      id: "sam8f-slots8-unfiltered",
      label: "SAM 8f / 8 slots unfiltered",
      slots: "8",
      samManifest: "outputs/masks/nerf-lego-sam-8f/mask-manifest.json",
      outputDir:
        "outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam8f-slots8-unfiltered-benchmark",
      samMaxFrames: "8",
      samMaxMasksPerFrame: "8",
      samMaxAreaFraction: "1.0",
      maskPolicy: "8 frames, 8 largest SAM masks per frame, no area cap",
    },
    {
      id: "sam8f-slots4-balanced03",
      label: "SAM 8f / 4 slots max-area 0.3",
      slots: "4",
      samManifest: "outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json",
      outputDir:
        "outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam8f-balanced03-slots4-benchmark",
      samMaxFrames: "8",
      samMaxMasksPerFrame: "4",
      samMaxAreaFraction: "0.3",
      maskPolicy: "8 frames, 4 largest SAM masks per frame after max_area_fraction=0.3",
    },
  ].map((variant) => ({
    ...variant,
    assetId: `nerf-lego-splatfacto-safe-2000-${variant.id}-benchmark`,
    benchmarkOutputDir: `${paths.suiteOutputDir}/${variant.id}`,
  }));
}

function buildVariantCommand(variant, modeFlag) {
  const command = [
    "node",
    "scripts/benchmark-splatfacto-balanced.mjs",
    modeFlag,
    "--label",
    variant.id,
    "--dataset",
    paths.dataset,
    "--input-ply",
    paths.inputPly,
    "--sam-manifest",
    variant.samManifest,
    "--sam-checkpoint",
    paths.samCheckpoint,
    "--output-dir",
    variant.outputDir,
    "--benchmark-output-dir",
    variant.benchmarkOutputDir,
    "--asset-id",
    variant.assetId,
    "--slots",
    variant.slots,
    "--sam-max-frames",
    variant.samMaxFrames,
    "--sam-max-masks-per-frame",
    variant.samMaxMasksPerFrame,
    "--sam-max-area-fraction",
    variant.samMaxAreaFraction,
    "--object-iterations",
    common.objectIterations,
    "--curve-iterations",
    common.curveIterations,
    "--eval-every",
    common.evalEvery,
    "--render-size",
    common.renderSize,
    "--learning-rate",
    common.learningRate,
    "--device",
    common.device,
  ];
  if (skipSam) {
    command.push("--skip-sam");
  }
  return command;
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
  ];
  for (const variant of variants) {
    checks.push(
      {
        label: `${variant.id} SAM manifest`,
        path: variant.samManifest,
        prepare: formatCommand(buildVariantCommand(variant, "--dry-run")),
      },
      {
        label: `${variant.id} summary`,
        path: `${variant.benchmarkOutputDir}/summary.json`,
        prepare: formatCommand(buildVariantCommand(variant, "--run")),
      },
    );
  }
  checks.push(
    {
      label: "variant suite summary",
      path: summaryJson,
      prepare: "npm run benchmark:splatfacto:variants -- --run",
    },
    {
      label: "variant suite CSV",
      path: summaryCsv,
      prepare: "npm run benchmark:splatfacto:variants -- --run",
    },
    {
      label: "variant suite Markdown",
      path: summaryMd,
      prepare: "npm run benchmark:splatfacto:variants -- --run",
    },
    {
      label: "variant suite report",
      path: reportHtml,
      prepare: "npm run benchmark:splatfacto:variants -- --run",
    },
  );

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
  if (existsSync(summaryJson)) {
    const summary = readJson(summaryJson);
    printSuiteSummary(summary);
  }
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
  if (skipSam) {
    for (const variant of variants) {
      required.push({
        label: `${variant.id} SAM manifest`,
        path: variant.samManifest,
        prepare: formatCommand(buildVariantCommand(variant, "--dry-run")),
      });
    }
  } else {
    required.push({
      label: "SAM checkpoint",
      path: paths.samCheckpoint,
      prepare: "export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth",
    });
  }
  return required.filter((item) => !existsSync(item.path));
}

function printMissing(missing) {
  console.error("benchmark suite inputs are missing:");
  for (const item of missing) {
    console.error(`missing=${item.label} path=${item.path}`);
    console.error(`prepare=${item.prepare}`);
  }
}

function writeSuiteArtifacts() {
  const summaries = variants.map((variant) => {
    const summary = readJson(`${variant.benchmarkOutputDir}/summary.json`);
    return flattenVariantSummary(variant, summary);
  });
  const reportCommand = [
    "uv",
    "run",
    "objgauss",
    "object-field",
    "emergence-report",
    ...summaries.map((summary) => summary.paths.curve),
    ...summaries.flatMap((summary) => ["--label", summary.id]),
    "--output",
    reportHtml,
    "--title",
    "ObjGauss Splatfacto Safe-2000 Mask Variant Benchmark",
  ];
  console.log("\n=== Build variant comparison report ===");
  console.log(formatCommand(reportCommand));
  runCapture(reportCommand);

  const summary = {
    kind: "splatfacto_safe_2000_variant_benchmark",
    passed: summaries.every((item) => item.passed),
    generated_at: new Date().toISOString(),
    paths: {
      dataset: paths.dataset,
      input_ply: paths.inputPly,
      suite_output_dir: paths.suiteOutputDir,
      summary: summaryJson,
      csv: summaryCsv,
      markdown: summaryMd,
      report: reportHtml,
    },
    variants: summaries,
    rankings: {
      by_render_occlusion_effect_score: rankBy(summaries, "render_occlusion_effect_score"),
      by_stability_ari: rankBy(summaries, "stability_ari"),
      by_object_emergence_score: rankBy(summaries, "object_emergence_score"),
    },
  };
  writeJson(summaryJson, summary);
  writeFileSync(summaryCsv, renderCsv(summaries), "utf-8");
  writeFileSync(summaryMd, renderMarkdown(summary), "utf-8");
  printSuiteSummary(summary);
  console.log(`\nsplatfacto_variant_benchmark=${summary.passed ? "passed" : "failed"}`);
  if (!summary.passed) {
    process.exitCode = 2;
  }
}

function flattenVariantSummary(variant, summary) {
  return {
    id: variant.id,
    label: variant.label,
    passed: summary.passed,
    mask_policy: variant.maskPolicy,
    configured_max_area_fraction: Number.parseFloat(variant.samMaxAreaFraction),
    paths: {
      sam_manifest: variant.samManifest,
      output_dir: variant.outputDir,
      curve: summary.paths.curve,
      summary: summary.paths.summary,
    },
    frames: summary.sam.frames,
    masks: summary.sam.masks,
    mask_pixels: summary.sam.mask_pixels,
    slots: summary.registration.slots,
    supervised_gaussians: summary.registration.supervised_gaussians,
    registration_initial_loss: summary.registration.initial_loss,
    registration_final_loss: summary.registration.final_loss,
    object_id_counts: summary.object_id_counts.map((item) => item.count),
    assignment_confidence: summary.emergence.assignment_confidence,
    effective_slots: summary.emergence.effective_slots,
    stability_ari: summary.emergence.stability_ari,
    object_emergence_score: summary.emergence.object_emergence_score,
    curve_object_emergence_score: summary.curve.final_object_emergence_score,
    render_occlusion_effect_score: summary.curve.final_render_occlusion_effect_score,
    final_projection_loss: summary.curve.final_projection_loss,
  };
}

function rankBy(rows, field) {
  return [...rows]
    .sort((left, right) => numberValue(right[field]) - numberValue(left[field]))
    .map((row, index) => ({
      rank: index + 1,
      id: row.id,
      value: row[field] ?? null,
    }));
}

function renderCsv(rows) {
  const headers = [
    "id",
    "label",
    "frames",
    "masks",
    "mask_pixels",
    "slots",
    "configured_max_area_fraction",
    "supervised_gaussians",
    "object_id_counts",
    "registration_final_loss",
    "final_projection_loss",
    "stability_ari",
    "object_emergence_score",
    "curve_object_emergence_score",
    "render_occlusion_effect_score",
  ];
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(
      headers
        .map((field) => {
          const value = field === "object_id_counts" ? row.object_id_counts.join("/") : row[field];
          return csvCell(value);
        })
        .join(","),
    );
  }
  return `${lines.join("\n")}\n`;
}

function renderMarkdown(summary) {
  const rows = [
    "# Splatfacto Safe-2000 Mask Variant Benchmark",
    "",
    `Generated: ${summary.generated_at}`,
    "",
    "| Variant | Frames | Masks | Slots | Supervised | Object IDs | ARI | OES | Curve OES | Render effect |",
    "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
  ];
  for (const item of summary.variants) {
    rows.push(
      `| ${item.id} | ${item.frames} | ${item.masks} | ${item.slots} | ${item.supervised_gaussians} | ${item.object_id_counts.join("/")} | ${formatNumber(item.stability_ari)} | ${formatNumber(item.object_emergence_score)} | ${formatNumber(item.curve_object_emergence_score)} | ${formatNumber(item.render_occlusion_effect_score)} |`,
    );
  }
  rows.push("", `Report: ${summary.paths.report}`, "");
  return `${rows.join("\n")}`;
}

function printSuiteSummary(summary) {
  const variants = summary.variants ?? [];
  console.log(`summary=${summary.paths?.summary ?? summaryJson}`);
  console.log(`variants=${variants.length}`);
  for (const item of variants) {
    console.log(
      `variant=${item.id} frames=${item.frames} masks=${item.masks} slots=${item.slots} object_id_counts=${item.object_id_counts.join("/")} ari=${formatNumber(item.stability_ari)} oes=${formatNumber(item.object_emergence_score)} render=${formatNumber(item.render_occlusion_effect_score)}`,
    );
  }
  const topRender = summary.rankings?.by_render_occlusion_effect_score?.[0];
  if (topRender) {
    console.log(`best_render_variant=${topRender.id} value=${formatNumber(topRender.value)}`);
  }
  console.log(`summary_status=${summary.passed ? "passed" : "failed"}`);
}

function toArray(value) {
  if (value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

function numberValue(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--run") parsed.run = true;
    else if (value === "--dry-run") parsed.run = false;
    else if (value === "--status") parsed.status = true;
    else if (value === "--skip-sam") parsed.skipSam = true;
    else if (value.startsWith("--")) {
      const key = value.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      const next = values[index + 1];
      if (!next || next.startsWith("--")) {
        throw new Error(`${value} requires a value`);
      }
      if (parsed[key] === undefined) {
        parsed[key] = next;
      } else if (Array.isArray(parsed[key])) {
        parsed[key].push(next);
      } else {
        parsed[key] = [parsed[key], next];
      }
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
