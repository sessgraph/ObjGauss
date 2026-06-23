import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawn } from "node:child_process";

const args = process.argv.slice(2);
const options = parseArgs(args);
const mode = options.run ? "run" : options.status ? "status" : "dry-run";

const paths = {
  semanticManifest: options.semanticManifest ?? "docs/benchmarks/semantic-smoke.json",
  semanticOutputDir: options.semanticOutputDir ?? "/tmp/objgauss-semantic-smoke-suite",
  sceneManifest: options.sceneManifest ?? "docs/benchmarks/splatfacto-scenes.json",
  sceneOutputDir: options.sceneOutputDir ?? "/tmp/objgauss-splatfacto-scene-suite",
  variantOutputDir:
    options.variantOutputDir ?? "/tmp/objgauss-splatfacto-safe-2000-variant-suite",
  outputDir: options.outputDir ?? "/tmp/objgauss-cross-scene-benchmark",
  samCheckpoint:
    options.samCheckpoint ?? process.env.SAM_CHECKPOINT ?? "/home/ljy/models/sam/sam_vit_b_01ec64.pth",
};

const skipSemantic = Boolean(options.skipSemantic);
const skipScenes = Boolean(options.skipScenes);
const skipVariants = Boolean(options.skipVariants);
const refreshSam = Boolean(options.refreshSam);

const summaryJson = `${paths.outputDir}/summary.json`;
const summaryCsv = `${paths.outputDir}/summary.csv`;
const summaryMd = `${paths.outputDir}/summary.md`;
const summaryHtml = `${paths.outputDir}/summary.html`;
const failureReport = `${paths.outputDir}/failure-report.md`;

const semanticSummaryJson = `${paths.semanticOutputDir}/summary.json`;
const sceneSummaryJson = `${paths.sceneOutputDir}/summary.json`;
const variantSummaryJson = `${paths.variantOutputDir}/summary.json`;

const semanticCommand = [
  "npm",
  "run",
  "acceptance:semantic",
  "--",
  "--manifest",
  paths.semanticManifest,
  "--output-dir",
  paths.semanticOutputDir,
];

const variantCommand = [
  "npm",
  "run",
  "benchmark:splatfacto:variants",
  "--",
  "--run",
  "--suite-output-dir",
  paths.variantOutputDir,
  "--sam-checkpoint",
  paths.samCheckpoint,
  ...(refreshSam ? [] : ["--skip-sam"]),
];

const sceneCommand = [
  "npm",
  "run",
  "benchmark:splatfacto:scenes",
  "--",
  "--run",
  "--manifest",
  paths.sceneManifest,
  "--suite-output-dir",
  paths.sceneOutputDir,
  "--sam-checkpoint",
  paths.samCheckpoint,
  ...(refreshSam ? [] : ["--skip-sam"]),
];

if (mode === "status") {
  printStatus();
  process.exit(0);
}

console.log(`mode=${mode}`);
console.log(`semantic_manifest=${paths.semanticManifest}`);
console.log(`semantic_output_dir=${paths.semanticOutputDir}`);
console.log(`scene_manifest=${paths.sceneManifest}`);
console.log(`scene_output_dir=${paths.sceneOutputDir}`);
console.log(`variant_output_dir=${paths.variantOutputDir}`);
console.log(`output_dir=${paths.outputDir}`);
console.log(`sam_checkpoint=${paths.samCheckpoint}`);
console.log(`skip_semantic=${skipSemantic ? "true" : "false"}`);
console.log(`skip_scenes=${skipScenes ? "true" : "false"}`);
console.log(`skip_variants=${skipVariants ? "true" : "false"}`);
console.log(`refresh_sam=${refreshSam ? "true" : "false"}`);

if (!skipSemantic) {
  console.log("\n=== Semantic smoke suite ===");
  console.log(formatCommand(mode === "run" ? semanticCommand : dryRunSemanticCommand()));
  if (mode === "run") {
    await run(semanticCommand);
  }
}

if (!skipScenes) {
  console.log("\n=== Splatfacto scene suite ===");
  console.log(formatCommand(mode === "run" ? sceneCommand : dryRunSceneCommand()));
  if (mode === "run") {
    await run(sceneCommand);
  }
}

if (!skipVariants) {
  console.log("\n=== Safe-2000 variant suite ===");
  console.log(formatCommand(mode === "run" ? variantCommand : dryRunVariantCommand()));
  if (mode === "run") {
    await run(variantCommand);
  }
}

if (mode === "dry-run") {
  console.log("\ndry_run=passed");
} else {
  const missing = collectMissingForAggregate();
  if (missing.length > 0) {
    printMissing(missing);
    process.exit(2);
  }
  mkdirSync(paths.outputDir, { recursive: true });
  const summary = buildCrossSceneSummary();
  writeJson(summaryJson, summary);
  writeFileSync(summaryCsv, renderCsv(summary.rows), "utf-8");
  writeFileSync(summaryMd, renderMarkdown(summary), "utf-8");
  writeFileSync(summaryHtml, renderHtml(summary), "utf-8");
  writeFileSync(failureReport, renderFailureReport(summary), "utf-8");
  printSummary(summary);
  console.log(`\ncross_scene_benchmark=${summary.passed ? "passed" : "failed"}`);
  if (!summary.passed) {
    process.exitCode = 2;
  }
}

function dryRunSemanticCommand() {
  return semanticCommand;
}

function dryRunVariantCommand() {
  const command = [
    "npm",
    "run",
    "benchmark:splatfacto:variants",
    "--",
    "--dry-run",
    "--suite-output-dir",
    paths.variantOutputDir,
    "--sam-checkpoint",
    paths.samCheckpoint,
  ];
  if (!refreshSam) {
    command.push("--skip-sam");
  }
  return command;
}

function dryRunSceneCommand() {
  const command = [
    "npm",
    "run",
    "benchmark:splatfacto:scenes",
    "--",
    "--dry-run",
    "--manifest",
    paths.sceneManifest,
    "--suite-output-dir",
    paths.sceneOutputDir,
    "--sam-checkpoint",
    paths.samCheckpoint,
  ];
  if (!refreshSam) {
    command.push("--skip-sam");
  }
  return command;
}

function printStatus() {
  const checks = [
    {
      label: "semantic summary",
      path: semanticSummaryJson,
      prepare: formatCommand(semanticCommand),
    },
    {
      label: "scene summary",
      path: sceneSummaryJson,
      prepare: formatCommand(sceneCommand),
    },
    {
      label: "variant summary",
      path: variantSummaryJson,
      prepare: formatCommand(variantCommand),
    },
    {
      label: "cross-scene summary",
      path: summaryJson,
      prepare: "npm run benchmark:cross-scene -- --run",
    },
    {
      label: "cross-scene CSV",
      path: summaryCsv,
      prepare: "npm run benchmark:cross-scene -- --run",
    },
    {
      label: "cross-scene Markdown",
      path: summaryMd,
      prepare: "npm run benchmark:cross-scene -- --run",
    },
    {
      label: "cross-scene HTML",
      path: summaryHtml,
      prepare: "npm run benchmark:cross-scene -- --run",
    },
    {
      label: "cross-scene failure report",
      path: failureReport,
      prepare: "npm run benchmark:cross-scene -- --run",
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
  if (existsSync(summaryJson)) {
    printSummary(readJson(summaryJson));
  }
}

function collectMissingForAggregate() {
  return [
    {
      label: "semantic summary",
      path: semanticSummaryJson,
      prepare: formatCommand(semanticCommand),
    },
    {
      label: "scene summary",
      path: sceneSummaryJson,
      prepare: formatCommand(sceneCommand),
    },
    {
      label: "variant summary",
      path: variantSummaryJson,
      prepare: formatCommand(variantCommand),
    },
  ].filter((item) => !existsSync(item.path));
}

function printMissing(missing) {
  console.error("cross-scene benchmark inputs are missing:");
  for (const item of missing) {
    console.error(`missing=${item.label} path=${item.path}`);
    console.error(`prepare=${item.prepare}`);
  }
}

function buildCrossSceneSummary() {
  const semanticSummary = readJson(semanticSummaryJson);
  const sceneSummary = readJson(sceneSummaryJson);
  const variantSummary = readJson(variantSummaryJson);
  const semanticRows = flattenSemanticRows(semanticSummary, paths.semanticManifest);
  const sceneRows = flattenSceneRows(sceneSummary);
  const variantRows = flattenVariantRows(variantSummary);
  const rows = [...semanticRows, ...sceneRows, ...variantRows];
  const summary = {
    kind: "object_emergence_cross_scene_benchmark",
    passed: rows.length > 0 && rows.every((row) => row.passed !== false),
    generated_at: new Date().toISOString(),
    paths: {
      semantic_manifest: paths.semanticManifest,
      semantic_summary: semanticSummaryJson,
      scene_manifest: paths.sceneManifest,
      scene_summary: sceneSummaryJson,
      variant_summary: variantSummaryJson,
      output_dir: paths.outputDir,
      summary: summaryJson,
      csv: summaryCsv,
      markdown: summaryMd,
      html: summaryHtml,
      failure_report: failureReport,
      semantic_report: reportPath(semanticSummary.report, `${paths.semanticOutputDir}/report.html`),
      scene_report: sceneSummary.paths?.report ?? `${paths.sceneOutputDir}/report.html`,
      variant_report: variantSummary.paths?.report ?? `${paths.variantOutputDir}/report.html`,
    },
    rows,
    rankings: {
      by_render_occlusion_effect_score: rankBy(rows, "render_occlusion_effect_score"),
      by_final_object_emergence_score: rankBy(rows, "final_object_emergence_score"),
      by_final_ari_to_initial: rankBy(rows, "final_ari_to_initial"),
    },
  };
  summary.best_by_scene = bestByScene(rows, "render_occlusion_effect_score");
  summary.stage_gates = evaluateStageGates(rows);
  return summary;
}

function flattenSceneRows(summary) {
  return (summary.scenes ?? []).map((scene) => ({
    suite: "splatfacto-scenes",
    scene_id: scene.id,
    scene_label: scene.label,
    variant_id: "default",
    variant_label: "default",
    mask_policy: scene.mask_policy,
    source_summary: scene.paths?.summary ?? sceneSummaryJson,
    curve: scene.paths?.curve ?? null,
    passed: scene.passed,
    gaussians: scene.gaussians ?? null,
    slots: scene.slots ?? null,
    frames: scene.frames ?? null,
    masks: scene.masks ?? null,
    mask_pixels: scene.mask_pixels ?? null,
    supervised_gaussians: scene.supervised_gaussians ?? null,
    object_id_counts: scene.object_id_counts ?? null,
    initial_projection_loss: scene.registration_initial_loss ?? null,
    final_projection_loss: scene.final_projection_loss ?? null,
    final_assignment_confidence: scene.assignment_confidence ?? null,
    final_ari_to_initial: scene.stability_ari ?? null,
    final_spatial_compactness_score: null,
    final_object_emergence_score:
      scene.curve_object_emergence_score ?? scene.object_emergence_score ?? null,
    render_occlusion_effect_score: scene.render_occlusion_effect_score ?? null,
    heldout_final_projection_loss:
      scene.heldout_projection_loss ?? scene.heldout?.final_projection_loss ?? null,
    heldout_supervised_gaussians:
      scene.heldout_supervised_gaussians ?? scene.heldout?.supervised_gaussians ?? null,
    heldout_render_occlusion_effect_score:
      scene.heldout_render_occlusion_effect_score ??
      scene.heldout?.render_occlusion_effect_score ??
      null,
  }));
}

function flattenSemanticRows(summary, manifestPath) {
  const manifest = readJson(manifestPath);
  const manifestRoot = benchmarkRoot(manifestPath, manifest);
  const manifestScenes = new Map((manifest.scenes ?? []).map((scene) => [scene.id, scene]));
  return (summary.scenes ?? []).map((scene) => {
    const manifestScene = manifestScenes.get(scene.id) ?? {};
    const maskStats = summarizeMaskManifest(resolvePath(manifestRoot, manifestScene.masks));
    const curve = existsSync(scene.curve) ? readJson(scene.curve) : null;
    const voteSummary = curve?.vote_summary ?? {};
    return {
      suite: "semantic-smoke",
      scene_id: scene.id,
      scene_label: scene.label,
      variant_id: "default",
      variant_label: "default",
      mask_policy: semanticMaskPolicy(scene.id),
      source_summary: semanticSummaryJson,
      curve: scene.curve,
      passed: scene.passed,
      gaussians: scene.gaussians ?? null,
      slots: scene.slots ?? null,
      frames: maskStats.frames ?? voteSummary.frames ?? null,
      masks: maskStats.masks,
      mask_pixels: maskStats.mask_pixels,
      supervised_gaussians: voteSummary.supervised_gaussians ?? null,
      object_id_counts: null,
      initial_projection_loss: scene.initial_projection_loss ?? null,
      final_projection_loss: scene.final_projection_loss ?? null,
      final_assignment_confidence: scene.final_assignment_confidence ?? null,
      final_ari_to_initial: scene.final_ari_to_initial ?? null,
      final_spatial_compactness_score: scene.final_spatial_compactness_score ?? null,
      final_object_emergence_score: scene.final_object_emergence_score ?? null,
      render_occlusion_effect_score: scene.final_render_occlusion_effect_score ?? null,
      heldout_final_projection_loss: scene.heldout?.final_projection_loss ?? null,
      heldout_supervised_gaussians: scene.heldout?.supervised_gaussians ?? null,
      heldout_render_occlusion_effect_score: scene.heldout?.render_occlusion_effect_score ?? null,
    };
  });
}

function flattenVariantRows(summary) {
  return (summary.variants ?? []).map((variant) => {
    const perVariantSummary =
      variant.paths?.summary && existsSync(variant.paths.summary) ? readJson(variant.paths.summary) : null;
    return {
      suite: "splatfacto-safe2000-variants",
      scene_id: "lego-splatfacto-safe-2000",
      scene_label: "Lego Splatfacto safe-2000",
      variant_id: variant.id,
      variant_label: variant.label,
      mask_policy: variant.mask_policy,
      source_summary: variant.paths?.summary ?? null,
      curve: variant.paths?.curve ?? null,
      passed: variant.passed,
      gaussians: perVariantSummary?.registration?.gaussians ?? null,
      slots: variant.slots ?? null,
      frames: variant.frames ?? null,
      masks: variant.masks ?? null,
      mask_pixels: variant.mask_pixels ?? null,
      supervised_gaussians: variant.supervised_gaussians ?? null,
      object_id_counts: variant.object_id_counts ?? null,
      initial_projection_loss: variant.registration_initial_loss ?? null,
      final_projection_loss: variant.final_projection_loss ?? null,
      final_assignment_confidence: variant.assignment_confidence ?? null,
      final_ari_to_initial: variant.stability_ari ?? null,
      final_spatial_compactness_score: perVariantSummary?.emergence?.spatial_compactness_score ?? null,
      final_object_emergence_score: variant.curve_object_emergence_score ?? variant.object_emergence_score ?? null,
      render_occlusion_effect_score: variant.render_occlusion_effect_score ?? null,
      heldout_final_projection_loss:
        variant.heldout_projection_loss ??
        perVariantSummary?.curve?.final_heldout_projection_loss ??
        null,
      heldout_supervised_gaussians:
        variant.heldout_supervised_gaussians ??
        perVariantSummary?.curve?.final_heldout_supervised_gaussians ??
        null,
      heldout_render_occlusion_effect_score:
        variant.heldout_render_occlusion_effect_score ??
        perVariantSummary?.curve?.final_heldout_render_occlusion_effect_score ??
        null,
    };
  });
}

function evaluateStageGates(rows) {
  const splatfactoScenes = new Set(
    rows.filter((row) => row.suite === "splatfacto-scenes").map((row) => row.scene_id),
  );
  const renderRows = rows.filter((row) => numberValue(row.render_occlusion_effect_score) > Number.NEGATIVE_INFINITY);
  const heldoutRows = rows.filter((row) => row.heldout_final_projection_loss !== null && row.heldout_final_projection_loss !== undefined);
  const bestRender = renderRows.reduce(
    (best, row) => Math.max(best, numberValue(row.render_occlusion_effect_score)),
    Number.NEGATIVE_INFINITY,
  );
  return {
    smoke: gate([
      check("rows_present", rows.length >= 3, rows.length, ">= 3"),
      check("all_rows_passed", rows.every((row) => row.passed !== false), rows.filter((row) => row.passed === false).length, "0 failed rows"),
      check("render_metric_rows", renderRows.length >= 3, renderRows.length, ">= 3"),
    ]),
    candidate: gate([
      check("real_splatfacto_scenes", splatfactoScenes.size >= 2, splatfactoScenes.size, ">= 2"),
      check("best_render_effect", bestRender >= 0.1, bestRender, ">= 0.100000"),
    ]),
    paper: gate([
      check("real_splatfacto_scenes", splatfactoScenes.size >= 3, splatfactoScenes.size, ">= 3"),
      check("heldout_eval_rows", heldoutRows.length >= 3, heldoutRows.length, ">= 3"),
      check("failure_report_available", true, failureReport, "written"),
    ]),
  };
}

function gate(checks) {
  return {
    passed: checks.every((item) => item.passed),
    checks,
  };
}

function check(name, passed, actual, expected) {
  return { name, passed: Boolean(passed), actual, expected };
}

function semanticMaskPolicy(sceneId) {
  if (sceneId === "plush-semantic") return "deterministic 2D color semantic masks";
  if (sceneId === "lego-alpha-proxy") return "NeRF Lego RGBA/color proxy masks";
  if (sceneId === "lego-splatfacto-smoke") return "SAM 2f / 8 slots on Splatfacto smoke";
  return "manifest masks";
}

function reportPath(value, fallback) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && typeof value.output === "string") return value.output;
  return fallback;
}

function summarizeMaskManifest(path) {
  if (!path || !existsSync(path)) {
    return { frames: null, masks: null, mask_pixels: null };
  }
  const manifest = readJson(path);
  const frames = manifest.frames ?? [];
  let masks = 0;
  let maskPixels = 0;
  for (const frame of frames) {
    for (const mask of frame.masks ?? []) {
      masks += 1;
      maskPixels += Number(mask.area ?? 0);
    }
  }
  return { frames: frames.length, masks, mask_pixels: maskPixels };
}

function benchmarkRoot(manifestPath, manifest) {
  const rootValue = manifest.root ?? ".";
  return resolve(dirname(manifestPath), rootValue);
}

function resolvePath(root, value) {
  if (!value) return null;
  return resolve(root, value);
}

function bestByScene(rows, field) {
  const best = new Map();
  for (const row of rows) {
    const current = best.get(row.scene_id);
    if (!current || numberValue(row[field]) > numberValue(current[field])) {
      best.set(row.scene_id, row);
    }
  }
  return [...best.values()].map((row) => ({
    scene_id: row.scene_id,
    variant_id: row.variant_id,
    value: row[field] ?? null,
  }));
}

function rankBy(rows, field) {
  return [...rows]
    .sort((left, right) => numberValue(right[field]) - numberValue(left[field]))
    .map((row, index) => ({
      rank: index + 1,
      scene_id: row.scene_id,
      variant_id: row.variant_id,
      value: row[field] ?? null,
    }));
}

function renderCsv(rows) {
  const headers = [
    "suite",
    "scene_id",
    "variant_id",
    "frames",
    "masks",
    "mask_pixels",
    "gaussians",
    "slots",
    "supervised_gaussians",
    "object_id_counts",
    "initial_projection_loss",
    "final_projection_loss",
    "final_assignment_confidence",
    "final_ari_to_initial",
    "final_object_emergence_score",
    "render_occlusion_effect_score",
    "heldout_final_projection_loss",
    "heldout_supervised_gaussians",
    "heldout_render_occlusion_effect_score",
  ];
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvCell(cellValue(row, header))).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function renderMarkdown(summary) {
  const lines = [
    "# ObjGauss Cross-Scene Emergence Benchmark",
    "",
    `Generated: ${summary.generated_at}`,
    "",
    "| Suite | Scene | Variant | Frames | Masks | Slots | Supervised | Object IDs | ARI | OES | Render effect | Held-out loss |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
  ];
  for (const row of summary.rows) {
    lines.push(
      `| ${row.suite} | ${row.scene_id} | ${row.variant_id} | ${empty(row.frames)} | ${empty(row.masks)} | ${empty(row.slots)} | ${empty(row.supervised_gaussians)} | ${objectCounts(row.object_id_counts)} | ${formatNumber(row.final_ari_to_initial)} | ${formatNumber(row.final_object_emergence_score)} | ${formatNumber(row.render_occlusion_effect_score)} | ${formatNumber(row.heldout_final_projection_loss)} |`,
    );
  }
  lines.push("", `Semantic report: ${summary.paths.semantic_report}`);
  if (summary.paths.scene_report) {
    lines.push(`Scene report: ${summary.paths.scene_report}`);
  }
  lines.push(`Variant report: ${summary.paths.variant_report}`);
  lines.push(`Failure report: ${summary.paths.failure_report}`);
  lines.push("");
  lines.push("## Stage Gates");
  lines.push("");
  lines.push("| Gate | Passed | Failed checks |");
  lines.push("| --- | --- | --- |");
  for (const [name, gateValue] of Object.entries(summary.stage_gates ?? {})) {
    const failed = gateValue.checks.filter((item) => !item.passed).map((item) => item.name).join(", ") || "-";
    lines.push(`| ${name} | ${gateValue.passed ? "yes" : "no"} | ${failed} |`);
  }
  return `${lines.join("\n")}\n`;
}

function renderHtml(summary) {
  const rows = summary.rows
    .map(
      (row) => `<tr><td>${escapeHtml(row.suite)}</td><td>${escapeHtml(row.scene_id)}</td><td>${escapeHtml(row.variant_id)}</td><td>${empty(row.frames)}</td><td>${empty(row.masks)}</td><td>${empty(row.slots)}</td><td>${empty(row.supervised_gaussians)}</td><td>${escapeHtml(objectCounts(row.object_id_counts))}</td><td>${formatNumber(row.final_ari_to_initial)}</td><td>${formatNumber(row.final_object_emergence_score)}</td><td>${formatNumber(row.render_occlusion_effect_score)}</td><td>${formatNumber(row.heldout_final_projection_loss)}</td></tr>`,
    )
    .join("\n");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ObjGauss Cross-Scene Emergence Benchmark</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 32px; color: #1f2937; }
    table { border-collapse: collapse; width: 100%; font-size: 14px; }
    th, td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }
    th { background: #f3f4f6; }
    td:nth-child(n+4) { text-align: right; }
    td:nth-child(8) { text-align: left; font-family: ui-monospace, monospace; font-size: 12px; }
  </style>
</head>
<body>
  <h1>ObjGauss Cross-Scene Emergence Benchmark</h1>
  <p>Generated: ${escapeHtml(summary.generated_at)}</p>
  <table>
    <thead><tr><th>Suite</th><th>Scene</th><th>Variant</th><th>Frames</th><th>Masks</th><th>Slots</th><th>Supervised</th><th>Object IDs</th><th>ARI</th><th>OES</th><th>Render effect</th><th>Held-out loss</th></tr></thead>
    <tbody>
${rows}
    </tbody>
  </table>
  <p>Semantic report: ${escapeHtml(summary.paths.semantic_report)}</p>
  ${summary.paths.scene_report ? `<p>Scene report: ${escapeHtml(summary.paths.scene_report)}</p>` : ""}
  <p>Variant report: ${escapeHtml(summary.paths.variant_report)}</p>
  <p>Failure report: ${escapeHtml(summary.paths.failure_report)}</p>
</body>
</html>
`;
}

function renderFailureReport(summary) {
  const lines = [
    "# ObjGauss Cross-Scene Failure Report",
    "",
    `Generated: ${summary.generated_at}`,
    `Overall passed: \`${String(summary.passed)}\``,
    "",
    "## Stage Gates",
    "",
    "| Gate | Passed | Check | Actual | Expected |",
    "| --- | --- | --- | ---: | --- |",
  ];
  for (const [gateName, gateValue] of Object.entries(summary.stage_gates ?? {})) {
    for (const item of gateValue.checks) {
      lines.push(
        `| ${gateName} | ${item.passed ? "yes" : "no"} | ${item.name} | ${escapeMarkdown(String(formatReportValue(item.actual)))} | ${escapeMarkdown(String(item.expected))} |`,
      );
    }
  }
  lines.push("", "## Failed Rows", "");
  const failedRows = summary.rows.filter((row) => row.passed === false);
  if (failedRows.length === 0) {
    lines.push("No failed smoke rows.");
  } else {
    for (const row of failedRows) {
      lines.push(`- ${row.suite}/${row.scene_id}/${row.variant_id}`);
    }
  }
  lines.push("", "## Paper-Readiness Gap", "");
  const paper = summary.stage_gates?.paper;
  const failedPaper = paper?.checks?.filter((item) => !item.passed) ?? [];
  if (failedPaper.length === 0) {
    lines.push("Paper gate passed.");
  } else {
    for (const item of failedPaper) {
      lines.push(`- ${item.name}: actual ${formatReportValue(item.actual)}, expected ${item.expected}`);
    }
  }
  return `${lines.join("\n")}\n`;
}

function printSummary(summary) {
  console.log(`summary=${summary.paths.summary}`);
  console.log(`rows=${summary.rows.length}`);
  for (const row of summary.rows) {
    console.log(
      `row=${row.suite}/${row.scene_id}/${row.variant_id} ari=${formatNumber(row.final_ari_to_initial)} oes=${formatNumber(row.final_object_emergence_score)} render=${formatNumber(row.render_occlusion_effect_score)}`,
    );
  }
  const bestRender = summary.rankings.by_render_occlusion_effect_score[0];
  if (bestRender) {
    console.log(
      `best_render=${bestRender.scene_id}/${bestRender.variant_id} value=${formatNumber(bestRender.value)}`,
    );
  }
  console.log(`summary_status=${summary.passed ? "passed" : "failed"}`);
  for (const [name, gateValue] of Object.entries(summary.stage_gates ?? {})) {
    console.log(`stage_gate=${name} passed=${gateValue.passed ? "true" : "false"}`);
  }
}

function cellValue(row, field) {
  if (field === "object_id_counts") return objectCounts(row.object_id_counts);
  return row[field];
}

function objectCounts(value) {
  return Array.isArray(value) ? value.join("/") : "";
}

function empty(value) {
  return value === null || value === undefined ? "" : String(value);
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
    else if (value === "--skip-semantic") parsed.skipSemantic = true;
    else if (value === "--skip-scenes") parsed.skipScenes = true;
    else if (value === "--skip-variants") parsed.skipVariants = true;
    else if (value === "--refresh-sam") parsed.refreshSam = true;
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
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(6) : "";
}

function formatReportValue(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(6) : value;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char];
  });
}

function escapeMarkdown(value) {
  return value.replaceAll("|", "\\|");
}

function run(command) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(command[0], command.slice(1), {
      env: process.env,
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolveRun();
      } else {
        reject(new Error(`${command[0]} exited with code ${code}`));
      }
    });
  });
}
