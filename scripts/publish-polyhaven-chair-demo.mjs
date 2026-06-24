import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const DEFAULT_SOURCE_DIR =
  "outputs/assets/gaussians/polyhaven-chair-splatfacto-smoke-sam8f-slots6-benchmark";
const DEFAULT_PUBLIC_DIR = "public/samples";
const DEFAULT_PUBLIC_NAME = "polyhaven_chair_demo";
const DEFAULT_OUTPUT_DIR = "/tmp/objgauss-polyhaven-chair-demo-publish";

const args = parseArgs(process.argv.slice(2));
const sourceDir = String(args.sourceDir ?? args["source-dir"] ?? DEFAULT_SOURCE_DIR);
const publicDir = String(args.publicDir ?? args["public-dir"] ?? DEFAULT_PUBLIC_DIR);
const publicName = String(args.publicName ?? args["public-name"] ?? DEFAULT_PUBLIC_NAME);
const outputDir = String(args.outputDir ?? args["output-dir"] ?? DEFAULT_OUTPUT_DIR);
const sourceSplat = String(args.sourceSplat ?? args["source-splat"] ?? path.join(sourceDir, "gaussians.splat"));
const sourceObjectPly = String(
  args.sourceObjectPly ?? args["source-object-ply"] ?? path.join(sourceDir, "object_aware_gaussians.ply"),
);
const sourceManifest = String(
  args.sourceManifest ?? args["source-manifest"] ?? path.join(sourceDir, "training-output-manifest.json"),
);

const publicSplat = path.join(publicDir, `${publicName}.splat`);
const publicObjectPly = path.join(publicDir, `${publicName}_objects.ply`);
const failures = [];

if (!existsSync(sourceSplat)) failures.push(`missing source splat: ${sourceSplat}`);
if (!existsSync(sourceObjectPly)) failures.push(`missing source object PLY: ${sourceObjectPly}`);

const summary = {
  mode: "polyhaven-chair-commercial-demo-publish-v1",
  generatedAt: new Date().toISOString(),
  assetId: "polyhaven-chair-commercial-demo-local",
  source: {
    asset: "Poly Haven School Chair 01",
    sourceUrl: "https://polyhaven.com/a/SchoolChair_01",
    license: "CC0",
    splat: sourceSplat,
    objectPly: sourceObjectPly,
    manifest: sourceManifest,
  },
  public: {
    directory: publicDir,
    publicName,
    splat: publicSplat,
    objectPly: publicObjectPly,
  },
  copied: false,
  files: {},
  failures,
  prepare: [
    "uv run objgauss assets pull polyhaven-school-chair-nerf",
    "npm run train:splatfacto:smoke -- --run --asset-id polyhaven-school-chair-nerf --dataset outputs/assets/training/polyhaven-school-chair-nerf --output-root outputs/training/polyhaven-chair-splatfacto-smoke --experiment chair-splatfacto-smoke --timestamp smoke-cuda --export-dir outputs/training/polyhaven-chair-splatfacto-smoke/export-smoke-cuda --object-field-dir outputs/training/polyhaven-chair-splatfacto-smoke/object-field-sam --sam-manifest outputs/masks/polyhaven-chair-sam-smoke/mask-manifest.json --data-parser blender-data --iterations 100 --steps-per-save 100 --vis tensorboard --cache-images cpu --camera-res-scale-factor 0.5 --cuda-home /tmp/objgauss-cuda13 --max-jobs 2 --device cuda --sam-max-frames 8 --sam-max-masks-per-frame 6 --sam-min-area 64 --sam-max-area-fraction 0.75 --slots 6 --object-iterations 80 --skip-benchmark",
  ],
};

if (failures.length === 0) {
  mkdirSync(publicDir, { recursive: true });
  copyFileSync(sourceSplat, publicSplat);
  copyFileSync(sourceObjectPly, publicObjectPly);
  summary.copied = true;
  summary.files = {
    splat: fileStatus(publicSplat),
    objectPly: fileStatus(publicObjectPly),
    sourceManifest: existsSync(sourceManifest) ? JSON.parse(readFileSync(sourceManifest, "utf8")) : null,
  };
}

writeReport(outputDir, summary);

if (!summary.copied) {
  console.error(`polyhaven_chair_demo_publish=failed report=${outputDir}/summary.md`);
  for (const failure of failures) console.error(`failure=${failure}`);
  process.exit(1);
}

console.log(
  `polyhaven_chair_demo_publish=passed asset=${JSON.stringify(summary.assetId)} ` +
    `splat=${JSON.stringify(publicSplat)} objectPly=${JSON.stringify(publicObjectPly)} ` +
    `splatBytes=${summary.files.splat.size} objectPlyBytes=${summary.files.objectPly.size} ` +
    `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
);

function fileStatus(filePath) {
  if (!filePath || !existsSync(filePath)) return { path: filePath, exists: false, size: 0 };
  const stat = statSync(filePath);
  return {
    path: filePath,
    exists: true,
    size: stat.size,
  };
}

function writeReport(outputDirPath, summary) {
  mkdirSync(outputDirPath, { recursive: true });
  writeFileSync(path.join(outputDirPath, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(path.join(outputDirPath, "summary.md"), renderMarkdown(summary));
}

function renderMarkdown(summary) {
  const lines = [
    "# Poly Haven Chair Commercial Demo Publish",
    "",
    `- Status: ${summary.copied ? "passed" : "failed"}`,
    `- Generated: ${summary.generatedAt}`,
    `- Asset id: ${summary.assetId}`,
    `- Source: ${summary.source.asset}`,
    `- License: ${summary.source.license}`,
    `- Source URL: ${summary.source.sourceUrl}`,
    "",
    "## Files",
    "",
    `- Source splat: ${summary.source.splat}`,
    `- Source object PLY: ${summary.source.objectPly}`,
    `- Public splat: ${summary.public.splat}`,
    `- Public object PLY: ${summary.public.objectPly}`,
    `- Splat bytes: ${summary.files?.splat?.size ?? 0}`,
    `- Object PLY bytes: ${summary.files?.objectPly?.size ?? 0}`,
    "",
  ];
  if (summary.failures.length > 0) {
    lines.push("## Failures", "", ...summary.failures.map((failure) => `- ${failure}`), "");
    lines.push("## Prepare", "", ...summary.prepare.map((command) => `- \`${command}\``), "");
  }
  return lines.join("\n");
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
