import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const args = parseArgs(process.argv.slice(2));
const profile = String(args.profile ?? "ci");
const outputDir = String(
  args.outputDir ?? args["output-dir"] ?? `/tmp/objgauss-renderer-acceptance-${profile}`,
);
const dryRun = flagEnabled(args.dryRun ?? args["dry-run"]);
const skipBuild = flagEnabled(args.skipBuild ?? args["skip-build"]);
const skipNativeRoute = flagEnabled(args.skipNativeRoute ?? args["skip-native-route"]);
const skipSplatIndexMapping = flagEnabled(
  args.skipSplatIndexMapping ?? args["skip-splat-index-mapping"],
);
const skipWebGpuTileSmoke = flagEnabled(args.skipWebGpuTileSmoke ?? args["skip-webgpu-tile-smoke"]);
const nativePort = String(args.nativePort ?? args["native-port"] ?? "5395");
const trainedPort = String(args.trainedPort ?? args["trained-port"] ?? "5395");
const noShAssets = String(
  args.noShAssets ??
    args["no-sh-assets"] ??
    "nerf-lego-alpha-closure-local,plush-semantic-closure-local",
);

const profileSpec = createProfileSpec(profile);
const report = {
  status: dryRun ? "dry-run" : "running",
  generatedAt: new Date().toISOString(),
  profile,
  outputDir,
  dryRun,
  decision: profileSpec.decision,
  steps: [],
};

try {
  for (const [label, command] of profileSpec.steps) {
    console.log(`\n=== ${label} ===`);
    if (dryRun) {
      console.log(command.join(" "));
      report.steps.push({ label, command: command.join(" "), exitCode: null, durationMs: 0 });
      continue;
    }
    const result = await run(label, command);
    report.steps.push({
      label,
      command: command.join(" "),
      exitCode: result.exitCode,
      durationMs: result.durationMs,
    });
  }
  report.status = dryRun ? "dry-run" : "passed";
  writeReport(outputDir, report);
} catch (error) {
  report.status = "failed";
  report.error = error?.message ?? String(error);
  if (error?.result) {
    report.steps.push({
      label: error.result.label,
      command: error.result.command.join(" "),
      exitCode: error.result.exitCode,
      durationMs: error.result.durationMs,
    });
  }
  writeReport(outputDir, report);
  throw error;
}

console.log(
  `\nacceptance_renderer_profile=${report.status} profile=${JSON.stringify(profile)} ` +
    `outputDir=${JSON.stringify(outputDir)} steps=${report.steps.length}`,
);

function createProfileSpec(name) {
  if (name === "ci") {
    return {
      decision: {
        defaultCiRequirement: true,
        includesTrainedShHeavySample: false,
        sparkCommercialRouteDefaultCi: false,
        reason:
          "fresh-clone CI must not require the local nerf-lego-trained-output-local sample",
      },
      steps: [
        ...(skipBuild ? [] : [["Build viewer", ["npm", "run", "build"]]]),
        ...(skipWebGpuTileSmoke
          ? []
          : [["WebGPU tile smoke", ["npm", "run", "audit:webgpu-tile-smoke"]]]),
        ...(skipSplatIndexMapping
          ? []
          : [
              [
                "No-SH public sample index mapping",
                [
                  "npm",
                  "run",
                  "audit:splat-index-mapping",
                  "--",
                  "--assets",
                  noShAssets,
                  "--output-dir",
                  path.join(outputDir, "splat-index-mapping"),
                ],
              ],
            ]),
        ...(skipNativeRoute
          ? []
          : [
              [
                "Spark no-SH native object mask route",
                ["npm", "run", "audit:spark-native-mask-gate", "--", "--port", nativePort],
              ],
            ]),
      ],
    };
  }

  if (name === "product") {
    return {
      decision: {
        defaultCiRequirement: false,
        includesTrainedShHeavySample: true,
        sparkCommercialRouteDefaultCi: false,
        reason:
          "product/demo review should require the local SH-heavy trained sample and fail fast when it is missing",
      },
      steps: [
        [
          "Spark commercial route acceptance",
          [
            "npm",
            "run",
            "acceptance:spark-commercial-route",
            "--",
            "--native-port",
            nativePort,
            "--trained-port",
            trainedPort,
            "--output-dir",
            path.join(outputDir, "spark-commercial-route"),
            ...(skipBuild ? ["--skip-build"] : []),
          ],
        ],
      ],
    };
  }

  throw new Error(`unknown renderer acceptance profile: ${name}; expected ci or product`);
}

function writeReport(outputDirPath, summary) {
  mkdirSync(outputDirPath, { recursive: true });
  writeFileSync(path.join(outputDirPath, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(path.join(outputDirPath, "summary.md"), renderMarkdown(summary));
}

function renderMarkdown(summary) {
  return [
    "# Renderer Acceptance Profile",
    "",
    `- Status: ${summary.status}`,
    `- Profile: ${summary.profile}`,
    `- Generated: ${summary.generatedAt}`,
    `- Default CI requirement: ${yesNo(summary.decision.defaultCiRequirement)}`,
    `- Includes trained SH-heavy sample: ${yesNo(summary.decision.includesTrainedShHeavySample)}`,
    `- Spark commercial route default CI: ${yesNo(summary.decision.sparkCommercialRouteDefaultCi)}`,
    `- Reason: ${summary.decision.reason}`,
    "",
    "## Steps",
    "",
    "| Step | Exit | Duration ms | Command |",
    "| --- | ---: | ---: | --- |",
    ...summary.steps.map(
      (step) =>
        `| ${escapeMarkdown(step.label)} | ${step.exitCode ?? ""} | ${step.durationMs ?? 0} | \`${escapeMarkdown(step.command)}\` |`,
    ),
    "",
  ].join("\n");
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

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function run(label, command) {
  return new Promise((resolve, reject) => {
    const startedAt = performance.now();
    const child = spawn(command[0], command.slice(1), {
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("close", (code) => {
      const result = {
        label,
        command,
        exitCode: code,
        durationMs: Math.round(performance.now() - startedAt),
      };
      if (code === 0) {
        resolve(result);
      } else {
        const error = new Error(`${command.join(" ")} exited with ${code}`);
        error.result = result;
        reject(error);
      }
    });
  });
}

function yesNo(value) {
  return value ? "yes" : "no";
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
}
