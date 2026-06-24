import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";

const args = parseArgs(process.argv.slice(2));
const nativePort = String(args.nativePort ?? args["native-port"] ?? "5347");
const trainedPort = String(args.trainedPort ?? args["trained-port"] ?? "5348");
const outputDir = String(
  args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-spark-commercial-route",
);
const skipBuild = flagEnabled(args.skipBuild ?? args["skip-build"]);
const skipTrainedSampleAudit = flagEnabled(
  args.skipTrainedSampleAudit ?? args["skip-trained-sample-audit"],
);

const steps = [
  ...(skipBuild ? [] : [["Build viewer", ["npm", "run", "build"]]]),
  ...(skipTrainedSampleAudit
    ? []
    : [
        [
          "Spark trained SH-heavy sample availability",
          [
            "npm",
            "run",
            "audit:spark-trained-sample",
            "--",
            "--output-dir",
            `${outputDir}/trained-sample`,
          ],
        ],
      ]),
  [
    "Spark no-SH native object mask route",
    ["npm", "run", "audit:spark-native-mask-gate", "--", "--port", nativePort],
  ],
  [
    "Spark SH-heavy packed object mask route",
    ["npm", "run", "audit:spark-trained-route", "--", "--port", trainedPort],
  ],
];

const report = {
  status: "running",
  generatedAt: new Date().toISOString(),
  outputDir,
  ports: {
    native: nativePort,
    trained: trainedPort,
  },
  skipBuild,
  skipTrainedSampleAudit,
  steps: [],
  routes: {
    native: [],
    trained: [],
  },
  contract: {
    colorMode: "source-color",
    boundary: "hard-object-mask-no-reoptimize",
    proves: [
      "no-SH generated samples use Spark native compact .splat object masking",
      "SH-heavy trained samples preserve degree-3 SH through the packed route",
      "commercial source/original edit route stays explicit about hard object masks",
    ],
    doesNotProve: [
      "deletion inpainting",
      "post-delete reoptimization",
      "WebGPU visual fidelity",
      "arbitrary third-party .splat object metadata",
    ],
  },
};

try {
  for (const [label, command] of steps) {
    console.log(`\n=== ${label} ===`);
    const result = await run(label, command);
    report.steps.push({
      label,
      command: command.join(" "),
      exitCode: result.exitCode,
      durationMs: result.durationMs,
    });
    appendRouteResults(report, result.stdout);
  }
  report.status = "passed";
  writeReport(report);
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
    appendRouteResults(report, error.result.stdout);
  }
  writeReport(report);
  throw error;
}

console.log(
  `\nacceptance_spark_commercial_route=passed nativePort=${nativePort} trainedPort=${trainedPort} ` +
    `summary=${outputDir}/summary.json report=${outputDir}/summary.md`,
);

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
    let stdout = "";
    let stderr = "";
    const child = spawn(command[0], command.slice(1), {
      stdio: ["ignore", "pipe", "pipe"],
    });
    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      process.stdout.write(chunk);
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      process.stderr.write(chunk);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      const result = {
        label,
        command,
        exitCode: code,
        durationMs: Math.round(performance.now() - startedAt),
        stdout,
        stderr,
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

function appendRouteResults(report, stdout) {
  for (const line of stdout.split(/\r?\n/)) {
    const native = parseNativeRouteLine(line);
    if (native) {
      report.routes.native.push(native);
      continue;
    }
    const trained = parseTrainedRouteLine(line);
    if (trained) report.routes.trained.push(trained);
  }
}

function parseNativeRouteLine(line) {
  const match = line.match(
    /^native_mask_gate_asset=passed asset=(?<asset>".*?") source=(?<source>".*?") route=(?<route>".*?") visible=(?<visible>\d+)\/(?<base>\d+) objectMask=(?<objectMask>.*?) mesh=(?<mesh>.*?) visual=(?<visual>.*?) screenshot=(?<screenshot>.+)$/,
  );
  if (!match?.groups) return null;
  return {
    gate: "spark-native-mask-gate",
    asset: jsonText(match.groups.asset),
    source: jsonText(match.groups.source),
    route: jsonText(match.groups.route),
    visibleGaussians: Number(match.groups.visible),
    baseGaussians: Number(match.groups.base),
    objectMask: match.groups.objectMask,
    mesh: match.groups.mesh,
    visual: match.groups.visual,
    screenshot: match.groups.screenshot,
    boundary: "hard-object-mask-no-reoptimize",
  };
}

function parseTrainedRouteLine(line) {
  const match = line.match(
    /^spark_trained_route_asset=passed asset=(?<asset>".*?") initial=(?<initial>.*?) delete=(?<delete>.*?) spark=(?<spark>.*?) visible=(?<visible>\d+)\/(?<base>\d+) objectMask=(?<objectMask>.*?) shRest=(?<shRest>.*?) screenshot=(?<screenshot>.+)$/,
  );
  if (!match?.groups) return null;
  return {
    gate: "spark-trained-route",
    asset: jsonText(match.groups.asset),
    initial: splitQuotedTuple(match.groups.initial),
    delete: splitQuotedTuple(match.groups.delete),
    spark: splitQuotedTuple(match.groups.spark),
    visibleGaussians: Number(match.groups.visible),
    baseGaussians: Number(match.groups.base),
    objectMask: match.groups.objectMask,
    shRest: splitSimpleTuple(match.groups.shRest),
    screenshot: match.groups.screenshot,
  };
}

function writeReport(report) {
  mkdirSync(outputDir, { recursive: true });
  const summaryPath = `${outputDir}/summary.json`;
  const markdownPath = `${outputDir}/summary.md`;
  writeFileSync(summaryPath, `${JSON.stringify(report, null, 2)}\n`);
  writeFileSync(markdownPath, renderMarkdown(report));
}

function renderMarkdown(report) {
  const lines = [
    "# Spark Commercial Route Acceptance",
    "",
    `- Status: ${report.status}`,
    `- Generated: ${report.generatedAt}`,
    `- Native port: ${report.ports.native}`,
    `- Trained port: ${report.ports.trained}`,
    "",
    "## Step Results",
    "",
    "| Step | Exit | Duration ms |",
    "| --- | ---: | ---: |",
    ...report.steps.map((step) => (
      `| ${escapeMarkdown(step.label)} | ${step.exitCode} | ${step.durationMs} |`
    )),
    "",
    "## Native no-SH Routes",
    "",
    "| Asset | Source | Route | Visible / Base | Boundary | Screenshot |",
    "| --- | --- | --- | ---: | --- | --- |",
    ...report.routes.native.map((route) => (
      `| ${escapeMarkdown(route.asset)} | ${escapeMarkdown(route.source)} | ${escapeMarkdown(route.route)} | ` +
      `${route.visibleGaussians} / ${route.baseGaussians} | ${escapeMarkdown(route.boundary)} | ${escapeMarkdown(route.screenshot)} |`
    )),
    "",
    "## SH-heavy Routes",
    "",
    "| Asset | Initial | Delete | Spark source | Visible / Base | SH rest | Screenshot |",
    "| --- | --- | --- | --- | ---: | --- | --- |",
    ...report.routes.trained.map((route) => (
      `| ${escapeMarkdown(route.asset)} | ${escapeMarkdown(route.initial.join(" / "))} | ` +
      `${escapeMarkdown(route.delete.join(" / "))} | ${escapeMarkdown(route.spark.join(" / "))} | ` +
      `${route.visibleGaussians} / ${route.baseGaussians} | ${escapeMarkdown(route.shRest.join(" / "))} | ${escapeMarkdown(route.screenshot)} |`
    )),
    "",
    "## Contract Boundary",
    "",
    `- Color mode: ${report.contract.colorMode}`,
    `- Source preview boundary: ${report.contract.boundary}`,
    "",
    "Proves:",
    ...report.contract.proves.map((entry) => `- ${entry}`),
    "",
    "Does not prove:",
    ...report.contract.doesNotProve.map((entry) => `- ${entry}`),
    "",
  ];
  if (report.error) {
    lines.push("## Error", "", report.error, "");
  }
  return lines.join("\n");
}

function jsonText(text) {
  return JSON.parse(text);
}

function splitQuotedTuple(text) {
  return text.split(":").map((entry) => jsonText(entry));
}

function splitSimpleTuple(text) {
  return text.split(":").map((entry) => {
    if (entry.startsWith("\"")) return jsonText(entry);
    return entry;
  });
}

function escapeMarkdown(value) {
  return String(value ?? "").replaceAll("|", "\\|");
}
