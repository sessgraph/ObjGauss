import { spawn } from "node:child_process";

const args = parseArgs(process.argv.slice(2));
const pullAssets = flagEnabled(args["pull-assets"]);
const skipSemanticBenchmark = flagEnabled(args["skip-semantic-benchmark"]);
const includeSparkCommercialRoute = flagEnabled(args["include-spark-commercial-route"]);
const sparkNativePort = optionalString(args["spark-native-port"]) || "5395";
const sparkTrainedPort = optionalString(args["spark-trained-port"]) || "5395";
const sparkRouteOutputDir =
  optionalString(args["spark-route-output-dir"]) ||
  "/tmp/objgauss-acceptance-demo-spark-commercial-route";
const skipSparkRouteBuild = flagEnabled(args["skip-spark-route-build"]);
const browserAuditMode = optionalString(args["browser-audit-mode"]) || "preview";
const browserAuditPort = optionalString(args["browser-audit-port"]) || "5395";
const browserAuditAssets = optionalString(args["browser-audit-assets"]);
const skipBrowserVisualResidual = flagEnabled(args["skip-browser-visual-residual"]);

const steps = [
  ...(pullAssets
    ? [
        ["Pull Plush sample", ["uv", "run", "objgauss", "assets", "pull", "plush-3dgs-local"]],
        ["Pull NeRF Lego dataset", ["uv", "run", "objgauss", "assets", "pull", "nerf-synthetic-lego"]],
      ]
    : []),
  [
    "Build Plush v1 closure",
    ["uv", "run", "objgauss", "demo", "v1-closure", "--iterations", "80"],
  ],
  ["Verify Plush v1 closure", ["uv", "run", "objgauss", "demo", "verify-v1-closure"]],
  [
    "Build Plush semantic closure",
    ["uv", "run", "objgauss", "demo", "plush-semantic-closure", "--iterations", "80"],
  ],
  [
    "Verify Plush semantic closure",
    ["uv", "run", "objgauss", "demo", "verify-plush-semantic-closure"],
  ],
  [
    "Build NeRF Lego alpha closure proxy",
    [
      "uv",
      "run",
      "objgauss",
      "demo",
      "lego-alpha-closure",
      "--max-frames",
      "12",
      "--sample-stride",
      "8",
      "--iterations",
      "120",
    ],
  ],
  [
    "Verify NeRF Lego alpha closure proxy",
    ["uv", "run", "objgauss", "demo", "verify-lego-alpha-closure"],
  ],
  ...(browserAuditMode === "preview"
    ? [["Build viewer for browser audit", ["npm", "run", "build"]]]
    : []),
  [
    "Browser audit for closure cards",
    [
      "npm",
      "run",
      "audit:demo",
      "--",
      "--server-mode",
      browserAuditMode,
      "--port",
      browserAuditPort,
      ...(browserAuditAssets ? ["--assets", browserAuditAssets] : []),
      ...(skipBrowserVisualResidual ? ["--skip-visual-residual"] : []),
    ],
  ],
  ...(includeSparkCommercialRoute
    ? [
        [
          "Spark commercial route acceptance",
          [
            "npm",
            "run",
            "acceptance:spark-commercial-route",
            "--",
            "--native-port",
            sparkNativePort,
            "--trained-port",
            sparkTrainedPort,
            "--output-dir",
            sparkRouteOutputDir,
            ...(skipSparkRouteBuild ? ["--skip-build"] : []),
          ],
        ],
      ]
    : []),
  ...(!skipSemanticBenchmark
    ? [["Semantic emergence benchmark suite", ["npm", "run", "acceptance:semantic"]]]
    : []),
];

for (const [label, command] of steps) {
  console.log(`\n=== ${label} ===`);
  await run(command);
}

console.log("\nacceptance_demo=passed");

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

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return "";
  const text = String(value).trim();
  return text || "";
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function run(command) {
  return new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), {
      stdio: "inherit",
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${command.join(" ")} exited with ${code}`));
      }
    });
  });
}
