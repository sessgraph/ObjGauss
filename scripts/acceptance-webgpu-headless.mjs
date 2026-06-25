import { spawn } from "node:child_process";

const args = parseArgs(process.argv.slice(2));
const outputDir = String(
  args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-webgpu-headless-acceptance",
);
const port = String(args.port ?? "5395");
const assets = optionalString(args.assets ?? args.asset);
const webGpuFlags = String(args.webGpuFlags ?? args["webgpu-flags"] ?? "unsafe");
const skipBuild = flagEnabled(args.skipBuild ?? args["skip-build"]);
const skipTileSmoke = flagEnabled(args.skipTileSmoke ?? args["skip-tile-smoke"]);

const steps = [
  ...(skipBuild ? [] : [["Build viewer", ["npm", "run", "build"]]]),
  ...(skipTileSmoke
    ? []
    : [["WebGPU tile smoke", ["npm", "run", "audit:webgpu-tile-smoke"]]]),
  [
    "WebGPU headless offscreen object transition",
    [
      "npm",
      "run",
      "audit:webgpu-offscreen-readback",
      "--",
      "--port",
      port,
      "--output-dir",
      `${outputDir}/offscreen-readback`,
      "--webgpu-flags",
      webGpuFlags,
      ...(assets ? ["--assets", assets] : []),
    ],
  ],
];

for (const [label, command] of steps) {
  console.log(`\n=== ${label} ===`);
  await run(command);
}

console.log(
  `\nacceptance_webgpu_headless=passed outputDir=${JSON.stringify(outputDir)} ` +
    `port=${port} webGpuFlags=${JSON.stringify(webGpuFlags)} assets=${JSON.stringify(assets || "default")}`,
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
