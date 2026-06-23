import { spawn } from "node:child_process";

const args = new Set(process.argv.slice(2));
const pullAssets = args.has("--pull-assets");
const skipSemanticBenchmark = args.has("--skip-semantic-benchmark");

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
  ["Browser audit for closure cards", ["npm", "run", "audit:demo"]],
  ...(!skipSemanticBenchmark
    ? [["Semantic emergence benchmark suite", ["npm", "run", "acceptance:semantic"]]]
    : []),
];

for (const [label, command] of steps) {
  console.log(`\n=== ${label} ===`);
  await run(command);
}

console.log("\nacceptance_demo=passed");

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
