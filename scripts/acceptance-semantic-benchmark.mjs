import { spawn } from "node:child_process";

const args = process.argv.slice(2);
const manifestIndex = args.indexOf("--manifest");
const outputIndex = args.indexOf("--output-dir");
const manifest =
  manifestIndex >= 0 && args[manifestIndex + 1]
    ? args[manifestIndex + 1]
    : "docs/benchmarks/semantic-smoke.json";
const outputDir =
  outputIndex >= 0 && args[outputIndex + 1]
    ? args[outputIndex + 1]
    : "/tmp/objgauss-semantic-smoke-suite";

await run([
  "uv",
  "run",
  "objgauss",
  "object-field",
  "emergence-benchmark",
  manifest,
  "--output-dir",
  outputDir,
  "--strict",
]);

console.log("\nacceptance_semantic_benchmark=passed");

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
