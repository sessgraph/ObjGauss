import { spawn } from "node:child_process";

const args = parseArgs(process.argv.slice(2));
const nativePort = String(args.nativePort ?? args["native-port"] ?? "5347");
const trainedPort = String(args.trainedPort ?? args["trained-port"] ?? "5348");
const skipBuild = flagEnabled(args.skipBuild ?? args["skip-build"]);

const steps = [
  ...(skipBuild ? [] : [["Build viewer", ["npm", "run", "build"]]]),
  [
    "Spark no-SH native object mask route",
    ["npm", "run", "audit:spark-native-mask-gate", "--", "--port", nativePort],
  ],
  [
    "Spark SH-heavy packed object mask route",
    ["npm", "run", "audit:spark-trained-route", "--", "--port", trainedPort],
  ],
];

for (const [label, command] of steps) {
  console.log(`\n=== ${label} ===`);
  await run(command);
}

console.log(
  `\nacceptance_spark_commercial_route=passed nativePort=${nativePort} trainedPort=${trainedPort}`,
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
