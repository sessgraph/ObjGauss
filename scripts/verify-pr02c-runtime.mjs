import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REQUIRED_RUNTIME = {
  python: "3.10.20",
  torch_distribution: "2.13.0",
  torch_runtime: "2.13.0+cu130",
  torch_cuda: "13.0",
};
const DISPLAY_RESERVE = 1024 ** 3;
const TRAINING_CAP = 12 * 1024 ** 3;

function sha256Bytes(payload) {
  return createHash("sha256").update(payload).digest("hex");
}

function sha256File(path) {
  return sha256Bytes(readFileSync(path));
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function stableBytes(value) {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableBytes(item)).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${stableBytes(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function packageTreeSha256() {
  const paths = [
    resolve(ROOT, "learning/pyproject.toml"),
    ...readdirSync(resolve(ROOT, "learning"))
      .filter((name) => name.endsWith(".json"))
      .sort()
      .map((name) => resolve(ROOT, "learning", name)),
    ...readdirSync(resolve(ROOT, "learning/src/objgauss_learning"))
      .filter((name) => name.endsWith(".py"))
      .sort()
      .map((name) => resolve(ROOT, "learning/src/objgauss_learning", name)),
  ];
  const entries = Object.fromEntries(paths
    .map((path) => [relative(ROOT, path).replaceAll("\\", "/"), sha256File(path)])
    .sort(([left], [right]) => left.localeCompare(right)));
  return sha256Bytes(`${stableBytes(entries)}\n`);
}

function atomicJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}`;
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
  try {
    renameSync(temporary, path);
  } finally {
    rmSync(temporary, { force: true });
  }
}

function parseArgs(argv) {
  const values = {
    report: "generated/pr02c/runtime/report.json",
    output: "generated/pr02c/runtime/verification-report.json",
  };
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    assert(value, `missing value for ${name}`);
    if (name === "--report") values.report = value;
    else if (name === "--output") values.output = value;
    else throw new Error(`unknown argument: ${name}`);
  }
  return values;
}

function repoPath(raw, prefix) {
  assert(!raw.startsWith("/"), `path must be repository-relative: ${raw}`);
  const path = resolve(ROOT, raw);
  assert(path.startsWith(`${ROOT}/`), `path escapes repository root: ${raw}`);
  if (prefix) assert(raw.startsWith(`${prefix}/`), `path must be below ${prefix}/: ${raw}`);
  return path;
}

const args = parseArgs(process.argv.slice(2));
const reportPath = repoPath(args.report, "generated/pr02c");
const outputPath = repoPath(args.output, "generated/pr02c");
const report = readJson(reportPath);
const gitHead = spawnSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" });
assert.equal(gitHead.status, 0, gitHead.stderr);
const manifestPath = resolve(ROOT, "learning/runtime-manifest.json");
const manifest = readJson(manifestPath);
const lockPath = resolve(ROOT, manifest.inputs.runtime_lock.path);
const gridPath = resolve(ROOT, manifest.inputs.hyperparameter_grid.path);
const checks = [];

function check(checkId, predicate) {
  assert(predicate, `PR-02C runtime verification failed: ${checkId}`);
  checks.push({ check_id: checkId, status: "supported" });
}

check("report-kind-version",
  report.report_kind === "objgauss.pr02c-runtime-probe"
  && report.report_version === "0.1.0");
check("runtime-verdict",
  report.verdict?.status === "supported"
  && report.verdict?.reason_code === "all_c0_gates_passed");
check("source-commit",
  /^[a-f0-9]{40,64}$/.test(report.producer?.source_commit ?? "")
  && report.producer.source_commit === gitHead.stdout.trim()
  && report.source?.worktree_clean === true
  && report.source?.head_matches_source_commit === true);
check("source-tree", report.producer?.source_tree_sha256 === packageTreeSha256());
check("runtime-versions", stableBytes(report.runtime) === stableBytes(REQUIRED_RUNTIME));
check("manifest-lineage", report.inputs?.runtime_manifest_sha256 === sha256File(manifestPath));
check("lock-lineage",
  report.inputs?.runtime_lock_sha256 === sha256File(lockPath)
  && manifest.inputs.runtime_lock.sha256 === sha256File(lockPath));
check("grid-lineage",
  report.inputs?.hyperparameter_grid_sha256 === sha256File(gridPath)
  && manifest.inputs.hyperparameter_grid.sha256 === sha256File(gridPath));
check("simulator-isolation",
  report.isolation?.forbidden_distributions_absent === true
  && report.isolation?.forbidden_modules_absent === true
  && report.isolation?.forbidden_modules_loaded === false);
check("runtime-offline", report.isolation?.network_policy === "offline-during-runtime");
check("explicit-offline-mode", report.isolation?.explicit_offline_mode === true);
check("gpu-display-reserve",
  report.gpu_probe?.status === "supported"
  && report.gpu_probe?.minimum_free_bytes >= DISPLAY_RESERVE
  && report.gpu_probe?.display_reserve_bytes === DISPLAY_RESERVE);
check("gpu-training-cap",
  report.gpu_probe?.training_allocation_cap_bytes > 0
  && report.gpu_probe?.training_allocation_cap_bytes <= TRAINING_CAP);
check("claim-boundary",
  report.claim_boundary?.supported_claim === "clean-isolated-pytorch-runtime-is-available"
  && report.claim_boundary?.excluded_claims?.includes("trainer-implemented")
  && report.claim_boundary?.excluded_claims?.includes("model-performance"));

const verification = {
  report_version: "0.1.0",
  report_kind: "objgauss.pr02c-runtime-verification",
  verdict: "supported",
  report_sha256: sha256File(reportPath),
  source_commit: report.producer.source_commit,
  checks,
  claim_boundary: "PR-02C C0 runtime, isolation, lineage, and GPU reserve only",
};
atomicJson(outputPath, verification);
process.stdout.write(`${JSON.stringify({
  verdict: verification.verdict,
  check_count: checks.length,
  report_sha256: verification.report_sha256,
  output: args.output,
})}\n`);
