import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalStringify } from "../src/pr00/canonical-json.mjs";
import { sha256Hex } from "../src/pr00/node-hash.mjs";
import { validateContract } from "../src/pr01/contract-dispatch.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURE_ROOT = resolve(ROOT, "contracts/fixtures/pr02a");
const OUTPUT = resolve(ROOT, "generated/pr02a");

const POSITIVE_FIXTURES = [
  "dynamics-experiment",
  "training-trial",
  "training-attempt",
  "checkpoint-manifest",
  "dynamics-prediction",
  "dynamics-evaluation-report",
];

function decodePointerToken(token) {
  return token.replaceAll("~1", "/").replaceAll("~0", "~");
}

function parentAtPointer(document, pointer) {
  const tokens = pointer.split("/").slice(1).map(decodePointerToken);
  const final = tokens.pop();
  let parent = document;
  for (const token of tokens) {
    parent = parent[Array.isArray(parent) ? Number(token) : token];
  }
  return { parent, final };
}

function applyMutation(document, item) {
  const { parent, final } = parentAtPointer(document, item.path);
  const key = Array.isArray(parent) ? Number(final) : final;
  if (item.operation === "delete") {
    if (Array.isArray(parent)) {
      parent.splice(key, 1);
    } else {
      delete parent[key];
    }
    return;
  }
  parent[key] = item.operation === "set-non-finite" ? Number.NaN : item.value;
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

const manifest = await readJson(resolve(FIXTURE_ROOT, "manifest.json"));
const checksumChecks = [];
for (const item of [
  ...manifest.frozen_contracts,
  ...manifest.schemas,
  ...manifest.fixtures,
]) {
  const observed = sha256Hex(await readFile(resolve(ROOT, item.path)));
  checksumChecks.push({
    path: item.path,
    expected_sha256: item.sha256,
    observed_sha256: observed,
    matches: observed === item.sha256,
  });
}

const positiveChecks = [];
for (const name of POSITIVE_FIXTURES) {
  const validation = validateContract(
    await readJson(resolve(FIXTURE_ROOT, `${name}.valid.json`)),
  );
  positiveChecks.push({
    fixture: `${name}.valid.json`,
    contract_key: validation.contract_key,
    valid: validation.valid,
    reason_code: validation.reason_code,
  });
}

const negativeFixture = await readJson(resolve(FIXTURE_ROOT, "negative-cases.json"));
const negativeChecks = [];
for (const item of negativeFixture.cases) {
  const document = await readJson(resolve(FIXTURE_ROOT, item.base));
  applyMutation(document, item);
  const validation = validateContract(document);
  negativeChecks.push({
    case_id: item.case_id,
    rejected: !validation.valid,
    expected_reason_code: item.expected_reason_code,
    observed_reason_code: validation.reason_code,
    matches: !validation.valid && validation.reason_code === item.expected_reason_code,
  });
}

const supported = checksumChecks.every((item) => item.matches)
  && positiveChecks.every((item) => item.valid)
  && negativeChecks.every((item) => item.matches);
const report = {
  slice: "PR-02A",
  endpoint: "versioned-dynamics-evidence-contract-is-strict-and-unambiguous",
  verdict: supported ? "supported" : "invalid",
  frozen_contract_count: manifest.frozen_contracts.length,
  schema_count: manifest.schemas.length,
  dispatchable_record_count: positiveChecks.length,
  positive_fixture_count: positiveChecks.length,
  negative_case_count: negativeChecks.length,
  checksum_checks: checksumChecks,
  positive_checks: positiveChecks,
  negative_checks: negativeChecks,
  claim_boundary: "PR-02A contract expressiveness, exact dispatch, and fail-closed semantics only; "
    + "no pilot, cohort, trainer, checkpoint, model metric, Gaussian, or robot-control claim",
};

await mkdir(OUTPUT, { recursive: true });
const reportBytes = canonicalStringify(report);
await writeFile(resolve(OUTPUT, "contract-report.json"), reportBytes, "utf8");
console.log(JSON.stringify({
  output: "generated/pr02a/contract-report.json",
  verdict: report.verdict,
  report_sha256: sha256Hex(reportBytes),
  frozen_contract_count: report.frozen_contract_count,
  schema_count: report.schema_count,
  dispatchable_record_count: report.dispatchable_record_count,
  positive_fixture_count: report.positive_fixture_count,
  negative_case_count: report.negative_case_count,
}, null, 2));

if (!supported) {
  process.exit(1);
}
