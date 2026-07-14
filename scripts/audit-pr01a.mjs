import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalStringify } from "../src/pr00/canonical-json.mjs";
import { sha256Hex } from "../src/pr00/node-hash.mjs";
import { validateContract } from "../src/pr01/contract-dispatch.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURE_ROOT = resolve(ROOT, "contracts/fixtures/pr01a");
const OUTPUT = resolve(ROOT, "generated/pr01a");

function decodePointerToken(token) {
  return token.replaceAll("~1", "/").replaceAll("~0", "~");
}

function setPointer(document, pointer, value) {
  const tokens = pointer.split("/").slice(1).map(decodePointerToken);
  const final = tokens.pop();
  let parent = document;
  for (const token of tokens) {
    parent = parent[Array.isArray(parent) ? Number(token) : token];
  }
  parent[Array.isArray(parent) ? Number(final) : final] = value;
}

function deletePointer(document, pointer) {
  const tokens = pointer.split("/").slice(1).map(decodePointerToken);
  const final = tokens.pop();
  let parent = document;
  for (const token of tokens) {
    parent = parent[Array.isArray(parent) ? Number(token) : token];
  }
  if (Array.isArray(parent)) {
    parent.splice(Number(final), 1);
  } else {
    delete parent[final];
  }
}

function applyMutation(document, item) {
  if (item.operation === "delete") {
    deletePointer(document, item.path);
    return;
  }
  const value = item.operation === "set-non-finite" ? Number.NaN : item.value;
  setPointer(document, item.path, value);
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

const manifest = await readJson(resolve(FIXTURE_ROOT, "manifest.json"));
const checksumChecks = [];
for (const item of [manifest.frozen_pr00_schema, ...manifest.schemas, ...manifest.fixtures]) {
  const bytes = await readFile(resolve(ROOT, item.path));
  const observed = sha256Hex(bytes);
  checksumChecks.push({
    path: item.path,
    expected_sha256: item.sha256,
    observed_sha256: observed,
    matches: observed === item.sha256,
  });
}

const positiveChecks = [];
for (const name of ["episode", "experiment", "attempt", "invariance-report"]) {
  const document = await readJson(resolve(FIXTURE_ROOT, `${name}.valid.json`));
  const validation = validateContract(document);
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
  slice: "PR-01A",
  endpoint: "versioned-contract-fixtures-and-negative-cases-pass",
  verdict: supported ? "supported" : "invalid",
  frozen_pr00_schema_sha256: manifest.frozen_pr00_schema.sha256,
  schema_count: manifest.schemas.length,
  positive_fixture_count: positiveChecks.length,
  negative_case_count: negativeChecks.length,
  checksum_checks: checksumChecks,
  positive_checks: positiveChecks,
  negative_checks: negativeChecks,
  claim_boundary: "PR-01A contract expressiveness and fail-closed dispatch only; "
    + "no runtime, writer, invariance evaluator, formal cohort, model, training, or robot claim",
};

await mkdir(OUTPUT, { recursive: true });
const reportBytes = canonicalStringify(report);
await writeFile(resolve(OUTPUT, "contract-report.json"), reportBytes, "utf8");
console.log(JSON.stringify({
  output: "generated/pr01a/contract-report.json",
  verdict: report.verdict,
  report_sha256: sha256Hex(reportBytes),
  schema_count: report.schema_count,
  positive_fixture_count: report.positive_fixture_count,
  negative_case_count: report.negative_case_count,
}, null, 2));

if (!supported) {
  process.exit(1);
}
