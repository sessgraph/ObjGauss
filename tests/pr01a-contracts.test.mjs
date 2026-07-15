import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { sha256Hex } from "../src/pr00/node-hash.mjs";
import {
  supportedContractKeys,
  validateContract,
} from "../src/pr01/contract-dispatch.mjs";
import { createSyntheticAudit } from "../src/pr00/synthetic-audit.mjs";

const FIXTURE_ROOT = "contracts/fixtures/pr01a";
const SCHEMA_ROOT = "contracts/objgauss/0.2.0";
const PR00_SCHEMA_PATH = "contracts/objgauss/0.1.0/episode.schema.json";
const PR00_SCHEMA_SHA256 = "b619618706a1bd8da370c465fb36ba8e8edb08ada3406663fc2e2ed2dfa0da9c";

const FIXTURES = new Map([
  ["episode", "0.2.0:objgauss.episode"],
  ["experiment", "0.2.0:objgauss.experiment"],
  ["attempt", "0.2.0:objgauss.attempt"],
  ["invariance-report", "0.2.0:objgauss.invariance_report"],
]);

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function fixture(name) {
  return readJson(`${FIXTURE_ROOT}/${name}.valid.json`);
}

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

test("0.1.0 remains byte-frozen and dispatches without migration", () => {
  const schemaBytes = readFileSync(PR00_SCHEMA_PATH);
  assert.equal(sha256Hex(schemaBytes), PR00_SCHEMA_SHA256);

  const { episode } = createSyntheticAudit();
  const result = validateContract(episode);
  assert.equal(result.valid, true);
  assert.equal(result.contract_key, "0.1.0:objgauss.episode");

  const notMigrated = structuredClone(episode);
  notMigrated.schema_version = "0.2.0";
  const rejected = validateContract(notMigrated);
  assert.equal(rejected.valid, false);
  assert.equal(rejected.reason_code, "schema-invalid");
});

test("the PR-01A manifest freezes every schema and fixture checksum", () => {
  const manifest = readJson(`${FIXTURE_ROOT}/manifest.json`);
  assert.equal(
    sha256Hex(readFileSync(manifest.frozen_pr00_schema.path)),
    manifest.frozen_pr00_schema.sha256,
  );
  for (const item of [...manifest.schemas, ...manifest.fixtures]) {
    assert.equal(sha256Hex(readFileSync(item.path)), item.sha256, item.path);
  }
});

test("the frozen registry prefix has one 0.1.0 entry and four 0.2.0 entries", () => {
  assert.deepEqual(supportedContractKeys.filter((key) => !key.startsWith("0.3.0:")), [
    "0.1.0:objgauss.episode",
    "0.2.0:objgauss.episode",
    "0.2.0:objgauss.experiment",
    "0.2.0:objgauss.attempt",
    "0.2.0:objgauss.invariance_report",
  ]);
});

test("all four 0.2.0 schemas and positive fixtures are strict and valid", () => {
  for (const [name, expectedKey] of FIXTURES) {
    const schema = readJson(`${SCHEMA_ROOT}/${name}.schema.json`);
    assert.equal(schema.$schema, "https://json-schema.org/draft/2020-12/schema");
    assert.equal(schema.properties.schema_version.const, "0.2.0");
    assert.equal(schema.unevaluatedProperties, false);

    const result = validateContract(fixture(name));
    assert.equal(result.valid, true, JSON.stringify(result));
    assert.equal(result.reason_code, "contract-valid");
    assert.equal(result.contract_key, expectedKey);
  }
});

test("fixture IDs remain manifest values rather than schema constants", () => {
  const episodeSchema = readJson(`${SCHEMA_ROOT}/episode.schema.json`);
  assert.deepEqual(
    episodeSchema.$defs.identity.properties.fixture_id,
    { $ref: "#/$defs/identifier" },
  );
  const experimentSchema = readJson(`${SCHEMA_ROOT}/experiment.schema.json`);
  assert.deepEqual(
    experimentSchema.$defs.identity.properties.fixture_id,
    { $ref: "#/$defs/identifier" },
  );
});

test("latest, unknown versions, and kind/version mismatches fail before schema dispatch", () => {
  for (const version of ["latest", "0.2", "0.4.0", undefined]) {
    const document = fixture("episode");
    document.schema_version = version;
    const result = validateContract(document);
    assert.equal(result.valid, false);
    assert.equal(result.reason_code, "unsupported-contract-version");
  }

  const wrongKind = fixture("episode");
  wrongKind.contract_kind = "objgauss.unknown";
  assert.equal(validateContract(wrongKind).reason_code, "unsupported-contract-kind");
});

test("unknown fields and non-finite numbers fail closed in every 0.2.0 document", () => {
  const numericPaths = new Map([
    ["episode", "/intervention/commanded_action/vector_W_N/0"],
    ["experiment", "/budgets/branch_timeout_s"],
    ["attempt", "/timing/wall_seconds"],
    ["invariance-report", "/counts/attempts"],
  ]);
  for (const [name] of FIXTURES) {
    const unknown = fixture(name);
    unknown.unexpected = true;
    assert.equal(validateContract(unknown).reason_code, "schema-invalid");

    const nonFinite = fixture(name);
    setPointer(nonFinite, numericPaths.get(name), Number.NaN);
    assert.equal(validateContract(nonFinite).reason_code, "schema-invalid");
  }
});

test("artifact paths reject traversal and absolute paths", () => {
  const cases = [
    ["episode", "/evidence/trajectory/uri"],
    ["experiment", "/runtime/lock_uri"],
    ["invariance-report", "/artifacts/machine_report_uri"],
  ];
  for (const [name, path] of cases) {
    for (const unsafe of ["../escape.json", "/absolute/path.json", "safe/../../escape.json"] ) {
      const document = fixture(name);
      setPointer(document, path, unsafe);
      assert.equal(validateContract(document).reason_code, "schema-invalid");
    }
  }
});

test("the committed negative fixture matrix returns stable reason codes", () => {
  const negative = readJson(`${FIXTURE_ROOT}/negative-cases.json`);
  assert.equal(negative.version, "0.2.0");
  for (const item of negative.cases) {
    const document = readJson(`${FIXTURE_ROOT}/${item.base}`);
    applyMutation(document, item);
    const result = validateContract(document);
    assert.equal(result.valid, false, item.case_id);
    assert.equal(result.reason_code, item.expected_reason_code, item.case_id);
  }
});

test("experiment arithmetic, attempt publication, and report verdict are independently checked", () => {
  const experiment = fixture("experiment");
  experiment.design.expected_group_count = 47;
  assert.equal(validateContract(experiment).reason_code, "experiment-count-mismatch");

  const attempt = fixture("attempt");
  attempt.outcome.status = "succeeded";
  assert.equal(validateContract(attempt).reason_code, "attempt-outcome-inconsistent");

  const report = fixture("invariance-report");
  report.verdict.reason_code = "structural_evidence_invalid";
  assert.equal(validateContract(report).reason_code, "verdict-reason-inconsistent");
});
