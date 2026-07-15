import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { sha256Hex } from "../src/pr00/node-hash.mjs";
import {
  supportedContractKeys,
  validateContract,
} from "../src/pr01/contract-dispatch.mjs";

const FIXTURE_ROOT = "contracts/fixtures/pr02a";
const SCHEMA_ROOT = "contracts/objgauss/0.3.0";

const FIXTURES = new Map([
  ["dynamics-experiment", "0.3.0:objgauss.dynamics_experiment"],
  ["training-trial", "0.3.0:objgauss.training_trial"],
  ["training-attempt", "0.3.0:objgauss.training_attempt"],
  ["checkpoint-manifest", "0.3.0:objgauss.checkpoint_manifest"],
  ["dynamics-prediction", "0.3.0:objgauss.dynamics_prediction"],
  ["dynamics-evaluation-report", "0.3.0:objgauss.dynamics_evaluation_report"],
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

function parentAtPointer(document, pointer) {
  const tokens = pointer.split("/").slice(1).map(decodePointerToken);
  const final = tokens.pop();
  let parent = document;
  for (const token of tokens) {
    parent = parent[Array.isArray(parent) ? Number(token) : token];
  }
  return { parent, final };
}

function setPointer(document, pointer, value) {
  const { parent, final } = parentAtPointer(document, pointer);
  parent[Array.isArray(parent) ? Number(final) : final] = value;
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

test("the exact registry preserves old kinds and adds six explicit 0.3.0 records", () => {
  assert.deepEqual(supportedContractKeys, [
    "0.1.0:objgauss.episode",
    "0.2.0:objgauss.episode",
    "0.2.0:objgauss.experiment",
    "0.2.0:objgauss.attempt",
    "0.2.0:objgauss.invariance_report",
    "0.3.0:objgauss.dynamics_experiment",
    "0.3.0:objgauss.training_trial",
    "0.3.0:objgauss.training_attempt",
    "0.3.0:objgauss.checkpoint_manifest",
    "0.3.0:objgauss.dynamics_prediction",
    "0.3.0:objgauss.dynamics_evaluation_report",
  ]);
});

test("the PR-02A manifest freezes all old contracts, new schemas, and fixtures", () => {
  const manifest = readJson(`${FIXTURE_ROOT}/manifest.json`);
  assert.equal(manifest.frozen_contracts.length, 5);
  assert.equal(manifest.schemas.length, 7);
  assert.equal(manifest.fixtures.length, 7);
  for (const item of [
    ...manifest.frozen_contracts,
    ...manifest.schemas,
    ...manifest.fixtures,
  ]) {
    assert.equal(sha256Hex(readFileSync(item.path)), item.sha256, item.path);
  }
});

test("all six 0.3.0 record schemas are strict and positive fixtures are valid", () => {
  for (const [name, expectedKey] of FIXTURES) {
    const schema = readJson(`${SCHEMA_ROOT}/${name}.schema.json`);
    assert.equal(schema.$schema, "https://json-schema.org/draft/2020-12/schema");
    assert.equal(schema.properties.schema_version.const, "0.3.0");
    assert.equal(schema.unevaluatedProperties, false);

    const result = validateContract(fixture(name));
    assert.equal(result.valid, true, JSON.stringify(result));
    assert.equal(result.reason_code, "contract-valid");
    assert.equal(result.contract_key, expectedKey);
  }
});

test("0.3.0 uses exact version/kind dispatch and never migrates old records", () => {
  for (const version of ["latest", "0.3", "0.4.0", undefined]) {
    const document = fixture("dynamics-experiment");
    document.schema_version = version;
    assert.equal(validateContract(document).reason_code, "unsupported-contract-version");
  }

  const oldEpisode = readJson("contracts/fixtures/pr01a/episode.valid.json");
  oldEpisode.schema_version = "0.3.0";
  assert.equal(validateContract(oldEpisode).reason_code, "unsupported-contract-kind");

  const newExperiment = fixture("dynamics-experiment");
  newExperiment.schema_version = "0.2.0";
  assert.equal(validateContract(newExperiment).reason_code, "unsupported-contract-kind");
});

test("unknown fields and non-finite values fail closed in every 0.3.0 record", () => {
  const numericPaths = new Map([
    ["dynamics-experiment", "/endpoint/delta"],
    ["training-trial", "/resources/gpu_hours"],
    ["training-attempt", "/timing/wall_seconds"],
    ["checkpoint-manifest", "/selection/validation_primary_error"],
    ["dynamics-prediction", "/predictions/0/objects/0/position_W_m/0"],
    ["dynamics-evaluation-report", "/baseline_comparisons/0/error_reduction"],
  ]);
  for (const [name] of FIXTURES) {
    const unknown = fixture(name);
    unknown.unexpected = true;
    assert.equal(validateContract(unknown).reason_code, "schema-invalid", name);

    const nonFinite = fixture(name);
    setPointer(nonFinite, numericPaths.get(name), Number.NaN);
    assert.equal(validateContract(nonFinite).reason_code, "schema-invalid", name);
  }
});

test("all committed artifact paths reject traversal and absolute paths", () => {
  const cases = [
    ["dynamics-experiment", "/training/hyperparameter_grid/uri"],
    ["training-attempt", "/outputs/training_log/value/uri"],
    ["checkpoint-manifest", "/payload/uri"],
    ["dynamics-prediction", "/inputs/source_episode/uri"],
    ["dynamics-evaluation-report", "/artifacts/machine_report/uri"],
  ];
  for (const [name, path] of cases) {
    for (const unsafe of ["../escape.json", "/absolute/path.json", "safe/../../escape.json"]) {
      const document = fixture(name);
      setPointer(document, path, unsafe);
      assert.equal(validateContract(document).reason_code, "schema-invalid", `${name}:${unsafe}`);
    }
  }
});

test("the committed PR-02A negative matrix returns stable reason codes", () => {
  const negative = readJson(`${FIXTURE_ROOT}/negative-cases.json`);
  assert.equal(negative.version, "0.3.0");
  assert.ok(negative.cases.length >= 30);
  for (const item of negative.cases) {
    const document = readJson(`${FIXTURE_ROOT}/${item.base}`);
    applyMutation(document, item);
    const result = validateContract(document);
    assert.equal(result.valid, false, item.case_id);
    assert.equal(result.reason_code, item.expected_reason_code, item.case_id);
  }
});

test("deterministic predictions forbid learned-model lineage", () => {
  const prediction = fixture("dynamics-prediction");
  prediction.identity.model_arm = "copy_state";
  prediction.identity.trial_id = { availability: "missing", reason: "not_applicable" };
  prediction.identity.checkpoint_id = { availability: "missing", reason: "not_applicable" };
  prediction.identity.training_seed = { availability: "missing", reason: "not_applicable" };
  assert.equal(validateContract(prediction).valid, true);

  prediction.identity.trial_id = { availability: "present", value: "forbidden-trial" };
  assert.equal(
    validateContract(prediction).reason_code,
    "prediction-model-lineage-inconsistent",
  );
});

test("training failure classifications cannot make scientific failures retryable", () => {
  const attempt = fixture("training-attempt");
  attempt.outcome = {
    status: "failed",
    classification: "infrastructure",
    reason_code: "non_finite_output",
    message: "Fixture scientific failure is deliberately misclassified.",
  };
  attempt.outputs.checkpoint = { availability: "missing", reason: "invalidated" };
  assert.equal(
    validateContract(attempt).reason_code,
    "training-attempt-classification-inconsistent",
  );
});

test("supported evaluation requires every baseline, control, count, and hard gate", () => {
  const report = fixture("dynamics-evaluation-report");
  assert.equal(validateContract(report).valid, true);

  report.counts.observed_predictions = 7;
  assert.equal(
    validateContract(report).reason_code,
    "supported-report-has-failed-gates",
  );
});

test("evaluation reports encode rejected, blocked, and invalid without relabeling them", () => {
  const rejected = fixture("dynamics-evaluation-report");
  rejected.baseline_comparisons[0] = {
    baseline: "copy_state",
    error_reduction: 0.04,
    confidence_interval_95: { lower: 0.02, upper: 0.06 },
    delta: 0.05,
    passed: false,
  };
  rejected.hard_gates[8].status = "rejected";
  rejected.verdict = { status: "rejected", reason_code: "scientific_gate_failed" };
  assert.equal(validateContract(rejected).valid, true);

  const blocked = fixture("dynamics-evaluation-report");
  blocked.counts.observed_predictions = 7;
  blocked.hard_gates[4].status = "blocked";
  blocked.verdict = { status: "blocked", reason_code: "evidence_incomplete" };
  assert.equal(validateContract(blocked).valid, true);

  const invalid = fixture("dynamics-evaluation-report");
  invalid.hard_gates[0].status = "invalid";
  invalid.verdict = { status: "invalid", reason_code: "structural_evidence_invalid" };
  assert.equal(validateContract(invalid).valid, true);
});
