import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createContractDispatcher } from "../src/pr01/contract-dispatch.mjs";
import {
  buildReport,
  EXIT_CODES,
  validateNegativeCases,
} from "../src/pr01/audit/evaluator.mjs";

const HASH = "a".repeat(64);
const NEGATIVE_CASES = [
  ["snapshot_changed", "rejected"],
  ["rng_changed", "rejected"],
  ["physics_changed", "rejected"],
  ["cross_split_leakage", "invalid"],
  ["action_missing_or_duplicate", "invalid"],
  ["ledger_missing", "invalid"],
  ["contact_tampered", "rejected"],
  ["checksum_mismatch", "invalid"],
  ["lineage_broken", "invalid"],
  ["attempt_timeout", "blocked"],
  ["non_finite_value", "invalid"],
].map(([caseId, status]) => ({
  case_id: caseId,
  outcome: "passed",
  expected_status: status,
  expected_reason_code: caseId,
}));

function evaluation(status) {
  const reasons = {
    supported: "all_hard_gates_passed",
    rejected: "scientific_gate_failed",
    blocked: "evidence_incomplete",
    invalid: "structural_evidence_invalid",
  };
  const supported = status === "supported";
  return {
    status,
    reason_code: reasons[status],
    identity: { experiment_id: "audit-fixture", fixture_id: "audit-fixture" },
    inputs: {
      experiment_manifest_sha256: HASH,
      episode_index_sha256: HASH,
      attempt_index_sha256: HASH,
      source_gate_report_sha256: HASH,
    },
    counts: {
      expected_groups: 1,
      observed_groups: supported ? 1 : 0,
      expected_episodes: 5,
      observed_episodes: supported ? 5 : 0,
      attempts: supported ? 5 : 0,
      failed_attempts: 0,
    },
    checks: [{
      check_id: "fixture-check",
      status,
      reason_code: supported ? "schema_valid" : status === "rejected" ? "snapshot_changed" : status === "blocked" ? "attempt_timeout" : "checksum_mismatch",
      group_id: { availability: "missing", reason: "not_applicable" },
      branch_id: { availability: "missing", reason: "not_applicable" },
      evidence_sha256: HASH,
    }],
  };
}

test("auditor has frozen four-state exit codes", () => {
  assert.deepEqual(EXIT_CODES, { supported: 0, rejected: 2, blocked: 3, invalid: 4 });
});

test("auditor source imports neither simulator, adapter nor writer", async () => {
  const source = await readFile(new URL("../src/pr01/audit/evaluator.mjs", import.meta.url), "utf8");
  const imports = [...source.matchAll(/^import\s+.*?from\s+["']([^"']+)["'];?$/gm)].map((match) => match[1]);
  assert.deepEqual(imports, [
    "node:crypto",
    "node:fs/promises",
    "node:path",
    "node:url",
    "../contract-dispatch.mjs",
  ]);
});

test("mutation matrix must be complete and passed", () => {
  assert.equal(validateNegativeCases(NEGATIVE_CASES), NEGATIVE_CASES);
  assert.throws(() => validateNegativeCases(NEGATIVE_CASES.slice(1)), /incomplete/);
  assert.throws(
    () => validateNegativeCases(NEGATIVE_CASES.map((item, index) => index === 0 ? { ...item, outcome: "failed" } : item)),
    /did not pass/,
  );
});

for (const status of ["supported", "rejected", "blocked", "invalid"]) {
  test(`machine report is contract-valid for ${status}`, async () => {
    const report = await buildReport({
      evaluation: evaluation(status),
      negativeCases: NEGATIVE_CASES,
      artifacts: {
        machine_report_uri: "generated/pr01d/audit-report.json",
        human_report_uri: "generated/pr01d/audit-report.md",
        checksums_uri: "generated/pr01d/checksums.sha256",
      },
    });
    const result = createContractDispatcher()(report);
    assert.equal(result.valid, true, JSON.stringify(result));
  });
}
