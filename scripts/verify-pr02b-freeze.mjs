#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { createContractDispatcher } from "../src/pr01/contract-dispatch.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function loadJson(path) {
  const bytes = await readFile(path);
  return { bytes, document: JSON.parse(bytes.toString("utf8")) };
}

function equal(left, right) {
  return isDeepStrictEqual(left, right);
}

function sorted(values) {
  return [...values].sort();
}

function requireCheck(checks, checkId, condition, detail = {}) {
  checks.push({ check_id: checkId, status: condition ? "supported" : "invalid", ...detail });
  if (!condition) throw new Error(`PR-02B verification failed: ${checkId}`);
}

function partitionProjection(partition) {
  return {
    object_identity_ids: sorted(Object.keys(partition.objects)),
    layout_ids: sorted(Object.keys(partition.layouts)),
    group_ids: partition.groups.map((item) => item.group_id),
    group_count: partition.groups.length,
  };
}

function cuboidMass(specification) {
  return 8 * specification.half_size_m.reduce((product, value) => product * value, 1)
    * specification.density_kg_m3;
}

function approximatelyEqual(left, right, tolerance = 1e-12) {
  return Number.isFinite(left) && Number.isFinite(right)
    && Math.abs(left - right) <= tolerance * Math.max(1, Math.abs(left), Math.abs(right));
}

export function verifyFreezeDocuments({
  report,
  reportSha256,
  experiment,
  formalData,
  formalDataSha256,
  gridSha256,
  lockSha256,
  pilotSpecSha256,
  sourceExperimentSha256,
  analyzerSha256,
  sourceCommit,
  repeatReportSha256,
  sourceAuditSha256,
  sourceAuditDocuments,
  pr01Spec,
}) {
  const checks = [];
  const validation = createContractDispatcher({ root: ROOT })(experiment);
  requireCheck(checks, "experiment-contract-valid", validation.valid, { validation });
  requireCheck(
    checks,
    "pilot-verdict-supported",
    report.verdict.status === "supported"
      && report.verdict.reason_code === "all_hard_gates_passed"
      && report.hard_gates.every((item) => item.status === "supported"),
  );
  requireCheck(
    checks,
    "pilot-hard-gate-set-complete",
    equal(report.hard_gates.map((item) => item.gate_id), [
      "source_audits_supported",
      "canonical_reverse_semantics_match",
      "direction_controls_supported",
      "horizon_covers_action_and_settling",
      "power_target_reached",
      "resources_within_hard_budgets",
      "gpu_display_reserve_supported",
    ]),
  );
  requireCheck(
    checks,
    "pilot-report-lineage",
    experiment.source.source_gate_report_sha256 === reportSha256
      && experiment.data_policy.pilot_freeze.report_sha256 === reportSha256,
  );
  requireCheck(
    checks,
    "frozen-input-checksums",
    report.inputs.pilot_spec_sha256 === pilotSpecSha256
      && report.inputs.source_experiment_sha256 === sourceExperimentSha256
      && report.inputs.runtime_lock_sha256 === lockSha256
      && report.inputs.hyperparameter_grid_sha256 === gridSha256
      && experiment.identity.preregistration_sha256 === pilotSpecSha256
      && experiment.source.runtime_lock_sha256 === lockSha256
      && experiment.training.hyperparameter_grid.sha256 === gridSha256,
  );
  requireCheck(
    checks,
    "producer-clean-source-lineage",
    report.producer.source_sha256 === analyzerSha256
      && report.producer.source_commit === sourceCommit,
  );
  requireCheck(
    checks,
    "repeat-and-audit-lineage",
    equal(report.inputs.repeat_report_sha256, repeatReportSha256)
      && equal(report.inputs.source_audit_sha256, sourceAuditSha256),
  );
  requireCheck(
    checks,
    "source-audit-contracts-valid",
    sourceAuditDocuments.every((document) => createContractDispatcher({ root: ROOT })(document).valid),
  );
  requireCheck(
    checks,
    "formal-data-freeze-lineage",
    report.freeze.formal_data_spec_sha256 === formalDataSha256,
  );
  for (const split of ["train", "validation", "test"]) {
    requireCheck(
      checks,
      `formal-${split}-projection`,
      equal(experiment.data_policy[split], partitionProjection(formalData.partitions[split])),
    );
  }
  const identitySets = ["calibration", "train", "validation", "test"]
    .map((split) => new Set(experiment.data_policy[split].object_identity_ids));
  const layoutSets = ["calibration", "train", "validation", "test"]
    .map((split) => new Set(experiment.data_policy[split].layout_ids));
  const disjoint = (sets) => sets.every((left, index) => sets.slice(index + 1)
    .every((right) => [...left].every((item) => !right.has(item))));
  requireCheck(checks, "object-layout-split-isolation", disjoint(identitySets) && disjoint(layoutSets));
  const formalObjects = Object.values(formalData.partitions)
    .flatMap((partition) => Object.values(partition.objects));
  const formalMaxMass = Math.max(...formalObjects.map(cuboidMass));
  const support = formalData.action_support_envelope;
  requireCheck(
    checks,
    "formal-action-support-within-pilot-envelope",
    support.formal_within_pilot_mass_envelope === true
      && support.formal_max_mass_kg === formalMaxMass
      && formalMaxMass <= support.pilot_max_mass_kg,
  );
  const formalObjectIds = new Set(Object.values(formalData.partitions)
    .flatMap((partition) => Object.keys(partition.objects)));
  const formalLayoutIds = new Set(Object.values(formalData.partitions)
    .flatMap((partition) => Object.keys(partition.layouts)));
  const formalResetSeeds = new Set(Object.values(formalData.partitions)
    .flatMap((partition) => partition.reset_seeds));
  const pr01Seeds = new Set([...pr01Spec.formal_reset_seeds, ...pr01Spec.preflight_reset_seeds]);
  requireCheck(
    checks,
    "pr01-evidence-excluded",
    formalData.pilot_exclusion.excluded_from_training === true
      && formalData.pilot_exclusion.excluded_from_final_statistics === true
      && disjoint([formalObjectIds, new Set(Object.keys(pr01Spec.object_specs))])
      && disjoint([formalLayoutIds, new Set(Object.keys(pr01Spec.layouts))])
      && disjoint([formalResetSeeds, pr01Seeds])
      && !JSON.stringify({ report, experiment, formalData }).includes("group-box-"),
  );
  requireCheck(
    checks,
    "freeze-values-projected",
    equal(experiment.horizon, report.freeze.horizon)
      && equal(experiment.endpoint.normalization_scales, report.freeze.normalization_scales)
      && experiment.endpoint.delta === report.freeze.delta
      && experiment.endpoint.delta_shuffle === report.freeze.delta_shuffle
      && equal(experiment.training.training_seeds, report.freeze.training_seed_values),
  );
  const gpu = report.gpu_probe;
  requireCheck(
    checks,
    "gpu-display-reserve",
    gpu.status === "supported"
      && Math.min(gpu.free_before_bytes, gpu.free_after_bytes) >= gpu.display_reserve_bytes
      && gpu.training_allocation_cap_bytes <= experiment.budgets.training_peak_vram_bytes_max
      && gpu.training_allocation_cap_bytes <= gpu.free_before_bytes - gpu.display_reserve_bytes,
  );
  const resources = report.resources;
  const budgets = resources.hard_budgets;
  requireCheck(
    checks,
    "resource-projection-within-budgets",
    resources.pilot_wall_seconds <= budgets.pilot_wall_seconds_max
      && resources.pilot_artifact_bytes <= budgets.pilot_artifact_bytes_max
      && resources.pilot_process_rss_peak_bytes <= budgets.pilot_process_rss_bytes_max
      && resources.projected_formal_cohort_cpu_wall_hours <= budgets.cohort_cpu_wall_hours_max
      && resources.projected_formal_artifact_bytes <= budgets.artifact_bytes_max
      && resources.scheduled_hpo_gpu_hours_max <= budgets.gpu_hours_pilot_hpo_max
      && resources.scheduled_formal_gpu_hours_max <= budgets.gpu_hours_formal_max
      && resources.scheduled_total_gpu_hours_max <= budgets.gpu_hours_total_max
      && resources.scheduled_hpo_gpu_hours_max
        === resources.scheduled_hpo_gpu_hours_base * (1 + resources.retry_gpu_reserve_fraction)
      && resources.scheduled_formal_gpu_hours_max
        === resources.scheduled_formal_gpu_hours_base * (1 + resources.retry_gpu_reserve_fraction)
      && experiment.retry_policy.max_extra_attempt_fraction
        === resources.retry_gpu_reserve_fraction,
  );
  requireCheck(
    checks,
    "power-freeze-supported",
    report.power.selected !== null
      && report.power.selected.supported === true
      && report.power.selected.power >= report.power.target_power
      && report.freeze.formal_group_counts.test === report.power.selected.test_groups
      && report.freeze.training_seed_values.length === report.power.selected.training_seeds,
  );
  const expectedDelta = Math.max(0.05, Math.min(0.10, 0.10 * report.power.median_normalized_effect));
  const expectedGroupSigma = expectedDelta * report.power.raw_group_effect_sigma
    / Math.abs(report.power.median_normalized_effect);
  const expectedSeedSigma = expectedDelta * report.power.raw_bootstrap_group_mean_sigma
    / Math.abs(report.power.median_normalized_effect);
  const supportedCandidates = report.power.candidates.filter((item) => item.supported).sort((left, right) => (
    left.test_groups * left.training_seeds - right.test_groups * right.training_seeds
    || left.test_groups - right.test_groups
    || left.training_seeds - right.training_seeds
  ));
  requireCheck(
    checks,
    "power-formulas-recomputed",
    report.power.positive_effect_scale_supported === true
      && approximatelyEqual(report.power.delta, expectedDelta)
      && approximatelyEqual(report.power.delta_shuffle, Math.max(0.03, 0.60 * expectedDelta))
      && approximatelyEqual(report.power.group_error_reduction_sigma_proxy, expectedGroupSigma)
      && approximatelyEqual(report.power.training_seed_error_reduction_sigma_proxy, expectedSeedSigma)
      && equal(report.power.selected, supportedCandidates[0]),
  );
  requireCheck(
    checks,
    "source-audits-supported",
    report.source_audits.length === 2
      && report.source_audits.every((item) => item.status === "supported")
      && report.counts.pilot_groups_per_repeat === 12
      && report.counts.pilot_repeats === 2
      && report.counts.pilot_episodes === 120,
  );
  return checks;
}

async function main() {
  const [
    rootArgument = "generated/pr02b/evidence",
    outputArgument,
    sourceCommit,
  ] = process.argv.slice(2);
  if (!/^[0-9a-f]{40}$/.test(sourceCommit ?? "")) {
    throw new Error("PR-02B verification requires the exact 40-character source commit");
  }
  const evidenceRoot = resolve(rootArgument);
  const output = resolve(outputArgument ?? `${evidenceRoot}/freeze/verification-report.json`);
  const reportLoaded = await loadJson(resolve(evidenceRoot, "freeze/pilot-report.json"));
  const experimentLoaded = await loadJson(resolve(evidenceRoot, "freeze/dynamics-experiment.json"));
  const formalLoaded = await loadJson(resolve(evidenceRoot, "freeze/formal-data-spec.json"));
  const gridBytes = await readFile(resolve(ROOT, "contracts/fixtures/pr02b/hyperparameter-grid.json"));
  const lockBytes = await readFile(resolve(ROOT, "sim/uv.lock"));
  const pilotSpecBytes = await readFile(resolve(ROOT, "contracts/fixtures/pr02b/pilot-spec.json"));
  const sourceExperimentBytes = await readFile(resolve(ROOT, "contracts/fixtures/pr02b/source-experiment.json"));
  const pr01Spec = JSON.parse(await readFile(
    resolve(ROOT, "contracts/fixtures/pr01e/cohort-spec.json"),
    "utf8",
  ));
  const analyzerBytes = await readFile(resolve(ROOT, "sim/src/objgauss_sim/pr02_pilot.py"));
  const repeatReports = await Promise.all([
    readFile(resolve(evidenceRoot, "repeat-a/cohort-report.json")),
    readFile(resolve(evidenceRoot, "repeat-b/cohort-report.json")),
  ]);
  const sourceAudits = await Promise.all([
    loadJson(resolve(evidenceRoot, "repeat-a/source-audit.json")),
    loadJson(resolve(evidenceRoot, "repeat-b/source-audit.json")),
  ]);
  const checks = verifyFreezeDocuments({
    report: reportLoaded.document,
    reportSha256: sha256(reportLoaded.bytes),
    experiment: experimentLoaded.document,
    formalData: formalLoaded.document,
    formalDataSha256: sha256(formalLoaded.bytes),
    gridSha256: sha256(gridBytes),
    lockSha256: sha256(lockBytes),
    pilotSpecSha256: sha256(pilotSpecBytes),
    sourceExperimentSha256: sha256(sourceExperimentBytes),
    analyzerSha256: sha256(analyzerBytes),
    sourceCommit,
    repeatReportSha256: repeatReports.map(sha256),
    sourceAuditSha256: sourceAudits.map((item) => sha256(item.bytes)),
    sourceAuditDocuments: sourceAudits.map((item) => item.document),
    pr01Spec,
  });
  const sourceBytes = await readFile(fileURLToPath(import.meta.url));
  const verification = {
    report_version: "0.1.0",
    report_kind: "objgauss.pr02b-freeze-verification",
    verdict: "supported",
    source_sha256: sha256(sourceBytes),
    pilot_report_sha256: sha256(reportLoaded.bytes),
    dynamics_experiment_sha256: sha256(experimentLoaded.bytes),
    formal_data_spec_sha256: sha256(formalLoaded.bytes),
    checks,
    claim_boundary: "PR-02B lineage, freeze projection, isolation, power, and resource gates only",
  };
  await writeFile(output, `${JSON.stringify(verification, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify({
    output,
    verdict: verification.verdict,
    check_count: checks.length,
    pilot_report_sha256: verification.pilot_report_sha256,
  })}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    process.stderr.write(`${error.stack ?? error.message}\n`);
    process.exitCode = 4;
  });
}
