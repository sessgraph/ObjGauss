import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";
import { canonicalStringify } from "../pr00/canonical-json.mjs";
import { createEpisodeValidator as createPr00EpisodeValidator } from "../pr00/contract-validator.mjs";
import { sha256Hex } from "../pr00/node-hash.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

const CONTRACT_ENTRIES = [
  {
    version: "0.1.0",
    kind: "objgauss.episode",
    relativePath: "contracts/objgauss/0.1.0/episode.schema.json",
    semanticValidator: "pr00-episode",
  },
  {
    version: "0.2.0",
    kind: "objgauss.episode",
    relativePath: "contracts/objgauss/0.2.0/episode.schema.json",
    semanticValidator: "pr01-episode",
  },
  {
    version: "0.2.0",
    kind: "objgauss.experiment",
    relativePath: "contracts/objgauss/0.2.0/experiment.schema.json",
    semanticValidator: "pr01-experiment",
  },
  {
    version: "0.2.0",
    kind: "objgauss.attempt",
    relativePath: "contracts/objgauss/0.2.0/attempt.schema.json",
    semanticValidator: "pr01-attempt",
  },
  {
    version: "0.2.0",
    kind: "objgauss.invariance_report",
    relativePath: "contracts/objgauss/0.2.0/invariance-report.schema.json",
    semanticValidator: "pr01-invariance-report",
  },
  {
    version: "0.3.0",
    kind: "objgauss.dynamics_experiment",
    relativePath: "contracts/objgauss/0.3.0/dynamics-experiment.schema.json",
    semanticValidator: "pr02-dynamics-experiment",
  },
  {
    version: "0.3.0",
    kind: "objgauss.training_trial",
    relativePath: "contracts/objgauss/0.3.0/training-trial.schema.json",
    semanticValidator: "pr02-training-trial",
  },
  {
    version: "0.3.0",
    kind: "objgauss.training_attempt",
    relativePath: "contracts/objgauss/0.3.0/training-attempt.schema.json",
    semanticValidator: "pr02-training-attempt",
  },
  {
    version: "0.3.0",
    kind: "objgauss.checkpoint_manifest",
    relativePath: "contracts/objgauss/0.3.0/checkpoint-manifest.schema.json",
    semanticValidator: "pr02-checkpoint-manifest",
  },
  {
    version: "0.3.0",
    kind: "objgauss.dynamics_prediction",
    relativePath: "contracts/objgauss/0.3.0/dynamics-prediction.schema.json",
    semanticValidator: "pr02-dynamics-prediction",
  },
  {
    version: "0.3.0",
    kind: "objgauss.dynamics_evaluation_report",
    relativePath: "contracts/objgauss/0.3.0/dynamics-evaluation-report.schema.json",
    semanticValidator: "pr02-dynamics-evaluation-report",
  },
];

const PR02_COMMON_SCHEMA_PATH = "contracts/objgauss/0.3.0/common.schema.json";

const SUPPORTED_VERSIONS = new Set(CONTRACT_ENTRIES.map((entry) => entry.version));

function key(version, kind) {
  return `${version}:${kind}`;
}

function semanticIssue(path, reasonCode, message) {
  return { path, reason_code: reasonCode, message };
}

function magnitude(vector) {
  return Math.hypot(...vector);
}

function equalJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function close(left, right, tolerance = 1e-12) {
  return Number.isFinite(left) && Number.isFinite(right) && Math.abs(left - right) <= tolerance;
}

function validateAction(action, path, issues) {
  const expectedDuration = action.applied_steps / action.sim_frequency_hz;
  if (!close(action.duration_s, expectedDuration)) {
    issues.push(semanticIssue(
      `${path}/duration_s`,
      "action-duration-mismatch",
      "duration_s must equal applied_steps / sim_frequency_hz",
    ));
  }
  const actionMagnitude = magnitude(action.vector_W_N);
  if (action.kind === "hold" && actionMagnitude !== 0) {
    issues.push(semanticIssue(
      `${path}/vector_W_N`,
      "hold-is-nonzero",
      "hold must use a zero force vector",
    ));
  }
  if (action.kind === "push" && actionMagnitude <= 0) {
    issues.push(semanticIssue(
      `${path}/vector_W_N`,
      "push-is-zero",
      "push must use a non-zero force vector",
    ));
  }
}

function validatePr01Episode(episode) {
  const issues = [];
  const { commanded_action: commanded, executed_action: executed, control_ledger: ledger } = episode.intervention;
  validateAction(commanded, "/intervention/commanded_action", issues);
  validateAction(executed, "/intervention/executed_action", issues);

  if (!equalJson(commanded, executed)) {
    issues.push(semanticIssue(
      "/intervention/executed_action",
      "action-ledger-mismatch",
      "the direct simulator force command and executed action must match exactly",
    ));
  }
  if (ledger.requested_steps !== commanded.applied_steps || ledger.applied_steps !== executed.applied_steps) {
    issues.push(semanticIssue(
      "/intervention/control_ledger",
      "control-step-count-mismatch",
      "control ledger step counts must match commanded and executed actions",
    ));
  }
  if (episode.intervention.target_object_id !== episode.environment.target_object_id) {
    issues.push(semanticIssue(
      "/environment/target_object_id",
      "target-object-mismatch",
      "intervention and environment target_object_id must match",
    ));
  }
  if (episode.initialization.snapshot_id !== episode.provenance.parent_snapshot_id) {
    issues.push(semanticIssue(
      "/provenance/parent_snapshot_id",
      "parent-snapshot-mismatch",
      "provenance parent snapshot must match initialization snapshot",
    ));
  }
  if (episode.evidence.trajectory.media_type !== "application/vnd.objgauss.trajectory+json") {
    issues.push(semanticIssue(
      "/evidence/trajectory/media_type",
      "trajectory-media-type-mismatch",
      "trajectory must use the trajectory media type",
    ));
  }
  if (episode.evidence.contact_ledger.media_type !== "application/vnd.objgauss.contact-ledger+json") {
    issues.push(semanticIssue(
      "/evidence/contact_ledger/media_type",
      "contact-media-type-mismatch",
      "contact ledger must use the contact-ledger media type",
    ));
  }

  const settled = episode.evidence.settling_result;
  const settlingConfig = episode.environment.settling;
  const withinThreshold = settled.final_linear_speed_m_s <= settlingConfig.linear_speed_max_m_s
    && settled.final_angular_speed_rad_s <= settlingConfig.angular_speed_max_rad_s;
  if (settled.settled !== withinThreshold) {
    issues.push(semanticIssue(
      "/evidence/settling_result/settled",
      "settling-result-inconsistent",
      "settled must equal the independently checkable speed-threshold result",
    ));
  }
  return issues;
}

function validatePr01Experiment(experiment) {
  const issues = [];
  const actionIds = experiment.actions.map((action) => action.branch_id);
  if (new Set(actionIds).size !== actionIds.length) {
    issues.push(semanticIssue(
      "/actions",
      "duplicate-branch-id",
      "action branch_id values must be unique",
    ));
  }
  const holdCount = experiment.actions.filter((action) => action.kind === "hold").length;
  if (holdCount !== 1) {
    issues.push(semanticIssue(
      "/actions",
      "hold-branch-count-mismatch",
      "the five-branch action set must contain exactly one hold",
    ));
  }
  experiment.actions.forEach((action, index) => validateAction(action, `/actions/${index}`, issues));

  const design = experiment.design;
  const expectedGroups = design.object_spec_ids.length
    * design.layout_ids.length
    * design.start_pose_ids.length
    * design.reset_seeds.length;
  const expectedEpisodes = expectedGroups * design.branches_per_group;
  if (design.expected_group_count !== expectedGroups || design.expected_episode_count !== expectedEpisodes) {
    issues.push(semanticIssue(
      "/design",
      "experiment-count-mismatch",
      "formal group and episode counts must equal the Cartesian design",
    ));
  }

  const allocation = experiment.split_policy.seed_rank_allocation;
  if (allocation.train + allocation.validation + allocation.test !== design.reset_seeds.length) {
    issues.push(semanticIssue(
      "/split_policy/seed_rank_allocation",
      "split-allocation-mismatch",
      "seed-rank allocation must consume every formal reset seed exactly once",
    ));
  }
  const splitCounts = experiment.split_policy.counts;
  if (
    splitCounts.train_groups + splitCounts.validation_groups + splitCounts.test_groups !== expectedGroups
    || splitCounts.train_episodes !== splitCounts.train_groups * design.branches_per_group
    || splitCounts.validation_episodes !== splitCounts.validation_groups * design.branches_per_group
    || splitCounts.test_episodes !== splitCounts.test_groups * design.branches_per_group
  ) {
    issues.push(semanticIssue(
      "/split_policy/counts",
      "split-count-mismatch",
      "split group and episode counts must reconcile with the formal design",
    ));
  }

  const preflight = experiment.preflight;
  const expectedPreflightGroups = design.object_spec_ids.length
    * design.layout_ids.length
    * preflight.start_pose_ids.length
    * preflight.reserved_reset_seeds.length;
  if (
    preflight.expected_group_count !== expectedPreflightGroups
    || preflight.expected_episode_count !== expectedPreflightGroups * design.branches_per_group
  ) {
    issues.push(semanticIssue(
      "/preflight",
      "preflight-count-mismatch",
      "preflight counts must equal its frozen object/layout/start/seed design",
    ));
  }
  const formalSeeds = new Set(design.reset_seeds);
  if (preflight.reserved_reset_seeds.some((seed) => formalSeeds.has(seed))) {
    issues.push(semanticIssue(
      "/preflight/reserved_reset_seeds",
      "preflight-seed-leakage",
      "preflight seeds must not appear in the formal cohort",
    ));
  }
  const maximumExtraAttempts = Math.floor(
    design.expected_episode_count * experiment.budgets.extra_attempt_fraction_max,
  );
  if (experiment.retry_policy.max_extra_attempts > maximumExtraAttempts) {
    issues.push(semanticIssue(
      "/retry_policy/max_extra_attempts",
      "retry-budget-exceeded",
      "max extra attempts exceeds the frozen fraction of formal episodes",
    ));
  }
  return issues;
}

function validatePr01Attempt(attempt) {
  const issues = [];
  const { timing, outcome, retry, publication } = attempt;
  if (
    timing.finished_monotonic_s < timing.started_monotonic_s
    || !close(timing.wall_seconds, timing.finished_monotonic_s - timing.started_monotonic_s)
  ) {
    issues.push(semanticIssue(
      "/timing",
      "attempt-timing-inconsistent",
      "wall time must equal the monotonic finish-start interval",
    ));
  }

  const succeeded = outcome.status === "succeeded";
  const outcomeConsistent = succeeded
    ? outcome.classification === "none" && outcome.reason_code === "none"
    : outcome.classification !== "none" && outcome.reason_code !== "none";
  const publicationConsistent = succeeded
    ? publication.final_episode_published && publication.episode_artifact.availability === "present"
    : !publication.final_episode_published && publication.episode_artifact.availability === "missing";
  if (!outcomeConsistent || !publicationConsistent) {
    issues.push(semanticIssue(
      "/outcome",
      "attempt-outcome-inconsistent",
      "attempt outcome, failure classification, and final episode publication must agree",
    ));
  }

  const previousAttemptConsistent = retry.ordinal === 1
    ? retry.previous_attempt_id.availability === "missing"
    : retry.previous_attempt_id.availability === "present";
  if (!previousAttemptConsistent) {
    issues.push(semanticIssue(
      "/retry/previous_attempt_id",
      "retry-lineage-inconsistent",
      "only retry ordinal 2 may reference a previous attempt",
    ));
  }
  const retryableReasons = new Set(["simulator_crash", "startup_timeout", "atomic_write_failure"]);
  const eligible = outcome.status === "failed" && retryableReasons.has(outcome.reason_code) && retry.ordinal < retry.max_attempts;
  if (retry.eligible !== eligible) {
    issues.push(semanticIssue(
      "/retry/eligible",
      "retry-eligibility-inconsistent",
      "retry eligibility must follow the frozen infrastructure-only policy",
    ));
  }
  return issues;
}

const VERDICT_REASONS = {
  supported: "all_hard_gates_passed",
  rejected: "scientific_gate_failed",
  blocked: "evidence_incomplete",
  invalid: "structural_evidence_invalid",
};

const REQUIRED_NEGATIVE_CASES = new Set([
  "snapshot_changed",
  "rng_changed",
  "physics_changed",
  "cross_split_leakage",
  "action_missing_or_duplicate",
  "ledger_missing",
  "contact_tampered",
  "checksum_mismatch",
  "lineage_broken",
  "attempt_timeout",
  "non_finite_value",
]);

function validatePr01InvarianceReport(report) {
  const issues = [];
  if (report.verdict.reason_code !== VERDICT_REASONS[report.verdict.status]) {
    issues.push(semanticIssue(
      "/verdict/reason_code",
      "verdict-reason-inconsistent",
      "verdict reason must match the four-state status",
    ));
  }
  const caseIds = report.negative_cases.map((item) => item.case_id);
  if (new Set(caseIds).size !== caseIds.length || !equalJson([...caseIds].sort(), [...REQUIRED_NEGATIVE_CASES].sort())) {
    issues.push(semanticIssue(
      "/negative_cases",
      "negative-case-matrix-incomplete",
      "the complete frozen mutation matrix must be present exactly once",
    ));
  }
  const checkIds = report.checks.map((item) => item.check_id);
  if (new Set(checkIds).size !== checkIds.length) {
    issues.push(semanticIssue(
      "/checks",
      "duplicate-check-id",
      "invariance check IDs must be unique",
    ));
  }
  if (report.verdict.status === "supported") {
    if (
      report.counts.observed_groups !== report.counts.expected_groups
      || report.counts.observed_episodes !== report.counts.expected_episodes
      || report.checks.some((check) => check.status !== "supported")
    ) {
      issues.push(semanticIssue(
        "/verdict/status",
        "supported-report-has-failed-gates",
        "supported requires complete counts and only supported hard-gate checks",
      ));
    }
  }
  if (report.counts.attempts < report.counts.observed_episodes + report.counts.failed_attempts) {
    issues.push(semanticIssue(
      "/counts/attempts",
      "attempt-count-mismatch",
      "attempt count cannot be less than successful episodes plus failed attempts",
    ));
  }
  return issues;
}

function strictlyIncreasing(values) {
  return values.every((value, index) => index === 0 || value > values[index - 1]);
}

function sameMembers(left, right) {
  return left.length === right.length
    && equalJson([...left].sort(), [...right].sort());
}

function validatePr02DynamicsExperiment(experiment) {
  const issues = [];
  const partitions = ["calibration", "train", "validation", "test"]
    .map((name) => experiment.data_policy[name]);
  for (const partition of partitions) {
    if (partition.group_count !== partition.group_ids.length) {
      issues.push(semanticIssue(
        "/data_policy",
        "split-count-inconsistent",
        "each split group_count must equal its unique group_id count",
      ));
      break;
    }
  }

  for (const field of ["object_identity_ids", "layout_ids", "group_ids"]) {
    const seen = new Set();
    if (partitions.some((partition) => partition[field].some((value) => {
      if (seen.has(value)) {
        return true;
      }
      seen.add(value);
      return false;
    }))) {
      issues.push(semanticIssue(
        "/data_policy",
        "split-leakage",
        `${field} values must be disjoint across calibration/train/validation/test`,
      ));
      break;
    }
  }

  const scoringTimes = experiment.horizon.scoring_times_s;
  if (
    !strictlyIncreasing(scoringTimes)
    || !close(scoringTimes.at(-1), experiment.horizon.duration_s)
  ) {
    issues.push(semanticIssue(
      "/horizon/scoring_times_s",
      "horizon-inconsistent",
      "scoring times must be strictly increasing and end at duration_s",
    ));
  }
  if (!close(
    experiment.budgets.gpu_hours_total_max,
    experiment.budgets.gpu_hours_pilot_hpo_max + experiment.budgets.gpu_hours_formal_max,
  )) {
    issues.push(semanticIssue(
      "/budgets/gpu_hours_total_max",
      "gpu-budget-inconsistent",
      "total GPU-hour ceiling must equal pilot/HPO plus formal ceilings",
    ));
  }
  return issues;
}

function validatePr02TrainingTrial(trial) {
  const issues = [];
  if (
    trial.outcome.optimizer_updates_completed > trial.configuration.optimizer_updates_max
    || trial.outcome.epochs_completed > trial.configuration.epochs_max
  ) {
    issues.push(semanticIssue(
      "/outcome",
      "trial-progress-exceeds-budget",
      "completed optimizer updates and epochs cannot exceed the frozen configuration",
    ));
  }
  const completed = trial.outcome.status === "completed";
  const hasMetric = trial.selection.validation_primary_error.availability === "present";
  const hasCheckpoint = trial.selection.checkpoint_id.availability === "present";
  const outcomeConsistent = completed
    ? trial.outcome.reason_code === "completed" && hasMetric && hasCheckpoint
    : trial.outcome.reason_code !== "completed" && !hasMetric && !hasCheckpoint && !trial.selection.selected;
  if (!outcomeConsistent || (trial.selection.selected && !completed)) {
    issues.push(semanticIssue(
      "/outcome",
      "trial-outcome-inconsistent",
      "trial status, selection, validation metric, and checkpoint availability must agree",
    ));
  }
  if (hasMetric && trial.selection.validation_primary_error.value < 0) {
    issues.push(semanticIssue(
      "/selection/validation_primary_error/value",
      "trial-metric-negative",
      "validation primary error cannot be negative",
    ));
  }
  return issues;
}

function validatePr02TrainingAttempt(attempt) {
  const issues = [];
  if (
    attempt.timing.finished_monotonic_s < attempt.timing.started_monotonic_s
    || !close(
      attempt.timing.wall_seconds,
      attempt.timing.finished_monotonic_s - attempt.timing.started_monotonic_s,
    )
  ) {
    issues.push(semanticIssue(
      "/timing",
      "attempt-timing-inconsistent",
      "wall time must equal the monotonic finish-start interval",
    ));
  }

  const succeeded = attempt.outcome.status === "succeeded";
  const outcomeConsistent = succeeded
    ? attempt.outcome.classification === "none"
      && attempt.outcome.reason_code === "none"
      && attempt.outputs.training_log.availability === "present"
      && attempt.outputs.checkpoint.availability === "present"
    : attempt.outcome.classification !== "none"
      && attempt.outcome.reason_code !== "none"
      && attempt.outputs.checkpoint.availability === "missing";
  if (!outcomeConsistent) {
    issues.push(semanticIssue(
      "/outcome",
      "training-attempt-outcome-inconsistent",
      "attempt outcome, classification, reason, and published checkpoint must agree",
    ));
  }
  if (!succeeded) {
    const reasonsByClassification = {
      infrastructure: new Set(["process_crash", "io_failure", "transient_oom"]),
      scientific: new Set(["non_finite_output", "not_converged"]),
      validation: new Set([
        "schema_invalid",
        "lineage_invalid",
        "display_vram_reserve_violated",
      ]),
    };
    if (!reasonsByClassification[attempt.outcome.classification]?.has(attempt.outcome.reason_code)) {
      issues.push(semanticIssue(
        "/outcome",
        "training-attempt-classification-inconsistent",
        "failure reason must belong to its infrastructure/scientific/validation class",
      ));
    }
  }

  const previousAttemptConsistent = attempt.identity.ordinal === 1
    ? attempt.retry.previous_attempt_id.availability === "missing"
    : attempt.retry.previous_attempt_id.availability === "present";
  if (!previousAttemptConsistent) {
    issues.push(semanticIssue(
      "/retry/previous_attempt_id",
      "retry-lineage-inconsistent",
      "only retry ordinal 2 may reference a previous attempt",
    ));
  }

  const retryableReasons = new Set(["process_crash", "io_failure", "transient_oom"]);
  const eligible = attempt.outcome.status === "failed"
    && attempt.outcome.classification === "infrastructure"
    && retryableReasons.has(attempt.outcome.reason_code)
    && attempt.identity.ordinal < attempt.retry.max_attempts;
  if (attempt.retry.eligible !== eligible) {
    issues.push(semanticIssue(
      "/retry/eligible",
      "retry-eligibility-inconsistent",
      "only first-attempt infrastructure failures may be retried",
    ));
  }
  return issues;
}

function validatePr02CheckpointManifest(checkpoint) {
  if (checkpoint.compatibility.runtime_lock_sha256 !== checkpoint.provenance.runtime_lock_sha256) {
    return [semanticIssue(
      "/compatibility/runtime_lock_sha256",
      "checkpoint-runtime-lineage-inconsistent",
      "checkpoint compatibility and provenance must reference the same runtime lock",
    )];
  }
  return [];
}

function validatePr02DynamicsPrediction(prediction) {
  const issues = [];
  const scoringTimes = prediction.horizon.scoring_times_s;
  const predictionTimes = prediction.predictions.map((item) => item.time_s);
  if (
    !sameMembers(scoringTimes, predictionTimes)
    || !strictlyIncreasing(scoringTimes)
    || !strictlyIncreasing(predictionTimes)
    || !close(scoringTimes.at(-1), prediction.horizon.duration_s)
  ) {
    issues.push(semanticIssue(
      "/predictions",
      "prediction-horizon-inconsistent",
      "prediction times must exactly match the strictly increasing frozen horizon",
    ));
  }

  const firstObjectIds = prediction.predictions[0].objects.map((item) => item.object_id);
  const objectSetValid = new Set(firstObjectIds).size === firstObjectIds.length
    && prediction.predictions.every((timePoint) => {
      const ids = timePoint.objects.map((item) => item.object_id);
      return new Set(ids).size === ids.length && sameMembers(ids, firstObjectIds);
    });
  if (!objectSetValid) {
    issues.push(semanticIssue(
      "/predictions",
      "prediction-object-set-inconsistent",
      "every time point must contain the same unique object IDs",
    ));
  }
  if (!prediction.predictions.every((item) => (
    item.objects.some((object) => object.object_id === prediction.inputs.target_object_id)
  ))) {
    issues.push(semanticIssue(
      "/inputs/target_object_id",
      "target-object-missing",
      "the target object must be predicted at every scoring time",
    ));
  }
  if (prediction.predictions.some((timePoint) => timePoint.objects.some((object) => (
    !close(magnitude(object.quaternion_WO_wxyz), 1, 1e-6)
  )))) {
    issues.push(semanticIssue(
      "/predictions",
      "quaternion-not-normalized",
      "all predicted wxyz quaternions must be unit length",
    ));
  }
  if (prediction.predictions.some((timePoint) => timePoint.objects.some((object) => {
    const firstNonZero = object.quaternion_WO_wxyz.find((value) => Math.abs(value) > 1e-12);
    return firstNonZero !== undefined && firstNonZero < 0;
  }))) {
    issues.push(semanticIssue(
      "/predictions",
      "quaternion-sign-noncanonical",
      "predicted wxyz quaternions must use deterministic first-nonzero-positive sign",
    ));
  }

  const learned = ["action_free", "action_conditioned"].includes(prediction.identity.model_arm);
  const lineageFields = [
    prediction.identity.trial_id,
    prediction.identity.checkpoint_id,
    prediction.identity.training_seed,
  ];
  const lineageConsistent = learned
    ? lineageFields.every((value) => value.availability === "present")
    : lineageFields.every((value) => value.availability === "missing");
  if (!lineageConsistent) {
    issues.push(semanticIssue(
      "/identity",
      "prediction-model-lineage-inconsistent",
      "learned arms require trial/checkpoint/seed lineage; deterministic baselines forbid it",
    ));
  }

  const observedPayloadHash = sha256Hex(canonicalStringify(prediction.predictions));
  if (prediction.prediction_payload_sha256 !== observedPayloadHash) {
    issues.push(semanticIssue(
      "/prediction_payload_sha256",
      "prediction-payload-checksum-mismatch",
      "prediction payload hash must match canonical raw predictions",
    ));
  }
  return issues;
}

const REQUIRED_PR02_HARD_GATES = new Set([
  "schema_valid",
  "lineage_valid",
  "split_isolated",
  "final_loader_isolated",
  "predictions_complete",
  "evaluator_independent",
  "resources_within_budget",
  "retry_policy_valid",
  "baseline_copy_passed",
  "baseline_constant_velocity_passed",
  "baseline_action_free_passed",
  "direction_push_pos_x_passed",
  "direction_push_neg_x_passed",
  "action_shuffle_passed",
]);

const PR02_STRUCTURAL_HARD_GATES = new Set([
  "schema_valid",
  "lineage_valid",
  "split_isolated",
  "final_loader_isolated",
  "predictions_complete",
  "evaluator_independent",
  "resources_within_budget",
  "retry_policy_valid",
]);

function intervalConsistent(estimate, interval) {
  return interval.lower <= estimate && estimate <= interval.upper;
}

function validatePr02DynamicsEvaluationReport(report) {
  const issues = [];
  if (report.verdict.reason_code !== VERDICT_REASONS[report.verdict.status]) {
    issues.push(semanticIssue(
      "/verdict/reason_code",
      "verdict-reason-inconsistent",
      "verdict reason must match the four-state status",
    ));
  }

  const comparisons = new Map(report.baseline_comparisons.map((item) => [item.baseline, item]));
  const expectedBaselines = ["copy_state", "constant_velocity", "action_free"];
  if (!sameMembers([...comparisons.keys()], expectedBaselines)) {
    issues.push(semanticIssue(
      "/baseline_comparisons",
      "baseline-matrix-incomplete",
      "all three preregistered baselines must appear exactly once",
    ));
  }
  for (const comparison of report.baseline_comparisons) {
    if (!intervalConsistent(comparison.error_reduction, comparison.confidence_interval_95)) {
      issues.push(semanticIssue(
        "/baseline_comparisons",
        "interval-inconsistent",
        "comparison estimate must lie within its ordered confidence interval",
      ));
      break;
    }
    const passed = comparison.confidence_interval_95.lower > report.endpoint.delta;
    if (!close(comparison.delta, report.endpoint.delta) || comparison.passed !== passed) {
      issues.push(semanticIssue(
        "/baseline_comparisons",
        "comparison-gate-inconsistent",
        "comparison pass requires the paired CI lower bound to exceed frozen delta",
      ));
      break;
    }
  }

  const directionMap = new Map(report.direction_gates.map((item) => [item.direction, item]));
  if (!sameMembers([...directionMap.keys()], ["push_pos_x", "push_neg_x"])) {
    issues.push(semanticIssue(
      "/direction_gates",
      "direction-matrix-incomplete",
      "positive and negative x direction gates must appear exactly once",
    ));
  }
  for (const direction of report.direction_gates) {
    const expectedSign = direction.direction === "push_pos_x" ? 1 : -1;
    const passed = direction.predicted_effect_sign === expectedSign
      && direction.gt_effect_sign === expectedSign;
    if (direction.passed !== passed) {
      issues.push(semanticIssue(
        "/direction_gates",
        "direction-gate-inconsistent",
        "direction pass must match predicted and GT effect signs",
      ));
      break;
    }
  }

  const shufflePassed = report.action_shuffle.confidence_interval_95.lower
    > report.endpoint.delta_shuffle;
  if (
    !intervalConsistent(
      report.action_shuffle.error_increase,
      report.action_shuffle.confidence_interval_95,
    )
    || !close(report.action_shuffle.delta_shuffle, report.endpoint.delta_shuffle)
    || report.action_shuffle.passed !== shufflePassed
  ) {
    issues.push(semanticIssue(
      "/action_shuffle",
      "shuffle-gate-inconsistent",
      "shuffle pass requires an ordered CI whose lower bound exceeds frozen delta_shuffle",
    ));
  }

  const gateIds = report.hard_gates.map((item) => item.gate_id);
  if (
    new Set(gateIds).size !== gateIds.length
    || !sameMembers(gateIds, [...REQUIRED_PR02_HARD_GATES])
  ) {
    issues.push(semanticIssue(
      "/hard_gates",
      "hard-gate-matrix-incomplete",
      "the complete preregistered hard-gate matrix must appear exactly once",
    ));
  }

  const hardGates = new Map(report.hard_gates.map((item) => [item.gate_id, item]));
  const scientificOutcomes = new Map([
    ["baseline_copy_passed", comparisons.get("copy_state")?.passed],
    ["baseline_constant_velocity_passed", comparisons.get("constant_velocity")?.passed],
    ["baseline_action_free_passed", comparisons.get("action_free")?.passed],
    ["direction_push_pos_x_passed", directionMap.get("push_pos_x")?.passed],
    ["direction_push_neg_x_passed", directionMap.get("push_neg_x")?.passed],
    ["action_shuffle_passed", report.action_shuffle.passed],
  ]);
  for (const [gateId, passed] of scientificOutcomes) {
    const observed = hardGates.get(gateId)?.status;
    const expected = passed ? "supported" : "rejected";
    if (["supported", "rejected"].includes(observed) && observed !== expected) {
      issues.push(semanticIssue(
        "/hard_gates",
        "hard-gate-status-inconsistent",
        "scientific hard-gate status must match its independently recomputed outcome",
      ));
      break;
    }
  }

  const expectedRetryFraction = report.retry_audit.extra_attempts
    / report.retry_audit.scheduled_tasks;
  if (!close(report.retry_audit.extra_attempt_fraction, expectedRetryFraction)) {
    issues.push(semanticIssue(
      "/retry_audit/extra_attempt_fraction",
      "retry-audit-inconsistent",
      "extra attempt fraction must equal extra_attempts / scheduled_tasks",
    ));
  }
  if (!close(
    report.resources.gpu_hours_total,
    report.resources.gpu_hours_pilot_hpo + report.resources.gpu_hours_formal,
  )) {
    issues.push(semanticIssue(
      "/resources/gpu_hours_total",
      "resource-accounting-inconsistent",
      "total GPU hours must equal pilot/HPO plus formal GPU hours",
    ));
  }

  const countsComplete = report.counts.expected_groups === report.counts.observed_groups
    && report.counts.expected_training_seeds === report.counts.observed_training_seeds
    && report.counts.expected_predictions === report.counts.observed_predictions;
  const gatesSupported = report.hard_gates.every((item) => item.status === "supported")
    && [...scientificOutcomes.values()].every(Boolean);
  if (report.verdict.status === "supported" && (!countsComplete || !gatesSupported)) {
    issues.push(semanticIssue(
      "/verdict/status",
      "supported-report-has-failed-gates",
      "supported requires complete counts and every preregistered hard gate",
    ));
  }

  const structuralEvidenceInvalid = report.hard_gates.some((item) => (
    item.status === "invalid"
    || (PR02_STRUCTURAL_HARD_GATES.has(item.gate_id) && item.status === "rejected")
  ));
  const evidenceBlocked = !countsComplete
    || report.hard_gates.some((item) => item.status === "blocked");
  const scientificGateRejected = [...scientificOutcomes.values()].includes(false)
    || report.hard_gates.some((item) => (
      !PR02_STRUCTURAL_HARD_GATES.has(item.gate_id) && item.status === "rejected"
    ));
  const expectedVerdict = structuralEvidenceInvalid
    ? "invalid"
    : evidenceBlocked
      ? "blocked"
      : scientificGateRejected
        ? "rejected"
        : "supported";
  if (report.verdict.status !== expectedVerdict) {
    issues.push(semanticIssue(
      "/verdict/status",
      "verdict-status-inconsistent",
      "four-state verdict must be derived from structural validity, completeness, and scientific gates",
    ));
  }
  return issues;
}

const SEMANTIC_VALIDATORS = {
  "pr01-episode": validatePr01Episode,
  "pr01-experiment": validatePr01Experiment,
  "pr01-attempt": validatePr01Attempt,
  "pr01-invariance-report": validatePr01InvarianceReport,
  "pr02-dynamics-experiment": validatePr02DynamicsExperiment,
  "pr02-training-trial": validatePr02TrainingTrial,
  "pr02-training-attempt": validatePr02TrainingAttempt,
  "pr02-checkpoint-manifest": validatePr02CheckpointManifest,
  "pr02-dynamics-prediction": validatePr02DynamicsPrediction,
  "pr02-dynamics-evaluation-report": validatePr02DynamicsEvaluationReport,
};

function schemaErrors(validate) {
  return (validate.errors ?? []).map((error) => ({
    path: error.instancePath || "/",
    reason_code: "schema-invalid",
    keyword: error.keyword,
    message: error.message,
  }));
}

export function createContractDispatcher({ root = ROOT } = {}) {
  const registry = new Map();
  const pr02CommonSchema = JSON.parse(
    readFileSync(resolve(root, PR02_COMMON_SCHEMA_PATH), "utf8"),
  );
  for (const entry of CONTRACT_ENTRIES) {
    const schema = JSON.parse(readFileSync(resolve(root, entry.relativePath), "utf8"));
    if (entry.semanticValidator === "pr00-episode") {
      registry.set(key(entry.version, entry.kind), {
        entry,
        validate(document) {
          const result = createPr00EpisodeValidator(schema)(document);
          return {
            schema_errors: result.schema_errors.map((error) => ({
              ...error,
              reason_code: "schema-invalid",
            })),
            semantic_errors: result.semantic_errors.map((error) => ({
              ...error,
              reason_code: error.category,
            })),
          };
        },
      });
      continue;
    }
    const ajv = new Ajv2020({
      allErrors: true,
      strict: true,
      strictNumbers: true,
      validateFormats: false,
    });
    if (entry.version === "0.3.0") {
      ajv.addSchema(pr02CommonSchema);
    }
    const validateSchema = ajv.compile(schema);
    const validateSemantics = SEMANTIC_VALIDATORS[entry.semanticValidator];
    registry.set(key(entry.version, entry.kind), {
      entry,
      validate(document) {
        const validSchema = validateSchema(document);
        return {
          schema_errors: validSchema ? [] : schemaErrors(validateSchema),
          semantic_errors: validSchema ? validateSemantics(document) : [],
        };
      },
    });
  }

  return function validateContract(document) {
    if (document === null || typeof document !== "object" || Array.isArray(document)) {
      return {
        valid: false,
        status: "invalid",
        reason_code: "contract-envelope-invalid",
        contract_key: null,
        schema_errors: [],
        semantic_errors: [],
      };
    }
    if (!SUPPORTED_VERSIONS.has(document.schema_version)) {
      return {
        valid: false,
        status: "invalid",
        reason_code: "unsupported-contract-version",
        contract_key: null,
        schema_errors: [],
        semantic_errors: [],
      };
    }
    const contractKey = key(document.schema_version, document.contract_kind);
    const registered = registry.get(contractKey);
    if (registered === undefined) {
      return {
        valid: false,
        status: "invalid",
        reason_code: "unsupported-contract-kind",
        contract_key: contractKey,
        schema_errors: [],
        semantic_errors: [],
      };
    }
    const result = registered.validate(document);
    const valid = result.schema_errors.length === 0 && result.semantic_errors.length === 0;
    return {
      valid,
      status: valid ? "valid" : "invalid",
      reason_code: valid
        ? "contract-valid"
        : result.schema_errors[0]?.reason_code ?? result.semantic_errors[0]?.reason_code,
      contract_key: contractKey,
      ...result,
    };
  };
}

export const validateContract = createContractDispatcher();
export const supportedContractKeys = Object.freeze(
  CONTRACT_ENTRIES.map((entry) => key(entry.version, entry.kind)),
);
