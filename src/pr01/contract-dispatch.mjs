import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import Ajv2020 from "ajv/dist/2020.js";
import { createEpisodeValidator as createPr00EpisodeValidator } from "../pr00/contract-validator.mjs";

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
];

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

const SEMANTIC_VALIDATORS = {
  "pr01-episode": validatePr01Episode,
  "pr01-experiment": validatePr01Experiment,
  "pr01-attempt": validatePr01Attempt,
  "pr01-invariance-report": validatePr01InvarianceReport,
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
