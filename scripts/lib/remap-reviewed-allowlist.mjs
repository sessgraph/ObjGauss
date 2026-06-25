import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

export const REVIEWED_ALLOWLIST_MODE = "object-boundary-remap-reviewed-allowlist-v1";
export const REVIEWED_ALLOWLIST_OWNER_DECISION = "approved-for-policy-gated-remap";

const STRICT_MAX_AFTER_COVERAGE_DISTANCE = 0.08;
const STRICT_MAX_AFTER_LUMA_DELTA = 0.04;
const STRICT_MAX_AFTER_CHROMA_DELTA = 0.04;

export function loadReviewedAllowlistManifest(inputPath, options = {}) {
  if (!inputPath) return null;
  if (!existsSync(inputPath)) {
    throw new Error(`reviewed allowlist is missing: ${inputPath}`);
  }
  const raw = JSON.parse(readFileSync(inputPath, "utf-8"));
  return validateReviewedAllowlistManifest(raw, {
    ...options,
    manifestPath: inputPath,
  });
}

export function validateReviewedAllowlistManifest(raw, options = {}) {
  const manifestPath = options.manifestPath ?? "";
  const errors = [];
  if (raw?.mode !== REVIEWED_ALLOWLIST_MODE) {
    errors.push(`unsupported reviewed allowlist mode: ${raw?.mode}`);
  }
  if (!Number.isInteger(raw?.version) || raw.version < 1) {
    errors.push("version must be a positive integer");
  }
  if ((raw?.defaultAction ?? "keep-hard-mask") !== "keep-hard-mask") {
    errors.push("defaultAction must be keep-hard-mask");
  }
  if (!Array.isArray(raw?.targets)) {
    errors.push("targets must be an array");
  }

  const targets = [];
  const seen = new Set();
  for (const [index, target] of (raw?.targets ?? []).entries()) {
    if (target?.approved === false) continue;
    const normalized = validateReviewedAllowlistTarget(target, {
      index,
      errors,
      rootDir: options.rootDir ?? process.cwd(),
      requireExistingEvidence: Boolean(options.requireExistingEvidence),
    });
    if (!normalized) continue;
    const key = `${normalized.assetId}:${normalized.targetObjectId}`;
    if (seen.has(key)) {
      errors.push(`targets[${index}] duplicates approved target ${key}`);
    }
    seen.add(key);
    targets.push(normalized);
  }

  if (errors.length > 0) {
    throw new Error(`invalid reviewed allowlist${manifestPath ? ` ${manifestPath}` : ""}: ${errors.join("; ")}`);
  }

  return {
    path: manifestPath || null,
    raw,
    targets,
    summary: {
      mode: raw.mode,
      path: manifestPath || null,
      defaultAction: raw.defaultAction ?? "keep-hard-mask",
      targetCount: targets.length,
      targets: targets.map((target) => ({
        assetId: target.assetId,
        targetObjectId: target.targetObjectId,
      })),
    },
  };
}

function validateReviewedAllowlistTarget(target, options) {
  const prefix = `targets[${options.index}]`;
  const assetId = stringOrEmpty(target?.assetId);
  const targetObjectId = Number(target?.targetObjectId);
  const errors = options.errors;

  if (!assetId) errors.push(`${prefix}.assetId is required`);
  if (!Number.isInteger(targetObjectId) || targetObjectId < 0) {
    errors.push(`${prefix}.targetObjectId must be a non-negative integer`);
  }
  requireNonEmptyString(errors, `${prefix}.reviewer`, target?.reviewer);
  requireDateString(errors, `${prefix}.reviewedAt`, target?.reviewedAt);
  requireNonEmptyString(errors, `${prefix}.reason`, target?.reason);

  const ownerApproval = target?.ownerApproval;
  if (!ownerApproval || typeof ownerApproval !== "object") {
    errors.push(`${prefix}.ownerApproval is required`);
  } else {
    requireNonEmptyString(errors, `${prefix}.ownerApproval.approvedBy`, ownerApproval.approvedBy);
    requireDateString(errors, `${prefix}.ownerApproval.approvedAt`, ownerApproval.approvedAt);
    if (ownerApproval.decision !== REVIEWED_ALLOWLIST_OWNER_DECISION) {
      errors.push(
        `${prefix}.ownerApproval.decision must be ${REVIEWED_ALLOWLIST_OWNER_DECISION}`,
      );
    }
  }

  const evidence = target?.evidence;
  if (!evidence || typeof evidence !== "object") {
    errors.push(`${prefix}.evidence is required`);
  } else {
    if (evidence.policyDecision !== "allowlist-candidate") {
      errors.push(`${prefix}.evidence.policyDecision must be allowlist-candidate`);
    }
    validateEvidencePath(errors, `${prefix}.evidence.policyReport`, evidence.policyReport, options);
    validateEvidencePath(errors, `${prefix}.evidence.residualReport`, evidence.residualReport, options);
    validateEvidencePath(errors, `${prefix}.evidence.originalScreenshot`, evidence.originalScreenshot, options);
    validateEvidencePath(
      errors,
      `${prefix}.evidence.remapPreviewScreenshot`,
      evidence.remapPreviewScreenshot,
      options,
    );
    requireNumber(errors, `${prefix}.evidence.hiddenGaussianDelta`, evidence.hiddenGaussianDelta, {
      maxExclusive: 0,
    });
    requireNumber(errors, `${prefix}.evidence.hiddenGaussianDeltaShare`, evidence.hiddenGaussianDeltaShare, {
      max: 0,
    });
    validateAfterDelta(errors, `${prefix}.evidence.afterDelta`, evidence.afterDelta);
    validateDeltaTriple(errors, `${prefix}.evidence.deleteDeltaChange`, evidence.deleteDeltaChange);
  }

  if (errors.length > 0 || !assetId || !Number.isInteger(targetObjectId)) return null;
  return {
    assetId,
    targetObjectId,
    reviewer: stringOrEmpty(target.reviewer),
    reviewedAt: stringOrEmpty(target.reviewedAt),
    reason: stringOrEmpty(target.reason),
    ownerApproval,
    evidence,
  };
}

function validateEvidencePath(errors, label, value, options) {
  const pathValue = stringOrEmpty(value);
  if (!pathValue) {
    errors.push(`${label} is required`);
    return;
  }
  if (!options.requireExistingEvidence) return;
  if (path.isAbsolute(pathValue)) {
    errors.push(`${label} must be a repository-relative path`);
    return;
  }
  const fullPath = path.resolve(options.rootDir, pathValue);
  if (!existsSync(fullPath)) {
    errors.push(`${label} does not exist: ${pathValue}`);
  }
}

function validateAfterDelta(errors, label, value) {
  validateDeltaTriple(errors, label, value);
  if (!value || typeof value !== "object") return;
  requireNumber(errors, `${label}.coverageRatio`, value.coverageRatio, {
    min: 1,
    max: 1 + STRICT_MAX_AFTER_COVERAGE_DISTANCE,
  });
  requireNumber(errors, `${label}.lumaDelta`, value.lumaDelta, {
    min: 0,
    max: STRICT_MAX_AFTER_LUMA_DELTA,
  });
  requireNumber(errors, `${label}.chromaDelta`, value.chromaDelta, {
    min: 0,
    max: STRICT_MAX_AFTER_CHROMA_DELTA,
  });
}

function validateDeltaTriple(errors, label, value) {
  if (!value || typeof value !== "object") {
    errors.push(`${label} is required`);
    return;
  }
  requireNumber(errors, `${label}.coverageRatio`, value.coverageRatio);
  requireNumber(errors, `${label}.lumaDelta`, value.lumaDelta);
  requireNumber(errors, `${label}.chromaDelta`, value.chromaDelta);
}

function requireNonEmptyString(errors, label, value) {
  if (!stringOrEmpty(value)) errors.push(`${label} is required`);
}

function requireDateString(errors, label, value) {
  const text = stringOrEmpty(value);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) {
    errors.push(`${label} must be YYYY-MM-DD`);
  }
}

function requireNumber(errors, label, value, constraints = {}) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    errors.push(`${label} must be finite`);
    return;
  }
  if (constraints.min !== undefined && number < constraints.min) {
    errors.push(`${label} must be >= ${constraints.min}`);
  }
  if (constraints.max !== undefined && number > constraints.max) {
    errors.push(`${label} must be <= ${constraints.max}`);
  }
  if (constraints.maxExclusive !== undefined && number >= constraints.maxExclusive) {
    errors.push(`${label} must be < ${constraints.maxExclusive}`);
  }
}

function stringOrEmpty(value) {
  return typeof value === "string" ? value.trim() : "";
}
