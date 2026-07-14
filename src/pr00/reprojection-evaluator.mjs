export const PRIMARY_ENDPOINT = "max_camera_reprojection_error_px";
export const PRIMARY_THRESHOLD_EXCLUSIVE_PX = 1;

function invalid(reason, details = {}) {
  return {
    endpoint: PRIMARY_ENDPOINT,
    threshold_exclusive_px: PRIMARY_THRESHOLD_EXCLUSIVE_PX,
    status: "invalid",
    reason,
    valid_point_count: 0,
    max_error_px: null,
    failures: [],
    ...details,
  };
}

export function evaluateReprojection({ episode, project }) {
  if (typeof project !== "function") {
    return invalid("projector-not-callable");
  }
  if (project.evaluatorReference === true) {
    return invalid("evaluator-projector-not-independent");
  }
  const points = episode?.audit?.primary_points;
  if (!Array.isArray(points) || points.length === 0) {
    return invalid("zero-primary-points");
  }
  const observations = new Map(
    episode.observations.map((observation) => [observation.observation_id, observation]),
  );
  const errors = [];
  const failures = [];
  for (const auditPoint of points) {
    const observation = observations.get(auditPoint.observation_id);
    if (observation === undefined) {
      failures.push({ case_id: auditPoint.case_id, reason: "missing-observation" });
      continue;
    }
    try {
      const projection = project({ observation, pointW: auditPoint.point_W_m });
      const dx = projection.pixel[0] - auditPoint.expected_pixel[0];
      const dy = projection.pixel[1] - auditPoint.expected_pixel[1];
      const errorPx = Math.hypot(dx, dy);
      if (!Number.isFinite(errorPx)) {
        failures.push({ case_id: auditPoint.case_id, reason: "non-finite-error" });
      } else {
        errors.push({ case_id: auditPoint.case_id, error_px: errorPx });
      }
    } catch (error) {
      failures.push({
        case_id: auditPoint.case_id,
        reason: error.code ?? "projection-error",
        message: error.message,
      });
    }
  }
  if (errors.length === 0) {
    return invalid("zero-valid-points", { failures });
  }
  if (failures.length > 0) {
    return invalid("primary-point-projection-failed", {
      valid_point_count: errors.length,
      failures,
    });
  }
  const maxErrorPx = Math.max(...errors.map((item) => item.error_px));
  return {
    endpoint: PRIMARY_ENDPOINT,
    threshold_exclusive_px: PRIMARY_THRESHOLD_EXCLUSIVE_PX,
    status: maxErrorPx < PRIMARY_THRESHOLD_EXCLUSIVE_PX ? "supported" : "rejected",
    reason: maxErrorPx < PRIMARY_THRESHOLD_EXCLUSIVE_PX
      ? "all-primary-points-below-threshold"
      : "at-least-one-primary-point-at-or-above-threshold",
    valid_point_count: errors.length,
    max_error_px: maxErrorPx,
    failures: errors.filter((item) => item.error_px >= PRIMARY_THRESHOLD_EXCLUSIVE_PX),
  };
}
