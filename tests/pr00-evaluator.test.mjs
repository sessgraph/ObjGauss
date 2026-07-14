import assert from "node:assert/strict";
import test from "node:test";
import { projectEpisodePoint } from "../src/pr00/projector.mjs";
import { evaluateReprojection } from "../src/pr00/reprojection-evaluator.mjs";
import { createSyntheticAudit } from "../src/pr00/synthetic-audit.mjs";

test("independent primary endpoint supports the frozen synthetic fixture", () => {
  const { episode } = createSyntheticAudit();
  const result = evaluateReprojection({ episode, project: projectEpisodePoint });
  assert.equal(result.status, "supported");
  assert.equal(result.valid_point_count, 36);
  assert.ok(result.max_error_px < 1e-9);
  assert.equal(result.threshold_exclusive_px, 1);
});

test("one pixel is rejected because the threshold is strict", () => {
  const { episode } = createSyntheticAudit();
  const shifted = (input) => {
    const projection = projectEpisodePoint(input);
    return { ...projection, pixel: [projection.pixel[0] + 1, projection.pixel[1]] };
  };
  const result = evaluateReprojection({ episode, project: shifted });
  assert.equal(result.status, "rejected");
  assert.ok(result.max_error_px >= 1);
  assert.ok(result.failures.length > 0);
});

test("zero points and a reference-marked projector are invalid, never pass", () => {
  const { episode } = createSyntheticAudit();
  const empty = structuredClone(episode);
  empty.audit.primary_points = [];
  assert.equal(evaluateReprojection({ episode: empty, project: projectEpisodePoint }).status, "invalid");
  const sharedReference = () => ({ pixel: [0, 0] });
  sharedReference.evaluatorReference = true;
  assert.equal(evaluateReprojection({ episode, project: sharedReference }).reason, "evaluator-projector-not-independent");
});
