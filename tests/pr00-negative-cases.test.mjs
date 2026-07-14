import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createEpisodeValidator } from "../src/pr00/contract-validator.mjs";
import { invertRigidTransform } from "../src/pr00/frame-math.mjs";
import { projectEpisodePoint } from "../src/pr00/projector.mjs";
import { evaluateReprojection } from "../src/pr00/reprojection-evaluator.mjs";
import { auditResources } from "../src/pr00/resource-audit.mjs";
import { createSyntheticAudit } from "../src/pr00/synthetic-audit.mjs";

const schema = JSON.parse(readFileSync("contracts/objgauss/0.1.0/episode.schema.json"));
const manifest = JSON.parse(readFileSync("contracts/fixtures/synthetic-audit-v0.manifest.json"));
const validate = createEpisodeValidator(schema);

function invalidAfter(mutator) {
  const { episode } = createSyntheticAudit();
  mutator(episode);
  return !validate(episode).valid;
}

test("the complete preregistered negative-case matrix fails closed", () => {
  const results = new Map();

  {
    const { episode } = createSyntheticAudit();
    episode.observations[0].T_WC.value.matrix_row_major = invertRigidTransform(
      episode.observations[0].T_WC.value.matrix_row_major,
    );
    results.set("swapped_transform_direction", evaluateReprojection({ episode, project: projectEpisodePoint }).status !== "supported");
  }
  results.set("row_column_vector_mix", invalidAfter((episode) => {
    const matrix = episode.observations[0].T_WC.value.matrix_row_major;
    episode.observations[0].T_WC.value.matrix_row_major = [
      matrix[0], matrix[4], matrix[8], matrix[12],
      matrix[1], matrix[5], matrix[9], matrix[13],
      matrix[2], matrix[6], matrix[10], matrix[14],
      matrix[3], matrix[7], matrix[11], matrix[15],
    ];
  }));
  results.set("centimeter_meter_mix", invalidAfter((episode) => {
    episode.coordinate_convention.length_unit = "centimeter";
  }));
  results.set("opencv_webgl_mix", invalidAfter((episode) => {
    episode.coordinate_convention.camera_axes = "+X-right,+Y-up,-Z-forward";
  }));
  results.set("non_unit_quaternion", invalidAfter((episode) => {
    episode.observations[0].objects[0].symmetry = {
      availability: "present",
      value: { kind: "finite", rotations_wxyz: [[2, 0, 0, 0]] },
    };
  }));

  {
    const { episode } = createSyntheticAudit();
    const observation = episode.observations[0];
    const matrix = observation.T_WC.value.matrix_row_major;
    const behind = [
      matrix[3] - matrix[2],
      matrix[7] - matrix[6],
      matrix[11] - matrix[10],
    ];
    assert.throws(() => projectEpisodePoint({ observation, pointW: behind }), /positive OpenCV camera depth/);
    results.set("behind_camera_point", true);
  }
  {
    const { episode } = createSyntheticAudit();
    const observation = episode.observations[0];
    const matrix = observation.T_WC.value.matrix_row_major;
    const outside = [
      matrix[3] + matrix[0] * 100 + matrix[2],
      matrix[7] + matrix[4] * 100 + matrix[6],
      matrix[11] + matrix[8] * 100 + matrix[10],
    ];
    assert.throws(() => projectEpisodePoint({ observation, pointW: outside }), /inside the image/);
    results.set("out_of_bounds_point", true);
  }
  results.set("singular_intrinsics", invalidAfter((episode) => {
    episode.observations[0].K.value.matrix_row_major = new Array(9).fill(0);
  }));
  results.set("hold_missing_action_mix", invalidAfter((episode) => {
    episode.interventions[0].commanded_action = { availability: "missing", reason: "not_provided" };
  }));
  results.set("duplicate_object_id", invalidAfter((episode) => {
    episode.observations[0].objects[1].object_id = episode.observations[0].objects[0].object_id;
  }));
  results.set("non_monotonic_time", invalidAfter((episode) => {
    episode.observations[1].episode_time_s = 0;
  }));
  {
    const { resources } = createSyntheticAudit();
    resources[0].bytes[0] ^= 0xff;
    results.set("broken_checksum", auditResources({ manifest, resources }).status === "invalid");
  }
  results.set("broken_lineage", invalidAfter((episode) => {
    episode.causal_lineage[0].input_refs = ["unknown-input"];
  }));
  results.set("sentinel_missing_value", invalidAfter((episode) => {
    episode.interventions[1].executed_action.value = {
      kind: "hold",
      vector_W_N: [0, 0, 0],
      duration_s: 0.2,
      source: { kind: "executed", ref: "intervention-push" },
    };
  }));

  const { episode } = createSyntheticAudit();
  assert.deepEqual([...results.keys()].sort(), [...episode.audit.required_negative_cases].sort());
  for (const [caseId, rejected] of results) {
    assert.equal(rejected, true, `${caseId} did not fail closed`);
  }
});
