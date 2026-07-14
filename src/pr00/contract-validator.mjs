import Ajv2020 from "ajv/dist/2020.js";
import {
  FrameMathError,
  assertIntrinsics,
  assertRigidTransform,
  quaternionNorm,
} from "./frame-math.mjs";
import { isSafeResourceUri } from "./resource-uri.mjs";

const REQUIRED_NEGATIVE_CASES = new Set([
  "swapped_transform_direction",
  "row_column_vector_mix",
  "centimeter_meter_mix",
  "opencv_webgl_mix",
  "non_unit_quaternion",
  "behind_camera_point",
  "out_of_bounds_point",
  "singular_intrinsics",
  "hold_missing_action_mix",
  "duplicate_object_id",
  "non_monotonic_time",
  "broken_checksum",
  "broken_lineage",
  "sentinel_missing_value",
]);

function issue(path, category, message) {
  return { path, category, message };
}

function duplicates(values) {
  const seen = new Set();
  return values.filter((value) => {
    if (seen.has(value)) {
      return true;
    }
    seen.add(value);
    return false;
  });
}

function checkUnitVector(value, path, issues) {
  const norm = Math.hypot(...value);
  if (!Number.isFinite(norm) || Math.abs(norm - 1) > 1e-8) {
    issues.push(issue(path, "non-unit-vector", "vector norm must equal 1"));
  }
}

function checkQuaternion(value, path, issues) {
  try {
    const norm = quaternionNorm(value);
    if (Math.abs(norm - 1) > 1e-8) {
      issues.push(issue(path, "non-unit-quaternion", "quaternion norm must equal 1"));
      return;
    }
    const firstNonZero = value.find((component) => Math.abs(component) > 1e-9) ?? 1;
    if (firstNonZero < 0) {
      issues.push(issue(path, "non-canonical-quaternion", "first non-zero quaternion component must be positive"));
    }
  } catch (error) {
    issues.push(issue(path, error.code ?? "invalid-quaternion", error.message));
  }
}

function checkTransform(available, path, issues) {
  if (available.availability !== "present") {
    issues.push(issue(path, "required-measurement-missing", "synthetic-audit-v0 requires a present transform"));
    return;
  }
  try {
    assertRigidTransform(available.value.matrix_row_major, path);
  } catch (error) {
    issues.push(issue(path, error.code ?? "invalid-transform", error.message));
  }
}

function expectedResourceByteLength(descriptor) {
  const elements = descriptor.shape.reduce((product, value) => product * value, 1);
  return elements * (descriptor.dtype === "float32le" ? 4 : 1);
}

function checkResource(available, path, issues, resources) {
  if (available.availability !== "present") {
    issues.push(issue(path, "required-resource-missing", "synthetic-audit-v0 requires a present resource"));
    return;
  }
  const descriptor = available.value;
  if (!isSafeResourceUri(descriptor.uri)) {
    issues.push(issue(`${path}/value/uri`, "unsafe-resource-uri", "resource URI must be a safe relative path"));
  }
  const expectedDtype = descriptor.media_type === "application/vnd.objgauss.depth-f32le"
    ? "float32le"
    : "uint8";
  if (descriptor.dtype !== expectedDtype) {
    issues.push(issue(`${path}/value/dtype`, "media-dtype-mismatch", `media type requires ${expectedDtype}`));
  }
  const expectedChannels = descriptor.media_type === "application/vnd.objgauss.rgb8" ? 3 : null;
  if (expectedChannels !== null && descriptor.shape.at(-1) !== expectedChannels) {
    issues.push(issue(`${path}/value/shape`, "media-shape-mismatch", "rgb8 resource must end in 3 channels"));
  }
  const existing = resources.get(descriptor.uri);
  const signature = `${descriptor.sha256}:${expectedResourceByteLength(descriptor)}`;
  if (existing !== undefined && existing !== signature) {
    issues.push(issue(path, "resource-descriptor-conflict", "same URI has conflicting checksum or shape"));
  }
  resources.set(descriptor.uri, signature);
}

export function createEpisodeValidator(schema) {
  const ajv = new Ajv2020({
    allErrors: true,
    strict: true,
    strictNumbers: true,
    validateFormats: false,
  });
  const validateSchema = ajv.compile(schema);
  return function validateEpisode(episode) {
    const schemaValid = validateSchema(episode);
    const schemaErrors = schemaValid ? [] : validateSchema.errors.map((error) => ({
      path: error.instancePath || "/",
      category: `schema:${error.keyword}`,
      message: error.message ?? "schema validation failed",
      params: error.params,
    }));
    if (!schemaValid) {
      return { valid: false, status: "invalid", schema_errors: schemaErrors, semantic_errors: [] };
    }

    const semanticErrors = [];
    const resources = new Map();
    const observationIds = episode.observations.map((observation) => observation.observation_id);
    for (const duplicate of duplicates(observationIds)) {
      semanticErrors.push(issue("/observations", "duplicate-observation-id", duplicate));
    }
    if (episode.observations[0].episode_time_s !== 0) {
      semanticErrors.push(issue("/observations/0/episode_time_s", "invalid-time-origin", "first observation must start at 0.0"));
    }
    for (let index = 1; index < episode.observations.length; index += 1) {
      if (episode.observations[index].episode_time_s <= episode.observations[index - 1].episode_time_s) {
        semanticErrors.push(issue(`/observations/${index}/episode_time_s`, "non-monotonic-time", "observation time must be strictly increasing"));
      }
    }

    let canonicalObjectIds = null;
    episode.observations.forEach((observation, observationIndex) => {
      const path = `/observations/${observationIndex}`;
      checkResource(observation.rgb, `${path}/rgb`, semanticErrors, resources);
      checkResource(observation.depth, `${path}/depth`, semanticErrors, resources);
      checkTransform(observation.T_WC, `${path}/T_WC`, semanticErrors);
      if (observation.K.availability !== "present") {
        semanticErrors.push(issue(`${path}/K`, "required-measurement-missing", "synthetic-audit-v0 requires present intrinsics"));
      } else {
        try {
          assertIntrinsics(
            observation.K.value.matrix_row_major,
            observation.K.value.image_size_px,
          );
        } catch (error) {
          semanticErrors.push(issue(`${path}/K`, error.code ?? "invalid-intrinsics", error.message));
        }
      }
      const objectIds = observation.objects.map((object) => object.object_id);
      for (const duplicate of duplicates(objectIds)) {
        semanticErrors.push(issue(`${path}/objects`, "duplicate-object-id", duplicate));
      }
      if (canonicalObjectIds === null) {
        canonicalObjectIds = [...objectIds].sort();
      } else if (JSON.stringify([...objectIds].sort()) !== JSON.stringify(canonicalObjectIds)) {
        semanticErrors.push(issue(`${path}/objects`, "unstable-object-set", "object IDs must stay stable across the synthetic episode"));
      }
      observation.objects.forEach((object, objectIndex) => {
        const objectPath = `${path}/objects/${objectIndex}`;
        checkTransform(object.T_WO, `${objectPath}/T_WO`, semanticErrors);
        checkResource(object.mask, `${objectPath}/mask`, semanticErrors, resources);
        if (object.symmetry.availability !== "present") {
          semanticErrors.push(issue(`${objectPath}/symmetry`, "symmetry-blocked", "synthetic fixture requires explicit symmetry metadata"));
        } else if (object.symmetry.value.kind === "finite") {
          object.symmetry.value.rotations_wxyz.forEach((quaternion, quaternionIndex) => {
            checkQuaternion(quaternion, `${objectPath}/symmetry/value/rotations_wxyz/${quaternionIndex}`, semanticErrors);
          });
        } else if (object.symmetry.value.kind === "continuous_axis") {
          checkUnitVector(object.symmetry.value.axis_O, `${objectPath}/symmetry/value/axis_O`, semanticErrors);
        }
      });
    });

    const interventionIds = episode.interventions.map((intervention) => intervention.intervention_id);
    for (const duplicate of duplicates(interventionIds)) {
      semanticErrors.push(issue("/interventions", "duplicate-intervention-id", duplicate));
    }
    const lastObservationTime = episode.observations.at(-1).episode_time_s;
    episode.interventions.forEach((intervention, index) => {
      const path = `/interventions/${index}`;
      if (index > 0 && intervention.episode_time_s < episode.interventions[index - 1].episode_time_s) {
        semanticErrors.push(issue(`${path}/episode_time_s`, "non-monotonic-time", "intervention time must not decrease"));
      }
      if (intervention.episode_time_s > lastObservationTime) {
        semanticErrors.push(issue(`${path}/episode_time_s`, "out-of-episode-time", "intervention must not occur after the final observation"));
      }
      if (!canonicalObjectIds.includes(intervention.target_object_id)) {
        semanticErrors.push(issue(`${path}/target_object_id`, "unknown-target-object", intervention.target_object_id));
      }
      if (intervention.commanded_action.availability !== "present") {
        semanticErrors.push(issue(`${path}/commanded_action`, "missing-commanded-action", "commanded action must be present"));
      } else {
        const action = intervention.commanded_action.value;
        const magnitude = Math.hypot(...action.vector_W_N);
        if (action.kind === "hold" && magnitude !== 0) {
          semanticErrors.push(issue(`${path}/commanded_action`, "hold-is-nonzero", "hold must carry a zero force vector"));
        }
        if (action.kind === "push" && magnitude <= 0) {
          semanticErrors.push(issue(`${path}/commanded_action`, "push-is-zero", "push must carry a non-zero force vector"));
        }
      }
    });

    const lineageIds = episode.causal_lineage.map((record) => record.lineage_id);
    for (const duplicate of duplicates(lineageIds)) {
      semanticErrors.push(issue("/causal_lineage", "duplicate-lineage-id", duplicate));
    }
    const allowedLineageInputs = new Set([episode.fixture_id, ...interventionIds]);
    const allowedLineageOutputs = new Set(observationIds);
    episode.causal_lineage.forEach((record, index) => {
      const path = `/causal_lineage/${index}`;
      if (!allowedLineageOutputs.has(record.output_ref)) {
        semanticErrors.push(issue(`${path}/output_ref`, "broken-lineage-output", record.output_ref));
      }
      for (const inputRef of record.input_refs) {
        if (!allowedLineageInputs.has(inputRef)) {
          semanticErrors.push(issue(`${path}/input_refs`, "broken-lineage-input", inputRef));
        }
      }
      if (record.config_sha256 !== episode.producer.config_sha256) {
        semanticErrors.push(issue(`${path}/config_sha256`, "lineage-config-mismatch", "lineage and producer config checksums must match"));
      }
    });

    const auditCaseIds = episode.audit.primary_points.map((point) => point.case_id);
    for (const duplicate of duplicates(auditCaseIds)) {
      semanticErrors.push(issue("/audit/primary_points", "duplicate-audit-case-id", duplicate));
    }
    episode.audit.primary_points.forEach((point, index) => {
      if (!observationIds.includes(point.observation_id)) {
        semanticErrors.push(issue(`/audit/primary_points/${index}/observation_id`, "unknown-audit-observation", point.observation_id));
      }
    });
    for (const requiredCase of REQUIRED_NEGATIVE_CASES) {
      if (!episode.audit.required_negative_cases.includes(requiredCase)) {
        semanticErrors.push(issue("/audit/required_negative_cases", "missing-negative-case", requiredCase));
      }
    }

    return {
      valid: semanticErrors.length === 0,
      status: semanticErrors.length === 0 ? "valid" : "invalid",
      schema_errors: [],
      semantic_errors: semanticErrors,
      resource_descriptors: [...resources.entries()].map(([uri, signature]) => ({ uri, signature })),
    };
  };
}

export function validateEpisodeOrThrow(validateEpisode, episode) {
  const result = validateEpisode(episode);
  if (!result.valid) {
    const errors = [...result.schema_errors, ...result.semantic_errors]
      .map((error) => `${error.path} [${error.category}] ${error.message}`)
      .join("\n");
    const error = new Error(`episode validation failed:\n${errors}`);
    error.validation = result;
    throw error;
  }
  return result;
}

export function isFrameMathError(error) {
  return error instanceof FrameMathError;
}
