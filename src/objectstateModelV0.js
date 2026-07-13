import { colorForObject } from "./palette.js";

export const OBJECTSTATE_MODEL_V0_STATE_SCHEMA = "objgauss-objectstate-model-v0-state-v1";
export const OBJECTSTATE_MODEL_V0_FEATURE_ORDER = [
  "x",
  "y",
  "z",
  "red",
  "green",
  "blue",
  "opacity",
];

export function inferObjectStateModelV0(points, checkpoint, precomputedPoints = null) {
  const model = validateObjectStateModelV0(checkpoint);
  if (!Array.isArray(points) || !points.length) {
    throw new Error("ObjectState Model v0 input PLY has no points");
  }
  if (precomputedPoints && precomputedPoints.length !== points.length) {
    throw new Error(
      `ObjectState Model v0 precomputed row mismatch: ${precomputedPoints.length}/${points.length}`,
    );
  }

  let agreementCount = 0;
  const inferredPoints = points.map((point, row) => {
    const probabilities = predictPoint(point, model);
    const objectId = argmax(probabilities);
    if (precomputedPoints && Number(precomputedPoints[row]?.objectId) === objectId) {
      agreementCount += 1;
    }
    return {
      ...point,
      objectId,
      objectColor: colorForObject(objectId),
      assignment: probabilities,
      confidence: probabilities[objectId],
      entropy: normalizedEntropy(probabilities),
      assignmentSource: OBJECTSTATE_MODEL_V0_STATE_SCHEMA,
    };
  });

  const precomputedAgreement = precomputedPoints
    ? agreementCount / inferredPoints.length
    : null;
  const telemetry = {
    schema: OBJECTSTATE_MODEL_V0_STATE_SCHEMA,
    modelFamily: model.model_family,
    rowCount: inferredPoints.length,
    objectCount: model.config.slots,
    precomputedRowCount: precomputedPoints?.length ?? null,
    agreementCount: precomputedPoints ? agreementCount : null,
    precomputedAgreement,
    status: precomputedPoints && agreementCount === inferredPoints.length
      ? "checkpoint_inference_match"
      : precomputedPoints
        ? "checkpoint_inference_mismatch"
        : "checkpoint_inference_uncompared",
  };
  if (telemetry.status === "checkpoint_inference_mismatch") {
    throw new Error(
      `ObjectState Model v0 checkpoint/precomputed mismatch: ${agreementCount}/${inferredPoints.length}`,
    );
  }
  return { points: inferredPoints, telemetry };
}

export function validateObjectStateModelV0(checkpoint) {
  if (!checkpoint || typeof checkpoint !== "object") {
    throw new Error("ObjectState Model v0 checkpoint must be an object");
  }
  if (checkpoint.schema !== OBJECTSTATE_MODEL_V0_STATE_SCHEMA) {
    throw new Error(`unsupported ObjectState Model v0 schema: ${checkpoint.schema ?? "missing"}`);
  }
  if (
    !Array.isArray(checkpoint.feature_order) ||
    checkpoint.feature_order.join(",") !== OBJECTSTATE_MODEL_V0_FEATURE_ORDER.join(",")
  ) {
    throw new Error("ObjectState Model v0 feature order mismatch");
  }
  const config = checkpoint.config;
  const slots = positiveInteger(config?.slots, "config.slots");
  const inputDim = positiveInteger(config?.input_dim, "config.input_dim");
  const hiddenDim = positiveInteger(config?.hidden_dim, "config.hidden_dim");
  if (inputDim !== OBJECTSTATE_MODEL_V0_FEATURE_ORDER.length) {
    throw new Error("ObjectState Model v0 input dimension mismatch");
  }
  const featureMean = finiteVector(checkpoint.feature_mean, inputDim, "feature_mean");
  const featureStd = finiteVector(checkpoint.feature_std, inputDim, "feature_std");
  if (featureStd.some((value) => value <= 0)) {
    throw new Error("ObjectState Model v0 feature_std must be positive");
  }
  return {
    ...checkpoint,
    config: { ...config, slots, input_dim: inputDim, hidden_dim: hiddenDim },
    feature_mean: featureMean,
    feature_std: featureStd,
    encoder_weight: finiteMatrix(checkpoint.encoder_weight, inputDim, hiddenDim, "encoder_weight"),
    encoder_bias: finiteVector(checkpoint.encoder_bias, hiddenDim, "encoder_bias"),
    assignment_weight: finiteMatrix(
      checkpoint.assignment_weight,
      hiddenDim,
      slots,
      "assignment_weight",
    ),
    assignment_bias: finiteVector(checkpoint.assignment_bias, slots, "assignment_bias"),
  };
}

function predictPoint(point, model) {
  const rgb = Array.isArray(point.modelFeatureRgb)
    ? point.modelFeatureRgb
    : (point.color ?? [0, 0, 0]).map((value) => Number(value) / 255);
  const features = [
    Number(point.x),
    Number(point.y),
    Number(point.z),
    Number(rgb[0]),
    Number(rgb[1]),
    Number(rgb[2]),
    Number(point.modelFeatureOpacity ?? point.opacity ?? 1),
  ];
  if (!features.every(Number.isFinite)) {
    throw new Error("ObjectState Model v0 input contains non-finite features");
  }
  const normalized = features.map(
    (value, index) => (value - model.feature_mean[index]) / model.feature_std[index],
  );
  const hidden = model.encoder_bias.map((bias, column) => {
    let value = bias;
    for (let row = 0; row < normalized.length; row += 1) {
      value += normalized[row] * model.encoder_weight[row][column];
    }
    return Math.tanh(value);
  });
  const logits = model.assignment_bias.map((bias, column) => {
    let value = bias;
    for (let row = 0; row < hidden.length; row += 1) {
      value += hidden[row] * model.assignment_weight[row][column];
    }
    return value;
  });
  return softmax(logits);
}

function softmax(values) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}

function normalizedEntropy(probabilities) {
  if (probabilities.length <= 1) return 0;
  const entropy = -probabilities.reduce(
    (sum, value) => sum + (value > 0 ? value * Math.log(value) : 0),
    0,
  );
  return entropy / Math.log(probabilities.length);
}

function positiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isInteger(number) || number <= 0) {
    throw new Error(`ObjectState Model v0 ${name} must be a positive integer`);
  }
  return number;
}

function finiteVector(values, length, name) {
  if (!Array.isArray(values) || values.length !== length) {
    throw new Error(`ObjectState Model v0 ${name} shape mismatch`);
  }
  const normalized = values.map(Number);
  if (!normalized.every(Number.isFinite)) {
    throw new Error(`ObjectState Model v0 ${name} must be finite`);
  }
  return normalized;
}

function finiteMatrix(values, rows, columns, name) {
  if (!Array.isArray(values) || values.length !== rows) {
    throw new Error(`ObjectState Model v0 ${name} shape mismatch`);
  }
  return values.map((row, index) => finiteVector(row, columns, `${name}[${index}]`));
}
