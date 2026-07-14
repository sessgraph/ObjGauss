import { canonicalStringify } from "./canonical-json.mjs";
import { sha256Hex } from "./node-hash.mjs";

export const SYNTHETIC_AUDIT_SEED = 0x50_52_30_30;
export const SYNTHETIC_AUDIT_VERSION = "0.1.0";
export const SYNTHETIC_AUDIT_FIXTURE_ID = "synthetic-audit-v0";

const WIDTH = 64;
const HEIGHT = 48;
const K = [58, 0, 31.5, 0, 58, 23.5, 0, 0, 1];
const FRAME_CONFIGS = [
  { id: "observation-000", time: 0, eye: [0, -6, 3], target: [0, 0.65, 0.75] },
  { id: "observation-001", time: 1, eye: [3, -5, 2.5], target: [0.15, 0.7, 0.7] },
  { id: "observation-002", time: 2, eye: [-3, -4.5, 2.2], target: [0.1, 0.85, 0.7] },
];

const PRIMARY_WORLD_POINTS = [
  [-1.4, -0.1, 0.2], [-0.8, 0.2, 0.5], [0, 0, 0.15], [0.8, 0.2, 0.5],
  [1.4, -0.1, 0.2], [-1.2, 1.1, 0.85], [-0.4, 1.5, 1.35], [0.4, 1.5, 1.35],
  [1.2, 1.1, 0.85], [-1.5, 2.1, 0.35], [0, 2.4, 1.1], [1.5, 2.1, 0.35],
];

const FIXTURE_CONFIG = Object.freeze({
  fixture_id: SYNTHETIC_AUDIT_FIXTURE_ID,
  seed: SYNTHETIC_AUDIT_SEED,
  version: SYNTHETIC_AUDIT_VERSION,
  image_size_px: [WIDTH, HEIGHT],
  intrinsics_row_major: K,
  frames: FRAME_CONFIGS,
  primary_world_points_m: PRIMARY_WORLD_POINTS,
  objects: ["object-asymmetric", "object-axial"],
});

function normalize(vector) {
  const norm = Math.hypot(...vector);
  if (!Number.isFinite(norm) || norm <= 1e-12) {
    throw new Error("reference vector must have non-zero finite norm");
  }
  return vector.map((value) => value / norm);
}

function subtract(left, right) {
  return left.map((value, index) => value - right[index]);
}

function cross(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

// Reference-only camera construction. It deliberately does not import frame-math.mjs.
function referenceCameraToWorld(eye, target) {
  const forward = normalize(subtract(target, eye));
  const right = normalize(cross(forward, [0, 0, 1]));
  const down = normalize(cross(forward, right));
  return [
    right[0], down[0], forward[0], eye[0],
    right[1], down[1], forward[1], eye[1],
    right[2], down[2], forward[2], eye[2],
    0, 0, 0, 1,
  ];
}

function referenceRigidTransformZ(angleRad, translation) {
  const cosine = Math.cos(angleRad);
  const sine = Math.sin(angleRad);
  return [
    cosine, -sine, 0, translation[0],
    sine, cosine, 0, translation[1],
    0, 0, 1, translation[2],
    0, 0, 0, 1,
  ];
}

// Reference-only projection used to create immutable GT pixels.
function referenceProject(T_WC, pointW) {
  const dx = pointW[0] - T_WC[3];
  const dy = pointW[1] - T_WC[7];
  const dz = pointW[2] - T_WC[11];
  const x = T_WC[0] * dx + T_WC[4] * dy + T_WC[8] * dz;
  const y = T_WC[1] * dx + T_WC[5] * dy + T_WC[9] * dz;
  const z = T_WC[2] * dx + T_WC[6] * dy + T_WC[10] * dz;
  if (z <= 0) {
    throw new Error("reference primary point fell behind the camera");
  }
  const pixel = [
    (K[0] * x + K[1] * y + K[2] * z) / z,
    (K[3] * x + K[4] * y + K[5] * z) / z,
  ];
  if (pixel[0] < 0 || pixel[1] < 0 || pixel[0] >= WIDTH || pixel[1] >= HEIGHT) {
    throw new Error(`reference primary point fell outside the image: ${pixel.join(",")}`);
  }
  return pixel;
}
referenceProject.evaluatorReference = true;

function present(value) {
  return { availability: "present", value };
}

function missing(reason) {
  return { availability: "missing", reason };
}

function source(kind, ref) {
  return { kind, ref };
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function makeRgb(frameIndex) {
  const random = seededRandom(SYNTHETIC_AUDIT_SEED + frameIndex * 0x9e37);
  const bytes = new Uint8Array(WIDTH * HEIGHT * 3);
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const offset = (y * WIDTH + x) * 3;
      const horizon = y / (HEIGHT - 1);
      const noise = Math.floor(random() * 5);
      bytes[offset] = Math.min(255, 24 + frameIndex * 12 + x * 2 + noise);
      bytes[offset + 1] = Math.min(255, 52 + Math.floor(horizon * 95) + noise);
      bytes[offset + 2] = Math.min(255, 88 + Math.floor((1 - horizon) * 90) + frameIndex * 8);
    }
  }
  const centers = frameIndex === 0 ? [[22, 29], [42, 28]]
    : frameIndex === 1 ? [[20, 29], [44, 27]]
      : [[24, 28], [45, 29]];
  const colors = [[205, 255, 99], [103, 230, 211]];
  for (let objectIndex = 0; objectIndex < centers.length; objectIndex += 1) {
    const [cx, cy] = centers[objectIndex];
    for (let y = cy - 5; y <= cy + 5; y += 1) {
      for (let x = cx - 5; x <= cx + 5; x += 1) {
        if (x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT || (x - cx) ** 2 + (y - cy) ** 2 > 25) {
          continue;
        }
        const offset = (y * WIDTH + x) * 3;
        bytes.set(colors[objectIndex], offset);
      }
    }
  }
  return bytes;
}

function makeDepth(frameIndex) {
  const bytes = new Uint8Array(WIDTH * HEIGHT * 4);
  const view = new DataView(bytes.buffer);
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const depth = 5.2 + frameIndex * 0.15 + y * 0.012 + Math.abs(x - WIDTH / 2) * 0.004;
      view.setFloat32((y * WIDTH + x) * 4, depth, true);
    }
  }
  return bytes;
}

function makeMask(frameIndex, objectIndex) {
  const bytes = new Uint8Array(WIDTH * HEIGHT);
  const centers = frameIndex === 0 ? [[22, 29], [42, 28]]
    : frameIndex === 1 ? [[20, 29], [44, 27]]
      : [[24, 28], [45, 29]];
  const [cx, cy] = centers[objectIndex];
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= 25) {
        bytes[y * WIDTH + x] = 255;
      }
    }
  }
  return bytes;
}

function resourceDescriptor(uri, mediaType, dtype, shape, bytes) {
  return {
    uri,
    media_type: mediaType,
    dtype,
    shape,
    sha256: sha256Hex(bytes),
    source: source("synthetic_producer", SYNTHETIC_AUDIT_FIXTURE_ID),
  };
}

function measurement(data) {
  return present({ data, confidence: 1, source: source("oracle", SYNTHETIC_AUDIT_FIXTURE_ID) });
}

function transformMeasurement(matrix) {
  return present({
    matrix_row_major: matrix,
    confidence: 1,
    source: source("oracle", SYNTHETIC_AUDIT_FIXTURE_ID),
  });
}

function createObject(frameIndex, objectIndex, maskDescriptor) {
  const asymmetric = objectIndex === 0;
  const position = asymmetric
    ? [-0.8 + frameIndex * 0.05, 0.35 + frameIndex * 0.08, 0.5]
    : [0.8 + Math.max(0, frameIndex - 1) * 0.28, 0.55, 0.6];
  const angle = asymmetric ? frameIndex * 0.08 : frameIndex * 0.22;
  return {
    object_id: asymmetric ? "object-asymmetric" : "object-axial",
    canonical_frame: {
      origin: "center_of_mass",
      axes_source: "producer_authored",
      handedness: "right",
    },
    symmetry: present(asymmetric
      ? { kind: "none" }
      : { kind: "continuous_axis", axis_O: [0, 0, 1] }),
    T_WO: transformMeasurement(referenceRigidTransformZ(angle, position)),
    linear_velocity_W_mps: measurement(asymmetric ? [0.05, 0.08, 0] : [frameIndex < 2 ? 0 : 0.28, 0, 0]),
    angular_velocity_W_radps: measurement(asymmetric ? [0, 0, 0.08] : [0, 0, 0.22]),
    visibility: measurement(1),
    existence: measurement(true),
    mask: present(maskDescriptor),
    estimate_phase: "oracle",
  };
}

export function createSyntheticAudit() {
  const configSha256 = sha256Hex(canonicalStringify(FIXTURE_CONFIG));
  const resources = [];
  const observations = FRAME_CONFIGS.map((frame, frameIndex) => {
    const rgbBytes = makeRgb(frameIndex);
    const depthBytes = makeDepth(frameIndex);
    const rgbUri = `assets/frame-${String(frameIndex).padStart(3, "0")}.rgb8`;
    const depthUri = `assets/frame-${String(frameIndex).padStart(3, "0")}.depth-f32le`;
    const rgbDescriptor = resourceDescriptor(
      rgbUri,
      "application/vnd.objgauss.rgb8",
      "uint8",
      [HEIGHT, WIDTH, 3],
      rgbBytes,
    );
    const depthDescriptor = resourceDescriptor(
      depthUri,
      "application/vnd.objgauss.depth-f32le",
      "float32le",
      [HEIGHT, WIDTH],
      depthBytes,
    );
    resources.push({ uri: rgbUri, bytes: rgbBytes }, { uri: depthUri, bytes: depthBytes });
    const masks = [0, 1].map((objectIndex) => {
      const bytes = makeMask(frameIndex, objectIndex);
      const uri = `assets/frame-${String(frameIndex).padStart(3, "0")}-object-${objectIndex}.mask-u8`;
      resources.push({ uri, bytes });
      return resourceDescriptor(
        uri,
        "application/vnd.objgauss.mask-u8",
        "uint8",
        [HEIGHT, WIDTH],
        bytes,
      );
    });
    const T_WC = referenceCameraToWorld(frame.eye, frame.target);
    return {
      observation_id: frame.id,
      episode_time_s: frame.time,
      rgb: present(rgbDescriptor),
      depth: present(depthDescriptor),
      K: present({
        matrix_row_major: K,
        image_size_px: [WIDTH, HEIGHT],
        confidence: 1,
        source: source("oracle", SYNTHETIC_AUDIT_FIXTURE_ID),
      }),
      T_WC: transformMeasurement(T_WC),
      objects: [createObject(frameIndex, 0, masks[0]), createObject(frameIndex, 1, masks[1])],
    };
  });

  const primaryPoints = [];
  for (const observation of observations) {
    const T_WC = observation.T_WC.value.matrix_row_major;
    PRIMARY_WORLD_POINTS.forEach((pointW, pointIndex) => {
      primaryPoints.push({
        case_id: `${observation.observation_id}-point-${String(pointIndex).padStart(2, "0")}`,
        observation_id: observation.observation_id,
        point_W_m: pointW,
        expected_pixel: referenceProject(T_WC, pointW),
      });
    });
  }

  const episode = {
    schema_version: "0.1.0",
    contract_kind: "objgauss.episode",
    episode_id: "episode-synthetic-audit-v0",
    sibling_group_id: "sibling-synthetic-audit-v0",
    fixture_id: SYNTHETIC_AUDIT_FIXTURE_ID,
    producer: {
      name: "objgauss.synthetic-audit",
      version: SYNTHETIC_AUDIT_VERSION,
      seed: SYNTHETIC_AUDIT_SEED,
      fixture_spec: SYNTHETIC_AUDIT_FIXTURE_ID,
      config_sha256: configSha256,
    },
    license_boundary: {
      project_policy: "all-rights-reserved",
      distribution: "internal-only",
      archive_migration: "none",
    },
    coordinate_convention: {
      id: "robotics-opencv-v1",
      matrix_storage: "row-major",
      vector_action: "column-vector-left-multiply",
      transform_notation: "T_AB-maps-B-to-A",
      world_handedness: "right",
      world_up: "+Z",
      length_unit: "meter",
      camera_axes: "+X-right,+Y-down,+Z-forward",
      quaternion_order: "wxyz",
      time_field: "episode_time_s",
      time_unit: "second",
    },
    observations,
    interventions: [
      {
        intervention_id: "intervention-hold",
        episode_time_s: 0.5,
        target_object_id: "object-asymmetric",
        commanded_action: present({
          kind: "hold",
          vector_W_N: [0, 0, 0],
          duration_s: 0.25,
          source: source("commanded", "intervention-hold"),
        }),
        executed_action: present({
          kind: "hold",
          vector_W_N: [0, 0, 0],
          duration_s: 0.25,
          source: source("executed", "intervention-hold"),
        }),
      },
      {
        intervention_id: "intervention-push",
        episode_time_s: 1,
        target_object_id: "object-axial",
        commanded_action: present({
          kind: "push",
          vector_W_N: [2, 0, 0],
          duration_s: 0.2,
          source: source("commanded", "intervention-push"),
        }),
        executed_action: missing("not_measured"),
      },
    ],
    causal_lineage: observations.map((observation, index) => ({
      lineage_id: `lineage-${observation.observation_id}`,
      output_ref: observation.observation_id,
      input_refs: index === 0
        ? [SYNTHETIC_AUDIT_FIXTURE_ID]
        : [SYNTHETIC_AUDIT_FIXTURE_ID, index === 1 ? "intervention-hold" : "intervention-push"],
      producer_version: SYNTHETIC_AUDIT_VERSION,
      transform_version: "pr00-frame-math-0.1.0",
      config_sha256: configSha256,
    })),
    audit: {
      primary_endpoint: "max_camera_reprojection_error_px",
      threshold_exclusive_px: 1,
      primary_points: primaryPoints,
      required_negative_cases: [
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
        "sentinel_missing_value"
      ],
    },
  };

  return { episode, resources, config: FIXTURE_CONFIG };
}

export const REFERENCE_PROJECTOR_MARKER = referenceProject.evaluatorReference;
