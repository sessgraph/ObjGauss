import { projectWorldPoint } from "./frame-math.mjs";

export function projectEpisodePoint({ observation, pointW }) {
  if (observation.T_WC.availability !== "present" || observation.K.availability !== "present") {
    const error = new Error("projection requires present T_WC and K");
    error.code = "MISSING_CAMERA_CALIBRATION";
    throw error;
  }
  return projectWorldPoint({
    T_WC: observation.T_WC.value.matrix_row_major,
    K: observation.K.value.matrix_row_major,
    imageSizePx: observation.K.value.image_size_px,
    pointW,
    requireInBounds: true,
  });
}
