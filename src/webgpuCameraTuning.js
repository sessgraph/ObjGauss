export const WEBGPU_CAMERA_TUNING_MODE = "runtime-camera-tuning-v1";
export const WEBGPU_CAMERA_MODE_EDIT_FIXED = "edit-fixed";
export const WEBGPU_CAMERA_MODE_SPARK_FRAME = "spark-frame";
export const WEBGPU_CAMERA_MODE_DEFAULT = WEBGPU_CAMERA_MODE_EDIT_FIXED;
export const WEBGPU_CAMERA_MODES = Object.freeze([
  WEBGPU_CAMERA_MODE_EDIT_FIXED,
  WEBGPU_CAMERA_MODE_SPARK_FRAME,
]);

export function normalizeWebGpuCameraTuning(tuning = null) {
  const rawMode =
    typeof tuning === "string"
      ? tuning
      : tuning?.cameraMode ?? tuning?.mode ?? WEBGPU_CAMERA_MODE_DEFAULT;
  const cameraMode = WEBGPU_CAMERA_MODES.includes(String(rawMode))
    ? String(rawMode)
    : WEBGPU_CAMERA_MODE_DEFAULT;
  return {
    tuningMode: WEBGPU_CAMERA_TUNING_MODE,
    cameraMode,
  };
}
