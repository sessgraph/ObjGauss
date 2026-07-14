function projectSchematic(point, width, height) {
  const scale = Math.min(width / 16, height / 7.5);
  return [
    width * 0.5 + (point[0] - point[1] * 0.58) * scale,
    height * 0.72 - point[2] * scale + point[1] * scale * 0.22,
  ];
}

function transformPoint(matrix, point) {
  return [
    matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
    matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
    matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
  ];
}

function drawLine(context, from, to, color, width = 1) {
  context.beginPath();
  context.moveTo(...from);
  context.lineTo(...to);
  context.strokeStyle = color;
  context.lineWidth = width;
  context.stroke();
}

function drawAxes(context, width, height) {
  const origin = projectSchematic([0, 0, 0], width, height);
  const axes = [
    { point: [1, 0, 0], color: "#ff7958", label: "+X" },
    { point: [0, 1, 0], color: "#67e6d3", label: "+Y" },
    { point: [0, 0, 1], color: "#c8ff63", label: "+Z" },
  ];
  for (const axis of axes) {
    const endpoint = projectSchematic(axis.point, width, height);
    drawLine(context, origin, endpoint, axis.color, 2);
    context.fillStyle = axis.color;
    context.font = "11px ui-monospace, monospace";
    context.fillText(axis.label, endpoint[0] + 4, endpoint[1]);
  }
}

function drawGround(context, width, height) {
  for (let index = -3; index <= 3; index += 1) {
    drawLine(
      context,
      projectSchematic([index, -1, 0], width, height),
      projectSchematic([index, 3, 0], width, height),
      "rgba(242,241,232,0.08)",
    );
    drawLine(
      context,
      projectSchematic([-3, index + 1, 0], width, height),
      projectSchematic([3, index + 1, 0], width, height),
      "rgba(242,241,232,0.08)",
    );
  }
}

function drawCamera(context, matrix, width, height, active) {
  const eye = [matrix[3], matrix[7], matrix[11]];
  const center = projectSchematic(eye, width, height);
  const cornersC = [[-0.35, -0.25, 0.7], [0.35, -0.25, 0.7], [0.35, 0.25, 0.7], [-0.35, 0.25, 0.7]];
  const corners = cornersC.map((point) => projectSchematic(transformPoint(matrix, point), width, height));
  const color = active ? "#f2f1e8" : "rgba(242,241,232,0.28)";
  corners.forEach((corner) => drawLine(context, center, corner, color, active ? 1.6 : 1));
  for (let index = 0; index < corners.length; index += 1) {
    drawLine(context, corners[index], corners[(index + 1) % corners.length], color, active ? 1.6 : 1);
  }
  context.fillStyle = active ? "#f2f1e8" : "rgba(242,241,232,0.4)";
  context.beginPath();
  context.arc(center[0], center[1], active ? 4 : 2.5, 0, Math.PI * 2);
  context.fill();
}

function drawTrajectory(context, episode, objectId, width, height, color) {
  const points = episode.observations.map((observation) => {
    const object = observation.objects.find((candidate) => candidate.object_id === objectId);
    const matrix = object.T_WO.value.matrix_row_major;
    return projectSchematic([matrix[3], matrix[7], matrix[11]], width, height);
  });
  context.setLineDash([4, 4]);
  for (let index = 1; index < points.length; index += 1) {
    drawLine(context, points[index - 1], points[index], color, 1.5);
  }
  context.setLineDash([]);
}

function drawObject(context, object, width, height, color) {
  const matrix = object.T_WO.value.matrix_row_major;
  const center = projectSchematic([matrix[3], matrix[7], matrix[11]], width, height);
  context.fillStyle = color;
  context.strokeStyle = "rgba(5,11,9,0.72)";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(center[0], center[1], object.object_id === "object-axial" ? 10 : 9, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = "#f2f1e8";
  context.font = "10px ui-monospace, monospace";
  context.fillText(object.object_id, center[0] + 13, center[1] + 3);
}

function drawAction(context, episode, frame, width, height) {
  const action = episode.interventions.find((candidate) => candidate.episode_time_s <= frame.episode_time_s
    && candidate.commanded_action.availability === "present"
    && candidate.commanded_action.value.kind === "push");
  if (action === undefined) {
    return;
  }
  const object = frame.objects.find((candidate) => candidate.object_id === action.target_object_id);
  const matrix = object.T_WO.value.matrix_row_major;
  const startW = [matrix[3], matrix[7], matrix[11] + 0.18];
  const vector = action.commanded_action.value.vector_W_N;
  const endW = [startW[0] + vector[0] * 0.45, startW[1] + vector[1] * 0.45, startW[2] + vector[2] * 0.45];
  const start = projectSchematic(startW, width, height);
  const end = projectSchematic(endW, width, height);
  drawLine(context, start, end, "#ff7958", 3);
  context.fillStyle = "#ff7958";
  context.beginPath();
  context.arc(end[0], end[1], 4, 0, Math.PI * 2);
  context.fill();
}

function renderSchematic(canvas, episode, frameIndex) {
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#07100d";
  context.fillRect(0, 0, width, height);
  drawGround(context, width, height);
  drawAxes(context, width, height);
  episode.observations.forEach((observation, index) => {
    drawCamera(context, observation.T_WC.value.matrix_row_major, width, height, index === frameIndex);
  });
  drawTrajectory(context, episode, "object-asymmetric", width, height, "rgba(200,255,99,0.58)");
  drawTrajectory(context, episode, "object-axial", width, height, "rgba(103,230,211,0.58)");
  const frame = episode.observations[frameIndex];
  drawObject(context, frame.objects[0], width, height, "#c8ff63");
  drawObject(context, frame.objects[1], width, height, "#67e6d3");
  drawAction(context, episode, frame, width, height);
}

function renderRgb(canvas, descriptor, bytes) {
  const [height, width, channels] = descriptor.shape;
  if (channels !== 3 || bytes.byteLength !== width * height * channels) {
    throw new Error("RGB resource shape does not match its bytes");
  }
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const rgba = new Uint8ClampedArray(width * height * 4);
  const rgb = new Uint8Array(bytes);
  for (let input = 0, output = 0; input < rgb.length; input += 3, output += 4) {
    rgba[output] = rgb[input];
    rgba[output + 1] = rgb[input + 1];
    rgba[output + 2] = rgb[input + 2];
    rgba[output + 3] = 255;
  }
  context.putImageData(new ImageData(rgba, width, height), 0, 0);
}

function depthRange(descriptor, bytes) {
  const expected = descriptor.shape.reduce((product, value) => product * value, 1) * 4;
  if (bytes.byteLength !== expected) {
    throw new Error("depth resource shape does not match its bytes");
  }
  const view = new DataView(bytes);
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let offset = 0; offset < bytes.byteLength; offset += 4) {
    const value = view.getFloat32(offset, true);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error("depth resource contains an invalid value");
    }
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return [minimum, maximum];
}

export function initPr00View() {
  const workbench = document.querySelector("#contract-workbench");
  const loadButton = document.querySelector("#load-contract");
  const closeButton = document.querySelector("#close-contract");
  const timeline = document.querySelector("#contract-timeline");
  const rgbCanvas = document.querySelector("#contract-rgb");
  const schematicCanvas = document.querySelector("#contract-world");
  const status = document.querySelector("#contract-status");
  const inspector = document.querySelector("#scene-inspector");
  const worldButton = document.querySelector("#load-world");
  let payload = null;
  let wasInspectorHidden = false;
  let resizeRequest = 0;

  function setStatus(kind, text) {
    status.dataset.kind = kind;
    status.textContent = text;
  }

  function hide() {
    workbench.hidden = true;
    loadButton.classList.remove("primary");
    worldButton.classList.add("primary");
    inspector.hidden = wasInspectorHidden;
  }

  function renderFrame(frameIndex) {
    const frame = payload.episode.observations[frameIndex];
    const rgb = frame.rgb.value;
    const depth = frame.depth.value;
    renderRgb(rgbCanvas, rgb, payload.resources.get(rgb.uri));
    renderSchematic(schematicCanvas, payload.episode, frameIndex);
    const [minDepth, maxDepth] = depthRange(depth, payload.resources.get(depth.uri));
    document.querySelector("#contract-frame").textContent = `${frame.observation_id} · t=${frame.episode_time_s.toFixed(1)} s`;
    document.querySelector("#contract-depth").textContent = `${minDepth.toFixed(2)}–${maxDepth.toFixed(2)} m`;
    document.querySelector("#contract-object-count").textContent = String(frame.objects.length);
    document.querySelector("#contract-action").textContent = frameIndex === 0 ? "hold" : "push → +X_W";
  }

  async function show() {
    workbench.hidden = false;
    loadButton.classList.add("primary");
    worldButton.classList.remove("primary");
    wasInspectorHidden = inspector.hidden;
    inspector.hidden = true;
    setStatus("loading", "VALIDATING");
    try {
      const module = await import("../generated/pr00/contract-consumer.mjs");
      payload = await module.loadPr00Contract();
      timeline.min = "0";
      timeline.max = String(payload.episode.observations.length - 1);
      timeline.value = "0";
      document.querySelector("#contract-verdict").textContent = payload.report.verdict.toUpperCase();
      document.querySelector("#contract-endpoint").textContent = payload.report.checks.primary_endpoint.max_error_px.toExponential(2);
      document.querySelector("#contract-checksum").textContent = `${payload.manifest.episode.sha256.slice(0, 12)}…`;
      renderFrame(0);
      setStatus("supported", "SUPPORTED");
    } catch (error) {
      payload = null;
      setStatus("blocked", "BLOCKED");
      document.querySelector("#contract-frame").textContent = error.message;
      document.querySelector("#contract-verdict").textContent = "INVALID";
    }
  }

  timeline.addEventListener("input", () => {
    if (payload !== null) {
      renderFrame(Number(timeline.value));
    }
  });
  function renderAfterResize() {
    if (payload === null || workbench.hidden) {
      return;
    }
    cancelAnimationFrame(resizeRequest);
    resizeRequest = requestAnimationFrame(() => {
      resizeRequest = 0;
      renderFrame(Number(timeline.value));
    });
  }
  window.addEventListener("resize", renderAfterResize);
  new ResizeObserver(renderAfterResize).observe(workbench);
  loadButton.addEventListener("click", show);
  closeButton.addEventListener("click", hide);
  for (const id of ["load-world", "load-lego", "file-input"]) {
    document.querySelector(`#${id}`).addEventListener("click", hide);
  }
  if (new URLSearchParams(window.location.search).get("mode") === "contract") {
    show();
  }
  return { hide, show };
}
