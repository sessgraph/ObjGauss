export const BRANCH_ORDER = [
  "hold",
  "push-neg-x-weak",
  "push-pos-x-weak",
  "push-pos-x-strong",
  "push-pos-y-weak",
];

export function recordAt(records, time) {
  let selected = records[0];
  for (const record of records) {
    if (record.episode_time_s > time + 1e-9) break;
    selected = record;
  }
  return selected;
}

export function worldToCanvas(canvas, position) {
  const scale = Math.min(canvas.width, canvas.height) / 0.44;
  return [canvas.width / 2 + position[0] * scale, canvas.height / 2 - position[1] * scale];
}

function arrow(context, from, vector, color, scale = 1) {
  const length = Math.hypot(vector[0], vector[1]);
  if (length < 1e-12) return;
  const to = [from[0] + vector[0] * scale, from[1] - vector[1] * scale];
  const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 2;
  context.beginPath(); context.moveTo(...from); context.lineTo(...to); context.stroke();
  context.beginPath(); context.moveTo(...to);
  context.lineTo(to[0] - 8 * Math.cos(angle - .45), to[1] - 8 * Math.sin(angle - .45));
  context.lineTo(to[0] - 8 * Math.cos(angle + .45), to[1] - 8 * Math.sin(angle + .45));
  context.closePath(); context.fill();
}

export function renderBranch(canvas, payload, time) {
  const context = canvas.getContext("2d");
  const { trajectory, contacts, episode } = payload;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#08120f"; context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "rgba(187,255,221,.08)"; context.lineWidth = 1;
  for (let i = 0; i <= 10; i += 1) {
    const x = i * canvas.width / 10; const y = i * canvas.height / 10;
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, canvas.height); context.stroke();
    context.beginPath(); context.moveTo(0, y); context.lineTo(canvas.width, y); context.stroke();
  }
  const visible = trajectory.records.filter((record) => record.episode_time_s <= time + 1e-9);
  context.strokeStyle = "#69d9ff"; context.lineWidth = 2; context.beginPath();
  visible.forEach((record, index) => {
    const point = worldToCanvas(canvas, record.actors.target.position_W_m);
    if (index === 0) context.moveTo(...point); else context.lineTo(...point);
  });
  context.stroke();
  const record = recordAt(trajectory.records, time);
  for (const [name, actor] of Object.entries(record.actors)) {
    const point = worldToCanvas(canvas, actor.position_W_m);
    context.fillStyle = name === "target" ? "#8bffbd" : "#60776c";
    context.fillRect(point[0] - (name === "target" ? 15 : 11), point[1] - 11, name === "target" ? 30 : 22, 22);
  }
  const targetPoint = worldToCanvas(canvas, record.actors.target.position_W_m);
  if (time <= episode.intervention.commanded_action.duration_s + 1e-9) {
    arrow(context, targetPoint, episode.intervention.commanded_action.vector_W_N, "#ffad66", 70);
  }
  const contact = time + 1e-9 < contacts.records[0].episode_time_s
    ? { contacts: [] }
    : recordAt(contacts.records, time);
  for (const pair of contact.contacts) {
    for (const point of pair.points) {
      const origin = worldToCanvas(canvas, point.position_W_m);
      context.fillStyle = "#ffad66"; context.beginPath(); context.arc(...origin, 3, 0, Math.PI * 2); context.fill();
      arrow(context, origin, point.normal_W, "#edf8f2", 18);
      arrow(context, origin, point.impulse_W_N_s, "#69d9ff", 900);
    }
  }
  context.fillStyle = "rgba(237,248,242,.7)"; context.font = "11px ui-monospace, monospace";
  context.fillText(`t=${time.toFixed(2)}s · ${record.phase}`, 12, 20);
}
