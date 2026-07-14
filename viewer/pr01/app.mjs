import { BRANCH_ORDER, renderBranch } from "./render.mjs";

const state = { time: 0, playing: false, previous: 0, branches: [] };

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

async function loadJson(uri, expectedSha256) {
  const response = await fetch(uri, { cache: "no-store" });
  if (!response.ok) throw new Error(`${uri}: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (expectedSha256 && await sha256(bytes) !== expectedSha256) throw new Error(`${uri}: checksum mismatch`);
  return JSON.parse(new TextDecoder().decode(bytes));
}

function actionLabel(action) {
  const vector = action.vector_W_N.map((value) => Number(value).toFixed(2)).join(", ");
  return `${action.kind} [${vector}] N · ${action.applied_steps} steps`;
}

function contactLabel(contacts) {
  const points = contacts.records.reduce((sum, record) => sum + record.contacts.reduce((inner, item) => inner + item.points.length, 0), 0);
  return `${points} traced points`;
}

function createCard(payload) {
  const fragment = document.querySelector("#branch-template").content.cloneNode(true);
  const card = fragment.querySelector("article");
  card.dataset.branch = payload.episode.identity.branch_id;
  fragment.querySelector(".branch-kind").textContent = payload.episode.intervention.commanded_action.kind.toUpperCase();
  fragment.querySelector(".branch-name").textContent = payload.episode.identity.branch_id;
  fragment.querySelector(".commanded").textContent = actionLabel(payload.episode.intervention.commanded_action);
  fragment.querySelector(".executed").textContent = actionLabel(payload.episode.intervention.executed_action);
  fragment.querySelector(".settling").textContent = payload.episode.evidence.settling_result.settled ? "settled · recomputed" : "NOT SETTLED";
  fragment.querySelector(".contact").textContent = contactLabel(payload.contacts);
  document.querySelector("#branches").append(fragment);
  payload.canvas = document.querySelector(`[data-branch="${payload.episode.identity.branch_id}"] canvas`);
}

function render() {
  document.querySelector("#timeline").value = String(state.time);
  document.querySelector("#time").value = `${state.time.toFixed(2)} s`;
  state.branches.forEach((payload) => renderBranch(payload.canvas, payload, state.time));
}

function frame(now) {
  if (!state.playing) return;
  const delta = Math.min(.05, (now - state.previous) / 1000);
  state.previous = now;
  state.time = Math.min(1.1, state.time + delta);
  if (state.time >= 1.1) { state.playing = false; document.querySelector("#play").textContent = "REPLAY"; }
  render();
  if (state.playing) requestAnimationFrame(frame);
}

async function start() {
  const verdict = document.querySelector("#verdict");
  try {
    const manifest = await loadJson("./demo-manifest.json");
    const report = await loadJson(manifest.audit_report_uri, manifest.audit_report_sha256);
    if (report.verdict.status !== "supported") throw new Error(`machine audit is ${report.verdict.status}`);
    for (const branch of manifest.branches) {
      const episode = await loadJson(branch.episode_uri, branch.episode_sha256);
      const trajectory = await loadJson(branch.trajectory_uri, episode.evidence.trajectory.sha256);
      const contacts = await loadJson(branch.contact_uri, episode.evidence.contact_ledger.sha256);
      state.branches.push({ episode, trajectory, contacts });
    }
    state.branches.sort((left, right) => BRANCH_ORDER.indexOf(left.episode.identity.branch_id) - BRANCH_ORDER.indexOf(right.episode.identity.branch_id));
    if (state.branches.length !== 5) throw new Error("five audited branches are required");
    state.branches.forEach(createCard);
    const first = state.branches[0].episode;
    document.querySelector("#group-id").textContent = first.identity.group_id;
    document.querySelector("#split").textContent = first.identity.split;
    document.querySelector("#snapshot").textContent = `${first.initialization.snapshot_sha256.slice(0, 12)}…`;
    document.querySelector("#rng").textContent = `${first.initialization.restored_rng_sha256.slice(0, 12)}…`;
    verdict.dataset.status = "supported"; verdict.querySelector("strong").textContent = "SUPPORTED";
    document.querySelector("#play").disabled = false;
    document.querySelector("#timeline").disabled = false;
    render();
  } catch (error) {
    verdict.dataset.status = "invalid"; verdict.querySelector("strong").textContent = "INVALID";
    document.querySelector("#group-id").textContent = error.message;
    throw error;
  }
}

document.querySelector("#play").addEventListener("click", () => {
  if (state.time >= 1.1) state.time = 0;
  state.playing = !state.playing;
  document.querySelector("#play").textContent = state.playing ? "PAUSE" : "PLAY";
  if (state.playing) { state.previous = performance.now(); requestAnimationFrame(frame); }
});
document.querySelector("#timeline").addEventListener("input", (event) => {
  state.playing = false; state.time = Number(event.target.value); document.querySelector("#play").textContent = "PLAY"; render();
});
window.addEventListener("resize", render);
start();
