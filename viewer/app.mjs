import {
  SPLAT_FORMAT,
  assertSplatByteLength,
  parseSplatV1,
} from "./splat-format.mjs";
import { SplatRenderer } from "./splat-renderer.mjs";
import { createSyntheticWorldSplat } from "./synthetic-world.mjs";

const SYNTHETIC_SAMPLE = Object.freeze({
  name: "ObjGauss synthetic Gaussian world",
  expectedBytes: 272_736,
  expectedCount: 8_523,
  expectedSha256: "4782f6ed4816aee54618bb4d1fcbce8df67e65301e23a89c155985084f51cfe6",
});

const LEGO_AUDIT_SAMPLE = Object.freeze({
  url: "../data/local-preview/legobrick-1267e213/legobrick.splat",
  name: "Legobrick · fixed local preview",
  commit: "1267e2135660e1f4197f94c045453fe40c209b0e",
  expectedBytes: 3_297_920,
  expectedCount: 103_060,
  expectedSha256: "d5131a664a12a8764da70552c85f567d276313110f63f1efd48424845917899e",
});

const elements = {
  canvas: document.querySelector("#splat-canvas"),
  dropZone: document.querySelector("#drop-zone"),
  overlay: document.querySelector("#load-overlay"),
  loadStatus: document.querySelector("#load-status"),
  statusDetail: document.querySelector("#status-detail"),
  statusHeading: document.querySelector("#status-heading"),
  statusCopy: document.querySelector("#status-copy"),
  stateBadge: document.querySelector("#state-badge"),
  sceneTitle: document.querySelector("#scene-title"),
  loadWorld: document.querySelector("#load-world"),
  loadLego: document.querySelector("#load-lego"),
  fileInput: document.querySelector("#file-input"),
  resetCamera: document.querySelector("#reset-camera"),
  autoRotate: document.querySelector("#auto-rotate"),
  splatScale: document.querySelector("#splat-scale"),
  scaleValue: document.querySelector("#scale-value"),
  renderCount: document.querySelector("#render-count"),
  renderFps: document.querySelector("#render-fps"),
  sortTime: document.querySelector("#sort-time"),
  splatCount: document.querySelector("#splat-count"),
  sampleSize: document.querySelector("#sample-size"),
  formatName: document.querySelector("#format-name"),
  checksum: document.querySelector("#checksum"),
  sourceMode: document.querySelector("#source-mode"),
  semanticKind: document.querySelector("#semantic-kind"),
  assetProvenance: document.querySelector("#asset-provenance"),
  licenseStatus: document.querySelector("#license-status"),
  renderClaimTitle: document.querySelector("#render-claim-title"),
  renderClaimCopy: document.querySelector("#render-claim-copy"),
  sourceLink: document.querySelector("#source-link"),
  sceneInspector: document.querySelector("#scene-inspector"),
};

let renderer = null;
let activeLoad = 0;
let activeRequest = null;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "—";
  }
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1_024 && unit < units.length - 1) {
    value /= 1_024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

async function sha256Hex(arrayBuffer) {
  if (!globalThis.crypto?.subtle) {
    throw new Error("浏览器缺少 Web Crypto，无法校验默认样例 SHA-256");
  }
  const digest = await crypto.subtle.digest("SHA-256", arrayBuffer);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function setState(kind, heading, detail) {
  elements.dropZone.dataset.state = kind;
  elements.stateBadge.dataset.kind = kind;
  elements.stateBadge.querySelector("span").textContent = kind.toUpperCase();
  elements.loadStatus.textContent = heading;
  elements.statusDetail.textContent = detail;
  elements.statusHeading.textContent = heading;
  elements.statusCopy.textContent = detail;
  elements.overlay.hidden = kind === "ready" || kind === "review";
  if (kind === "ready" || kind === "review") {
    elements.renderClaimTitle.textContent = "3D Gaussian 已在浏览器中实际渲染。";
    elements.renderClaimCopy.textContent = "首帧 WebGL2 draw、covariance 投影与 far-to-near 合成均已完成。";
  } else if (kind === "blocked") {
    elements.renderClaimTitle.textContent = "未获得可声明的渲染结果。";
    elements.renderClaimCopy.textContent = "输入或渲染链路已 fail-closed；旧场景和肯定声明均已清除。";
  } else {
    elements.renderClaimTitle.textContent = "渲染验证尚未完成。";
    elements.renderClaimCopy.textContent = "完成输入校验、首次深度排序和首个 WebGL2 draw 后才会更新声明。";
  }
}

function clearLedger() {
  elements.sceneTitle.textContent = "没有可渲染场景";
  elements.splatCount.textContent = "—";
  elements.sampleSize.textContent = "—";
  elements.formatName.textContent = "—";
  elements.checksum.textContent = "—";
  elements.sourceMode.textContent = "—";
  elements.semanticKind.textContent = "—";
  elements.assetProvenance.textContent = "—";
  elements.licenseStatus.textContent = "—";
  elements.sourceLink.hidden = true;
  elements.sourceLink.removeAttribute("href");
  elements.renderCount.textContent = "0 / 0";
}

function block(heading, detail, { clearRenderer = true } = {}) {
  if (clearRenderer && renderer !== null) {
    renderer.clear("blocked");
  }
  clearLedger();
  setState("blocked", heading, detail);
}

function renderLedger({
  parsed,
  byteLength,
  sha256,
  name,
  mode,
  semanticKind,
  provenance,
  license,
  sourceUrl = null,
  sourceLabel = "来源",
}) {
  elements.sceneTitle.textContent = name;
  elements.splatCount.textContent = parsed.count.toLocaleString("zh-CN");
  elements.sampleSize.textContent = formatBytes(byteLength);
  elements.formatName.textContent = parsed.format;
  elements.checksum.textContent = `${sha256.slice(0, 12)}…${sha256.slice(-8)}`;
  elements.checksum.title = sha256;
  elements.sourceMode.textContent = mode;
  elements.semanticKind.textContent = semanticKind;
  elements.assetProvenance.textContent = provenance;
  elements.licenseStatus.textContent = license;
  elements.sourceLink.hidden = sourceUrl === null;
  if (sourceUrl === null) {
    elements.sourceLink.removeAttribute("href");
  } else {
    elements.sourceLink.href = sourceUrl;
    elements.sourceLink.querySelector("strong").textContent = sourceLabel;
  }
}

async function renderBytes(arrayBuffer, {
  name,
  mode,
  semanticKind,
  provenance,
  license,
  sourceUrl = null,
  sourceLabel = "来源",
  displayScale = 1,
  cameraMode = "fit",
  readyHeading = "3D Gaussian 已就绪",
  expected = null,
  loadId,
}) {
  assertSplatByteLength(arrayBuffer.byteLength);
  if (expected !== null && arrayBuffer.byteLength !== expected.expectedBytes) {
    throw new Error(
      `登记 fixture 大小不符：expected ${expected.expectedBytes}, got ${arrayBuffer.byteLength}`,
    );
  }

  setState("loading", "正在校验 Gaussian", "计算 SHA-256，并验证每条 32-byte splat record…");
  const sha256 = await sha256Hex(arrayBuffer);
  if (expected !== null && sha256 !== expected.expectedSha256) {
    throw new Error(`登记 fixture SHA-256 不符：${sha256}`);
  }
  if (loadId !== activeLoad) {
    return;
  }

  const parsed = parseSplatV1(arrayBuffer);
  if (expected !== null && parsed.count !== expected.expectedCount) {
    throw new Error(`登记 fixture splat 数量不符：${parsed.count}`);
  }
  renderLedger({
    parsed,
    byteLength: arrayBuffer.byteLength,
    sha256,
    name,
    mode,
    semanticKind,
    provenance,
    license,
    sourceUrl,
    sourceLabel,
  });
  elements.splatScale.value = String(displayScale);
  elements.scaleValue.textContent = `${displayScale}×`;
  renderer.setScaleMultiplier(displayScale);
  setState("loading", "正在投影 3D covariance", "上传 GPU 后进行首次 far-to-near 深度排序…");
  await renderer.load(parsed, { cameraMode });
  if (loadId !== activeLoad) {
    return;
  }

  if (expected === null) {
    setState(
      "review",
      "本地 Gaussian 已渲染",
      "结构有效，但该文件的来源、许可与 checksum 未进入项目台账。",
    );
  } else {
    setState(
      "ready",
      readyHeading,
      `${parsed.count.toLocaleString("zh-CN")} 个 splat 已通过固定大小与 SHA-256，并完成首次深度排序。`,
    );
  }
}

async function loadSyntheticWorld() {
  const loadId = ++activeLoad;
  activeRequest?.abort();
  activeRequest = null;
  setState("loading", "正在生成 Gaussian 世界", "构造地形、路径、建筑、树与环境粒子…");
  try {
    await renderBytes(createSyntheticWorldSplat(), {
      name: SYNTHETIC_SAMPLE.name,
      mode: "synthetic-world · deterministic",
      semanticKind: "synthetic-gaussian-world",
      provenance: "generated-in-browser",
      license: "third-party review: not-required · project license: unresolved",
      displayScale: 0.75,
      cameraMode: "immersive",
      readyHeading: "Gaussian 世界已就绪",
      expected: SYNTHETIC_SAMPLE,
      loadId,
    });
  } catch (error) {
    if (loadId !== activeLoad || renderer?.blocked) {
      return;
    }
    block("合成世界不可用", error.message);
  }
}

async function loadLegoAuditSample() {
  const loadId = ++activeLoad;
  activeRequest?.abort();
  activeRequest = new AbortController();
  setState("loading", "正在读取固定样例", "从 Git ignored data/ 目录加载 legobrick.splat…");
  try {
    const response = await fetch(LEGO_AUDIT_SAMPLE.url, {
      cache: "no-store",
      signal: activeRequest.signal,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const contentLength = response.headers.get("content-length");
    if (contentLength !== null && Number(contentLength) !== LEGO_AUDIT_SAMPLE.expectedBytes) {
      throw new Error(`响应大小不符：${contentLength}`);
    }
    await renderBytes(await response.arrayBuffer(), {
      name: LEGO_AUDIT_SAMPLE.name,
      mode: `pinned-preview · ${LEGO_AUDIT_SAMPLE.commit.slice(0, 8)}`,
      semanticKind: "point-derived-splat",
      provenance: "unverified",
      license: "review-pending · container MIT; asset rights/provenance unverified",
      sourceUrl: "https://github.com/GitHubDragonFly/GitHubDragonFly.github.io/blob/1267e2135660e1f4197f94c045453fe40c209b0e/viewers/examples/legobrick.splat",
      sourceLabel: "Lego 审计样例来源",
      displayScale: 18,
      expected: LEGO_AUDIT_SAMPLE,
      loadId,
    });
  } catch (error) {
    if (loadId !== activeLoad || error.name === "AbortError") {
      return;
    }
    if (renderer?.blocked) {
      return;
    }
    block(
      "Lego 审计样例不可用",
      `${error.message}。先运行 bash scripts/fetch-gaussian-preview.sh，再从仓库根目录启动服务器。`,
    );
  }
}

async function loadLocalFile(file) {
  const loadId = ++activeLoad;
  activeRequest?.abort();
  activeRequest = null;
  setState("loading", "正在读取本地文件", `${file.name} · ${formatBytes(file.size)}`);
  try {
    assertSplatByteLength(file.size);
    await renderBytes(await file.arrayBuffer(), {
      name: file.name,
      mode: "local-file · unverified",
      semanticKind: "local-splat",
      provenance: "unverified",
      license: "third-party review: unverified · project license: unresolved",
      displayScale: 1,
      loadId,
    });
  } catch (error) {
    if (loadId !== activeLoad) {
      return;
    }
    if (renderer?.blocked) {
      return;
    }
    block("本地文件被拒绝", error.message);
  }
}

try {
  renderer = new SplatRenderer(elements.canvas, {
    onFatal(message) {
      block("渲染器已阻断", message, { clearRenderer: false });
    },
    onStats({ fps, rendered, total, sortMs }) {
      elements.renderCount.textContent = `${rendered.toLocaleString("zh-CN")} / ${total.toLocaleString("zh-CN")}`;
      elements.renderFps.textContent = rendered > 0 ? fps.toFixed(0) : "—";
      elements.sortTime.textContent = rendered > 0
        ? (sortMs < 0.1 ? "<0.1 ms" : `${sortMs.toFixed(1)} ms`)
        : "—";
    },
  });
} catch (error) {
  block("WebGL2 不可用", error.message, { clearRenderer: false });
}

elements.loadWorld.addEventListener("click", loadSyntheticWorld);
elements.loadLego.addEventListener("click", loadLegoAuditSample);
elements.fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file !== undefined) {
    loadLocalFile(file);
  }
});
elements.resetCamera.addEventListener("click", () => renderer?.resetCamera());
elements.splatScale.addEventListener("input", () => {
  const value = Number(elements.splatScale.value);
  elements.scaleValue.textContent = `${value}×`;
  renderer?.setScaleMultiplier(value);
});
elements.autoRotate.addEventListener("click", () => {
  const enabled = elements.autoRotate.getAttribute("aria-pressed") !== "true";
  elements.autoRotate.setAttribute("aria-pressed", String(enabled));
  elements.autoRotate.querySelector("output").textContent = enabled ? "ON" : "OFF";
  renderer?.setAutoRotate(enabled);
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("is-dragging");
  });
}
elements.dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file !== undefined) {
    loadLocalFile(file);
  }
});

if (renderer !== null) {
  if (matchMedia("(max-width: 900px)").matches) {
    elements.sceneInspector.open = false;
  }
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    elements.autoRotate.setAttribute("aria-pressed", "false");
    elements.autoRotate.querySelector("output").textContent = "OFF";
    renderer.setAutoRotate(false);
  }
  elements.formatName.textContent = SPLAT_FORMAT;
  loadSyntheticWorld();
}
