import { spawn, execFileSync } from "node:child_process";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const DEFAULT_PORT = 5395;
const DEFAULT_HOST = "127.0.0.1";
const RESTART_WAIT_MS = 3000;
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = realpathSync(path.resolve(scriptDir, ".."));

const options = parseArgs(process.argv.slice(2));
const port = Number(options.port ?? DEFAULT_PORT);
const host = String(options.host ?? DEFAULT_HOST);
const strictPort = options.strictPort ?? true;
const restartProjectServer = options.restartProjectServer ?? true;

if (!Number.isInteger(port) || port <= 0) {
  throw new Error(`invalid dev server port: ${options.port}`);
}

await preparePort({ port, restartProjectServer });
await startVite({ port, host, strictPort, passthrough: options.passthrough });

async function preparePort({ port, restartProjectServer }) {
  const listeners = findPortListeners(port);
  if (!listeners.length) return;

  const projectListeners = listeners.filter(isProjectDevServer);
  const foreignListeners = listeners.filter((listener) => !isProjectDevServer(listener));
  if (foreignListeners.length > 0) {
    const details = foreignListeners.map(formatListener).join("; ");
    throw new Error(`port ${port} is already used by a non-ObjGauss process: ${details}`);
  }
  if (!restartProjectServer) {
    const details = projectListeners.map(formatListener).join("; ");
    throw new Error(`port ${port} is already used by ObjGauss dev server: ${details}`);
  }

  const pids = [...new Set(projectListeners.map((listener) => listener.pid))];
  for (const pid of pids) {
    process.kill(pid, "SIGTERM");
  }
  if (await waitForPortFree(port, RESTART_WAIT_MS)) return;

  for (const pid of pids) {
    try {
      process.kill(pid, "SIGKILL");
    } catch {
      // The process may have exited between the grace wait and this fallback.
    }
  }
  if (!(await waitForPortFree(port, RESTART_WAIT_MS))) {
    throw new Error(`port ${port} is still busy after restarting ObjGauss dev server`);
  }
}

async function waitForPortFree(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (findPortListeners(port).length === 0) return true;
    await sleep(100);
  }
  return findPortListeners(port).length === 0;
}

async function startVite({ port, host, strictPort, passthrough }) {
  const viteBin = path.join(
    projectRoot,
    "node_modules",
    ".bin",
    process.platform === "win32" ? "vite.cmd" : "vite",
  );
  if (!existsSync(viteBin)) {
    throw new Error("missing local Vite binary; run npm install before starting the dev server");
  }
  const viteArgs = ["--host", host, "--port", String(port)];
  if (strictPort) viteArgs.push("--strictPort");
  viteArgs.push(...passthrough);

  const child = spawn(viteBin, viteArgs, {
    cwd: projectRoot,
    stdio: "inherit",
    env: process.env,
  });
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, () => {
      child.kill(signal);
    });
  }
  const exit = await new Promise((resolve) => {
    child.on("exit", (code, signal) => resolve({ code, signal }));
  });
  if (exit.signal) {
    process.kill(process.pid, exit.signal);
  }
  process.exit(exit.code ?? 0);
}

function findPortListeners(port) {
  let output = "";
  try {
    output = execFileSync("lsof", ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN", "-F", "pc"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch (error) {
    if (error.status === 1) return [];
    throw error;
  }

  const listeners = [];
  let current = null;
  for (const line of output.split("\n")) {
    if (!line) continue;
    const type = line[0];
    const value = line.slice(1);
    if (type === "p") {
      if (current) listeners.push(current);
      current = { pid: Number(value), command: "", cwd: "", cmdline: "" };
    } else if (type === "c" && current) {
      current.command = value;
    }
  }
  if (current) listeners.push(current);
  return listeners
    .filter((listener) => Number.isInteger(listener.pid) && listener.pid > 0)
    .map((listener) => ({
      ...listener,
      cwd: processCwd(listener.pid),
      cmdline: processCmdline(listener.pid),
    }));
}

function isProjectDevServer(listener) {
  const cwdMatches = listener.cwd ? isInside(projectRoot, listener.cwd) : false;
  const cmdlineMatches = listener.cmdline.includes(projectRoot);
  const devCommand =
    /\bvite\b/.test(listener.cmdline) ||
    /\bdev-server\.mjs\b/.test(listener.cmdline) ||
    /\bnpm\b.*\brun\b.*\bdev\b/.test(listener.cmdline);
  return (cwdMatches || cmdlineMatches) && devCommand;
}

function processCwd(pid) {
  try {
    return realpathSync(`/proc/${pid}/cwd`);
  } catch {
    return "";
  }
}

function processCmdline(pid) {
  try {
    return readFileSync(`/proc/${pid}/cmdline`, "utf8").replaceAll("\0", " ").trim();
  } catch {
    return "";
  }
}

function isInside(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function formatListener(listener) {
  return `${listener.pid}:${listener.command || "unknown"}:${listener.cwd || "unknown-cwd"}`;
}

function parseArgs(rawArgs) {
  const parsed = {
    passthrough: [],
  };
  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];
    if (arg === "--port") {
      parsed.port = rawArgs[index + 1];
      index += 1;
    } else if (arg.startsWith("--port=")) {
      parsed.port = arg.slice("--port=".length);
    } else if (arg === "--host") {
      parsed.host = rawArgs[index + 1];
      index += 1;
    } else if (arg.startsWith("--host=")) {
      parsed.host = arg.slice("--host=".length);
    } else if (arg === "--strictPort") {
      parsed.strictPort = true;
    } else if (arg === "--no-strictPort") {
      parsed.strictPort = false;
    } else if (arg === "--restart") {
      parsed.restartProjectServer = true;
    } else if (arg === "--no-restart") {
      parsed.restartProjectServer = false;
    } else {
      parsed.passthrough.push(arg);
    }
  }
  return parsed;
}
