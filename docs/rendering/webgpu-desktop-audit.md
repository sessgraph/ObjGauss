# WebGPU Desktop Runtime Audit

> Status: RENDER-005Q runbook

This audit separates a real desktop Chrome/WebGPU presentation result from the
known headless unsafe-WebGPU presentation backend loss.

For CI/headless compute, storage, readback, and object-state transition
coverage, run `npm run acceptance:webgpu-headless` instead. Passing headless
acceptance is not a substitute for this desktop presentation audit.

## When To Run

Run this after `RENDER-005P` when a machine has a desktop session and a
WebGPU-capable Chrome/Chromium build. The current headless probe result is not
enough to claim the final WebGPU tile renderer works, because `clear-only`
already loses the device in that environment.

## Command

```bash
npm run build
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local
```

The command starts `vite preview` on `127.0.0.1:5230`, launches system
Chrome/Chromium in headed mode with `--enable-unsafe-webgpu`, and runs:

```text
clear-only
texture-display-only
full
```

Expected desktop pass:

```text
webgpu_desktop_audit=passed classification="desktop-webgpu-runtime-passed"
```

## Useful Options

```bash
# Pick an installed Playwright channel.
npm run audit:webgpu-desktop -- --browser-channel chrome

# Pick a specific browser binary.
npm run audit:webgpu-desktop -- --executable-path /usr/bin/google-chrome

# Reuse an already running server.
npm run audit:webgpu-desktop -- --url http://127.0.0.1:5230/ --no-server

# Diagnostic headless run; this is expected to fail on the current machine.
npm run audit:webgpu-desktop -- --headless --allow-failures
```

## Interpretation

- `desktop-webgpu-runtime-passed`: desktop Chrome presentation works. If
  headless still fails, classify the blocker as a headless unsafe-WebGPU
  presentation limitation and keep compute/readback probes for CI evidence.
- `desktop-webgpu-unavailable`: browser did not expose a WebGPU adapter. Use a
  newer Chrome/Chromium build, check GPU drivers, or try `--browser-channel
  chrome`.
- `desktop-webgpu-presentation-backend-loss`: even desktop Chrome lost the
  device. Continue debugging the presentation path before claiming C-runtime
  readiness.

## Scope

This is a runtime/presentation audit. It does not change the renderer, train
data, object labels, or benchmark outputs.
