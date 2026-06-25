# WebGPU Headless Acceptance

> Status: RENDER-005T-AS runbook

This runbook defines the CI/headless WebGPU gate for the ObjGauss C-path
renderer. It validates compute, storage, object-state filtering, GPU readback,
and object edit transitions without relying on canvas presentation.

## Command

```bash
npm run acceptance:webgpu-headless
```

The command runs:

```text
npm run build
npm run audit:webgpu-tile-smoke
npm run audit:webgpu-offscreen-readback
```

By default the offscreen suite covers:

```text
nerf-lego-alpha-closure-local
plush-semantic-closure-local
```

and writes:

```text
/tmp/objgauss-webgpu-headless-acceptance/offscreen-readback/summary.json
/tmp/objgauss-webgpu-headless-acceptance/offscreen-readback/summary.md
```

## What It Proves

- WebGPU storage buffers upload successfully.
- Pixel compute dispatch completes and `queue.onSubmittedWorkDone()` resolves.
- `pixelResolvedRgba` can be copied to a `MAP_READ` buffer.
- Isolate and delete transitions change the GPU object-state checksum.
- Isolate and delete transitions change the offscreen readback checksum.
- Large scene coverage includes Plush semantic at 281k Gaussians.

This gate intentionally does not create a WebGPU canvas render pass.

## What It Does Not Prove

- Canvas presentation works.
- The fullscreen display pass works in a headed browser.
- Spark/edit visual residual is acceptable.
- The commercial Spark filtered edit path changed.

Those are covered by desktop/presentation and visual gates, not by the headless
acceptance command.

## Presentation Gate

Use the desktop audit when the question is whether browser presentation works:

```bash
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --probes full
```

Use the presentation performance smoke when the question is whether the full
WebGPU canvas path still renders within the current headed browser timing
envelope across the default small and large scenes:

```bash
npm run audit:webgpu-presentation-performance
```

Use the presentation transition suite when the question is whether headed
full-canvas WebGPU object selection, isolate, and delete transitions stay on
the C-path renderer instead of falling back to Spark:

```bash
npm run audit:webgpu-presentation-transition
```

For visual fidelity / tuning regressions, use the existing coverage gate:

```bash
npm run audit:webgpu-coverage-gate
```

For runtime timing regressions in the same offscreen object-transition path, use
the performance smoke gate:

```bash
npm run audit:webgpu-runtime-performance
```

It reruns `audit:webgpu-offscreen-readback`, parses the browser-visible
`storageTiming*` telemetry, and writes:

```text
/tmp/objgauss-webgpu-runtime-performance/summary.json
/tmp/objgauss-webgpu-runtime-performance/summary.md
```

This timing gate proves the current update / submit / queue-done envelope for
WebGPU object edits. It is not an FPS benchmark and does not replace headed
canvas presentation or visual residual gates.

Interpretation:

- `acceptance:webgpu-headless` passing means WebGPU compute/storage/object
  editing is healthy in CI/headless conditions.
- `audit:webgpu-desktop` passing means a real desktop browser can present the
  WebGPU output to canvas.
- If headless presentation fails while offscreen acceptance passes, classify it
  as a headless WebGPU presentation/backend limitation unless the desktop gate
  also fails.

## Useful Options

```bash
# Rerun a single scene.
npm run acceptance:webgpu-headless -- --assets nerf-lego-alpha-closure-local

# Use a different output directory.
npm run acceptance:webgpu-headless -- --output-dir /tmp/objgauss-webgpu-headless

# Reuse an existing build.
npm run acceptance:webgpu-headless -- --skip-build

# Keep local browser audits on the fixed preview port.
npm run acceptance:webgpu-headless -- --port 5395
```

Use `npm run audit:webgpu-offscreen-readback -- --skip-object-transition` only
when intentionally debugging the older first-readback-only probe.
