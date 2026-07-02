# ObjGauss Open Source Reference Map

> Status: draft / 2026-07-02
> Purpose: record external references for the rebuild direction without copying
> their architecture wholesale.

## Rebuild Target

The rebuilt product should open directly into an immersive Three.js world:

- No permanent sidebars as the primary interaction model.
- The main surface is a 3D exhibition / VR-like scene.
- All models appear in the scene as draggable objects.
- Object and model information appears in frosted-glass in-world overlays.
- Frontend owns experience, interaction, and rendering.
- Backend / pipeline owns model supply, training output processing, core-point
  extraction, object-aware compression, manifests, and per-object chunk delivery.
- ObjGauss-owned frontend Gaussian renderer algorithms remain first-class
  frontend algorithms: Gaussian OIT, WebGPU tile / compute paths, shaders,
  object-state buffers, picking, Spark bridge, and OGC browser decoder contracts.

## References To Borrow From

### mkkellogg/GaussianSplats3D

Repository: <https://github.com/mkkellogg/GaussianSplats3D>

Useful ideas:

- Three.js-native Gaussian splat viewer rather than a separate native renderer.
- Multi-scene loading through scene descriptors.
- A compressed internal scene format (`.ksplat`) that loads faster than raw
  `.ply` / `.splat`.
- Drop-in viewer concept that can coexist with normal Three.js scene objects.
- WebXR support as a product direction.

Use in ObjGauss:

- Borrow the product shape: Three.js world + splat scenes as spatial objects.
- Borrow the loading lesson: browser delivery should use a packed internal
  format that matches runtime memory layout.
- Do not replace ObjGauss renderer kernels with this project. Our Gaussian OIT,
  WebGPU tile / compute, object-state buffer, picking, Spark bridge, and OGC
  decoder remain ours.

### antimatter15/splat

Repository: <https://github.com/antimatter15/splat>

Useful ideas:

- Minimal browser Gaussian viewer that treats splats as navigable scene content.
- Movement / orbit control model for a spatial experience.
- Clear explanation of transparency sorting and why browser splat renderers
  need specialized ordering / compositing.

Use in ObjGauss:

- Borrow the simplicity of opening into a navigable 3D scene.
- Keep our renderer research line for transparency, sorting, and WebGPU kernels.

### playcanvas/supersplat

Repository: <https://github.com/playcanvas/supersplat>

Useful ideas:

- Editor-oriented workflow for Gaussian splats: load, inspect, optimize, publish.
- Asset-processing is separate from the viewer surface.
- Scene objects are manipulated spatially instead of through a dense form-first
  admin UI.

Use in ObjGauss:

- Borrow the workflow boundary: frontend manipulates objects; backend/pipeline
  produces optimized delivery artifacts.
- Do not move training, model processing, or compression policy into the browser
  UI.

### A-Frame

Repository: <https://github.com/aframevr/aframe>

Useful ideas:

- Web framework for browser-based 3D / AR / VR experiences.
- Entity-component mental model for spatial objects.
- Mature WebXR product vocabulary.

Use in ObjGauss:

- Borrow the spatial interaction model and WebXR vocabulary.
- Keep implementation on direct Three.js for now because the existing frontend
  renderer algorithms are Three.js / WebGPU / Spark oriented.

### Hubs Foundation / Hubs

Repository: <https://github.com/Hubs-Foundation/hubs>

Useful ideas:

- Full-screen virtual-world product model.
- In-world UI surfaces instead of dense application sidebars.
- Desktop, mobile, and VR as a shared target.

Use in ObjGauss:

- Borrow the in-world UI principle: information should feel attached to the
  scene, using frosted-glass overlays and spatial selection.
- Do not inherit the multi-user stack, networking, identity, or backend shape.

## Immediate Implementation Direction

1. Replace the current first screen with a full-viewport Three.js exhibition.
2. Load safe local model samples into the scene as draggable groups.
3. Show core point per processed model as a visible anchor.
4. Show selected model and compression metadata in a frosted-glass floating
   overlay, not a sidebar.
5. Keep old renderer algorithm files in the repo, but remove them from the new
   default entry path until they are reintroduced as explicit renderer modules.
6. Record the future model delivery contract:
   processed 3D Gaussian -> object core point -> per-object compressed storage
   -> lazy chunk load by selected object / proximity / LOD.
