# ADR 0005: WebGPU Tile-Based Object-Aware Gaussian Renderer

> 状态: Proposed / implementation-ready
> 日期: 2026-06-23

## 背景

ObjGauss 当前前端已经有两条渲染路径：

- `真实查看`: Spark 读取 `.splat`，能给出真实 3DGS 外观，但不能按 ObjGauss 的 `object_id` 做对象删除、隔离或隐藏。
- `对象编辑`: `PointCloudViewport` 读取带 `object_id` 的 Gaussian PLY，已经用 `ShaderMaterial`、screen-space Gaussian kernel、weighted blended OIT 和 GPU object-state texture 做对象级编辑预览。

这个中间态已经能验收对象绑定，但仍不是终局 renderer：

- 删除后的 `原始颜色（编辑预览）` 仍是 PLY 原色编辑预览，不是完整 3DGS 重渲染。
- 当前 shader 用 `THREE.Points` + `gl_PointCoord`，受 point-size、屏幕 sprite 和 WebGL blending 约束。
- 没有真正的 per-tile Gaussian binning / accumulation。
- 没有为 100k 到 1M Gaussians 设计稳定的数据布局、溢出处理和性能 telemetry。

因此 RENDER-004 的目标不是继续调 Three.js 点云近似，而是把 ObjGauss 的对象语义状态接入一个可演进到完整 Gaussian splatting 的 WebGPU tile renderer。

## 决策

采用 WebGPU tile-based renderer 作为 ObjGauss object-aware renderer 的终局架构，同时保留现有 Gaussian OIT WebGL path 作为 fallback/debug path。

RENDER-004 按 staged delivery 执行：

1. **RENDER-004A / Renderer boundary**
   - 新增 renderer capability detection：`navigator.gpu`、adapter、device、required limits。
   - 建立 renderer boundary，使 UI 可显式区分 `真实 Splat`、`Gaussian OIT 编辑`、`WebGPU Tile 编辑` 和 fallback。
   - WebGPU 初始化失败时回退到现有 `Gaussian OIT 编辑`，并暴露可审计状态，不静默伪装成功。

2. **RENDER-004B / GPU data layout**
   - 将 ObjGauss PLY 点数据打包为 WebGPU storage buffers。
   - 最小字段：
     - `position_radius`: `vec4<f32>`，`xyz` 为 Gaussian center，`w` 预留 bounding radius。
     - `color_opacity`: `vec4<f32>`，`rgb` 为当前颜色模式结果，`a` 为 opacity。
     - `scale_rotation`: `vec4<f32>`，`xy` 为 screen/kernel scale seed，`z` 为 rotation，`w` 预留。
     - `object_index`: `u32` dense object index。
     - `object_state`: `u32` visibility/removal/isolation state。
   - 相机、viewport、point-size scale、kernel cutoff、color mode 写入 uniform buffer。
   - 当前 PLY parser 的 `scale_0/1/2`、`rot_0..3`、`opacity` 继续作为第一版数据来源。

3. **RENDER-004C / Tile binning**
   - 默认 tile size 为 `16x16` pixels。
   - projection pass 每个 Gaussian 计算 clip position、screen center、screen-space ellipse/conic 近似和 tile bounding rect。
   - 第一版实现可使用 fixed-capacity tile lists：
     - `tile_counts`: per-tile atomic count。
     - `tile_entries`: `tile_count * max_entries_per_tile` 的 storage buffer。
     - 超出容量时记录 `overflow_count`，并在 UI/audit 中暴露。
   - 后续高质量实现升级为 prefix-sum compacted tile lists，避免 fixed-capacity 浪费和 overflow。

4. **RENDER-004D / Per-tile accumulation and resolve**
   - accumulation pass 每个 tile / pixel 遍历 tile list，计算 Gaussian kernel coverage。
   - 第一版使用与当前 WebGL path 对齐的 weighted blended OIT：
     - `sum_rgb += color * weight`
     - `sum_weight += weight`
     - resolve 为 `sum_rgb / sum_weight` 和 `1 - exp(-sum_weight * opacity_scale)`。
   - 后续可在 tile 内加入 depth-aware ordering 或 hybrid sorting，但不作为第一版阻塞项。

5. **RENDER-004E / Object editing contract**
   - `object_state` 必须与当前 GPU object-state filtering 语义一致：
     - 隐藏 object 不参与 binning 或 accumulation。
     - 隔离 object 只保留目标 object。
     - 删除预览从同一个 buffer state 生效，不重新上传全量 geometry。
   - 画布选中可以先保留 CPU raycast fallback；后续用 WebGPU pick buffer 或 object-id resolve target。
   - 选择高亮可作为单独 overlay pass，不混入主 accumulation 语义。

## 非目标

RENDER-004 第一轮不要求一次做到完整论文级 3DGS renderer：

- 不要求 SH view-dependent color。
- 不要求 tile 内完全 depth-sorted alpha compositing。
- 不要求替换 Spark 的 `.splat` 真实查看 path。
- 不要求提交训练产物、`outputs/`、checkpoint 或大型 demo assets。
- 不要求在不支持 WebGPU 的浏览器里模拟 WebGPU；这些环境走现有 WebGL fallback。

第一轮也不把 fixed-capacity tile list 伪装成最终方案。只要使用 fixed capacity，就必须暴露 overflow telemetry。

## Data Contract

输入仍是 ObjGauss 当前 object-aware Gaussian cloud：

```text
Gaussian {
  xyz: f32[3]
  color: f32[3] | SH DC converted rgb
  opacity: f32
  scale: f32[2] minimum first version, f32[3] later
  rotation: f32 first version, quaternion later
  object_id: i32
}
```

前端加载后建立 dense object index：

```text
object_id -> dense_object_index -> object_state[dense_object_index]
```

renderer 不直接依赖 sparse `object_id`，只消费 dense `object_index`。这保持 WebGPU buffer 紧凑，也和 RENDER-003 的 object-state texture 语义一致。

## Render Pipeline

第一版 pipeline：

```text
React state / ObjGauss scene
        |
        v
pack Gaussian buffers
        |
        v
clear tile counts + accumulation texture
        |
        v
project + bin visible Gaussians into tiles
        |
        v
accumulate per tile / pixel
        |
        v
fullscreen resolve to canvas
        |
        v
overlay grid / axes / selected object affordance
```

必须保留以下审计信息：

- `renderer`: `webgpu-tile` 或 fallback renderer id。
- `tileSize`: 默认 `16`。
- `gaussianCount`。
- `visibleCount`。
- `tileOverflowCount`。
- `objectFilter`: `gpu-object-state-buffer`。
- `fallbackReason`，仅 fallback 时存在。

## Fallback Contract

WebGPU renderer 只有在以下条件都满足时启用：

- `navigator.gpu` 存在。
- adapter/device 创建成功。
- required buffer/texture limits 满足当前 scene。
- shader module 和 pipeline 创建成功。
- 首帧 render 成功并产生非空 canvas。

否则必须回退到 `Gaussian OIT 编辑`，并在 DOM/audit 中暴露原因。

不允许出现以下状态：

- UI 文案显示 WebGPU，但实际走 WebGL。
- WebGPU shader/pipeline 失败后画布空白。
- object hide/isolate/delete 在 WebGL fallback 中可用，在 WebGPU 中无效。

## 验收

RENDER-004 完成必须满足：

- `npm run build` 通过。
- WebGPU 可用环境中，三个默认闭环样例可进入 `WebGPU Tile 编辑`。
- DOM 暴露 `data-renderer="webgpu-tile"` 和 `data-object-filter="gpu-object-state-buffer"`。
- 画布非空，且 browser audit 能检测到 edit pixels。
- 对象列表选择、画布选中、隔离、删除预览仍工作。
- 删除预览后使用 `原始颜色（编辑预览）` 显示剩余整体场景。
- `visibleCount` 与 object-state 更新一致。
- `tileOverflowCount` 被记录；如果大于 0，audit 不应无条件通过高质量 gate。
- 不支持 WebGPU 的浏览器明确 fallback 到 `Gaussian OIT 编辑`，现有 demo audit 仍通过。

## 实施顺序

建议按这些 PR 切：

1. `RENDER-004A`: capability detection + renderer boundary + audit contract。
2. `RENDER-004B`: WebGPU buffers + clear/project/bin smoke path，先输出 debug occupancy，不显示最终图。
3. `RENDER-004C`: tile accumulation + fullscreen resolve，显示 WebGPU 编辑画面。
4. `RENDER-004D`: object-state buffer 接入隐藏 / 隔离 / 删除。
5. `RENDER-004E`: browser audit、overflow telemetry、fallback hardening。

完成 `RENDER-004A` 前，不应删除或削弱当前 `Gaussian OIT 编辑` path；它是 WebGPU 不可用环境的验收底线。

## 风险

- WebGPU browser/driver 覆盖不稳定，fallback 必须长期保留。
- Fixed-capacity tile list 容易在密集视角 overflow；需要 telemetry 和后续 prefix-sum list。
- JavaScript packing 大场景 buffer 会产生交互延迟；后续可能需要 incremental upload 或 worker。
- 当前 `scale_0/1/2` 到 screen-space conic 的映射仍是近似；完整 covariance projection 应作为后续质量 PR。
- WebGPU picking / selection 不能阻塞第一版，可以先复用 CPU raycast。

## 与现有 ADR 的关系

- ADR 0001 选择 Spark 作为真实 `.splat` viewer，并保留对象编辑 fallback。
- ADR 0004 已完成 B -> B+，即 ShaderMaterial Gaussian kernel、weighted OIT 和 GPU object-state texture。
- 本 ADR 接续 ADR 0004 的 Phase 3，把 object-aware editing 从 WebGL OIT preview 推进到 WebGPU tile architecture。
