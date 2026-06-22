# ObjGauss 当前状态总览

> 最近更新: 2026-06-22

## 当前阶段

MVP 原型可运行，已完成流程化基线提交，已接入真实 3DGS splat renderer，并具备可复现的 ObjGauss v1 闭环验收 demo。

## 已完成能力

- Python CLI:
  - `objgauss convert-splat`
  - `objgauss cluster`
  - `objgauss colorize`
  - `objgauss filter`
  - `objgauss stats`
  - `objgauss assets list/pull`
  - `objgauss masks from-nerf-alpha`
  - `objgauss object-field init/export/stats/inspect-nerf/vote-masks`
- 前端:
  - 中文 UI。
  - Spark / Three.js 真实 3DGS splat 预览。
  - Three.js 高斯中心点云编辑 fallback。
  - 自身颜色 / 对象聚类色切换。
  - 对象列表、隔离、删除预览。
  - 素材库卡片和本地 Plush 样例加载。
- 素材:
  - `plush-3dgs-local` 可自动拉取。
  - Plush `.splat` 用于真实 renderer，`plush_objects.ply` 用于对象级编辑。
  - `polyhaven-school-chair-1k` 可自动拉取到 mesh Demo 输入目录。
  - `nerf-synthetic-lego` 可自动拉取到训练素材目录。
  - ARKitScenes、ScanNet、OmniObject3D、Google Scanned Objects、Poly Haven、Mip-NeRF 360、Tanks and Temples 已登记为候选来源。
- Object Field:
  - 已有 `object_logits: (N, K)` 软分区文件格式。
  - 可从现有 Gaussian PLY warm start，并导出 hard `object_id` PLY 复用前端。
  - 可检查 NeRF-style `transforms_*.json` 训练素材完整性。
  - 可从 NeRF Synthetic RGBA alpha 通道生成真实图片 mask manifest。
  - 可消费预计算 SAM / CLIP / 2D mask manifest，并投影投票到 Gaussian。
  - 可通过 projection loss 更新 Object Field logits。
- Demo:
  - `objgauss demo v1-closure` 可生成当前 v1 闭环验收包。
  - 前端素材库已有 `ObjGauss v1 闭环样例`，加载后可查看真实 splat 外观并执行对象隔离/删除预览。
- 流程:
  - `docs/development-flow.md` 已建立。
  - `AGENTS.md` 和 `CLAUDE.md` 已指向统一流程。
  - baseline commit: `c8dcef7`.

## 最近验证

2026-06-22:

```bash
uv run --extra dev pytest
npm run build
```

结果：

- Python 测试: 14 passed。
- 前端构建: 通过。
- 浏览器验证: 桌面 1440x920 与移动端 390x844 均渲染非空、无前端错误。
- ASSET-001: Poly Haven School Chair 实际拉取 5 个文件；NeRF Synthetic Lego 实际抽取 805 个文件。
- OBJFIELD-001: Plush PLY 可初始化 6-slot Object Field；NeRF Lego 检查 400 frames、缺图 0、无效 pose 0。
- SEG-001 / OBJFIELD-002: synthetic projection mask vote 可训练 Object Field，并输出 `object_id` PLY。
- DEMO-001: Plush v1 闭环 demo 生成 281498 个 Gaussian、6 个对象、3 个投影视角、18 个 masks；projection loss 1.791760 -> 1.201637；浏览器验证可加载 `ObjGauss v1 闭环样例` 并执行对象选择、隔离、删除预览。
- MASK-001: NeRF Lego 真实 RGBA 图片 alpha 生成 mask manifest，8 frames / 8 masks / 800x800 / 299242 foreground pixels。
- 已知提示: Vite 报 Spark / Three.js chunk 超过 500KB，不影响当前预览。

## 当前限制

- 对象聚类色、隐藏、隔离、删除预览仍通过点云编辑 fallback 完成，不是对象级 splat shader。
- 当前 v1 闭环 demo 的 mask manifest 由已有对象标签派生，用于验收闭环；NeRF Lego alpha mask 已能从真实图片生成，但只是前景 mask，不等价于 SAM / CLIP 实例语义分割。
- 仓库内还不运行 SAM / CLIP 模型。
- 当前训练循环是 projection supervision，不是完整 3DGS render loss 联合训练。
- Poly Haven mesh Demo 还不能直接进入现有 3DGS viewer，需要后续 mesh 多视角渲染和 3DGS 训练。
- 训练素材目录已接入 NeRF Lego，但还没有对应训练出的 Lego Gaussian PLY。

## 下一步主线

1. 执行 SEG-002: 接入可选 SAM / CLIP mask 生成器，输出当前 mask manifest。
2. 执行 TRAIN-001: 训练 NeRF Lego 得到 Gaussian PLY，再跑 `vote-masks` 验收真实数据。
3. 建立 Poly Haven mesh -> 多视角渲染 -> 3DGS 训练的 Demo 转换链。
4. 后续 renderer 优化: Spark 按需加载或拆包，降低首屏 bundle。
