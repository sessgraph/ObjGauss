# ObjGauss 行动队列

> 最近更新: 2026-06-22

## Open

### ACTION-016: 用真实 SAM checkpoint 跑小场景 mask manifest

- 原因: `SEG-002A` 已有可选 SAM CLI，但本机尚未提供 SAM checkpoint，因此还没有真实模型输出的 mask manifest 作为验收证据。
- 推荐: Owner 提供本地 checkpoint 路径，或单独确认下载策略；然后对 NeRF Lego 小帧数运行 `objgauss masks from-nerf-sam` 并接 `vote-masks`。
- 退出条件: 一个真实 SAM manifest 被 `objgauss object-field vote-masks` 消费并输出 `object_id` PLY。

### ACTION-006: 接入 SAM / CLIP mask 生成器

- 原因: `SEG-002A` 已接入可选 SAM manifest 生成入口，但还缺真实 checkpoint 小场景验收、CLIP 语义命名和跨视角 slot 对齐。
- 推荐: 不把模型权重放入仓库；先用 Owner 提供的本地 SAM checkpoint 跑 NeRF Lego 小帧数，再决定是否接 CLIP。
- 退出条件: 真实 SAM / CLIP 小场景 mask manifest 被 `objgauss object-field vote-masks` 消费，并输出对象级 PLY。

### ACTION-004: 建立 Poly Haven mesh 到 3DGS 的 Demo 转换链

- 原因: `polyhaven-school-chair-1k` 已可拉取，但仍是 glTF mesh，不能直接进入 3DGS viewer。
- 推荐: 先做 Blender/Three 离线多视角渲染，再接 3DGS 训练。
- 退出条件: 产出 School Chair `.splat` / ObjGauss PLY，并可前端加载。

## Closed

### ACTION-020: 固化 mask vote quality audit

- 完成 commit: 待提交
- 结果: `objgauss object-field vote-masks` summary、外部训练输出登记和三个闭环 demo manifest 现在包含 `vote_quality`；verifier 会检查 `mask_vote_quality_audit_available`，覆盖监督比例、每槽覆盖、冲突比例、normalized target entropy 和观测权重统计。

### ACTION-019: 生成真实 3DGS + 2D 语义 mask 统一闭环样例

- 完成 commit: `ae83594`
- 结果: `objgauss demo plush-semantic-closure` 可从真实 Plush `.splat` 和原始 Gaussian PLY 生成非 KMeans 的 2D color mask manifest，训练 Object Field，导出保留原色的 `object_id` PLY；`verify-plush-semantic-closure`、`audit-v1-goal` 和 `npm run acceptance:demo` 均通过。

### ACTION-018: 固化 ObjGauss v1 阶段目标完成度审计

- 完成 commit: `85943d4`
- 结果: `objgauss demo audit-v1-goal` 可审计阶段目标证据；接入 `plush-semantic-closure` 后当前输出 unified evidence，completion_blockers=`-`。

### ACTION-017: 固化 mask guidance 改变 Object Field 的验收

- 完成 commit: `e5e5154`
- 结果: `verify-v1-closure` 和 `verify-lego-alpha-closure` 现在检查 `mask_guidance_changed_object_field`；本地 `acceptance:demo` 证明 Plush 196457 个 Gaussian、Lego proxy 4960 个 Gaussian 的 hard label 被 mask guidance 改变。

### ACTION-016A: 接入可选 SAM automatic mask manifest 生成器

- 完成 commit: `8c3c80e`
- 结果: `objgauss masks from-nerf-sam` 已接入，可在本地具备 `segment-anything` 和 checkpoint 时输出 `vote-masks` manifest；fake generator 测试已覆盖 manifest 和 `.npy` 写出逻辑。

### ACTION-015: 固化外部 3DGS 训练输出接入命令

- 完成 commit: `721ac49`
- 结果: `objgauss training register-output` 可登记外部训练器产出的 Gaussian PLY / `.splat`，生成 viewer `.splat`，并在提供 mask manifest 时跑 Object Field 投票和导出 `object_id` PLY；真实 NeRF Lego 训练产物仍归 `TRAIN-001`。

### ACTION-014: 固化 NeRF Lego 多 slot 真实 2D mask 生成入口

- 完成 commit: `5302cfe`
- 结果: `objgauss masks from-nerf-rgba-colors` 可从 NeRF Lego 真实 RGBA 图片生成 `yellow`、`red`、`dark`、`other` 四类 mask manifest；独立 `vote-masks` 已消费该 manifest 并输出带 `object_id` 的 PLY。

### ACTION-013: 固化一键闭环总验收命令

- 完成 commit: `81f1d0b`
- 结果: `npm run acceptance:demo` 会重新生成并验证 Plush v1 closure、重新生成并验证 NeRF Lego proxy closure，然后执行 `npm run audit:demo` 浏览器闭环验收；本地验证输出 `acceptance_demo=passed`。

### ACTION-012: 固化闭环 demo 浏览器交互验收

- 完成 commit: `f3e5c62`，截图输出补充 commit: `f1b1190`
- 结果: `npm run audit:demo` 会启动临时 Vite 服务，加载 `ObjGauss v1 闭环样例` 和 `NeRF Lego 闭环代理样例`，检查 splat / 点云编辑 canvas 非空，并执行对象选择、只看所选和预览删除；本地验证 passed，并输出截图到 `/tmp/objgauss-audit-*.png`。

### ACTION-011: 固化 NeRF Lego proxy 闭环机器验收命令

- 完成 commit: `7a250d9`
- 结果: `objgauss demo verify-lego-alpha-closure` 会检查 NeRF 源图像、mask 文件、proxy `.splat`、Object Field `.npz`、loss 下降、`object_id` PLY、public assets 和前端素材注册；本地真实 Lego proxy demo 验证 passed=true。

### ACTION-010: 生成 NeRF Lego 闭环代理样例

- 完成 commit: `db3441a`
- 结果: `objgauss demo lego-alpha-closure` 可从 NeRF Lego 真实多视角 RGBA + pose 生成 `lego_proxy.splat`、真实 2D color mask manifest、Object Field 和 `lego_v1_objects.ply`；前端素材库可加载 `NeRF Lego 闭环代理样例` 并执行对象隔离/删除预览。

### ACTION-009: 固化 v1 闭环机器验收命令

- 完成 commit: `b6236bd`
- 结果: `objgauss demo verify-v1-closure` 会检查真实 `.splat`、mask manifest、Object Field `.npz`、loss 下降、`object_id` PLY、public copy 和前端素材注册；本地真实 Plush demo 验证 passed=true。

### ACTION-008: 生成 NeRF Lego 真实图片 alpha mask manifest

- 完成 commit: `e96b024`
- 结果: `objgauss masks from-nerf-alpha` 可从 NeRF Synthetic Lego RGBA alpha 通道生成 boolean `.npy` masks 和 `vote-masks` manifest；本地验证 8 frames / 8 masks / 800x800 / 299242 foreground pixels。

### ACTION-007: 固化 v1 闭环验收 demo

- 完成 commit: `6802e7f`
- 结果: `objgauss demo v1-closure` 可生成 `outputs/demos/v1-closure/` 和 `public/samples/plush_v1_objects.ply`，前端素材库可加载 `ObjGauss v1 闭环样例`。

### ACTION-005: 建立 Object Field 真实训练循环

- 完成 commit: `af825f8`
- 结果: 已通过 `objgauss object-field vote-masks` 实现 projection supervision 训练循环；完整 3DGS render loss 另行立项。

### ACTION-003: 选择首个训练数据最小子集

- 完成 commit: `9c88666`
- 结果: 选择并接入 `nerf-synthetic-lego`，实际抽取 805 个文件到 `outputs/assets/training/nerf-synthetic-lego/`。

### ACTION-002: 确认公开 Demo 许可策略

- 完成 commit: `9c88666`
- 结果: 选择并接入 Poly Haven CC0 `SchoolChair_01` 作为首个许可干净 Demo 输入源；完整前端 Demo 仍需 ACTION-004。

### ACTION-001: 建立 baseline commit

- 完成 commit: `c8dcef7`
- 结果: 创建第一个可运行 MVP commit，并在 `project-status.md` / `pr-queue.md` 回填。
