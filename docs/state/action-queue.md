# ObjGauss 行动队列

> 最近更新: 2026-06-22

## Open

### ACTION-006: 接入 SAM / CLIP mask 生成器

- 原因: `vote-masks` 已能消费 mask manifest，`MASK-001` 也能从 NeRF Lego alpha 生成前景 mask；但仓库还不能自己从图片生成 SAM / CLIP 语义或实例 mask。
- 推荐: 先做可选依赖和离线命令，不把模型权重放入仓库。
- 退出条件: 小场景图片可生成 mask manifest，并被 `objgauss object-field vote-masks` 消费。

### ACTION-004: 建立 Poly Haven mesh 到 3DGS 的 Demo 转换链

- 原因: `polyhaven-school-chair-1k` 已可拉取，但仍是 glTF mesh，不能直接进入 3DGS viewer。
- 推荐: 先做 Blender/Three 离线多视角渲染，再接 3DGS 训练。
- 退出条件: 产出 School Chair `.splat` / ObjGauss PLY，并可前端加载。

## Closed

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
