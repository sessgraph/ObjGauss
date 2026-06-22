# ObjGauss 行动队列

> 最近更新: 2026-06-22

## Open

### ACTION-006: 接入 SAM / CLIP mask 生成器

- 原因: `vote-masks` 已能消费 mask manifest，但仓库还不能自己从图像生成语义/实例 mask。
- 推荐: 先做可选依赖和离线命令，不把模型权重放入仓库。
- 退出条件: 小场景图片可生成 mask manifest，并被 `objgauss object-field vote-masks` 消费。

### ACTION-004: 建立 Poly Haven mesh 到 3DGS 的 Demo 转换链

- 原因: `polyhaven-school-chair-1k` 已可拉取，但仍是 glTF mesh，不能直接进入 3DGS viewer。
- 推荐: 先做 Blender/Three 离线多视角渲染，再接 3DGS 训练。
- 退出条件: 产出 School Chair `.splat` / ObjGauss PLY，并可前端加载。

## Closed

### ACTION-007: 固化 v1 闭环验收 demo

- 完成 commit: pending
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
