# ObjGauss 行动队列

> 最近更新: 2026-06-22

## Open

### ACTION-004: 建立 Poly Haven mesh 到 3DGS 的 Demo 转换链

- 原因: `polyhaven-school-chair-1k` 已可拉取，但仍是 glTF mesh，不能直接进入 3DGS viewer。
- 推荐: 先做 Blender/Three 离线多视角渲染，再接 3DGS 训练。
- 退出条件: 产出 School Chair `.splat` / ObjGauss PLY，并可前端加载。

## Closed

### ACTION-003: 选择首个训练数据最小子集

- 完成 commit: `9c88666`
- 结果: 选择并接入 `nerf-synthetic-lego`，实际抽取 805 个文件到 `outputs/assets/training/nerf-synthetic-lego/`。

### ACTION-002: 确认公开 Demo 许可策略

- 完成 commit: `9c88666`
- 结果: 选择并接入 Poly Haven CC0 `SchoolChair_01` 作为首个许可干净 Demo 输入源；完整前端 Demo 仍需 ACTION-004。

### ACTION-001: 建立 baseline commit

- 完成 commit: `c8dcef7`
- 结果: 创建第一个可运行 MVP commit，并在 `project-status.md` / `pr-queue.md` 回填。
