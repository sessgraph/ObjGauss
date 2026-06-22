# ObjGauss 行动队列

> 最近更新: 2026-06-22

## Open

### ACTION-001: 建立 baseline commit

- 原因: 当前仓库历史为空，所有文件仍是未跟踪改动。
- 退出条件: 创建第一个可运行 MVP commit，并在 `project-status.md` / `pr-queue.md` 回填。

### ACTION-002: 确认公开 Demo 许可策略

- 原因: Plush 来源许可混合，只能本地测试；公开 Demo 需要许可更干净的素材。
- 推荐: 优先用 Poly Haven CC0 素材做公开展示样例。
- 退出条件: `docs/asset-library.md` 标明首个可公开 Demo 素材和转换链路。

### ACTION-003: 选择首个训练数据最小子集

- 原因: ARKitScenes / OmniObject3D / ScanNet 规模较大，不能盲目下载全量数据。
- 推荐: 先选 OmniObject3D 单对象或 ARKitScenes 单房间子集。
- 退出条件: 新建任务文档，明确下载规模、许可、转换命令、验收标准。

## Closed

- 暂无。
