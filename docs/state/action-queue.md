# ObjGauss 行动队列

> 最近更新: 2026-07-07

## Open

### ACTION-006: 接入 SAM / CLIP mask 生成器

- 原因: `SEG-002` 已完成真实 SAM checkpoint 小场景验收，`SEG-CLIP-001` 已完成 manifest-level 跨视角 slot alignment，`CLIP-SCORE-001` 已完成可选 CLIP score cache contract；`CLIP-RUN-001` 已跑通真实 `transformers` CLIP inference，`CLIP-QUALITY-001`、`CLIP-SLOT-QUALITY-002`、`CLIP-BASELINE-003`、`CLIP-QUALITY-004`、`CLIP-BALANCE-001` 和 `CLIP-COVERAGE-001` 已落地 mask-level / slot-level naming quality gate、baseline comparison、promotion policy、slot naming diversity policy、slot support rebalance policy 与显式 foreground coverage recovery 机制。真实 CLIP 语义路线的 slot balance blocker 已清除，但整体仍保持 `do-not-promote`。
- 推荐: 不把模型权重放入仓库；真实 CLIP / SAM 证据链已用
  `align-slots --recover-foreground-coverage -> vote-masks -> compare-baselines` 重跑，
  supervised fraction 只从 `0.114283` 提升到 `0.114960`，仍低于 promotion policy 的
  `>=0.200000`，且 mask-level 背景 dominant blocker 仍存在。下一步应改进 SAM / CLIP
  mask selection、crop 策略或 label policy 来扩大真实 foreground coverage；promotion 前
  仍必须通过 `objgauss masks compare-baselines`。
- 退出条件: 真实 SAM / CLIP 小场景 mask manifest 被真实 CLIP 分数稳定语义命名、跨视角对齐，并通过 mask-level / slot-level 命名覆盖率、slot balance、vote quality、training summary 与 color-mask / KMeans baseline 对比。

### ACTION-004: 建立 Poly Haven mesh 到 3DGS 的 Demo 转换链

- 原因: `polyhaven-school-chair-1k` 已可拉取，但仍是 glTF mesh，不能直接进入 3DGS viewer。
- 推荐: 先做 Blender/Three 离线多视角渲染，再接 3DGS 训练。
- 退出条件: 产出 School Chair `.splat` / ObjGauss PLY，并可前端加载。

## Closed

### ACTION-023: 实现 bounded evidence normalization candidate

- 完成 commit: 本提交
- 结果: `real-sample-v2-sample-aware-weight-policy` 新增 `bounded-normalized` 候选和
  `objgauss-bounded-evidence-normalization-v1` summary，记录 feature / position blend、
  bounded confidence gain 和 entropy reduction。Lego 仍选择 `promoted`，
  `mixed_gaussians=0`、`hard_regression=0`；Polyhaven 选择 `bounded-normalized`，
  `evidence_normalization_status=satisfied_by_bounded_normalization`，selected
  `hard_regression=0`，blocked promoted candidate 仍记录 `hard_regression=1814`。
  本项不解冻 geometry / camera / dynamic-K，不引入 diffusion / rollout / replay buffer。

### ACTION-022: 下载真实 Gaussian cloud 并清理 viewer demo 入口

- 完成 commit: 本提交
- 结果: 新增 `nike-3dgs-local` asset registry，已用 `uv run objgauss assets pull
  nike-3dgs-local --force` 下载 `nike.splat` 并生成 ignored
  `public/samples/nike.splat` / `public/samples/nike_objects.ply`。该样例为
  `270,491` Gaussians / 4 个 object，counts `84,781 / 69,968 / 74,734 / 41,008`。
  Viewer catalog 新增 `nike-real-splat-demo`，首屏 dock / 默认 stage 收敛为 5 个 curated
  demos；near-1M、OGC、trainable artifact 和旧 closure 保留在高级模型版本 / URL 调试路径。

### ACTION-021: 让主展示台真实 `.splat` 对象子集随对象移动

- 完成 commit: `1ced889`、`67343a1`；本轮 `GAUSSIAN-OBJECT-PROCESS-FLOW-001` 再次验证。
- 结果: `NATIVE-SPLAT-OBJECT-TRANSFORM-001` 已在主 `ThreeWorld` 建立
  `.splat index -> object_id -> object translate` 路径；`NATIVE-SPLAT-MOTION-HARDEN-001`
  已把真实绑定状态显示到 UI，并证明 selected source splat 子集移动、peer 对象不动。
  本轮 raw -> handoff -> object layer flow 的 desktop browser check 进一步验证生成结果模型
  `real-sample-v2-sample-aware-lego::object-0` 移动后 native source splat motion
  `active=true`、`transformedObjects=1`、`selectedScreenDelta=28.513px`，peer world delta 为 `0`。

### ACTION-016: 用真实 SAM checkpoint 跑小场景 mask manifest

- 完成 commit: `18ac234`
- 结果: 本机使用 `segment-anything`、`torch 2.12.1+cu130`、RTX 5060 Ti 和本地 `sam_vit_b_01ec64.pth` 生成 NeRF Lego 2 帧真实 SAM manifest；`objgauss object-field vote-masks` 消费该 manifest，监督 5567 / 5696 个 Gaussian，loss 3.902681 -> 0.120758，并输出带 `object_id` 的 `outputs/demos/lego-sam-smoke/lego_sam_objects.ply`。

### ACTION-020: 固化 mask vote quality audit

- 完成 commit: `6a32018`
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
