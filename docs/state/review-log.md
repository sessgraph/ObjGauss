# 审查日志

> 追加式记录。每次多角度评审/审查的结论写为一条带日期的条目。
> P0/红线项须同时回写到 `risks.md` 与 `action-queue.md` 三审。

---

## 2026-07-07 · 全项目多角度评审

**评审方法**：5 个并行子代理分别覆盖架构、Python 正确性、前端 WebGPU、测试质量、仓库卫生，所有结论带 file:line 证据并交叉验证。

### 总体结论

这是一个**流程纪律远超平均、但核心算法存在高危缺陷**的项目。仓库卫生、commit 规范、素材许可分层、门禁零松动都做得很好（B+）；但 Python 训练/稳定性管线有 3 处必须立即处理的高危问题，其中一个疑似触犯了开发流程的红线。前端则是可运行但维护债重的状态（C+）。

### 🔴 P0 — 必须立即处理

#### 1. 疑似红线违规：commit `a10d6c5` 调松证据归一化门并同步改弱测试

`objgauss/core/real_sample_v2_sample_aware_weight_policy.py:688-719` 中，当 promoted 权重硬回归时 `hard_safety_blend=0.0`，导致 `bounded_feature_weight == baseline_feature_weight`——即"候选"用与 baseline 完全相同的 cloud/handoff/seed/weights 重渲染，**bit-identical 于 baseline**，却被命名为 `bounded-normalized` 判定 eligible，使 `bounded_satisfied=True` → `requires_evidence_normalization=False`。

同一 commit `a10d6c5` 把测试 `test_sample_aware_weight_policy_selects_baseline_when_promoted_regresses_policy` 的断言从 `baseline` 改成 `bounded-normalized` + `requires_evidence_normalization is False`，**测试名仍叫 selects_baseline 却断言 bounded-normalized**。门从"阻止全局权重提升"翻转为"全局放行"，靠的是把 baseline 改名。

直接违反 `docs/development-flow.md` §6「不要为通过验证而调松门禁或改弱测试断言」和 §10 红线。**建议**：要求 `blend>0` 才能 `bounded_satisfied`；回滚测试改弱部分；回写 `risks.md`。

#### 2. 监督交叉熵梯度在 `1e-8` clip 边界爆炸（训练发散/NaN 源）

`objgauss/core/assignment_losses.py:214-222`：`assignment[i,j]=0` 且 `target>0` 时，`clipped=1e-8`，梯度 `-target/1e-8 ≈ -1e8`。fp16/AMP 下溢为 inf 并 NaN 传播。且损失用 `log(clipped)`（eps 以下平坦）但梯度用 `1/clipped` 当 clip 恒等——**数学不一致**，clip 区域真实次梯度应为 0。触发条件是常见的"零质量 slot + 目标期望有质量"早期训练状态。

commit `a10d6c5` 标题声称"bounded evidence normalization"，但实际只动了 viewer-preview 的标量权重 clamp，**完全未触及损失管线**（H1）。

#### 3. 稳定性门可"空跑绿门"

`v2_stability_gate.py:135-156` + `v2_stability_diagnostics.py:378-379`：未提供 `predicted_assignments` 时回退到用 oracle 自己的期望当"预测"，所有 gate 返回 True，CI 可在求解器从未运行的情况下报绿。配套：`zip` 静默截断长度不匹配的预测（H5，`:382`）、从不校验预测矩阵列数等于 fixture slot 数（H6，`:549-570`，2 列分配在 log(2) 尺度下 entropy 永远 ≤1.0 误判通过）。

**建议**：未提供预测时 raise 或返回 sentinel 强制 fail；函数顶部 `assert len(predicted)==len(observations)`；强制 `matrix.shape[1]==fixture.world.oracle.slots`。

### 各角度要点

#### 角度 1 · 架构与代码组织

- **巨型文件已产出真实 bug**：`cli.py` 4318 行中 `_format_optional_float` 重复定义两次（L2277 返回 `"none"`，L4313 返回 `"-"`），后定义覆盖先定义，所有早期调用方拿到 `"-"`。`App.jsx` 9442 行内 `App` 组件单体 ~1866 行、`ThreeWorld` ~1495 行、`DebugPanel` 接收 **61 个 prop**。
- **v1/v2 分层其实干净**：顶层 `objgauss/*.py` 全是纯 re-export shim，单一权威来源在 `core/`，无重复实现。但"v2"后缀已无 v1 对照，是历史包袱。
- **`core/__init__.py` 902 行是懒加载注册表**，非上帝模块，可接受。
- **ADR 0005 状态漂移**：标 `Proposed` 但代码已实现全部验收标准（`WebGpuTileViewport.jsx` 的 DOM 审计属性齐备），应升为 `Accepted/Implemented`。
- **`real_sample_v2_*` 9 文件切碎**：promoted_weights 与 sample_aware_weight_policy import 同一组符号且都做"cross-sample 权重扫描+gate"（合计 1449 行职责重叠），应合并。

#### 角度 2 · Python 核心正确性

P0 三个已列。其余中危（多为一行修复）：
- **M1** 温度前向 clamp 到 1e-8、反向除以未 clamp 的 1e-12，`T=1e-12` 合法时梯度被放大 1e4 倍（`assignment_solver_v2.py:828 vs :760`）。
- **M2** 0 行分配矩阵被 `validate_assignment_matrix` 放行 → `np.mean(empty)` NaN 污染总损失（`object_state.py:165-188`）。
- **M4** `_validate_weight` 只查 `value<0`，`nan<0`/`inf<0` 均为 False，放行 NaN/inf 权重（`assignment_losses.py:317-319`）。
- **M7** `make_synthetic_world_state` 的 `seed` 参数是死代码，不同 seed 产出同一世界。
- **M9** `_BORDERLINE_ENTROPY_MARGIN=0.03` 硬编码无依据。

**稳健面**：softmax 用 max-subtraction 无溢出、除法普遍 `max(...,eps)` 保护、纯 numpy 无 autograd 泄漏、seed 用局部 PCG64 严格可复现、checkpoint 往返干净、收敛判定不可欺骗（无基于收敛的早停）。

#### 角度 3 · 前端 React + WebGPU

- **关键事实**：`App.jsx` 实际只挂载 `<ThreeWorld>`（three.js WebGL + spark），而 `WebGpuTileViewport.jsx`/`SplatViewport.jsx`/`PointCloudViewport.jsx` **在 App 中零引用**——并存未接线的旁路渲染路径。降低 WebGPU 隐患对线上的直接影响，但增加双轨漂移风险。
- **每渲染重建派生对象**：`selectedDebugSnapshot` 等 ~12 个汇总对象未 `useMemo` 且作 effect 依赖（`App.jsx:228/257/262` 等），window 全局每渲染 toggle，易修。
- **WebGPU 旁路隐患**：零 `context.unconfigure()`（canvas 重建泄漏）、device lost 仅置位无恢复、readback 在帧路径上 `mapAsync` 阻塞主线程——启用 tile 管线前必修。
- **解析器相对健壮**：`ply.js`/`ogcDecoder.js` 有 magic 校验、显式 little-endian、嵌套长度越界校验。缺口：`ply.js:166-175` 循环内读取前不校验 `offset+size<=byteLength`；全文件一次性加载无流式。
- **构建**：`@vitejs/plugin-react` 误入 `dependencies`；`three ^0.180` 语义不安全（three 不守 semver）；无 chunking 致 9442 行 App 产出单一大 bundle。
- **亮点**：ThreeWorld mount effect cleanup **完整彻底**（RAF/事件/controls/renderer/spark/DOM 全清理）；CSS 干净（0 `!important`、4 `z-index`）。

#### 角度 4 · 测试质量与门禁

- **门禁零松动是最大亮点**：0 `skip`、0 `xfail`、0 `assert True`、0 吞异常、0 容差加宽历史（git log 验证）。54 个 audit/acceptance 脚本里 0 个空壳/MOCK，40 个真验收，2 个 SOFT 也诚实自标。
- **主要短板是断言质量**：`test_core_namespace.py` ~50 处 `assert <symbol> is not None`（全仓库最大弱断言簇）；3 处 self-oracle（`test_solver_decoder_training.py:895` scalar_count 自指、`test_objgauss_mvp.py:2442-2472` 自报 loss_decreased、`test_core_namespace.py` 的 `validate_X is X` 簇 13 处——no-op 实现也能过）。
- **2 个 SOFT audit 脚本**默认无证据通过（`audit-commercial-demo-readiness`、`audit-near1m-production-gap`），进 CI 前应默认 require-flags 或改名 `report-*`。
- **稳健性缺口**：全仓库 0 seed + 0 fixture，目前不是 flake 源但一旦引入随机路径会立即 flake。
- 三个榜样文件（`test_model_manifest`/`test_object_emergence_solver`/`test_v2_stability_foundation`）用固定期望值、hashlib 对比、allclose、负路径，是可复用范式。

#### 角度 5 · 仓库卫生与流程一致性

- **最突出问题：状态文件体积失控**。`pr-queue.md` **836 KB / 11037 行 / 367 条目**（99% 是 Done 历史），`project-status.md` 344 KB / 3174 行，均已超 Read 单次上限。每次 feat commit 追加 60-90 行从不停留快照，违背 §1.1「注意力比代码更贵」。**建议**：只保留 ready/in-progress/planned + 最近 ~20 条 done，历史归档到 `docs/state/archive/`。
- **`AGENTS.md`（9KB）与 `docs/development-flow.md` 大段重复**，红线/完成定义文字不完全一致（AGENTS.md 多一条"不删除他人未提交改动"），违背 §1.3 单一权威来源。`CLAUDE.md` 则正确地只是 pointer。建议 AGENTS.md 也收成 pointer。
- **亮点**：commit 规范 100%（502 个 commit 全 conventional）、工作区干净、状态回写及时、素材许可分层清晰、0 个大型二进制文件误入 git、`risks.md` 记的是真实风险。
- `src/assetLibrary.js`（20 id）与 `objgauss/assets.py`（14 id）是 dev-flow §2 明确定义的不同用途事实源，非违规，但 14 个重叠拉取型资产元数据双处硬编码无校验，建议加 contract test 比对。

### 优先级修复清单

| 优先级 | 事项 | 角度 |
|---|---|---|
| **P0** | 复核 commit `a10d6c5` 是否调松证据归一化门 + 改弱测试（红线） | Python |
| **P0** | 修 `assignment_losses.py:214-222` 梯度爆炸（clip 区域梯度置零） | Python |
| **P0** | 修稳定性门空跑绿门 + zip 截断 + 列数不校验（H4/H5/H6） | Python |
| **P1** | 删 `cli.py` 重复定义的 `_format_optional_float`（L4313） | 架构 |
| **P1** | 修 M1 温度 clamp 不一致、M2 0 行矩阵 NaN、M4 NaN/inf 权重 | Python |
| **P1** | 状态文件瘦身 + 归档机制（pr-queue.md 836KB） | 卫生 |
| **P1** | AGENTS.md 收敛为 pointer，消除与 dev-flow 重复 | 卫生 |
| **P2** | App.jsx 派生对象 `useMemo`（低风险高收益） | 前端 |
| **P2** | 提升 `test_core_namespace.py` 50 处 `is not None` 为行为断言；修 3 处 self-oracle | 测试 |
| **P2** | 依赖修正：`plugin-react` 移入 devDeps、`three` 锁 `~0.180` | 前端 |
| **P2** | ADR 0005 状态升为 Implemented | 架构 |
| **P3** | 拆 App.jsx 纯函数层 + 引入 DebugContext/Zustand | 架构 |
| **P3** | 合并 `real_sample_v2_*` 强耦合文件、去掉 v2 后缀 | 架构 |
| **P3** | WebGPU unconfigure + device-lost 恢复（启用 tile 管线前必修） | 前端 |
| **P3** | 删除/隔离 3 套未接线的渲染路径 | 前端 |

### 值得肯定的地方

- **门禁纪律**：零 skip/xfail/容差加宽，audit 脚本无空壳。
- **流程骨架**：commit 规范 100%、状态回写及时、素材许可分层清晰、无大型素材误入 git、risks.md 记真实风险。
- **数值稳定主路径**：softmax/logsumexp 溢出安全、纯 numpy 无 autograd 泄漏、seed 可复现、checkpoint 往返干净。
- **前端 cleanup 彻底**：ThreeWorld mount effect 的资源清理无真实泄漏。
- **解析器健壮**：ply/ogc 有 magic 校验、显式端序、嵌套长度越界校验。

### 后续动作（待三审回写）

- [x] P0-1（红线）：已复核 commit `a10d6c5`；历史结论成立，但当前 HEAD 已由
  `3cc6cd0 fix(training): harden sample-aware cross-sample gate` 修正，并在
  `ACTION-025` 关闭记录中覆盖。
- [x] P0-2 / P0-3：已写入 `action-queue.md` 与 `risks.md`，等待实现切片。
- [ ] P1 状态文件瘦身：写入 `action-queue.md`。

### 本次验证（2026-07-07）

- P0-1 当前状态：**历史有效，当前已修**。`a10d6c5` 确实把 Polyhaven promoted
  regression case 从 `baseline` 改成 `bounded-normalized`，并让
  `requires_evidence_normalization=False`；当前 `3cc6cd0` 已把
  `feature_weight_blend=0` 的 bounded candidate 改为
  `bounded_evidence_normalization_noop_baseline_fallback`，selected policy 回到
  `baseline`。
- P0-2 当前状态：**仍有效**。运行时探针复现
  `supervised_assignment_loss_and_gradient([[0, 1]], target=[[1, 0]])`
  输出 gradient `[-100000000.0, -0.0]`。
- P0-3 当前状态：**部分有效**。无 `predicted_slots` / `predicted_assignments` 时
  `evaluate_synthetic_stability_gate(...)` 仍用 oracle expected slots 并返回
  `synthetic_stability_gate_pass`；`predicted_assignments` 长度不匹配当前已显式
  `ValueError`，日志中的 zip 静默截断子项不再成立；assignment column count 仍无显式
  slot 数校验，应在 gate 输入层 fail fast。
- 验证命令：
  - `uv run --extra dev pytest tests/test_real_sample_v2_sample_aware_weight_policy.py tests/test_real_sample_v2_bounded_normalization_cross_sample.py tests/test_v2_stability_gate.py tests/test_v2_stability_diagnostics.py -q`
  - runtime probes for supervised CE gradient and stability gate fallback behavior.

---

## 2026-07-11 · stabilization correctness follow-up

- 2026-07-07 的 CE 风险已按坐标空间拆清：clip plateau 上的 probability-space 导数为
  `0`；刚高于 `EPS` 时精确导数仍是大幅的 `-target / p`，不能用梯度裁剪伪装修复。
  训练消费者现直接使用解析 softmax-logit VJP
  `p * sum(active_target) - active_target`，避免 materialize `1 / p` 和二次 Jacobian。
- `b054e96` 已覆盖 `EPS` 两侧、logits finite-difference 与两个 solver 的单步 loss 下降；
  全量 806 个 Python tests 通过。当前风险口径以 `risks.md` 的 R-015 为准。
- controlled evidence 相关 follow-up 分别由 `506b793`、`ca22c71`、`21f2744` 与
  `c60b069` 收紧 association、handoff 重算、GT-preassociation 和 trainable ObjectState
  artifact ABI；
  这些修复不等于真实 3-scene gate 已通过。
- `3acfe73` 将 viewer full/truth audit 对齐当前 evidence UI，并让任何显式 failed status
  阻断顶层通过；该审计闭环不改变 Viewer 的 research-prototype 定位。
