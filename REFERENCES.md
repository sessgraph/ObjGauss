# Project Resource Registry

本文件是新项目对**候选数据、归档资源、外部地址、许可状态和历史证据边界**的唯一台账。
记录在此不代表资源已获准下载、移植、训练、再分发或用于研究结论。

## 状态词汇

| 状态 | 含义 |
| --- | --- |
| `verified_archive_fact` | 已从只读归档标签核对；未在新分支重新运行 |
| `verified_upstream_fact` | 已从固定的官方一手来源核对版本、内容或元数据 |
| `approved_local_preview` | Owner 已批准下载到 ignored 本地目录，仅用于声明的预览范围 |
| `owner_brief_candidate` | 来自 Owner 提供的研究材料；尚未独立核验上游现状 |
| `requires_upstream_review` | 使用前必须核对官方版本、字段、许可、大小和条款 |
| `reference` | 只参考思想、边界或失败经验，不复制实现 |
| `candidate-port` | 许可和技术栈批准后，可按具体文件连同测试评审移植 |
| `do-not-migrate` | 不应整体恢复或作为新项目事实源 |

## Stage-0 固定本地预览

Owner 于 2026-07-14 明确要求看到渲染后的 3D Gaussian，而不是 notebook，因此批准下载下表
固定小型 `.splat` 到 ignored 本地目录，并仅用于 Stage-0 渲染审计样例。页面默认世界由
`viewer/synthetic-world.mjs` 确定性生成，不含第三方输入；下表外部文件只在用户选择
“Lego 审计样例”时加载。该授权不延伸到其他资产、训练数据、模型或再分发。

| 项 | 当前事实与边界 |
| --- | --- |
| 上游文件 | `GitHubDragonFly/GitHubDragonFly.github.io` 的 `viewers/examples/legobrick.splat` |
| 固定版本 | commit `1267e2135660e1f4197f94c045453fe40c209b0e` |
| 固定 URL | <https://raw.githubusercontent.com/GitHubDragonFly/GitHubDragonFly.github.io/1267e2135660e1f4197f94c045453fe40c209b0e/viewers/examples/legobrick.splat> |
| 获取命令 | `bash scripts/fetch-gaussian-preview.sh`；已有文件复用前与新下载落盘前均校验大小和 SHA-256 |
| 本地位置 | `data/local-preview/legobrick-1267e213/legobrick.splat`（Git ignored） |
| 大小与记录数 | `3,297,920` bytes；`103,060` 条固定长度记录 |
| SHA-256 | `d5131a664a12a8764da70552c85f567d276313110f63f1efd48424845917899e` |
| 格式 | `antimatter15-splat-v1`；每条 `32` bytes，包含 position、scale、RGBA 与 quaternion |
| 语义状态 | `semantic_kind=point-derived-splat`；`asset_provenance=unverified` |
| 许可边界 | 容器仓库存在 MIT 声明，但当前没有核验该样例的作者、生成链、输入来源或逐资产许可；仓库许可不能替代资产 provenance，因此只批准 ignored 本地预览，不批准提交或再分发 |
| 允许声明 | 固定文件可被下载、校验、解析，并以 covariance、Gaussian alpha 与深度排序在本地 WebGL2 页面渲染 |
| 禁止声明 | 不能称为 trained 3DGS、ObjGauss 模型输出或重建/对象状态/动力学/规划证据，也不能据此宣称任何研究门通过 |

页面采用的 `32`-byte `.splat` 布局和 WebGL Gaussian 渲染方法参考
[`antimatter15/splat`](https://github.com/antimatter15/splat)；该参考实现为 MIT。参考实现的许可
只覆盖相应代码与格式参考，不能补齐上述外部审计样例的 provenance 或许可链。

## 新项目候选数据源

以下入口来自 2026-07-14 Owner 研究材料。包括 Kubric / MOVi 在内的原始数据当前仍为
`owner_brief_candidate + requires_upstream_review`；Stage-0 `.splat` 预览不是其中任一原始
数据集的已接入证明。表中的“候选职责”是 PRD 的规划输入，不是对数据字段或研究适用性的
已验证陈述。

| 候选 | Owner 材料中的入口 | 候选职责 | 不能直接支持的声明 |
| --- | --- | --- | --- |
| Kubric / MOVi | <https://github.com/google-research/kubric/blob/main/challenges/movi/README.md> | 坐标、实例、物性和 Oracle Gaussian 基础 | 真实动作或 sim-to-real |
| HO-Cap | <https://irvlutd.github.io/HOCap/> | 真实多视角 ObjectBelief 与 pose 外测 | 严格动作反事实 |
| HOT3D | <https://facebookresearch.github.io/hot3d/> | 遮挡、重识别和头戴视角压力测试 | measured action 因果 |
| ManiSkill 3 | <https://maniskill.readthedocs.io/en/latest/user_guide/index.html> | snapshot/reset 和 sibling actions | 真实机器人价值 |
| CausalWorld | <https://causal-world.readthedocs.io/en/latest/> | 显式物性干预与 OOD 组合 | sim-to-real |
| Physion++ | <https://dingmyu.github.io/physion_v2/> | 隐藏物理外部验证 | 机器人动作执行价值 |
| RH20T | <https://rh20t.github.io/> | 真实接触、力和执行偏差校准候选 | 严格同起点反事实，除非另行构造 |
| DROID | <https://droid-dataset.github.io/> | 后期真实视觉—动作预训练候选 | 核心因果真值 |
| CALVIN | <https://github.com/mees/calvin> | 后期长时序 rollout/规划候选 | 真实机器人最终价值 |

BridgeData、RoboNet、Open X-Embodiment 暂只记录为后期预训练候选。`RES-001` 必须使用
上游一手资料补齐每个候选的版本、许可、字段矩阵、下载范围、校验和和存储预算；在此之前
不得把附件中的规模、字段或许可证描述提升为 `confirmed_fact`。

## 归档资源处置台账

所有路径均相对于 `archive/objgauss-final-2026-07-14`。

| 资源 | 级别 | 候选价值与边界 |
| --- | --- | --- |
| `docs/adr/0009-objectstate-multi-object-evidence.md` | `reference` | whole-scene split、独立 GT、hard cases、简单 baseline 与失败可见性 |
| `docs/state/project-status.md`、`docs/state/risks.md` | `reference` | 归档结束时的权威负证据与技术债，不是新项目状态 |
| `docs/architecture/objgauss-v1-kernel-contract.md` | `reference` | ObjectState/Gaussian/object_id 的概念边界 |
| `docs/architecture/objectstate-model-contract.md` | `reference` | `Gaussian/AssignmentEvidence -> A[N,K] -> ObjectState -> gate` 的审计链 |
| `docs/dataset/controlled-reality-contract.md` | `reference` | episode/object/action/transition 与 readiness 分账；缺 sibling lineage 等新需求 |
| `docs/architecture/objectstate-teacher-evidence-contract.md` | `reference` | teacher provenance/confidence/uncertainty 与 GT 泄漏禁令 |
| `objgauss/core/assignment_evidence.py`、`objgauss/core/object_state.py` | `candidate-port` | 小型数据结构参考；需重新核对语义、许可和测试 |
| `objgauss/datasets/objectstate_controlled_capture.py`、`objectstate_transition_dataset.py` | `candidate-port` | validator 原型；旧字段不足，不能原样冻结为新 contract |
| `objgauss/evaluation/objectstate_reality_gate.py` | `candidate-port` | pass/fail/blocked 与指标独立重算思想；旧阈值不可继承 |
| `objgauss/core/io_ply.py`、`io_splat.py`、`features.py`、`assignment_metrics.py` | `candidate-port` | 可能独立提取的纯基础件；必须连同行为测试逐文件评审 |
| `objgauss/model_manifest.py`、`objgauss/core/chunk_index.py`、`quantization.py`、`ogc_payload.py` | `reference` | 仅在确认浏览器交付需求后评估 manifest/LOD/chunk 思路 |
| 旧 CLI、完整 `objgauss/core`、pipelines/evaluation 双层、viewer 与 compatibility 壳 | `do-not-migrate` | 体量和耦合过高，不整体恢复 |
| 全量旧 tests | `do-not-migrate` | 只随选定基础件提取最小相关行为测试，不恢复旧测试树 |

## 不继承的历史证据

下列均为 `verified_archive_fact`，用于避免重复犯错；它们没有在新分支重新运行，不能写成
新项目能力或当前基线：

- `docs/state/project-status.md`：旧 M2 Model v0 Hungarian mIoU `0.754572`，低于
  3D connected-components `0.825829`；三 seed 最佳 class-semantic proxy 均值
  `0.758923`，hard-case cross-view swap rate 仍为 `0.5`。
- 同一状态源：BOP current ledger 为 `0 pass / 6 fail / 3 blocked`；RBO canonical ledger
  为 `0 pass / 7 fail / 2 blocked`；严格 controlled scene 为 `0/3`。
- NeRF Lego M1 的单物体四色结果只是 wiring smoke，不是实例分割或持久身份证据。
- `docs/state/huggingface-release.md`：4.5M 全量 PLY 的旧本机浏览器结果约 4.4 FPS；1M
  sampled 产物在单台 RTX 5060 Ti 上约 33.8 FPS，均不能外推为新项目或其他硬件 SLA。
- 旧项目有大量 CLI/core/viewer/compatibility 技术债；这支持“不整体迁移”，不支持对尚未
  评审的单个基础模块作质量结论。

## 许可与资产边界

- 归档根 `LICENSE` 是 all rights reserved，未授予复制、修改或分发许可。即使 Owner 相同，
  新项目也要先明确许可证和允许移植的具体文件范围。
- 归档 Git tree 不包含真实训练集或大型 checkpoint；大资产过去位于 ignored `outputs/`
  或外部 Hugging Face，不能把归档树当作数据仓库。
- 第三方数据、模型、checkpoint 和派生产物继续服从各自来源条款；发布、公开 Demo 或商业
  使用前必须再次从官方来源核验。
- 旧记录把 NeRF Lego/HF near-1M 标为 research-use-only、HF `license: other` 和
  development-stage；在新项目中只可作为待核验候选。

## Git

- GitHub：<https://github.com/sessgraph/ObjGauss>
- 本地 remote：`git@github.com:sessgraph/ObjGauss.git`
- 完整归档标签：`archive/objgauss-final-2026-07-14`
- 完整归档提交：`e891bbf`
- 旧版完整 `AGENTS.md`：
  `git show archive/objgauss-final-2026-07-14:AGENTS.md`
- 旧版 Hugging Face 发布记录：
  `git show archive/objgauss-final-2026-07-14:docs/state/huggingface-release.md`

归档标签当前只存在于本地 Git；没有在本次清理中推送远端。

## ObjGauss Hugging Face 仓库

- Dataset：<https://huggingface.co/datasets/jianyong365/objgauss-nerf-lego-near1m>
- Model：<https://huggingface.co/jianyong365/objgauss-nerf-lego-near1m-model>

这两个仓库是旧项目的 development-stage release。其实际文件、校验和与许可边界应以归档
中的 `docs/state/huggingface-release.md` 和远端现状重新核对，不应直接当作新项目资产。

## 旧项目使用过的外部 Hugging Face 来源

- Gaussian splat 样例：<https://huggingface.co/cakewalk/splat-data>
- BOP HOPE：<https://huggingface.co/datasets/bop-benchmark/hope>
- BOP LMO：<https://huggingface.co/datasets/bop-benchmark/lmo>

`cakewalk/splat-data` 在旧项目记录中属于多来源、混合许可素材，只能在重新核验具体文件许可后
使用。BOP 数据也应按各自上游条款重新确认用途。
