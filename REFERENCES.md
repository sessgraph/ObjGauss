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
| `conditional_candidate` | 上游能力已核验，但本地 pilot、预算或许可闭环未通过，不能标为 approved |
| `blocked_pending_budget` | 已有可执行 pilot 建议，但 Owner 尚未批准安装、下载或资源上限 |
| `approved_local_pilot` | Owner 已批准固定边界内的 ignored 本地依赖安装或运行；不等于资源、仿真器或研究假设获批 |
| `verified_local_runtime` | 固定本地环境的安装、import 和设备 probe 已通过；不等于 simulator 语义或研究假设通过 |
| `approved_pr01_primitive_cpu_push_source` | Snapshot/RNG fork 与程序化 CPU external-force action/contact gate 已通过；只批准 PR-01 primitive push source，不代表完整 simulator、formal cohort 已实现或机器人动作成立 |
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
| 获取与验证命令 | `bash scripts/fetch-gaussian-preview.sh`，随后运行 `npm run test:preview`；获取脚本在已有文件复用前与新下载落盘前校验大小和 SHA-256，专项测试再校验记录数和严格解析；clean-checkout `npm run check` 不依赖该 ignored 外部文件 |
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

## PR-00 技术标准候选与决定

以下是 `PR-00` decision packet 的官方一手资料核验。Owner 已选择 JSON Schema Draft 2020-12
作为唯一机器 contract 源，并选择 JavaScript ESM、Node 24 LTS、npm、Ajv 8、esbuild 与
`node:test` 作为 Web-first 工具链；GitHub Actions 与 `npm run check` 是 CI/本地唯一验收入口。
Robotics/OpenCV 坐标约定也已冻结：

| 候选 | 已核验的上游事实 | 当前边界 |
| --- | --- | --- |
| [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12) | JSON Schema 官方 specification 页面将 2020-12 列为当前 released version，并提供固定 meta-schema URI | `decision`：`PR-00` 的唯一机器 contract 源；数组固定使用带 `uri/media_type/dtype/shape/sha256` 的 descriptor，不做 codegen |
| [Node.js Releases](https://nodejs.org/en/about/previous-releases) | Node 官方发布页在 2026-07-14 将 v24 列为 LTS，并建议生产应用只使用 Active LTS 或 Maintenance LTS | `decision`：`.node-version` 与 CI 固定 Node `24.18.0` |
| [Ajv JSON Schema](https://ajv.js.org/json-schema.html) | Ajv 官方文档明确 v8 支持 Draft 2020-12 全部关键词，并要求使用独立的 2020-12 class | `decision`：`package-lock.json` 固定 Ajv `8.20.0`；不得回退到默认 draft-07 instance |
| [esbuild Getting Started](https://esbuild.github.io/getting-started/) | esbuild 官方文档提供 npm 安装、bundle 与 browser target 的构建入口 | `decision`：`package-lock.json` 固定 esbuild `0.28.1`，只负责 browser consumer bundle；不引入应用框架 |
| [GitHub Actions: Building and testing Node.js](https://docs.github.com/en/actions/tutorials/build-and-test-code/nodejs) | GitHub 官方文档推荐用 `setup-node` 固定 Node 版本，并以 `npm ci` 按 lockfile 安装依赖 | `decision`：PR 与 `main` 运行 GitHub Actions；`npm run check` 是本地/CI 唯一验收入口，不包含部署或发布 |
| [Semantic Versioning 2.0.0](https://semver.org/) | SemVer 官方规范说明 `0.y.z` 用于初始开发、已发布版本不可修改，并建议初始版本从 `0.1.0` 开始 | `decision`：schema 从 `0.1.0` 开始；ObjGauss 另行收紧 `0.x` 规则，改变合法实例集合必须升 MINOR，PATCH 不改变 contract |
| [OpenCV Perspective-n-Point](https://docs.opencv.org/master/d5/d1f/calib3d_solvePnP.html) | OpenCV 官方 calib3d 文档明确 camera frame 为 `+X` 右、`+Y` 下、`+Z` 前，并把求得的外参写成 object/world 到 camera 的变换 | `decision`：Camera 采用该轴语义；ObjGauss 同时冻结 `T_AB · p_B = p_A`、World 右手 `+Z` 向上、meter，WebGL bridge 仅属于 Viewer consumer |

上述页面于 2026-07-14 核验。实现时必须锁定实际 validator/version，不能把标准草案名称等同于
某个库已经正确支持全部 vocabulary。

`PR-00` 的 `synthetic-audit-v0` 只由仓库内固定 seed/config producer 生成，不使用本台账中的
任何外部数据或旧归档资源；该选择不表示下游候选来源已完成许可审核。

### PR-00 仓库内 synthetic 资源

| 项 | 当前事实与边界 |
| --- | --- |
| 权威 producer | `src/pr00/synthetic-audit.mjs`；version `0.1.0`，seed `1347563568` |
| 固定 config SHA-256 | `90c4f4a397957af6c0b889fe4bec37d0018a818aaf0d52167740242a4c1668e2` |
| 预期 manifest | `contracts/fixtures/synthetic-audit-v0.manifest.json`；只保存 spec、producer identity 与预期 checksums |
| 派生位置 | `generated/pr00/`；Git ignored，可由 `npm run build` 确定性重建，不得手工编辑或提交 |
| Episode | 3 observations、2 objects、36 primary points；SHA-256 `1c553bf941fe63d4457d0e8965fb667b932b25a2996bb432d39fb8edb3be049e` |
| 数组资源 | 3 RGB、3 depth、6 object masks；共 12 个 descriptor 和 12 个独立 checksum |
| 来源/许可 | 全部由当前仓库 producer 从零生成，不联网、不下载、不读取旧归档；适用新项目当前私有 all-rights-reserved 政策 |
| 允许声明 | `synthetic-audit-v0` contract、Robotics/OpenCV 坐标链、资源 lineage 与独立重投影门得到支持 |
| 禁止声明 | 不支持真实数据、Gaussian 重建、世界模型、动力学、因果或规划价值 |

## RES-001：ManiSkill 3 官方审计（2026-07-14）

当前结论是 **上游能力、隔离 GPU runtime、CPU/no-render snapshot/RNG fork 与程序化 primitive
action/contact outcome 已核验**。ManiSkill 当前为 `approved_pr01_primitive_cpu_push_source`，
可以进入 `PR-01` adapter/writer；这个窄状态不批准 robot controller、render/GPU simulator、
外部 asset 或完整 simulator 声明。

| 审计项 | 官方一手证据 | 当前判断 |
| --- | --- | --- |
| 固定版本 | [GitHub v3.0.1 release](https://github.com/mani-skill/ManiSkill/releases/tag/v3.0.1) 于 2026-04-21 发布，release 指向 commit `a4a4f92`；[PyPI](https://pypi.org/project/mani-skill/) 同日发布 `mani-skill 3.0.1` | `verified_upstream_fact`：RES-001 只评估 `3.0.1`，不使用 nightly 或浮动 `latest` |
| 包与校验和 | PyPI wheel `mani_skill-3.0.1-py3-none-any.whl` 为 `101.7 MB`，SHA-256 为 `685de2f03c300b1ede49881a1bf6306ad062082d39c8d3be8b8e85603f32e33a`；要求 Python `>=3.9` | `sim/uv.lock` 已成为 production resolved lock，SHA-256 为 `69d3b8c1ede6e2839b1b0bfd8cdcee5eb657aa0e60f525cfa4726e359f47a52e`；A-1 ignored freeze 只保留为 pilot provenance |
| 许可 | [官方仓库](https://github.com/mani-skill/ManiSkill) 标为 Apache-2.0，并说明 rigid-body environments 使用 permissive licenses；同时明确 assets 使用 CC BY-NC 4.0，且部分资产/算法有独立来源 | 框架代码和资产必须分账；PR-01 首个证据禁止引入未逐项审查的外部 assets、demonstrations 或 datasets，优先使用程序化 primitive geometry |
| 安装与平台 | [官方安装页](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html) 说明 state-only simulation 不要求渲染依赖；渲染需要 Vulkan，Linux + NVIDIA 同时支持 CPU sim、GPU sim 和 rendering；额外资产默认不随安装下载 | 首个 pilot 分为 state-only 与 render 两门；官方 `download_asset` 源码显示 `MS_SKIP_ASSET_DOWNLOAD_PROMPT=1` 会自动同意下载，因此本项目必须 `unset` 该变量，并用只读 `MS_ASSET_DIR` fail closed |
| Snapshot 内容 | [Advanced Features](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html) 说明 `get_state_dict()` 默认包含注册 actor/articulation 的 pose、velocity、joint state，并要求 custom task 显式扩展未自动登记状态 | 对刚体/关节动态状态有直接 API；PR-01 task-specific state 必须显式加入并往返测试 |
| 恢复 API | [BaseEnv API](https://maniskill.readthedocs.io/en/latest/api/mani_skill/) 提供 `reset(options={"reset_to_env_states": ...})`，可用 dict/flat tensor 恢复并跳过普通 episode initialization；v3.0.1 release 还包含 cached reset state-shape 修复 | 具备实现 sibling fork 的必要接口，但仍需本地验证恢复后的 controller/contact/cache/renderer 状态 |
| RNG 与缺口 | [官方 RNG 文档](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html) 明确 state 不包含 object texture、固定相机 pose、controller stiffness 等；相同 geometry/texture 还依赖同 seed 和 batched episode RNG，且 GPU/CPU reproducibility 有额外复杂性 | `set_state_dict` 不是完整 snapshot；camera、light、material、controller config、physics config、RNG state 和 backend 必须独立冻结并哈希 |
| 初始任务候选 | [PushCube-v1 文档](https://maniskill.readthedocs.io/en/latest/api/mani_skill/envs/tasks/tabletop/push_cube/index.html) 提供 cube push、Panda/Fetch、状态观测和可重写 reset/load hooks | 只作 API smoke/reference；其目标、动作空间和资产不能原样成为 PR-01 primary evidence，正式 cohort 仍需最小自定义任务 |

### 本机只读容量快照

以下是 2026-07-14 的本机命令结果，不是 ManiSkill 上游最低要求：

| 资源 | 已核验本地事实 | 审计边界 |
| --- | --- | --- |
| OS / Python | Linux `6.17.0-35-generic` x86_64；系统 Python `3.12.3`；pilot 使用 ignored uv-managed CPython `3.10.20` | 未修改系统 Python；3.10 是 `toppra` wheel compatibility 选择，不是生产栈 ADR |
| GPU | NVIDIA GeForce RTX 5060 Ti；driver `595.71.05`；总显存 `16,311 MiB`，核验时空闲 `15,174 MiB` | Owner 宿主终端的 Torch CUDA probe 已通过；Agent 沙箱不可见 GPU；CPU snapshot 已实测，GPU simulator/render 仍未实测 |
| RAM | `31 GiB` 总内存，核验时约 `16 GiB` available；另有 `8 GiB` swap | 瞬时快照，不是可长期占用预算 |
| Disk | 当前文件系统约 `656 GiB` available；`data/res001/` 实占 `5,976 MiB`，其中为最终门补充的 ignored Node `24.18.0` 与 npm cache 分别约 `184 MiB` / `51 MiB` | 低于批准的 `10 GB` 上限；不授权自动资产或 demonstration 下载 |
| Vulkan | `nvidia_icd.json` 与 `10_nvidia.json` 存在；官方列为 optional 的 `nvidia_layers.json` 不存在；`vulkaninfo` 命令未安装 | 只能说明 ICD 配置候选存在；渲染门仍为 `not-run` |

### Owner 已批准的 runtime pilot guardrail

Owner 于 2026-07-14 选择 A，随后明确允许使用 GPU，并继续批准程序化 CPU primitive
external-force action/contact gate。该授权不自动批准 rendering、GPU simulator、robot controller、
第三方 asset、demo 或 dataset。

Owner 随后批准 PR-01 production tooling 的最小隔离范围：固定已验证版本并写入独立 `sim`
optional extra，只用于离线 primitive episode 生成；允许 adapter、原子 writer、CLI、真实 smoke、
行为测试和 CI。Runtime 禁止网络和外部 asset，auditor 不依赖 ManiSkill。该批准不改变下述
render 门，也不授权模型、训练、Gaussian dynamics、外部数据或机器人控制。

1. **A — state-only（`approved_local_pilot`）**：隔离环境、固定 ManiSkill `3.0.1`，允许安装
   与本机 driver `595.71.05` / CUDA compatibility `13.2` 匹配的 PyTorch `2.13.0` CUDA 13.0
   wheel；不渲染、不调用 asset/demo downloader，使用程序化 primitive。网络累计上限按 GPU
   runtime 放宽至 `5 GB`、新增磁盘上限 `10 GB`、进程 RSS 上限 `8 GB`、显存上限 `8 GB`、
   墙钟上限 `30 min`。只验证 runtime import/GPU probe，随后验证 get/set/reset、seed 和五个
   sibling 初态 hash。
2. **B — render（仍待动作级批准）**：先验证 Vulkan，再添加固定相机 RGB/depth/mask；不得
   静默下载 ManiSkill assets。任何确需第三方 asset 的路径先回到许可审核。
3. **C — production tooling（`implemented_local_supported_uncommitted`）**：固定 runtime 已成为
   隔离 `sim` extra；clean-venv install、network/asset fail-closed 行为测试和真实 five-branch
   smoke 已通过。远端 runner 仍须由最终 commit SHA 的实际 CI 裁决，不得用本地结果代替。

这些数字是基于本机余量的保守本地 guardrail，不是上游最低要求。RES-001 当前为
`verified_local_runtime`，snapshot/RNG fork 和 programmatic CPU primitive action/contact 两个
窄命题与 PR-01B production runtime 均为本地 `supported`；没有下载任何 dataset、demo 或额外 asset。

### Runtime pilot 尝试账本

| 尝试 | 固定输入 | 结果与证据 | 下一动作 |
| --- | --- | --- | --- |
| `A-0` | 系统 CPython `3.12.3`；`mani-skill==3.0.1`；`torch==2.13.0` / `cu130`；`--only-binary :all:` | `failed_setup`：`mani-skill` 固定依赖 `mplib==0.1.1`，后者需要 `toppra>=0.4.0`；resolver 找不到匹配 `cp312` 的可用 `toppra` wheel。失败发生在安装前，只创建了 ignored `data/res001/venv` | 不允许通过删除 `--only-binary` 静默转为本地源码构建；改用存在匹配 wheel 的 CPython 3.10 |
| `A-1` | uv-managed CPython `3.10.20`，安装在 ignored `data/res001/python/`；其他 direct pins 与 guardrail 不变 | `verified_local_runtime`：Owner 宿主终端确认 ManiSkill `3.0.1`、SAPIEN `3.0.3`、Torch `2.13.0+cu130`、CUDA `13.0` 和 RTX 5060 Ti tensor probe；`installed.txt` SHA-256 `b0aac6dff475d8b1287583da185e17aa9c7b390d30ee535398fd54b618d39e67`；实占 `5,741 MiB`；只读 `no-assets/` 为空 | Snapshot/RNG fork 与 programmatic CPU primitive action/contact 已通过；进入唯一 adapter/writer 切片，不扩大到完整 simulator 声明 |
| `B-1` | uv `0.11.17`；CPython `3.10.20`；`sim/uv.lock`；Linux x86_64；`runtime` extra；外部依赖 `--no-build`；执行期 offline + 空只读 asset gate | `implemented_local_supported_uncommitted`：全新临时 venv 安装 102 个 locked external packages 与本地 `objgauss-sim`；10 tests 通过；canonical/reverse 五分支 evidence 均为 `8a2013f1c8af839ad47f038b6bf3df8306191114cd4e23c1434779c84b571cb0`，reverse 为 `supported`；约 `1.93–1.96 s/process`、峰值约 `787 MiB` | runtime 门保持支持；GitHub clean runner 尚未实际执行，不能记为远端集成完成 |

Agent 在受限沙箱中独立复核了 freeze、版本、磁盘和空资产目录，但沙箱不暴露 GPU，复跑
`torch.cuda.is_available()` 为 false；因此 GPU 成功证据明确归属于 Owner 宿主终端，而不是
Agent 沙箱。该权限差异不覆盖 Owner 已回传的宿主 probe，也不构成 render 或 simulator 证据。

### Snapshot/reset pilot 预注册（运行前冻结）

**唯一 primary endpoint**：固定 seed `24071401` 生成一个仅含两个程序化 dynamic box 的
`physx_cpu` / `render_backend=none` snapshot。分别为 `hold`、`push_pos_x_weak`、
`push_pos_x_strong`、`push_neg_x_weak`、`push_pos_y_weak` 恢复同一 snapshot，在任何 action 或
simulation step 之前，对“ManiSkill physical state + main/episode RNG state”做规范化 SHA-256；
五个 hash 必须与 source snapshot 完全相同。任何一个不同即 `rejected`，API 异常、额外资产、
非有限值或环境缺失即 `invalid`/`blocked`，不得用容差或 secondary check 替代。

固定输入与必要 secondary checks：

- ManiSkill `3.0.1`、SAPIEN `3.0.3`、CPython `3.10.20`；单环境、`physx_cpu`、无 agent、
  `obs_mode=none`、`reward_mode=none`、`render_backend=none`；两个 box 只含程序化 collision shape。
- 同 seed 的两次正常 reset 必须产生相同 full snapshot hash；不同 seed 必须产生不同 hash，
  作为 seed sensitivity 负对照。
- 对物理 state 做固定扰动后 hash 必须变化；随后 `set_state_dict` 必须精确恢复 physical hash。
- `reset(options={"reset_to_env_states": ...})` 必须精确恢复 physical hash。由于上游 state 明确
  不含 RNG，pilot 必须同时显示 state-only restore 的 RNG gap，并验证显式恢复 ManiSkill-owned
  main/episode RNG 后 full snapshot hash 精确回到 source。
- `MS_SKIP_ASSET_DOWNLOAD_PROMPT` 必须未设置；`MS_ASSET_DIR` 指向只读 ignored 空目录，运行前后
  都不得出现文件；RSS 上限 `8 GB`、墙钟上限 `15 min`。

规范化只接受排序后的 dict/list、有限数值、显式 dtype/shape 和 little-endian tensor bytes；
报告保存到 ignored `data/res001/evidence/`。两次独立进程运行的稳定 evidence hash 必须相同，
runtime telemetry 不进入稳定 hash。该门只裁决 snapshot fork feasibility，不执行 action、不验证
contact dynamics，也不把 ManiSkill 提升为 PR-01 approved simulator。

**运行结果：`supported`（仅 snapshot fork feasibility）**

- 权威脚本为 `scripts/res001_snapshot_pilot.py`；spec SHA-256 为
  `907d872c3fe2304509fe04596026430fc89da09cab99efb3546cc5c39bfb4696`。两个独立进程的稳定
  evidence SHA-256 均为 `1affc32d51ce176712b831ede8b98db8fa82dc72e205432b968316118254e80b`。
- Source physical/RNG/full hash 分别为
  `77b2518837ca4b6427387696af38d1004d019e273f36142587863e6f7dfce658`、
  `03ce0308fa8ae7a638e688f4c4c56bd000d2729f00c71174dce8b958e307843f`、
  `d2b78e33073008f7a855ecc2841ae3ff751bfe3b78e859696da9e914befc2b5f`；五个 sibling 的三个
  hash 均逐项等于 source。
- 同 seed repeat、不同 seed 负对照、固定 state mutation 负对照、`set_state_dict` physical
  roundtrip 和 `reset_to_env_states` physical roundtrip 全部通过。
- 仅调用 `reset_to_env_states` 时 physical hash 已恢复，但 RNG/full hash 分别变为
  `11a2e76fa8cced431f0e35944ddabfb24e9e3b86ee3c50ce38c0f2ea6c24e5f4` /
  `166b157f79c03c4c5580742a0811174fbed136476be338f931ac5bd90364e1e2`；
  显式恢复 ManiSkill main/episode RNG 后 full hash 才回到 source。由此确认上游 state 不是完整
  sibling snapshot，PR-01 adapter 必须拥有并版本化 RNG capture/restore。
- 两次运行墙钟分别约 `0.55 s` / `0.26 s`，最大 RSS `786,564 KiB` / `788,636 KiB`，远低于
  预注册预算；backend 为 `physx_cpu` / `render_backend=none`，无 agent、sensor 或 camera，
  `no-assets/` 运行前后保持 mode `555` 且为空。沙箱中的 NVML warning 不影响 CPU pilot，
  也不构成 GPU simulator 或 render 证据。

因此 RES-001 的 snapshot fork 子命题为 `supported`，可以进入下一节的 PR-01
action/contact pilot；这一段本身不包含 action outcome 证据，当前综合状态以下一节结果为准。

### PR-01 sibling action/contact pilot 预注册（运行前冻结）

**唯一 primary endpoint**：在 snapshot pilot 已支持的 fixed runtime 上，固定 seed `24071401`
和同一份“physical state + ManiSkill main/episode RNG”source snapshot，按不同进程中的 canonical
与 reverse 顺序执行五个 sibling。对每个 branch，干预前 full snapshot hash、executed-action
ledger、最终 physical-state hash 和逐 simulation-step 的 target-related contact trace hash 必须
跨进程逐项完全相同；同时，四个非 hold branch 相对 hold 必须产生与声明方向一致且达到冻结
阈值的 target 位移。任一条件不满足即 `rejected`；API 异常、非有限值、版本/资产门漂移或
缺少任一 branch 即 `invalid`，不得用 renderer、视觉相似或 secondary metric 替代。

固定场景、动作与时间语义：

- Pilot ID 为 `pr01-maniskill-sibling-action-v0`；ManiSkill `3.0.1`、SAPIEN `3.0.3`、
  CPython `3.10.20`、Torch `2.13.0+cu130`；单环境、`physx_cpu`、100 Hz、无 agent/sensor/
  camera/render 和外部 asset。
- 场景仅含程序化 collision box：顶部为 `z=0` 的 static floor、target 和远离干预路径的
  context；三者 static/dynamic friction 均为 `0.5`、restitution 为 `0.0`。Target half-size 为
  `[0.03, 0.02, 0.025] m`、density 为 `500 kg/m^3`；reset 后先执行 20 个无动作 warm-up step，
  再捕获唯一 source snapshot。
- 五个 branch 为 `hold`、`push_pos_x_weak`、`push_pos_x_strong`、`push_neg_x_weak`、
  `push_pos_y_weak`。除 hold 外，分别在 target 质心连续 10 个 simulation step 施加
  `[+0.35, 0, 0]`、`[+0.70, 0, 0]`、`[-0.35, 0, 0]`、`[0, +0.35, 0] N` 的 external force；
  hold 使用 `[0, 0, 0] N` 和相同 10-step duration。随后所有 branch 都执行固定 100-step
  no-force settling；不以睡眠状态提前终止。
- `commanded_action` 与 `executed_action` 分账。后者记录实际 force、applied step count、action/
  settling duration 和 sim frequency；必须与冻结命令逐字段相同。该 pilot 只核验程序化
  external-force intervention，不把它描述为 robot controller action 或真实机器人 measured action。

冻结的 branch-level checks：

- 每个 branch 的 pre-action physical/RNG/full hash 必须等于 source；target 初态 position、
  quaternion、linear/angular velocity 也必须逐值相同。
- Hold 从 source 到最终的水平漂移不得超过 `0.002 m`。`+x weak`、`-x weak`、`+y weak`
  相对 hold 在各自主轴上的有符号位移必须至少为 `0.005 m`；`+x strong` 相对 hold 的 `+x`
  位移必须至少为 `0.005 m`，且至少比 `+x weak` 多 `0.005 m`。方向只由 target 最终位置与
  hold 的 paired difference 裁决，不用绝对世界位置替代。
- 每个 branch 在 action + settling 的 110 steps 内必须观察到至少一个 target-floor contact
  point；target-context contact 必须为零。逐 step trace 对 body pair、point position/normal/
  impulse/separation 做确定性排序后哈希，summary 同时记录 contact step count、contact count、
  point count 和 impulse-norm sum。
- 固定 settling 结束时 target linear speed 不得超过 `0.01 m/s`，angular speed 不得超过
  `0.01 rad/s`。这只证明本 primitive cohort 达到冻结终态，不外推到其他材质或任务。
- Evaluator 负对照必须拒绝符号反转的 `+x weak` paired displacement，并拒绝 applied step
  count 少一的 executed-action ledger。两次进程的 branch 执行顺序不进入稳定 evidence hash，
  其他 spec、outcome、checks 与 claim boundary 全部进入。

沿用 snapshot pilot 的空只读 `MS_ASSET_DIR`、未设置 `MS_SKIP_ASSET_DOWNLOAD_PROMPT`、
`8 GB` RSS 和 `15 min` 墙钟 guardrail；报告写入 ignored `data/res001/evidence/`。该门通过也只把
ManiSkill 提升为“programmatic CPU sibling source approved for PR-01’s primitive push slice”；
renderer、robot controller、GPU simulator、外部资产、正式 cohort、dataset、训练、
Gaussian dynamics、因果或规划声明仍不在证据范围。

**运行结果：`supported`（仅 programmatic CPU primitive push sibling source）**

- 权威脚本为 `scripts/pr01_sibling_action_pilot.py`；spec SHA-256 为
  `faa5014d7a05801fc507be9b5d309b601e4986d7b80e812e6b0da33da7c9ae2b`。Canonical 与 reverse
  两个独立进程的稳定 evidence SHA-256 均为
  `3c2c8a7dec7d7626d158907b6fc7981a89e27645b342a8116c4a8ac1094616f2`，并确认两份报告的
  branch execution order 恰好相反。稳定证据还固定了 action pilot 源码 SHA-256
  `39f3f8f7085aef032b98062aa3dc632f874ac4fe88150c97f84c391b1382ba04` 与 snapshot helper
  SHA-256 `8f414140f2e36073f40a2fec3650a476ab7dc086fca1d1cec3c3ae2c7cdd7f25`。
- Source physical/RNG/full hash 分别为
  `718fc8aadbf62e15579e601a19765d2d6c4ea59ef1a34c8c661310d104df5606`、
  `244cb61c0fcc89370a3264fbc50cf14bface31ffa9b683a4f5764a7bb33135d3`、
  `7112a99c777455b44593f22dfaa7f3c0f0a5ffce0cbc845f7e65a31972cb8be4`；五个 branch 的
  pre-action full hash 和 target state 均逐项等于 source。
- 相对 paired hold，`+x weak`、`+x strong`、`-x weak`、`+y weak` 的主轴位移分别约为
  `+0.005741 m`、`+0.080497 m`、`-0.005555 m`、`+0.005927 m`；hold 水平漂移约
  `7.83e-7 m`。四个 push 全部越过运行前冻结的 `0.005 m` 有符号效应门，`+x strong` 也超过
  `+x weak` 至少 `0.005 m`。
- 每个 branch 的 110 个 action + settling step 都观察到 target-floor contact，共 440 个
  contact points；target-context contact 为零。固定 settling 后五个 target 的 linear/angular
  speed 都为零。Final physical-state 与逐 step contact-trace hash 在两个进程中逐 branch 完全
  一致；commanded/executed external-force ledger 也逐字段一致。
- Evaluator 内置的 force-sign flip 与 missing applied-step 负对照均被拒绝。外层错误
  `MS_SKIP_ASSET_DOWNLOAD_PROMPT` 环境得到 `invalid`；canonical 与 canonical 的同序比较不能
  冒充相反执行顺序，得到 `rejected`。
- 两个有效进程墙钟约 `0.345 s` / `0.340 s`，最大 RSS `790,244 KiB` / `790,400 KiB`；空只读
  `no-assets/` 前后不变。沙箱 NVML warning 不影响该 CPU/no-render pilot，也不构成 GPU
  simulator 或 render 证据。

据此，ManiSkill 状态提升为 `approved_pr01_primitive_cpu_push_source`，可以继续实现 PR-01 的
唯一 adapter/writer 切片；仍不能把它概括为完整 approved simulator，也不能把 external-force
intervention 写成 robot controller 或 measured real action。

## 新项目候选数据源

以下入口来自 2026-07-14 Owner 研究材料。ManiSkill 已完成上节所述官方审计、固定 runtime、
CPU/no-render snapshot fork 和 programmatic primitive action/contact pilot；其他来源当前仍为
`owner_brief_candidate + requires_upstream_review`。Stage-0
`.splat` 预览不是其中任一原始数据集的已接入证明。表中的“候选职责”是 PRD 的规划输入，
不是对数据字段或研究适用性的已验证陈述。

| 候选 | Owner 材料中的入口 | 候选职责 | 不能直接支持的声明 |
| --- | --- | --- | --- |
| Kubric / MOVi | <https://github.com/google-research/kubric/blob/main/challenges/movi/README.md> | 坐标、实例、物性和 Oracle Gaussian 基础 | 真实动作或 sim-to-real |
| HO-Cap | <https://irvlutd.github.io/HOCap/> | 真实多视角 ObjectBelief 与 pose 外测 | 严格动作反事实 |
| HOT3D | <https://facebookresearch.github.io/hot3d/> | 遮挡、重识别和头戴视角压力测试 | measured action 因果 |
| ManiSkill 3 | `3.0.1`；官方审计见上节 | `approved_pr01_primitive_cpu_push_source`：snapshot/RNG fork、程序化 external-force siblings 与 contact/settling | robot controller、render/GPU、正式 cohort、跨 backend determinism 或真实机器人价值 |
| CausalWorld | <https://causal-world.readthedocs.io/en/latest/> | 显式物性干预与 OOD 组合 | sim-to-real |
| Physion++ | <https://dingmyu.github.io/physion_v2/> | 隐藏物理外部验证 | 机器人动作执行价值 |
| RH20T | <https://rh20t.github.io/> | 真实接触、力和执行偏差校准候选 | 严格同起点反事实，除非另行构造 |
| DROID | <https://droid-dataset.github.io/> | 后期真实视觉—动作预训练候选 | 核心因果真值 |
| CALVIN | <https://github.com/mees/calvin> | 后期长时序 rollout/规划候选 | 真实机器人最终价值 |

BridgeData、RoboNet、Open X-Embodiment 暂只记录为后期预训练候选。除上述 ManiSkill
审计范围外，`RES-001` 仍须使用上游一手资料补齐每个候选的版本、许可、字段矩阵、下载范围、
校验和和存储预算；在此之前不得把附件中的规模、字段或许可证描述提升为 `confirmed_fact`。

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

- Owner 已决定新项目当前保持私有并按 all rights reserved 管理；对外发布前必须重新做
  许可证决策。当前决定不改变任何第三方资产条款。
- 归档根 `LICENSE` 也是 all rights reserved，未授予复制、修改或分发许可。它与新项目是
  独立授权边界；`PR-00` 已明确为全新实现、零归档文件移植。未来即使 Owner 相同，仍须对
  每个候选文件另行授权并记录许可审查。
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
