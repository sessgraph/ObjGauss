# ObjGauss

当前工作区已完成 Demo A 的两个前置切片：Stage-0 在浏览器中直接渲染环境级 3D Gaussian；
`PR-00` 建立状态、干预和坐标的唯一机器 contract，并在固定 synthetic episode 上给出可复现
裁决。后续 `RES-001` 在固定 ManiSkill 3.0.1 runtime 上支持了无渲染 snapshot fork，并由
`PR-01` 首个 action/contact gate 支持了程序化 CPU primitive push sibling source；完整
里程碑现按 `PR-01A`–`PR-01F` 推进。`0.2.0` sibling evidence contract、隔离 production
runtime、原子 adapter/writer、independent audit 与冻结 cohort 均已支持；无 RGB Delivery、
checksums 与一键命令也已实现，并会在 dirty worktree 上 fail closed。代码承载验收 SHA
`234ba00` 已由 `./scripts/accept-pr01` 完整重建为 `supported`，且该 SHA 的 PR-00、runtime、
writer、independent audit、frozen cohort 与 acceptance delivery 六项远端 Actions 全部成功，
因此 PR-01 严格 sibling evidence 里程碑已关闭。robot controller、RGB/GPU renderer 和训练仍未
实现。项目保持
私有并按 all rights reserved 管理，对外发布前必须重新决策许可证。

`PR-00` 当前本地机器结果为 `supported`：JSON Schema Draft 2020-12 contract、资源 checksum、
lineage 和 14 类预注册反例全部通过，36 个 primary points 的独立最大重投影误差为
`1.005e-14 px`，严格低于 `< 1.0 px` 门。这个结果只支持 `synthetic-audit-v0` 的 contract、
Robotics/OpenCV 坐标链与独立重投影门，不支持真实数据、Gaussian 重建、世界模型、动力学或
规划价值声明。

## 运行 PR-00 Web 证据页

使用 [`.node-version`](.node-version) 固定的 Node `24.18.0`，从仓库根目录运行：

```bash
npm ci
npm run check
python3 -m http.server 8000 --bind 127.0.0.1
```

然后打开 <http://127.0.0.1:8000/viewer/?mode=contract>。页面只消费 `npm run check` 生成在
ignored `generated/pr00/` 中的 episode、资源与机器报告；浏览器再次验证 schema、manifest、
episode、全部资源和 report 后，才同步显示 RGB、对象、相机、坐标轴、轨迹和窄声明账本。

唯一机器 schema 是
[`contracts/objgauss/0.1.0/episode.schema.json`](contracts/objgauss/0.1.0/episode.schema.json)，
工具链、坐标约定、版本和回滚决策见
[`docs/adr/0002-pr00-contract-stack.md`](docs/adr/0002-pr00-contract-stack.md)。

## 查看 Stage-0 Gaussian 世界

打开 <http://127.0.0.1:8000/viewer/>。页面默认在浏览器内确定性生成 `8,523` 个严格 splat
records，组成可环绕、平移、移动和缩放的环境级 Gaussian scene。可选的 `103,060`-splat Lego
审计样例需先运行：

```bash
bash scripts/fetch-gaussian-preview.sh
npm run test:preview
```

获取脚本先校验固定大小和 SHA-256；专项测试再校验记录数与严格解析。该测试显式依赖 ignored
外部文件，不属于必须在无外部资产 clean checkout 中成立的 `npm run check`。该文件只保存到
ignored `data/`，其 asset provenance 仍为 `unverified`。Stage-0 只证明
本地 WebGL2 Gaussian 渲染链可见，不是 `PR-00` episode，也不构成任何模型或研究门证据。
准确资源与许可边界见 [`REFERENCES.md`](REFERENCES.md)，局部渲染决策见
[`docs/adr/0001-stage-0-preview-stack.md`](docs/adr/0001-stage-0-preview-stack.md)。

## 复现 RES-001 snapshot pilot

完成 `REFERENCES.md` 登记的 ignored CPython 3.10 runtime 安装后，从仓库根目录运行：

```bash
env -u MS_SKIP_ASSET_DOWNLOAD_PROMPT \
  MS_ASSET_DIR="$PWD/data/res001/no-assets" \
  XDG_CACHE_HOME="$PWD/data/res001/xdg-cache" \
  MPLCONFIGDIR="$PWD/data/res001/mpl-cache" \
  data/res001/venv-py310/bin/python scripts/res001_snapshot_pilot.py
```

脚本只构建两个程序化 collision box，使用 `physx_cpu` 并关闭 renderer。当前稳定 evidence hash
为 `1affc32d51ce176712b831ede8b98db8fa82dc72e205432b968316118254e80b`；它支持 get/set/reset、
seed、显式 RNG 恢复和五 sibling 初态 hash，不支持 action、contact、render、dynamics 或规划声明。
准确预注册、负结果和资源边界只在 [`REFERENCES.md`](REFERENCES.md) 维护。

## 复现 PR-01 primitive sibling action/contact gate

先运行 canonical baseline；它通过本进程 checks 后会停在 `pending_repeat`，不会提前给出跨进程
裁决：

```bash
env -u MS_SKIP_ASSET_DOWNLOAD_PROMPT \
  MS_ASSET_DIR="$PWD/data/res001/no-assets" \
  XDG_CACHE_HOME="$PWD/data/res001/xdg-cache" \
  MPLCONFIGDIR="$PWD/data/res001/mpl-cache" \
  data/res001/venv-py310/bin/python scripts/pr01_sibling_action_pilot.py \
  --order canonical \
  --output data/res001/evidence/sibling-action-pilot-run1.json
```

再用相反 branch 顺序运行独立进程并比较：

```bash
env -u MS_SKIP_ASSET_DOWNLOAD_PROMPT \
  MS_ASSET_DIR="$PWD/data/res001/no-assets" \
  XDG_CACHE_HOME="$PWD/data/res001/xdg-cache" \
  MPLCONFIGDIR="$PWD/data/res001/mpl-cache" \
  data/res001/venv-py310/bin/python scripts/pr01_sibling_action_pilot.py \
  --order reverse \
  --compare data/res001/evidence/sibling-action-pilot-run1.json \
  --output data/res001/evidence/sibling-action-pilot-run2.json
```

当前第二进程 verdict 为 `supported`，稳定 evidence hash 为
`3c2c8a7dec7d7626d158907b6fc7981a89e27645b342a8116c4a8ac1094616f2`。该结论只批准
程序化 CPU external-force primitive push source，不能描述成完整 PR-01、机器人 action、
render/GPU 或 dynamics 能力。

## 验证 PR-01A Contract 0.2.0

`0.1.0` 保持字节冻结；`0.2.0` 使用 episode、experiment、attempt、invariance-report 四份
Schema，并按精确 `schema_version + contract_kind` 分派：

```bash
npm run contract:pr01a
node --test tests/pr01a-contracts.test.mjs
npm run check
```

正负 fixtures 和 checksum manifest 位于
[`contracts/fixtures/pr01a/`](contracts/fixtures/pr01a/)，架构、隔离 runtime 和无 RGB Demo
边界见 [`ADR-003`](docs/adr/0003-pr01-sibling-evidence-stack.md)。这一步只建立可审计记录语义，
不表示 writer、正式 cohort 或完整 PR-01 已通过。

当前 PR-01A machine report 为 `supported`，SHA-256 为
`f250ef05a82d0f0cb90b0a1ba6dcb0248cecac8beb0df454035485d4d9dd64e1`；该报告位于 ignored
`generated/pr01a/`，可由上述命令重建。

## 验证 PR-01B 隔离 Runtime

PR-01B 使用 uv `0.11.17`、CPython `3.10.20` 和 [`sim/uv.lock`](sim/uv.lock) 的精确解析，
ManiSkill/SAPIEN/Torch 只存在于 `sim` package 的 `runtime` optional extra。以下命令始终创建
新的临时 venv，先用 wheel-only 门安装外部依赖，再构建本地 package；真正执行 simulator 前
切换为 offline、空只读 asset 目录，并运行 canonical/reverse 两个独立进程：

```bash
./scripts/check-pr01b-runtime
```

当前本地门的 10 个行为测试通过，两个进程的稳定 evidence SHA-256 均为
`8a2013f1c8af839ad47f038b6bf3df8306191114cd4e23c1434779c84b571cb0`，reverse verdict 为
`supported`。报告位于 ignored `generated/pr01b/`；本结果只支持可复现、无渲染、无外部资产的
`physx_cpu` 五分支 runtime，不表示 writer、独立 audit、正式 cohort 或完整 PR-01 已完成。
代码承载验收 SHA `234ba00` 的 GitHub runtime workflow 已成功；该远端证据不扩大本切片声明。

## 验证 PR-01C 原子 Writer

PR-01C 把真实五分支 simulator 输出映射为 `0.2.0` episode/attempt，并通过临时目录、
`fsync`、校验与单次 rename 发布不可变 branch；同一逻辑 key 与同一语义内容为 no-op，内容
冲突或中途失败不会覆盖已发布 episode。以下门从全新临时 venv 构建 package，运行行为/负例
测试，再用两个独立真实进程生成 canonical/reverse golden group：

```bash
./scripts/check-pr01c-writer
```

当前本地门的 22 个 Python 测试通过；五个 branch 各含 111 条 trajectory records 与 110 条 contact
records；canonical/reverse 在同一 clean source 下得到相同 evidence SHA-256。该 digest 纳入
source commit/tree lineage，因此不作为跨提交常量。这只支持单一 golden group 的 adapter、原子
幂等 writer 与 attempt failure semantics；独立 auditor、正式 cohort 与 Delivery 的结论由后续
切片各自提供。代码承载验收 SHA `234ba00` 的远端 writer job 已成功。

## 验证 PR-01D 独立审计

PR-01D evaluator 不导入 simulator、adapter 或 writer；它从 raw episode、trajectory、contact、
attempt 与 publication 重新计算 14 个 hard gates，并以固定 `0/1/2/3/4` 退出码区分
`supported/internal error/rejected/blocked/invalid`。以下命令会先重新生成真实 golden group，
再运行完整 mutation matrix：

```bash
./scripts/check-pr01d-audit
```

当前 14 个 baseline gates 与 11 个 mutation cases 全部通过。Machine report 含运行生成的
attempt/index hashes，因此不把单次运行 SHA 伪装成跨运行常量。本结果只支持这个
golden fixture 的证据可独立审计；该段本身不替代后续 cohort、最终 lineage 或完整 PR-01。
代码承载验收 SHA `234ba00` 的远端 independent audit job 已成功。

## 验证 PR-01E Preflight 与正式 Cohort

PR-01E 先用隔离 reserved seeds 生成 12 groups / 60 episodes；provisional `0.005 m` effect
threshold 正确拒绝了 `box-b` weak push，失败证据被保留且没有重试、换 seed 或删 group。
冻结为 `0.0014 m` 后完整 preflight 通过，并把正式预算冻结为 300 秒、50 MiB、8 GiB 与最多
12 个额外 attempt。以下命令先复跑 preflight，再生成正式 48 groups / 240 episodes：

```bash
./scripts/check-pr01e-cohort
```

当前正式结果为 24/12/12 train/validation/test groups、0 failed/extra attempts，wall time
`126.24 s`、artifacts `41,403,093 bytes`、RSS `808,157,184 bytes`；独立 audit verdict 为
`supported`。Active spec SHA-256 为 `d1dff647…b339ef`，manifest SHA-256 为
`c0dc325c…dc7ef`；阈值测量时的 `fdab5f78…c8623` spec 作为历史 fixture 保留，机器测试证明两者
只在 source-commit provenance 策略上不同。数据与运行输出只在 ignored `generated/pr01e/`，
不进入 Git；代码承载验收 SHA `234ba00` 的 clean 验收与远端 frozen cohort job 均已
`supported`，运行证据绑定该 HEAD。

## 完成 PR-01F Delivery 验收

正式入口要求 Git worktree 完全干净、Node `24.18.0` 与 uv `0.11.17`；任何 tracked、staged 或
非 ignored 的 untracked 改动都会以 `invalid` 退出，避免把旧 HEAD 冒充最终 source commit：

```bash
./scripts/accept-pr01
```

它会从锁定依赖开始，依次运行全库 Node 门、PR-01B 独立 runtime/smoke、真实 Writer/Audit、
12/60 preflight、48/240 formal cohort、独立审计、Delivery build 与 checksum verifier。成功后
输出 ignored `artifacts/pr01/`：
正式 dataset 的每个 branch 内含 episode、trajectory、contact、成功 attempt 和 publication；
独立 `attempts/` 只保存没有最终 episode 的失败尝试；另含 experiment manifest、机器/人类报告、
checksum index 与五联无 RGB Demo。

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

随后打开 <http://127.0.0.1:8000/artifacts/pr01/demo/>。当前五联页面与 fail-closed checksum/audit
行为已通过本地测试和浏览器目视核对。代码承载验收 SHA `234ba00` 的 clean 一键验收得到
48 groups / 240 episodes、24/12/12 split、0 failed/extra attempts、独立 audit `supported`，并生成
含 1210 个条目的 checksum index；Delivery 的 source commit 与该 HEAD 一致。该 SHA 的六项
GitHub Actions 全部成功，PR-01F 与完整 PR-01 远端门已关闭；任何后续 `main` HEAD 若未保持
这些门成功，状态自动重开。

## 验证 PR-02A Contract 0.3.0

`0.1.0` 与 `0.2.0` 保持字节冻结；`0.3.0` 使用 dynamics experiment、training trial/attempt、
checkpoint manifest、raw dynamics prediction 和 independent evaluation report 六种可分派记录，
并通过 shared schema 复用严格 identifier、artifact、availability、provenance 与 verdict 定义：

```bash
npm run contract:pr02a
node --test tests/pr02a-contracts.test.mjs
npm run check
```

当前本地 machine report 为 `supported`，SHA-256 为
`3b1e64a02120da2ca5ef5616af08f8ee068498139dfdb54b4e471045acccca3f`：5 个旧 contract 文件哈希
保持冻结，7 个 `0.3.0` schema 文件、6 个正向 fixtures 和 39 个负例全部通过。报告位于 ignored
`generated/pr02a/contract-report.json`，可由上述命令重建。本结果只支持 PR-02A contract 的
表达力、精确版本分派和 fail-closed 语义；不表示 pilot、cohort、trainer、checkpoint、模型指标、
Gaussian dynamics 或机器人控制已经实现。

## 验证 PR-02B Pilot/Data Freeze

PR-02B 使用与 PR-01 及正式 PR-02 cohort 都隔离的两组对象、两组 layout 和三个 reserved
reset seeds，分别以 canonical/reverse branch 顺序生成 `12 groups / 60 episodes`。两遍都先由
PR-01D 独立 auditor 重算 source gates，随后校准器才冻结物理时间 horizon、四类 ObjectState
归一化尺度、`δ`、`δ_shuffle`、正式 split/group 数、training seeds、小型 Object GNN grid 和
资源调度。唯一完整入口要求 clean checkout，并始终离线运行：

```bash
./scripts/check-pr02b-pilot
```

该命令要求 Node `24.18.0`、uv `0.11.17` 和冻结的 `sim/uv.lock`，会先跑全库 Node/Python 回归，
再创建临时 venv、执行两遍真实 simulator source、两份独立 audit、16 MiB CUDA 探针和
`0.3.0` dynamics experiment verifier。GPU 探针必须在 12 GiB 单进程上限之外，为桌面显示保留
至少 1 GiB 实际可用显存。产物只写入 ignored `generated/pr02b/evidence/`，不得提交。

排障运行发现并保留两条负证据：首版过重 calibration object 的 weak
push 未过既有 `0.0014 m` source 门；首版 power analyzer 又把物理 effect 的绝对方差直接当作
error-reduction 方差，因量纲错误被判为 `blocked`。修正对象支持范围和相对异质性代理后，
代码承载 SHA `04ddb18` 的 clean acceptance 中，canonical/reverse 两份 source audit 与 freeze
verifier 均通过；权威冻结值为 horizon
`1.1 s`、评分点 `[0.1, 0.2, 0.5, 1.1] s`、正式 `48/12/12` groups、3 个 training seeds、
`δ=0.1`、`δ_shuffle=0.06`，基础调度 `6 + 4 = 10 GPU-hours`，含 5% 技术重试保留后为
`6.3 + 4.2 = 10.5 GPU-hours`。宿主 16 MiB probe 后仍有 15.64 GB 可用显存并保留 1 GiB。
两遍各生成 12 groups / 60 episodes、0 failed/extra attempts，21 项 verification 与总 checksum
index 全部通过；pilot report SHA-256 为 `47ad53c6…944cc`，ignored evidence 位于
`generated/pr02b/evidence/`。

PR-02B 因此为本地 `supported`，但尚无远端 CI。本切片不创建 `learning/`、不训练模型，也不
支持任何模型性能、Gaussian dynamics、外部数据泛化或机器人控制声明。Owner 已另行授权
PR-02C 进入规划。Owner 已选择延迟物化 final test：PR-02C 只生成 train/validation 60 groups，
12 个 test groups 在 PR-02E 前只保留冻结 spec；rollout 已冻结为四个评分区间共享、显式 `Δt`
与 commanded-action schedule 的 residual transition；HPO config 按 validation groups 和 3 seeds
两级等权平均选择。ADR-006 已 accepted；PR-02C C0 独立 runtime/contract gate 已实现，仍未
生成 formal cohort、实现模型/trainer 或运行训练。
长期取舍见
[`ADR-005`](docs/adr/0005-pr02b-pilot-data-freeze.md)。

PR-02C 的实施与验收边界见
[`ADR-006`](docs/adr/0006-pr02c-trainer-baselines.md)。它只负责独立纯 PyTorch runtime、四个
预注册 arms、training lineage 和 reproducibility gate；独立科学 evaluator 与 final verdict
仍属于 PR-02D/PR-02E。

### 验证 PR-02C C0 Runtime

```bash
./scripts/check-pr02c-runtime
```

该门只在 clean checkout 上运行：使用 `learning/uv.lock` 创建全新纯 PyTorch venv，要求
CPython `3.10.20`、`torch==2.13.0+cu130`、CUDA `13.0`、离线运行且不存在 ManiSkill、SAPIEN
或 `objgauss-sim`，并独立复核 HEAD、package tree、lock、PR-02B grid、12 GiB cap 和至少
1 GiB 的桌面显示显存保留。成功证据原子写入 ignored `generated/pr02c/runtime/`。

当前状态是 `c0_committed_local_supported`：代码提交 `fc20023` 的 clean gate 在 Node
`24.18.0` 下通过 77 项全库测试、12 个 Python 行为/失败测试和 14 项独立 verification checks；
真实 RTX 5060 Ti probe 保留超过 1 GiB 空闲显存，并把训练 allocation cap 限为 12 GiB。C0 只
支持“独立纯 PyTorch runtime 可用”这一窄声明，不支持 trainer、模型性能、科学比较或 Gaussian
dynamics 价值声明；当前尚无远端 CI。

### 验证 PR-02C C1 Data Boundary

```bash
./scripts/check-pr02c-data
```

该 clean gate 先重建 PR-02B freeze 并复跑 C0，然后只从冻结 spec 物化 48 train + 12 validation
sibling groups。`sim/` producer 生成 300 个 `0.2.0` episodes；独立 `learning/` loader 重算
publication/descriptor checksums 和 lineage，只把初态、commanded action、target 与四个物理 rollout
times 暴露为 model inputs，future ObjectState 只进入独立 labels。Node verifier 不导入 producer 或
loader 实现，并独立重算 contract、split、final isolation 与同一 data index。锁定且校验哈希的
package install 是唯一可联网阶段；执行 producer、loader 与所有验收代码前会切换为 offline。

当前状态是 `c1_committed_local_supported`。代码承载 SHA `adb1a62` 的 clean gate 生成 60 groups /
300 branches、0 failed attempts；producer、loader 与 16 项独立 checks 对同一 data index
`2501ebc2…17a81b5` 均为 `supported`。C1 没有物化 12 个 test groups，也不支持 baseline、trainer、
模型性能或科学结论。

### 验证 PR-02C C2 Deterministic Baselines

```bash
./scripts/check-pr02c-baselines
```

该 clean gate 先完整复跑 C1，再把 loader 输出投影成只含 validation 初态、commanded action
schedule 与非未来 metadata 的 sanitized bundle。独立 baseline 进程不读取 source trajectory 或
future GT，离线生成 copy-state 和 constant-velocity 两个确定性 arms 的 120 份 `0.3.0`
prediction artifacts。Node verifier 不导入 Python producer，独立重算 source projection、public
contract、payload/artifact checksum、两种 baseline 数学、lineage、final/future isolation、sibling
初态不变性与 canonical/reverse repeat，并必须拒绝被篡改的 prediction；成功证据原子发布到 ignored
`generated/pr02c/baselines/`。

当前状态是 `c2_committed_local_supported`。代码承载 SHA `9ea2b92` 的 clean gate 重建 C1 的
60 groups / 300 branches / 0 failures，并对 data index `970b9359…2e745` 完成 16 项检查；随后在
12 validation groups / 60 branches 上发布 120 份 deterministic predictions，18 项独立 checks
全部通过，canonical/reverse semantic index 均为 `17488a15…7c647`。篡改单份 prediction 后
verifier 以 exit 4 拒绝，并同时标记 contract、payload checksum、constant-velocity 数学和
reverse repeat 失败。当前没有远端 CI；C2 本身不支持 learned model、HPO、checkpoint、test
prediction 或科学比较。

### 验证 PR-02C C3 Minimal Object GNN/Trainer

```bash
./scripts/check-pr02c-trainer
```

C3 在纯 PyTorch `learning/` 中实现共享两层 object encoder、两层 pairwise message MLP、一次
mean aggregation 与两层 residual head。Action-conditioned 只向 target object 注入裁剪到当前
区间的力、目标对象坐标系 COM 作用点、active duration/fraction；action-free 保留同形状 action
encoder 与 learned mask token。两 arm 复用同一四区间 variable-`Δt` transition，只在初态
teacher-force，并保持参数量、updates、数据顺序、grid 和 seed 公平。

当前状态是 `c3_implemented_pending_clean_acceptance`。工作区的 CPU tiny CLI 已通过；宿主 GPU
canonical/reverse diagnostic repeat 由独立 verifier 完成 24/24 checks，semantic index 为
`709f6f76…d3db`，每 arm 35,734 个参数，峰值显存 68,277,760 bytes，最低空闲显存
15,470,034,944 bytes；test split 与参数公平性账本 mutation 均以 exit 4 被拒绝。由于代码尚未
提交，要求 clean checkout 的正式 gate 尚未运行，因此这些是 diagnostic evidence，不是已发布
C3 acceptance。没有运行 HPO、formal training 或 final test，也没有冻结正式 checkpoint 或模型
性能/科学比较结论。

## 项目事实源

- [`docs/PRD.md`](docs/PRD.md)：问题、用户、概念数据语义、声明门和开放决策。
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)：按可证伪假设拆分的 `PR-00`–`PR-11` 路径。
- [`docs/state/pr-queue.md`](docs/state/pr-queue.md)：PR-01A–F、PR-02A–F 等切片的当前动态状态。
- [`REFERENCES.md`](REFERENCES.md)：固定预览、候选数据、归档资源和许可状态。
- [`AGENTS.md`](AGENTS.md)：稳定协作、授权、安全和验收规则。

不要据此预建训练框架、下载大型数据或恢复旧项目。旧 ObjGauss 只读恢复点是标签
`archive/objgauss-final-2026-07-14`（提交 `e891bbf`）；归档代码、模型、数据合同和研究结论都
不是当前项目资产或事实基线。
