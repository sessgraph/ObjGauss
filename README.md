# ObjGauss

当前工作区已完成 Demo A 的两个前置切片：Stage-0 在浏览器中直接渲染环境级 3D Gaussian；
`PR-00` 建立状态、干预和坐标的唯一机器 contract，并在固定 synthetic episode 上给出可复现
裁决。后续 `RES-001` 在固定 ManiSkill 3.0.1 runtime 上支持了无渲染 snapshot fork，并由
`PR-01` 首个 action/contact gate 支持了程序化 CPU primitive push sibling source；完整
里程碑现按 `PR-01A`–`PR-01F` 推进。`0.2.0` sibling evidence contract、隔离 production
runtime、原子 adapter/writer、independent audit 与冻结 cohort 均已在本地门支持；无 RGB
Delivery、checksums 与一键命令也已实现，并会在 dirty worktree 上 fail closed。PR-01A–F 已由
实现提交 `71d4e39` 固化，并在该 clean HEAD 上由 `./scripts/accept-pr01` 完整重建为
`supported`；当前只缺最终提交 SHA 的远端 CI，不能把 PR-01 标为关闭。robot controller、RGB/GPU
renderer 和训练仍未实现。项目保持
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
```

该外部文件只保存到 ignored `data/`，其 asset provenance 仍为 `unverified`。Stage-0 只证明
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
GitHub runtime workflow 已建立但尚未在远端运行，本地结果不能替代最终 commit SHA 的 CI。

## 验证 PR-01C 原子 Writer

PR-01C 把真实五分支 simulator 输出映射为 `0.2.0` episode/attempt，并通过临时目录、
`fsync`、校验与单次 rename 发布不可变 branch；同一逻辑 key 与同一语义内容为 no-op，内容
冲突或中途失败不会覆盖已发布 episode。以下门从全新临时 venv 构建 package，运行行为/负例
测试，再用两个独立真实进程生成 canonical/reverse golden group：

```bash
./scripts/check-pr01c-writer
```

当前本地门的 22 个 Python 测试通过；五个 branch 各含 111 条 trajectory records 与 110 条 contact
records，canonical/reverse 稳定 evidence SHA-256 均为
`d25a635c4f1f691428687a138571e85577de19485403807ce6418fe92322dfb4`。这只支持单一 golden
group 的 adapter、原子幂等 writer 与 attempt failure semantics；独立 auditor、正式 cohort 与
Delivery 的结论由后续切片各自提供，远端 CI 和完整 PR-01 关闭仍未完成。

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
golden fixture 的证据可独立审计；该段本身不替代后续 cohort、最终 lineage、远端 CI 或完整 PR-01。

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
不进入 Git；实现提交 `71d4e39` 的 clean 验收已把运行证据绑定到该 HEAD，远端 CI 仍未完成。

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
行为已通过本地测试和浏览器目视核对。实现提交 `71d4e39` 的 clean 一键验收得到 48 groups /
240 episodes、24/12/12 split、0 failed/extra attempts、独立 audit `supported`，并生成含 1210 个
条目的 checksum index；Delivery 的 source commit 与该 HEAD 一致。最终提交 SHA 的远端 GitHub
Actions 仍未运行，因此 PR-01F 尚未完成远端门，PR-01 也尚未关闭。

## 项目事实源

- [`docs/PRD.md`](docs/PRD.md)：问题、用户、概念数据语义、声明门和开放决策。
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)：按可证伪假设拆分的 `PR-00`–`PR-11` 路径。
- [`docs/state/pr-queue.md`](docs/state/pr-queue.md)：PR-01A–F 等切片的当前动态状态。
- [`REFERENCES.md`](REFERENCES.md)：固定预览、候选数据、归档资源和许可状态。
- [`AGENTS.md`](AGENTS.md)：稳定协作、授权、安全和验收规则。

不要据此预建训练框架、下载大型数据或恢复旧项目。旧 ObjGauss 只读恢复点是标签
`archive/objgauss-final-2026-07-14`（提交 `e891bbf`）；归档代码、模型、数据合同和研究结论都
不是当前项目资产或事实基线。
