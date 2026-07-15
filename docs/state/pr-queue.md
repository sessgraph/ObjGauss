# PR 执行队列

> 动态状态源；稳定假设、范围和验收定义见 [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。
> 日期：2026-07-15

| 切片 | 状态 | 当前证据 | 下一门 |
| --- | --- | --- | --- |
| Stage-0 | `committed_local` | commit `c1927b1` | 不扩大声明 |
| PR-00 | `committed_remote_supported` | commit `b4107fa`；本地门 supported；验收 SHA `234ba00` 的远端 PR-00 check 成功 | 保持窄声明与回归门 |
| RES-001 snapshot/RNG | `committed_local_supported` | commit `71d4e39`；两进程 evidence `1affc32d…e80b` | 不扩大来源授权 |
| PR-01 source action/contact gate | `committed_local_supported` | commit `71d4e39`；canonical/reverse evidence `3c2c8a7d…16f2` | 不扩大为 robot controller/render 能力 |
| PR-01A Contract | `committed_remote_supported` | 验收 SHA `234ba00`；Node 24.18.0 全库门；4 schemas、4 positive fixtures、11 negatives；远端 check 成功 | 保持 `0.1.0` 冻结与精确分派 |
| PR-01B Runtime | `committed_remote_supported` | 验收 SHA `234ba00`；`sim/uv.lock`；clean venv；10 tests；canonical/reverse evidence `8a2013f1…71cb0`；远端 smoke 成功 | 不扩大 simulator 授权 |
| PR-01C Writer | `committed_remote_supported` | 验收 SHA `234ba00`；22 Python tests；同一 clean source 下 canonical/reverse evidence 一致；远端 golden group 成功 | 保持原子/幂等与 attempt 分账 |
| PR-01D Audit | `committed_remote_supported` | 验收 SHA `234ba00`；14 hard gates；11 mutations；固定四态/退出码；远端 audit 成功 | evaluator 继续独立 |
| PR-01E Cohort | `committed_remote_supported` | 验收 SHA `234ba00`；48 groups / 240 episodes；24/12/12；0 failed/extra；远端 frozen cohort 成功 | 冻结 spec、split 与预算 |
| PR-01F Delivery | `committed_remote_supported` | 验收 SHA `234ba00`；clean `accept-pr01` supported；1210 checksum entries；远端 delivery 成功 | 进入 PR-02 前不扩大声明 |

状态只描述当前仓库与已运行证据；代码合并不自动把研究假设变为 `supported`。
