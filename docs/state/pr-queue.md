# PR 执行队列

> 动态状态源；稳定假设、范围和验收定义见 [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。
> 日期：2026-07-14

| 切片 | 状态 | 当前证据 | 下一门 |
| --- | --- | --- | --- |
| Stage-0 | `committed_local` | commit `c1927b1` | 不扩大声明 |
| PR-00 | `committed_local_supported` | commit `b4107fa`；本地 `npm run check` supported | 远端 CI 尚未运行 |
| RES-001 snapshot/RNG | `committed_local_supported` | commit `71d4e39`；两进程 evidence `1affc32d…e80b` | 不扩大来源授权 |
| PR-01 source action/contact gate | `committed_local_supported` | commit `71d4e39`；canonical/reverse evidence `3c2c8a7d…16f2` | 不扩大为 robot controller/render 能力 |
| PR-01A Contract | `committed_local_supported_pending_remote` | commit `71d4e39`；Node 24.18.0 全库门；4 schemas、4 positive fixtures、11 negatives；report `f250ef05…64e1` | 最终 SHA 远端 CI |
| PR-01B Runtime | `committed_local_supported_pending_remote` | commit `71d4e39`；`sim/uv.lock`；clean venv；10 tests；canonical/reverse evidence `8a2013f1…71cb0` | 最终 SHA 远端 CI |
| PR-01C Writer | `committed_local_supported_pending_remote` | commit `71d4e39`；22 Python tests；canonical/reverse evidence `d25a635c…2dfb4`；每 branch 111 trajectory / 110 contact records | 最终 SHA 远端 CI |
| PR-01D Audit | `committed_local_supported_pending_remote` | commit `71d4e39`；14 hard gates；11 mutations；固定四态与退出码 | 最终 SHA 远端 CI |
| PR-01E Cohort | `committed_local_supported_pending_remote` | commit `71d4e39`；正式 48 groups / 240 episodes；24/12/12；0 failed/extra；独立 audit supported | 最终 SHA 远端 CI |
| PR-01F Delivery | `committed_local_supported_pending_remote` | commit `71d4e39` clean `accept-pr01` supported；1210 checksum entries；source commit 绑定 HEAD | 状态同步后的最终 SHA 本地复跑与远端 CI |

状态只描述当前仓库与已运行证据；代码合并不自动把研究假设变为 `supported`。
