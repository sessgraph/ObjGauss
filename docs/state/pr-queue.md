# PR 执行队列

> 动态状态源；稳定假设、范围和验收定义见 [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。
> 日期：2026-07-14

| 切片 | 状态 | 当前证据 | 下一门 |
| --- | --- | --- | --- |
| Stage-0 | `committed_local` | commit `c1927b1` | 不扩大声明 |
| PR-00 | `committed_local_supported` | commit `b4107fa`；本地 `npm run check` supported | 远端 CI 尚未运行 |
| RES-001 snapshot/RNG | `implemented_local_supported_uncommitted` | 两进程 evidence `1affc32d…e80b` | 与 PR-01A 一并评审提交边界 |
| PR-01 source action/contact gate | `implemented_local_supported_uncommitted` | canonical/reverse evidence `3c2c8a7d…16f2` | 与 PR-01A 一并评审提交边界 |
| PR-01A Contract | `implemented_local_supported_uncommitted` | Node 24.18.0 全库门；4 schemas、4 positive fixtures、11 negatives；report `f250ef05…64e1` | 范围复核与 Owner commit 授权；远端 CI 未运行 |
| PR-01B Runtime | `implemented_local_supported_uncommitted` | `sim/uv.lock`；clean venv；10 tests；canonical/reverse evidence `8a2013f1…71cb0`；CI job 已定义 | 范围复核与 Owner commit 授权；远端 CI 未运行 |
| PR-01C Writer | `implemented_local_supported_uncommitted` | 22 Python tests；canonical/reverse golden evidence `d25a635c…2dfb4`；每 branch 111 trajectory / 110 contact records | 最终 commit lineage；范围复核与 Owner commit 授权；远端 CI 未运行 |
| PR-01D Audit | `implemented_local_supported_uncommitted` | 14 hard gates；11 mutations；固定四态与退出码 | 范围复核与 Owner commit 授权；远端 CI 未运行 |
| PR-01E Cohort | `implemented_local_supported_uncommitted` | 保留 provisional rejection；正式 48 groups / 240 episodes；24/12/12；0 failed/extra；独立 audit supported | 范围复核与 Owner commit 授权；远端 CI 未运行 |
| PR-01F Delivery | `implemented_local_pending_clean_commit` | 五联无 RGB Demo、report/checksums、`accept-pr01`、CI 与 clean-head guard 已实现；dirty checkout 负例通过 | Owner commit 授权；最终 SHA clean checkout supported；远端 CI supported |

状态只描述当前仓库与已运行证据；代码合并不自动把研究假设变为 `supported`。
