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
| PR-01F Delivery | `committed_remote_supported` | 验收 SHA `234ba00`；clean `accept-pr01` supported；1210 checksum entries；远端 delivery 成功 | 保持回归门，不向 PR-02 外推声明 |
| PR-02A Contract | `committed_local_supported` | 7 schemas；6 records/positive fixtures；39 negatives；旧 5 contract hashes 冻结；report `3b1e64a0…acccca3f`；`npm run check` 通过 | 远端未运行，不扩大为数据/模型证据 |
| PR-02B Pilot/Data Freeze | `committed_local_supported` | `b99b5f1` + `04ddb18`；代码承载 SHA `04ddb18` clean acceptance：两遍各 12 groups / 60 episodes、0 failed/extra，source audits、GPU 1 GiB reserve、21 checks 与 checksums supported；report `47ad53c6…944cc`；保留 source rejection 与 power 量纲错误负证据 | 无远端 CI；后续代码 HEAD 必须重跑唯一验收门 |
| PR-02C Trainer/Baselines | `c1_committed_local_supported` | C0 `fc20023` clean supported；C1 代码承载 SHA `adb1a62` 的 clean gate 为 60 groups / 300 branches / 0 failures、16 checks，producer/loader/verifier data index `2501ebc2…17a81b5` 一致 | 无远端 CI；下一片只实现 C2 copy-state 与 constant-velocity baselines |
| PR-02D Independent Audit | `planned_blocked_by_pr02c` | evaluator 独立性与 hard gates 已预注册；尚无实现 | PR-02C 通过后实现 mutation/audit 门 |
| PR-02E Formal Experiment | `planned_blocked_by_pr02d` | final 隔离、统计与 verdict 语义已预注册；尚未运行 | PR-02D `supported` 后一次性运行 frozen experiment |
| PR-02F Delivery/CI | `planned_blocked_by_pr02e` | Web/CI 声明边界已确认；尚无 Delivery | PR-02E 产生有效 verdict 后构建验收投影 |

状态只描述当前仓库与已运行证据；代码合并不自动把研究假设变为 `supported`。
