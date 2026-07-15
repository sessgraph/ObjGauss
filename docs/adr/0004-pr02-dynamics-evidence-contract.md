# ADR-004：PR-02 Dynamics Evidence Contract 0.3.0

- 状态：accepted
- 日期：2026-07-15
- 范围：PR-02A Contract

## 背景

PR-02 需要在任何 pilot、数据生成和训练开始前，先冻结一套能区分实验规格、训练尝试、模型
选择、prediction-before-observation 与独立评估的机器记录。PR-00 `0.1.0` 和 PR-01 `0.2.0`
已经发布，不能原地增加训练字段，也不能通过自动迁移把旧 episode 伪装成模型证据。

## 决策

1. 保持 `0.1.0` 与 `0.2.0` 字节冻结，新增精确版本 `0.3.0`；consumer 必须使用
   `schema_version + contract_kind` 分派，不提供隐式 migration。
2. `0.3.0` 有六种可分派记录：
   - `objgauss.dynamics_experiment`；
   - `objgauss.training_trial`；
   - `objgauss.training_attempt`；
   - `objgauss.checkpoint_manifest`；
   - `objgauss.dynamics_prediction`；
   - `objgauss.dynamics_evaluation_report`。
3. `common.schema.json` 只保存 identifier、artifact descriptor、availability、provenance、model
   arm、component weights 和 verdict 等共享定义；它不是可分派 record kind。
4. 原始 simulator episode 继续引用 `0.2.0`，`0.3.0` 不复制 episode schema。实验、trial、
   checkpoint、prediction 和 report 通过 SHA-256、URI、source/runtime lineage 关联。
5. JSON Schema Draft 2020-12 负责严格类型、常量、枚举、路径和未知字段拒绝；dispatcher 的
   独立 semantic validators 负责 split 隔离、计数/时间算术、retry 分类、checkpoint lineage、
   quaternion、prediction payload checksum、置信区间和全门 verdict 等跨字段不变量。
6. PR-02A fixture 中的 horizon、scale、`δ`、seed/group 数均是 contract 测试值，不是实验
   freeze。正式数值只能由 PR-02B 的隔离 calibration/power pilot 产生并写入新 experiment spec。

## 后果

- PR-02B 必须消费通过 `0.3.0` validator 的 experiment spec，不能使用未版本化配置代替。
- Trainer 不得读取 final GT；raw prediction 先独立、原子发布，evaluation report 才能读取 GT
  并重算 hard gates。
- Schema-valid 只证明记录可表达、可拒绝错误输入，不支持数据充分性、模型性能、Gaussian
  dynamics 或机器人控制声明。
- 未来若改变合法实例集合，必须新增 SemVer 版本并提供显式、可测试、保留 checksum/lineage
  的迁移；不得修改本版本后继续称为 `0.3.0`。

## 验证

```bash
npm run contract:pr02a
node --test tests/pr02a-contracts.test.mjs
npm run check
```

当前本地结果：5 个旧 contract 文件哈希保持冻结；7 个 `0.3.0` schema 文件、6 个正向 fixtures
和 39 个负例通过；ignored machine report verdict 为 `supported`，SHA-256 为
`3b1e64a02120da2ca5ef5616af08f8ee068498139dfdb54b4e471045acccca3f`。该结果已提交，但尚无远端
CI 证据。
