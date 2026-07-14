# ADR-002：PR-00 Contract、坐标与 Web-first 验证栈

> 状态：Accepted
> 日期：2026-07-14
> 决策者：Owner（PR-00 Decision Freeze）
> 适用范围：`PR-00` 状态、干预与坐标协议

## 背景

Stage-0 已证明浏览器能够实际渲染严格 `.splat`，但它没有机器 episode contract、对象状态、
坐标门、同步观测或研究 verdict。`PR-00` 只验证一个假设：显式且可审计的 contract 能让固定
synthetic episode 的状态、干预与相机投影保持一致。它不引入外部数据、旧归档实现、训练或
Gaussian 建模。

## 决策

### 唯一 contract 与兼容

- 唯一机器源是
  [`contracts/objgauss/0.1.0/episode.schema.json`](../../contracts/objgauss/0.1.0/episode.schema.json)，
  使用 JSON Schema Draft 2020-12。Web、测试、producer 和未来 Python 只能作为 consumer，
  不得维护第二份字段真值。
- 首版为 `0.1.0`；版本化 `$id` 与记录的 `schema_version` 必须精确匹配，object schema 拒绝
  未知字段。已发布版本不可原地修改；`0.x` 中改变合法实例集合升级 MINOR，PATCH 不改变
  contract。迁移必须显式、可测试并记录前后 checksum/lineage，禁止静默升级。
- RGB、depth 与 mask 只通过 `uri/media_type/dtype/shape/sha256` descriptor 引用，不内嵌数组。
  `availability` tagged union 是唯一缺失表达；`null`、哨兵值和 NaN 均不是缺失。

### 坐标、姿态与时间

- `T_AB · p_B = p_A`，矩阵以 row-major 序列化并左乘列向量。World 为右手系、`+Z` 向上、
  meter；OpenCV Camera 为 `+X` 右、`+Y` 下、`+Z` 前。`T_WC` 是 Camera → World，投影使用
  `T_CW = inverse(T_WC)`；WebGL bridge 只属于 Viewer consumer。
- Quaternion 是有限、归一化、确定性符号的 `[w,x,y,z]`。`episode_time_s` 从首个
  Observation 的 `0.0` 开始；Observation 严格递增，同步字段共时，事件可同刻但不得倒退。
- Canonical object frame 由 producer 定义且 episode 内不变；synthetic 原点是刚体质心，轴是
  producer-authored 右手语义轴。Symmetry 显式为 `none`、有限旋转集合或连续轴；未知值阻塞
  姿态指标，不能默认 `none`。

### Fixture、evaluator 与 verdict

- `src/pr00/synthetic-audit.mjs` 用固定 seed/config 生成 `synthetic-audit-v0`。Git 只提交
  producer、fixture spec/checksum manifest；`generated/pr00/` 派生 episode、资源、报告和浏览器
  bundle 全部 ignored。PR-00 构建不联网、不下载数据、不读取旧归档。
- Subject frame math 位于 `src/pr00/frame-math.mjs`。Primary GT 由 producer 内独立 reference
  投影生成；`src/pr00/reprojection-evaluator.mjs` 不拥有 GT 投影公式，只比较 subject prediction
  与冻结像素。
- 唯一 endpoint 是全部 36 个 primary points 的
  `max_camera_reprojection_error_px < 1.0`。任一点未达门为 `rejected`；零有效点、共享 GT 逻辑
  或失效审计链为 `invalid`。Round-trip、schema、资源 checksum 和 14 类反例是 correctness
  gates，不能替代 endpoint。
- Viewer 动态加载构建后的 browser consumer；consumer 重新校验 schema、episode、资源和报告
  后才显示 RGB、对象、相机、坐标轴与轨迹。失败时显示 `BLOCKED`，Stage-0 渲染可见性不改变
  PR-00 verdict。

### 工具链与唯一门禁

- Node 固定为 `.node-version` 中的 `24.18.0`，使用 JavaScript ESM、npm、Ajv `8.20.0`、
  esbuild `0.28.1` 和 `node:test`；没有服务端框架。
- `package-lock.json` 是依赖解析事实源。安装使用 `npm ci`，不接受未锁定安装作为 CI 证据。
- 本地与 GitHub Actions 的唯一验收入口是 `npm run check`，其顺序为 build、contract audit、
  行为/负例测试和语法检查。GitHub Actions 在 PR 与 `main` 上运行，actions 固定到不可变提交
  SHA；workflow 不部署、不发布也不写外部系统。

## 目录责任

| 路径 | 唯一责任 |
| --- | --- |
| `contracts/objgauss/0.1.0/` | 唯一 JSON Schema contract |
| `contracts/fixtures/` | 固定 fixture spec、预期 checksum 与 producer identity |
| `src/pr00/` | frame math、validator、producer、evaluator、verdict 与 browser consumer 源码 |
| `scripts/build-pr00.mjs` | 生成并核对 ignored PR-00 派生产物，构建 browser consumer |
| `scripts/audit-pr00.mjs` | 从生成文件重新计算 schema/resource/endpoint/report 证据 |
| `generated/pr00/` | ignored 派生 episode、数组、报告和浏览器 bundle；不是事实源 |
| `viewer/pr00-view.mjs` | 只消费已验证 contract，绘制同步 Web 证据视图 |
| `tests/pr00-*.test.mjs` | contract、frame、endpoint 和预注册负例行为门 |

## 回滚与影响

删除 PR-00 新增源码、contract、workflow 与 package files，并恢复 Viewer 的 contract 入口即可
回到 Stage-0；ignored `generated/pr00/` 可直接丢弃后重建。已冻结的 `0.1.0` schema 一旦提交便
不得原地修改，后续兼容变化必须新增版本和显式 migrator。

通过 PR-00 最多允许声明：`synthetic-audit-v0` 的 schema/语义、Robotics/OpenCV 坐标链、资源
lineage 与独立重投影门得到支持。不允许外推为真实数据有效、Gaussian 重建、世界模型、动力学
或规划价值。旧归档保持零文件移植；未来候选移植必须逐文件另行授权和审查。
