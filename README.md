# ObjGauss

当前工作区已完成 Demo A 的两个前置切片：Stage-0 在浏览器中直接渲染环境级 3D Gaussian；
`PR-00` 建立状态、干预和坐标的唯一机器 contract，并在固定 synthetic episode 上给出可复现
裁决。项目仍保持私有并按 all rights reserved 管理，对外发布前必须重新决策许可证。

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

## 项目事实源

- [`docs/PRD.md`](docs/PRD.md)：问题、用户、概念数据语义、声明门和开放决策。
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)：按可证伪假设拆分的 `PR-00`–`PR-11` 路径。
- [`REFERENCES.md`](REFERENCES.md)：固定预览、候选数据、归档资源和许可状态。
- [`AGENTS.md`](AGENTS.md)：稳定协作、授权、安全和验收规则。

不要据此预建训练框架、下载大型数据或恢复旧项目。旧 ObjGauss 只读恢复点是标签
`archive/objgauss-final-2026-07-14`（提交 `e891bbf`）；归档代码、模型、数据合同和研究结论都
不是当前项目资产或事实基线。
