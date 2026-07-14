# ObjGauss

当前项目正从文档立项进入 Demo A 的 Stage-0 可见切片。研究方向是
**对象中心、动作条件的 Gaussian 世界模型**；机器 contract、训练技术栈、新项目许可证和
研究阈值仍未冻结，Stage-0 页面不代表这些门已经通过。

## 先看页面

从仓库根目录运行：

```bash
bash scripts/fetch-gaussian-preview.sh
node --test tests/*.test.mjs
python3 -m http.server 8000 --bind 127.0.0.1
```

然后打开 <http://127.0.0.1:8000/viewer/>。页面默认在浏览器内确定性生成一个由 `8,523` 个
真实 splat records 组成的环境场景，包含 Gaussian 地形、路径、建筑、树和环境粒子，再用
WebGL2 投影、排序和合成。默认相机位于环境内部；可以拖动环绕、`Shift` + 拖动平移、用
`WASD` 移动、滚轮或双指缩放，并可调整 Gaussian 半径。页面还可切换到 `103,060`-splat 的
固定 Lego 审计样例或打开本地 `.splat`；外部下载仍位于 ignored `data/`，不会进入 Git。

这条链路证明的是“浏览器能生成或读取严格 `.splat`，并把环境级输入实际渲染为 3D
Gaussian”。默认世界是 `synthetic-gaussian-world`，只是 Web viewer fixture；Lego 审计样例是
`point-derived-splat`，来源链仍是 `unverified`。两者都不是训练好的 ObjGauss 输出，也不支持
重建质量、对象状态、动力学、规划价值或研究门声明。准确来源和许可边界见
[`REFERENCES.md`](REFERENCES.md)，局部技术决策见
[`docs/adr/0001-stage-0-preview-stack.md`](docs/adr/0001-stage-0-preview-stack.md)。

项目入口：

- [`docs/PRD.md`](docs/PRD.md)：问题、用户、范围、概念数据协议和分阶段研究门槛。
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)：按可证伪假设拆分的 `PR-00`–`PR-11` 路径。
- [`REFERENCES.md`](REFERENCES.md)：固定预览、候选数据、归档资源、许可状态与恢复入口。
- [`viewer/index.html`](viewer/index.html)：Stage-0 本地 3D Gaussian 渲染页面。

当前只批准了无生产依赖的 Stage-0 页面和固定小型预览。不要据此预建训练框架、引入生产
依赖或下载大型数据集。原 ObjGauss 源码、旧 Viewer、训练数据、构建产物和本地依赖仍只在
归档或外部存储中，不属于当前实现。

## ObjGauss 归档

- 归档标签：`archive/objgauss-final-2026-07-14`
- 归档提交：`e891bbf`
- Git tree：`1e0a57b59775cad752d97835ff2b2fd6a9d07d95`

建议用独立 worktree 只读查看旧项目，避免覆盖当前目录：

```bash
git worktree add --detach ../ObjGauss-archive archive/objgauss-final-2026-07-14
```

远端仓库和 Hugging Face 资料见 [`REFERENCES.md`](REFERENCES.md)。归档中的代码、模型、
数据合同和研究结论都只是候选输入，不自动成为新项目资产或事实基线。
