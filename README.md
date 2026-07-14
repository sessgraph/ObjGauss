# ObjGauss 新项目规划工作区

当前分支是新项目规划工作区，正在进行文档优先的立项。拟议方向是研究
**对象中心、动作条件的 Gaussian 世界模型**，但技术栈、许可证、机器数据协议和最终
验收命令尚未由 Owner 批准；当前文档不代表这些决策已经冻结。

规划入口：

- [`docs/PRD.md`](docs/PRD.md)：问题、用户、范围、概念数据协议和分阶段研究门槛。
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)：按可证伪假设拆分的 `PR-00`–`PR-11` 路径。
- [`REFERENCES.md`](REFERENCES.md)：候选数据、归档资源、许可状态与恢复入口。

在规划门通过前，不预建框架、不引入生产依赖、不下载大型数据集。原 ObjGauss 源码、
3D Viewer、测试、下载数据、训练数据、构建产物和本地依赖已从当前分支清理。

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
