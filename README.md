# Clean Project Workspace

这是一个供下个项目使用的干净工作区。原 ObjGauss 源码、3D Viewer、测试、下载数据、
训练数据、构建产物和本地依赖已从当前分支清理。

## ObjGauss 归档

- 归档标签：`archive/objgauss-final-2026-07-14`
- 归档提交：`e891bbf`
- Git tree：`1e0a57b59775cad752d97835ff2b2fd6a9d07d95`

建议用独立 worktree 只读查看旧项目，避免覆盖当前目录：

```bash
git worktree add --detach ../ObjGauss-archive archive/objgauss-final-2026-07-14
```

远端仓库和 Hugging Face 资料见 [`REFERENCES.md`](REFERENCES.md)。开始新项目时，请先确定
项目目标、技术栈、许可证和验收命令，再更新本文件与 `AGENTS.md`。
