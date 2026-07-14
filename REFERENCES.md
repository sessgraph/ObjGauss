# Archived Project References

本文件只保留旧 ObjGauss 项目对下个项目可能有用的外部地址和恢复入口，不代表这些素材
适合新项目使用，也不继承其许可证或研究结论。

## Git

- GitHub：<https://github.com/sessgraph/ObjGauss>
- 本地 remote：`git@github.com:sessgraph/ObjGauss.git`
- 完整归档标签：`archive/objgauss-final-2026-07-14`
- 完整归档提交：`e891bbf`
- 旧版完整 `AGENTS.md`：
  `git show archive/objgauss-final-2026-07-14:AGENTS.md`
- 旧版 Hugging Face 发布记录：
  `git show archive/objgauss-final-2026-07-14:docs/state/huggingface-release.md`

归档标签当前只存在于本地 Git；没有在本次清理中推送远端。

## ObjGauss Hugging Face 仓库

- Dataset：<https://huggingface.co/datasets/jianyong365/objgauss-nerf-lego-near1m>
- Model：<https://huggingface.co/jianyong365/objgauss-nerf-lego-near1m-model>

这两个仓库是旧项目的 development-stage release。其实际文件、校验和与许可边界应以归档
中的 `docs/state/huggingface-release.md` 和远端现状重新核对，不应直接当作新项目资产。

## 旧项目使用过的外部 Hugging Face 来源

- Gaussian splat 样例：<https://huggingface.co/cakewalk/splat-data>
- BOP HOPE：<https://huggingface.co/datasets/bop-benchmark/hope>
- BOP LMO：<https://huggingface.co/datasets/bop-benchmark/lmo>

`cakewalk/splat-data` 在旧项目记录中属于多来源、混合许可素材，只能在重新核验具体文件许可后
使用。BOP 数据也应按各自上游条款重新确认用途。
