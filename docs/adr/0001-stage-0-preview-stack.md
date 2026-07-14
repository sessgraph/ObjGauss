# ADR-001：Stage-0 本地预览技术栈

> 状态：Accepted for Stage-0 preview only
> 日期：2026-07-14
> 决策者：Owner（明确要求看到渲染后的 3D Gaussian，不使用 notebook）

## 背景

ObjGauss 已有研究路线，但此前工作区只有文档，Owner 无法看到第一阶段的任何可运行结果。
核心训练、机器 contract、研究数据落盘格式和长期前端栈仍未冻结，因此本决策只解决一个
问题：如何用最小、可撤销、可审计的方式，在浏览器中实际渲染固定的 3D Gaussian splat。

## 决策

- 使用原生 HTML、CSS 和浏览器 ES modules 实现 `viewer/`，不引入 npm、框架或生产依赖。
- 使用 WebGL2 instanced quads 投影 3D covariance，以 Gaussian alpha 和 premultiplied alpha
  blend 合成画面；不绘制点云 fallback、假地面网格或未冻结的坐标轴。小于等于 150,000
  splats 的首次 far-to-near 排序同步完成以保证首屏可见，后续相机排序放在 ES module Web
  Worker 中；更大本地文件的首次排序也使用 Worker。
- 只接受当前 `antimatter15-splat-v1` 固定布局：每条小端 `32` bytes，依次为 position `3 ×
  float32`、scale `3 × float32`、RGBA `4 × uint8` 和 wxyz quaternion `4 × uint8`。
- 默认场景由 `viewer/synthetic-world.mjs` 确定性生成严格 `.splat` records：固定
  `272,736` bytes、`8,523` records 与 SHA-256
  `4782f6ed4816aee54618bb4d1fcbce8df67e65301e23a89c155985084f51cfe6`，包含 Gaussian
  地形、路径、建筑、树和环境粒子。它只用于让 Web viewer 呈现环境上下文，不是 `PR-00`
  episode 或模型输出。
- 默认相机从环境内部沿路径观察，支持环绕、平移、`WASD` 移动与缩放；首屏不得从场景外部
  展示完整有限底板。自动环绕只作缓慢视差提示，不能形成商品转台观感。
- 外部审计样例固定为 commit `1267e2135660e1f4197f94c045453fe40c209b0e` 的
  `legobrick.splat`，通过 `scripts/fetch-gaussian-preview.sh` 下载到 Git ignored
  `data/local-preview/legobrick-1267e213/`。脚本与浏览器都校验固定字节数和 SHA-256，浏览器
  还校验 `103,060` 条记录；它不再承担默认“世界”语义。
- 使用 Node.js 内置 test runner 验证格式解析、covariance、深度排序、固定样例完整性与页面
  声明边界；使用 Python 标准库 HTTP server 做本地预览，无构建步骤。
- 解析器对空文件、非 `32` 倍数、超过 `48,000,000` bytes、非有限或绝对值超过 `1,000,000`
  的 position、非正或超过 `10` 的 scale、零范数 quaternion、在最大 `32×` 显示尺度下不能
  保持 finite float32 的 covariance 和全透明输入 fail closed。渲染分辨率单边限制为 `4,096`
  像素；GPU 上传和深度排序前统一减去 bounds center，避免大绝对坐标在 view matrix 中消差；
  该 renderer-local 平移不定义或暗示 `PR-00` 世界坐标约定。screen covariance 特征值使用
  缩放后的稳定计算。任一登记 fixture 的大小、哈希或记录数不符，
  以及 WebGL2、GPU texture、Worker、排序或首帧 draw 失败/30 秒无响应时也必须显示 `blocked`，
  不得把部分或旧结果称为就绪。
- 用户可打开本地 `.splat`，但只要其来源、许可和 checksum 未进入台账，页面就标为
  `local-file · unverified`，不能提升为项目证据。
- 页面必须同时显示 provenance、许可审查边界和禁止声明。默认 synthetic fixture 标为
  `semantic_kind=synthetic-gaussian-world`、`asset_provenance=generated-in-browser` 和
  `third_party_asset_review=not-required`，同时明确 `project_license=unresolved`；可选 Lego
  审计样例保持
  `semantic_kind=point-derived-splat`、`asset_provenance=unverified` 和
  `license_status=review-pending`，并显示“容器 MIT、逐资产权利与生成链未核验”。实际渲染
  Gaussian 不等于 trained 3DGS，更不等于 ObjGauss 重建、对象状态、动力学、规划价值或研究门
  通过。

权威运行与验证命令维护在 [`../../AGENTS.md`](../../AGENTS.md) 第 4 节。

## 影响

该决策提供一个无需构建步骤的环境级可见入口，也验证了 synthetic records、ignored 外部
数据、固定来源、哈希、`.splat` 解析和浏览器 Gaussian 渲染链路。它没有批准 Python 训练栈、
包管理器、模型框架、长期 UI 框架、机器 schema、原始研究数据下载或外部资产再分发；这些
仍需后续 ADR/contract 决策。

WebGL2 页面、Worker 和 `antimatter15-splat-v1` 解析器都是 Stage-0 的局部实现，不自动成为
长期前端、训练、模型 I/O 或公共 contract。后续若需要真实 episode、通用格式、LOD、时间轴
或模型输出接入，应在对应单假设 PR 中另行裁决，不把本 viewer 扩张成通用数据 adapter。
