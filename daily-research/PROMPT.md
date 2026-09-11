# WebNN 每日调研 — 手动触发 Prompt

> 用法：想跑当日调研时，把下面「===」之间的整段发给 Claude 即可。任务 1 只聚焦 **LiteRT backend**（不涉及 ORT / CoreML / DML）。

===

每日 WebNN 成长任务。请执行以下两项调研，并把完整中文报告写入归档文件。

【环境与资料】
- 设计文档目录：/home/junwei/workspace/webnn（含 weights-file-quota-design.md / .en.md、weights-file-quota-impl-guide.md、"Support LiteRT OpenVINO plugin in Chromium.pdf"）。
- Chromium 源码树（真实活代码，务必直接用）：/home/junwei/workspace/chromium/src，WebNN 实现在 services/webnn/，LiteRT/TFLite 后端在 services/webnn/tflite/（关键文件：graph_builder_tflite.cc、graph_impl_litert.cc、context_impl_litert.cc、context_impl_tflite.cc、op_resolver.cc、webnn_graph_builder_impl.cc）。
- 用户背景：Chromium WebNN 工程师（Intel，fujunwei），近期工作：LiteRT backend 从 GPU 迁移到 renderer 进程、weights file quota 设计、通过 LiteRT 集成 OpenVINO NPU plugin 到 Chromium。

【任务 1 — 提升 CL 数量（仅限 WebNN LiteRT backend）】
- **只聚焦 LiteRT/TFLite 后端**（services/webnn/tflite/ 与 LiteRT 路径相关的公共代码），不要涉及 ORT / CoreML / DirectML 后端。
- 直接在 services/webnn/tflite/ 下 grep 活代码（重点 grep：`TODO(`、`base::unexpected`、`NotSupported`、`not supported`、`crbug.com`），找出待完善功能、未实现算子/op、已知 bug、性能优化空间。
- 交叉 WebNN spec 与 LiteRT 实现的差距、Chromium issue tracker、上游 webmachinelearning/webnn issues。
- 产出 3-5 个具体、粒度适合单个 CL 的改进点，每个含：问题描述、涉及文件（带真实路径，尽量给行号）、预估工作量、为什么有价值。务必确认 TODO 未被他人抢先关闭（对比 git log / blame 或活代码现状）。
- 尽量给出与前几日 daily-research/*.md 不同的新候选或新角度（避免重复）。

【任务 2 — 探索 WebNN 创新点（发散，覆盖整个技术栈）】
- 横跨完整技术栈：Application 层、Framework 层（ONNX Runtime Web / TF.js / Transformers.js）、Runtime/Backend 层（LiteRT / OpenVINO / 其它）、Models 层（LLM / 扩散 / 视觉 / 语音）、架构/系统层（跨进程、weights 内存管理、NPU/GPU 调度、沙箱）。
- 结合用户三条工作主线。
- 产出 3-5 个有深度的创新点，每个标注：所属层、技术价值、对个人学习成长的价值、可行性评估。鼓励跨层结合。

【产出与归档】
- 用 WebSearch 补充最新 WebNN/LiteRT/OpenVINO/模型生态进展（注意当前时间）。
- 先用 `date +%F` 取当天日期，把完整报告写入 /home/junwei/workspace/webnn/daily-research/<YYYY-MM-DD>.md（目录不存在就创建；若当天文件已存在则追加"同日补充"章节而非覆盖）。文件顶部写日期和一句话摘要，正文结构清晰、每条建议具体可执行、避免空泛。
- 最后在对话里给出简短摘要（表格 + 一句话建议）并附上归档文件路径。

===
