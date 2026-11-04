# 请求接入

本目录写 HTTP 或 Engine 上的一份请求，如何变成 scheduler 队列里的 `Req`。树是 `297e54fd105e974cdf401e5b4d20253a22c16f90`（标签 `v0.5.23`）。运行时行为标 **源码路径推断，未实测**。说不准的标 **待核**。

调度循环、组 batch、前向不在这里展开。流式块怎么 yield、SSE 以外的增量合并归 [04-流式与非流式响应](../04-流式与非流式响应/README.md)。字段表仍在 [附录 06](../../附录-选项索引/06-Client请求选项大全.md)。

## 先读哪一页

讲解顺序是整段，再按请求走过的函数拆开。

| 顺序 | 页 | 覆盖 |
| --- | --- | --- |
| 0 | [00-阶段总览](00-阶段总览.md) | F10、F11。S7 到入队前的位置、场景、两条进程 |
| 1 | [01-协议入口](01-协议入口.md) | F140，以及 A.10 列出的其它协议。HTTP、OpenAI、Engine、gRPC |
| 2 | [02-校验与规范化](02-校验与规范化.md) | F141、F142。`GenerateReqInput` 与 `normalize_batch_and_arguments` |
| 3 | [03-分词与模板](03-分词与模板.md) | F143、F173–F178、F207。模板在协议层，encode 在 tokenizer |
| 4 | [04-下发与入队](04-下发与入队.md) | F144–F146。ZMQ、DP 控制器、`Req`、等待队列或 PD 队列 |
| 5 | [05-批对象与采样参数](05-批对象与采样参数.md) | F147–F149。`SamplingParams` 在接入时定稿。`ScheduleBatch` / `ForwardBatch` 只写对象边界 |

核对表：[../../调研笔记/00-全量知识点清单.md](../../调研笔记/00-全量知识点清单.md)。场景行在 [场景与硬件前提总表](../../01-体系总览/03-场景与硬件前提总表.md)。

## 状态

请求接入正文已按上面的页展开。调度与推理已写在 [03-调度与推理](../03-调度与推理/README.md)。仍空着的是流式与非流式响应。

## 本目录拥有的 ID

| 范围 | ID | 页 |
| --- | --- | --- |
| 粗阶段 | F10、F11 | [00-阶段总览](00-阶段总览.md) |
| 原生与其它协议入口 | F140 | [01-协议入口](01-协议入口.md) |
| 请求体与规范化 | F141、F142 | [02-校验与规范化](02-校验与规范化.md) |
| 分词 | F143 | [03-分词与模板](03-分词与模板.md) |
| Tokenizer 与模板 | F173–F178、F207 | [03-分词与模板](03-分词与模板.md) |
| 下发、收包、`Req` | F144、F145、F146 | [04-下发与入队](04-下发与入队.md) |
| 批对象与采样参数 | F147、F148、F149 | [05-批对象与采样参数](05-批对象与采样参数.md) |

F12、F13 以及 A.11 的调度循环不归本目录。F14 以及 A.12 的回包不归本目录。F169 解释为何非流式也进 `_stream_one_response` 的后半（yield 规则）在响应篇；本目录只写入口处 `stream` 如何选择 SSE 或一次 `__anext__`。

F177 的 `resolve_auto_parsers` 调用点在启动的进程拓扑。parser 名怎么落地、以及 `deepseek-v41` 不单在 `reasoning_parser_names.py`，写在分词页。

## 本批没写死的缺口

- 原生 gRPC 的 Rust 扩展在把 dict 交给 `RuntimeHandle.submit_request` 之前是否再做 proto 校验。Python 侧看不到。待核。
- `ScheduleBatch.init_new` 的调用者与 `ForwardMode` 的选择在调度篇。这里只钉类型和 `init_new` 的角色。
- `grammar_queue` 里编译完成后挪进 `waiting_queue` 的循环在调度篇。这里只写未命中缓存时先不进等待队列。
- 各多模态处理器的模型分支不逐个展开。目录是 `srt/multimodal/processors/`。
- OpenAI、Anthropic 的每个 JSON 字段不对照附录 06。
- 扩散服务的 HTTP 不进 LLM 的 `TokenizerManager`。
- `RayEngine` 继承 `Engine.generate`。actor 如何替换子进程，启动篇已标待核。
- `SGLANG_RUST_SERVER` 不走 Python tokenizer。启动篇 F102、F204。
- 多卡、DP 路由和 PD 队列没有实跑记录。

返回 [流程总览](../00-总览.md)。
