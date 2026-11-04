# 流式与非流式响应

本目录写 scheduler 把 token id 交出去之后的一段：detokenize、增量合并、流式 yield 或一次 JSON，以及中止和用量计数怎么挂上。树是 `297e54fd105e974cdf401e5b4d20253a22c16f90`（标签 `v0.5.23`）。运行时行为标 **源码路径推断，未实测**。说不准的标 **待核**。

调度侧停在 `SchedulerOutputStreamer.stream_output`，见 [03-调度与推理](../03-调度与推理/README.md)。HTTP 入口如何在 `stream` 上选择外壳，见 [02-请求接入](../02-请求接入/README.md)。采样函数在调度步的调用点在调度篇；采样后端的名字和语法后端在本目录。

## 先读哪一页

讲解顺序是整段，再按 token id 离开 scheduler 之后走过的函数拆开。

| 顺序 | 页 | 覆盖 |
| --- | --- | --- |
| 0 | [00-阶段总览](00-阶段总览.md) | F14。S11 的位置、三条场景轴、一条非流式请求的函数序 |
| 1 | [01-采样后端与语法](01-采样后端与语法.md) | F165、F166。后端如何定稿，`Sampler.forward` 的分叉，语法失败如何变成 abort |
| 2 | [02-detokenize进程](02-detokenize进程.md) | F167。进程与旁路、两套 id、`DecodeStatus` 增量解码 |
| 3 | [03-增量合并与yield](03-增量合并与yield.md) | F168、F169、F172、F198。`ReqState`、同一生成器、增量合并 |
| 4 | [04-HTTP与Engine回包](04-HTTP与Engine回包.md) | F170、F171。SSE 与一次 JSON、Engine、中止、`usage` |

核对表：[../../调研笔记/00-全量知识点清单.md](../../调研笔记/00-全量知识点清单.md)。场景行在 [场景与硬件前提总表](../../01-体系总览/03-场景与硬件前提总表.md)。

## 状态

流式与非流式响应正文已按上面的页展开。流程 A 里归本目录的 ID 都写在这些页上。特性已写在 [03-特性](../../03-特性/README.md)。模型和分布式仍是空壳。

## 本目录拥有的 ID

| 范围 | ID | 页 |
| --- | --- | --- |
| 粗阶段 | F14 | [00-阶段总览](00-阶段总览.md) |
| 采样后端与语法后端 | F165、F166 | [01-采样后端与语法](01-采样后端与语法.md) |
| detokenizer 循环 | F167 | [02-detokenize进程](02-detokenize进程.md) |
| tokenizer 等待、同一生成器、增量流 | F168、F169、F172 | [03-增量合并与yield](03-增量合并与yield.md) |
| HTTP 流式与非流式 | F170、F171 | [04-HTTP与Engine回包](04-HTTP与Engine回包.md) |
| 为何非流式也进 `_stream_one_response` | F198 | [03-增量合并与yield](03-增量合并与yield.md) |

F12、F13 以及 A.11 的调度循环不归本目录。F164 的 `stream_output` 和套接字「谁可以 PUSH」在调度篇；本目录从 payload 的字段和 detokenizer 的 PULL 写起。F10、F11 的路由选择在请求接入；本目录写 `serve_generate_request` 拿到生成器之后如何包帧。语法库的算法是 B48。采样旗标的逐项说明在附录 05。

## 本批没写死的缺口

- SSE 字节、断连时序、多卡 TP 上只有 0 号发送，都没有实跑记录。
- `skip_tokenizer_init` 时 detokenizer 进程仍被拉起，生成流量不进它的 PULL。`FreezeGCReq` 和 `ConfigureLoggingReq` 是否还可能发到这个空转进程，发送端没有逐条对完。待核。
- 非增量流式的中间块把 `output_ids` 指到 `state.output_ids` 这同一个列表。`batch_notify_size` 触发的 `sleep(0)` 和 HTTP `dumps_json` 交错之后，客户端会不会看到被后续 token 续写的列表，待核。
- NPU 上 `has_flashinfer` 的取值未知。`platform_hook` 不会自动把 `sampling_backend` 写成 `ascend`。待核。
- `flashinfer` / `intel_xpu` 的 sampling 符号导入失败时的异常路径没有逐条展开。待核。
- 自定义采样后端的工厂能否赶在 CLI choices 固化之前注册，只按启动篇 F18 的插件时序推断。未实测。
- `skip_tokenizer_init` 且请求带停止串时，scheduler 上 `req.tokenizer` 为 None，`tail_str` 仍会 `decode`。有没有更早的拒绝，待核。
- beam 的 `decode_beam_search_output`、多 detokenizer router 的转发、OpenAI reasoning / tool 的 delta 切分，没有逐行展开。
- xgrammar、outlines、llguidance 的编译算法归特性篇。
- Rust server 的 detokenizer 不在这棵 Python 路径上。调度篇只写到 `push_generation` / `push_abort`。

返回 [流程总览](../00-总览.md)。
