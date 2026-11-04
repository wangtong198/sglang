# 调度与推理

本目录写 scheduler 收包之后的一轮：选哪条事件循环、从 `waiting_queue` / `grammar_queue` 组 batch、前向、读写 KV、采样，再把 token id 交出去。树是 `297e54fd105e974cdf401e5b4d20253a22c16f90`（标签 `v0.5.23`）。运行时行为标 **源码路径推断，未实测**。说不准的标 **待核**。

请求如何进队列在 [02-请求接入](../02-请求接入/README.md)。detokenize、SSE、非流式的一次 yield 在 [04-流式与非流式响应](../04-流式与非流式响应/README.md)。本目录写到 `SchedulerOutputStreamer.stream_output` 为止。

## 先读哪一页

讲解顺序是整段，再按一次请求在 scheduler 和 model runner 里走过的函数拆开。

| 顺序 | 页 | 覆盖 |
| --- | --- | --- |
| 0 | [00-阶段总览](00-阶段总览.md) | F12、F13。S9–S10 的位置、场景、一条请求的函数序 |
| 1 | [01-事件循环](01-事件循环.md) | F150–F157。`dispatch_event_loop`、普通循环与 overlap |
| 2 | [02-队列与策略](02-队列与策略.md) | F158、F159。`grammar_queue`、`waiting_queue`、`SchedulePolicy`、优先级 |
| 3 | [03-组batch](03-组batch.md) | F160。连续批、chunk、`ScheduleBatch`、prefill / decode / MIXED |
| 4 | [04-前缀与KV](04-前缀与KV.md) | F161、F200。匹配、分配、checkpoint，以及树为何在 scheduler 上 |
| 5 | [05-前向与采样](05-前向与采样.md) | F13、F162、F193。`ForwardBatch`、`ModelRunner.forward`、采样、空的 attention DP 组 |
| 6 | [06-结果回传](06-结果回传.md) | F163、F164。`process_batch_result`、overlap 下的延迟采样、送出 token id |
| 7 | [07-PD分离](07-PD分离.md) | F179–F188、F203。prefill / decode 循环、`PREBUILT`、mux、角色切换 |

核对表：[../../调研笔记/00-全量知识点清单.md](../../调研笔记/00-全量知识点清单.md)。场景行在 [场景与硬件前提总表](../../01-体系总览/03-场景与硬件前提总表.md)。

## 状态

调度与推理正文已按上面的页展开。仍空着的是流式与非流式响应。

## 本目录拥有的 ID

| 范围 | ID | 页 |
| --- | --- | --- |
| 粗阶段 | F12、F13 | [00-阶段总览](00-阶段总览.md)，F13 的前向在 [05-前向与采样](05-前向与采样.md) |
| 事件循环 | F150–F157 | [01-事件循环](01-事件循环.md) |
| 排队、chunk、前缀、前向、结果、回送 | F158–F164 | [02](02-队列与策略.md) 至 [06](06-结果回传.md) |
| PD | F179–F188 | [07-PD分离](07-PD分离.md) |
| 为何 | F193、F200、F203 | [05](05-前向与采样.md)、[04](04-前缀与KV.md)、[07](07-PD分离.md) |

F10、F11 以及 A.10 的入队不归本目录。F14 以及 A.12 的 detokenize 和 SSE 不归本目录。F147–F149 的对象边界在请求接入；本目录写这些对象在调度步被构造和消费的调用点。F196（PP 关掉 overlap）在启动篇，本目录只消费那个改写。投机算法内部、HiCache 搬运、HiSparse、dLLM 的去噪步归特性篇，本目录只写它们插进调度循环的分叉。

PD 作为并行角色在 [04-分布式/05-分离多机与通信](../../04-分布式/05-分离多机与通信.md)。特性侧的 B37–B39 在 [03-特性/06-调度与PD](../../03-特性/06-调度与PD/README.md)。

## 本批没写死的缺口

- `schedule_policy` 的 CLI choices 含 `"priority"`，`SchedulePolicy` 的两个枚举都不收这个字符串，`arg_groups` 里也没看到把它改写成 `fcfs` 的 hook。正式优先级是 `enable_priority_scheduling`。`"priority"` 这个 choice 会在何处失败，待核。
- `num_continuous_decode_steps` 的 help 写连续多步 decode，默认 1。`python/sglang/srt` 里除字段声明外没有读到使用点。是否仍生效，待核。
- `SpeculativeAlgorithm.is_eagle()` 把 `FROZEN_KV_MTP` 算进去，旁边有 FIXME，写明 scheduler 支持尚未立住。MTP 在调度路径上的重叠和语法屏障，待核。算法本体不在本目录。
- 各 PD 传输后端的握手和页布局没有逐个展开。循环只调用发送和队列。
- dLLM、HiSparse、beam、LoRA 准入、HiCache 的 host/storage 搬运只标了分叉。
- MLX overlap 循环的逐步对齐，启动篇已标待核。
- 多卡集体通信、PD 双进程、校验和拦住坏页，都没有实跑记录。

返回 [流程总览](../00-总览.md)。
