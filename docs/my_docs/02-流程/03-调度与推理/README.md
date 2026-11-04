# 调度与推理

占位。scheduler 收包之后：选哪条事件循环、组 batch、前向、处理结果。PD 的循环分叉在 A.14，和 A.11 的 `dispatch_event_loop` 放在同一节。本页不写循环正文。

核对：[../../调研笔记/00-全量知识点清单.md](../../调研笔记/00-全量知识点清单.md) 的 A.11、A.14，A.1 的 F12、F13，以及下表的 A.15。

| 范围 | ID | 清单节 | 这一节要讲的事 |
| --- | --- | --- | --- |
| 粗阶段 | F12、F13 | A.1 | S9 组 batch；S10 `ScheduleBatch` 到 `ForwardBatch` |
| 调度与 KV | F150–F164 | A.11 | 事件循环选择、排队、chunk、前缀匹配、前向、回 tokenizer |
| PD | F179–F188 | A.14 | `disaggregation_mode`、传输后端、prefill/decode 循环、角色切换、mux、编码器分离 |

A.15 归本目录的问（只列问题，不回答）：

| ID | 问题 |
| --- | --- |
| F193 | 为何空的 attention DP 组仍要 `ForwardMode.IDLE` |
| F200 | 为何前缀树挂在 scheduler，而不是在 `ModelRunner.forward` 里现查 |
| F203 | 为何 PD decode 用 `PREBUILT` |

PD 作为并行角色（D13、D14）在 [04-分布式/05-分离多机与通信](../../04-分布式/05-分离多机与通信.md)。特性侧的 B37–B39 在 [03-特性/06-调度与PD](../../03-特性/06-调度与PD/README.md)。展开时三处互相指，各自写自己的 ID。

返回 [流程总览](../00-总览.md)。
