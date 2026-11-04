# 调度与 PD

占位。调度策略、重叠、prefill/decode 分离在特性表里的三行。事件循环的 F 号不在这里。

| ID | 特性 |
| --- | --- |
| B29 | Overlap schedule |
| B30 | Two-batch overlap |
| B37 | PD 分离 |
| B38 | PD mux |
| B39 | PD 角色切换 |
| B50 | 优先级调度 |
| B51 | 调度策略 |
| B52 | 连续 decode |
| B64 | Beam search |
| B73 | Prefill delayer |

循环和 `ForwardMode` 是 F150–F164、F179–F188，归 [02-流程/03-调度与推理](../../02-流程/03-调度与推理/README.md)。PD 作为角色分裂是 D13、D14，归 [04-分布式/05-分离多机与通信](../../04-分布式/05-分离多机与通信.md)。

返回 [特性目录](../README.md)。
