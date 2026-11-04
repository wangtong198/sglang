# MoE 与 DeepEP

占位。专家计算、all-to-all 后端、专家负载和专家包。清单写明 a2a 与 GEMM runner 是两套枚举。本页不展开互斥条件。

| ID | 特性 |
| --- | --- |
| B19 | DeepEP |
| B20 | mori |
| B21 | 其它 a2a |
| B22 | MoE GEMM runner |
| B43 | Shared experts TP |
| B44 | EPLB |
| B45 | Elastic EP |
| B46 | DWDP |
| B47 | Expert pack |

这些后端是 D08 的通信实现，并行轴在 [04-分布式/04-专家并行](../../04-分布式/04-专家并行.md)（含 F201）。清单的边界句：DeepEP / mori / megamoe 不另算一条并行轴。

返回 [特性目录](../README.md)。
