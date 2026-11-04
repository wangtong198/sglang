# 量化与图优化

量化、权重格式、CUDA graph、编译、融合核、稠密 GEMM。图的捕获步骤在启动篇 F116、F117、F195。

运行时行为 **源码路径推断，未实测**。

| 页 | ID |
| --- | --- |
| [01-CUDA-graph](01-CUDA-graph.md) | B31、B32 |
| [02-torch-compile](02-torch-compile.md) | B33 |
| [03-量化](03-量化.md) | B34 |
| [04-权重格式](04-权重格式.md) | B35 |
| [05-融合核](05-融合核.md) | B59 |
| [06-稠密GEMM](06-稠密GEMM.md) | B90 |

`--moe-runner-backend` 属于 [04-MoE与DeepEP](../04-MoE与DeepEP/04-GEMM-runner.md)，不在稠密 GEMM 页里。

返回 [特性目录](../README.md)。
