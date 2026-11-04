# 并行切分

占位。清单 B 里和并行轴同名的特性行，加上自定义集合通信。轴的定义、组的个数和初始化在 [04-分布式](../../04-分布式/README.md)，不在本页重写。

| ID | 特性 | 对应的 D |
| --- | --- | --- |
| B17 | Attention DP | D04、D18 |
| B18 | 模型副本 DP | D03 |
| B40 | Prefill CP | D06 |
| B41 | Decode CP | D07 |
| B42 | PP | D02 |
| B70 | 自定义集合通信 | D17 |
| B82 | CP-TP 组共享 | D20 |
| B85 | CP 下 decode 的 attention TP 切权 | 组仍是 D06 / D01，不另开轴 |
| B92 | Aiter allreduce 融合 | 与 D17 相邻，不是新轴 |
| B93 | FlashInfer allreduce 融合 | 与 D17 相邻，不是新轴 |

返回 [特性目录](../README.md)。
