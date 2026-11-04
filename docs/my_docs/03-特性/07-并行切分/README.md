# 并行切分

清单 B 里和并行轴同名的特性。轴的定义、组的个数和初始化在 [04-分布式](../../04-分布式/README.md)。本目录写特性开关把哪些尺寸和通信实现定下来。

运行时行为 **源码路径推断，未实测**。

| 页 | ID | 对应的 D |
| --- | --- | --- |
| [01-Attention-DP](01-Attention-DP.md) | B17 | D04、D18 |
| [02-模型副本DP](02-模型副本DP.md) | B18 | D03 |
| [03-Prefill-CP](03-Prefill-CP.md) | B40、B82、B85 | D06、D20。B85 不另开轴 |
| [04-Decode-CP](04-Decode-CP.md) | B41 | D07 |
| [05-PP](05-PP.md) | B42 | D02 |
| [06-集合通信](06-集合通信.md) | B70、B92、B93 | D17 及相邻融合，不是新轴 |

Attention DP 与 `dp_size>1` 互斥。多节点禁止副本 DP，提示改用 `--attn-dp-size`。

返回 [特性目录](../README.md)。
