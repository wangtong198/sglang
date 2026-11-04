# MoE 与 DeepEP

专家 all-to-all 与专家 GEMM 是两套枚举。并行轴在 [04-分布式/04-专家并行](../../04-分布式/04-专家并行.md)。域地图在 [00-域总览](00-域总览.md)。

运行时行为 **源码路径推断，未实测**。DeepEP 与弹性扩缩没有在本环境跑过。

| 页 | ID |
| --- | --- |
| [00-域总览](00-域总览.md) | 本域全部 |
| [01-DeepEP](01-DeepEP.md) | B19 |
| [02-mori](02-mori.md) | B20 |
| [03-其它a2a](03-其它a2a.md) | B21 |
| [04-GEMM-runner](04-GEMM-runner.md) | B22 |
| [05-Shared-experts-TP](05-Shared-experts-TP.md) | B43 |
| [06-EPLB](06-EPLB.md) | B44 |
| [07-Elastic-EP](07-Elastic-EP.md) | B45 |
| [08-DWDP](08-DWDP.md) | B46 |
| [09-Expert-pack](09-Expert-pack.md) | B47 |
| [10-Waterfill](10-Waterfill.md) | B94 |

DSPARK 加 attention DP 时，a2a 只允许 `none`、`megamoe`、`mori`。DeepEP 不在这三项里。

返回 [特性目录](../README.md)。
