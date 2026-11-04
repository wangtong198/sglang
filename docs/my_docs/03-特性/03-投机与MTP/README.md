# 投机与 MTP

目标模型验证草稿多猜出来的 token。启动时建草稿 worker 是 F113。V4.1 只放行 DSPARK，拒绝表归 [05-模型](../../05-模型/04-模型与特性矩阵.md)。

运行时行为 **源码路径推断，未实测**。

| 页 | ID |
| --- | --- |
| [00-框架](00-框架.md) | B01 |
| [01-EAGLE](01-EAGLE.md) | B02 |
| [02-NEXTN](02-NEXTN.md) | B03 |
| [03-FROZEN-KV-MTP](03-FROZEN-KV-MTP.md) | B04 |
| [04-STANDALONE](04-STANDALONE.md) | B05 |
| [05-NGRAM](05-NGRAM.md) | B06 |
| [06-DFLASH](06-DFLASH.md) | B07 |
| [07-UNO](07-UNO.md) | B08 |
| [08-DSPARK](08-DSPARK.md) | B09 |
| [09-解耦投机](09-解耦投机.md) | B10 |
| [10-草稿attention](10-草稿attention.md) | B91 |

`SpecInputType` 按算法分开。`is_eagle()` 目前含 `FROZEN_KV_MTP`，源码 FIXME 写明 scheduler 支持未立稳。

返回 [特性目录](../README.md)。
