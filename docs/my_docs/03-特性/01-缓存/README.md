# 缓存

前缀树、分层缓存、KV 池形态、权重按层卸到 CPU。调度步怎么匹配和写回在 [04-前缀与KV](../../02-流程/03-调度与推理/04-前缀与KV.md)。池和树的启动次序在 [07-KV与前缀树](../../02-流程/01-启动/07-KV与前缀树.md)。

运行时行为 **源码路径推断，未实测**。域地图在 [00-域总览](00-域总览.md)。

| 页 | ID | 写什么 |
| --- | --- | --- |
| [00-域总览](00-域总览.md) | B23–B28、B49、B68、B69、B89 | 选型主链。分层缓存挂在同一棵 `UnifiedRadixCache` 上 |
| [01-RadixCache](01-RadixCache.md) | B23 | 统一树与 legacy 树、五种驱逐、锁 |
| [02-chunked-prefix](02-chunked-prefix.md) | B24 | MLA 长前缀改走 MHA chunked-KV。不改树的插入 |
| [03-SWA](03-SWA.md) | B25 | 纯 SWA 与 hybrid component。V4 池的例外 |
| [04-HiCache](04-HiCache.md) | B26 | host / L3、`cache` 与 `buffer_only`、预取和写回 |
| [05-存储后端](05-存储后端.md) | B27 | 注册表里的每一种 L3，以及 LMCache / FlexKV |
| [06-Chunked-prefill](06-Chunked-prefill.md) | B28 | 长提示拆轮。attention DP 整除块大小 |
| [07-Session](07-Session.md) | B49 | Legacy 请求树与 radix 引用计数 |
| [08-Unified-memory](08-Unified-memory.md) | B68 | 多子池一块字节。post-capture 要求 CUDA |
| [09-KV分片](09-KV分片.md) | B69 | PD prefill 的逻辑页。pipeline 跑两次 |
| [10-层CPU-offload](10-层CPU-offload.md) | B89 | 权重组预取。与 PLE offload 互斥 |

和 HiSparse 的交界：HiSparse 要求 `--disable-radix-cache`，见 [02-注意力与稀疏](../02-注意力与稀疏/01-HiSparse.md)。

返回 [特性目录](../README.md)。
