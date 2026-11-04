# 缓存

占位。前缀缓存、分层缓存、KV 池形态。不写树的操作和驱逐。

核对：清单 B 的下列行。

| ID | 特性 |
| --- | --- |
| B23 | RadixCache |
| B24 | chunked prefix cache |
| B25 | SWA / hybrid 树 |
| B26 | HiCache |
| B27 | HiCache 存储后端 |
| B28 | Chunked prefill |
| B49 | Session |
| B68 | Unified memory |
| B69 | KV 分片 / page interleave |
| B89 | 层 CPU offload |

池和树在启动链上的次序是 F126–F135，归 [02-流程/01-启动](../../02-流程/01-启动/README.md)。前缀树为何挂在 scheduler 是 F200，归 [02-流程/03-调度与推理](../../02-流程/03-调度与推理/README.md)。

返回 [特性目录](../README.md)。
