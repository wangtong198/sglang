# 量化与图优化

占位。量化与权重格式、CUDA graph、编译、融合核。不写捕获步骤。

| ID | 特性 |
| --- | --- |
| B31 | CUDA graph |
| B32 | Piecewise / breakable 图 |
| B33 | torch.compile |
| B34 | 量化 |
| B35 | 权重格式 |
| B59 | Megakernel / Monokernel |
| B90 | 稠密 GEMM runner |

图在启动链上的位置是 F116、F117 和 F195，归 [02-流程/01-启动](../../02-流程/01-启动/README.md)。量化方法名单在清单 B34，展开时从 `QUANTIZATION_CHOICES` 核对。旗标在 [附录 05](../../附录-选项索引/05-Server选项大全.md)。

返回 [特性目录](../README.md)。
