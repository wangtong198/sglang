# 注意力与稀疏

稀疏注意力、注意力后端、线性注意力、V4 的 Engram / mHC、Qwen4 PLE、确定性。Attention DP 在 [07-并行切分](../07-并行切分/01-Attention-DP.md)。域地图在 [00-域总览](00-域总览.md)。

运行时行为 **源码路径推断，未实测**。

| 页 | ID |
| --- | --- |
| [00-域总览](00-域总览.md) | 本域全部 |
| [01-HiSparse](01-HiSparse.md) | B11 |
| [02-DSA](02-DSA.md) | B12 |
| [03-QSA](03-QSA.md) | B13 |
| [04-MiniMax稀疏](04-MiniMax稀疏.md) | B14 |
| [05-Qwen稀疏](05-Qwen稀疏.md) | B15 |
| [06-Quest](06-Quest.md) | B16 |
| [07-Mamba与线性注意力](07-Mamba与线性注意力.md) | B56 |
| [08-KDA](08-KDA.md) | B77 |
| [09-GDN](09-GDN.md) | B78 |
| [10-Lightning](10-Lightning.md) | B79 |
| [11-ShortConv](11-ShortConv.md) | B80 |
| [12-Attention后端目录](12-Attention后端目录.md) | B57 |
| [13-Engram](13-Engram.md) | B58 |
| [14-mHC](14-mHC.md) | B76 |
| [15-PLE](15-PLE.md) | B81 |
| [16-DSA-prefill核](16-DSA-prefill核.md) | B83 |
| [17-DSA-cache切层](17-DSA-cache切层.md) | B84 |
| [18-确定性推理](18-确定性推理.md) | B61 |
| [19-batch-invariant](19-batch-invariant.md) | B62 |

V4.1 拒绝 HiSparse 的原文在 `validate_deepseek_v41_features`。整张拒绝表归 [05-模型](../../05-模型/04-模型与特性矩阵.md)。

返回 [特性目录](../README.md)。
