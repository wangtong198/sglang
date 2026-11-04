# 注意力与稀疏

占位。稀疏注意力、注意力后端、线性注意力 / Mamba、确定性相关开关。不写 kernel。

核对：清单 B 的下列行。Attention 后端名单在清单 B57 下的那一段，展开时从 `ATTENTION_BACKEND_CHOICES` 核对，不在本页抄名单。

| ID | 特性 |
| --- | --- |
| B11 | HiSparse |
| B12 | DSA |
| B13 | QSA |
| B14 | MiniMax 稀疏 |
| B15 | Qwen 稀疏 / Qwen4-Exp |
| B16 | Quest |
| B56 | Mamba / 线性注意力 |
| B57 | Attention 后端目录 |
| B58 | Engram |
| B61 | 确定性推理 |
| B62 | Batch invariant |
| B76 | mHC |
| B77 | KDA |
| B78 | GDN |
| B79 | Lightning attention |
| B80 | Short conv |
| B81 | Qwen4 PLE |
| B83 | DSA / DSV4 稀疏 prefill 核 |
| B84 | DSA cache 按 CP 切层 |

V4.1 拒绝 HiSparse 是 F197，归 [05-模型](../../05-模型/04-模型与特性矩阵.md)。

返回 [特性目录](../README.md)。
