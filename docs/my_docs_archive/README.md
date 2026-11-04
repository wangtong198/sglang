# 归档：分层之前的扁平笔记

这里是 `docs/my_docs/` 改成五层目录之前的正文和生成材料。文件还在，只是退出主线。现行结构、写作门和展开顺序见 [../my_docs/README.md](../my_docs/README.md)。

主线仍留在 `docs/my_docs/` 的三份是：`README.md`、`00-规划与学习路径.md`、`调研笔记/00-全量知识点清单.md`。`05` / `06` 没有留在本目录，它们作为附录放在 [../my_docs/附录-选项索引](../my_docs/附录-选项索引/README.md)。

归档篇之间的相对链接按旧的扁平布局保留。清单里「02 已写」一类话指的是本目录里的旧篇。

## 迁入本目录的路径

| 原路径 | 现路径 |
| --- | --- |
| `docs/my_docs/01-概览：作用与竞品对比.md` | `01-概览：作用与竞品对比.md` |
| `docs/my_docs/02-启动到HTTP请求完整流程.md` | `02-启动到HTTP请求完整流程.md` |
| `docs/my_docs/03-模块地图.md` | `03-模块地图.md` |
| `docs/my_docs/04-特性详解.md` | `04-特性详解.md` |
| `docs/my_docs/07-调度器与KV缓存.md` | `07-调度器与KV缓存.md` |
| `docs/my_docs/08-Radix前缀缓存.md` | `08-Radix前缀缓存.md` |
| `docs/my_docs/09-MoE路由与采样.md` | `09-MoE路由与采样.md` |
| `docs/my_docs/10-多模态.md` | `10-多模态.md` |
| `docs/my_docs/assets/deepseek-ai_DeepSeek-V4.1-Flash_config.json` | `assets/deepseek-ai_DeepSeek-V4.1-Flash_config.json` |
| `docs/my_docs/调研笔记/解析流水线.md` | `调研笔记/解析流水线.md` |
| `docs/my_docs/调研笔记/gen_option_docs.py` | `调研笔记/gen_option_docs.py` |
| `docs/my_docs/调研笔记/render_specific.py` | `调研笔记/render_specific.py` |
| `docs/my_docs/调研笔记/specific_zh.py` | `调研笔记/specific_zh.py` |
| `docs/my_docs/调研笔记/zh_parts/` | `调研笔记/zh_parts/` |

`解析流水线.md` 是更早的 S2 压缩表，漏了后来加进 `pipeline.py` 的 hook。对照以清单 A.4 为准。`render_specific.py` 和 `gen_option_docs.py` 仍把输出目录写成 `docs/my_docs/`（分层之前 05 和 06 的位置）。不要直接跑它们，否则会在主线根目录又写出这两份文件。现行附录在 `docs/my_docs/附录-选项索引/`。
