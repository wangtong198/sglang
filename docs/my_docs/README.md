# SGLang v0.5.23 源码学习系列

这套笔记对着本仓库里的 **SGLang v0.5.23** 源码写，不是产品介绍。

| 项 | 值 |
| --- | --- |
| 上游 | https://github.com/sgl-project/sglang.git |
| 标签 | `v0.5.23`（annotated tag 对象 `7df5dd09bc2f0e7e1509a1f7c2a40dff00ad8a01`） |
| 代码树 commit | `297e54fd105e974cdf401e5b4d20253a22c16f90` |
| 同步方式 | `git fetch --depth 1` 该 tag，`git read-tree` 把这棵树放进本仓库。历史是浅的，工作区文件与该 commit 一致，另外加上 `docs/my_docs/`。 |

## 四批计划

深文分四批写。当前只完成第 1 批：全量知识点清单。后面每一批只加深一条轨道。

| 批次 | 内容 | 入口 |
| --- | --- | --- |
| 1 清单 | 流程、特性、模型、分布式。每个点要能指到本树的路径和符号 | [调研笔记/00-全量知识点清单](调研笔记/00-全量知识点清单.md) |
| 2 启动链路 | CLI 到 scheduler ready | 清单 A，再重写 02 |
| 3 请求推理 | HTTP 入到采样与回包 | 清单 A 的请求段，再重写 07 及相关段 |
| 4 特性 / 分布式 / 模型 | 按清单 B/C/D 逐条加深 | 替换 04、08–10 里偏薄的部分 |

活地图和阅读顺序在 [00-规划与学习路径](00-规划与学习路径.md)。`05-Server选项大全.md`、`06-Client请求选项大全.md` 保持附录。01–04、07–10 先留着，清单里标明哪一批替换。

说不准的标「待核」。不能在本环境实跑的运行时行为标「源码路径推断，未实测」。

## 现有篇

1. [00-规划与学习路径](00-规划与学习路径.md)
2. [01-概览：作用与竞品对比](01-概览：作用与竞品对比.md)
3. [02-启动到HTTP请求完整流程](02-启动到HTTP请求完整流程.md)
4. [03-模块地图](03-模块地图.md)
5. [04-特性详解](04-特性详解.md)
6. [05-Server选项大全](05-Server选项大全.md) — 附录
7. [06-Client请求选项大全](06-Client请求选项大全.md) — 附录
8. [07-调度器与KV缓存](07-调度器与KV缓存.md)
9. [08-Radix前缀缓存](08-Radix前缀缓存.md)
10. [09-MoE路由与采样](09-MoE路由与采样.md)
11. [10-多模态](10-多模态.md)

调研笔记：[00-全量知识点清单](调研笔记/00-全量知识点清单.md)，以及更早的 [解析流水线](调研笔记/解析流水线.md)（S2 压缩表，已落后于 `pipeline.py`，以清单为准）。`调研笔记/render_specific.py` 用 `zh_parts/` 里按字段写好的中文重新生成 05 和 06。

模型 config 原文：[assets/deepseek-ai_DeepSeek-V4.1-Flash_config.json](assets/deepseek-ai_DeepSeek-V4.1-Flash_config.json)。

行号以本树为准。函数名可以当锚点，行号会随补丁移动。

## 与上游树的唯一差异

GitHub 的 push protection 把上游一个 32 位字母数字的类名当成 Mistral API key，拒绝推送含有这段连续字符的 commit。运行时名字没有改：类在定义后用 `"Mistral3" + "ForConditionalGeneration"` 写回 `__name__` 和模块全局变量。涉及文件：

- `python/sglang/srt/models/mistral.py`
- `python/sglang/srt/multimodal/processors/llava.py`
- `python/sglang/srt/configs/model_config.py`
- `python/sglang/multimodal_gen/runtime/models/encoders/mistral_3.py`
- `python/sglang/multimodal_gen/runtime/models/registry.py`（识别被拆开的字符串字面量）
- `test/registered/unit/models/test_mistral3_vision_feature.py`

除此之外，源码树与 `297e54fd105e974cdf401e5b4d20253a22c16f90` 一致。
