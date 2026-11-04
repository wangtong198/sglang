# 请求接入

占位。HTTP 体变成内部请求，分词，再下发到 scheduler。本页不写字段表，字段在 [附录 06](../../附录-选项索引/06-Client请求选项大全.md)。

核对：[../../调研笔记/00-全量知识点清单.md](../../调研笔记/00-全量知识点清单.md) 的 A.10、A.13，以及 A.1 的 F10、F11。

| 范围 | ID | 清单节 | 这一节要讲的事 |
| --- | --- | --- | --- |
| 粗阶段 | F10、F11 | A.1 | S7 HTTP 体变成 `GenerateReqInput`；S8 分词并 ZMQ 下发 |
| 请求对象与入口 | F140–F149 | A.10 | `/generate` 与其它协议入口、`Req` / `ScheduleBatch` / `ForwardBatch`、`SamplingParams` |
| Tokenizer 与模板 | F173–F178、F207 | A.13，A.16 | `TokenizerManager`、`TemplateManager`、多模态处理器、auto parser。F207 是 Python chat 模板替换 |

A.10 末尾列出的其它 HTTP 入口（chat、embedding、Anthropic、Ollama 等）随 F140–F149 一起写，不另开目录。

返回 [流程总览](../00-总览.md)。
