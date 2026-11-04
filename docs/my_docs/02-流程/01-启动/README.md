# 启动

占位。从 CLI 到 scheduler ready。下一步按清单展开的就是这一节。本页不写 hook 正文，也不贴调用栈。

核对：[../../调研笔记/00-全量知识点清单.md](../../调研笔记/00-全量知识点清单.md) 的 A.1（F01–F09）、A.2–A.9、A.15 中下表这些问。

| 范围 | ID | 清单节 | 这一节要讲的事 |
| --- | --- | --- | --- |
| 粗阶段 | F01–F09 | A.1 | S0 到 ready，以及 S4 上 tokenizer 进程的建立 |
| CLI 与分叉 | F15–F24 | A.2 | `serve`、插件、HTTP / encoder / gRPC / Ray，扩散另起一条 |
| S1 声明 | F25–F30 | A.3 | `ServerArgs` 怎么来，封口前还没有改写 |
| S2 hook 全序 | F31–F91 | A.4 | `run_resolution_pipeline` 的次序。以清单本表和 `pipeline.py` 为准 |
| 进程拓扑 | F92–F104 | A.5 | `Engine._launch_subprocesses` |
| 分布式 init 插在哪 | F105–F110 | A.6 | 发生在 `Scheduler.__init__`、早于 `init_model_worker`。轴的定义归 D01–D20 |
| 复审补步 | F205、F206 | A.16 | CP-TP 组共享落在 S2 的 context parallelism；草稿架构改名落在 `ModelConfig` |
| 模型与权重 | F111–F125 | A.7 | `init_model_worker` 与 `ModelRunner.initialize` 的次序 |
| KV 与前缀树 | F126–F135 | A.8 | 池和树不是同一步 |
| ready | F136–F139 | A.9 | pipe、`wait_for_ready`、健康检查 |

A.15 归本目录的问（只列问题，不回答）：

| ID | 问题 |
| --- | --- |
| F189 | 为何 tokenizer、scheduler、detokenizer 拆进程 |
| F190 | 为何 `resolve_once` 只能成功一次 |
| F191 | 为何 S1 的默认值不能拿来解释运行时 |
| F192 | 为何 attention DP 不增加 GPU，却增加 scheduler |
| F194 | 为何草稿和目标各有一块池 |
| F195 | 为何 CUDA graph 在 KV 池之后、树之前或穿插 post-capture resize |
| F196 | 为何 `pp_size>1` 强制关掉 overlap schedule |
| F199 | 为何 `node_rank>=1` 不提供业务 HTTP |
| F202 | 为何 chunked prefill 在 attention DP 下会被改小 |
| F204 | 为何 Rust server 跳过 Python tokenizer |

平台分叉在 F44、F61、F73、F87，它们落在 F31–F91 里，仍归本目录。场景行回填到 [场景与硬件前提总表](../../01-体系总览/03-场景与硬件前提总表.md)。

返回 [流程总览](../00-总览.md)。
