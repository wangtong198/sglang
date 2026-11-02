#!/usr/bin/env python3
"""旧的生成器。现在请用同目录的 render_specific.py。

它读取 zh_parts/*.json 里按字段写好的中文，缺任何字段都会失败，
避免再把命名空间说明抄进每一行。
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

ROOT = Path("/workspace")
OUT = ROOT / "docs" / "my_docs"
SPEC = importlib.util.spec_from_file_location(
    "catalog", "/tmp/gen_sglang_catalogs.py"
)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)

NS_TITLE = catalog.NS_TITLE

NS_ESSAY = {
    "device": "设备命名空间决定进程看见的加速器。它不改变 TP 宽度。选错设备会在 S2 的硬件校验失败，而不是在第一条请求才报错。HIP 机器上的设备字符串常常仍是 cuda，真正的 ROCm 分支看的是平台对象的 is_hip。",
    "disagg": "PD 分离在 S2 定模式，在 S9 换事件循环。prefill 与 decode 是两套进程，用 bootstrap 字段配对。多节点模型副本 DP 仍然禁止。V4.1 加 DSPARK 时传输后端还要是 mooncake 或 nixl，并且 ragged verify 为 static。",
    "exec.features": "这里是横切开关：精度、返回值、实验路径。多数不改变进程数。和 V4.1 冲突的（two-batch overlap、非 DSPARK 投机）在模型特征检查里拒绝，不一定在本字段的声明处。",
    "exec.kernel": "Attention、GEMM、采样 kernel 的选择。声明值 auto 会在 S2 被模型 override 改掉。V4.1-Flash 的 attention 会被改成 dsv4，page_size 改成 256。不要把 auto 当成运行时后端。",
    "exec.mamba": "线性注意力 / Mamba 的状态槽，和 Transformer KV 页分开记账，见 ReqKvInfo 的 mamba 字段。V4.1-Flash 的主体不是 Mamba，这些开关对它通常不起作用。",
    "exec.graph": "CUDA graph 分 decode 与 prefill 两相，在 S5 末尾捕获。DeepEP normal、DeepEP v2 的 prefill、DWDP 会把对应相关掉。废弃旗标 --disable-cuda-graph 只是别名。",
    "exec.comm": "集合通信融合与 NVTX。它们不改变 ep_size。Aiter all-reduce 融合和 prefill CP 互斥。",
    "exec.moe": "MoE 通信与专家 GEMM。spanning all-to-all（deepep、deepep_v2、mori、megamoe 等）会把 ep_size 强制改成 tp_size。DSPARK 加 attention DP 时白名单只有 none、megamoe、mori。deepep_v2 的 direct 不能跨节点。",
    "exec.overlap": "微批重叠。pp_size>1 会强制关掉 overlap schedule。V4.1 禁止 two-batch overlap，并且禁止 PP。",
    "exec.offload": "把一部分权重或 KV 放到 CPU。会挤压 S6 的 KV 预算。和统一内存池、某些 PD 传输后端互相检查。",
    "exec.dllm": "掩码扩散语言模型，仍走 Scheduler，ForwardMode 是 DLLM_EXTEND。不是 multimodal_gen 的图像扩散。",
    "exec.deterministic": "批不变算子，用来复现采样。会限制可用的 attention 后端。和投机、某些图捕获可能冲突，以 S2 的确定性 hook 为准。",
    "lora": "S5 装载适配器，S8 按请求选择。提供 lora 路径时 enable_lora 会被打开。部分 MoE runner 会因此关掉 shared expert 融合。attention DP 与个别 LoRA kernel 不兼容。",
    "memory": "S6 的预算、页、前缀树和 HiCache。mem_fraction_static 决定权重加 KV 池占显存的比例，调小是为了给激活留地方。disable_radix_cache 会让 LPM 策略失效。V4 的 page_size 会被模型 override 改成 256。",
    "mm": "VLM 的处理器开关，在 S2、S4、S8 起作用。对非多模态模型，enable_multimodal 是空操作。mm_enable_dp_encoder 在 tp=1 时没有并行度。encoder 分离是另一条进程，见 disagg。",
    "model": "权重从哪来、以什么格式加载。S1 只记录路径，S2 的 handle_model_source_paths 和 expert_pack 可能改写本地目录。量化方法要和 config.json 的 quantization_config 一致，否则 loader 在 S5 失败。",
    "observability": "日志、指标、trace、崩溃转储。不改变数值，但 crash dump 相关环境变量必须在 CUDA 初始化之前设好，所以放在 S2 后段。",
    "parallel": "全部并行宽度。恒等式 attn_tp=tp/attn_dp/attn_cp，moe_tp=tp/ep/moe_dp。多节点禁止 dp_size>1。V4.1 禁止 PP。HIP 上的 V4.1 禁止 prefill CP。DCP 与 prefill CP 互斥。旧的 enable_dp_attention 会把 dp_size 收成 attn_dp_size。",
    "schedule": "S9 的预算和策略。attention DP 会把 schedule_conservativeness 乘 0.3，并把 chunked_prefill_size 除以组数。这些是改写，不是你在命令行里看到的声明默认。",
    "serving": "S4 与 S7：地址、解析器、tokenizer 批、API key、响应存储。reasoning 与 tool 解析器为 auto 时在 init_tokenizer_manager 按模板选定。V4.1 的 reasoning 名是 deepseek-v41。",
    "spec": "S2 决定算法和草稿路径，S5 建 draft worker。V4.1 只允许 DSPARK，步数会被收成 1。NEXTN 在通用路径折成 EAGLE，对 V4.1 仍然非法。DP attention 下必须 enable_dp_lm_head，且 a2a 只能是 none、megamoe 或 mori。",
}

# Full overrides: four-part Chinese, no English help dump.
CUSTOM = {
    "tp_size": "作用：张量并行宽度，也是单副本的 GPU 数因子之一（再乘 pp 与模型副本 dp）。默认 1，单卡，不做切分。交互：S3 用它起进程；(tp*pp) 必须整除 nnodes。attention 与 MoE 的子宽度必须乘回这个数。spanning all-to-all 会把 ep_size 改成它。易错：把它当成「只切 attention」。KV 头为 1 的 V4.1 不应靠把 tp 拉到 16 来切 attention，应配 attn_dp_size。",
    "pp_size": "作用：流水并行段数，增加 GPU。默认 1，不切层。交互：大于 1 时 S2 强制关掉 overlap schedule。V4.1 的特征检查直接拒绝 PP。moe_dp>1、DWDP、elastic EP 也要求它为 1。易错：用 PP 把 16 卡分成两段来跑 V4.1-Flash，进程会在 S2 退出。",
    "dp_size": "作用：整模副本数，GPU 数再乘一份。默认 1。交互：与 attn_dp_size 不能同时大于 1。nnodes 不是 1 时 check_server_args 直接 assert，提示改用 attn_dp_size。易错：四机各写 --dp-size 4 --tp 4。那是四个副本，而这棵树不允许副本跨节点。",
    "attn_dp_size": "作用：在已有 TP 组里划出多少个 attention 数据并行组，每组一个 scheduler，FFN/MoE 仍可 TP 或 EP。默认 1，不划分。不增加 GPU。交互：tp 必须能被它整除。大于 1 时保守系数乘 0.3，chunked prefill 除以它，ZMQ 改用 TCP。旧旗标 --dp-size N --enable-dp-attention 会被收成 attn_dp_size=N 且 dp_size=1。易错：和 dp_size 同时放大；或在 HIP 上再开 encoder SWA bounded replay。",
    "enable_dp_attention": "作用：废弃的内部标记，没有稳定 CLI。默认 False。交互：DeprecatedStoreTrueAction 把旧写法收成 attn_dp_size。解析结束后对外报告的 enable_dp_attention 表示 attn_dp_size>1，供旧客户端读 /server_info。易错：在新配置里继续写这个旗标，并以为 dp_size 仍是副本数。",
    "ep_size": "作用：专家并行宽度，不增加 GPU，rank 仍在 TP 组里。默认 1，专家不按 EP 切开。交互：mori、deepep、deepep_v2、megamoe 等 spanning 后端在 S2 把它改成 tp_size，并打日志。与 moe_dp_size 相乘必须回到 tp。易错：手写一个较小的 ep，又开了 mori，运行值会被改掉。",
    "moe_dp_size": "作用：MoE 的数据并行宽度。默认 1。交互：大于 1 时要求 pp 为 1；若 ep 也大于 1，则 ep 乘 moe_dp 必须等于 tp。attn_cp 与它不相等时，只允许 moe_dp 为 1。易错：把它当成 attn_dp_size。两者切的不是同一组算子。",
    "dcp_size": "作用：decode 上下文并行，在每个 TP 组内切 decode 的 KV。默认 1。不增加 GPU。交互：大于 1 不能同时开 prefill CP。fi_a2a 只允许 CUDA。必须整除 attention TP。易错：和 --enable-prefill-cp 一起开，S2 的 handle_decode_context_parallelism 会拒绝。",
    "attn_cp_size": "作用：attention 上下文并行宽度。默认 1。交互：prefill CP 打开时，DeepSeek-V4 的校验会把它改写成 tp/attn_dp。HIP 上的 V4.1 和多节点在改写前就会失败。易错：以为自己设成 2 就会生效。模型 hook 可能把它改成铺满剩余 TP。",
    "enable_prefill_cp": "作用：预填充阶段按序列切分上下文，也就是笔记里的 PCP。默认关。交互：必须同时给 cp_strategy（interleave 或 zigzag）。与 dcp_size>1 互斥。V4 zigzag 只允许 NPU。HIP 上 V4.1、多节点、tp>8 都拒绝。易错：在四机 BW1100 上用它来拉长上下文。",
    "cp_strategy": "作用：预填充 CP 的切分方式，zigzag 或 interleave。默认空。交互：enable_prefill_cp 为真时必填，否则 ValueError。V4 在 CUDA/HIP 上只能 interleave。易错：只写 enable，不写策略。",
    "nnodes": "作用：节点数。默认 1。交互：与 node_rank、dist_init_addr 一起在 S3 建进程组。(tp*pp) 必须整除它。dp_size>1 只允许它为 1。易错：四台机器都写 nnodes=1，各自成群。",
    "node_rank": "作用：本进程所在节点的序号，从 0 开始。默认 0。交互：大于 0 的节点不起 HTTP，只跑 scheduler 直到 ready。易错：每台机器都写 0，端口冲突，进程组建不起来。",
    "dist_init_addr": "作用：分布式初始化的主机地址，形如 host:port。别名 --nccl-init-addr。默认空。交互：多节点时所有 rank 必须能连到 rank 0 上的这个地址。易错：写成每台机器自己的主机名。",
    "nccl_port": "作用：NCCL 环境的端口。默认空，PortArgs 选一个临时端口以下的空闲端口。交互：多机防火墙要放行。它不决定 TP 宽度。易错：随机端口被安全组丢掉，卡在 init_distributed_environment。",
    "moe_a2a_backend": "作用：MoE token 的 all-to-all 实现。默认 none，token 不跨 rank 搬家，专家按 TP 切。交互：deepep、deepep_v2、mori、megamoe 等会强制 ep=tp。deepep normal 关 CUDA graph。deepep_v2 direct 禁止多节点。DSPARK 加 attention DP 时只能 none、megamoe、mori。易错：四机 V4.1 写 deepep。应写 mori。",
    "deepep_mode": "作用：DeepEP v1 的调度模式，auto、normal 或 low_latency。默认 auto。交互：normal 关掉两相 CUDA graph。CuteDSL FP4 会强制 low_latency，normal 直接报错。它不控制 deepep_v2。易错：以为 auto 在 mori 后端也有意义。mori 不读这个字段。",
    "deepep_v2_mode": "作用：DeepEP v2 的模式，direct 或 hybrid。与 v1 的 deepep_mode 无关。交互：direct 在 nnodes>1 时 ValueError。v2 还限制 runner 为 deep_gemm，并关 prefill 图和 shared expert 融合。易错：跨节点仍写 direct。",
    "enable_dp_lm_head": "作用：attention DP 下 LM head 走数据并行。默认关。交互：必须已经打开 attention DP。DSPARK 在 DP attention 下强制它。与 enable_tp_lm_head_all_to_all 互斥。易错：开了 DSPARK 和 attn_dp_size 却忘了它，_handle_dspark 会拒绝启动。",
    "speculative_algorithm": "作用：投机算法名。默认空，不建 draft worker。内建 EAGLE、EAGLE3、NEXTN、STANDALONE、NGRAM、DFLASH、DSPARK、UNO。交互：V4.1 只留下 DSPARK。NEXTN 会先被折成 EAGLE，然后仍被 V4.1 拒绝。易错：把 config 里的 num_nextn_predict_layers 理解成还要再开 NEXTN。",
    "speculative_dspark_block_size": "作用：DSPARK 一次草稿块的大小。默认由解析填。V4.1-Flash 的 config 是 5。交互：应与 checkpoint 的 dspark_block_size 一致。易错：沿用别的模型的 block。",
    "speculative_num_steps": "作用：草稿迭代步数。默认空。交互：DSPARK 会把它收成 1，大于 1 只打警告并改写。易错：把它设得很大指望接受长度变长。DSPARK 不吃这个自由度。",
    "speculative_draft_model_path": "作用：草稿权重路径。默认空。交互：DSPARK 若发现主 checkpoint 自带草稿，会把它改成 model_path 或子目录；否则 ValueError，必须自己传。易错：V4.1 不传路径，也不确认权重里有没有草稿子目录。",
    "chunked_prefill_size": "作用：一轮预填充最多吃多少 token。-1 表示关闭，整段 extend。交互：attention DP 时 S2 会除以 attn_dp_size。V4 走 trtllm attention 时要求它大于 0。易错：开了 attn_dp_size 4 仍按未除过的块估计显存。",
    "schedule_conservativeness": "作用：调度器估计新 decode 还能塞多少的系数，越大越保守。交互：attention DP 打开时乘 0.3。易错：在命令行设了 1，运行中却发现更激进，因为改写发生在 S2。",
    "max_running_requests": "作用：同时处于 running 的请求上限。交互：V4 模型 hook 在未设置时可能改成 256。它受 KV 池容量约束，设得再大也不会超过 S6 算出来的槽。易错：把它当成吞吐旋钮，显存已经满了仍继续加大。",
    "mem_fraction_static": "作用：静态占用（权重加 KV 池）占显存的比例。交互：S6 的 _profile_available_bytes 用它留出激活。OOM 时调小。Engram 若在 GPU 上，会和这个预算抢。易错：V4.1 的 Engram 表没放 host，却把比例设得很高。",
    "page_size": "作用：KV 页的 token 数。交互：V4 override 把它改成 256（NPU 128）。Radix 命中按页对齐。易错：手写 16，以为能提高短前缀命中率。S2 会改掉。",
    "attention_backend": "作用：注意力 kernel。默认往往是 auto。交互：V4.1 被改成 dsv4。HIP 上的 prefill CP 还要求 prefill 与 decode 都是 dsv4。易错：指定 triton 或 flashinfer 来「加速」V4.1，覆盖了模型要求的后端。",
    "disable_radix_cache": "作用：关掉前缀树。默认开着树。交互：关掉后 SchedulePolicy 会把 LPM 换成不看缓存的策略，每次请求从零写 KV。易错：排查命中率时关掉后忘记打开，延迟整体变差。",
    "radix_eviction_policy": "作用：树的驱逐策略。lru 最近最少使用，lfu 最不经常使用，slru 分段 LRU，priority 先驱逐低优先级，tlru 带尾部的 LRU。交互：disable 时内部仍用 lru 且不记录。priority 要请求带 priority。易错：策略名写错，不在 choices 里，S1 就 argparse 失败。",
    "enable_hicache": "作用：给前缀树接分层 host/L3。默认关。交互：hicache_hook 与某类 unified external linker 互斥。它不替代 Radix 的匹配规则。易错：把它当成扩大 GPU KV 的开关。GPU 页数仍由 mem_fraction_static 决定，HiCache 是换出之后的下一层。",
    "disaggregation_mode": "作用：PD 角色。null 是合部，prefill 只做提示，decode 只做解码。默认 null。交互：S9 换循环。decode 侧用 ForwardMode.PREBUILT。V4.1 加 DSPARK 时还有传输后端和 ragged 限制。易错：两台机器都写 prefill，没有 decode。",
    "model_path": "作用：权重目录或仓库 id。S0 的位置参数会变成它。交互：none 或 dummy 时 S2 提前返回，不下载。expert_pack 可能把它换成生成的本地元数据。易错：路径指向只有 safetensors、没有 config 的目录，S5 的 get_config 失败。",
    "trust_remote_code": "作用：允许加载仓库里的自定义建模代码。默认关。V4.1 的处理器和配置经常需要它。交互：只影响 S5 的加载，不改变并行。易错：关着它加载 DeepSeek 权重，tokenizer 或 config 类找不到。",
    "reasoning_parser": "作用：把思考段从正文里拆出来。choices 含 auto 和 reasoning_parser_names。默认按字段声明。交互：auto 在 S4 按 chat template 写成具体名，V4.1 对应 deepseek-v41。易错：手写 deepseek-r1 去解析 V4.1 的模板。",
    "tool_call_parser": "作用：工具调用的解析器。auto 在 S4 选定。V4.1 有 DeepSeekV41Detector。交互：和 chat template 必须配套，sampler 不负责拆工具。易错：只开 parser、模板却不是工具模板，调用边界对不齐。",
    "enable_two_batch_overlap": "作用：两个微批重叠。默认关。交互：V4.1 特征检查拒绝。PP 也会关 overlap schedule，但是另一个字段。易错：从旧的 DeepSeek-V3 启动脚本抄过来。",
    "language_model_only": "作用：V4.1 等多模态架构只建文本塔，不建 ViT。默认关，会建视觉塔（若 config 有视觉层）。交互：S5 的 DeepseekV4ForCausalLM 构造函数看它。易错：纯文本服务却没开，144GB 上多占一份 ViT。",
    "kv_cache_dtype": "作用：KV 存储精度。交互：和 attention 后端、量化一起在 S2 检查。fp8 KV 配错后端会在 S6 或第一次前向失败。易错：为了省显存改 fp8，但 dsv4 池不支持该组合。",
    "load_format": "作用：权重容器格式。auto 会按文件猜。choices 含 safetensors、gguf、remote 等。交互：expert_pack 是一条窄路径，只验证过特定 GGUF。易错：对 fp8 safetensors 手写 gguf。",
    "quantization": "作用：在线或加载时的量化方法。交互：应与 config 的 quant_method 一致。V4.1-Flash 是 fp8 加专家 fp4。乱指定 awq 会在 S5 失败。易错：把 expert_dtype=fp4 理解成还要再写 --quantization mxfp4。",
    "enable_lora": "作用：打开 LoRA。若提供了 lora 路径，S2 会把它设为真以保持兼容。交互：检查 max_loras_per_batch。部分 MoE 融合会被关掉。易错：路径给了但 batch 上限是 0。",
    "cuda_graph_config": "作用：按 JSON 指定 decode/prefill 两相的图后端和 batch 上限。后端可以是 full、breakable、disabled。JSON 优先于分相的便捷旗标。交互：DeepEP normal 仍会把两相关掉，JSON 写了 full 也会被改。易错：只改 JSON，没看 S2 日志里的覆盖。",
    "schedule_policy": "作用：waiting 队列的排序。lpm 等看前缀树，fcfs 等不看。交互：disable_radix_cache 时看树的策略会被换掉。易错：关了 radix 仍期待 lpm 生效。",
    "enable_metrics": "作用：打开 Prometheus 风格指标。默认按字段。交互：不影响数值。/server_info 里的 internal_states 是另一条状态，DSPARK 测试用它看接受长度。易错：只开 metrics 却去 /server_info 找直方图。",
    "host": "作用：HTTP 绑定地址。默认常见是 127.0.0.1。交互：0.0.0.0 时内部自检改用 127.0.0.1。多机 rank 0 要能被其它节点访问的是 dist_init_addr，不是这个 host。易错：把 host 当成 NCCL 的地址。",
    "port": "作用：HTTP 端口。默认 30000。交互：node_rank>0 不起这个端口。易错：四台机器都去 curl 本机 30000，只有 rank 0 有服务。",
    "skip_tokenizer_init": "作用：不在主进程建分词器。scheduler 把 token id 直接送回。交互：调用方必须自己传 input_ids，S11 的文本路径被跳过。易错：仍用 /v1/chat/completions 传文本。",
    "context_length": "作用：覆盖模型的最大上下文。默认用 config。V4.1-Flash 是 1048576。交互：超过此长度的请求在 S8 被拒。它不增加 KV 池，池仍由显存决定。易错：把 context 改到一百万，却发现并发只有个位数，因为 144GB 放不下那么多页。",
    "dwdp_size": "作用：分布式权重数据并行。大于 1 时 MoE 预填充用权重预取代替 token all-to-all。交互：必须等于 tp，强制纯 DP attention、ep=tp、a2a none、无 PP、无投机、无 CUDA graph。易错：和 DSPARK 一起开。",
    "enable_elastic_ep": "作用：运行中扩 EP。交互：要求 attention DP、dp lm head、ep==tp、a2a 为 nixl、无 CP、无 PP。和静态 mori 部署不是一条路径。易错：在四机 mori 上打开它指望热扩容。",
    "moe_runner_backend": "作用：专家 GEMM 的 kernel。auto 会按量化改写。交互：LoRA runner 和若干 flashinfer runner 会关 shared expert 融合。V4.1 的 MXFP8 dense 快路径不覆盖 HIP。易错：在海光上指定 flashinfer_cutedsl。",
    "prefill_attention_backend": "作用：只覆盖预填充相的 attention。默认跟主后端。交互：HIP 上 V4 prefill CP 要求它和 decode 都是 dsv4。易错：预填充用一个后端、解码用另一个，触发 V4 的双相检查。",
    "decode_attention_backend": "作用：只覆盖解码相的 attention。约束同 prefill 那一项。易错：同上。",
    "enable_encoder_swa_bounded_replay": "作用：V4.1 encoder 的滑动窗口有界重放。默认关。交互：只适用于 V4.1。HIP 上不能和 attention DP、prefill CUDA graph、CP、LoRA、session radix、PD 同时开。易错：在四机 DP attention 配置上打开它。",
    "max_prefill_tokens": "作用：一批预填充 token 的上限，和 chunk 一起限制 S9 第 2 步。交互：受 KV 剩余页数约束。易错：设得比 chunked_prefill_size 还小，chunk 形同虚设。",
    "api_key": "作用：若设置，HTTP 要求客户端带同样的密钥。默认空，不校验。交互：只影响 S7，不进 scheduler。易错：把密钥写进进程列表。05 不会重复密钥的值。",
    "served_model_name": "作用：/v1/models 和响应里展示的名字，可以和路径不同。默认用路径推断。交互：LoRA 可以用 base:adapter 这种名字。易错：客户端 model 字段必须对上这个名字，对不上会 404。",
    "revision": "作用：Hugging Face 或魔搭风格仓库的版本。默认主分支。交互：和 model_path 一起在 S5 下载。易错：revision 不存在，卡在下载而不是断言。",
    "download_dir": "作用：权重缓存目录。默认用库的默认缓存。交互：多机若目录不共享，每台都要能读到同一份权重。易错：只在 rank 0 下载。",
}


PHRASES = [
    ("Defaults to a random port", "不填则选一个空闲端口"),
    ("Defaults to auto-detection if not specified", "不填则自动探测"),
    ("Disabled by default", "默认关闭"),
    ("defaults to", "默认是"),
    ("Default", "默认"),
    ("If set", "若打开"),
    ("If not set", "若不设置"),
    ("must be", "必须是"),
    ("must equal", "必须等于"),
    ("only supported", "仅支持"),
    ("not supported", "不支持"),
    ("is required", "是必需的"),
    ("Enable ", "打开"),
    ("Disable ", "关闭"),
    ("Whether ", "是否"),
    ("The number of ", ""),
    ("The maximum ", "最大"),
    ("The minimum ", "最小"),
    ("The path ", "路径："),
    ("The port ", "端口："),
    ("The host ", "主机："),
    ("Choose the ", "选择"),
    ("Specify the ", "指定"),
    ("speculative decoding", "投机解码"),
    ("speculative", "投机"),
    ("data parallel", "数据并行"),
    ("tensor parallel", "张量并行"),
    ("pipeline parallel", "流水并行"),
    ("context parallel", "上下文并行"),
    ("expert parallel", "专家并行"),
    ("attention", "注意力"),
    ("prefill", "预填充"),
    ("decode", "解码"),
    ("KV cache", "KV 缓存"),
    ("kv cache", "KV 缓存"),
    ("radix", "前缀树"),
    ("chunked prefill", "分块预填充"),
    ("cuda graph", "CUDA 图"),
    ("CUDA graph", "CUDA 图"),
    ("weights", "权重"),
    ("checkpoint", "检查点"),
    ("tokenizer", "分词器"),
    ("requests", "请求"),
    ("request", "请求"),
    ("backend", "后端"),
    ("memory", "显存"),
    ("batch", "批"),
    ("scheduler", "调度器"),
    ("model", "模型"),
    ("draft", "草稿"),
    ("target", "目标模型"),
    ("overlap", "重叠"),
    ("offload", "换出到 CPU"),
    ("quantization", "量化"),
    ("disaggregation", "PD 分离"),
    ("LoRA", "LoRA"),
    ("grammar", "语法约束"),
    ("logprobs", "对数概率"),
    ("streaming", "流式"),
    ("timeout", "超时"),
    ("worker", "工作进程"),
    ("rank", "rank"),
    ("node", "节点"),
    ("server", "服务"),
    ("cache", "缓存"),
    ("pool", "池"),
    ("page", "页"),
    ("token", "token"),
    ("layer", "层"),
    ("kernel", "kernel"),
    ("group", "组"),
    ("size", "宽度"),
    ("path", "路径"),
    ("file", "文件"),
    ("json", "JSON"),
    ("JSON", "JSON"),
    ("GPU", "GPU"),
    ("CPU", "CPU"),
    ("HTTP", "HTTP"),
    ("and ", "并且"),
    ("with ", "与"),
    ("when ", "当"),
    ("for ", "用于"),
    ("from ", "来自"),
    ("into ", "到"),
    ("only ", "仅"),
    ("not ", "不"),
    ("the ", ""),
    ("The ", ""),
    ("a ", ""),
    ("an ", ""),
    ("of ", "的"),
    ("to ", "到"),
    ("in ", "在"),
    ("on ", "在"),
    ("is ", "是"),
    ("are ", "是"),
    ("use ", "使用"),
    ("Use ", "使用"),
    ("set ", "设置"),
    ("Set ", "设置"),
    ("this ", "该"),
    ("that ", "该"),
    ("which ", "该"),
    ("will ", "会"),
    ("can ", "可以"),
]


KEEP = {
    "cuda", "gpu", "cpu", "npu", "hpu", "xpu", "json", "http", "https", "zmq",
    "lora", "moe", "fp8", "fp4", "fp32", "bf16", "api", "ssl", "tls", "nvtx",
    "nccl", "yaml", "url", "ipc", "tcp", "udp", "radix", "hicache", "deepep",
    "mori", "dspark", "eagle", "gguf", "safetensors", "prometheus", "openai",
    "anthropic", "ollama", "grpc", "rdma", "nvlink", "sm100", "hip", "rocm",
    "triton", "flashinfer", "cutlass", "marlin", "awq", "gptq", "bitsandbytes",
    "lm", "kv", "tp", "pp", "dp", "ep", "cp", "dcp", "sse", "id", "ids",
}


def _tr_piece(s: str) -> str:
    out = s
    for src, dst in PHRASES:
        out = out.replace(src, dst)
    return re.sub(r"\s+", " ", out).strip(" .,;")


def english_left(s: str) -> int:
    n = 0
    for w in re.findall(r"[A-Za-z]{4,}", s):
        if w.lower() not in KEEP:
            n += 1
    return n


def gloss(text: str) -> str:
    """Prefer a Chinese sentence. Drop a half-translated English clause."""
    if not text:
        return ""
    raw = " ".join(text.split())
    patterns = [
        (r"^If set, (.+)$", "打开之后，{0}。"),
        (r"^Whether (?:to )?(.+)$", "是否{0}。"),
        (r"^Enable (.+)$", "打开{0}。"),
        (r"^Disable (.+)$", "关闭{0}。"),
        (r"^The (.+)$", "{0}。"),
        (r"^Choose (.+)$", "选择{0}。"),
        (r"^Specify (.+)$", "指定{0}。"),
        (r"^Set (.+)$", "设置{0}。"),
        (r"^Use (.+)$", "使用{0}。"),
        (r"^Path (.+)$", "路径：{0}。"),
        (r"^Maximum (.+)$", "上限：{0}。"),
        (r"^Minimum (.+)$", "下限：{0}。"),
        (r"^Number of (.+)$", "数量：{0}。"),
        (r"^How many (.+)$", "数量：{0}。"),
        (r"^When (.+)$", "当{0}。"),
    ]
    body = raw.rstrip(".")
    for pat, tmpl in patterns:
        m = re.match(pat, body, flags=re.I)
        if not m:
            continue
        piece = _tr_piece(m.group(1))
        sent = tmpl.format(piece)
        if english_left(sent) == 0:
            return sent
        break
    piece = _tr_piece(body)
    if english_left(piece) == 0:
        return piece + "。"
    return ""


NAME_TOK = [
    ("enable_", "打开"),
    ("disable_", "关闭"),
    ("max_", "最大"),
    ("min_", "最小"),
    ("num_", "数量"),
    ("speculative_", "投机的"),
    ("chunked_prefill", "分块预填充"),
    ("prefill", "预填充"),
    ("decode", "解码"),
    ("attention", "注意力"),
    ("backend", "后端"),
    ("cache", "缓存"),
    ("memory", "显存"),
    ("cuda_graph", "CUDA 图"),
    ("graph", "图"),
    ("overlap", "重叠"),
    ("offload", "换出"),
    ("quant", "量化"),
    ("lora", "LoRA"),
    ("moe", "MoE"),
    ("expert", "专家"),
    ("draft", "草稿"),
    ("tokenizer", "分词器"),
    ("scheduler", "调度"),
    ("request", "请求"),
    ("batch", "批"),
    ("token", "token"),
    ("page", "页"),
    ("port", "端口"),
    ("host", "主机"),
    ("path", "路径"),
    ("model", "模型"),
    ("rank", "rank"),
    ("node", "节点"),
    ("size", "宽度"),
    ("policy", "策略"),
    ("timeout", "超时"),
    ("snapshot", "快照"),
    ("publish", "发布"),
    ("interval", "间隔"),
    ("backup", "备份"),
    ("weights", "权重"),
    ("weight", "权重"),
    ("remote", "远程"),
    ("instance", "实例"),
    ("loader", "加载器"),
    ("linear", "线性"),
    ("verify", "校验"),
    ("custom", "自定义"),
    ("all_reduce", "all-reduce"),
    ("metric", "指标"),
    ("log", "日志"),
    ("ssl", "TLS"),
    ("api", "API"),
    ("worker", "工作进程"),
    ("stream", "流"),
    ("hidden", "隐状态"),
    ("grammar", "语法"),
    ("tool", "工具"),
    ("reason", "推理过程"),
    ("dp", "数据并行"),
    ("tp", "张量并行"),
    ("pp", "流水并行"),
    ("ep", "专家并行"),
    ("cp", "上下文并行"),
    ("kv", "KV"),
    ("fp32", "FP32"),
    ("fp16", "FP16"),
    ("bf16", "BF16"),
    ("fp8", "FP8"),
    ("fp4", "FP4"),
    ("lm_head", "LM head"),
    ("head", "head"),
    ("logits", "logits"),
    ("gpu", "GPU"),
    ("cpu", "CPU"),
    ("npu", "NPU"),
]


def name_zh(name: str) -> str:
    raw = name
    parts = []
    s = name
    # greedy token scan
    used = []
    i = 0
    tokens = sorted(NAME_TOK, key=lambda x: -len(x[0]))
    while s:
        hit = False
        for tok, zh in tokens:
            if s.startswith(tok):
                used.append(zh)
                s = s[len(tok) :]
                s = s.lstrip("_")
                hit = True
                break
        if not hit:
            # take one snake piece
            piece, _, rest = s.partition("_")
            used.append(piece)
            s = rest
        i += 1
        if i > 20:
            break
    return " ".join(used) if used else raw


def default_effect(default: str, no_cli: bool, kind: str) -> str:
    if kind == "derived" or default == "（发布时计算）":
        return "它不是启动参数，在配置发布时由函数算出来。命令行写不了它。"
    if no_cli:
        return f"声明默认是 `{default}`。它没有常规 CLI，多半由别的旗标或解析阶段写入。"
    if default in ("None", "null", "（必填）"):
        return f"声明默认是 `{default}`。保持不填时，S2 可能用模型或平台填上一个具体值；没被填上就按「关闭或自动」处理。"
    if default in ("False", "false"):
        return "声明默认是关闭。不写这个旗标就不会走这条路径。"
    if default in ("True", "true"):
        return "声明默认是打开。要用反向旗标或显式 false 才会关掉。"
    if default in ("1",):
        return "声明默认是 1。并行宽度上的 1 表示这一维不切开；计数类字段则表示一个单位。是否会被 S2 改写，看本组说明和该字段是否标了可改写。"
    if default in ("-1",):
        return "声明默认是 -1。在预填充块大小这类字段上，-1 表示关闭分块。"
    if default in ("auto", "'auto'", '"auto"'):
        return "声明默认是 auto。S2 会按模型和平台改成具体后端，请看启动日志里的覆盖，而不是这个字符串。"
    return f"声明默认是 `{default}`。不在命令行覆盖它，就会带着这个值进入 S2；S2 仍可能改写。"


NS_PIT = {
    "device": "设备字符串和真实平台可能不一致。HIP 上经常仍写 cuda，ROCm 分支看 is_hip，不看这个字符串。",
    "disagg": "prefill 和 decode 要成对启动。只起一边，请求会停在 bootstrap。",
    "exec.features": "实验开关常常和某条模型特征检查冲突，失败点在 S2 后段，不一定在这个字段旁边。",
    "exec.kernel": "auto 会被模型改成具体 kernel。手写一个后端可能覆盖 V4 需要的 dsv4。",
    "exec.mamba": "V4.1-Flash 不是 Mamba 主体。打开这些开关通常没有收益，还可能在 S6 多占状态槽。",
    "exec.graph": "图捕获发生在权重和池之后。DeepEP normal、DWDP 会把图关掉，配置写了 full 也会被改。",
    "exec.comm": "融合通信不改变 ep 或 tp。和 CP、某些量化叠在一起会在 S2 被关掉或拒绝。",
    "exec.moe": "通信后端和 GEMM 后端是两件事。开了 spanning all-to-all 却手写小 ep，运行时会被改成 tp。",
    "exec.overlap": "V4.1 禁止 two-batch overlap。PP 会另关 schedule overlap。",
    "exec.offload": "换出省 GPU，但拉高延迟，并和 KV 预算抢 CPU 内存。",
    "exec.dllm": "这是掩码扩散语言模型，不是图像扩散。不要和 multimodal_gen 的参数混用。",
    "exec.deterministic": "确定性会限制 kernel。和投机同时开时，以 S2 的检查为准。",
    "lora": "给了路径却把每批上限留成 0，S2 会失败。MoE 融合可能被自动关掉。",
    "memory": "把比例或页大小调大不会变出显存。OOM 时先减小 mem_fraction_static，而不是加大 max 请求数。",
    "mm": "非多模态模型上打开多模态开关是空操作。tp=1 时复制 ViT 没有并行度。",
    "model": "路径、revision、量化三者不一致时，失败发生在 S5 加载，而不是 argparse。",
    "observability": "指标开关不改变数值。崩溃转储要在 CUDA 初始化前生效。",
    "parallel": "宽度必须满足整除和互斥。多节点不要用模型副本 dp_size。V4.1 不要开 PP 和 HIP 上的 prefill CP。",
    "schedule": "attention DP 会改写保守系数和分块大小。命令行上的数不一定是运行数。",
    "serving": "解析器要和 chat template 配套。auto 在 S4 才落地成具体名字。",
    "spec": "V4.1 只接受 DSPARK。草稿路径缺失会在 S2 直接退出。",
}


def pitfalls(f) -> str:
    bits = []
    if f.get("resolvable"):
        bits.append("该字段允许 S2 改写，以启动日志和 /server_info 为准。")
    if f.get("kind") == "derived":
        bits.append("派生量写进命令行不会生效。")
    bits.append(NS_PIT.get(f["ns"], "先看本组开头的约束，再改这个字段。"))
    return "".join(bits)


def explain_field(f) -> str:
    if f["name"] in CUSTOM:
        return CUSTOM[f["name"]]
    title = NS_TITLE.get(f["ns"], f["ns"])
    essay = NS_ESSAY.get(f["ns"], "")
    purpose_name = name_zh(f["name"])
    glossed = gloss(f.get("help") or "")
    if glossed:
        purpose = f"用来处理「{purpose_name}」。具体行为：{glossed}"
    else:
        purpose = (
            f"用来处理「{purpose_name}」。"
            "它参与这一组在 S2 的检查或在对应阶段被读取；"
            "没有单独的强制改写时，就保持声明值直到该阶段使用。"
        )
    if f.get("choices"):
        purpose += f" 可选值：{f['choices']}。"
    if f.get("aliases"):
        purpose += f" 别名：{f['aliases']}。"
    return (
        f"作用：属于「{title}」。{purpose} "
        f"默认与效果：{default_effect(f['default'], f.get('no_cli'), f.get('kind', 'cli'))} "
        f"交互与改写：{essay} "
        f"易错：{pitfalls(f)}"
    )


def md_cell(text: str) -> str:
    text = " ".join((text or "").replace("\n", " ").split())
    return text.replace("|", "\\|")


def render_llm(fields) -> str:
    groups: dict[str, list] = {}
    for f in fields:
        groups.setdefault(f["ns"], []).append(f)
    order = list(NS_TITLE) + [k for k in groups if k not in NS_TITLE]
    lines = []
    lines.append("# 05 Server 选项大全")
    lines.append("")
    lines.append(
        "范围是 v0.5.23 的两套参数。LLM 的 `ServerArgs` 由 `arg_groups/fields/` 声明，"
        "S1 注册 CLI，S2 `run_resolution_pipeline` 才改写。表内「含义」是中文说明，分成作用、默认与效果、交互与改写、易错。"
        "声明默认不等于运行值。Diffusion 的 dataclass 在文末，走另一条启动，不进这套 S2。"
    )
    lines.append("")
    lines.append(
        f"LLM 字段 **{len(fields)}** 项（含派生和 no_cli）。"
        f"带常规 CLI 的输入字段 **{sum(1 for f in fields if not f['no_cli'] and f['kind']=='cli')}** 项。"
    )
    lines.append("")
    lines.append("## 手写旗标")
    lines.append("")
    lines.append("| CLI | 含义 |")
    lines.append("| --- | --- |")
    manual = [
        ("--config", "作用：从 YAML 合并进 argv，发生在 parse 之前。默认不读文件。交互：文件里的键要能对上 CLI。易错：YAML 把布尔写成字符串，合并后类型错。"),
        ("--sampling-backend", "作用：采样 kernel。choices 来自 SAMPLING_BACKEND_CHOICES，KV canary 打开时多一个 token_oracle。默认仍是字段 sampling_backend。交互：确定性模式会限制可用后端。易错：写了未注册的名字，S1 就失败。"),
        ("--reasoning-parser", "作用：思考段解析器，含 auto。auto 在 S4 按模板落地，V4.1 是 deepseek-v41。易错：用 V3 的 deepseek-r1 解析 V4.1。"),
        ("--tool-call-parser", "作用：工具调用解析器，含 auto。和 chat template 配套。易错：只改解析器不改模板。"),
        ("--kv-canary-real-data", "作用：KV 金丝雀检查真实缓存的范围。none 关闭，partial 查每槽前 16 字节，all 查整槽。易错：在生产打开 all，带宽被检查打满。"),
        ("--disable-cuda-graph", "作用：废弃。等价于两相 backend=disabled。易错：和 cuda_graph_config 同时写，以解析结果为准。"),
        ("--enable-dp-attention", "作用：废弃。与 --dp-size N 一起会被收成 attn_dp_size=N、dp_size=1。计划在 2026-12-31 后删除。易错：以为 dp_size 仍是副本。"),
        ("--monokernel-backend", "作用：废弃，改用 --enable-amd-monokernel。易错：和 NVIDIA megakernel 一起开，S2 会把两个原因一次报出来。"),
        ("--enable-flashinfer-allreduce-fusion", "作用：废弃，改用 --flashinfer-allreduce-fusion-backend=auto。易错：在 HIP 上打开 NVIDIA 融合。"),
    ]
    for cli, text in manual:
        lines.append(f"| `{cli}` | {md_cell(text)} |")
    lines.append("")
    lines.append("## LLM ServerArgs")
    lines.append("")
    for ns in order:
        items = groups.get(ns) or []
        if not items:
            continue
        lines.append(f"### {NS_TITLE.get(ns, ns)}")
        lines.append("")
        lines.append(
            f"命名空间 `{ns}`，类 `{items[0]['class']}`，文件 `{items[0]['file']}`，{len(items)} 项。"
        )
        lines.append("")
        lines.append(NS_ESSAY.get(ns, ""))
        lines.append("")
        lines.append("| 字段 | CLI | 类型 | 默认 | 含义 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for f in items:
            flag = "--" + f["name"].replace("_", "-")
            extra = []
            if f.get("aliases"):
                extra.append(f["aliases"])
            if f.get("no_cli") or f.get("kind") == "derived":
                extra.append("无常规 CLI")
            if f.get("resolvable"):
                extra.append("S2 可改写")
            if extra:
                flag += "（" + "；".join(extra) + "）"
            lines.append(
                "| `{name}` | {cli} | `{typ}` | `{default}` | {help} |".format(
                    name=f["name"],
                    cli=md_cell(flag),
                    typ=md_cell(f["type"]),
                    default=md_cell(f["default"]),
                    help=md_cell(explain_field(f)),
                )
            )
        lines.append("")
    return "\n".join(lines)


def explain_client(f, section: str) -> str:
    name = f["name"]
    if name.startswith("_") or f.get("internal"):
        return (
            f"作用：内部字段 `{name}`，声明在 `{f['owner']}`。不作为公开 JSON。"
            "默认与效果：由服务端在 S7 或 S8 填写。"
            "交互与改写：客户端传了也会被忽略或覆盖。"
            "易错：把它写进请求体指望生效。"
        )
    glossed = gloss(f.get("help") or "")
    purpose = glossed or f"请求字段「{name_zh(name)}」。"
    stage = "它在 S7 进入。"
    if section.startswith("原生"):
        stage = "原生接口直接绑定这个结构，S8 的 TokenizerManager 读取它。"
    elif "sampling" in section:
        stage = "它在 S8 被 normalize，S9 的采样步使用。HTTP 层的同名参数会先抄进这里。"
    elif "OpenAI" in section or "Anthropic" in section or "Ollama" in section:
        stage = "它先留在协议对象上，OpenAI/Anthropic/Ollama serving 再抄进 GenerateReqInput 或 SamplingParams，然后进入 S8。"
    return (
        f"作用：{purpose} {stage} "
        f"默认与效果：默认 `{f['default']}`。不传就用这个默认；null 往往在 normalize 时再收成采样器的默认。"
        "交互与改写：批请求时不少字段允许标量或列表。列表长度要和批大小一致，否则 S8 的 normalize 抛错。"
        "PD 的 bootstrap_*、routed_dp_rank 只影响送到哪个进程，不改变采样。"
        "易错：在 chat 接口上传了只在 /generate 生效的字段，或相反。以该结构所在的路由为准。"
    )


CLIENT_STAGE = {
    "GenerateReqInput": "原生 /generate，S7 直接绑定，S8 分词。",
    "EmbeddingReqInput": "原生 /encode 与 /classify。不走解码采样。",
    "SamplingParams": "S8 规范化后的采样合同，S9 使用。",
    "ChatCompletionRequest": "POST /v1/chat/completions。S7 转成 GenerateReqInput。",
    "CompletionRequest": "POST /v1/completions。",
    "EmbeddingRequest": "POST /v1/embeddings。",
    "ClassifyRequest": "POST /v1/classify。",
    "ScoringRequest": "POST /v1/score。",
    "DecisionRequest": "POST /v1/decisions。",
    "V1RerankReqInput": "POST /v1/rerank。",
    "ResponsesRequest": "POST /v1/responses。可后台，可引用 response store。",
    "TokenizeRequest": "只分词，不进 S9。",
    "DetokenizeRequest": "只把 id 变文本。",
    "TranscriptionRequest": "音频转写。",
    "AnthropicMessagesRequest": "POST /v1/messages。",
    "AnthropicCountTokensRequest": "只计数。",
    "OllamaChatRequest": "POST /api/chat。",
    "OllamaGenerateRequest": "POST /api/generate。",
}


def render_client() -> str:
    lines = []
    lines.append("# 06 Client 请求选项大全")
    lines.append("")
    lines.append(
        "客户端字段按它们进入的结构分组。公开 JSON 在 S7 被 FastAPI 或 Pydantic 接住，"
        "再抄进 `GenerateReqInput` 或 `SamplingParams`，S8 之后 scheduler 只看见 `Req`。"
        "含义列是中文：作用、默认与效果、交互、易错。继承来的 PD 字段（`PDRoutingFields`）算在子类表里。"
    )
    lines.append("")
    lines.append("内部字段（下划线或 PrivateAttr）仍列出，避免漏掉服务端会写的状态，但它们不是稳定的公开 API。")
    lines.append("")
    total = 0
    for title, rel, names in catalog.CLIENT_SPECS:
        path = ROOT / rel
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"文件 `{rel}`。")
        lines.append("")
        classes = catalog.classes_of(path)[2]
        for name in names:
            cls = classes.get(name)
            lines.append(f"### `{name}`")
            lines.append("")
            if cls is None:
                lines.append("源码中未找到该类。")
                lines.append("")
                continue
            fields = catalog.collect_class_fields(path, cls)
            total += len(fields)
            blurb = CLIENT_STAGE.get(name, "")
            lines.append(
                f"定义于第 {cls.lineno} 行，含继承字段 {len(fields)} 项。{blurb}"
            )
            lines.append("")
            lines.append("| 字段 | 声明类 | 默认 | 含义 |")
            lines.append("| --- | --- | --- | --- |")
            for f in fields:
                mark = f["name"] + ("（内部）" if f["internal"] else "")
                lines.append(
                    "| `{name}` | `{owner}` | `{default}` | {help} |".format(
                        name=mark,
                        owner=f["owner"],
                        default=md_cell(f["default"]),
                        help=md_cell(explain_client(f, title)),
                    )
                )
            lines.append("")
    lines.append(f"字段条目合计 {total}。")
    lines.append("")
    return "\n".join(lines)


def explain_diff(f) -> str:
    glossed = gloss(f.get("help") or "")
    purpose = glossed or f"扩散服务字段「{name_zh(f['name'])}」。"
    return (
        f"作用：属于 Diffusion ServerArgs（`{f['owner']}`），不进入 LLM 的 S2。{purpose} "
        f"默认与效果：默认 `{f['default']}`。不传则用该默认；必填项缺失时扩散进程在自己的 argparse 失败。"
        "交互与改写：这套参数由 multimodal_gen 自己的加载逻辑解释，不会触发 deepep 或 attn_dp 的断言。"
        "易错：把 LLM 的 --tp / --moe-a2a-backend 写进扩散命令。两边旗标不是同一份。"
    )


def render_diffusion(fields) -> str:
    lines = []
    lines.append("## Diffusion ServerArgs")
    lines.append("")
    lines.append(
        f"类在 `multimodal_gen/runtime/server_args/server_args.py`，含 `DisaggServerArgsMixin`，共 {len(fields)} 项。"
        "它在 S0 被选成扩散后端之后单独解析。下面的含义是中文说明，注释来自字段上方的源码注释并译述。"
    )
    lines.append("")
    lines.append("| 字段 | 声明类 | 默认 | 含义 |")
    lines.append("| --- | --- | --- | --- |")
    for f in fields:
        lines.append(
            "| `{name}` | `{owner}`:{lineno} | `{default}` | {help} |".format(
                name=f["name"],
                owner=f["owner"],
                lineno=f["lineno"],
                default=md_cell(f["default"]),
                help=md_cell(explain_diff(f)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main():
    fields = catalog.collect_llm_fields()
    text = render_llm(fields)
    diff = catalog.collect_diffusion_fields()
    text += "\n" + render_diffusion(diff)
    (OUT / "05-Server选项大全.md").write_text(text, encoding="utf-8")
    (OUT / "06-Client请求选项大全.md").write_text(render_client(), encoding="utf-8")
    print("05", (OUT / "05-Server选项大全.md").stat().st_size)
    print("06", (OUT / "06-Client请求选项大全.md").stat().st_size)


if __name__ == "__main__":
    main()
