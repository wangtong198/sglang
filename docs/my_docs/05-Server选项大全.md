# 05 Server 选项大全

每一行的「含义」只讲这个字段：它做什么、声明默认会怎样、解析阶段会不会改它、用错时会怎样。
材料来自 `arg_groups/fields/` 的 help，以及 `*_hook.py` / `validation_hook.py` 里点名该字段的断言或改写。
声明默认不是运行值。S2 `run_resolution_pipeline` 之后才是最终配置。

LLM 字段 584 项。Diffusion 的 dataclass 在文末，不进这套解析。

## 不在 Struct 上、由 add_cli_args 另外注册的旗标

| CLI | 含义 |
| --- | --- |
| `--config` | 作用：在 parse 之前把 YAML 里的键合并进命令行。键名要能对上已注册的旗标。默认与效果：不写这个旗标就不读文件。交互与改写：文件里的值和后面的命令行同名旗标冲突时，以合并器的规则为准，布尔不要写成字符串。易错：YAML 把 true 写成 yes，类型对不上 argparse。 |
| `--sampling-backend` | 作用：采样层用哪套 kernel。choices 在 S1 动态生成，打开 KV canary 的 token oracle 时会多一个选项。默认与效果：默认空，按平台选。交互与改写：确定性推理会限制可用后端。易错：写了未注册的名字，argparse 直接失败。 |
| `--reasoning-parser` | 作用：把思考段从正文里拆出来。choices 含 auto 和 reasoning_parser_names。 默认与效果：默认按字段声明。 交互与改写：auto 在 S4 按 chat template 写成具体名，V4.1 对应 deepseek-v41。 易错：手写 deepseek-r1 去解析 V4.1 的模板。 |
| `--tool-call-parser` | 作用：工具调用的解析器。auto 在 S4 选定。V4.1 有 DeepSeekV41Detector。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：和 chat template 必须配套，sampler 不负责拆工具。 易错：只开 parser、模板却不是工具模板，调用边界对不齐。 |
| `--disable-cuda-graph` | 作用：旧的总开关，CLI 上是废弃旗标，指向两相都 disabled。字段本身 no_cli。默认与效果：默认 false。交互与改写：新配置应写分相后端。易错：继续写 --disable-cuda-graph，和分相 JSON 混在一起。 |
| `--enable-dp-attention` | 作用：废弃的内部标记，没有稳定 CLI。 默认与效果：默认 False。 交互与改写：DeprecatedStoreTrueAction 把旧写法收成 attn_dp_size。解析结束后对外报告的 enable_dp_attention 表示 attn_dp_size>1，供旧客户端读 /server_info。 易错：在新配置里继续写这个旗标，并以为 dp_size 仍是副本数。 |

## LLM ServerArgs

### 模型与加载

命名空间 `model`，类 `Model`，`python/sglang/srt/arg_groups/fields/model.py`，45 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `is_startup_weight_load_overlap` | --is-startup-weight-load-overlap（fn=sglang.srt.arg_groups.model_override_base.startup_weight_load_overlap_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：权重加载是否和启动的其他步骤重叠。派生量，由模型路径和加载模式算出来。默认与效果：无 CLI。交互与改写：重叠时启动日志的阶段时间会叠在一起。易错：用它判断加载是否结束。 |
| `shared_experts` | --shared-experts（fn=sglang.srt.configs.shared_experts.default_shared_experts_config；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：正在构建的模型的共享专家布局，给执行路径判断共享专家有几份、怎么融合。默认与效果：从模型配置来，不是用户平时设置的。交互与改写：和 disable_shared_experts_fusion 一起决定是不是独立计算共享专家。易错：在 CLI 上改它来增减专家个数，专家个数在 config 里。 |
| `model_path` | --model-path（--model） | `str` | `（必填）` | 作用：权重目录或仓库 id。S0 的位置参数会变成它。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：none 或 dummy 时 S2 提前返回，不下载。expert_pack 可能把它换成生成的本地元数据。 易错：路径指向只有 safetensors、没有 config 的目录，S5 的 get_config 失败。 |
| `load_format` | --load-format | `str` | `'auto'` | 作用：权重容器格式。auto 会按文件猜。choices 含 safetensors、gguf、remote 等。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：expert_pack 是一条窄路径，只验证过特定 GGUF。 易错：对 fp8 safetensors 手写 gguf。 |
| `model_loader_extra_config` | --model-loader-extra-config | `str` | `'{}'` | 作用：按 load_format 传给对应加载器的额外 JSON。presharded 可以含 presharded_path 和草稿的路径。默认与效果：默认空。交互与改写：别的 load_format 会忽略不认识的键或报错。易错：safetensors 路径里塞了 presharded 的键。 |
| `trust_remote_code` | --trust-remote-code | `bool` | `False` | 作用：允许加载仓库里的自定义建模代码。 默认与效果：默认关。V4.1 的处理器和配置经常需要它。 交互与改写：只影响 S5 的加载，不改变并行。 易错：关着它加载 DeepSeek 权重，tokenizer 或 config 类找不到。 |
| `context_length` | --context-length | `Optional[int]` | `None` | 作用：覆盖模型的最大上下文。 默认与效果：默认用 config。V4.1-Flash 是 1048576。 交互与改写：超过此长度的请求在 S8 被拒。它不增加 KV 池，池仍由显存决定。 易错：把 context 改到一百万，却发现并发只有个位数，因为 144GB 放不下那么多页。 |
| `is_embedding` | --is-embedding | `bool` | `False` | 作用：把 CausalLM 当成向量模型用，走 embedding 而不是生成。默认与效果：默认否。交互与改写：打开后服务的主路径是 /v1/embeddings 或 /encode，生成接口不应再当主用途。架构没有池化头时会在加载或第一次编码失败。易错：聊天服务误开，客户端仍调 chat completions。 |
| `revision` | --revision | `Optional[str]` | `None` | 作用：Hugging Face 或魔搭风格仓库的版本。 默认与效果：默认主分支。 交互与改写：和 model_path 一起在 S5 下载。 易错：revision 不存在，卡在下载而不是断言。 |
| `model_impl` | --model-impl | `str` | `'auto'` | 作用：用哪套模型实现。auto 优先 SGLang 自己的类，没有则退回 Transformers。sglang 强制用仓库里的实现，没有就失败。默认与效果：默认 auto。交互与改写：V4.1 的类在 models/deepseek_v4.py，auto 会找到它。易错：强制 sglang 但该架构还没注册。 |
| `model_config_parser` | --model-config-parser | `str` | `'auto'` | 作用：用谁解析 config.json。auto 时名字像 Mistral 就用 mistral 解析器，否则用 hf 的 AutoConfig。插件可以用 register_model_config_parser 再注册。默认与效果：默认 auto。交互与改写：V4.1 走 HF 配置再经 normalize_deepseek_v41_config。易错：DeepSeek 被误判成 mistral 解析器。 |
| `json_model_override_args` | --json-model-override-args | `str` | `'{}'` | 作用：用 JSON 字符串覆盖模型配置里的默认项，在读完 config.json 之后套上去。默认与效果：默认空，不覆盖。交互与改写：覆盖架构名或层数会让权重对不上。易错：用它改 n_routed_experts 去「减专家」，权重形状立刻不匹配。 |
| `dtype` | --dtype（S2 可改写） | `str` | `'auto'` | 作用：权重和激活的数据类型。auto 对 FP32/FP16 模型用 FP16，对 BF16 模型用 BF16。half 和 float16 都是 FP16，AWQ 推荐 half。bfloat16 在精度和范围之间折中。float 是 FP32。默认与效果：默认通常是 auto，跟检查点。交互与改写：和 kv_cache_dtype、量化方法是三套精度，不要混成一个开关。易错：AWQ 权重却写成 bfloat16。 |
| `quantization` | --quantization（S2 可改写） | `Optional[str]` | `None` | 作用：在线或加载时的量化方法。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：应与 config 的 quant_method 一致。V4.1-Flash 是 fp8 加专家 fp4。乱指定 awq 会在 S5 失败。 易错：把 expert_dtype=fp4 理解成还要再写 --quantization mxfp4。 |
| `quantization_param_path` | --quantization-param-path | `Optional[str]` | `None` | 作用：KV 缩放因子的 JSON。KV dtype 是 FP8 时一般要给。不给则缩放默认为 1.0，精度可能不对。默认与效果：默认空。交互与改写：和 kv_cache_dtype 一起用。权重自身的 scale 不在这个文件。易错：FP8 KV 不给文件，输出悄悄变差。 |
| `kv_cache_dtype` | --kv-cache-dtype（S2 可改写） | `str` | `'auto'` | 作用：KV 存储精度。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：和 attention 后端、量化一起在 S2 检查。fp8 KV 配错后端会在 S6 或第一次前向失败。 易错：为了省显存改 fp8，但 dsv4 池不支持该组合。 |
| `qsa_indexer_dtype` | --qsa-indexer-dtype | `str` | `'auto'` | 作用：压缩 QSA indexer 缓存和索引 query 的存储类型，给 Qwen4-Exp。fp8_e4m3 以普通 e4m3 存储，并用 fp8 GEMM 给块打分，需要 CUDA SM90 及以上。主 KV 缓存不受影响。默认与效果：默认按模型。交互与改写：不是 DeepSeek indexer 的开关。易错：在 V4.1 上设置它，以为能改 index_topk 的精度。 |
| `modelopt_quant` | --modelopt-quant | `Optional[Union[str, Dict]]` | `None` | 作用：用 NVIDIA Model Optimizer 做量化，取值 fp8、int4_awq、w4a8_awq、nvfp4、nvfp4_awq。需要安装 nvidia-modelopt。默认与效果：默认不启用。交互与改写：和模型自带的 quantization_config 不要两套同时量化。易错：没装库就写 nvfp4。 |
| `modelopt_checkpoint_restore_path` | --modelopt-checkpoint-restore-path | `Optional[str]` | `None` | 作用：从已经保存的 ModelOpt 量化检查点恢复，跳过再量化，直接加载。默认与效果：默认空，不恢复。交互与改写：给出路径后量化过程被跳过。易错：路径是原始 BF16 权重，不是量化检查点。 |
| `modelopt_checkpoint_save_path` | --modelopt-checkpoint-save-path | `Optional[str]` | `None` | 作用：把这次 ModelOpt 量化的结果存到该路径，下次可以用 restore 直接加载。默认与效果：默认空，不另存。交互与改写：和 restore 同时给出时，应以 restore 为准跳过量化。易错：路径没有写权限，量化做完才在保存时失败。 |
| `modelopt_export_path` | --modelopt-export-path | `Optional[str]` | `None` | 作用：把 ModelOpt 量化模型导出到这个目录，给后续服务或别的运行时用。默认与效果：默认空，不导出。交互与改写：导出格式要和 modelopt_quant 的取值一致。易错：导出路径和 save 路径写成同一个，互相覆盖。 |
| `quantize_and_serve` | --quantize-and-serve | `bool` | `False` | 作用：启动时先按 ModelOpt 量化再立刻服务，而不是只量化或只加载。默认与效果：默认不这样做。交互与改写：需要 modelopt 库和量化配置。易错：已有量化检查点仍走量化再服务，白等一次量化。 |
| `rl_quant_profile` | --rl-quant-profile | `Optional[str]` | `None` | 作用：为 RL 场景选择一套量化剖面，让服务端量化和训练端对齐。默认与效果：默认空，不套 RL 剖面。交互与改写：和 rl_on_policy_target 一起决定数值要对齐谁。易错：训练用了另一种量化，这里留空。 |
| `startup_weight_load_mode` | --startup-weight-load-mode | `Literal['serial', 'overlap']` | `'serial'` | 作用：启动时怎么读权重，例如是否重叠 IO 和初始化。默认与效果：默认按实现。交互与改写：会决定 is_startup_weight_load_overlap 这个派生量。易错：模式和存储介质不匹配，网络盘上用了假设本地 NVMe 的模式。 |
| `custom_weight_loader` | --custom-weight-loader | `Optional[List[str]]` | `None` | 作用：指定自定义权重加载器的入口，替代 load_format 对应的默认加载器。默认与效果：默认空，用内置加载器。交互与改写：加载器必须认识当前量化格式。易错：路径指到不存在的可导入对象。 |
| `weight_loader_disable_mmap` | --weight-loader-disable-mmap | `bool` | `False` | 作用：读权重时不用 mmap，改成普通读。某些网络文件系统上 mmap 会出错。默认与效果：默认允许 mmap。交互与改写：关掉后峰值内存和启动时间都会变。易错：在本地 NVMe 上关掉，启动变慢且没有收益。 |
| `weight_loader_prefetch_checkpoints` | --weight-loader-prefetch-checkpoints | `bool` | `False` | 作用：加载当前分片时预取后续检查点，重叠读盘。默认与效果：默认按实现。交互与改写：预取线程数由 weight_loader_prefetch_num_threads 控制。易错：磁盘已经打满仍加大预取。 |
| `weight_loader_prefetch_num_threads` | --weight-loader-prefetch-num-threads | `int` | `4` | 作用：预取检查点的线程数。默认与效果：默认按实现。交互与改写：只在预取打开时有用。易错：线程数超过磁盘队列深度。 |
| `weight_loader_drop_cache_after_load` | --weight-loader-drop-cache-after-load | `bool` | `False` | 作用：权重加载完后丢掉页缓存，把主机内存还给 HiCache 或别的用途。默认与效果：默认不丢。交互与改写：丢掉后如果还要第二次读同一文件会重新读盘。易错：调试时反复加载同一路径，每次都冷读。 |
| `remote_instance_weight_loader_seed_instance_ip` | --remote-instance-weight-loader-seed-instance-ip | `Optional[str]` | `None` | 作用：从另一台已经起来的实例拉权重时，种子实例的 IP。默认与效果：默认空，不走远程实例加载。交互与改写：要和端口、通信组端口、后端一起给。易错：只写 IP 不写端口。 |
| `remote_instance_weight_loader_seed_instance_service_port` | --remote-instance-weight-loader-seed-instance-service-port | `Optional[int]` | `None` | 作用：种子实例上提供权重服务的端口。默认与效果：默认空。交互与改写：这是服务端口，不是 NCCL 端口。易错：写成 HTTP 30000，而传输走的是另一组端口。 |
| `remote_instance_weight_loader_send_weights_group_ports` | --remote-instance-weight-loader-send-weights-group-ports | `Optional[List[int]]` | `None` | 作用：种子实例把权重组播或点对点送出时用的端口列表。默认与效果：默认空。交互与改写：多 rank 时每个 rank 要有自己的端口。易错：列表长度和 tp 不一致。 |
| `remote_instance_weight_loader_backend` | --remote-instance-weight-loader-backend | `Literal['transfer_engine', 'nccl', 'modelexpress']` | `'nccl'` | 作用：从远程实例加载权重的后端，例如 nccl 或 transfer_engine。默认与效果：help 里的默认是 nccl。交互与改写：transfer_engine 还要种子侧先把引擎拉起来。易错：后端写成 nccl 但机器之间没有那套集合通信。 |
| `remote_instance_weight_loader_start_seed_via_transfer_engine` | --remote-instance-weight-loader-start-seed-via-transfer-engine | `bool` | `False` | 作用：是否由本进程通过 transfer engine 把种子侧也拉起来。默认与效果：默认否。交互与改写：为真时种子 IP 必须可达。易错：种子已经在跑又让这边再启动一次。 |
| `engine_info_bootstrap_port` | --engine-info-bootstrap-port | `int` | `6789` | 作用：引擎信息引导服务的端口，别的进程来这里拿 rank 和地址。默认与效果：默认按实现分配。交互与改写：和 HTTP 端口、NCCL 端口都要错开。易错：多实例撞同一个引导端口。 |
| `modelexpress_config` | --modelexpress-config | `Optional[str]` | `None` | 作用：ModelExpress 加载路径的配置。默认与效果：默认空，不用 ModelExpress。交互与改写：和普通 safetensors 加载二选一。易错：配置指向的清单和 model_path 不是同一份权重。 |
| `download_dir` | --download-dir | `Optional[str]` | `None` | 作用：权重缓存目录。默认用库的。 默认与效果：默认缓存。 交互与改写：多机若目录不共享，每台都要能读到同一份权重。 易错：只在 rank 0 下载。 |
| `model_checksum` | --model-checksum | `Optional[str]` | `None` | 作用：期望的模型校验和。加载后比对，不一致就失败。默认与效果：默认空，不校验。交互与改写：用于确认没有下到半截文件。易错：校验和是旧版本权重的，新下载必然失败。 |
| `delete_ckpt_after_loading` | --delete-ckpt-after-loading | `bool` | `False` | 作用：权重加载进 GPU 后删除磁盘上的检查点。默认与效果：默认不删。交互与改写：删了就不能热重启时再读同一路径。易错：在共享缓存目录上打开，别的进程还要用这份文件。 |
| `decrypted_config_file` | --decrypted-config-file | `Optional[str]` | `None` | 作用：解密后的模型配置文件路径，给加密检查点用。默认与效果：默认空。交互与改写：和 decrypted_draft_config_file 是两份，草稿另算。易错：把明文 config.json 和这个解密文件搞混。 |
| `decrypted_draft_config_file` | --decrypted-draft-config-file | `Optional[str]` | `None` | 作用：草稿模型解密后的配置路径。默认与效果：默认空。交互与改写：只在投机草稿也加密时需要。易错：只解密了目标模型。 |
| `checkpoint_engine_wait_weights_before_ready` | --checkpoint-engine-wait-weights-before-ready | `bool` | `False` | 作用：在对外报 ready 之前先等检查点引擎把权重送齐。默认与效果：默认按实现。交互与改写：打开后父进程的 wait_for_ready 会更晚返回。易错：权重源没启动，服务一直不 ready。 |
| `weight_cache_mode` | --weight-cache-mode | `str` | `'off'` | 作用：权重缓存守护进程的模式。off 关闭，daemon 起缓存服务，client 去连已有守护进程。默认与效果：默认 off。交互与改写：client 模式要有 socket 路径。易错：多机都写成 daemon，各自一份缓存，没有共享。 |
| `weight_cache_socket` | --weight-cache-socket | `Optional[str]` | `None` | 作用：权重缓存守护进程的套接字路径。默认与效果：默认空。交互与改写：client 和 daemon 必须用同一个路径。易错：路径在容器里不可见。 |
| `weight_cache_timeout` | --weight-cache-timeout | `int` | `1800` | 作用：连权重缓存或等它响应的超时。默认与效果：默认按实现。交互与改写：超时太短，大文件还在写入就被判定失败。易错：和 HTTP 超时混成一个数。 |

### 并行（TP / PP / DP / EP / CP / DCP）

命名空间 `parallel`，类 `Parallel`，`python/sglang/srt/arg_groups/fields/parallel.py`，73 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `nccl_port` | --nccl-port | `Optional[int]` | `None` | 作用：NCCL 环境的端口。 默认与效果：默认空，PortArgs 选一个临时端口以下的空闲端口。 交互与改写：多机防火墙要放行。它不决定 TP 宽度。 易错：随机端口被安全组丢掉，卡在 init_distributed_environment。 |
| `dist_timeout` | --dist-timeout | `Optional[int]` | `None` | 作用：torch.distributed 初始化的超时秒数。多机建组慢时要加大。默认与效果：默认空，用 PyTorch 默认超时。交互与改写：超时发生在 S3 建进程组，不是 HTTP。易错：超时太短，NCCL 还在握手就被判定失败。 |
| `dist_init_addr` | --dist-init-addr（--nccl-init-addr） | `Optional[str]` | `None` | 作用：分布式初始化的主机地址，形如 host:port。别名 --nccl-init-addr。 默认与效果：默认空。 交互与改写：多节点时所有 rank 必须能连到 rank 0 上的这个地址。 易错：写成每台机器自己的主机名。 |
| `gated_launch_port` | --gated-launch-port | `Optional[int]` | `None` | 作用：门控启动的控制端口。设置后，每个 rank 在分布式环境建好之后、大块显存分配之前停住，直到向 rank 0 那台机器的这个端口 POST /gate/activate。空表示不门控。默认与效果：默认空，不暂停。交互与改写：用来让外部编排把占内存的阶段推迟到安全窗口。易错：设了端口却没有人发 activate，所有 rank 一直等。 |
| `nnodes` | --nnodes | `int` | `1` | 作用：节点数。 默认与效果：默认 1。 交互与改写：与 node_rank、dist_init_addr 一起在 S3 建进程组。(tp*pp) 必须整除它。dp_size>1 只允许它为 1。 易错：四台机器都写 nnodes=1，各自成群。 |
| `node_rank` | --node-rank | `int` | `0` | 作用：本进程所在节点的序号，从 0 开始。 默认与效果：默认 0。 交互与改写：大于 0 的节点不起 HTTP，只跑 scheduler 直到 ready。 易错：每台机器都写 0，端口冲突，进程组建不起来。 |
| `tp_size` | --tp-size（--tensor-parallel-size） | `int` | `1` | 作用：张量并行宽度，也是单副本的 GPU 数因子之一（再乘 pp 与模型副本 dp）。 默认与效果：默认 1，单卡，不做切分。 交互与改写：S3 用它起进程；(tp*pp) 必须整除 nnodes。attention 与 MoE 的子宽度必须乘回这个数。spanning all-to-all 会把 ep_size 改成它。 易错：把它当成「只切 attention」。KV 头为 1 的 V4.1 不应靠把 tp 拉到 16 来切 attention，应配 attn_dp_size。 |
| `dcp_size` | --dcp-size（--decode-context-parallel-size） | `int` | `1` | 作用：decode 上下文并行，在每个 TP 组内切 decode 的 KV。 默认与效果：默认 1。不增加 GPU。 交互与改写：大于 1 不能同时开 prefill CP。fi_a2a 只允许 CUDA。必须整除 attention TP。 易错：和 --enable-prefill-cp 一起开，S2 的 handle_decode_context_parallelism 会拒绝。 |
| `pp_size` | --pp-size（--pipeline-parallel-size） | `int` | `1` | 作用：流水并行段数，增加 GPU。 默认与效果：默认 1，不切层。 交互与改写：大于 1 时 S2 强制关掉 overlap schedule。V4.1 的特征检查直接拒绝 PP。moe_dp>1、DWDP、elastic EP 也要求它为 1。 易错：用 PP 把 16 卡分成两段来跑 V4.1-Flash，进程会在 S2 退出。 |
| `pp_max_micro_batch_size` | --pp-max-micro-batch-size | `Optional[int]` | `None` | 作用：流水并行里一个微批最多多少请求。空表示自动算。必须是正整数或空。默认与效果：默认空，自动。交互与改写：V4.1 根本不允许 PP。和 min_free_slots_delay 不能一起用，因为微批槽位被这个上限卡住，阈值可能永远达不到。易错：V4.1 上设置它。 |
| `pp_async_batch_depth` | --pp-async-batch-depth | `int` | `0` | 作用：流水并行异步堆积多少个微批。默认与效果：默认 0，不额外异步堆积。交互与改写：大于 0 会增加在途激活的显存。V4.1 没有 PP。易错：把深度设得很大，激活把 KV 挤掉。 |
| `dp_size` | --dp-size（--data-parallel-size；S2 可改写） | `int` | `1` | 作用：整模副本数，GPU 数再乘一份。 默认与效果：默认 1。 交互与改写：与 attn_dp_size 不能同时大于 1。nnodes 不是 1 时 check_server_args 直接 assert，提示改用 attn_dp_size。 易错：四机各写 --dp-size 4 --tp 4。那是四个副本，而这棵树不允许副本跨节点。 |
| `attn_dp_size` | --attn-dp-size（--attention-data-parallel-size；S2 可改写） | `int` | `1` | 作用：在已有 TP 组里划出多少个 attention 数据并行组，每组一个 scheduler，FFN/MoE 仍可 TP 或 EP。 默认与效果：默认 1，不划分。不增加 GPU。 交互与改写：tp 必须能被它整除。大于 1 时保守系数乘 0.3，chunked prefill 除以它，ZMQ 改用 TCP。旧旗标 --dp-size N --enable-dp-attention 会被收成 attn_dp_size=N 且 dp_size=1。 易错：和 dp_size 同时放大；或在 HIP 上再开 encoder SWA bounded replay。 |
| `load_balance_method` | --load-balance-method | `str` | `'auto'` | 作用：数据并行把请求分到哪个 scheduler。可选 auto、round_robin、follow_bootstrap_room、total_requests、total_tokens。默认与效果：默认 auto。交互与改写：attention DP 时这是组间负载，不是专家负载。follow_bootstrap_room 用于 PD 把同一房间送到同一组。易错：想粘住前缀却用了 round_robin。 |
| `attn_cp_size` | --attn-cp-size（--attention-context-parallel-size；S2 可改写） | `int` | `1` | 作用：attention 上下文并行宽度。 默认与效果：默认 1。 交互与改写：prefill CP 打开时，DeepSeek-V4 的校验会把它改写成 tp/attn_dp。HIP 上的 V4.1 和多节点在改写前就会失败。 易错：以为自己设成 2 就会生效。模型 hook 可能把它改成铺满剩余 TP。 |
| `moe_dp_size` | --moe-dp-size（--moe-data-parallel-size） | `int` | `1` | 作用：MoE 的数据并行宽度。 默认与效果：默认 1。 交互与改写：大于 1 时要求 pp 为 1；若 ep 也大于 1，则 ep 乘 moe_dp 必须等于 tp。attn_cp 与它不相等时，只允许 moe_dp 为 1。 易错：把它当成 attn_dp_size。两者切的不是同一组算子。 |
| `dwdp_size` | --dwdp-size | `int` | `1` | 作用：分布式权重数据并行。大于 1 时 MoE 预填充用权重预取代替 token all-to-all。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：必须等于 tp，强制纯 DP attention、ep=tp、a2a none、无 PP、无投机、无 CUDA graph。 易错：和 DSPARK 一起开。 |
| `dcp_comm_backend` | --dcp-comm-backend（S2 可改写） | `Optional[str]` | `None` | 作用：解码上下文并行归约用的通信。ag_rs 是 all-gather 加 reduce-scatter；a2a 是一次 all-to-all 交换输出和 LSE 再本地合并；fi_a2a 用 FlashInfer 的 MNNVL all-to-all，要 Blackwell 且 DCP 组在同一个 MNNVL 域。不填时，支持的地方用 fi_a2a，CUDA/ROCm/XPU 否则用 a2a，再不然 ag_rs。默认与效果：默认空，按平台选。交互与改写：a2a 和 fi_a2a 要求 dcp_size 大于 1。fi_a2a 在非 CUDA 上直接拒绝。易错：海光上写 fi_a2a。 |
| `dcp_replicate_q_proj` | --dcp-replicate-q-proj（S2 可改写） | `Optional[bool]` | `None` | 作用：MLA 的 DCP 在 a2a 或 fi_a2a 上，是否每个 rank 都算完整的 Q，省掉按头维 all-gather Q。用 --no-dcp-replicate-q-proj 关掉模型默认。默认与效果：默认空，按模型。交互与改写：必须 dcp_size 大于 1，且通信后端是 a2a 或 fi_a2a。易错：ag_rs 上打开它。 |
| `enable_prefill_cp` | --enable-prefill-cp | `bool` | `False` | 作用：预填充阶段按序列切分上下文，也就是笔记里的 PCP。 默认与效果：默认关。 交互与改写：必须同时给 cp_strategy（interleave 或 zigzag）。与 dcp_size>1 互斥。V4 zigzag 只允许 NPU。HIP 上 V4.1、多节点、tp>8 都拒绝。 易错：在四机 BW1100 上用它来拉长上下文。 |
| `cp_strategy` | --cp-strategy | `Optional[str]` | `None` | 作用：预填充 CP 的切分方式，zigzag 或 interleave。 默认与效果：默认空。 交互与改写：enable_prefill_cp 为真时必填，否则 ValueError。V4 在 CUDA/HIP 上只能 interleave。 易错：只写 enable，不写策略。 |
| `enable_cp_tp_group_sharing` | --enable-cp-tp-group-sharing（无常规 CLI；S2 可改写） | `bool` | `False` | 作用：派生。混合线性注意力模型上，CP 组就是 TP 组：残差、注意力和 indexer 按 CP 切，MoE 和线性注意力仍按 TP。从架构解析，没有 CLI。默认与效果：默认 false，直到模型说要。交互与改写：用户不能直接写这个名字。易错：在命令行找 --enable-cp-tp-group-sharing。 |
| `enable_dsa_cache_layer_split` | --enable-dsa-cache-layer-split | `bool` | `False` | 作用：把 DSA 的 GPU KV 和 indexer 缓存按层切到上下文并行的各个 rank，减少每卡 KV。目前只支持 mooncake 或 mooncake_tcp 传输。mori 和 nixl 以后才会有。默认与效果：默认关闭。交互与改写：传输后端不是 mooncake 时不能开。易错：四机 mori 配置打开它。 |
| `enable_cp_decode_attn_tp` | --enable-cp-decode-attn-tp | `bool` | `False` | 作用：上下文并行且 cp 大于 1 时，解码阶段把原本复制的注意力线性层按本地 CP 分片切，去掉重复的解码 GEMM。默认与效果：默认关闭。交互与改写：只在 CP 解码时有收益。V4.1 的 HIP 预填充 CP 本身就不允许。易错：没有 CP 却打开。 |
| `enable_kv_cache_sharding` | --enable-kv-cache-sharding | `bool` | `False` | 作用：把原本在一组里复制的静态 KV 按页切到该组各 rank，提高这一组的不重复 KV 容量。GQA 在预填充 CP 下按 attention-CP 切；MLA 按 attention-TP 切。只给 PD 的 prefill worker。默认与效果：默认关闭。交互与改写：页大小、注意力后端和预算都定了之后才会校验，S2 末尾还会再跑一次。易错：decode 节点打开它。 |
| `enable_dp_attention` | --enable-dp-attention（无常规 CLI；S2 可改写） | `bool` | `False` | 作用：废弃的内部标记，没有稳定 CLI。 默认与效果：默认 False。 交互与改写：DeprecatedStoreTrueAction 把旧写法收成 attn_dp_size。解析结束后对外报告的 enable_dp_attention 表示 attn_dp_size>1，供旧客户端读 /server_info。 易错：在新配置里继续写这个旗标，并以为 dp_size 仍是副本数。 |
| `enable_dp_attention_local_control_broadcast` | --enable-dp-attention-local-control-broadcast | `bool` | `False` | 作用：attention DP 下，控制消息只发给每个 DP 组的组长，再在 attn_tp 组内广播，而不是每次调度都在全体 tp 上做一次 gloo 同步。默认与效果：默认关闭。交互与改写：必须已经是 attention DP。易错：没有 DP attention 却打开，没有组长可发。 |
| `enable_dp_lm_head` | --enable-dp-lm-head（S2 可改写） | `bool` | `False` | 作用：attention DP 下 LM head 走数据并行。 默认与效果：默认关。 交互与改写：必须已经打开 attention DP。DSPARK 在 DP attention 下强制它。与 enable_tp_lm_head_all_to_all 互斥。 易错：开了 DSPARK 和 attn_dp_size 却忘了它，_handle_dspark 会拒绝启动。 |
| `enable_tp_lm_head_all_to_all` | --enable-tp-lm-head-all-to-all（S2 可改写） | `Optional[bool]` | `None` | 作用：attention DP 下，TP 切分的 LM head 用 all-to-all，而不是先 all-gather 再按 DP 散开。默认只在 PD 的 decode、纯 DP attention（tp 等于 attn_dp 且大于 1）且没有 CP 时打开。和 enable_dp_lm_head 互斥。批的行数对不齐时退回 all-gather。默认与效果：默认空，按上面的条件。交互与改写：可用 --no- 关掉。易错：和 dp lm head 同时开。 |
| `enable_attn_tp_input_scattered` | --enable-attn-tp-input-scattered（S2 可改写） | `bool` | `False` | 作用：注意力 TP 的输入保持分散，不先 all-gather 成完整隐藏向量。默认与效果：默认关闭。交互与改写：要和后续的 gather 开关配套，否则残差对不上。易错：只开分散不关掉多余的 gather。 |
| `enable_shared_experts_attn_tp` | --enable-shared-experts-attn-tp | `bool` | `False` | 作用：共享专家走注意力那一路的 TP，而不是 MoE 的 EP。默认与效果：默认关闭。交互与改写：shared_experts_tp_size 必须整除注意力 TP。只支持 Kimi-K3。易错：在 DeepSeek 上打开。 |
| `shared_experts_tp_size` | --shared-experts-tp-size | `Optional[int]` | `None` | 作用：共享专家的 TP 宽度，必须是注意力 TP 的正约数。只支持 Kimi-K3，并且 a2a 后端要是 deepep、megamoe、mooncake、ascend_fuseep 或 mori。默认与效果：默认空。交互与改写：不是 Kimi-K3 会在 handle_shared_experts_tp 拒绝。易错：V4.1 上设置它。 |
| `enable_dense_mlp_attn_tp` | --enable-dense-mlp-attn-tp | `bool` | `False` | 作用：稠密 MLP 按注意力 TP 切，而不是按 MoE 的切分。默认与效果：默认关闭。交互与改写：moe_dense_tp_size 是另一条约束，只能是空、1 或 tp。易错：两个都设成互相矛盾的宽度。 |
| `enable_layernorm_sp` | --enable-layernorm-sp | `bool` | `False` | 作用：LayerNorm 做序列并行。默认与效果：默认关闭。交互与改写：要在投机算法定稿之后才校验。易错：和某些融合 all-reduce 同时开，归约顺序冲突。 |
| `enable_megamoe_for_layernorm_sp` | --enable-megamoe-for-layernorm-sp | `bool` | `False` | 作用：LayerNorm 序列并行时，MoE 用 MegaMOE。默认与效果：默认关闭。交互与改写：MegaMOE 要求 attn_dp 等于 tp。易错：attn_dp 只有 4、tp 是 16 仍打开。 |
| `disable_attn_tp_gather` | --disable-attn-tp-gather | `bool` | `False` | 作用：注意力 TP 之后不再把结果 gather 回来。默认与效果：默认不禁用 gather。交互与改写：只有后续层能吃分散输入时才能关。易错：关掉后残差形状不对。 |
| `enable_p2p_check` | --enable-p2p-check | `bool` | `False` | 作用：启动时检查 GPU 之间的点对点访问。默认与效果：默认按实现。交互与改写：检查失败说明拓扑不适合某些自定义 all-reduce。易错：在容器里 P2P 被禁，检查失败被当成权重错误。 |
| `ep_size` | --ep-size（--expert-parallel-size, --ep；S2 可改写） | `int` | `1` | 作用：专家并行宽度，不增加 GPU，rank 仍在 TP 组里。 默认与效果：默认 1，专家不按 EP 切开。 交互与改写：mori、deepep、deepep_v2、megamoe 等 spanning 后端在 S2 把它改成 tp_size，并打日志。与 moe_dp_size 相乘必须回到 tp。 易错：手写一个较小的 ep，又开了 mori，运行值会被改掉。 |
| `moe_dense_tp_size` | --moe-dense-tp-size（S2 可改写） | `Optional[int]` | `None` | 作用：MoE 模型里稠密 MLP 的 TP。大 TP 时 MLP 维可能小于 GEMM 允许的最小维，这时要把稠密层的 TP 设成 1。只能是空、1 或整个 tp。默认与效果：默认空。交互与改写：DeepSeek 的预填充 CP 会把它改成 1。易错：设成 2 这种既不是 1 也不是 tp 的数。 |
| `ep_join_rank_offset` | --ep-join-rank-offset | `int` | `0` | 作用：弹性 EP 加入组的全局 rank 偏移。扩容加入者必须把它设成当前有效的 EP 大小。默认与效果：默认 0。交互与改写：和 elastic_ep_initial_size 一起描述扩容前的布局。易错：偏移写成 0，新 rank 和旧 rank 重叠。 |
| `elastic_ep_initial_size` | --elastic-ep-initial-size | `Optional[int]` | `None` | 作用：用来固定每个 rank 专家存储布局的 EP 大小，布局一旦定下就不变。扩容加入者必须用主部署启动时的 EP。默认与效果：默认空。交互与改写：和 max_ep_size 不同，一个是布局，一个是能扩到的上限。易错：加入者用了自己的 tp，而不是主部署的 EP。 |
| `max_ep_size` | --max-ep-size | `Optional[int]` | `None` | 作用：运行中 EP 最多扩到多大。会按这个大小预分配活跃 rank 状态和后端缓冲。默认是启动时的世界大小。默认与效果：默认空，等于启动世界大小。交互与改写：预分配会占额外显存。易错：设得比机器能提供的卡还大。 |
| `attn_tp_size` | --attn-tp-size（fn=sglang.srt.runtime_context.attn_tp_size_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：注意力张量并行宽度，等于 tp 除以 attn_dp 再除以 attn_cp。派生量。默认与效果：无 CLI。交互与改写：ParallelContext 会检查它和另外两项乘回 tp。易错：在命令行写 --attn-tp-size，不会被接受。 |
| `attn_dp_enabled` | --attn-dp-enabled（fn=sglang.srt.runtime_context.attn_dp_enabled_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：注意力是否按数据并行跑。attn_dp_size 大于 1，或者这个进程是弹性 EP 的扩容加入者（它自己的组可能只有 1 宽）时为真。派生量。默认与效果：无 CLI。交互与改写：不要用它回写配置，/server_info 里旧的 enable_dp_attention 表示的是宽度大于 1，不是这个派生量。易错：把扩容加入者当成没有 attention DP。 |
| `num_dp_ranks` | --num-dp-ranks（fn=sglang.srt.runtime_context.num_dp_ranks_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：有多少个数据并行 rank，每个一个 scheduler，等于 dp_size 乘 attn_dp_size。弹性扩容时会变大。派生量。默认与效果：无 CLI。交互与改写：四机建议配置是 1×4，所以是 4 个 scheduler。易错：用 GPU 数去除，把节点数算进去。 |
| `attn_dcp_size` | --attn-dcp-size（fn=sglang.srt.runtime_context.attn_dcp_size_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：注意力 TP 组内部的解码上下文并行宽度。dcp 没开时是 1。派生量。默认与效果：无 CLI。交互与改写：必须整除注意力 TP。易错：和 attn_cp_size 看成同一个数。 |
| `moe_ep_size` | --moe-ep-size（fn=sglang.srt.runtime_context.moe_ep_size_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：从配置归一化之后的 MoE 专家并行宽度。spanning 后端会把它收成 tp。派生量。默认与效果：无 CLI。交互与改写：和用户写的 ep_size 在改写后应该一致。易错：日志里两个数不同，说明改写发生了，以这个派生量为准。 |
| `moe_tp_size` | --moe-tp-size（fn=sglang.srt.runtime_context.moe_tp_size_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：去掉专家并行和 MoE DP 之后，tp 还剩下的 MoE 张量并行宽度。派生量。默认与效果：无 CLI。交互与改写：ep 等于 tp 且 moe_dp 为 1 时，它是 1。易错：以为 MoE 还有一份和注意力一样的 TP。 |
| `dcp_enabled` | --dcp-enabled（fn=sglang.srt.runtime_context.dcp_enabled_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：dcp_size 大于 1 时为真，这时才会建 DCP 进程组。派生量。默认与效果：无 CLI。交互与改写：和 prefill CP 互斥。易错：dcp_size 为 1 却期待组已经建好。 |
| `tp_rank` | --tp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程在张量并行组里的序号。派生量，由启动布局算出来。默认与效果：无 CLI。交互与改写：公式是 (attn_dp_rank * attn_cp + attn_cp_rank) * attn_tp + attn_tp_rank。易错：用节点号当成 tp_rank。 |
| `pp_rank` | --pp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程在流水并行组里的段号。派生量。默认与效果：无 CLI。pp 为 1 时就是 0。交互与改写：V4.1 不允许 PP，这个值保持 0。易错：多机时用 node_rank 代替它。 |
| `moe_ep_rank` | --moe-ep-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程在专家并行组里的序号。派生量。默认与效果：无 CLI。交互与改写：ep 等于 tp 时它和对应的 MoE 布局一致。易错：和 tp_rank 假设永远相等，中间还有 moe_dp 时不相等。 |
| `moe_dp_rank` | --moe-dp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程在 MoE 数据并行组里的序号。派生量。默认与效果：无 CLI。moe_dp 为 1 时是 0。交互与改写：和 attn_dp_rank 不是同一个组。易错：用注意力 DP 的组号去取 MoE 权重。 |
| `moe_tp_rank` | --moe-tp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程在 MoE 张量并行组里的序号。派生量。默认与效果：无 CLI。ep 铺满 tp 时这组宽度是 1，序号是 0。交互与改写：宽度见 moe_tp_size。易错：ep=tp 时还按 tp_rank 去切专家矩阵。 |
| `attn_tp_rank` | --attn-tp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程在注意力张量并行组里的序号。派生量。默认与效果：无 CLI。交互与改写：四机建议配置里每组 4 个，组内序号是 0 到 3。易错：用全局 rank 去除 4 却没按组重置。 |
| `attn_cp_rank` | --attn-cp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程在注意力上下文并行组里的序号。派生量。默认与效果：无 CLI。CP 没开时是 0。交互与改写：决定这条序列的哪一段 token 由本 rank 算。易错：CP 关闭时仍按它切序列。 |
| `dcp_rank` | --dcp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程在解码上下文并行组里的序号。派生量。默认与效果：无 CLI。交互与改写：只在 dcp_size 大于 1 时有多于一个的序号。易错：和 attn_cp_rank 混用。 |
| `attn_dcp_rank` | --attn-dcp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：注意力组内部的 DCP 序号。派生量。默认与效果：无 CLI。交互与改写：和全局 dcp_rank 的计数范围可能不同，因为它在注意力 TP 组内。易错：用全局序号去索引组内 KV。 |
| `attn_dp_rank` | --attn-dp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本进程属于第几个注意力数据并行组。派生量。默认与效果：无 CLI。交互与改写：DataParallelController 按这个组投递请求。四机建议里它等于节点号。易错：请求的 routed_dp_rank 和这个进程的组号对不上还硬算。 |
| `dp_rank` | --dp-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：模型副本数据并行的序号。派生量。默认与效果：无 CLI。dp_size 为 1 时是 0。交互与改写：多节点禁止 dp_size 大于 1，所以跨节点时这个值不该被用来复制整模。易错：把 attn_dp_rank 打印成 dp_rank。 |
| `launch_world_rank` | --launch-world-rank（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本次启动的全局 rank，含弹性 EP 的偏移。派生量。默认与效果：无 CLI。交互与改写：等于 ep_join_rank_offset 加上组内位置。易错：用它代替 tp_rank 去切权重。 |
| `launch_world_size` | --launch-world-size（fn=sglang.srt.runtime_context.launch_world_size_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：本次启动的全局进程数，是偏移加上 tp 乘 pp。派生量。默认与效果：无 CLI。交互与改写：attention DP 不增加这个数。易错：把 dp_size 也乘进去，多算了进程。 |
| `max_world_size` | --max-world-size（fn=sglang.srt.runtime_context.max_world_size_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：预分配时允许的最大世界大小，给弹性扩容留缓冲。派生量。默认与效果：无 CLI。交互与改写：和 max_ep_size 一起决定缓冲。易错：把它当成当前正在跑的进程数。 |
| `world_group` | --world-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：全体进程的通信组句柄。运行时对象，没有 CLI。默认与效果：建组之后才有。交互与改写：在 publish 之前读取会失败。易错：在 tokenizer 进程里用它做集合通信。 |
| `tp_group` | --tp-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：张量并行通信组句柄。运行时对象。默认与效果：无 CLI。交互与改写：连续 rank 组成一组。易错：把句柄序列化进请求。 |
| `pp_group` | --pp-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：流水并行通信组句柄。运行时对象。默认与效果：pp 为 1 时组是平凡的。交互与改写：V4.1 不建多段流水。易错：对平凡组做 send/recv。 |
| `moe_ep_group` | --moe-ep-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：专家并行通信组句柄。运行时对象。默认与效果：无 CLI。交互与改写：all-to-all 发生在这个组上。易错：用 tp_group 做专家交换。 |
| `moe_dp_group` | --moe-dp-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：MoE 数据并行通信组句柄。运行时对象。默认与效果：无 CLI。交互与改写：moe_dp 为 1 时不需要跨 rank 交换专家批次。易错：和 attn 的 DP 组当成同一个。 |
| `moe_tp_group` | --moe-tp-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：MoE 张量并行通信组句柄。运行时对象。默认与效果：无 CLI。交互与改写：ep 铺满时这个组只有本进程。易错：仍在这个组上做 all-reduce。 |
| `attn_tp_group` | --attn-tp-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：注意力张量并行通信组句柄。运行时对象。默认与效果：无 CLI。交互与改写：DP attention 的控制广播可以只在这个组里做。易错：用整个 tp_group 做注意力 all-reduce，把别的 DP 组卷进来。 |
| `attn_cp_group` | --attn-cp-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：注意力上下文并行通信组句柄。运行时对象。默认与效果：无 CLI。交互与改写：预填充 CP 的序列交换走这里。易错：CP 没开时对它发通信。 |
| `shared_experts_tp_group` | --shared-experts-tp-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：共享专家张量并行的通信组句柄。运行时对象。默认与效果：无 CLI。交互与改写：只在 Kimi-K3 的 shared_experts_tp_size 生效时有非平凡组。易错：DeepSeek 上使用它。 |
| `dcp_group` | --dcp-group（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：解码上下文并行的通信组句柄。运行时对象。默认与效果：无 CLI。交互与改写：归约后端 ag_rs 或 a2a 在这个组上做。易错：用 attn_cp_group 做解码归约。 |

### 显存、KV 与前缀缓存

命名空间 `memory`，类 `Memory`，`python/sglang/srt/arg_groups/fields/memory.py`，31 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `radix_eviction_policy` | --radix-eviction-policy（S2 可改写） | `str` | `'lru'` | 作用：树的驱逐策略。lru 最近最少使用，lfu 最不经常使用，slru 分段 LRU，priority 先驱逐低优先级，tlru 带尾部的 LRU。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：disable 时内部仍用 lru 且不记录。priority 要请求带 priority。 易错：策略名写错，不在 choices 里，S1 就 argparse 失败。 |
| `_radix_eviction_policy_explicitly_set` | ---radix-eviction-policy-explicitly-set（无常规 CLI） | `bool` | `False` | 作用：内部标记，区分「用户没写驱逐策略」和「用户显式写了 lru」。模型特有的默认策略必须保留显式选择。没有 CLI。默认与效果：由 prepare_server_args 看 argv 里有没有 --radix-eviction-policy。交互与改写：不要在代码里清掉它，否则模型默认会覆盖用户的 lru。易错：在测试里直接构造 ServerArgs 却没设这个标记，行为和命令行不一致。 |
| `radix_eviction_policy_config` | --radix-eviction-policy-config | `Optional[Dict[str, Any]]` | `None` | 作用：传给驱逐策略的 JSON 参数。slru 接受 protected_threshold，默认 2；tlru 另有自己的阈值。默认与效果：默认空，用策略内置参数。交互与改写：策略名和 JSON 的键必须匹配。易错：策略是 lru 却传了 slru 的键。 |
| `disable_radix_cache` | --disable-radix-cache（S2 可改写） | `bool` | `False` | 作用：关掉前缀树。 默认与效果：默认开着树。 交互与改写：关掉后 SchedulePolicy 会把 LPM 换成不看缓存的策略，每次请求从零写 KV。 易错：排查命中率时关掉后忘记打开，延迟整体变差。 |
| `enable_page_major_kv_layout` | --enable-page-major-kv-layout | `bool` | `False` | 作用：把 Mamba 状态和全量/SWA KV 收成按 token 槽、每个槽含所有层的单页缓冲，而不是每层一块缓冲。enable_unified_memory 会隐含打开它。默认与效果：默认关闭。交互与改写：注意力后端必须支持这种布局，不支持的会在建池时失败。易错：普通 MHA 模型打开它。 |
| `enable_unified_memory` | --enable-unified-memory | `bool` | `False` | 作用：用一块字节缓冲动态切给全量 KV、SWA 和 Mamba，替代事先分死的多个池。需要注意力、线性注意力和 Mamba 后端认识这种步长。DeepSeek-V4 在建池时被排除。默认与效果：默认关闭。交互与改写：会隐含 page-major 布局。易错：V4.1 打开它，建池直接拒绝。 |
| `enable_session_radix_cache` | --enable-session-radix-cache | `bool` | `False` | 作用：在 UnifiedRadixCache 上按会话记引用。驱逐时先拿掉没有引用的项；关闭会话只减引用，不立刻删 KV。默认与效果：默认关闭。交互与改写：V4.1 的 encoder SWA 有界重放明确不支持它。易错：和 encoder 重放一起开。 |
| `radix_cache_backend` | --radix-cache-backend | `Optional[str]` | `None` | 作用：选用已经用 register_radix_cache_backend 注册过的前缀树后端。不写则走内置选择链。默认与效果：默认空，用内置 Radix 或 Unified。交互与改写：flexkv 也可以通过 enable_flexkv 进入这条链。易错：写了没注册的名字。 |
| `enable_hierarchical_cache` | --enable-hierarchical-cache（--enable-hicache） | `bool` | `False` | 作用：打开分层缓存 HiCache，GPU 页下面还有 host 或存储层。默认与效果：默认关闭。交互与改写：和直接连外部存储的 unified external linker 互斥。易错：两个一起开。 |
| `hicache_host_memory_mode` | --hicache-host-memory-mode | `str` | `'cache'` | 作用：主机内存是持久的 cache 层，还是 GPU 和存储之间的临时 buffer_only。buffer_only 必须再指定存储后端。默认与效果：默认按 cache。交互与改写：比例的默认值随这个模式变：cache 是 2.0，buffer_only 是 1.2。易错：buffer_only 却不给存储后端。 |
| `hicache_ratio` | --hicache-ratio | `Optional[float]` | `None` | 作用：主机 KV 池相对设备池的大小比例。cache 模式默认 2.0，buffer_only 默认 1.2，只做备份的 decode 收缩 host 池是 0.2。默认与效果：默认空，按模式套上面的数。交互与改写：hicache_size 若给出，按 GB 覆盖这个比例。易错：两种都写，以为比例还生效。 |
| `hicache_size` | --hicache-size | `int` | `0` | 作用：主机 KV 池的 GB 数，会覆盖 hicache_ratio。默认与效果：默认空，用比例。交互与改写：和 Engram host 表、cpu_offload 抢同一台机器的内存。易错：设得比机器内存还大。 |
| `hicache_host_memory_fraction` | --hicache-host-memory-fraction | `Optional[float]` | `0.8` | 作用：本机所有 rank 的 HiCache host 池一共可以用多少可用主机内存。上限会看 cgroup 的 memory.max/memory.high，并先留 10 GiB。只在既没写 size 也没写 ratio 时适用。默认与效果：默认按实现。交互与改写：多 rank 共享这一份预算，不是每卡一份。易错：四卡各自按 0.5 估算，加起来超过机器。 |
| `hicache_write_policy` | --hicache-write-policy | `str` | `'write_through'` | 作用：分层缓存的写策略，决定新 KV 何时写到 host 或存储。默认与效果：默认按实现。交互与改写：写得更勤会增加 PCIe 流量，命中别的请求更快。易错：和 io 后端的能力不匹配。 |
| `hicache_io_backend` | --hicache-io-backend | `str` | `'kernel'` | 作用：CPU 和 GPU 之间搬 KV 用的 IO 后端。默认与效果：默认按平台。交互与改写：和存储后端不是同一层，一个是 GPU-CPU，一个是 CPU-远端。易错：把 mooncake 写到这里，它属于存储后端。 |
| `hicache_mem_layout` | --hicache-mem-layout | `str` | `'page_first'` | 作用：分层缓存主机池的内存布局。默认与效果：默认按实现。交互与改写：必须和 GPU 侧 KV 布局能互转，V4 的压缩布局不一定能直接落盘。易错：V4.1 换了一个不认识压缩 KV 的布局。 |
| `hicache_storage_backend` | --hicache-storage-backend | `Optional[str]` | `None` | 作用：分层 KV 的存储后端。内置有 file、mooncake、npu_memcache、hf3fs、nixl、aibrix、tensorcast、seaweedfs。自定义后端用 extra_config 给 backend_name、module_path、class_name。默认与效果：默认空。交互与改写：buffer_only 模式必须有它。易错：模块路径不在 PYTHONPATH。 |
| `hicache_storage_prefetch_policy` | --hicache-storage-prefetch-policy | `str` | `'timeout'` | 作用：何时停止从存储预取。默认与效果：默认 timeout。交互与改写：和重试次数、轮询间隔一起决定排队请求等多久。易错：timeout 很短，预取总是半途停。 |
| `hicache_storage_backend_extra_config` | --hicache-storage-backend-extra-config | `Optional[str]` | `None` | 作用：存储后端的额外配置。可以是 JSON 字符串，或以 @ 开头指向 JSON/YAML/TOML 文件。默认与效果：默认空。交互与改写：动态后端的类名和模块路径放在这里。易错：@ 文件路径不存在。 |
| `hicache_storage_prefetch_retry_poll_interval` | --hicache-storage-prefetch-retry-poll-interval | `int` | `8` | 作用：预取没找到、备份可能还在提交时，排队请求再隔多少个调度轮次重新检查。等到暂存或匹配移动则下一轮就重发。只有走到预填充调度的轮次才计数。0 表示不重试未命中。默认与效果：默认 8。交互与改写：负载高时第一次检查可能早于提交完成，所以才要重试。易错：设成 0，竞态下的未命中不再重试。 |
| `hicache_storage_prefetch_retry_max_attempts` | --hicache-storage-prefetch-retry-max-attempts | `int` | `8` | 作用：排队请求最多重新检查存储多少次。超过后用设备上已有的 KV 接纳。0 表示不再重发。默认与效果：默认 8。交互与改写：和轮询间隔相乘大致是最长等待。易错：设得很大，请求在队列里等存储，TTFT 变长。 |
| `enable_unified_cache_external_linker` | --enable-unified-cache-external-linker | `bool` | `False` | 作用：UnifiedRadixCache 直接连外部 KV 存储，当作 L3，中间没有 host 缓存层。默认与效果：默认关闭。交互与改写：和普通 HiCache 的分层互斥。易错：两个都开。 |
| `unified_cache_external_linker_backend` | --unified-cache-external-linker-backend | `str` | `'mooncake'` | 作用：上面这条直连用的存储后端。默认与效果：默认 mooncake。交互与改写：动态后端的类名放在 linker 配置里，不在这个字符串里。易错：写成 file，但直连路径没实现 file。 |
| `unified_cache_external_linker_config` | --unified-cache-external-linker-config | `Optional[Dict[str, Any]]` | `None` | 作用：直连后端的 JSON。内置后端直接写存储选项。动态后端需要 linker 类名、linker_module_path，以及可选的 linker_extra_config。即使直连还没开也可以先写配置。对内置后端，给出它就替代 hicache 的 extra config；外部链接再回退到旧旗标已废弃。普通 HiCache 用旧旗标不受影响。默认与效果：默认空。交互与改写：动态后端缺模块路径会在导入时失败。易错：把 HiCache 的配置格式原样塞进来。 |
| `enable_linker_mla_dedup` | --enable-linker-mla-dedup | `bool` | `False` | 作用：MLA 的复制 KV 只在 rank 0 加载，再用 Mooncake linker 按层广播。默认与效果：默认关闭。交互与改写：需要 linker 后端是 mooncake。易错：后端是别的，却打开去重。 |
| `enable_hisparse` | --enable-hisparse | `bool` | `False` | 作用：打开分层稀疏注意力 HiSparse。默认与效果：默认关闭。交互与改写：V4.1 特征检查拒绝 HiSparse。PD decode radix 也和它不兼容。易错：V4.1 打开它。 |
| `hisparse_config` | --hisparse-config（--hierarchical-sparse-attention-extra-config） | `Optional[str]` | `None` | 作用：HiSparse 的 JSON，例如 top_k、device_buffer_size、host_to_device_ratio。默认与效果：默认空。交互与改写：只在 enable_hisparse 为真时读取。易错：JSON 缺 top_k。 |
| `enable_lmcache` | --enable-lmcache | `bool` | `False` | 作用：用 LMCache 作为另一种分层缓存。默认与效果：默认关闭。交互与改写：和 flexkv、内置 HiCache 都在自动选择链上，不要多个同时当默认。易错：没给 YAML 就打开。 |
| `lmcache_config_file` | --lmcache-config-file | `Optional[str]` | `None` | 作用：LMCache 的 YAML 配置路径。默认与效果：默认空。交互与改写：enable_lmcache 为真时必须能读到。易错：路径是 JSON。 |
| `enable_flexkv` | --enable-flexkv | `bool` | `False` | 作用：默认前缀树走 FlexKV 的 KVManager，把 KV 换到 CPU、SSD 或远端。等价于 radix_cache_backend=flexkv，并且和 enable_lmcache 一起参与自动选择。默认与效果：默认关闭。交互与改写：配置文件或环境变量 FLEXKV_CONFIG_PATH 二选一。易错：和 LMCache 同时打开，选择链不确定你以为的那个。 |
| `flexkv_config_file` | --flexkv-config-file | `Optional[str]` | `None` | 作用：FlexKV 的 YAML 或 JSON 配置。等价于设置 FLEXKV_CONFIG_PATH。默认与效果：默认空。交互与改写：只在走 FlexKV 时读取。易错：环境变量和这个旗标指向不同文件。 |

### 调度

命名空间 `schedule`，类 `Schedule`，`python/sglang/srt/arg_groups/fields/schedule.py`，41 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `mem_fraction_static` | --mem-fraction-static | `Optional[float]` | `None` | 作用：静态占用（权重加 KV 池）占显存的比例。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：S6 的 _profile_available_bytes 用它留出激活。OOM 时调小。Engram 若在 GPU 上，会和这个预算抢。 易错：V4.1 的 Engram 表没放 host，却把比例设得很高。 |
| `max_running_requests` | --max-running-requests | `Optional[int]` | `None` | 作用：同时处于 running 的请求上限。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：V4 模型 hook 在未设置时可能改成 256。它受 KV 池容量约束，设得再大也不会超过 S6 算出来的槽。 易错：把它当成吞吐旋钮，显存已经满了仍继续加大。 |
| `max_queued_requests` | --max-queued-requests | `Optional[int]` | `None` | 作用：等待队列最多放多少请求，超过就拒绝新请求。默认与效果：默认按实现或空表示不额外封顶。交互与改写：和 max_running_requests 不同，一个是排队，一个是正在跑。易错：只加大 running，队列仍被这个数卡住。 |
| `max_total_tokens` | --max-total-tokens | `Optional[int]` | `None` | 作用：池里允许的 token 总数上限。空则由显存 profiling 决定。默认与效果：默认空，S6 按剩余显存算。交互与改写：手写的上限不能超过实际页数。易错：写成上下文长度乘并发，大于 144GB 能放下的页。 |
| `chunked_prefill_size` | --chunked-prefill-size | `Optional[int]` | `None` | 作用：一轮预填充最多吃多少 token。-1 表示关闭，整段 extend。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：attention DP 时 S2 会除以 attn_dp_size。V4 走 trtllm attention 时要求它大于 0。 易错：开了 attn_dp_size 4 仍按未除过的块估计显存。 |
| `prefill_decode_interval` | --prefill-decode-interval（S2 可改写） | `Optional[int]` | `None` | 作用：每隔多少步才允许插入一次预填充，用来保护解码延迟。默认与效果：默认按实现。交互与改写：还有一个最小批大小的配套字段。易错：设成很大，新请求的 TTFT 变得极差。 |
| `prefill_decode_interval_min_bs` | --prefill-decode-interval-min-bs | `int` | `0` | 作用：解码批至少有这么大，才执行 prefill_decode_interval 的节流。默认与效果：默认按实现。交互与改写：批很小时间隔不生效，预填充可以插进来。易错：两个字段只改了一个。 |
| `enable_dynamic_chunking` | --enable-dynamic-chunking | `bool` | `False` | 作用：按负载动态调整分块预填充的块大小，而不是始终用 chunked_prefill_size。默认与效果：默认关闭。交互与改写：attention DP 仍会先把基准块除以组数。易错：开了动态分块又把块大小设成 -1，没有块可调。 |
| `max_prefill_tokens` | --max-prefill-tokens | `int` | `16384` | 作用：一批预填充 token 的上限，和 chunk 一起限制 S9 第 2 步。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：受 KV 剩余页数约束。 易错：设得比 chunked_prefill_size 还小，chunk 形同虚设。 |
| `prefill_max_requests` | --prefill-max-requests | `Optional[int]` | `None` | 作用：一批里最多同时预填充多少个请求。默认与效果：默认按实现。交互与改写：和 max_prefill_tokens 一起卡住预填充。易错：请求数没到上限，token 数先到了，看起来像这个开关没生效。 |
| `schedule_policy` | --schedule-policy | `str` | `'fcfs'` | 作用：waiting 队列的排序。lpm 等看前缀树，fcfs 等不看。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：disable_radix_cache 时看树的策略会被换掉。 易错：关了 radix 仍期待 lpm 生效。 |
| `enable_priority_scheduling` | --enable-priority-scheduling | `bool` | `False` | 作用：按请求的 priority 排序，而不是只按到达顺序。默认与效果：默认关闭。交互与改写：关闭时如果请求仍带优先级，abort_on_priority_when_disabled 决定是拒绝还是忽略。易错：客户端传了 priority，服务没开这个开关。 |
| `disable_priority_preemption` | --disable-priority-preemption | `bool` | `False` | 作用：即使开了优先级，也不允许高优先级抢占正在跑的请求。默认与效果：默认不禁用抢占，也就是允许抢占（在优先级调度打开时）。交互与改写：名字是否定式。易错：想禁止抢占却把这个旗标留成 false。 |
| `default_priority_value` | --default-priority-value | `Optional[int]` | `None` | 作用：请求没写 priority 时用的默认优先级。默认与效果：默认按实现，通常是 0 或中性值。交互与改写：schedule_low_priority_values_first 会翻转谁先跑。易错：默认值和客户端以为的「越大越优先」相反。 |
| `abort_on_priority_when_disabled` | --abort-on-priority-when-disabled | `bool` | `False` | 作用：优先级调度关着时，如果请求仍带了 priority，是直接拒绝还是忽略该字段。默认与效果：默认按实现。交互与改写：打开后，旧客户端带优先级会被 4xx。易错：网关总是塞 priority，这个开关一开全失败。 |
| `schedule_low_priority_values_first` | --schedule-low-priority-values-first | `bool` | `False` | 作用：数值小的 priority 先跑。不设则数值大的先跑。默认与效果：默认按「大的先」。交互与改写：和 default_priority_value 一起决定缺省请求排在哪。易错：一半服务开了这个翻转，队列顺序对不上。 |
| `priority_scheduling_preemption_threshold` | --priority-scheduling-preemption-threshold | `int` | `10` | 作用：优先级差到这个阈值才允许抢占，避免一点点差别就打断正在跑的请求。默认与效果：默认按实现。交互与改写：disable_priority_preemption 为真时阈值没有机会触发。易错：阈值设成 0，任何更高优先级都抢占。 |
| `retraction_policy` | --retraction-policy | `str` | `'length'` | 作用：显存不够时怎么收缩正在跑的请求，例如丢掉谁的 KV。默认与效果：默认按实现。交互与改写：PD decode 的 retraction backup 决定收缩后 KV 还在不在。易错：策略会中止请求，但客户端没有重试。 |
| `schedule_conservativeness` | --schedule-conservativeness | `float` | `1.0` | 作用：调度器估计新 decode 还能塞多少的系数，越大越保守。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：attention DP 打开时乘 0.3。 易错：在命令行设了 1，运行中却发现更激进，因为改写发生在 S2。 |
| `page_size` | --page-size（S2 可改写） | `Optional[int]` | `None` | 作用：KV 页的 token 数。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：V4 override 把它改成 256（NPU 128）。Radix 命中按页对齐。 易错：手写 16，以为能提高短前缀命中率。S2 会改掉。 |
| `c128_page_size` | --c128-page-size | `int` | `16` | 作用：压缩比为 128 的那类 KV 的页大小，和普通 page_size 分开。默认与效果：默认按实现。交互与改写：V4 的 compress_ratios 里没有 128 时这个页用不上。易错：只改 page_size，C128 层仍用旧页。 |
| `swa_full_tokens_ratio` | --swa-full-tokens-ratio（S2 可改写） | `Optional[float]` | `None` | 作用：滑动窗口池相对全量 token 容量的比例。V4 非 V4.1 在未设置时会被改成 0.1。V4.1 不套这个 0.1。默认与效果：默认空。交互与改写：显式设置会被 _swa_full_tokens_ratio_explicitly_set 记住，模型默认不能覆盖。易错：V4.1 上以为会被改成 0.1。 |
| `_swa_full_tokens_ratio_explicitly_set` | ---swa-full-tokens-ratio-explicitly-set（无常规 CLI） | `Optional[bool]` | `None` | 作用：内部标记，区分用户写了 SWA 比例还是留空。没有 CLI。默认与效果：由 argv 决定。交互与改写：和 radix 策略的显式标记是同一类机制。易错：测试里不设它，模型钩子覆盖了你以为的默认。 |
| `swa_prefix_tails` | --swa-prefix-tails | `Optional[int]` | `None` | 作用：SWA 前缀在树上保留的尾部长度，命中后还能接上窗口。默认与效果：默认按实现。交互与改写：和 sliding_window 本身不是同一个数。易错：设得比窗口还短，命中后接不上。 |
| `disable_hybrid_swa_memory` | --disable-hybrid-swa-memory（S2 可改写） | `bool` | `False` | 作用：不用混合 SWA 的单独内存池，改成别的布局。默认与效果：默认不禁用。交互与改写：V4 依赖混合池来同时放压缩 KV 和窗口。易错：V4.1 关掉它，建池失败或窗口层写错地方。 |
| `prefill_only_disable_kv_cache` | --prefill-only-disable-kv-cache | `bool` | `False` | 作用：只做预填充、不保留 KV。适合只要提示 logprob、不要解码的场景。默认与效果：默认关闭。交互与改写：S2 前半和后半各校验一次，因为要等注意力后端定稿。易错：还想解码却打开它，没有 KV 可续。 |
| `disable_chunked_prefix_cache` | --disable-chunked-prefix-cache | `bool` | `False` | 作用：关掉分块前缀缓存。某些注意力后端才支持分块前缀，不支持时必须关。默认与效果：默认不关。交互与改写：和 disable_radix_cache 不同，一个关的是分块写入，一个关整棵树。易错：后端不支持却留着开。 |
| `disable_overlap_schedule` | --disable-overlap-schedule（S2 可改写） | `bool` | `False` | 作用：关掉把下一步 CPU 调度和上一步 GPU 重叠的调度。默认与效果：默认不关。交互与改写：pp 大于 1 时解析会强制关掉。易错：V3 脚本依赖 overlap，PP 打开后被静默关掉。 |
| `num_continuous_decode_steps` | --num-continuous-decode-steps | `int` | `1` | 作用：连续做多少步解码再回调度器看新请求。默认与效果：默认按实现。交互与改写：步数大则 TTFT 变差、解码吞吐更好。易错：设得很大，新请求长时间进不了批。 |
| `scheduler_recv_interval` | --scheduler-recv-interval | `int` | `1` | 作用：调度器隔多少步才非阻塞收一次 ZMQ。默认与效果：默认按实现，通常每步都收。交互与改写：调大能减 CPU，但请求进入变慢。易错：和 decode_log_interval 看成同一个旋钮。 |
| `enable_mixed_chunk` | --enable-mixed-chunk | `bool` | `False` | 作用：分块预填充和正在解码的请求放进同一批。默认与效果：默认关闭。交互与改写：V4.1 的 encoder SWA 重放不支持它。易错：打开后 MIXED 批的形状和纯解码图对不上。 |
| `max_mamba_cache_size` | --max-mamba-cache-size | `Optional[int]` | `None` | 作用：Mamba 状态缓存的上限。默认与效果：默认按实现。交互与改写：和 mamba_max_states_per_path 一个管总量一个管每条路径。易错：在没有 Mamba 的模型上把它当 KV 上限。 |
| `mamba_full_memory_ratio` | --mamba-full-memory-ratio（S2 可改写） | `Optional[float]` | `None` | 作用：混合模型里 Mamba 状态占内存预算的比例。默认与效果：默认按实现。交互与改写：和 swa_full_tokens_ratio 一起分割同一块预算。易错：比例加起来超过 1。 |
| `enable_prefill_delayer` | --enable-prefill-delayer | `bool` | `False` | 作用：在高水位时推迟预填充，把解码让出来。默认与效果：默认关闭。交互与改写：推迟多久由下面一组 delayer 字段决定。易错：只开开关，水位和延迟都是默认，行为和预期差一截。 |
| `prefill_delayer_max_delay_passes` | --prefill-delayer-max-delay-passes | `int` | `30` | 作用：预填充最多被推迟多少个调度轮次。默认与效果：默认按实现。交互与改写：和 max_delay_ms 谁先到谁算数。易错：轮次很大但每轮很短，实际延迟仍可接受；或相反。 |
| `prefill_delayer_token_usage_low_watermark` | --prefill-delayer-token-usage-low-watermark | `Optional[float]` | `None` | 作用：token 占用低于这个水位才允许被推迟的预填充进来。默认与效果：默认按实现。交互与改写：水位高于日常占用时，预填充会一直被推迟。易错：水位设成 0.99，几乎从不推迟，开关像没开。 |
| `prefill_delayer_forward_passes_buckets` | --prefill-delayer-forward-passes-buckets | `Optional[List[float]]` | `None` | 作用：推迟了多少轮的直方图桶。默认与效果：默认内置。交互与改写：只影响指标。易错：当成推迟策略本身。 |
| `prefill_delayer_wait_seconds_buckets` | --prefill-delayer-wait-seconds-buckets | `Optional[List[float]]` | `None` | 作用：预填充等待秒数的直方图桶。默认与效果：默认内置。交互与改写：同上，只统计。易错：单位设成毫秒。 |
| `prefill_delayer_queue_min_ratio` | --prefill-delayer-queue-min-ratio | `Optional[float]` | `None` | 作用：等待队列相对运行批至少到这个比例，推迟逻辑才考虑动手。默认与效果：默认按实现。交互与改写：队列很短时不会推迟。易错：比例设成 1，只有队列不短于运行批才推迟。 |
| `prefill_delayer_max_delay_ms` | --prefill-delayer-max-delay-ms | `Optional[float]` | `None` | 作用：单次预填充最多被推迟多少毫秒。默认与效果：默认按实现。交互与改写：和最大轮次共同封顶。易错：设成 0，等于不允许推迟。 |
| `min_free_slots_delay` | --min-free-slots-delay | `Optional[int]` | `None` | 作用：空闲槽位少到这个数就推迟接纳。和 PP 一起用会被拒绝，因为微批槽位被 pp_max_micro_batch_size 卡住，阈值可能永远达不到。默认与效果：默认空，不启用。交互与改写：validation_hook 在 pp 大于 1 时 assert。易错：PP 上设置它。 |

### HTTP 服务、解析器与 API

命名空间 `serving`，类 `Serving`，`python/sglang/srt/arg_groups/fields/serving.py`，64 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `enable_response_store` | --enable-response-store | `bool` | `False` | 作用：在内存里保存 Responses API 的结果，用来取回、串联和后台请求。默认关闭。不支持 PD 分离。存储没有 TTL 也没有大小上限。默认与效果：默认关闭。交互与改写：PD 模式下打开会在解析时失败。易错：长跑打开它，内存随历史响应涨。 |
| `tokenizer_path` | --tokenizer-path | `Optional[str]` | `None` | 作用：分词器路径。空则用模型路径。默认与效果：默认空。交互与改写：gRPC 模式若空，run_server 会先把它设成 model_path 再解析。易错：分词器和权重不是同一套词表。 |
| `tokenizer_mode` | --tokenizer-mode | `str` | `'auto'` | 作用：分词器模式，例如 slow 或 fast。默认与效果：默认 auto 或 fast。交互与改写：和 tokenizer_backend 一个是实现库，一个是快慢模式。易错：fast 模式和某些自定义 tokenizer.json 不兼容。 |
| `tokenizer_backend` | --tokenizer-backend | `str` | `'huggingface'` | 作用：分词库。huggingface 用 Hugging Face tokenizers；fastokens 更快，但要安装对应包。默认与效果：默认 huggingface。交互与改写：fastokens 没装会在 S4 失败。易错：为了速度切换后，特殊 token 和 HF 不一致。 |
| `tokenizer_worker_num` | --tokenizer-worker-num | `int` | `1` | 作用：分词管理器的工作进程数。默认与效果：默认 1。交互与改写：大于 1 时每个 worker 有自己的 IPC，多模态预处理预算还会被拆开。易错：设得很大，模型加载被复制多份。 |
| `detokenizer_worker_num` | --detokenizer-worker-num | `int` | `1` | 作用：反分词进程数。大于 1 时会再起一个路由器。默认与效果：默认 1。交互与改写：流式输出的顺序由路由器保证。易错：多个 worker 但客户端假定单连接有序，看到乱序片段。 |
| `skip_tokenizer_init` | --skip-tokenizer-init | `bool` | `False` | 作用：不在主进程建分词器。scheduler 把 token id 直接送回。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：调用方必须自己传 input_ids，S11 的文本路径被跳过。 易错：仍用 /v1/chat/completions 传文本。 |
| `host` | --host | `str` | `'127.0.0.1'` | 作用：HTTP 绑定地址。 默认与效果：默认常见是 127.0.0.1。 交互与改写：0.0.0.0 时内部自检改用 127.0.0.1。多机 rank 0 要能被其它节点访问的是 dist_init_addr，不是这个 host。 易错：把 host 当成 NCCL 的地址。 |
| `port` | --port | `int` | `30000` | 作用：HTTP 端口。 默认与效果：默认 30000。 交互与改写：node_rank>0 不起这个端口。 易错：四台机器都去 curl 本机 30000，只有 rank 0 有服务。 |
| `fastapi_root_path` | --fastapi-root-path | `str` | `''` | 作用：服务挂在反向代理的子路径后面时，FastAPI 的 root_path。默认与效果：默认空，挂在根上。交互与改写：不影响路由匹配本身，影响生成的 URL。易错：代理剥了前缀但这里仍写子路径，重定向错。 |
| `enable_http_worker_reuseport` | --enable-http-worker-reuseport | `bool` | `False` | 作用：每个 Uvicorn worker 用 SO_REUSEPORT 各自监听。默认与效果：默认关闭。交互与改写：多 worker 时内核做负载均衡。易错：和单进程的 scheduler 数量搞混，HTTP worker 不是 DP 组。 |
| `smg_grpc_mode` | --smg-grpc-mode | `bool` | `False` | 作用：用旧的 SMG gRPC 服务代替 HTTP。默认与效果：默认关闭。交互与改写：--grpc-mode 已废弃并折进这个开关。原生 gRPC 是另一个开关 grpc_port。易错：想开原生 gRPC 却写了这个。 |
| `grpc_mode` | --grpc-mode | `bool` | `False` | 作用：已废弃，改用 --smg-grpc-mode。默认与效果：默认关闭。交互与改写：解析时折进 smg_grpc_mode。易错：和新的 grpc_port 同时写，开了两套 gRPC。 |
| `grpc_port` | --grpc-port | `Optional[int]` | `None` | 作用：原生 gRPC 端口，和 HTTP 一起起。设置它或环境变量 SGLANG_GRPC_PORT 才启用。默认关。跟随节点只暴露 GetServerInfo。在旧 SMG 模式下这是 SMG 端口，默认是 HTTP 端口加 10000。默认与效果：默认空，不起原生 gRPC。交互与改写：边车需要这个端口。易错：和 30000 撞车。 |
| `grpc_response_timeout_secs` | --grpc-response-timeout-secs | `int` | `300` | 作用：等每一个原生 gRPC 响应块的秒数，包括第一块。必须是正数。默认与效果：默认 300。交互与改写：长预填充可能超过它。易错：设成 1，首 token 还没出来就超时。 |
| `grpc_worker_threads` | --grpc-worker-threads（无常规 CLI） | `Optional[int]` | `None` | 作用：原生 gRPC 的工作线程数，主要来自环境变量 SGLANG_GRPC_WORKER_THREADS，放进字段是为了配置投影能看见。默认与效果：默认空。交互与改写：不是 HTTP worker 数。易错：在命令行找这个旗标，它可能只认环境变量。 |
| `sidecar` | --sidecar | `Optional[str]` | `None` | 作用：对着原生 gRPC 起一个本地边车。模块要有 main(argv)，并从 SGLANG_GRPC_ENDPOINT 读地址。需要 grpc_port。Rust server 时每个有监听的节点起一个边车。默认与效果：默认空，不起边车。交互与改写：模块导入失败会让启动失败。易错：没开 grpc_port 就写边车。 |
| `sidecar_args` | --sidecar-args | `Optional[List[str]]` | `None` | 作用：传给边车 main 的 JSON 数组。--sidecar-shutdown-timeout 由 SGLang 自己消费。默认与效果：默认空。交互与改写：必须是 JSON 数组。易错：写成普通命令行字符串。 |
| `skip_server_warmup` | --skip-server-warmup | `bool` | `False` | 作用：跳过启动预热。默认与效果：默认不跳过。交互与改写：跳过则第一次真实请求承担编译和图捕获剩余成本。易错：为了快启动跳过，线上第一个用户很慢。 |
| `warmups` | --warmups | `Optional[str]` | `None` | 作用：启动监听前要跑的预热函数名，逗号分隔，函数在 warmup.py 里。默认与效果：默认空，跑内置预热。交互与改写：和 skip_server_warmup 同时用时，跳过赢。易错：函数名拼错，启动失败。 |
| `enable_http2` | --enable-http2 | `bool` | `False` | 作用：用 Granian 代替 Uvicorn，从而协商 HTTP/1.1 和 HTTP/2。需要安装 sglang[http2]。默认与效果：默认关闭。交互与改写：下面两个窗口参数只在这里打开时生效。易错：没装可选依赖就打开。 |
| `http2_max_concurrent_streams` | --http2-max-concurrent-streams | `int` | `200` | 作用：每个 HTTP/2 连接广告的最大并发流，范围 1 到 2^32-1。默认与效果：默认 200。交互与改写：只在 enable_http2 时使用。易错：设得比网关还大，服务先被打满。 |
| `http2_initial_connection_window_size` | --http2-initial-connection-window-size | `int` | `1024 * 1024` | 作用：HTTP/2 连接级接收窗口的初始字节数，范围 1024 到 2^31-1。默认与效果：默认 1 MiB。交互与改写：只在 HTTP/2 打开时使用。易错：窗口太小，大请求被流控卡住。 |
| `ssl_keyfile` | --ssl-keyfile | `Optional[str]` | `None` | 作用：HTTPS 私钥路径。默认与效果：默认空，走 HTTP。交互与改写：和证书必须成对。给出证书后，内部 URL 用 https。易错：只给了证书没给私钥。 |
| `ssl_certfile` | --ssl-certfile | `Optional[str]` | `None` | 作用：HTTPS 证书路径。默认与效果：默认空。交互与改写：有证书时 url() 用 https。易错：证书和私钥不匹配，监听起不来。 |
| `ssl_ca_certs` | --ssl-ca-certs | `Optional[str]` | `None` | 作用：校验客户端证书时用的 CA 包。默认与效果：默认空，不校验客户端证书。交互与改写：只在双向 TLS 时需要。易错：把服务器证书填到这里。 |
| `ssl_keyfile_password` | --ssl-keyfile-password | `Optional[str]` | `None` | 作用：私钥密码。默认与效果：默认空，私钥没有密码。交互与改写：会出现在进程参数里，注意不要打进日志。易错：密码错，TLS 监听失败。 |
| `enable_ssl_refresh` | --enable-ssl-refresh | `bool` | `False` | 作用：运行中重新加载证书。默认与效果：默认关闭。交互与改写：依赖证书路径在进程里仍可读。易错：证书换了路径，刷新仍读旧文件。 |
| `api_key` | --api-key | `Optional[str]` | `None` | 作用：若设置，HTTP 要求客户端带同样的密钥。 默认与效果：默认空，不校验。 交互与改写：只影响 S7，不进 scheduler。 易错：把密钥写进进程列表。05 不会重复密钥的值。 |
| `admin_api_key` | --admin-api-key | `Optional[str]` | `None` | 作用：管理接口用的密钥，和普通 api_key 分开。默认与效果：默认空，管理接口不单独鉴权。交互与改写：权重更新、清缓存这类接口应该用它。易错：和管理密钥、用户密钥设成同一个，权限没有分开。 |
| `served_model_name` | --served-model-name | `Optional[str]` | `None` | 作用：/v1/models 和响应里展示的名字，可以和路径不同。 默认与效果：默认用路径推断。 交互与改写：LoRA 可以用 base:adapter 这种名字。 易错：客户端 model 字段必须对上这个名字，对不上会 404。 |
| `weight_version` | --weight-version | `str` | `'default'` | 作用：对外报告的权重版本号，热更新后用来让客户端知道权重变了。默认与效果：默认空或初始版本。交互与改写：update_weight_version 接口会改它。易错：热更新了权重但没bump 版本，客户端仍用旧缓存。 |
| `chat_template` | --chat-template | `Optional[str]` | `None` | 作用：对话模板路径或内置名。空则用分词器自带的。默认与效果：默认空。交互与改写：reasoning 和 tool 的 auto 检测依赖最终模板。易错：模板和解析器不是同一代模型的。 |
| `trust_request_chat_template` | --trust-request-chat-template | `bool` | `False` | 作用：允许请求自己带模板。默认关闭，避免客户端注入模板。默认与效果：默认不信任。交互与改写：打开后请求可以覆盖服务端模板。易错：公网打开它。 |
| `hf_chat_template_name` | --hf-chat-template-name | `Optional[str]` | `None` | 作用：当仓库里有多套模板时，选 Hugging Face 模板的名字。默认与效果：默认空，用默认那套。交互与改写：和 chat_template 同时给时要看谁覆盖谁。易错：名字在 tokenizer_config 里不存在。 |
| `completion_template` | --completion-template | `Optional[str]` | `None` | 作用：补全接口用的模板，和对话模板分开。默认与效果：默认空。交互与改写：/v1/completions 走它，chat 不走。易错：只改了对话模板，补全接口仍是旧的。 |
| `file_storage_path` | --file-storage-path | `Optional[str]` | `None` | 作用：文件接口存放上传文件的目录。默认与效果：默认按实现。交互与改写：和媒体 URL 下载不是同一个目录。易错：目录不可写，上传接口失败。 |
| `enable_cache_report` | --enable-cache-report | `bool` | `False` | 作用：在响应里报告前缀缓存命中了多少 token。默认与效果：默认关闭。交互与改写：和 return_cached_tokens_details 这类请求字段配套。易错：服务没开，客户端要了详情却是空的。 |
| `return_input_ids` | --return-input-ids | `bool` | `False` | 作用：服务默认在响应里带上输入 token id。默认与效果：默认按字段。交互与改写：请求级 return_prompt_token_ids 还可以再覆盖。易错：默认打开后响应变大。 |
| `return_output_ids` | --return-output-ids | `bool` | `False` | 作用：服务默认在响应里带上输出 token id。默认与效果：默认按字段。交互与改写：和流式增量输出一起用时，每帧都可能带 id。易错：带宽被 id 列表占满。 |
| `reasoning_parser` | --reasoning-parser | `Optional[str]` | `None` | 作用：把思考段从正文里拆出来。choices 含 auto 和 reasoning_parser_names。 默认与效果：默认按字段声明。 交互与改写：auto 在 S4 按 chat template 写成具体名，V4.1 对应 deepseek-v41。 易错：手写 deepseek-r1 去解析 V4.1 的模板。 |
| `default_chat_template_kwargs` | --default-chat-template-kwargs | `Optional[Dict[str, Any]]` | `None` | 作用：调用模板时默认传入的参数字典。默认与效果：默认空。交互与改写：请求里的 chat_template_kwargs 可以覆盖。易错：默认把 thinking 设死，请求关不掉。 |
| `strip_thinking_cache` | --strip-thinking-cache | `bool` | `False` | 作用：写入前缀缓存前去掉思考段，避免思考 token 被后续请求复用。默认与效果：默认按实现。交互与改写：和 reasoning parser 要配套，否则切不干净。易错：关掉后，下一次请求命中了上一次的思考过程。 |
| `enable_strict_thinking` | --enable-strict-thinking | `bool` | `False` | 作用：严格思考模式，预算和结束标记才会被真正执行，而不是只当元数据。默认与效果：默认关闭。交互与改写：请求的 max_thinking_tokens 需要它才强制截断。易错：客户端传了思考预算，服务没开严格模式，预算被忽略。 |
| `tool_call_parser` | --tool-call-parser | `Optional[str]` | `None` | 作用：工具调用的解析器。auto 在 S4 选定。V4.1 有 DeepSeekV41Detector。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：和 chat template 必须配套，sampler 不负责拆工具。 易错：只开 parser、模板却不是工具模板，调用边界对不齐。 |
| `tool_server` | --tool-server | `Optional[str]` | `None` | 作用：工具调用要回调的外部工具服务地址。默认与效果：默认空，不回调。交互与改写：和 tool_call_parser 一个负责拆调用，一个负责执行。易错：只配了解析器，工具实际没执行。 |
| `sampling_defaults` | --sampling-defaults | `str` | `'model'` | 作用：服务端采样参数的默认 JSON，请求没写的字段用这里的。默认与效果：默认空，用 SamplingParams 的代码默认。交互与改写：请求显式值优先。易错：默认 temperature 设成 0，所有没写温度的请求变成贪心。 |
| `asr_max_buffer_seconds` | --asr-max-buffer-seconds | `int` | `60` | 作用：语音识别会话最多缓冲多少秒音频。默认与效果：默认按实现。交互与改写：超过就截断或拒绝。易错：长音频会话被默默截断。 |
| `asr_max_concurrent_sessions` | --asr-max-concurrent-sessions | `int` | `32` | 作用：同时多少路语音识别会话。默认与效果：默认按实现。交互与改写：和文本请求的 max_running_requests 是两套并发。易错：只加大文本并发，语音仍被这个数卡住。 |
| `preferred_sampling_params` | --preferred-sampling-params | `Optional[str]` | `None` | 作用：服务建议的采样参数，可以出现在模型信息里供客户端采用。默认与效果：默认空。交互与改写：不强制覆盖请求。易错：以为它会改掉请求里的 temperature。 |
| `allow_auto_truncate` | --allow-auto-truncate | `bool` | `False` | 作用：输入超过上下文时自动截断，而不是直接报错。默认与效果：默认不截断。交互与改写：截断从哪一头切要看实现，通常丢前面的历史。易错：打开后用户以为整段提示都在。 |
| `stream_interval` | --stream-interval | `int` | `1` | 作用：流式输出每隔多少个 token 推一帧。默认与效果：默认按实现，往往是 1。交互与改写：请求级 sampling 里也可以再设。易错：设得很大，客户端长时间没有增量。 |
| `batch_notify_size` | --batch-notify-size | `int` | `16` | 作用：攒够多少个结果再通知一次分词端。默认与效果：默认按实现。交互与改写：调大能减 IPC，延迟变差。易错：和 stream_interval 叠乘，帧更稀。 |
| `stream_response_default_include_usage` | --stream-response-default-include-usage | `bool` | `False` | 作用：流式响应默认是否在帧里带 usage。默认与效果：默认按字段。交互与改写：请求的 stream_options 可以覆盖。易错：默认带 usage，每帧都多一块 JSON。 |
| `incremental_streaming_output` | --incremental-streaming-output | `bool` | `False` | 作用：流式每帧只发新增文本，而不是累计全文。默认与效果：默认按实现。交互与改写：客户端如果按累计拼接会重复。易错：网关按累计协议解析增量帧。 |
| `enable_streaming_session` | --enable-streaming-session | `bool` | `False` | 作用：把流式连接登记成会话，便于取消和续写。默认与效果：默认按实现。交互与改写：和 session_id 一起用。易错：没开它却依赖服务端记住流。 |
| `constrained_json_whitespace_pattern` | --constrained-json-whitespace-pattern | `Optional[str]` | `None` | 作用：JSON 约束解码里空白用什么正则。默认与效果：默认按语法后端。交互与改写：和 disable_any_whitespace 一起收紧格式。易错：模式写错，合法 JSON 也被拒。 |
| `constrained_json_disable_any_whitespace` | --constrained-json-disable-any-whitespace | `bool` | `False` | 作用：JSON 约束里不允许任意空白。默认与效果：默认允许空白。交互与改写：和最大空白次数是两种限制。易错：模型习惯在冒号后加空格，被直接约束失败。 |
| `constrained_json_max_whitespace_cnt` | --constrained-json-max-whitespace-cnt | `Optional[int]` | `None` | 作用：JSON 约束里最多允许多少处空白。默认与效果：默认按实现。交互与改写：超过则语法失败。易错：设成 0，几乎所有格式化 JSON 都失败。 |
| `enable_dynamic_batch_tokenizer` | --enable-dynamic-batch-tokenizer | `bool` | `False` | 作用：把多个请求的分词攒成一批再做。默认与效果：默认关闭。交互与改写：批大小和超时由下面两个字段决定。易错：超时设得很大，单请求也要等凑批。 |
| `dynamic_batch_tokenizer_batch_size` | --dynamic-batch-tokenizer-batch-size | `int` | `32` | 作用：动态分词批最多多少个请求。默认与效果：默认按实现。交互与改写：先到批大小或先到超时就发出。易错：设成 1，等于没批。 |
| `dynamic_batch_tokenizer_batch_timeout` | --dynamic-batch-tokenizer-batch-timeout | `float` | `0.002` | 作用：动态分词批最多等多少时间。默认与效果：默认按实现。交互与改写：低 QPS 时由超时触发。易错：超时是 0，批次永远凑不齐或立即发出。 |
| `enable_tokenizer_batch_encode` | --enable-tokenizer-batch-encode | `bool` | `False` | 作用：分词编码走批处理接口。默认与效果：默认关闭。交互与改写：和动态攒批是两层，一个是 API，一个是等待。易错：只开这个，请求仍是一个一个进来。 |
| `disable_tokenizer_batch_decode` | --disable-tokenizer-batch-decode | `bool` | `False` | 作用：关掉批量反分词，改成逐条解。默认与效果：默认不关。交互与改写：排错乱序时可以关。易错：高 QPS 关掉后 CPU 成为瓶颈。 |

### 投机解码（MTP / EAGLE / DSPARK）

命名空间 `spec`，类 `Spec`，`python/sglang/srt/arg_groups/fields/spec.py`，43 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `speculative_algorithm` | --speculative-algorithm | `Optional[str]` | `None` | 作用：投机算法名。 默认与效果：默认空，不建 draft worker。内建 EAGLE、EAGLE3、NEXTN、STANDALONE、NGRAM、DFLASH、DSPARK、UNO。 交互与改写：V4.1 只留下 DSPARK。NEXTN 会先被折成 EAGLE，然后仍被 V4.1 拒绝。 易错：把 config 里的 num_nextn_predict_layers 理解成还要再开 NEXTN。 |
| `uno_lora_path` | --uno-lora-path | `Optional[str]` | `None` | 作用：UNO 投机算法用的 LoRA 路径。默认与效果：默认空。交互与改写：只在算法是 UNO 时读取。易错：DSPARK 配置里填了它，不会被用。 |
| `speculative_draft_model_path` | --speculative-draft-model-path（--speculative-draft-model） | `Optional[str]` | `None` | 作用：草稿权重路径。 默认与效果：默认空。 交互与改写：DSPARK 若发现主 checkpoint 自带草稿，会把它改成 model_path 或子目录；否则 ValueError，必须自己传。 易错：V4.1 不传路径，也不确认权重里有没有草稿子目录。 |
| `speculative_boundary_reduction` | --speculative-boundary-reduction（无常规 CLI；S2 可改写） | `Literal['ar', 'rs', 'rsv', 'rs+rsv']` | `'rs+rsv'` | 作用：草稿侧 FFN 边界规约，语义和目标模型的 boundary_reduction 相同，但只作用于草稿。默认与效果：默认跟目标或 auto。交互与改写：草稿和目标可以不同。易错：以为改了目标的开关，草稿也会变。 |
| `speculative_draft_model_revision` | --speculative-draft-model-revision | `Optional[str]` | `None` | 作用：草稿仓库的版本。默认与效果：默认跟目标的 revision。交互与改写：DSPARK 若把草稿路径收成 model_path，revision 也会收成目标的。易错：草稿和目标用了不同 commit。 |
| `speculative_draft_load_format` | --speculative-draft-load-format | `Optional[str]` | `None` | 作用：草稿权重的加载格式，可以和目标不同。默认与效果：默认跟目标。交互与改写：远程传输引擎要按草稿自己的格式决定。易错：草稿是 GGUF，格式仍写 safetensors。 |
| `speculative_num_steps` | --speculative-num-steps | `Optional[int]` | `None` | 作用：草稿迭代步数。 默认与效果：默认空。 交互与改写：DSPARK 会把它收成 1，大于 1 只打警告并改写。 易错：把它设得很大指望接受长度变长。DSPARK 不吃这个自由度。 |
| `speculative_eagle_topk` | --speculative-eagle-topk | `Optional[int]` | `None` | 作用：EAGLE 每步保留的分支数。V4 只允许 1。V4.1 根本不允许 EAGLE。默认与效果：默认空，按算法。交互与改写：大于 1 是树状草稿，ReplaySSM spec 要求它是空或 1。易错：V4 上设成 4。 |
| `speculative_num_draft_tokens` | --speculative-num-draft-tokens | `Optional[int]` | `None` | 作用：一轮草稿最多提多少个 token。默认与效果：默认按算法。交互与改写：和接受长度、图的捕获长度一起决定显存。V4 在线压缩和这个数有断言。易错：设得比图能校验的长度还长。 |
| `speculative_dflash_block_size` | --speculative-dflash-block-size | `Optional[int]` | `None` | 作用：DFLASH 的块大小。默认与效果：默认按算法。交互与改写：不是 DSPARK 的 block。易错：把 V4.1 的 5 填到这里。 |
| `speculative_domino_candidate_pool_size` | --speculative-domino-candidate-pool-size | `int` | `2048` | 作用：Domino 候选池大小。默认与效果：默认按实现。交互与改写：只在对应算法启用时分配。易错：没开该算法仍按这个数预留显存。 |
| `speculative_dspark_block_size` | --speculative-dspark-block-size | `Optional[int]` | `None` | 作用：DSPARK 一次草稿块的大小。 默认与效果：默认由解析填。V4.1-Flash 的 config 是 5。 交互与改写：应与 checkpoint 的 dspark_block_size 一致。 易错：沿用别的模型的 block。 |
| `speculative_dspark_sps_table_path` | --speculative-dspark-sps-table-path | `Optional[str]` | `None` | 作用：DSPARK 的 SPS 表路径。默认与效果：默认空，从表是否打进检查点决定。交互与改写：缺表时草稿不能按块采样。易错：路径是另一版模型的表。 |
| `speculative_dspark_confidence_sts_path` | --speculative-dspark-confidence-sts-path | `Optional[str]` | `None` | 作用：DSPARK 置信度统计文件的路径。默认与效果：默认空。交互与改写：和接受阈值一起决定丢弃哪些草稿。易错：文件和 block size 不匹配。 |
| `speculative_dspark_align_verify_tokens_to_graph_tier` | --speculative-dspark-align-verify-tokens-to-graph-tier | `bool` | `False` | 作用：把 DSPARK 校验长度对齐到 CUDA 图的档位，避免每一种长度都捕获一张图。默认与效果：默认按实现。交互与改写：对齐会多算几个 token 的校验。易错：关了对齐，图的数量随长度爆炸。 |
| `speculative_accept_threshold_single` | --speculative-accept-threshold-single | `float` | `1.0` | 作用：单个草稿 token 的接受概率阈值，低于它就丢掉。默认与效果：默认按算法。交互与改写：设得越高，接受长度越短、越稳。易错：设成 1，几乎不接受草稿。 |
| `speculative_accept_threshold_acc` | --speculative-accept-threshold-acc | `float` | `1.0` | 作用：累积接受概率的阈值。默认与效果：默认按算法。交互与改写：和 single 阈值一起用。易错：只调一个，另一个仍很严。 |
| `speculative_use_rejection_sampling` | --speculative-use-rejection-sampling | `bool` | `False` | 作用：用拒绝采样决定草稿 token 留不留，而不是简单阈值。默认与效果：默认按算法。交互与改写：和确定性模式一起时，随机数序列必须可复现。易错：开了拒绝采样又没种子，接受与否不可复现。 |
| `speculative_use_block_verification` | --speculative-use-block-verification | `bool` | `False` | 作用：按块校验草稿，而不是逐 token。默认与效果：默认按算法。交互与改写：DSPARK 的块大小要和它一致。易错：块校验打开但 block size 是 1，没有块的收益。 |
| `speculative_token_map` | --speculative-token-map | `Optional[str]` | `None` | 作用：草稿小词表到目标词表的映射文件。默认与效果：默认空。交互与改写：词表一样的草稿不需要它。易错：映射和草稿词表版本不一致，接受的 id 是错的。 |
| `speculative_attention_mode` | --speculative-attention-mode（S2 可改写） | `str` | `'prefill'` | 作用：投机里目标校验和草稿扩展用的注意力路径，prefill 或 decode。默认与效果：默认 prefill。交互与改写：选 decode 会走解码核，长校验可能不对。易错：树状草稿用 decode 模式。 |
| `speculative_draft_attention_backend` | --speculative-draft-attention-backend（S2 可改写） | `Optional[str]` | `None` | 作用：只给草稿的注意力后端。默认与效果：默认跟目标或自动。交互与改写：草稿可以比目标更轻。易错：草稿指定了目标没有的 dsv4 变体。 |
| `speculative_dsa_topk_backend` | --speculative-dsa-topk-backend | `str` | `'sgl-kernel'` | 作用：草稿 worker 的 DSA indexer top-k 后端，可选 sgl-kernel、torch、flashinfer。torch 需要 SGLANG_DSA_FUSE_TOPK=false。默认与效果：默认 sgl-kernel。交互与改写：和目标模型的 dsa_topk_backend 分开。易错：只改了目标的，草稿仍是另一个。 |
| `speculative_draft_kv_cache_dtype` | --speculative-draft-kv-cache-dtype（S2 可改写） | `Optional[str]` | `None` | 作用：只改草稿 KV 的 dtype。草稿和目标共享槽位下标，槽数按目标 token 算，所以小草稿的池仍可能很大：5 层 DFLASH 在 bf16 下每 token 10240 字节。设成 fp8_e4m3 能把草稿池减半，省下的显存要靠提高 mem_fraction_static 才变成 KV 容量。默认跟 kv_cache_dtype。默认与效果：默认跟目标。交互与改写：减半不会自动扩大 KV，除非你提高静态比例。易错：改了 dtype 却没调 mem_fraction_static，容量看起来没变。 |
| `speculative_draft_window_size` | --speculative-draft-window-size | `Optional[int]` | `None` | 作用：草稿只看最近 N 个 key。Llama EAGLE-3、DFLASH，以及 Triton/FlashInfer 上的内置 EAGLE/MTP 解码会遵守。其他 EAGLE-3 后端会静默忽略。校验侧不受影响。DFLASH 在分页后端上可能在左边多留一页对齐。草稿自己有滑动窗口时，这个旗标会被忽略并警告。默认是全量注意力。默认与效果：默认空，全量。交互与改写：和 sink size 一起构成 StreamingLLM 式窗口。易错：设了窗口但后端静默忽略，你以为省了 KV。 |
| `speculative_draft_sink_size` | --speculative-draft-sink-size | `Optional[int]` | `None` | 作用：草稿始终额外关注的前缀 sink token 数，加在最近窗口之外。只有内置 EAGLE/MTP 的 Triton 和 FlashInfer 草稿解码会用。Llama EAGLE-3 和 DFLASH 忽略它。0 或未设置表示只有最近窗口。必须同时给 draft_window_size。默认与效果：默认空。交互与改写：没有窗口时设置 sink 会失败。易错：只设 sink 不设窗口。 |
| `speculative_moe_runner_backend` | --speculative-moe-runner-backend（S2 可改写） | `Optional[str]` | `None` | 作用：草稿 MoE 的 GEMM 后端。默认跟目标的 moe_runner_backend，但 flashinfer_megamoe 会把草稿改成 flashinfer_trtllm，因为草稿层通常没量化，megamoe 的全局专家号分发不支持。默认与效果：默认空，按上面的规则。交互与改写：和目标后端可以不同。易错：草稿未量化却强制 megamoe。 |
| `speculative_moe_a2a_backend` | --speculative-moe-a2a-backend（S2 可改写） | `Optional[str]` | `None` | 作用：草稿的 MoE all-to-all。默认跟目标，但 flashinfer_megamoe 会把草稿改成 none。DSPARK 加 attention DP 时，如果显式设置，必须和目标后端相同。默认与效果：默认空。交互与改写：DSPARK 会拒绝目标和草稿 a2a 不一致。易错：目标是 mori，草稿写成 deepep。 |
| `speculative_enable_w4a4_mxfp4_megamoe` | --speculative-enable-w4a4-mxfp4-megamoe | `Optional[bool]` | `None` | 作用：草稿的 MXFP4 MegaMoE 是否用 W4A4。不填则跟目标的同名开关。--no- 可以让草稿留在 fp8xfp4，目标仍是 W4A4。MMA 类型和目标不同时，可能在 KV 池算完之后再分配一块 MegaMoE 对称缓冲，图捕获 OOM 时要降低 mem_fraction_static。默认与效果：默认空，跟目标。交互与改写：额外缓冲发生在池大小已经定了之后。易错：捕获图时 OOM，却只去加大 KV 比例。 |
| `speculative_draft_model_quantization` | --speculative-draft-model-quantization | `Optional[str]` | `None` | 作用：草稿的量化方法。默认与效果：默认空，继承目标。交互与改写：内部标记会记住用户有没有显式写过。易错：草稿是 FP8，方法却继承成目标的 FP4。 |
| `_speculative_draft_quantization_explicitly_set` | ---speculative-draft-quantization-explicitly-set（无常规 CLI） | `Optional[bool]` | `None` | 作用：内部标记，区分草稿量化是用户写的还是从目标继承的。没有 CLI。默认与效果：由解析记住。交互与改写：序列化 ServerArgs 时要保住它，否则下一轮会再次继承。易错：测试里丢掉这个标记，继承逻辑重跑。 |
| `enable_multi_layer_eagle` | --enable-multi-layer-eagle（S2 可改写） | `bool` | `False` | 作用：多层 EAGLE，草稿不止一层。默认与效果：默认关闭。交互与改写：V4 只允许 topk 为 1 的 EAGLE，V4.1 不允许 EAGLE。易错：V4.1 打开它。 |
| `speculative_adaptive` | --speculative-adaptive | `bool` | `False` | 作用：按接受率动态调整草稿步数。默认与效果：默认关闭。交互与改写：DSPARK 会把步数强制收成 1，自适应没有空间。易错：和 DSPARK 一起开，步数仍是 1。 |
| `speculative_adaptive_config` | --speculative-adaptive-config | `Optional[str]` | `None` | 作用：自适应投机的 JSON 配置路径。默认与效果：默认空。交互与改写：只在 speculative_adaptive 打开时读。易错：文件里的步数和算法允许的步数冲突。 |
| `spec_trace_dir` | --spec-trace-dir | `Optional[str]` | `None` | 作用：解耦投机的追踪文件目录。默认与效果：默认空，不写追踪。交互与改写：只在 decoupled spec 角色不是 null 时有内容。易错：目录不可写，追踪失败被当成投机失败。 |
| `speculative_ngram_min_bfs_breadth` | --speculative-ngram-min-bfs-breadth | `int` | `1` | 作用：n-gram 投机在树上做 BFS 时的最小分支宽度。默认与效果：默认 1。交互与改写：不能大于最大宽度。易错：最小大于最大。 |
| `speculative_ngram_max_bfs_breadth` | --speculative-ngram-max-bfs-breadth | `int` | `10` | 作用：n-gram BFS 的最大分支宽度。默认与效果：默认 10。交互与改写：越大，草稿候选越多，校验越贵。易错：设得很大，校验批爆显存。 |
| `speculative_ngram_match_type` | --speculative-ngram-match-type | `Literal['BFS', 'PROB']` | `'BFS'` | 作用：n-gram 缓存树的匹配方式，默认 BFS。默认与效果：默认 BFS。交互与改写：和前缀树的 radix 匹配不是同一个结构。易错：改成不认识的字符串。 |
| `speculative_ngram_max_trie_depth` | --speculative-ngram-max-trie-depth | `int` | `18` | 作用：n-gram trie 的最大深度。默认与效果：默认 18。交互与改写：深度越大，能匹配的片段越长，内存越大。易错：深度超过常见重复片段，只增加内存。 |
| `speculative_ngram_capacity` | --speculative-ngram-capacity | `int` | `10 * 1000 * 1000` | 作用：n-gram 缓存能放多少条目。默认与效果：默认一千万。交互与改写：和外部语料的 token 上限是两套预算。易错：容量小于外部语料，预加载时被挤掉。 |
| `speculative_ngram_external_corpus_path` | --speculative-ngram-external-corpus-path | `Optional[str]` | `None` | 作用：启动时灌进 SAM 的外部 JSONL 语料。运行中还可以 POST /add_external_corpus。默认与效果：默认空，不灌。交互与改写：语料 token 数不能超过 external_corpus_max_tokens。易错：JSONL 不是一行一个样例。 |
| `speculative_ngram_external_sam_budget` | --speculative-ngram-external-sam-budget | `int` | `0` | 作用：为外部 SAM 子树预留多少草稿节点。默认与效果：默认 0，不预留。交互与改写：为 0 时外部语料没有节点额度。易错：灌了语料但预算是 0。 |
| `speculative_ngram_external_corpus_max_tokens` | --speculative-ngram-external-corpus-max-tokens | `int` | `10000000` | 作用：外部 n-gram 语料分词后超过这么多个 token 就拒绝启动。按 CPU 内存来调。默认与效果：默认一千万。交互与改写：这是 CPU 预算，不是 GPU KV。易错：语料很大仍留着默认，启动直接失败。 |

### Prefill/Decode 分离

命名空间 `disagg`，类 `Disagg`，`python/sglang/srt/arg_groups/fields/disagg.py`，30 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `decoupled_spec_bind_endpoint` | --decoupled-spec-bind-endpoint | `Optional[str]` | `None` | 作用：解耦投机时本引擎入站 ZMQ 地址。校验端用它拉结果，草稿端用它拉控制消息。默认与效果：默认空。交互与改写：角色不是 null 时，这个地址、对端列表和 rank 三者缺一就在 PortArgs.init_new 里抛错。易错：只写了角色，没写绑定地址。 |
| `decoupled_spec_connect_endpoints` | --decoupled-spec-connect-endpoints | `Optional[List[str]]` | `None` | 作用：要连上去的对端入站地址，按对端 rank 排序。默认与效果：默认空。交互与改写：必须和 bind 地址、rank 一起给。易错：顺序和 rank 不一致，草稿结果送到错误的校验端。 |
| `decoupled_spec_rank` | --decoupled-spec-rank | `Optional[int]` | `None` | 作用：本引擎在自己角色空间里的序号，校验端一套、草稿端一套。默认与效果：默认空。交互与改写：解耦投机启用时必填。易错：校验端和草稿端都写成 0，连接表对不上。 |
| `decoupled_spec_role` | --decoupled-spec-role | `Literal['null', 'verifier', 'drafter']` | `'null'` | 作用：解耦投机的角色。null 表示关闭；verifier 只跑目标模型校验；drafter 只跑草稿。默认与效果：默认 null，目标和草稿仍在同一个 scheduler 里。交互与改写：不是 null 时必须同时给 bind、connect 和 rank。易错：两个进程都写成 verifier，没有人产草稿。 |
| `disaggregation_mode` | --disaggregation-mode | `Literal['null', 'prefill', 'decode']` | `'null'` | 作用：PD 分离角色。prefill 只做提示，decode 只做解码。不填就是 null，预填充和解码在同一进程。默认与效果：默认 null。交互与改写：S9 会换成对应的分离事件循环。V4.1 加 DSPARK 时还要求传输后端是 mooncake 或 nixl。易错：两台机器都写成 prefill。 |
| `disaggregation_decode_allocation_policy` | --disaggregation-decode-allocation-policy | `Literal['early', 'prefill_complete']` | `'early'` | 作用：分离服务里何时为请求预留 decode 侧 KV。early 是尽早留；prefill_complete 是等 prefill 结束再在 decode 上接纳。两边必须设成同一个策略。默认与效果：默认 early。交互与改写：策略不一致时 KV 生命周期对不齐。易错：只在 decode 上改成 prefill_complete。 |
| `disaggregation_transfer_backend` | --disaggregation-transfer-backend | `str` | `'mooncake'` | 作用：PD 之间传 KV 的后端。默认与效果：默认 mooncake。交互与改写：decode 侧 radix、host 接收、校验和各自还有允许的后端列表。fake 后端和 decode radix 不兼容。易错：机器没有 Mooncake 却留着默认值，启动卡在传输初始化。 |
| `disaggregation_enable_kv_checksum` | --disaggregation-enable-kv-checksum | `bool` | `False` | 作用：prefill 对每个请求的 KV 页算 Adler-32，decode 核对。不一致就中止请求，CI 里会直接抛错。decode 有 radix 时摘要只覆盖 prefill 传来的页。布局不同（TP、PP、页大小、dtype）的 prefill 只警告一次、不校验。两边都要打开。默认与效果：默认关闭。交互与改写：打开会增加传输 CPU。易错：只在一边打开，另一边没有摘要可比。 |
| `disaggregation_bootstrap_port` | --disaggregation-bootstrap-port | `int` | `8998` | 作用：prefill 上 bootstrap 服务的端口，decode 靠它来配对。默认与效果：默认 8998。交互与改写：请求里的 bootstrap_port 要指到这里。易错：和 encoder 的 8997 写反。 |
| `disaggregation_ib_device` | --disaggregation-ib-device | `Optional[str]` | `None` | 作用：分离传输用的 InfiniBand 设备。可以是单块、逗号分隔的共享列表、按 GPU 的 JSON，或指向该 JSON 的文件。Mooncake 和 NIXL UCX 支持。键是进程可见的 GPU 号（已经过 CUDA_VISIBLE_DEVICES）。NIXL 不要带端口后缀。若设置了非空 UCX_NET_DEVICES，它优先于本选项。默认与效果：默认空，由后端自己选设备。交互与改写：不改变 EP 宽度。易错：JSON 的 GPU 号用的是机器物理号，而进程只看见两张卡。 |
| `disaggregation_decode_enable_radix_cache` | --disaggregation-decode-enable-radix-cache | `bool` | `False` | 作用：decode 进程上也建前缀树，避免重复传输已经有的 KV。不能和 HiSparse、fake 传输后端，以及 EAGLE、EAGLE3、DSPARK 以外的投机算法一起用。默认与效果：默认关闭。交互与改写：打开后命中发生在 decode 本地，不是 prefill 的树。易错：用 NEXTN 或 NGRAM 同时打开它，解析会拒绝。 |
| `disaggregation_decode_enable_offload_kvcache` | --disaggregation-decode-enable-offload-kvcache | `bool` | `False` | 作用：decode 侧异步把 KV 换出。默认与效果：默认关闭。交互与改写：要有 host 侧空间，和 retraction backup 的存储方式相关。易错：没有 host 池却打开，换出失败变成请求中止。 |
| `disaggregation_decode_retraction_backup` | --disaggregation-decode-retraction-backup | `Optional[str]` | `None` | 作用：decode 收缩时 KV 备份到哪。cpu_tensor 是每请求一块 CPU 张量；host_pool 用预留的 HiCache 池，耗尽不回退；none 不备份，被收缩的请求以 503 让客户端重试。省略则按 decode 的 KV 池推断。默认与效果：默认空，走推断。交互与改写：host_pool 依赖 HiCache 容量。易错：以为省略等于 none，实际上可能已经选了 cpu_tensor。 |
| `num_reserved_decode_tokens` | --num-reserved-decode-tokens | `int` | `512` | 作用：新请求进入 running 批时，为 decode 预留多少 token 的显存。默认与效果：默认 512。交互与改写：预留过大，能同时跑的请求变少。易错：长输出模型仍用 512，跑到后面频繁 retraction。 |
| `disaggregation_decode_extra_slots` | --disaggregation-decode-extra-slots | `Optional[int]` | `None` | 作用：PD 模式下为正在传输的请求额外预留的 req_to_token 槽。不填时，小批是每 worker running 批的 2 倍，否则 0。默认与效果：默认空，按上面的规则。交互与改写：槽不够时传输中的请求进不了表。易错：手写成 0，高并发传输被丢掉。 |
| `disaggregation_decode_polling_interval` | --disaggregation-decode-polling-interval | `int` | `1` | 作用：decode 进程轮询请求的间隔。大于 1 可以降低轮询开销。默认与效果：默认 1，每次都看。交互与改写：调大则接纳传输完成的请求更慢。易错：设成几十，decode 侧看起来像卡住。 |
| `enable_pd_role_switch` | --enable-pd-role-switch | `bool` | `False` | 作用：允许运行中通过 /pd_role_switch 在 prefill 和 decode 之间换角色。默认与效果：默认关闭。交互与改写：只在 PD 模式有意义。易错：合部服务打开它，接口存在但没有角色可换。 |
| `optimistic_prefill_attempts` | --optimistic-prefill-attempts | `int` | `0` | 作用：跳过 bootstrap 等待、乐观做几次 prefill 前向的次数。默认与效果：默认 0，不乐观，先等配对。交互与改写：大于 0 时配对失败会浪费已经算过的提示。易错：网络不稳时设得很大，白算。 |
| `encoder_only` | --encoder-only | `bool` | `False` | 作用：多模态模型只启动编码器服务，不跑语言模型。默认与效果：默认关闭。交互与改写：run_server 看到它会离开主 LLM 循环，去 encoder 的 HTTP 或 gRPC。易错：语言进程没有 encoder_urls，图像请求没人算。 |
| `language_only` | --language-only | `bool` | `False` | 作用：VLM 只加载语言模型权重，编码器在别的进程。默认与效果：默认关闭，权重一起加载。交互与改写：这是编码器/解码器分离的一部分，和 language_model_only 不是同一个开关。易错：两个都开，行为以「完全不建塔」还是「分离」为准要看解析顺序。 |
| `language_model_only` | --language-model-only | `bool` | `False` | 作用：完全不建多模态塔，权重也不加载，省下的显存给 KV。多模态请求会被拒绝。它是独立模式，不是 PD 分离的一环。默认与效果：默认关闭，V4.1 会建 ViT。交互与改写：DeepseekV4ForCausalLM 构造函数直接看它。易错：纯文本服务忘了开，144GB 上多占 32 层 ViT。 |
| `encoder_transfer_backend` | --encoder-transfer-backend | `str` | `'auto'` | 作用：编码器分离时特征怎么传。auto 会按模型和 TP 选。默认与效果：默认 auto。交互与改写：Kimi K3 且 TP>1 时解析会选 zmq_to_tokenizer，见 resolve_encoder_transfer_backend。易错：手写一个后端，和模型的 TP 不匹配。 |
| `encoder_urls` | --encoder-urls | `List[str]` | `msgspec.field(default_factory=list)` | 作用：编码器服务的 URL 列表，语言进程把图像工作发给它们。默认与效果：默认空列表。交互与改写：也可以靠 encoder 启动时向 bootstrap 注册，动态拿到列表。易错：列表是空的又没注册，多模态请求没有去处。 |
| `encoder_bootstrap_port` | --encoder-bootstrap-port | `int` | `8997` | 作用：跑在 language-only tokenizer 进程里的 EncoderBootstrapServer 端口。编码器来注册，语言侧来拉当前 URL。默认与效果：默认 8997。交互与改写：不要和 PD 的 8998 撞车。易错：防火墙只放了 8998。 |
| `encoder_register_urls` | --encoder-register-urls | `List[str]` | `msgspec.field(default_factory=list)` | 作用：本编码器启动时要去注册的 bootstrap URL，可以多个。例如两个 prefill 的 8997。和 --encoder-only 一起用。默认与效果：默认空，不主动注册。交互与改写：语言侧要开着 bootstrap 才能被发现。易错：URL 写成了模型 HTTP 端口而不是 8997。 |
| `enable_adaptive_dispatch_to_encoder` | --enable-adaptive-dispatch-to-encoder | `bool` | `False` | 作用：language_only 的 EPD 模式下，多图请求发给编码器，单图请求本地算。默认与效果：默认关闭，按固定路径分发。交互与改写：只在 language_only 分离时有区别。易错：合部服务打开它，没有编码器可派。 |
| `enable_pdmux` | --enable-pdmux | `bool` | `False` | 作用：打开 PD 复用，prefill 和 decode 跑在 greenctx 流上。默认与效果：默认关闭。交互与改写：需要 pdmux 配置文件描述切分。易错：只开开关不给配置路径。 |
| `pdmux_config_path` | --pdmux-config-path | `Optional[str]` | `None` | 作用：PD 复用的配置文件路径。默认与效果：默认空。交互与改写：enable_pdmux 为真时才读。易错：路径不存在，启动失败点在读文件而不是并行断言。 |
| `sm_group_num` | --sm-group-num | `int` | `8` | 作用：SM 分区组的数量，给 PD 复用这类按 SM 切流的路径用。默认与效果：默认 8。交互与改写：必须和 GPU 的 SM 数相配，过大没有额外组可切。易错：在小卡上照抄 8。 |
| `disaggregation_decode_host_receive_threshold` | --disaggregation-decode-host-receive-threshold | `float` | `1.0` | 作用：设备上 token 占用到这个比例时，进来的 KV 改收进 decode 收缩用的 host 池，不把可驱逐的缓存页算进去。范围 0 到 1；1 表示从不走 host 接收；0 表示总是走。容量用 hicache-size 或 hicache-ratio。需要稠密 NHD MHA 缓冲，以及支持 host 目的地的传输后端。目前没有内置后端支持。默认与效果：默认 1.0，等于关闭 host 接收。交互与改写：help 写明尚无内置后端，打开也收不到。易错：设成 0 以为能省显存，实际后端不支持。 |

### LoRA

命名空间 `lora`，类 `Lora`，`python/sglang/srt/arg_groups/fields/lora.py`，14 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `enable_lora` | --enable-lora | `Optional[bool]` | `None` | 作用：打开 LoRA。若提供了 lora 路径，S2 会把它设为真以保持兼容。 默认与效果：沿用字段声明的默认值；若解析阶段改写，以启动日志为准。 交互与改写：检查 max_loras_per_batch。部分 MoE 融合会被关掉。 易错：路径给了但 batch 上限是 0。 |
| `enable_lora_overlap_loading` | --enable-lora-overlap-loading | `Optional[bool]` | `None` | 作用：异步把 LoRA 权重从主机拷到 GPU，和计算重叠。适配器切换很频繁、加载变成瓶颈时才该开。默认与效果：默认关闭，加载是同步的。交互与改写：要已经 enable_lora。易错：适配器很少切换却打开，多一次异步状态机。 |
| `max_lora_rank` | --max-lora-rank | `Optional[int]` | `None` | 作用：适配器秩的上限。不填则从 lora_paths 里的适配器推断。默认与效果：默认空，从文件推断。交互与改写：实际秩超过上限会在加载时失败。易错：手写 8，适配器其实是 64。 |
| `lora_target_modules` | --lora-target-modules | `Optional[Union[set[str], List[str]]]` | `None` | 作用：要挂 LoRA 的模块集合。不填则从适配器推断。写成 all 则所有支持的模块都挂。默认与效果：默认空，走推断。交互与改写：和适配器里真正有的模块取交集。易错：写了 all，不支持的模块被静默跳过或报错，取决于严格加载。 |
| `lora_paths` | --lora-paths | `Optional[Union[dict[str, str], List[dict[str, str]], List[str], List[LoRARef]]]` | `None` | 作用：要加载的适配器列表。每项可以是路径、名字=路径，或 JSON {lora_name, lora_path, pinned}。默认与效果：默认空，不加载适配器。交互与改写：一旦给出，enable_lora 会被自动设为真。易错：JSON 缺字段，argparse 或加载失败。 |
| `max_loaded_loras` | --max-loaded-loras | `Optional[int]` | `None` | 作用：同一时刻留在 CPU 内存里的适配器个数上限。必须大于等于 max_loras_per_batch。默认与效果：默认不额外限制，能装多少装多少。交互与改写：超过后按 lora_eviction_policy 驱逐。易错：设得比每批上限还小，解析会拒绝。 |
| `max_loras_per_batch` | --max-loras-per-batch | `int` | `8` | 作用：一个 running 批里最多几个适配器，包含只跑基座的请求。默认与效果：默认必须是正数，检查里会断言大于 0。交互与改写：triton_v2 等后端不支持 attention DP。易错：设成 0。 |
| `lora_eviction_policy` | --lora-eviction-policy | `str` | `'lru'` | 作用：CPU 侧适配器池满了怎么赶。lru 是最近最少使用，默认，缓存更有效；fifo 是先进先出。默认与效果：默认 lru。交互与改写：和前缀树的 radix 驱逐不是同一个策略。易错：把 radix 的 lfu 写到这里。 |
| `lora_backend` | --lora-backend | `str` | `'csgmv'` | 作用：多 LoRA 的 kernel 后端。默认与效果：默认按实现选择。交互与改写：csgmv 才会读 max_lora_chunk_size。易错：块大小设得很大但后端不是 csgmv。 |
| `max_lora_chunk_size` | --max-lora-chunk-size | `Optional[int]` | `16` | 作用：ChunkedSGMV LoRA 后端的块大小。只在 lora_backend=csgmv 时使用。更大可能更快。默认与效果：默认按该后端。交互与改写：其他后端忽略它。易错：在 triton LoRA 上调它，没有变化。 |
| `experts_shared_outer_loras` | --experts-shared-outer-loras | `Optional[bool]` | `None` | 作用：MoE 上强制共享外层 LoRA。打开后 w1/w3 的 lora_A 和 w2 的 lora_B 在专家之间共享，expert_dim 为 1。--no- 可强制关掉。默认按适配器自动检测。默认与效果：默认自动检测。交互与改写：和虚拟专家是两套 MoE LoRA 方式。易错：适配器不是共享外层，却强制打开，形状对不上。 |
| `lora_use_virtual_experts` | --lora-use-virtual-experts | `bool` | `False` | 作用：MoE 使用虚拟专家计算来挂 LoRA。默认与效果：默认关闭。交互与改写：会改变专家维的计算图。易错：和共享外层 LoRA 同时开，两套假设冲突。 |
| `lora_strict_loading` | --lora-strict-loading | `bool` | `False` | 作用：严格加载适配器。权重里缺键或多键就报错，而不是跳过。默认与效果：默认不严格。交互与改写：排错适配器是否匹配基座时应该打开。易错：生产打开后，一个可选模块缺失就整次加载失败。 |
| `lora_drain_wait_threshold` | --lora-drain-wait-threshold | `float` | `0.0` | 作用：某个适配器请求等待超过这么多秒，调度器会有选择地排空一个正在跑的适配器来腾位置，用来压高负载下的长尾。默认与效果：默认按字段。交互与改写：排空会打断那个适配器上的请求。易错：阈值太小，适配器被频繁换入换出。 |

### 多模态

命名空间 `mm`，类 `Mm`，`python/sglang/srt/arg_groups/fields/mm.py`，18 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `enable_multimodal` | --enable-multimodal | `Optional[bool]` | `None` | 作用：为当前模型打开多模态功能。如果模型本身不是多模态，打开也没有任何效果。默认与效果：默认空，按模型是否有视觉塔。交互与改写：V4.1 有 vision_config，不打开 language_model_only 就会建 ViT。易错：非多模态模型打开它，以为服务会开始收图片。 |
| `mm_attention_backend` | --mm-attention-backend | `Optional[str]` | `None` | 作用：多模态塔自己的注意力后端，和语言模型的 attention_backend 分开。默认与效果：默认空，跟多模态实现的默认。交互与改写：文本模型没有这张塔。易错：把 dsv4 写到这里，ViT 不走 dsv4。 |
| `enable_broadcast_mm_inputs_process` | --enable-broadcast-mm-inputs-process | `bool` | `False` | 作用：在调度器里广播多模态输入的处理结果，让各 rank 拿到同一份特征。默认与效果：默认关闭。交互与改写：TP 下如果只有一个 rank 做了预处理，必须有广播或别的传输。易错：TP>1 却不广播，其他 rank 没有图像特征。 |
| `enable_prefix_mm_cache` | --enable-prefix-mm-cache | `bool` | `False` | 作用：多模态前缀缓存。目前只支持纯多模态、没有混排文本的情况。默认与效果：默认关闭。交互与改写：和普通 radix 的 token 前缀不是同一条缓存。易错：图文混排请求指望它命中。 |
| `mm_enable_dp_encoder` | --mm-enable-dp-encoder（S2 可改写） | `Optional[bool]` | `None` | 作用：视觉编码器按数据并行复制，并行度会被设成 tp。默认与效果：默认空。交互与改写：tp 为 1 时 S2 会警告没有可分的工作。会复制 ViT 权重。易错：小图场景打开，复制权重大于算力收益。 |
| `mm_process_config` | --mm-process-config | `Optional[Dict[str, Any]]` | `None` | 作用：多模态预处理 JSON，键可以是 image、video、audio。默认与效果：默认空，用处理器默认。交互与改写：和请求里的 mm_processor_kwargs 是两层，服务端默认和请求覆盖。易错：JSON 键名写错，被静默忽略。 |
| `mm_processor_worker_num` | --mm-processor-worker-num | `int` | `0` | 作用：调用多模态处理器的线程数。0 表示用模型自己的默认。只有支持隔离 worker 的处理器才能大于 1。默认与效果：默认 0。交互与改写：大于 1 时处理器必须是线程安全的。易错：不支持的模型设成 8，启动或第一次请求失败。 |
| `mm_io_worker_num` | --mm-io-worker-num | `int` | `0` | 作用：加载和解码图像、视频、音频的线程数。0 用模型默认。参数为 0 时环境变量 SGLANG_IO_WORKERS 仍可作为覆盖。默认与效果：默认 0。交互与改写：和处理器线程是两套，一个管 IO，一个管处理器。易错：IO 线程远多于磁盘或网络带宽，只增加切换。 |
| `allowed_media_domains` | --allowed-media-domains | `List[str]` | `msgspec.field(default_factory=list)` | 作用：客户端给的 http(s) 图片、视频、音频 URL 只能是这些主机名。重定向目标也要在名单里。不设置则任意域名都允许。默认与效果：默认不限制。交互与改写：这是安全边界，和 media 大小上限一起用。易错：名单写了域名但漏了 CDN 重定向后的主机。 |
| `media_url_max_file_size_mb` | --media-url-max-file-size-mb | `int` | `64` | 作用：单个远程媒体下载的 MiB 上限，边下边检查。0 表示不限制。默认与效果：默认按字段。交互与改写：超过就中止这次下载，请求失败。易错：设成 0 在公网服务上等于不设防。 |
| `mm_preprocess_cache_size_mb` | --mm-preprocess-cache-size-mb（S2 可改写） | `Optional[int]` | `None` | 作用：按内容寻址的多模态预处理结果能用多少 CPU 内存。不设置时用模型默认，Kimi-K3 是 256 MiB。0 关闭缓存。预算会分给各个 tokenizer worker，不是每个 worker 都拿满。默认与效果：默认按模型。交互与改写：和 GPU 上的 ViT 全局缓存不是同一块。易错：多 worker 时按 256 估算总内存，实际是总额再拆分。 |
| `trust_mm_content_hashes` | --trust-mm-content-hashes | `bool` | `False` | 作用：信任调用方给的多模态 SHA-256。热的元数据缓存命中时可以不再读媒体字节。只有调用方保证哈希对应不可变字节时才能开。默认与效果：默认不信任，服务自己读字节再哈希。交互与改写：和请求字段 mm_content_hashes 一起用。易错：公网打开它，调用方可以伪造哈希去撞缓存。 |
| `limit_mm_data_per_request` | --limit-mm-data-per-request | `Optional[Union[str, Dict[str, int]]]` | `None` | 作用：每个请求最多几张图、几段视频、几段音频，例如 {"image": 1, "video": 1, "audio": 1}。默认与效果：默认不限制或按模型。交互与改写：超过则请求在 S7/S8 被拒绝。易错：JSON 写成列表而不是按模态的字典。 |
| `enable_mm_global_cache` | --enable-mm-global-cache | `bool` | `False` | 作用：全局多模态嵌入缓存，相同图像不再重复跑 ViT。默认与效果：默认关闭。交互与改写：存储后端由 mm_global_cache_backend 指定。易错：打开了但后端没配，缓存建不起来。 |
| `image_processor_backend` | --image-processor-backend | `Literal['auto', 'torchvision', 'pil']` | `'auto'` | 作用：图像处理器后端。auto 让 Transformers 选可用的最好后端。默认与效果：默认 auto。交互与改写：旧的 disable_fast_image_processor 已改为这里写 pil。易错：继续写废弃旗标。 |
| `mm_global_cache_backend` | --mm-global-cache-backend | `str` | `'mooncake'` | 作用：全局图像嵌入缓存的存储后端。只有 enable_mm_global_cache 打开才用。默认与效果：默认按实现。交互与改写：和 HiCache 的存储不是同一份数据。易错：把 KV 的 mooncake 配置套到嵌入缓存上。 |
| `disable_fast_image_processor` | --disable-fast-image-processor | `bool` | `False` | 作用：已废弃。改用 image_processor_backend=pil。默认与效果：默认不禁用。交互与改写：新配置不要再写它。易错：文档旧脚本仍带着它。 |
| `mm_feature_transport` | --mm-feature-transport（S2 可改写） | `Optional[Literal['cpu', 'cuda_ipc', 'cuda_vmm']]` | `None` | 作用：多模态特征怎么送进 GPU：走 CPU 内存、有界 CUDA IPC 池，或有界 CUDA VMM 池。不设置时用 cpu；已经验证过的多机 GB200/GB300 MNNVL 模型在有 IMEX 通道时用 cuda_vmm。默认与效果：默认按上面的规则。交互与改写：V4.1 的 CudaVmmFeatureTransport 只在选择 VMM 时启用。易错：没有 NVLink 域却写 cuda_vmm。 |

### 设备与进程启动

命名空间 `device`，类 `Device`，`python/sglang/srt/arg_groups/fields/device.py`，13 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `device` | --device | `Optional[str]` | `None` | 作用：选择计算设备，取值 cuda、xpu、hpu、npu、cpu、musa。不填则按当前机器自动探测。默认与效果：声明默认是空，因此启动时探测，不会假定一定是 NVIDIA。交互与改写：HIP/海光上探测结果的设备字符串经常仍是 cuda，ROCm 分支看平台的 is_hip，不看你有没有写 cuda。易错：在 DCU 上把 device 写成 cpu，权重和 KV 会离开加速器。 |
| `base_gpu_id` | --base-gpu-id | `int` | `0` | 作用：从哪块可见 GPU 开始往上分配。同一台机器跑多个进程时，用它错开卡。默认与效果：默认 0，从第一块可见卡开始。交互与改写：和 gpu_id_step、CUDA_VISIBLE_DEVICES 一起决定子进程看见的卡号。易错：VISIBLE_DEVICES 已经重排过，再设一个很大的 base，会指到不存在的卡。 |
| `gpu_id_step` | --gpu-id-step | `int` | `1` | 作用：相邻进程占用的 GPU 编号间隔。设成 2 就是 0、2、4。默认与效果：默认 1，连续占用。交互与改写：只影响编号，不改变 tp 需要的卡数。易错：间隔大于机器上的卡数，后面的 rank 没有设备。 |
| `gpu_id` | --gpu-id（无常规 CLI） | `Derived` | `（发布时计算）` | 作用：这个进程实际绑的设备号。用户不填，父进程在拉起子进程时写进发布包：可能先缩小可见设备，Ray 则从自己的池里分。None 表示这个角色根本不占设备。默认与效果：派生量，没有 CLI。交互与改写：tokenizer 和多数非计算进程保持 None。易错：在代码里假设每个进程都有 gpu_id，读 tokenizer 进程会得到空。 |
| `random_seed` | --random-seed | `Optional[int]` | `None` | 作用：全局随机种子，影响采样和部分初始化。默认与效果：默认空，不固定种子，输出不可复现。交互与改写：要和 enable_deterministic_inference 一起才谈得上跨批稳定；只设种子不够。易错：设了种子却开着不同的 CUDA 图或注意力后端，结果仍然对不齐。 |
| `mlx_enable_sampling` | --mlx-enable-sampling | `bool` | `False` | 作用：只对 MLX 后端。打开后 decode 按 temperature、top-k、top-p、min-p 采样，而不是贪心 argmax。采样在惰性 MLX 图里做，所以能和 overlap 调度一起用；prefill 的首 token 也会采样。贪心请求仍是精确 argmax。同时打开语法 mask、自定义 logit 处理器（会打断逐步链式 decode，自定义处理器只在纯 decode 步跑）、logit_bias、输出 logprob，以及确定性模式下的按请求种子。penalty 不算。默认与效果：默认关闭，MLX 走贪心。交互与改写：非 MLX 设备写了它没有采样效果。易错：以为打开后 prompt 的 logprob 也会算，help 写明不算输入段 logprob。 |
| `watchdog_timeout` | --watchdog-timeout | `float` | `300` | 作用：看门狗秒数。一次前向超过这个时间，进程主动崩溃，避免永久挂死。默认与效果：默认 300 秒。交互与改写：和 soft_watchdog_timeout 不同，这个是硬退出。易错：长上下文或第一次编译超过 300 秒，服务会被自己杀掉。 |
| `soft_watchdog_timeout` | --soft-watchdog-timeout | `Optional[float]` | `None` | 作用：软看门狗。前向超时只打调试信息，不退出。默认与效果：默认空，不启用软看门狗。交互与改写：应小于 watchdog_timeout，否则还没转储就被硬看门狗杀掉。易错：两个都设成同一个很大的数，等于没有提前报警。 |
| `sleep_on_idle` | --sleep-on-idle | `bool` | `False` | 作用：空闲时降低 CPU 占用。默认与效果：默认关闭，空闲也按正常节奏轮询。交互与改写：可能拉高空闲后的第一条请求延迟。易错：在线服务为了省 CPU 打开它，TTFT 抖动变大。 |
| `use_ray` | --use-ray | `bool` | `False` | 作用：用 Ray actor 管理 scheduler 进程，而不是本机 multiprocessing。默认与效果：默认关闭。交互与改写：打开后走 ray/http_server.py，需要安装 sglang[ray]。易错：没装 Ray 却打开，导入阶段失败。 |
| `custom_sigquit_handler` | --custom-sigquit-handler | `Optional[Callable]` | `None` | 作用：自定义 SIGQUIT 处理。声明处没有 help。默认与效果：默认空，用进程默认的退出处理。交互与改写：给调试钩子用，不改变推理数值。易错：处理器自己再发信号，会造成递归退出。 |
| `numa_node` | --numa-node | `Optional[List[int]]` | `None` | 作用：给子进程绑 NUMA 节点。列表第 i 项对应第 i 个子进程。不填则在 NUMA 机器上自动探测。默认与效果：默认空，走自动探测。交互与改写：绑错节点会让 CPU 和 GPU 跨 NUMA，all-reduce 前的准备变慢。易错：列表长度和子进程数不一致。 |
| `gc_threshold` | --gc-threshold | `Optional[List[int]]` | `None` | 作用：设置 Python GC 阈值，也就是回收频率。接受 1 到 3 个整数。默认与效果：默认空，用解释器默认阈值。交互与改写：和 enable_cudagraph_gc 是两件事，一个管平时 GC，一个管捕获图时是否冻结 GC。易错：把阈值设得极低，服务循环里频繁 GC，decode 间隔抖动。 |

### 可观测性

命名空间 `observability`，类 `Observability`，`python/sglang/srt/arg_groups/fields/observability.py`，45 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `log_level` | --log-level | `str` | `'info'` | 作用：所有日志器的级别，例如 debug、info、warning。默认与效果：默认 info。交互与改写：不改变数值，只改变打印多少。HTTP 访问日志另有 log_level_http。易错：把 debug 开在生产，磁盘被日志打满。 |
| `log_level_http` | --log-level-http | `Optional[str]` | `None` | 作用：HTTP 访问日志的级别，和全局 log_level 分开。默认与效果：默认跟全局或 uvicorn 默认。交互与改写：可以用前缀排除 /health 这类路径。易错：全局调成 warning 仍看到大量访问日志，因为这条没改。 |
| `log_requests` | --log-requests | `bool` | `False` | 作用：记录每个请求的元数据、输入和输出。详细程度由 log_requests_level 决定。默认与效果：默认关闭。交互与改写：级别 3 会记下全部输入输出，可能含用户内容。易错：生产打开级别 3。 |
| `log_requests_level` | --log-requests-level | `int` | `2` | 作用：0 只记元数据不含采样参数；1 加采样参数；2 再加部分输入输出；3 记下全部输入输出。默认与效果：默认 2。交互与改写：只有 log_requests 打开才有输出。易错：没开总开关却调级别，什么都不会多打。 |
| `log_requests_format` | --log-requests-format | `str` | `'text'` | 作用：请求日志是给人读的 text，还是结构化 json。默认与效果：默认 text。交互与改写：json 方便采集，text 方便看终端。易错：采集器按 JSON 解析，格式仍是 text。 |
| `log_requests_target` | --log-requests-target | `Optional[List[str]]` | `None` | 作用：请求日志写到哪。可以是 stdout，也可以是目录，可以多个，例如 stdout 加一个路径。默认与效果：默认空，走进程默认输出。交互与改写：目录必须可写。易错：路径是文件不是目录，或反过来。 |
| `uvicorn_access_log_exclude_prefixes` | --uvicorn-access-log-exclude-prefixes | `List[str]` | `msgspec.field(default_factory=list)` | 作用：请求路径以这些前缀开头的，不打 uvicorn 访问日志。默认空列表，等于不排除。例如去掉 /metrics 和 /health。默认与效果：默认空。交互与改写：只影响访问日志，不影响 log_requests。易错：写成不带斜杠的 health，前缀对不上。 |
| `crash_dump_folder` | --crash-dump-folder | `Optional[str]` | `None` | 作用：崩溃前约 5 分钟的请求会倒到这个目录。不填则不做崩溃转储。默认与效果：默认空，关闭。交互与改写：必须在 CUDA 初始化前就定好目录，所以放在解析后段。易错：目录不存在，真崩溃时写不进去。 |
| `show_time_cost` | --show-time-cost | `bool` | `False` | 作用：打印自定义计时点的耗时。默认与效果：默认关闭。交互与改写：只影响日志。易错：和 enable_metrics 混淆，以为会有 Prometheus 直方图。 |
| `enable_metrics` | --enable-metrics | `bool` | `False` | 作用：打开 Prometheus 风格指标。 默认与效果：默认按字段。 交互与改写：不影响数值。/server_info 里的 internal_states 是另一条状态，DSPARK 测试用它看接受长度。 易错：只开 metrics 却去 /server_info 找直方图。 |
| `smg_http_sidecar_port` | --smg-http-sidecar-port（--grpc-http-sidecar-port） | `Optional[int]` | `None` | 作用：旧的 SMG gRPC 模式下，HTTP 边车的端口，用来提供 Prometheus 和剖析。默认是主端口加 1。普通 HTTP 模式不用。默认与效果：默认空，需要时用 port+1。交互与改写：和原生 grpc_port 不是同一个。易错：HTTP 模式去连这个端口。 |
| `enable_mfu_metrics` | --enable-mfu-metrics | `bool` | `False` | 作用：打开估算的 MFU 相关 Prometheus 指标。默认与效果：默认关闭。交互与改写：需要 enable_metrics，否则没有出口。易错：只开 MFU 不开总指标。 |
| `enable_metrics_for_all_schedulers` | --enable-metrics-for-all-schedulers | `bool` | `False` | 作用：让每个 TP rank 上的 scheduler 都单独记请求指标，不只是 TP 0。attention DP 时特别有用，否则指标看起来全来自 TP 0。默认与效果：默认关闭。交互与改写：指标基数会乘上 scheduler 数。易错：DP attention 没开它，监控把所有组算成一个。 |
| `load_snapshot_publish_interval` | --load-snapshot-publish-interval | `int` | `15` | 作用：每隔多少次解码迭代把负载快照写到共享内存。预填充和空闲总是立刻发布。默认与效果：默认 15。交互与改写：load_publish_endpoint 的刷新也跟着这个间隔，设得很大，路由器看到的负载会旧。易错：把它理解成秒。它是解码迭代次数。 |
| `tokenizer_metrics_custom_labels_header` | --tokenizer-metrics-custom-labels-header | `str` | `'x-custom-labels'` | 作用：请求用哪个 HTTP 头把自定义标签传给 tokenizer 指标。默认与效果：默认 x-custom-labels。交互与改写：头里的键还必须在允许列表里。易错：改了头名，客户端仍写旧头。 |
| `tokenizer_metrics_allowed_custom_labels` | --tokenizer-metrics-allowed-custom-labels | `Optional[List[str]]` | `None` | 作用：允许出现在指标里的自定义标签名。请求头里的字典只有这些键会被收下。默认与效果：默认空，不收自定义标签。交互与改写：未列出的键被丢掉，不会报错。易错：以为任意键都会变成标签，基数爆炸或干脆没生效。 |
| `extra_metric_labels` | --extra-metric-labels | `Optional[Dict[str, str]]` | `None` | 作用：给指标加的固定标签，JSON 如 {"label1": "value1"}。默认与效果：默认空。交互与改写：这是进程级标签，不是按请求的。易错：和请求头标签混用，不知道值从哪来。 |
| `bucket_time_to_first_token` | --bucket-time-to-first-token | `Optional[List[float]]` | `None` | 作用：首 token 延迟直方图的桶边界，一串浮点数。默认与效果：默认用内置桶。交互与改写：桶要递增。易错：单位当成毫秒，实现是秒，桶全部挤在一头。 |
| `bucket_inter_token_latency` | --bucket-inter-token-latency | `Optional[List[float]]` | `None` | 作用：token 间隔延迟直方图的桶。默认与效果：默认内置。交互与改写：和 TTFT、端到端是三条直方图。易错：三条用了同一组不合适的桶。 |
| `bucket_e2e_request_latency` | --bucket-e2e-request-latency | `Optional[List[float]]` | `None` | 作用：请求端到端延迟直方图的桶。默认与效果：默认内置。交互与改写：长输出的请求会落到最后一个桶。易错：最大桶小于实际延迟，直方图失真。 |
| `prompt_tokens_buckets` | --prompt-tokens-buckets | `Optional[List[str]]` | `None` | 作用：提示 token 数直方图的分桶规则。default 用预置桶；tse 中间值 底数 个数 生成两侧指数桶；custom 后面跟自定义边界。默认与效果：默认按 default 规则。交互与改写：和生成 token 的桶是分开的。易错：tse 的三个参数顺序写反。 |
| `generation_tokens_buckets` | --generation-tokens-buckets | `Optional[List[str]]` | `None` | 作用：生成 token 数直方图的分桶规则，语法和提示 token 的桶相同。默认与效果：默认 default。交互与改写：不改变采样，只改变指标。易错：把 prompt 的桶抄过来，长生成全部进溢出桶。 |
| `gc_warning_threshold_secs` | --gc-warning-threshold-secs | `float` | `0.0` | 作用：一次 GC 超过这么多秒就打警告。0 表示不警告。默认与效果：默认 0，关闭警告。交互与改写：不改变 GC 频率，那是 gc_threshold。易错：设成 0 以为是「超过 0 秒就警告」，实际是关闭。 |
| `decode_log_interval` | --decode-log-interval | `int` | `40` | 作用：解码批每隔多少次迭代打一次日志和指标。默认与效果：默认 40。交互与改写：间隔大则日志少，指标也更稀。易错：把它当成秒。 |
| `enable_request_time_stats_logging` | --enable-request-time-stats-logging | `bool` | `False` | 作用：按请求把时间拆分打到日志。默认与效果：默认关闭。交互与改写：和 show_time_cost 的自定义标记不是同一套。易错：打开后日志量随 QPS 线性涨。 |
| `kv_events_config` | --kv-events-config | `Optional[str]` | `None` | 作用：NVIDIA Dynamo 风格的 KV 事件发布配置，JSON。写了这个旗标就会发布。给路由器补拉丢批可以设 replay_endpoint，例如 tcp://*:5558，端口会出现在 /server_info。负载发布是另一个开关 load_publish_endpoint。默认与效果：默认空，不发布。交互与改写：路由器要能订到这个地址。易错：和负载发布混成一个配置。 |
| `load_publish_endpoint` | --load-publish-endpoint | `Optional[str]` | `None` | 作用：给负载感知路由器订的运行时负载 PUB。默认关。auto 会在 KV 事件端口段后面紧挨着留出 dp_size 个端口；也可以写 tcp://*:6000 这种通配地址，rank r 绑定 port+r。必须已经有 kv_events_config，否则启动失败。auto 会从 KV 基址再留 2*dp_size 个端口。刷新间隔跟 load_snapshot_publish_interval。默认与效果：默认不发布。交互与改写：端口和 KV 事件重叠会启动失败。易错：auto 和别的进程抢了同一段端口。 |
| `enable_forward_pass_metrics` | --enable-forward-pass-metrics | `bool` | `False` | 作用：通过 ZMQ 把每次前向的指标发出去，外部规划器来订 forward_pass_metrics_ipc_name。默认与效果：默认关闭。交互与改写：打开后每步都有 IPC。易错：没有消费者仍打开，白占带宽。 |
| `forward_pass_metrics_worker_id` | --forward-pass-metrics-worker-id | `str` | `''` | 作用：前向指标里标记这个 worker 的 id。声明处没有展开 help。默认与效果：默认空字符串。交互与改写：多 worker 时用来区分是谁发出的。易错：所有进程留空，消费者分不清。 |
| `forward_pass_metrics_ipc_name` | --forward-pass-metrics-ipc-name | `Optional[str]` | `None` | 作用：前向指标的 ZMQ 地址。声明处 help 为空，由实现在打开指标时填上。默认与效果：默认空。交互与改写：消费者必须订这个名字。易错：在命令行写了一个和实现生成的不一致的地址。 |
| `enable_trace` | --enable-trace | `bool` | `False` | 作用：打开 OpenTelemetry 追踪。默认与效果：默认关闭。交互与改写：还要有端点和要追踪的模块。易错：只开开关，没有导出端点，追踪停在进程内。 |
| `trace_modules` | --trace-modules | `str` | `'request'` | 作用：追踪哪些组件。可选 request 和 mooncake，用逗号分隔。默认与效果：默认 request。交互与改写：mooncake 只在用了该传输时有跨度。易错：写成列表 JSON 而不是逗号分隔。 |
| `otlp_traces_endpoint` | --otlp-traces-endpoint | `str` | `'localhost:4317'` | 作用：OTLP 追踪导出的地址。默认与效果：默认空。交互与改写：enable_trace 为真时应该能连上。易错：写了 HTTP 完整路径，协议其实要 grpc 或反过来。 |
| `otlp_service_name` | --otlp-service-name | `Optional[str]` | `None` | 作用：导出追踪时用的服务名。默认与效果：默认按实现或 sglang。交互与改写：多实例要用不同名字才分得开。易错：所有副本同一个名字，追踪混在一起。 |
| `export_metrics_to_file` | --export-metrics-to-file | `bool` | `False` | 作用：把指标写到文件，而不是只留在拉取端点。默认与效果：默认关闭。交互与改写：目录由 export_metrics_to_file_dir 指定。易错：打开了但目录不可写。 |
| `export_metrics_to_file_dir` | --export-metrics-to-file-dir | `Optional[str]` | `None` | 作用：指标文件的目录。默认与效果：默认按实现。交互与改写：只在导出到文件打开时使用。易错：和 crash dump 目录写成同一个，互相覆盖。 |
| `stat_loggers` | --stat-loggers | `Optional[Dict[str, type]]` | `None` | 作用：额外的统计日志器列表。没有常规 CLI 时由代码挂上。默认与效果：默认空。交互与改写：不影响数值。易错：在 YAML 里写了它，argparse 不认识。 |
| `kv_canary` | --kv-canary | `str` | `'none'` | 作用：打开 KV 金丝雀，用已知模式检查缓存是否被写坏。默认与效果：默认关闭。交互与改写：真实数据检查范围由 kv_canary_real_data 决定。易错：生产长期打开全量检查，带宽被占满。 |
| `kv_canary_real_data` | --kv-canary-real-data | `str` | `'none'` | 作用：金丝雀检查真实 KV 的范围。none 不查真实数据，partial 查每槽前 16 字节，all 查整槽。默认与效果：默认 none。交互与改写：choices 来自 RealKvHashMode。易错：all 开在大缓存上。 |
| `kv_canary_sweep_interval` | --kv-canary-sweep-interval | `int` | `0` | 作用：金丝雀扫描的间隔。默认与效果：默认按实现。交互与改写：间隔越小，检查越勤，越影响吞吐。易错：设成每步都扫。 |
| `debug_tensor_dump_output_folder` | --debug-tensor-dump-output-folder | `Optional[str]` | `None` | 作用：调试时把张量倒到这个目录。默认与效果：默认空，不倒。交互与改写：还要指定层和输入文件才有内容。易错：目录在很小的磁盘上，一次前向就写满。 |
| `debug_tensor_dump_layers` | --debug-tensor-dump-layers | `Optional[List[int]]` | `None` | 作用：要倒出哪些层。默认与效果：默认空。交互与改写：层号按模型实现的编号，不是 Hugging Face 的名字。易错：层号越界。 |
| `debug_tensor_dump_input_file` | --debug-tensor-dump-input-file | `Optional[str]` | `None` | 作用：指定用哪个输入文件驱动这次张量转储。默认与效果：默认空。交互与改写：和在线请求的转储不是同一条路径。易错：文件格式不对，转储还没开始就失败。 |
| `forward_hooks` | --forward-hooks | `Optional[List[dict[str, Any]]]` | `None` | 作用：在前向的指定位置挂调试钩子。默认与效果：默认空，不挂。交互与改写：钩子会拖慢前向，并可能改变显存。易错：生产配置留着钩子。 |
| `msprobe_dump_config` | --msprobe-dump-config | `Optional[str]` | `None` | 作用：昇腾 msprobe 的转储配置。默认与效果：默认空。交互与改写：只在 NPU 路径有这个工具。易错：在 CUDA 上填写它。 |

### 执行特性开关

命名空间 `exec.features`，类 `ExecFeatures`，`python/sglang/srt/arg_groups/fields/exec_.py`，27 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `enable_fp32_lm_head` | --enable-fp32-lm-head | `bool` | `False` | 作用：LM head 的 logits 用 FP32 计算，避免低精度把分数挤坏。默认与效果：默认关闭，logits 跟权重精度走。交互与改写：只影响最后一层得分，不改变 KV 和并行。易错：为了省显存关掉它，采样在 FP8 logits 上不稳定。打开则多占带宽。 |
| `enable_tf32_matmul` | --enable-tf32-matmul（S2 可改写） | `Optional[bool]` | `None` | 作用：允许 float32 矩阵乘用 TensorFloat32，通过 torch.set_float32_matmul_precision。仅 CUDA。不填则跟模型偏好，否则关闭。默认与效果：默认空，按模型偏好。交互与改写：TF32 会改变数值，和确定性模式冲突时以确定性为准。易错：在 HIP 上打开，没有 TensorFloat32。 |
| `enable_memory_saver` | --enable-memory-saver | `bool` | `False` | 作用：允许用 release_memory_occupation 和 resume_memory_occupation 把占用让出来再拿回来。默认与效果：默认关闭，那两个接口不能真正释放。交互与改写：和权重 CPU 备份是两条路径，一个放 KV/占用，一个放权重。易错：没开这个开关就调用释放接口。 |
| `enable_weights_cpu_backup` | --enable-weights-cpu-backup | `bool` | `False` | 作用：release_weights_occupation 时把主模型和草稿的权重都拷到 CPU，resume 时拷回。默认与效果：默认关闭，释放权重就是卸掉，没有备份。交互与改写：CPU 内存要装得下整份权重。易错：和 draft 专用备份一起开，草稿被备份两次。 |
| `enable_draft_weights_cpu_backup` | --enable-draft-weights-cpu-backup | `bool` | `False` | 作用：只把草稿权重备份到 CPU，主模型不备份。默认与效果：默认关闭。交互与改写：没有投机草稿时没有对象可备份。易错：开了 DSPARK 却只开主模型备份，草稿在释放后丢了。 |
| `enable_custom_logit_processor` | --enable-custom-logit-processor | `bool` | `False` | 作用：允许请求里带自定义 logit 处理器。默认关闭是为了安全，因为处理器相当于在进程里跑调用方代码。默认与效果：默认关闭，请求里的 custom_logit_processor 不会被执行。交互与改写：MLX 路径另有自己的开关 mlx_enable_sampling。易错：客户端传了处理器但服务没开，请求被忽略或拒绝，生成结果和本地预期不一致。 |
| `enable_watermark` | --enable-watermark | `bool` | `False` | 作用：打开 Aaronson-Gumbel 文本水印。默认与效果：默认关闭，采样不加扰动。交互与改写：具体参数来自 watermark_config，而不是零散字段。易错：只开开关不给配置，密钥和窗口用解析后的默认，水印强度可能不是你要的。 |
| `watermark_config` | --watermark-config | `Optional[str]` | `None` | 作用：水印配置，可以是 JSON 文件路径或内联 JSON。字段包括 key、key_b、context_window、mixing_probability、max_probability、default_enabled、enforce_all。默认与效果：默认空。交互与改写：下面几个 watermark_* 字段都从这里解析出来，不要两头各写一套。易错：文件路径写错，解析阶段读不到密钥。 |
| `watermark_key` | --watermark-key（无常规 CLI；S2 可改写） | `Optional[str]` | `None` | 作用：水印密钥，从 watermark_config 解析，不是独立 CLI。默认与效果：默认空，直到配置提供。交互与改写：和 key_b 一起决定扰动序列。易错：在命令行上找 --watermark-key，找不到是因为设计上不直接暴露。 |
| `watermark_key_b` | --watermark-key-b（无常规 CLI；S2 可改写） | `Optional[str]` | `None` | 作用：水印的第二密钥，同样只从 watermark_config 来。默认与效果：默认空。交互与改写：和 watermark_key 成对。易错：只改 key 不改 key_b，检测端对不上。 |
| `watermark_context_window` | --watermark-context-window（无常规 CLI；S2 可改写） | `int` | `4` | 作用：水印使用的上文窗口长度，从配置解析。默认与效果：声明默认 4。交互与改写：窗口越大，对前文越敏感，也越贵。易错：配置 JSON 里写了别的数，声明默认 4 不会再生效。 |
| `watermark_mixing_probability` | --watermark-mixing-probability（无常规 CLI；S2 可改写） | `float` | `0.5` | 作用：水印混入采样分布的概率，从配置解析。默认与效果：声明默认 0.5。交互与改写：越高越容易检测，对原文分布干扰越大。易错：设成 1 又把 max_probability 留在 1，文本质量明显下降。 |
| `watermark_max_probability` | --watermark-max-probability（无常规 CLI；S2 可改写） | `float` | `1.0` | 作用：水印允许把某个 token 概率抬到的上限，从配置解析。默认与效果：声明默认 1.0，等于不额外封顶。交互与改写：应不小于正常采样概率，否则和 temperature 打架。易错：设得很小，采样几乎被水印锁死。 |
| `watermark_default_enabled` | --watermark-default-enabled（无常规 CLI；S2 可改写） | `bool` | `False` | 作用：请求没写水印字段时是否默认加水印，从配置解析。默认与效果：声明默认关闭。交互与改写：enforce_all 为真时这个默认不再重要，全部请求都被加水印。易错：以为打开 enable_watermark 就会给每个请求加水印。 |
| `watermark_enforce_all` | --watermark-enforce-all（无常规 CLI；S2 可改写） | `bool` | `False` | 作用：强制所有请求加水印，从配置解析。默认与效果：声明默认关闭。交互与改写：打开后客户端无法按请求关掉。易错：对外服务误开，所有用户文本都带水印偏移。 |
| `enable_return_hidden_states` | --enable-return-hidden-states | `bool` | `False` | 作用：允许响应里带回完整隐状态，等价于 return_hidden_states_mode=full。默认与效果：默认关闭，响应不含隐状态。交互与改写：full 模式很占带宽和显存。和辅助隐状态开关一起用时，返回的是拼接后的 aux，不是最后一层。易错：客户端写了 return_hidden_states 但服务没开模式，字段被拒绝。 |
| `return_hidden_states_mode` | --return-hidden-states-mode | `Optional[str]` | `None` | 作用：服务允许的最大隐状态返回级别。last 只允许请求要最后一层或不要；full 才允许请求要全部。默认与效果：默认空。交互与改写：enable_return_hidden_states 会把它收成 full；辅助隐状态在未设置时也会把它收成 full。易错：设成 last 却让请求要完整序列隐状态。 |
| `enable_return_aux_hidden_states` | --enable-return-aux-hidden-states | `bool` | `False` | 作用：在目标模型上抓辅助隐状态，但不跑投机草稿。此时 speculative_algorithm 只用来点名抓取约定（EAGLE3、DFLASH 或 DSPARK），然后被重置成空。请求若要隐状态，拿到的是拼接后的 aux，而不是最后一层。未设置返回模式时会把它设为 full。要求 pp_size 为 1。默认与效果：默认关闭。交互与改写：会清掉投机算法，不要和真的 DSPARK 同时指望两者都在。易错：开了它又开 DSPARK 草稿，算法名被重置，草稿不会跑。 |
| `aux_hidden_state_capture` | --aux-hidden-state-capture（无常规 CLI；S2 可改写） | `Optional[str]` | `None` | 作用：辅助隐状态的抓取约定，由 enable_return_aux_hidden_states 和 speculative_algorithm 解析出来。默认与效果：派生，无独立 CLI。交互与改写：决定层号列表怎么解释。易错：在命令行上找这个名字。 |
| `return_aux_hidden_state_layer_ids` | --return-aux-hidden-state-layer-ids | `Optional[List[int]]` | `None` | 作用：按选定约定，目标模型上要抓的层号。它们可以和草稿检查点里的层号不同。DFLASH 和 DSPARK 必须给；EAGLE3 默认用模型自己的 EAGLE3 层。默认与效果：默认空。交互与改写：层号越界会在建模型时失败。易错：把草稿的层号抄过来，和目标层对不上。 |
| `enable_return_routed_experts` | --enable-return-routed-experts | `bool` | `False` | 作用：响应里带回每一层选中的专家。默认与效果：默认关闭。交互与改写：和请求字段 return_routed_experts 同时为真才返回。数据量随层数和 top-k 涨。易错：只在请求里开，服务端开关是关的。 |
| `enable_return_indexer_topk` | --enable-return-indexer-topk | `bool` | `False` | 作用：响应里带回带 indexer 的层的 top-k 下标。默认与效果：默认关闭。交互与改写：只对有 indexer 的层有内容，V4 的 index 层才有。易错：在普通 GQA 模型上打开，字段是空的。 |
| `enable_encoder_swa_bounded_replay` | --enable-encoder-swa-bounded-replay | `bool` | `False` | 作用：V4.1 编码器滑动窗口的有界重放：只缓存主 KV 和 indexer 的 key，前缀命中时重建属于请求的 SWA 窗口。实验功能，只支持 CUDA 和 gfx950。默认与效果：默认关闭。交互与改写：HIP 上不能和 attention DP、prefill CUDA 图、CP、LoRA、session radix、PD 同时开，见 validate_deepseek_v41_features。易错：在四机 DP attention 上打开。 |
| `enable_decoder_swa_bounded_replay` | --enable-decoder-swa-bounded-replay（S2 可改写） | `Optional[bool]` | `None` | 作用：V4.1 解码器 SWA 有界重放。支持的配置上默认开。数值上不等于完整 prefill；要 prompt logprob 的请求会把每一行都算出来。用 --no-enable-decoder-swa-bounded-replay 可以留住完整 prefill，给必须和训练器对齐的 RL 用。默认与效果：默认空，支持时会被打开。交互与改写：和 encoder 那条重放是两个开关。易错：RL 没关它，logprob 和训练对不上。 |
| `sampling_mask_max_tokens` | --sampling-mask-max-tokens | `int` | `4096` | 作用：返回的采样掩码里最多多少个 token id。实际支持集超过就中止请求。PD 两边要设成同一个数。客户端应把 top_k 留在这个上限之下，给并列截断留余量。默认与效果：默认 4096。交互与改写：top_k 大于它时请求会被中止。易错：只在 prefill 上改，decode 仍是另一个数。 |
| `disable_outlines_disk_cache` | --disable-outlines-disk-cache | `bool` | `False` | 作用：关掉 outlines 的磁盘缓存，避免文件系统或高并发下崩溃。默认与效果：默认关闭这个「关掉」，也就是允许磁盘缓存。交互与改写：只影响 grammar 后端是 outlines 的时候。易错：名字是否定之否定，写成 true 是禁用缓存。 |
| `enable_mis` | --enable-mis | `bool` | `False` | 作用：多条目打分。把查询和多个条目拼成一条序列，方便一次批处理。要求 attention 后端是 flashinfer，并会自动关掉 CUDA 图、前缀树和分块预填充。默认与效果：默认关闭。交互与改写：打开后 radix 和 chunk 都不在。易错：指望 MIS 还能命中前缀缓存。 |

### Kernel、Attention、GEMM

命名空间 `exec.kernel`，类 `ExecKernel`，`python/sglang/srt/arg_groups/fields/exec_.py`，29 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `attention_backend` | --attention-backend（S2 可改写） | `Optional[str]` | `None` | 作用：注意力 kernel 的总选择。预填充和解码还可以分别覆盖。默认与效果：默认空，按模型和平台选。交互与改写：V4.1 会被改成 dsv4，page_size 改成 256。HIP 上的 V4 prefill CP 要求两相都是 dsv4。易错：手写 triton 覆盖了 V4 需要的 dsv4。 |
| `decode_attention_backend` | --decode-attention-backend（S2 可改写） | `Optional[str]` | `None` | 作用：只覆盖解码相的注意力 kernel，优先级高于 attention_backend。默认与效果：默认空，跟总后端。交互与改写：和 prefill 设成不同值时，V4 的双相检查可能失败。易错：解码用一个、预填充用另一个，却开了要求两相一致的路径。 |
| `enable_lean_attention` | --enable-lean-attention | `Optional[bool]` | `None` | 作用：长上下文解码用 Lean（以工作为中心）注意力 kernel。空表示自动：长上下文用 Lean，短的退回普通 kernel。True 强制开，False 强制关。默认与效果：默认空，走自动门。交互与改写：不是所有后端都实现了 Lean。易错：短请求强制开，自动门本来会避开的开销被你打开了。 |
| `enable_nvidia_megakernel` | --enable-nvidia-megakernel | `bool` | `False` | 作用：用 sgl-nvidia-mega 包里的 NVIDIA megakernel。支持的解码和校验步走融合 kernel，其他步仍是普通前向。配置不支持时，启动会列出原因并报错。实验功能。默认与效果：默认关闭。交互与改写：和 AMD monokernel 同时开，S2 会把两边的原因一次报齐。易错：在海光上打开。 |
| `enable_amd_monokernel` | --enable-amd-monokernel | `bool` | `False` | 作用：用 sgl-amd-mono 的持久 kernel。支持的解码和校验步走它，其他步普通前向。不支持的配置会列出原因并失败。实验功能。默认与效果：默认关闭。旧旗标 --monokernel-backend amd 指向它。交互与改写：和 NVIDIA megakernel 互斥。易错：包没装就打开。 |
| `prefill_attention_backend` | --prefill-attention-backend（S2 可改写） | `Optional[str]` | `None` | 作用：只覆盖预填充相的注意力 kernel，优先级高于总后端。默认与效果：默认空。交互与改写：HIP 上 V4 的 prefill CP 要求它和 decode 都是 dsv4。易错：只改预填充后端，解码仍是别的。 |
| `prefill_kv_cache_dequant_dtype` | --prefill-kv-cache-dequant-dtype（S2 可改写） | `str` | `'auto'` | 作用：当 KV 以 nvfp4 存储时，预填充注意力在线反量化成什么。nvfp4 表示直接读打包缓存；fp8_e4m3 先反量化到临时 FP8 工作区。不改变库存的 dtype。auto 在 SM100 上选不再额外反量化的 NVFP4，其他平台选 FP8 E4M3。默认与效果：默认 auto。交互与改写：KV 不是 nvfp4 时这个选项没有对象。易错：以为它会把 KV 存成 FP8。 |
| `sampling_backend` | --sampling-backend（无常规 CLI；S2 可改写） | `Optional[str]` | `None` | 作用：采样层用哪套 kernel。choices 在 S1 动态生成，打开 KV canary 的 token oracle 时会多一个选项。默认与效果：默认空，按平台选。交互与改写：确定性推理会限制可用后端。易错：写了未注册的名字，argparse 直接失败。 |
| `grammar_backend` | --grammar-backend | `Optional[str]` | `None` | 作用：语法约束解码的后端，例如 xgrammar、llguidance、outlines。默认与效果：默认空，按可用性选。交互与改写：outlines 还有磁盘缓存开关。MIS 等路径可能根本不走语法。易错：机器没装对应包，第一次带 schema 的请求才失败。 |
| `fp8_gemm_runner_backend` | --fp8-gemm-runner-backend（S2 可改写） | `str` | `'auto'` | 作用：块状 FP8 GEMM 用哪个执行器。auto 按硬件选：MXFP8 稠密在 SM100/SM103 上选 flashinfer_cutedsl，其他受支持的 Blackwell 选 FlashInfer CUTLASS。还可显式选 deep_gemm、flashinfer_trtllm、flashinfer_cutlass、flashinfer_cutedsl、flashinfer_deepgemm（仅 Hopper 小 M）、cutlass（偏 SM120）、triton、aiter（仅 ROCm）。默认与效果：默认 auto。交互与改写：V4.1 在 CUDA、非 HIP、SM100、FP8 块 32×32 且 ue8m0 时会被改成 flashinfer_cutedsl。易错：在海光上指定 flashinfer_cutedsl。 |
| `fp4_gemm_runner_backend` | --fp4-gemm-runner-backend（S2 可改写） | `str` | `'auto'` | 作用：NVFP4 GEMM 的执行器。auto 在 SM100 上选 flashinfer_cutedsl，SM80 到 SM90 选 marlin，其余含 SM120 选 flashinfer_cutlass。还可选 flashinfer_cudnn、flashinfer_trtllm（权重要按它的洗牌方式准备）、marlin（SM80+ 的 W4A16 回退）。默认与效果：默认 auto。交互与改写：和权重的量化格式必须匹配，选错会在加载或第一次 GEMM 失败。易错：trtllm 后端却用了 cutlass 的权重布局。 |
| `bf16_gemm_backend` | --bf16-gemm-backend | `str` | `'auto'` | 作用：未量化的 BF16 GEMM。auto 在 SM10x 上选 cutedsl，确定性推理则选 torch，其他情况用 cuBLAS。cutedsl 会在小 M 的 Split-K、CuTe DSL 和 cuBLAS 之间分发，SGLANG_ENABLE_BF16_SPLITK_GEMM=0 可关 Split-K。还可选 gemv 或 torch。默认与效果：默认 auto。交互与改写：确定性模式会把 auto 拉到 torch。易错：在非 SM10x 上强制 cutedsl。 |
| `dsa_prefill_backend` | --dsa-prefill-backend（S2 可改写） | `Optional[str]` | `None` | 作用：DeepSeek 稀疏注意力的预填充后端。不填则按硬件和 kv_cache_dtype 自动选。默认与效果：默认空，走自动。交互与改写：和 dsv4_prefill_backend 不是同一个开关，V4 用后者。易错：给 V4.1 只设了 DSA 预填充后端，V4 路径不读它。 |
| `dsv4_prefill_backend` | --dsv4-prefill-backend | `str` | `'auto'` | 作用：DeepSeek-V4 稀疏预填充。auto 和 flashmla_sparse 走现有 BF16 稀疏预填充；flashmla_sparse_q8 走 Q8KV8。默认与效果：默认 auto。交互与改写：和 dsv4_attn_backend 一起决定稀疏核，不是普通 MHA 后端。易错：KV 不是 Q8 却选 q8 路径。 |
| `dsa_triton_union` | --dsa-triton-union | `int` | `0` | 作用：triton_sparse_mla 预填充里，相邻多少个 query token 共用一套搜集到的索引。0 关闭，2 或 4 表示 G 个相邻 token 共享，所有权掩码再把每个 token 自己的 softmax 支持恢复回来，数学上仍然精确。默认与效果：默认 0，不共享。交互与改写：只在该 triton 后端下生效。易错：设成 3 这种不支持的组大小。 |
| `dsa_decode_backend` | --dsa-decode-backend（S2 可改写） | `Optional[str]` | `None` | 作用：DSA 解码后端。不填则按硬件和 KV dtype 自动选。默认与效果：默认空。交互与改写：和预填充后端可以不同。易错：解码选了预填充才有的核。 |
| `dsv4_attn_backend` | --dsv4-attn-backend（S2 可改写） | `str` | `'auto'` | 作用：DeepSeek V4 注意力实现，预填充和解码共用。auto 在 flashmla_mega 不支持时，或非 PD 且注意力 TP 大于 1 时，解析成 flashmla，否则 flashmla_mega。trtllm 是 SM100/SM103 加 FP8 KV 的可选项，把 SWA 和压缩 KV 收成统一的 512 维 FP8，解码和稀疏预填充走 flashinfer 的 trtllm-gen 稀疏 MLA。flashmla_mega 是 SM100/SM103 上 V4.1 的融合 Q-RoPE、稀疏注意力、逆 RoPE 和 MXFP8。默认与效果：默认 auto。交互与改写：V4.1 配 TRT-LLM 时，prefill CUDA 图必须是 disabled 或 breakable。易错：多卡注意力 TP>1 却以为会走到 flashmla_mega。 |
| `dsa_paged_mqa_logits_backend` | --dsa-paged-mqa-logits-backend | `str` | `'auto'` | 作用：DSA indexer 的分页 MQA logits kernel。auto 在 CUDA 上用 DeepGEMM，ROCm 上用 aiter。cutedsl 只在 SM100，小批长上下文更合适。默认与效果：默认 auto。交互与改写：ROCm 上选 deepgemm 不会有实现。易错：在 gfx 上指定 cutedsl。 |
| `dsa_topk_backend` | --dsa-topk-backend（S2 可改写） | `str` | `'sgl-kernel'` | 作用：目标模型 DSA indexer 的 top-k 后端，可选 sgl-kernel、torch、flashinfer。torch 目前要求环境变量 SGLANG_DSA_FUSE_TOPK=false。默认与效果：默认 sgl-kernel。交互与改写：草稿有自己的 speculative_dsa_topk_backend。易错：切到 torch 却没关融合 topk，启动或第一次 indexer 失败。 |
| `disable_flashinfer_autotune` | --disable-flashinfer-autotune | `bool` | `False` | 作用：关掉 FlashInfer 的自动调优。默认与效果：默认不关，允许调优。交互与改写：调优发生在启动，关掉能缩短启动、可能损失已调好的形状。易错：V4.1 测试脚本里会关它，生产照抄会错过调优。 |
| `flashinfer_autotune_skip_ops` | --flashinfer-autotune-skip-ops | `Optional[List[str]]` | `None` | 作用：自动调优时跳过的 FlashInfer 自定义算子名。被跳过的用启发式回退。默认与效果：默认空，全部参与调优。交互与改写：名字必须是 FlashInfer 认识的标识。易错：写错算子名，该项仍被调优，启动更慢且没有报错。 |
| `triton_attention_reduce_in_fp32` | --triton-attention-reduce-in-fp32 | `bool` | `False` | 作用：Triton 注意力的中间结果用 FP32 归约，避免 FP16 溢出崩溃。只影响 Triton 核。默认与效果：默认关闭。交互与改写：其他后端不读这个开关。易错：用 flashinfer 时打开它，以为数值会变，实际不变。 |
| `triton_attention_num_kv_splits` | --triton-attention-num-kv-splits | `int` | `8` | 作用：Flash decoding 的 Triton 核把 KV 切成几段。上下文越长，更大的切分通常更好。默认与效果：默认 8。交互与改写：只影响 Triton flash decoding。易错：短上下文设得很大，切分开销超过收益。 |
| `triton_attention_split_tile_size` | --triton-attention-split-tile-size | `Optional[int]` | `None` | 作用：上述 Triton 核里每一段 KV tile 的大小，给确定性推理用。默认与效果：默认空，用核自己的 tile。交互与改写：和 enable_deterministic_inference 一起才有稳定含义。易错：不开启确定性却依赖这个值去对齐比特。 |
| `flashinfer_mla_disable_ragged` | --flashinfer-mla-disable-ragged | `bool` | `False` | 作用：跑 FlashInfer MLA 时不用 ragged 预填充包装。默认与效果：默认不禁用，也就是使用 ragged 包装。交互与改写：变长预填充没有包装时，形状必须自己对齐。易错：名字是否定式，设 true 是禁用包装。 |
| `enable_fused_qk_norm_rope` | --enable-fused-qk-norm-rope（S2 可改写） | `Optional[bool]` | `None` | 作用：把 Q/K 归一化和 RoPE 融进一个核。XPU 上默认开，其他地方默认关。可用正向或 --no- 强制。默认与效果：默认空，按平台。交互与改写：融合核不支持的头维度会在选择后端时被拒绝。易错：在 CUDA 上以为默认是开的。 |
| `enable_precise_embedding_interpolation` | --enable-precise-embedding-interpolation | `bool` | `False` | 作用：嵌入网格缩放时做角点对齐，插值更准但更慢。默认与效果：默认关闭，用更快的插值。交互与改写：主要影响视觉或多模态位置嵌入。易错：文本模型打开它，几乎没有收益。 |
| `enable_deepseek_v4_fp4_indexer` | --enable-deepseek-v4-fp4-indexer | `bool` | `False` | 作用：V4 的实验性 FP4 C4 indexer。默认仍用现有 indexer。默认与效果：默认关闭。交互与改写：和 dsv4 注意力后端要配套，实验路径。易错：生产 V4.1 打开它却没有对应权重布局。 |
| `enable_dsa_fused_indexer` | --enable-dsa-fused-indexer（S2 可改写） | `Optional[bool]` | `None` | 作用：用四核融合的 DSA indexer 解码，替代 12 次启动的 aiter/torch 链。默认在 gfx950、aiter preshuffle、fp8 e4m3fn 索引缓存都具备时打开，缺任何一条会在日志里说明并退回。可用 --no- 强制走标准路径。默认与效果：默认空，按支持情况。交互与改写：不是 gfx950 就不要指望它开着。易错：强制打开，机器缺 aiter。 |

### Mamba / 线性注意力

命名空间 `exec.mamba`，类 `ExecMamba`，`python/sglang/srt/arg_groups/fields/exec_.py`，19 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `enable_mamba_extra_buffer` | --enable-mamba-extra-buffer（fn=sglang.srt.arg_groups.model_override_base.mamba_extra_buffer_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：混合 Mamba 的前缀树是否保留额外状态缓冲。条件是树开着且策略要求这块缓冲。派生量。默认与效果：发布时计算，无 CLI。交互与改写：由 mamba_radix_cache_strategy 决定。易错：在命令行上找这个名字。 |
| `enable_mamba_extra_buffer_lazy` | --enable-mamba-extra-buffer-lazy（fn=sglang.srt.arg_groups.overrides.mamba_extra_buffer_lazy_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：策略恰好是 extra_buffer_lazy 时为真。派生量。默认与效果：无 CLI。交互与改写：和上一字段互斥地描述两种策略。易错：两个都当成可以手写的开关。 |
| `mamba_backend` | --mamba-backend | `str` | `'triton'` | 作用：Mamba SSM 核。triton 或 flashinfer（需要带 Mamba 的 FlashInfer）。默认与效果：默认 triton。交互与改写：随机舍入在 triton 上还要求 SM100。易错：没装对应 FlashInfer 却选 flashinfer。 |
| `mamba_ssm_dtype` | --mamba-ssm-dtype | `Optional[str]` | `None` | 作用：Mamba 缓存里 SSM 状态的数据类型。不填则读模型配置里的 mamba_ssm_dtype。默认与效果：默认空，跟配置。交互与改写：FP16 随机舍入要求这里是 float16。易错：配置是 bf16，这里又写成 float16，两边不一致。 |
| `mamba_max_states_per_path` | --mamba-max-states-per-path | `int` | `-1` | 作用：每条从根到叶的路径上最多留多少个 Mamba 状态。-1 表示不限制。启用后，插入之后会删掉超出上限的最浅内部状态，完整 KV 仍留着。尾、分叉和被锁的节点保留。必须是 -1 或正整数。默认与效果：默认 -1。交互与改写：限制的是 Mamba 状态不是 KV 页。易错：设成 0，解析会拒绝。 |
| `enable_mamba_cache_stochastic_rounding` | --enable-mamba-cache-stochastic-rounding | `bool` | `False` | 作用：写 FP16 Mamba SSM 缓存时做随机舍入。需要 mamba_ssm_dtype=float16 和 CUDA。triton 后端还要 SM100。默认与效果：默认关闭。交互与改写：舍入轮数看 mamba_cache_philox_rounds。易错：dtype 仍是 bf16 就打开。 |
| `mamba_cache_philox_rounds` | --mamba-cache-philox-rounds | `int` | `0` | 作用：FP16 Mamba 随机舍入的 Philox 轮数。0 对 Triton 表示用 Triton 默认，对 FlashInfer 表示 10 轮。默认与效果：默认 0。交互与改写：只在随机舍入打开时有意义。易错：两个后端对 0 的含义不同。 |
| `mamba_radix_cache_strategy` | --mamba-radix-cache-strategy（S2 可改写） | `str` | `'auto'` | 作用：Mamba 前缀树用哪种策略。auto 会解析成具体策略，并决定要不要额外缓冲。默认与效果：默认 auto。交互与改写：ReplaySSM 要求 no_buffer，而那通常就是默认。易错：手写 extra_buffer 又开 ReplaySSM。 |
| `uses_mamba_radix_cache` | --uses-mamba-radix-cache（无常规 CLI；S2 可改写） | `bool` | `False` | 作用：这个模型是否走混合 Mamba 前缀树。从架构解析，没有 CLI。默认与效果：派生，默认 false 直到模型说是。交互与改写：普通 Transformer 为假，Mamba 字段对其无效果。易错：在 V4.1 上检查这个值并期待为真。 |
| `mamba_track_interval` | --mamba-track-interval | `int` | `256` | 作用：解码过程中每隔多少 token 记一次 Mamba 状态。默认与效果：默认 256。交互与改写：间隔大则状态更稀，回滚或投机更粗。易错：设成 1，状态写入变成每步开销。 |
| `enable_int8_mamba_checkpoint` | --enable-int8-mamba-checkpoint | `bool` | `False` | 作用：把前缀树里的线性注意力状态另存在 int8 检查点池，同样显存大约能多留一倍前缀。默认与效果：默认关闭。交互与改写：槽数默认是活跃 Mamba 池的 2 倍，可用 int8_mamba_ckpt_size 改。易错：打开后仍按 bf16 状态估算容量。 |
| `int8_mamba_ckpt_size` | --int8-mamba-ckpt-size | `Optional[int]` | `None` | 作用：int8 Mamba 检查点槽的个数。默认是活跃池的 2 倍。默认与效果：默认空，用 2 倍。交互与改写：必须先打开 int8 检查点才有池。易错：设得比活跃池还小，缓存前缀很快被挤掉。 |
| `linear_attn_backend` | --linear-attn-backend | `str` | `'triton'` | 作用：线性注意力（GDN/KDA）的默认核。decode 和 prefill 还可以分别覆盖。Helion 只支持 KDA。默认与效果：默认 triton。交互与改写：兼容的 SM100 GDN 预填充可能自动改选 FlashInfer。易错：给 GDN 指定 Helion。 |
| `linear_attn_decode_backend` | --linear-attn-decode-backend | `Optional[str]` | `None` | 作用：只覆盖线性注意力的解码核。不填则用 linear_attn_backend。默认与效果：默认空。交互与改写：校验后端默认跟着解码后端走。易错：解码用 flashinfer、校验却以为仍是 triton。 |
| `linear_attn_prefill_backend` | --linear-attn-prefill-backend | `Optional[str]` | `None` | 作用：只覆盖线性注意力的预填充/扩展核。不填则用总后端；兼容的 SM100 GDN 可能自动选 FlashInfer。默认与效果：默认空。交互与改写：和总后端可以不同。易错：预填充选了解码才支持的核。 |
| `linear_attn_verify_backend` | --linear-attn-verify-backend | `Optional[str]` | `None` | 作用：投机目标校验时的线性注意力核。不填则跟随解码：解码是 flashinfer 则校验也是，否则 triton。KDA 支持 triton、nv_cutedsl 和 flashinfer。默认与效果：默认空。交互与改写：只在投机校验步有区别。易错：没有投机却调这个值，不会影响普通解码。 |
| `enable_linear_replayssm` | --enable-linear-replayssm | `bool` | `False` | 作用：打开 ReplaySSM，只缓冲输出的线性注意力解码核。主要加速 GDN（标量门），批大小至少 64 时大约 1.2 到 1.5 倍。KDA 的每 K 门环更大，ReplaySSM 往往比打包的 KDA 解码更慢，打开前要自己测。需要 Triton 线性注意力解码，或 KDA 用 Helion，并且 mamba 策略是 no_buffer。默认与效果：默认关闭。交互与改写：和 extra buffer 策略冲突。易错：KDA 模型照着 GDN 的数字打开。 |
| `linear_replayssm_cache_len` | --linear-replayssm-cache-len | `int` | `16` | 作用：ReplaySSM 环缓冲长度 L。每 L 个解码步把完整循环状态刷回显存。默认与效果：默认 16。交互与改写：只在 ReplaySSM 打开时使用。易错：L 设成 1，等于每步都刷，带宽收益没了。 |
| `enable_linear_replayssm_spec` | --enable-linear-replayssm-spec | `bool` | `False` | 作用：投机校验用 ReplaySSM：每次提交时折叠，用每个槽的原始输入窗口代替按草稿保存的完整状态。只支持 GDN 或 KDA 混合线性注意力，并且是链状投机（eagle topk 为空或 1）。默认与效果：默认关闭。交互与改写：树状草稿不能用。易错：eagle topk 大于 1 仍打开。 |

### CUDA Graph

命名空间 `exec.graph`，类 `ExecGraph`，`python/sglang/srt/arg_groups/fields/exec_.py`，19 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `cuda_graph_config` | --cuda-graph-config | `Optional[CudaGraphConfig]` | `None` | 作用：按 JSON 指定解码和预填充两相的 CUDA 图。每相后端可以是 full、breakable、disabled，还可以写 max_bs。JSON 优先于分相的便捷旗标和旧旗标。默认与效果：默认空，用分相旗标和平台默认。交互与改写：DeepEP normal、DeepEP v2 的预填充、DWDP 仍会把对应相关掉，JSON 写了 full 也会被改。易错：只改 JSON，不看日志里的覆盖。 |
| `cuda_graph_backend_decode` | --cuda-graph-backend-decode | `Optional[Literal['full', 'breakable', 'disabled']]` | `None` | 作用：解码相图后端，会折进 cuda_graph_config 的 decode.backend。默认与效果：默认空。交互与改写：disable_decode_cuda_graph 等价于这里写 disabled。易错：JSON 和这个旗标同时写，以 JSON 为准。 |
| `cuda_graph_backend_prefill` | --cuda-graph-backend-prefill | `Optional[Literal['full', 'breakable', 'disabled']]` | `None` | 作用：预填充相图后端，折进 prefill.backend。默认与效果：默认空。交互与改写：V4.1 配 TRT-LLM 时必须是 disabled 或 breakable。易错：写成 full，V4.1 trtllm 路径在解析阶段失败。 |
| `cuda_graph_max_bs_decode` | --cuda-graph-max-bs-decode | `Optional[int]` | `None` | 作用：解码图捕获的最大批大小。默认与效果：默认空，用内置阶梯。交互与改写：实际捕获列表还可以被 cuda_graph_bs_decode 显式覆盖。易错：设得比 max_running_requests 大很多，捕获时间很长，多出来的图用不上。 |
| `cuda_graph_max_bs_prefill` | --cuda-graph-max-bs-prefill | `Optional[int]` | `None` | 作用：预填充图捕获的最大批大小。默认与效果：默认空。交互与改写：长序列还受 max_seq_len_prefill 限制。易错：只加大 batch，序列一长仍回落到 eager。 |
| `cuda_graph_max_seq_len_prefill` | --cuda-graph-max-seq-len-prefill | `Optional[int]` | `None` | 作用：预填充图重放能接受的最长序列，更长的批走 eager。折进配置的 max_seq_len。默认与效果：默认空。交互与改写：V4.1 测试里会把它和 chunk 一起设。易错：设得比 chunked_prefill_size 小，分块后的批也进不了图。 |
| `cuda_graph_bs_decode` | --cuda-graph-bs-decode | `Optional[List[int]]` | `None` | 作用：显式列出解码图要捕获的批大小，而不是用最大批生成阶梯。默认与效果：默认空，用阶梯。交互与改写：列表里没有的批大小会回退或填充到最近的图。易错：列表漏了常见批大小，那些批每次都填充，延迟不稳。 |
| `cuda_graph_bs_prefill` | --cuda-graph-bs-prefill | `Optional[List[int]]` | `None` | 作用：显式列出预填充图的批大小。默认与效果：默认空。交互与改写：和 max_bs_prefill 二选一更清晰，同时写时以显式列表为准。易错：列表含 0 或负数。 |
| `cuda_graph_prefill_max_context` | --cuda-graph-prefill-max-context（--context-bucket） | `Optional[int]` | `None` | 作用：预填充图能覆盖的上下文上限。声明处没有展开 help。默认与效果：默认空，由收尾函数 finalize_cuda_graph_prefill_max_context 填。交互与改写：它在 S2 末尾才定稿。易错：启动前读这个字段，还是空的。 |
| `disable_prefill_cuda_graph` | --disable-prefill-cuda-graph | `bool` | `False` | 作用：关掉预填充相 CUDA 图，等价于 backend=disabled。默认与效果：默认不关。交互与改写：和 cuda_graph_backend_prefill 写不同值时，以解析后的配置为准。易错：和 JSON 里的 full 矛盾，不知道谁赢。 |
| `disable_decode_cuda_graph` | --disable-decode-cuda-graph | `bool` | `False` | 作用：关掉解码相 CUDA 图。默认与效果：默认不关。交互与改写：解码图是吞吐的主要来源，关掉后退回 eager。易错：只为了调试预填充，却把解码图也关了。 |
| `disable_cuda_graph` | --disable-cuda-graph（无常规 CLI） | `bool` | `False` | 作用：旧的总开关，CLI 上是废弃旗标，指向两相都 disabled。字段本身 no_cli。默认与效果：默认 false。交互与改写：新配置应写分相后端。易错：继续写 --disable-cuda-graph，和分相 JSON 混在一起。 |
| `disable_cuda_graph_padding` | --disable-cuda-graph-padding | `bool` | `False` | 作用：需要填充到图的批大小时不用图；不需要填充时仍用图。默认与效果：默认关闭，允许填充后重放。交互与改写：和 torch.compile 不能同时开，否则每种批大小都会单独编译，启动会拖很久，validation_hook 会拒绝。易错：两个一起开。 |
| `enable_profile_cuda_graph` | --enable-profile-cuda-graph | `bool` | `False` | 作用：对 CUDA 图捕获做性能剖析。默认与效果：默认关闭。交互与改写：只影响启动时的捕获，不改变重放数值。易错：长期开着，每次重启都多一段剖析开销。 |
| `enable_cudagraph_gc` | --enable-cudagraph-gc | `bool` | `False` | 作用：捕获 CUDA 图期间是否允许垃圾回收。默认关闭，捕获时冻住 GC，捕获更快。默认与效果：默认不在捕获中 GC。交互与改写：和 gc_threshold 是平时的 GC，不是这一段。易错：捕获时内存紧张仍冻住 GC，捕获本身 OOM。 |
| `debug_cuda_graph` | --debug-cuda-graph | `bool` | `False` | 作用：用可打断的 CUDA 图做调试。打开后插入图打断，每个算子仍走捕获/重放路径但实际是 eager。用来查捕获和重放不一致。默认与效果：默认关闭。交互与改写：不是生产后端。易错：线上打开，吞吐掉回 eager。 |
| `enable_torch_compile` | --enable-torch-compile | `bool` | `False` | 作用：用 torch.compile 优化模型。实验功能。默认与效果：默认关闭。交互与改写：和 disable_cuda_graph_padding 互斥。最大批看 torch_compile_max_bs。易错：和 CUDA 图同时开，两套编译叠加，启动极慢。 |
| `enable_torch_compile_debug_mode` | --enable-torch-compile-debug-mode | `bool` | `False` | 作用：torch.compile 的调试模式。默认与效果：默认关闭。交互与改写：只在 compile 打开时有意义。易错：单独打开它，compile 本身是关的。 |
| `torch_compile_max_bs` | --torch-compile-max-bs | `int` | `32` | 作用：torch.compile 覆盖的最大批大小。默认与效果：默认 32。交互与改写：超过的批不走编译后的图。易错：服务 max_running_requests 是 256，这里仍是 32，大批没有编译收益。 |

### 集合通信

命名空间 `exec.comm`，类 `ExecComm`，`python/sglang/srt/arg_groups/fields/exec_.py`，16 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `enable_layerwise_nvtx_marker` | --enable-layerwise-nvtx-marker | `bool` | `False` | 作用：给模型每一层打 NVTX 标记，方便 Nsight 看分层耗时。默认与效果：默认关闭。交互与改写：只影响剖析，不改变数值。易错：不在剖析时开着，每层都有标记开销。 |
| `enable_nccl_nvls` | --enable-nccl-nvls | `bool` | `False` | 作用：在可用时对预填充较重的请求使用 NCCL NVLS。默认与效果：默认关闭。交互与改写：需要硬件支持 NVLS，否则没有这条路径。易错：多机没有 NVLink 域却打开。 |
| `enable_symm_mem` | --enable-symm-mem（S2 可改写） | `bool` | `False` | 作用：为快速集合通信打开 NCCL 对称内存。默认与效果：默认关闭。交互与改写：显存预分配会在 handle_gpu_memory_settings 里读这个开关，所以它必须先定下来。易错：打开后 KV 预算变小，却没把 mem_fraction_static 调低。 |
| `disable_custom_all_reduce` | --disable-custom-all-reduce（S2 可改写） | `bool` | `False` | 作用：不用自定义 all-reduce，退回 NCCL。默认与效果：默认不禁用，能用自定义核就用。交互与改写：某些世界大小或机器上自定义核会错，这时应打开本开关。易错：名字是否定式，排错时搞反。 |
| `enable_mscclpp` | --enable-mscclpp | `bool` | `False` | 作用：用 MSCCL++ 跑调过的 AllReduce 和 AllGather，不行再退回 NCCL。默认与效果：默认关闭。交互与改写：需要对应库和拓扑。易错：没装库就打开。 |
| `enable_torch_symm_mem` | --enable-torch-symm-mem | `bool` | `False` | 作用：all-reduce 用 torch 对称内存，否则退回 NCCL。只支持 SM90 及以上。SM90 的世界大小可以是 4、6、8；SM100 是 6、8。默认与效果：默认关闭。交互与改写：tp 不是这些数时不能用。易错：tp=16 仍打开。 |
| `enable_scattered_sconv` | --enable-scattered-sconv | `bool` | `False` | 作用：Inkling 专用。把注意力/MLP 输出的 all-reduce 换成按隐藏维的 reduce-scatter，在 [T, H/P] 分片上做通道短卷积，残差相加前再 all-gather。卷积缓存按 TP 切开，通信量不变。默认与效果：默认关闭。交互与改写：非 Inkling 模型没有这条卷积。易错：在 DeepSeek 上打开。 |
| `pre_warm_nccl` | --pre-warm-nccl | `bool` | `False` | 作用：启动时预热 NCCL/RCCL，降低第一次请求的 P99 TTFT。AMD/HIP 默认开，NVIDIA 默认关。默认与效果：默认 false，平台钩子可能改成开。交互与改写：多机预热会在建组后多一次通信。易错：在 NVIDIA 上看日志以为没预热就是故障。 |
| `boundary_reduction` | --boundary-reduction（S2 可改写） | `Literal['auto', 'ar', 'rs', 'rsv', 'rs+rsv']` | `'auto'` | 作用：FFN 边界怎么规约。ar 是先 all-reduce 再重分发 token；rs 和 rsv 分别是定长和变长 reduce-scatter；rs+rsv 两者都允许，优先 RSv。不支持的路径退回 ar。只对带阶段边界的解码器 FFN 有意义，auto 用模型默认，其他模型忽略。注意力和 MoE 的集合通信以及 all-reduce 融合不受影响。默认与效果：默认 auto。交互与改写：投机还有自己的 speculative_boundary_reduction。易错：改了它以为 MoE all-to-all 也会变。 |
| `enable_quant_communications` | --enable-quant-communications | `Optional[bool]` | `False` | 作用：把张量并行通信量化成 INT8。支持范围有限。默认与效果：默认关闭。交互与改写：会改变通信数值。易错：和确定性或水印同时开，误差来源分不清。 |
| `enable_flashinfer_allreduce_fusion` | --enable-flashinfer-allreduce-fusion（无常规 CLI） | `bool` | `False` | 作用：旧旗标，请改用 flashinfer_allreduce_fusion_backend=auto。声明处 help 为空，CLI 上标了废弃。默认与效果：默认关闭。交互与改写：新字段不设置时，受支持的 MiMo 用 cutedsl，其他受支持模型用 auto。易错：继续写旧旗标。 |
| `enforce_disable_flashinfer_allreduce_fusion` | --enforce-disable-flashinfer-allreduce-fusion | `bool` | `False` | 作用：强制关掉 FlashInfer all-reduce 融合，即使用户或模型想开。默认与效果：默认不强制关。交互与改写：排错融合数值时用。易错：和 backend=auto 同时写，强制关赢。 |
| `flashinfer_allreduce_fusion_backend` | --flashinfer-allreduce-fusion-backend（S2 可改写） | `Optional[Literal['auto', 'trtllm', 'mnnvl', 'cutedsl']]` | `None` | 作用：打开 FlashInfer all-reduce 融合并选后端。需要 SM90 或 SM10x。不填时，受支持的 MiMo 用 cutedsl，其他受支持模型用 auto。auto 在 Blackwell（含多机）选 mnnvl，SM90 单机选 trtllm。trtllm 只单机。mnnvl 在 SM90 单机或 SM100/SM103 上经 MNNVL。cutedsl 是 Blackwell 的 bf16 MNNVL，还能把 MoE 收尾和共享专家加融合进通信。融合的是 all-reduce 加残差加 RMSNorm。默认与效果：默认空。交互与改写：和 Aiter 融合是两条路径，不要在 ROCm 上选这个。易错：多机写 trtllm。 |
| `enable_aiter_allreduce_fusion` | --enable-aiter-allreduce-fusion（S2 可改写） | `bool` | `False` | 作用：打开 Aiter 的 AllReduce 融合，给 ROCm。默认与效果：默认关闭。交互与改写：预填充 CP 和 moe_dp 路径会断言不能开这个融合。模型钩子在部分 HIP 模型上会自动打开。易错：和 prefill CP 一起开。 |
| `disable_aiter_allreduce_fusion_in_prefill` | --disable-aiter-allreduce-fusion-in-prefill（S2 可改写） | `bool` | `False` | 作用：预填充类批（EXTEND、MIXED、SPLIT_PREFILL）不用 Aiter 融合，解码仍用。只有在总开关打开时才有意义。默认与效果：默认不单独关预填充。交互与改写：和「解码相关闭」可以分别设。易错：总开关是关的，这个旗标没有任何效果。 |
| `disable_aiter_allreduce_fusion_in_decode` | --disable-aiter-allreduce-fusion-in-decode（S2 可改写） | `bool` | `False` | 作用：解码类批（DECODE、TARGET_VERIFY、草稿扩展、IDLE）不用 Aiter 融合，预填充仍用。默认与效果：默认不单独关解码。交互与改写：同上，依赖总开关。易错：只关了解码，预填充的数值差异还在。 |

### MoE 与 DeepEP / A2A

命名空间 `exec.moe`，类 `ExecMoe`，`python/sglang/srt/arg_groups/fields/exec_.py`，41 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `is_ep_joiner` | --is-ep-joiner（fn=sglang.srt.arg_groups.model_override_base.ep_joiner_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：这个进程是不是弹性 EP 的加入者（扩容或故障恢复）。派生量。默认与效果：无 CLI。交互与改写：由启动角色决定，不是 ep_size。易错：在静态 16 卡部署里读它并期待为真。 |
| `is_ep_scale_joiner` | --is-ep-scale-joiner（fn=sglang.srt.arg_groups.model_override_base.ep_scale_joiner_of；无常规 CLI） | `Derived` | `（发布时计算）` | 作用：是不是扩容加入者，而不是故障恢复。派生量。默认与效果：无 CLI。交互与改写：扩容加入者即使组宽暂时为 1，attn_dp_enabled 仍可能为真。易错：用 attn_dp_size==1 判断「没有 attention DP」。 |
| `enable_fused_moe_sum_all_reduce` | --enable-fused-moe-sum-all-reduce | `bool` | `False` | 作用：把 MoE 的 triton 计算和求和 all-reduce 融合。默认与效果：默认关闭。交互与改写：只对 triton MoE 有这条融合。易错：runner 已是 flashinfer 仍打开。 |
| `moe_a2a_backend` | --moe-a2a-backend（S2 可改写） | `Literal['none', 'deepep', 'mooncake', 'nixl', 'mori', 'ascend_fuseep', 'flashinfer', 'megamoe', 'deepep_v2', 'ascend_tp', 'pplx', 'flashinfer_megamoe']` | `'none'` | 作用：MoE token 的 all-to-all 实现。none 表示不搬家，专家按 TP 切。deepep、deepep_v2、mori、megamoe、flashinfer、nixl、mooncake 等属于铺满 TP 的后端，解析会把 ep_size 改成 tp_size。默认与效果：默认 none。交互与改写：DSPARK 加 attention DP 时只能 none、megamoe、mori。deepep normal 会关 CUDA 图。deepep_v2 的 direct 不能跨节点。四机 HIP 的 V4.1 应写 mori。易错：写了 deepep 又开 DSPARK 和 attn_dp。 |
| `enable_w4a4_mxfp4_megamoe` | --enable-w4a4-mxfp4-megamoe | `bool` | `False` | 作用：MegaMoE 走 W4A4 MXFP4，使用 DeepGEMM 的 mxf4xmxf4。要和 moe_a2a_backend=megamoe 一起用。默认与效果：默认关闭。交互与改写：草稿可以单独用 speculative_enable_w4a4_mxfp4_megamoe 选择不同 MMA。易错：a2a 不是 megamoe 就打开。 |
| `deepep_v2_mode` | --deepep-v2-mode | `Literal['direct', 'hybrid']` | `'direct'` | 作用：DeepEP v2 的通信拓扑，启动时固定。direct 是单机 NVLink，hybrid 是多机。布局和分组 GEMM、解码 CUDA 图按推理阶段选，和这个开关无关，也不等于 v1 的 normal/low_latency。默认与效果：默认 direct。交互与改写：nnodes>1 时 direct 会 ValueError。易错：四机仍留着默认 direct。 |
| `moe_runner_backend` | --moe-runner-backend（S2 可改写） | `str` | `'auto'` | 作用：专家 GEMM 的执行器，和 all-to-all 是两件事。auto 按量化改写。若干 flashinfer 和 LoRA runner 会关掉共享专家融合。默认与效果：默认 auto。交互与改写：V4.1 的 MXFP8 稠密快路径不覆盖 HIP。易错：在海光上指定 flashinfer_cutedsl 或 flashinfer_trtllm。 |
| `flashinfer_mxfp4_moe_precision` | --flashinfer-mxfp4-moe-precision | `Literal['default', 'bf16', 'fp8']` | `'default'` | 作用：选择 FlashInfer MXFP4 MoE 的计算精度。SM90 上 fp8 走 Humming 风格的 MXFP4 权重乘 FP8 激活，需要 FlashInfer 至少 0.6.18。默认与效果：声明默认是 default，不强制改成 fp8。交互与改写：只在 flashinfer 的 mxfp4 MoE 路径上有区别。易错：FlashInfer 版本不够却写 fp8。 |
| `deepep_mode` | --deepep-mode | `Literal['auto', 'normal', 'low_latency']` | `'auto'` | 作用：DeepEP v1 的调度模式，auto、normal 或 low_latency。 默认与效果：默认 auto。 交互与改写：normal 关掉两相 CUDA graph。CuteDSL FP4 会强制 low_latency，normal 直接报错。它不控制 deepep_v2。 易错：以为 auto 在 mori 后端也有意义。mori 不读这个字段。 |
| `fuseep_mode` | --fuseep-mode | `Literal[1, 2]` | `2` | 作用：昇腾 FuseEP MoE 的模式。1 执行 dispatch_gmm_combine_decode；2 执行 dispatch_ffn_combine，并且 2 才支持混合部署。默认与效果：默认 2。交互与改写：只在 moe_a2a_backend 为 ascend_fuseep 时被读取。易错：在 CUDA 上改这个数，不会换成昇腾核。 |
| `deepep_dispatcher_output_dtype` | --deepep-dispatcher-output-dtype | `Literal['auto', 'bf16', 'fp8', 'int8', 'nvfp4']` | `'auto'` | 作用：DeepEP dispatcher 输出张量的数据类型。默认与效果：默认 auto，由 dispatcher 按权重精度选。交互与改写：和 deepep_mode 独立，改 dtype 不改变 normal/low_latency。易错：手写一个和专家权重不一致的 dtype，通信后的 GEMM 对不上。 |
| `flashinfer_a2a_dispatch_type` | --flashinfer-a2a-dispatch-type | `Optional[Literal['auto', 'bf16', 'nvfp4', 'mxfp8']]` | `None` | 作用：FlashInfer all-to-all dispatcher 使用的激活数据类型。默认与效果：默认空，跟 FlashInfer 自己的选择。交互与改写：只在 moe_a2a_backend 为 flashinfer 或 flashinfer_megamoe 时生效。MegaMOE 还要求 attn_dp 等于 tp。易错：后端是 mori 时设置它，没有任何效果。 |
| `ep_num_redundant_experts` | --ep-num-redundant-experts | `int` | `0` | 作用：专家并行里额外复制多少个冗余专家，用来给热点专家多一份副本。默认与效果：默认 0，不复制。交互与改写：大于 0 时要有 ep_dispatch_algorithm 决定副本放在哪些 rank。易错：专家总数加冗余后超过实现能登记的槽。 |
| `ep_dispatch_algorithm` | --ep-dispatch-algorithm | `Optional[Literal['static', 'dynamic', 'fake', 'lp']]` | `None` | 作用：冗余专家放到哪些 rank 的算法。默认与效果：默认空，没有冗余分配算法。交互与改写：ep_num_redundant_experts 为 0 时这个算法不会被用到。易错：开了冗余却不指定算法，副本位置保持初始布局。 |
| `init_expert_location` | --init-expert-location | `str` | `'trivial'` | 作用：EP 专家的初始放置。trivial 表示按顺序均分，不做负载感知的初始重排。默认与效果：默认 trivial。交互与改写：EPLB 打开后会在运行中再搬，这个只决定起点。易错：以为 trivial 已经按请求热点摆好。 |
| `enable_eplb` | --enable-eplb | `bool` | `False` | 作用：打开专家并行负载均衡，运行中把专家从忙的 rank 迁到闲的 rank。默认与效果：默认关闭。交互与改写：需要记录专家分布，并受重平衡间隔和利用率阈值约束。和静态 mori 部署可以同时存在，但是另一套运行时逻辑。易错：没开记录器就指望它知道热点。 |
| `eplb_algorithm` | --eplb-algorithm | `str` | `'auto'` | 作用：EPLB 用哪种重平衡算法。auto 按实现选择。默认与效果：默认 auto。交互与改写：只在 enable_eplb 为真时执行。易错：写了算法名但 EPLB 开关是关的。 |
| `eplb_rebalance_num_iterations` | --eplb-rebalance-num-iterations | `int` | `1000` | 作用：每隔多少次迭代自动做一次 EPLB 重平衡。默认与效果：默认 1000。交互与改写：间隔太小会频繁中断前向去搬权重。易错：设成 1，每步都重平衡。 |
| `eplb_rebalance_layers_per_chunk` | --eplb-rebalance-layers-per-chunk | `Optional[int]` | `None` | 作用：每次前向最多重平衡多少层，用来把搬迁拆开，避免一次搬完全部层。默认与效果：默认空，由实现一次处理。交互与改写：层数应能被这个块整除，否则最后一块更小。易错：设成比层数还大，等于没拆。 |
| `eplb_min_rebalancing_utilization_threshold` | --eplb-min-rebalancing-utilization-threshold | `float` | `1.0` | 作用：GPU 平均利用率至少到这个比例才触发 EPLB。范围必须在 0 到 1。默认与效果：默认 1.0，等于几乎只有打满才搬。交互与改写：调低会更积极地搬专家。易错：写成 10，超出 [0,1]，解析失败。 |
| `expert_distribution_recorder_mode` | --expert-distribution-recorder-mode | `Optional[Literal['stat', 'stat_approx', 'per_pass', 'per_token']]` | `None` | 作用：专家分布记录器的模式，决定统计写到哪里、记多细。默认与效果：默认空，不按额外模式记录。交互与改写：HTTP 上的 start/stop/dump expert distribution 依赖记录器已经开。易错：只调 HTTP 不设模式，dump 是空的。 |
| `expert_distribution_recorder_buffer_size` | --expert-distribution-recorder-buffer-size | `Optional[int]` | `None` | 作用：专家分布记录的环形缓冲长度。-1 表示不截断。默认与效果：默认空，用实现自己的长度。交互与改写：缓冲太小，长跑之后早期分布被覆盖。易错：设成 0，等于不保留样本。 |
| `expert_balancedness_report_mode` | --expert-balancedness-report-mode | `Literal['off', 'server_log', 'prometheus', 'both']` | `'off'` | 作用：专家均衡度报到哪。off 不报，server_log 打日志，prometheus 打指标，both 两边都报。默认与效果：默认 off。交互与改写：prometheus 还要 enable_metrics。易错：只开 report 没开 metrics，prometheus 模式没有出口。 |
| `deepep_config` | --deepep-config | `Optional[str]` | `None` | 作用：给本集群调过的 DeepEP 配置，可以是 JSON 字符串或文件路径。默认与效果：默认空，用 DeepEP 内置配置。交互与改写：只在 moe_a2a_backend 为 deepep 时读取。易错：文件路径写成了 JSON 对象，或反过来。 |
| `elastic_ep_backend` | --elastic-ep-backend | `Literal[None, 'mooncake', 'nixl']` | `None` | 作用：弹性 EP 扩缩时用的集合通信后端，支持 mooncake 和 nixl。默认与效果：默认空。交互与改写：静态扩容检查要求 nixl，不是 mori。易错：四机静态 mori 服务再打开弹性 EP。 |
| `enable_elastic_expert_backup` | --enable-elastic-expert-backup | `bool` | `False` | 作用：弹性专家备份，故障或扩容时有专家副本可接。默认与效果：默认关闭。交互与改写：要和弹性 EP 的加入模式一起配。易错：没留备份容量就打开，显存不够。 |
| `mooncake_ib_device` | --mooncake-ib-device | `Optional[str]` | `None` | 作用：Mooncake 传输用的 InfiniBand 设备，可以逗号分隔多块，例如 mlx5_0,mlx5_1。空则在 Mooncake 后端启用时自动探测。默认与效果：默认空。交互与改写：和 PD 的 disaggregation_ib_device 不是同一个字段。易错：两个字段只填了一个，另一条 Mooncake 路径仍在自动选设备。 |
| `enable_waterfill` | --enable-waterfill | `bool` | `False` | 作用：把融合后的共享专家当成一个额外的路由槽，派到当前最空的 EP rank。支持 DeepEP 和 MegaMOE，会隐式打开共享专家融合。和 DeepEP 一起时 deepep_mode 可以是 auto、normal 或 low_latency；生产解码要用 auto 或 low_latency，图才还在。支持 DeepSeek-V3/R1 且 EP 至少 2。默认与效果：默认关闭。交互与改写：若 a2a 不是 deepep 或 megamoe，解析会把后端改成 deepep。和 DSPARK 加 attention DP 的白名单冲突。易错：V4.1 四机 DSPARK 配置打开它。 |
| `ep_join_mode` | --ep-join-mode | `Optional[Literal['scale', 'recover']]` | `None` | 作用：弹性 EP 的加入方式。recover 是故障后回到原槽；scale 是作为超出原组大小的新 rank 加入，并且要求 node_rank 为 1。默认与效果：默认空，不加入。交互与改写：scale 还要 ep 等于 tp、attention DP、dp lm head、nixl、无 CP、无 PP。易错：scale 时 node_rank 写成 0。 |
| `elastic_ep_scale_timeout` | --elastic-ep-scale-timeout | `float` | `600` | 作用：一次弹性 EP 扩容最多等多少秒。默认与效果：默认 600。交互与改写：超时后这次加入失败，已有组继续服务。易错：设得比权重同步还短，扩容总是超时。 |
| `disable_flashinfer_cutlass_moe_fp4_allgather` | --disable-flashinfer-cutlass-moe-fp4-allgather | `bool` | `False` | 作用：FlashInfer CUTLASS MoE 的 FP4 路径上，all-gather 之前不做量化。默认与效果：默认不禁用，也就是仍会先量化再 all-gather。交互与改写：只影响这条 FP4 CUTLASS 路径。易错：把名字理解成关掉整个 all-gather。 |
| `disable_shared_experts_fusion` | --disable-shared-experts-fusion（S2 可改写） | `bool` | `False` | 作用：关掉 DeepSeek V3/R1 内置的共享专家融合。Waterfill 会把共享专家当成额外 MoE 槽，因此 Waterfill 打开时共享专家不会从 MoE 路径上拆开，这个关闭会被改回去。默认与效果：默认不关融合。交互与改写：若干 flashinfer runner 和 LoRA MoE 会在解析时强制把它设为真。易错：和水印式地以为自己关了融合，日志里又被 Waterfill 打开。 |
| `enforce_shared_experts_fusion` | --enforce-shared-experts-fusion | `bool` | `False` | 作用：即使通常会关（例如 DeepEP），也强制融合共享专家。对排除了共享专家量化的 Quark MXFP4 Qwen3.5/Qwen3.8 检查点，加载时还会把该共享专家量化成 MXFP4 以便融合，权重和激活都变成 MXFP4，而独立路径原本是 BF16，用之前要先对精度。和 disable_shared_experts_fusion 互斥。默认与效果：默认不强制。交互与改写：和 disable 同时开，解析会拒绝。易错：不看精度就在 Quark 检查点上强制融合。 |
| `enable_paged_experts` | --enable-paged-experts | `bool` | `False` | 作用：每层 GPU 上只常驻 paged_experts_num_resident 个路由专家，其余从锁页主机内存按需换入。仅单卡。专家必须跑在 triton 或 Marlin runner 上。解码 CUDA 图（full）最多捕获到 K/top_k 个请求；预填充图必须是 breakable 或 disabled。默认与效果：默认关闭。交互与改写：和多卡 EP 不能一起用。易错：tp>1 仍打开。 |
| `paged_experts_num_resident` | --paged-experts-num-resident | `Optional[int]` | `None` | 作用：每层留在 GPU 上的路由专家个数。打开分页专家时必填，并且至少要有路由器的 top-k 那么多。默认与效果：默认空。交互与改写：小于 top-k 时一个 token 的专家放不下。易错：V4.1 的 top-k 是 6，这里写成 4。 |
| `kt_weight_path` | --kt-weight-path | `Optional[str]` | `None` | 作用：ktransformers 的 AMX 核使用的量化专家权重目录，必须是本地文件夹。默认与效果：默认空。交互与改写：只在走 CPU 专家的 ktransformers 路径时读取。易错：写成 Hugging Face id 而不是本地目录。 |
| `kt_method` | --kt-method | `str` | `'AMXINT4'` | 作用：ktransformers 在 CPU 上执行时的量化格式。默认与效果：默认 AMXINT4。交互与改写：要和 kt_weight_path 里的权重格式一致。易错：权重是别的量化，方法仍是 AMXINT4。 |
| `kt_cpuinfer` | --kt-cpuinfer | `Optional[int]` | `None` | 作用：ktransformers 的 CPUInfer 线程数。默认与效果：默认空，由库自己定。交互与改写：和 NUMA 线程池数量是两套线程。易错：设得超过物理核，CPU 专家反而更慢。 |
| `kt_threadpool_count` | --kt-threadpool-count | `int` | `2` | 作用：ktransformers 线程池个数，应和 NUMA 节点一一对应。默认与效果：默认 2。交互与改写：机器只有 1 个 NUMA 时设 2 会跨节点。易错：照抄 2 到单路机器。 |
| `kt_num_gpu_experts` | --kt-num-gpu-experts | `Optional[int]` | `None` | 作用：ktransformers 混合部署里留在 GPU 上的专家数，其余去 CPU。默认与效果：默认空。交互与改写：和 kt_max_deferred_experts_per_token 一起决定每个 token 有多少专家下到 CPU。易错：GPU 专家数大于总专家数。 |
| `kt_max_deferred_experts_per_token` | --kt-max-deferred-experts-per-token | `Optional[int]` | `None` | 作用：每个 token 最多把多少个专家推迟到 CPU。除最后一层外都用这个值，最后一层始终是 0。默认与效果：默认空。交互与改写：只对 ktransformers 的 CPU 专家生效。易错：以为最后一层也会下到 CPU。 |

### Batch overlap

命名空间 `exec.overlap`，类 `ExecOverlap`，`python/sglang/srt/arg_groups/fields/exec_.py`，3 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `enable_two_batch_overlap` | --enable-two-batch-overlap | `bool` | `False` | 作用：两个微批重叠。 默认与效果：默认关。 交互与改写：V4.1 特征检查拒绝。PP 也会关 overlap schedule，但是另一个字段。 易错：从旧的 DeepSeek-V3 启动脚本抄过来。 |
| `enable_single_batch_overlap` | --enable-single-batch-overlap | `bool` | `False` | 作用：在同一个微批里让计算和通信重叠。默认与效果：默认关闭。交互与改写：DeepEP v2 会禁止 single-batch overlap。和 two-batch overlap 不是同一个开关。易错：V4.1 开了 deepep_v2 再开这个。 |
| `tbo_token_distribution_threshold` | --tbo-token-distribution-threshold | `float` | `0.48` | 作用：两批 token 分布差到这个阈值时，在 two-batch overlap 和 two-chunk overlap 之间选择。设 0 表示关掉 two-chunk overlap。默认与效果：默认 0.48。交互与改写：只在 two-batch overlap 打开时有意义。V4.1 禁止 two-batch overlap。易错：没开 TBO 却调这个阈值。 |

### 权重 offload

命名空间 `exec.offload`，类 `ExecOffload`，`python/sglang/srt/arg_groups/fields/exec_.py`，8 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `cpu_offload_gb` | --cpu-offload-gb | `int` | `0` | 作用：为 CPU offload 预留多少 GB 主机内存。默认与效果：默认 0，不预留，等于不做这种 offload。交互与改写：预留会挤占 HiCache 和 Engram host 表的内存。易错：和 Engram host 表同时要几百 GB，主机内存不够。 |
| `offload_group_size` | --offload-group-size | `int` | `-1` | 作用：offload 时每组多少层。-1 表示不用分组或由实现决定整组。默认与效果：默认 -1。交互与改写：和 offload_num_in_group 一起决定一组里实际换出几层。易错：组大小小于组内换出层数。 |
| `offload_num_in_group` | --offload-num-in-group | `int` | `1` | 作用：每一组里真正换出到 CPU 的层数。默认与效果：默认 1。交互与改写：这不是并行宽度。1 表示每组换出 1 层，不是「不切分」。易错：把它理解成 TP=1。 |
| `offload_prefetch_step` | --offload-prefetch-step | `int` | `1` | 作用：offload 时提前几步把下一层权重拉回来。默认与效果：默认按字段声明。交互与改写：步数太小会在用到该层时还没预取完。易错：设成 0，每次用层都同步等待。 |
| `offload_mode` | --offload-mode | `str` | `'cpu'` | 作用：权重换出的具体模式。默认与效果：默认按实现。交互与改写：和 cpu_offload_gb 的预留量要匹配该模式的峰值。易错：模式需要的内存超过预留的 GB。 |
| `ple_offload_embedding` | --ple-offload-embedding（S2 可改写） | `Optional[bool]` | `None` | 作用：把 Qwen4 PLE 的 n-gram 嵌入表换到 CPU 锁页内存。BF16 的 Qwen4-Exp 在 CUDA 上默认开。用 --no-ple-offload-embedding 关掉。默认与效果：默认按模型，Qwen4-Exp BF16 CUDA 会开。交互与改写：V4.1 的 Engram 是另一张表，不由这个开关控制。易错：在 DeepSeek 上打开，没有 PLE 表。 |
| `ple_offload_backend` | --ple-offload-backend | `str` | `'pinned'` | 作用：换出的 Qwen4 PLE 表放哪。pinned 用 CPU 锁页内存；file 在 ple_offload_dir 下映射稀疏文件，gather 核直接读，适合统一内存设备。默认与效果：默认 pinned。交互与改写：file 必须同时给目录。易错：统一内存机器仍用 pinned，占满锁页内存。 |
| `ple_offload_dir` | --ple-offload-dir | `Optional[str]` | `None` | 作用：file 后端存放 PLE 表的目录。默认是缓存目录下的 ple/模型路径，每个检查点一个目录。文件是稀疏的，重启可复用，应放在快速本地盘。默认与效果：默认走缓存目录。交互与改写：只在 backend=file 时使用。易错：目录在网络盘上，gather 直接读会很慢。 |

### 扩散语言模型（dLLM）

命名空间 `exec.dllm`，类 `ExecDllm`，`python/sglang/srt/arg_groups/fields/exec_.py`，3 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `dllm_algorithm` | --dllm-algorithm | `Optional[str]` | `None` | 作用：掩码扩散语言模型的算法，例如 LowConfidence。这是 dLLM，不是图像扩散。默认与效果：默认按字段。交互与改写：要和 dllm_algorithm_config 的 YAML 配套。ForwardMode 会走 DLLM_EXTEND。易错：把 multimodal_gen 的算法名写到这里。 |
| `dllm_algorithm_config` | --dllm-algorithm-config | `Optional[str]` | `None` | 作用：dLLM 算法的 YAML 配置文件。默认与效果：默认空。交互与改写：算法需要参数时，空文件会在初始化 dLLM 时失败。易错：写成 JSON。help 要求 YAML。 |
| `dllm_fdfo` | --dllm-fdfo | `bool` | `True` | 作用：dLLM 使用先完成先出的调度。默认开。--no-dllm-fdfo 退回同步按块调度。默认与效果：默认打开。交互与改写：只影响 dLLM 的调度顺序，不改变 Transformer 的 S9 循环。易错：普通 CausalLM 打开它，没有 dLLM 块可调度。 |

### 确定性推理

命名空间 `exec.deterministic`，类 `ExecDeterministic`，`python/sglang/srt/arg_groups/fields/exec_.py`，2 项。

| 字段 | CLI | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `enable_deterministic_inference` | --enable-deterministic-inference | `bool` | `False` | 作用：用与批大小无关的算子做确定性推理，同一输入和种子应得到同一输出。默认与效果：默认关闭。交互与改写：会限制 attention 和 GEMM 后端，BF16 GEMM 的 auto 会改成 torch。易错：只设随机种子不打开它，跨批仍然不稳定。 |
| `rl_on_policy_target` | --rl-on-policy-target | `Optional[str]` | `None` | 作用：声明 SGLang 要对齐的训练系统，使 on-policy 采样和训练端一致。默认与效果：默认按字段声明。交互与改写：choices 在 rl_on_policy 的选项列表里。和确定性、logprob 返回一起用。易错：训练端是另一个栈，这里却留空，logprob 对不齐。 |

## Diffusion ServerArgs

这些字段属于 `multimodal_gen` 的 dataclass，不进入 LLM 的 `run_resolution_pipeline`。
共 173 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `DISAGG_RESULT_PORT_OFFSETS` | `DisaggServerArgsMixin`:18 | `{RoleType.ENCODER: 1, RoleType.DENOISER: 2, RoleType.DECODER: 3, RoleType.PROMPT_ENCODER: 4, RoleType.IMAGE_ENCODER: 5}` | 作用：扩散分离里各角色结果端口相对主端口的偏移，编码器和下采样器不同。 默认与效果：按角色字典给出固定偏移。 交互与改写：只在 pool 分离模式使用，不进 LLM 的端口分配。 易错：和 HTTP 端口算重。 |
| `model_path` | `ServerArgs`:299 | `（必填）` | 作用：扩散模型权重目录或仓库 id。这是必填项。 默认与效果：没有默认。 交互与改写：子目录、变体、组件路径都是在它之上的覆盖。 易错：路径只有权重没有 model_index.json。 |
| `model_subfolder` | `ServerArgs`:300 | `None` | 作用：权重在仓库里的子目录。 默认与效果：默认空，用仓库根。 交互与改写：和 model_path 拼在一起。 易错：子目录名和仓库布局不一致。 |
| `model_variant` | `ServerArgs`:301 | `None` | 作用：同一模型的变体名，例如 fp16。 默认与效果：默认空。 交互与改写：加载器按变体选文件名。 易错：变体文件不存在。 |
| `model_id` | `ServerArgs`:304 | `None` | 作用：显式指定模型 id，例如 Qwen-Image，用来选 pipeline，而不是靠路径猜。 默认与效果：默认空，从路径推断。 交互与改写：和 pipeline_class_name 同时给时要一致。 易错：id 和权重不是同一个模型。 |
| `served_model_name` | `ServerArgs`:307 | `None` | 作用：扩散服务在 /v1/models 和响应里展示的名字。 默认与效果：默认空，用路径名。 交互与改写：客户端的 model 字段要对上它。 易错：对外名字和磁盘路径不同，客户端仍写路径。 |
| `backend` | `ServerArgs`:310 | `Backend.AUTO` | 作用：用 SGLang 自己的实现还是 diffusers。AUTO 按模型选。 默认与效果：默认 AUTO。 交互与改写：和 LLM 的 attention_backend 不是同一个参数对象。 易错：强制 diffusers，但该模型只有原生实现。 |
| `attention_backend` | `ServerArgs`:313 | `None` | 作用：扩散 DiT 的注意力后端。 默认与效果：默认空，按实现选。 交互与改写：还可以按组件覆盖。 易错：写成 LLM 的 dsv4。 |
| `enable_attention_backend_autotune` | `ServerArgs`:317 | `False` | 作用：预热时在模型自己的张量上给候选注意力后端计时，每层留最快的。只在 SM90 和 SM12x 上测过。 默认与效果：默认关闭。 交互与改写：会拉长启动。 易错：在没测过的架构上打开，选出的后端可能不稳定。 |
| `attention_backend_config` | `ServerArgs`:318 | `None` | 作用：注意力后端的额外配置。 默认与效果：默认空。 交互与改写：和按组件的覆盖表一起生效，组件表优先。 易错：JSON 键和后端不认识的参数。 |
| `component_attention_backends` | `ServerArgs`:319 | `field(default_factory=dict)` | 作用：按组件名指定注意力后端，例如只有 transformer 用某个核。 默认与效果：默认空字典。 交互与改写：不要和全局 attention_backend 写成矛盾还不看谁覆盖谁。 易错：组件名不在 model_index.json 里。 |
| `_requested_component_attention_backends` | `ServerArgs`:322 | `field(default=None, repr=False, compare=False)` | 作用：用户原始请求的组件后端，解析前的副本，避免解析把表改掉后丢失原意。 默认与效果：默认空。 交互与改写：没有 CLI，内部用。 易错：当它是最终生效值。 |
| `component_attention_backend_roles` | `ServerArgs`:328 | `field(default_factory=dict)` | 作用：按组件加角色的后端覆盖，键是 component.role，避免污染按组件名查找的那张表。 默认与效果：默认空字典。 交互与改写：分离角色里同一组件不同角色可以不同核。 易错：把角色键写进普通组件表。 |
| `_requested_component_attention_backend_roles` | `ServerArgs`:331 | `field(default=None, repr=False, compare=False)` | 作用：角色覆盖的原始请求副本。 默认与效果：默认空。 交互与改写：内部字段。 易错：从外部 JSON 填它。 |
| `cache_dit_config` | `ServerArgs`:334 | `None` | 作用：cache-dit 的配置，给 diffusers 路径复用 DiT 计算。 默认与效果：默认空，不启用。 交互与改写：和 CUDA 图可能冲突，要看该 pipeline 是否支持。 易错：配置打开了但后端是不支持缓存的核。 |
| `nccl_port` | `ServerArgs`:339 | `None` | 作用：扩散分布式初始化端口。 默认与效果：默认空，自动选。 交互与改写：和 LLM 的 nccl_port 不是同一个进程的参数。 易错：多机防火墙没放行。 |
| `enable_nccl_nvls` | `ServerArgs`:340 | `False` | 作用：扩散集合通信尝试使用 NCCL NVLS。 默认与效果：默认关闭。 交互与改写：需要 NVLink 域。 易错：没有 NVLink 仍打开。 |
| `trust_remote_code` | `ServerArgs`:343 | `False` | 作用：允许加载仓库里的自定义扩散代码。 默认与效果：默认关闭。 交互与改写：只影响扩散加载，不影响 LLM。 易错：不打开导致 pipeline 类找不到。 |
| `revision` | `ServerArgs`:344 | `None` | 作用：扩散仓库的版本。 默认与效果：默认空，主分支。 交互与改写：和 model_path 一起下载。 易错：revision 不存在。 |
| `num_gpus` | `ServerArgs`:347 | `1` | 作用：扩散使用的 GPU 数。跨节点时它是所有节点的总数，每节点跑 num_gpus 除以 nnodes。 默认与效果：默认 1。 交互与改写：和 tp、sp、dp 的乘积要一致。 易错：写成每节点卡数，又设了 nnodes。 |
| `performance_mode` | `ServerArgs`:348 | `'auto'` | 作用：扩散的性能预设。auto 按模型选。 默认与效果：默认 auto。 交互与改写：可能改写编译、图和 offload。 易错：手写的开关被预设覆盖，却没看日志。 |
| `base_gpu_id` | `ServerArgs`:349 | `0` | 作用：从哪块 GPU 开始数。 默认与效果：默认 0。 交互与改写：和 gpu_ids 同时给时以列表为准。 易错：VISIBLE_DEVICES 已经重排又设了很大的起点。 |
| `gpu_ids` | `ServerArgs`:350 | `None` | 作用：显式列出要使用的 GPU 号。 默认与效果：默认空，从 base 连续数。 交互与改写：长度应等于本节点的卡数。 易错：列表长度和 num_gpus 不一致。 |
| `numa_node` | `ServerArgs`:352 | `None` | 作用：第 i 项是本机第 i 块 GPU 上 worker 的 NUMA 节点。空则自动探测。 默认与效果：默认空。 交互与改写：绑错会跨 NUMA 拷权重。 易错：列表短于 GPU 数。 |
| `nnodes` | `ServerArgs`:356 | `1` | 作用：扩散跨节点的节点数。 默认与效果：默认 1。 交互与改写：num_gpus 是总卡数，不是每节点卡数。 易错：每台机器都写 nnodes=1。 |
| `node_rank` | `ServerArgs`:357 | `0` | 作用：本节点序号。 默认与效果：默认 0。 交互与改写：大于 0 的节点通常不对外服务或只参加集合通信。 易错：都写成 0。 |
| `dist_init_addr` | `ServerArgs`:358 | `None` | 作用：分布式初始化地址 host:port。 默认与效果：默认空。 交互与改写：所有节点必须能连到 rank 0。 易错：写成各自的主机名。 |
| `tp_size` | `ServerArgs`:359 | `None` | 作用：扩散 DiT 的张量并行。空表示按 num_gpus 推断。 默认与效果：默认空。 交互与改写：和 sp、ring、ulysses 一起分卡。 易错：各项乘积超过 num_gpus。 |
| `sp_degree` | `ServerArgs`:360 | `None` | 作用：序列并行度。空表示不切序列或自动。 默认与效果：默认空。 交互与改写：ulysses 和 ring 是两种序列并行实现。 易错：同时把 sp、ulysses、ring 设成互不相容的组合。 |
| `ulysses_degree` | `ServerArgs`:362 | `None` | 作用：Ulysses 序列并行度，用 all-to-all 交换头。 默认与效果：默认空。 交互与改写：和 ring、kv_gather 是不同的序列切法。 易错：头数不能被这个度整除。 |
| `ring_degree` | `ServerArgs`:363 | `None` | 作用：Ring 序列并行度。 默认与效果：默认空。 交互与改写：和 Ulysses 不要无意义叠加。 易错：环的度数和卡数不匹配。 |
| `kv_gather_degree` | `ServerArgs`:367 | `None` | 作用：注意力内部按行切开，用一次 K/V all-gather 交换，而不是 Ulysses 的 a2a 或 ring。 默认与效果：默认空。 交互与改写：和 ulysses、ring 互斥的切法。 易错：三种序列并行都设了大于 1。 |
| `sp_split_auto` | `ServerArgs`:369 | `False` | 作用：序列切分是不是自动分配的。为真时各层可以按调用再回退。 默认与效果：默认 false。 交互与改写：用户显式设了 sp 就不应再当自动。 易错：手写了度数又把这个设成 true，回退逻辑把度数改掉。 |
| `dp_size` | `ServerArgs`:372 | `1` | 作用：扩散数据并行组数，每组一份权重、不同请求。 默认与效果：默认 1。 交互与改写：和 CFG 并行再分每组里的卡。 易错：dp 乘每组卡数超过 num_gpus。 |
| `enable_cfg_parallel` | `ServerArgs`:375 | `None` | 作用：是否把 classifier-free guidance 的条件和无条件分支并行到不同 GPU。空表示按卡数自动。 默认与效果：默认空。 交互与改写：和 cfg_parallel_degree 一起决定每组几张卡。 易错：只有一张卡仍强制开。 |
| `cfg_parallel_degree` | `ServerArgs`:377 | `None` | 作用：每个 CFG 并行组的 GPU 数。空为自动，1 为关闭，大于 1 为打开。 默认与效果：默认空。 交互与改写：必须能整除该数据并行组的卡数。 易错：设成 3 但组内只有 2 张卡。 |
| `encoder_parallel` | `ServerArgs`:382 | `'auto'` | 作用：多 rank 副本里编码器怎么放：auto、fold、dp 或 replicate。 默认与效果：默认 auto。 交互与改写：replicate 会复制编码器权重。 易错：编码器很大仍 replicate。 |
| `hsdp_replicate_dim` | `ServerArgs`:384 | `1` | 作用：HSDP 的复制维。 默认与效果：默认 1，不额外复制。 交互与改写：和 shard 维一起切 FSDP 风格的推理。 易错：复制维和卡数乘积对不上。 |
| `hsdp_shard_dim` | `ServerArgs`:385 | `None` | 作用：HSDP 的切分维。 默认与效果：默认空。 交互与改写：use_fsdp_inference 打开时才有意义。 易错：没开 FSDP 却设了切分。 |
| `dist_timeout` | `ServerArgs`:386 | `3600` | 作用：扩散分布式初始化超时，默认更长，因为权重大。 默认与效果：默认 3600 秒。 交互与改写：和 scheduler_rpc_timeout 是两段超时。 易错：超时短于权重加载。 |
| `scheduler_rpc_timeout` | `ServerArgs`:387 | `None` | 作用：调度器 RPC 超时。 默认与效果：默认空，用内置。 交互与改写：批处理很慢时要加大。 易错：设成几秒，长视频步数还没跑完就超时。 |
| `pipeline_config` | `ServerArgs`:389 | `field(default_factory=PipelineConfig, repr=False)` | 作用：pipeline 的结构化配置对象。 默认与效果：默认构造一个 PipelineConfig。 交互与改写：命令行通常不直接填这个对象，而用其他旗标组装。 易错：当它是普通字符串。 |
| `pipeline_class_name` | `ServerArgs`:392 | `None` | 作用：强制使用的 pipeline 类名。 默认与效果：默认空，按模型 id 选。 交互与改写：类必须是已注册的 EntryClass。 易错：类名拼错。 |
| `lora_path` | `ServerArgs`:398 | `None` | 作用：扩散 LoRA 权重路径。放在服务参数里，而不是 pipeline 配置，避免 pipeline 变复杂。 默认与效果：默认空。 交互与改写：和 lora_scale、nickname 一起用。 易错：路径是 LLM 的 LoRA，形状对不上 DiT。 |
| `lora_nickname` | `ServerArgs`:399 | `'default'` | 作用：这套扩散 LoRA 的名字。 默认与效果：默认 default。 交互与改写：多套 LoRA 靠名字区分。 易错：两套都叫 default。 |
| `lora_scale` | `ServerArgs`:400 | `1.0` | 作用：扩散 LoRA 的强度。 默认与效果：默认 1.0。 交互与改写：和 alpha 一起决定实际缩放。 易错：设成很大，图像崩掉。 |
| `lora_alpha` | `ServerArgs`:401 | `None` | 作用：LoRA 的 alpha。空表示用权重里的值。 默认与效果：默认空。 交互与改写：实际比例常常是 scale * alpha / rank。 易错：alpha 和训练时不一致。 |
| `lora_merge_mode` | `ServerArgs`:402 | `'auto'` | 作用：LoRA 是合并进基座还是动态加。auto 按实现。 默认与效果：默认 auto。 交互与改写：合并后不能再按请求换适配器。 易错：需要热切换却选了合并。 |
| `lora_weight_name` | `ServerArgs`:403 | `None` | 作用：适配器文件里要用的权重名。 默认与效果：默认空，用默认文件名。 交互与改写：一个目录里有多个 safetensors 时要指定。 易错：文件名不存在。 |
| `component_paths` | `ServerArgs`:406 | `field(default_factory=dict)` | 作用：按 model_index.json 的组件名覆盖组件路径。 默认与效果：默认空字典。 交互与改写：值是路径，不是单个权重文件。 易错：键名和 model_index 不一致。 |
| `component_weights_paths` | `ServerArgs`:408 | `field(default_factory=dict)` | 作用：精确指定某个组件的权重文件，但保留该组件原来的配置。 默认与效果：默认空字典。 交互与改写：和 component_paths 一个换整个组件一个只换文件。 易错：文件和组件配置的形状不一致。 |
| `component_direct_gpu_weight_loading` | `ServerArgs`:411 | `field(default_factory=dict)` | 作用：让某一个组件走加载器自己的直接上 GPU 路径。 默认与效果：默认空字典。 交互与改写：全局 direct_gpu_weight_loading 是所有组件的总开关。 易错：组件不支持这条路径仍打开。 |
| `component_quantizations` | `ServerArgs`:414 | `field(default_factory=dict)` | 作用：给某个组件显式指定量化方法。自描述的检查点仍可自动识别。 默认与效果：默认空字典。 交互与改写：和全局 quantization 冲突时组件项优先。 易错：方法该组件的加载器不认识。 |
| `component_precisions` | `ServerArgs`:417 | `field(default_factory=dict)` | 作用：给声明了精度接口的组件指定加载和运行精度。 默认与效果：默认空字典。 交互与改写：只对加载器声明支持的组件有效。 易错：给 VAE 写了 DiT 才支持的精度。 |
| `component_quantization_ignored_layers` | `ServerArgs`:419 | `field(default_factory=dict)` | 作用：在线量化时按组件跳过的层名模式。 默认与效果：默认空字典。 交互与改写：和全局 quantization_ignored_layers 是两级。 易错：正则写错，该跳的层被量化。 |
| `load_diffusion_decoder` | `ServerArgs`:423 | `False` | 作用：LTX-2.5 的解码器很大，只有请求时才加载。 默认与效果：默认 false，不加载。 交互与改写：要出最终视频时必须开。 易错：没开就请求视频，解码阶段缺模块。 |
| `transformer_weights_path` | `ServerArgs`:426 | `None` | 作用：预量化的 transformer 权重，safetensors 文件、目录或 GGUF。 默认与效果：默认空，用模型自己的权重。 交互与改写：和 quantization 同时给时不要重复量化。 易错：GGUF 和 safetensors 路径写反。 |
| `minimax_h3_adaln_cache_path` | `ServerArgs`:428 | `None` | 作用：MiniMax H3 推理用的预计算 AdaLN 输出路径。 默认与效果：默认空。 交互与改写：和 minimax_h3_adaln_online 二选一。 易错：缓存是别的检查点算的。 |
| `minimax_h3_adaln_online` | `ServerArgs`:430 | `False` | 作用：每个请求从检查点现算 AdaLN，不需要旁路缓存文件。 默认与效果：默认 false。 交互与改写：打开后可以不提供 cache path。 易错：两个都空，既不在线也不读缓存。 |
| `minimax_h3_adaln_plan_width` | `ServerArgs`:433 | `4` | 作用：重建 AdaLN 时 slab 按多宽的时间步计划来分配。上限见 MINIMAX_H3_ADALN_MAX_PLAN_WIDTH。 默认与效果：默认 4。 交互与改写：计划比这个宽会装不下。 易错：设得超过实现上限。 |
| `minimax_h3_adaln_host_cache_gb` | `ServerArgs`:438 | `8.0` | 作用：AdaLN 计划的锁页主机缓存，十进制 GB。0 表示不要这块缓存。GPU 上被赶出的计划放这里。 默认与效果：默认 8GB。 交互与改写：和别的 host 缓存抢内存。 易错：设成 0 又没有足够 GPU 常驻。 |
| `quantization` | `ServerArgs`:441 | `None` | 作用：显式量化方法，例如 mxfp8、fp8、modelslim。设置后 transformer 按这个方法在线量化。 默认与效果：默认空，跟检查点。 交互与改写：组件级覆盖优先。 易错：检查点已经是 FP8 又在线量化一次。 |
| `quantization_ignored_layers` | `ServerArgs`:443 | `None` | 作用：在线量化时跳过的层名模式。 默认与效果：默认空，全部量化。 交互与改写：敏感层被量化会坏图。 易错：模式太宽，几乎没量化。 |
| `lora_target_modules` | `ServerArgs`:447 | `None` | 作用：限制 LoRA 打在哪些层，例如只要 q_proj。默认会打 q、k、v、o。 默认与效果：默认空，用默认的 qkvo。 交互与改写：模块名必须在 DiT 里存在。 易错：写成 LLM 的层名。 |
| `component_residency` | `ServerArgs`:451 | `None` | 作用：组件或组件组的常驻方式，决定在 GPU 还是 CPU。 默认与效果：默认空。 交互与改写：和各个 *_cpu_offload 旗标是同一类决策的不同入口。 易错：同一组件两种入口设成冲突。 |
| `cpu_offload_components` | `ServerArgs`:453 | `None` | 作用：按 model_index 的组件键，或旧的组件组名，把这些组件放 CPU。 默认与效果：默认空。 交互与改写：精确键优先于旧组名。 易错：组名拼写用了新键的格式。 |
| `dit_cpu_offload` | `ServerArgs`:454 | `None` | 作用：整个 DiT 是否放 CPU。 默认与效果：默认空，按性能模式。 交互与改写：和逐层 offload 不要一起当成两套都全开。 易错：DiT 全在 CPU，步数时间不可用。 |
| `direct_gpu_weight_loading` | `ServerArgs`:456 | `False` | 作用：用更高的加载峰值换更快的普通 DiT 启动，权重直接进 GPU。 默认与效果：默认 false。 交互与改写：组件还可以单独 opt-in。 易错：显存紧的机器打开，加载时 OOM。 |
| `dit_layerwise_offload` | `ServerArgs`:458 | `None` | 作用：为真时选择 DiT 的逐层换出组。 默认与效果：默认空。 交互与改写：常驻层数、预取、策略、寿命四个旋钮配合。 易错：只开逐层不设预取，每层都同步等。 |
| `layerwise_offload_components` | `ServerArgs`:459 | `None` | 作用：除了 DiT 以外还要对哪些组件做逐层换出。 默认与效果：默认空。 交互与改写：那些组件的加载器必须支持逐层。 易错：对 VAE 打开逐层，但 VAE 没有层接口。 |
| `dit_offload_prefetch_size` | `ServerArgs`:460 | `0.0` | 作用：DiT 逐层换出时提前拉回多少。 默认与效果：默认 0，不预取。 交互与改写：0 意味着用到该层才加载。 易错：预取大于常驻，显存峰值反而更高。 |
| `dit_layerwise_resident_layers` | `ServerArgs`:462 | `0.0` | 作用：有多少 DiT 层留在 GPU 上。 默认与效果：默认 0，不额外常驻。 交互与改写：哪些层由 residency_policy 决定。 易错：常驻层数大于总层数。 |
| `dit_layerwise_residency_policy` | `ServerArgs`:464 | `RESIDENCY_POLICY_LEADING` | 作用：常驻的是开头几层，还是在堆叠上均匀散开。 默认与效果：默认开头。 交互与改写：均匀散开更适合前后层都重的网络。 易错：策略名写错。 |
| `dit_layerwise_residency_lifetime` | `ServerArgs`:467 | `RESIDENCY_LIFETIME_FORWARD` | 作用：常驻层保持多久：一次请求的 DiT 前向期间，或整个服务进程活着的时候。 默认与效果：默认前向期间。 交互与改写：进程级常驻更占显存、更少来回拷。 易错：服务级常驻把显存占满，批处理上不去。 |
| `layerwise_prefetch_size` | `ServerArgs`:470 | `field(default_factory=dict)` | 作用：按组件覆盖预取大小，有条目的组件以条目为准。 默认与效果：默认空字典。 交互与改写：覆盖全局 dit_offload_prefetch_size。 易错：组件名写错，覆盖没生效。 |
| `layerwise_resident_layers` | `ServerArgs`:471 | `field(default_factory=dict)` | 作用：按组件覆盖常驻层数。 默认与效果：默认空字典。 交互与改写：和全局 DiT 旋钮是同一含义的分组件版。 易错：把 DiT 的层数抄到层数更少的组件。 |
| `layerwise_residency_policy` | `ServerArgs`:474 | `field(default_factory=dict)` | 作用：按组件覆盖常驻策略。 默认与效果：默认空字典。 交互与改写：取值要和全局策略同一套名字。 易错：写了全局不认识的策略。 |
| `layerwise_residency_lifetime` | `ServerArgs`:477 | `field(default_factory=dict)` | 作用：按组件覆盖常驻寿命。 默认与效果：默认空字典。 交互与改写：同上。 易错：一个组件进程级、一个请求级，显存峰值不好估。 |
| `offload_during_compile` | `ServerArgs`:480 | `True` | 作用：编译期间也允许 offload，降低编译峰值。 默认与效果：默认 true。 交互与改写：关掉后编译更快但更吃显存。 易错：显存紧时关掉它，编译 OOM。 |
| `text_encoder_cpu_offload` | `ServerArgs`:481 | `None` | 作用：文本编码器放 CPU。 默认与效果：默认空，按模式。 交互与改写：编码很频繁时会变成瓶颈。 易错：和 encoder_parallel=replicate 叠加，CPU 上复制多份。 |
| `image_encoder_cpu_offload` | `ServerArgs`:482 | `None` | 作用：图像编码器放 CPU。 默认与效果：默认空。 交互与改写：图生视频时每张图都要编码。 易错：放到 CPU 后首帧延迟不可接受。 |
| `vae_cpu_offload` | `ServerArgs`:483 | `False` | 作用：VAE 放 CPU。 默认与效果：默认 false。 交互与改写：解码步会把潜变量拷回 CPU。 易错：高分辨率视频打开后解码极慢。 |
| `use_fsdp_inference` | `ServerArgs`:484 | `None` | 作用：推理时用 FSDP 式切分。 默认与效果：默认空。 交互与改写：和 hsdp 的两个维一起用。 易错：和张量并行重复切同一维。 |
| `pin_cpu_memory` | `ServerArgs`:485 | `True` | 作用：offload 的 CPU 缓冲是否锁页。 默认与效果：默认 true。 交互与改写：锁页能让 GPU 直接 DMA，但占不可换出的内存。 易错：主机内存紧仍锁页，其他进程被挤掉。 |
| `ltx2_two_stage_device_mode` | `ServerArgs`:486 | `None` | 作用：LTX-2 两阶段分别放在什么设备上。 默认与效果：默认空，按实现。 交互与改写：和 load_diffusion_decoder 一起决定第二阶段在不在。 易错：第二阶段指定了没有的设备。 |
| `_explicit_arg_names` | `ServerArgs`:487 | `field(default_factory=set, repr=False)` | 作用：用户在命令行上显式写过的参数名集合，用来区分默认值和显式值。 默认与效果：默认空集合。 交互与改写：内部字段。 易错：当它是业务配置。 |
| `_required_resident_components` | `ServerArgs`:488 | `field(default_factory=set, init=False, repr=False)` | 作用：必须留在 GPU 上的组件集合，由 pipeline 填写。 默认与效果：默认空。 交互与改写：offload 不能把它们换走。 易错：强制 offload 这些组件。 |
| `_fsdp_disabled_components` | `ServerArgs`:491 | `field(default_factory=set, init=False, repr=False)` | 作用：不允许 FSDP 的组件。 默认与效果：默认空。 交互与改写：由加载器根据能力填写。 易错：手写进配置文件。 |
| `_component_layerwise_capabilities` | `ServerArgs`:494 | `field(default_factory=dict, init=False, repr=False)` | 作用：每个组件支不支持逐层 offload。 默认与效果：默认空。 交互与改写：内部探测结果。 易错：对不支持的组件打开逐层。 |
| `comfyui_mode` | `ServerArgs`:499 | `False` | 作用：按 ComfyUI 插件的方式跑，接口和生命周期不同。 默认与效果：默认 false。 交互与改写：和普通 HTTP 服务的假设不完全一样。 易错：当它只是换端口。 |
| `enable_torch_compile` | `ServerArgs`:502 | `False` | 作用：用 torch.compile 编译扩散模块。 默认与效果：默认 false。 交互与改写：和可打断 CUDA 图可以分别开。 易错：编译缓存写在只读目录。 |
| `regional_compile` | `ServerArgs`:503 | `False` | 作用：按区域编译而不是整网一次编译。 默认与效果：默认 false。 交互与改写：通常和 torch.compile 一起。 易错：没开 compile 只开区域。 |
| `allow_cudnn_tf32` | `ServerArgs`:514 | `True` | 作用：允许 cuDNN 用 TF32。可打断 CUDA 图把 DiT 前向切成多段，段边界在注意力。 默认与效果：默认 true。 交互与改写：关掉更慢、数值更接近 FP32。 易错：要比特对齐仍留着 true。 |
| `allow_bf16_reduced_precision_reduction` | `ServerArgs`:515 | `True` | 作用：BF16 归约允许降精度。 默认与效果：默认 true。 交互与改写：和确定性数值目标冲突时要关。 易错：关了之后某些核没有实现。 |
| `enable_breakable_cuda_graph` | `ServerArgs`:519 | `None` | 作用：把 DiT 前向捕获成可打断的 CUDA 图，在注意力处切开。图按分辨率捕获，预热过的分辨率才能重放。 默认与效果：默认空，按模型和性能模式。 交互与改写：未预热的分辨率回退 eager。 易错：服务分辨率和预热分辨率不同。 |
| `bcg_text_buckets` | `ServerArgs`:525 | `None` | 作用：可打断图里文本或提示序列的填充档。条件输入会填到这些档位。 默认与效果：默认空，用内置档。 交互与改写：提示比最大档还长就进不了图。 易错：档位不是递增的。 |
| `enable_layerwise_nvtx_marker` | `ServerArgs`:528 | `False` | 作用：给扩散的层打 NVTX，方便 Nsight。 默认与效果：默认 false。 交互与改写：只用于剖析。 易错：生产开着。 |
| `warmup_mode` | `ServerArgs`:537 | `None` | 作用：预热模式。off 表示不预热。取值在 WARMUP_MODES 里。 默认与效果：默认按实现。 交互与改写：可打断图依赖预热过的分辨率。 易错：off 之后第一张图很慢，或图根本没捕获。 |
| `warmup_resolutions` | `ServerArgs`:539 | `None` | 作用：预热哪些分辨率。 默认与效果：默认空，用代表性分辨率。 交互与改写：服务时的分辨率最好落在这里。 易错：只预热了 512，请求是 1024。 |
| `warmup_num_frames` | `ServerArgs`:540 | `None` | 作用：预热用的帧数。 默认与效果：默认空。 交互与改写：视频模型的图和帧数有关。 易错：预热 1 帧，请求是长视频。 |
| `warmup_steps` | `ServerArgs`:541 | `1` | 作用：预热跑多少步。 默认与效果：默认 1。 交互与改写：太少可能没覆盖到后几步的形状。 易错：设得和真实步数一样大，启动极慢。 |
| `warmup_sampling_params` | `ServerArgs`:545 | `None` | 作用：用 JSON 覆盖合成预热请求的形状。 默认与效果：默认空。 交互与改写：要和真实请求的分辨率、帧数接近。 易错：JSON 字段名和采样参数不一致。 |
| `disable_autocast` | `ServerArgs`:547 | `None` | 作用：关掉自动混合精度。 默认与效果：默认空。 交互与改写：关掉后可能更慢或 OOM。 易错：量化模型仍关掉 autocast，类型不匹配。 |
| `nunchaku_config` | `ServerArgs`:550 | `field(default_factory=NunchakuSVDQuantArgs, repr=False)` | 作用：Nunchaku SVDQuant 的配置对象。 默认与效果：默认构造一套。 交互与改写：不用 Nunchaku 时保持默认即可。 易错：配置和权重的 SVDQuant 版本不一致。 |
| `kv_cache_quant_config` | `ServerArgs`:556 | `field(default_factory=QVGKVQuantArgs, repr=False)` | 作用：视频生成的 KV 量化参数，默认关闭，对应 LLM 的 kv cache dtype 思路。 默认与效果：默认关闭。 交互与改写：只影响扩散里带 KV 的模块。 易错：当成 LLM 的 kv_cache_dtype 来写。 |
| `master_port` | `ServerArgs`:561 | `30005` | 作用：分布式推理的 master 端口。 默认与效果：默认 30005。 交互与改写：不要和 HTTP 端口相同。 易错：多实例都用 30005。 |
| `host` | `ServerArgs`:564 | `'127.0.0.1'` | 作用：扩散 HTTP 监听地址。 默认与效果：默认 127.0.0.1。 交互与改写：对外服务要写成 0.0.0.0。 易错：和 LLM 的 host 设在同一端口。 |
| `port` | `ServerArgs`:565 | `30000` | 作用：扩散 HTTP 端口。 默认与效果：默认 30000。 交互与改写：和 LLM 同时跑要错开。 易错：两个服务都绑 30000。 |
| `http_graceful_shutdown_timeout` | `ServerArgs`:566 | `None` | 作用：HTTP 优雅退出最多等多少秒。 默认与效果：默认空，用服务器默认。 交互与改写：超时后强制断开还在跑的步。 易错：设得比一步去噪还短。 |
| `enable_metrics` | `ServerArgs`:567 | `False` | 作用：打开扩散服务的指标。 默认与效果：默认 false。 交互与改写：和 LLM 的指标端口不是同一个进程。 易错：采集器仍去 LLM 的端口。 |
| `webui` | `ServerArgs`:570 | `False` | 作用：是否起一个简单网页。 默认与效果：默认 false。 交互与改写：端口是 webui_port，要先确认没被占用。 易错：和 API 端口混成一个。 |
| `webui_port` | `ServerArgs`:571 | `12312` | 作用：网页端口。 默认与效果：默认 12312。 交互与改写：只在 webui 打开时监听。 易错：端口被占，strict_ports 为真时启动失败。 |
| `scheduler_port` | `ServerArgs`:573 | `5555` | 作用：扩散内部调度器端口。 默认与效果：默认 5555。 交互与改写：多副本时用 scheduler_ports 列表。 易错：和 HTTP 端口冲突。 |
| `scheduler_ports` | `ServerArgs`:575 | `None` | 作用：每个数据并行副本一个入口端口。空表示还没分配。 默认与效果：默认空。 交互与改写：由启动过程填好后再服务。 易错：手写的长度和 dp 不一致。 |
| `batching_mode` | `ServerArgs`:576 | `'dynamic'` | 作用：请求怎么拼批。dynamic 表示按时间和大小凑批。 默认与效果：默认 dynamic。 交互与改写：和 batching_max_size、delay 一起用。 易错：模式名写错。 |
| `batching_max_size` | `ServerArgs`:577 | `1` | 作用：一批最多几个扩散请求。 默认与效果：默认 1，等于不把多个请求拼一批。 交互与改写：视频模型一批多个很吃显存。 易错：设得很大，显存爆。 |
| `batching_delay_ms` | `ServerArgs`:578 | `0.0` | 作用：凑批最多等多少毫秒。 默认与效果：默认 0，不等。 交互与改写：0 加上 max_size 1 就是来一个算一个。 易错：等得太久，交互式请求变慢。 |
| `batching_config` | `ServerArgs`:579 | `None` | 作用：凑批的完整配置，覆盖上面几个简单字段。 默认与效果：默认空。 交互与改写：给了它就以它为准。 易错：和简单字段互相矛盾。 |
| `enable_batching_metrics` | `ServerArgs`:580 | `False` | 作用：记录凑批的指标。 默认与效果：默认 false。 交互与改写：要先有指标出口。 易错：没开 enable_metrics。 |
| `async_output_save` | `ServerArgs`:581 | `False` | 作用：生成结果异步写盘，不堵住下一步。 默认与效果：默认 false。 交互与改写：进程退出时可能还没写完。 易错：马上读输出文件，文件还不完整。 |
| `sleep_on_idle` | `ServerArgs`:582 | `False` | 作用：空闲时降低 CPU 占用。 默认与效果：默认 false。 交互与改写：下一条请求的启动会稍慢。 易错：在线服务为了省 CPU 打开。 |
| `strict_ports` | `ServerArgs`:585 | `False` | 作用：请求的端口被占用就失败，而不是自动换一个。 默认与效果：默认 false，会自动换。 交互与改写：编排系统需要固定端口时应打开。 易错：打开后端口冲突，整个池起不来。 |
| `output_path` | `ServerArgs`:587 | `'outputs/'` | 作用：生成文件的输出目录。 默认与效果：默认 outputs/。 交互与改写：进程要有写权限。 易错：相对路径在不同工作目录下指向别处。 |
| `input_save_path` | `ServerArgs`:588 | `'inputs/uploads'` | 作用：上传输入的保存目录。 默认与效果：默认 inputs/uploads。 交互与改写：和输出目录分开。 易错：目录不可写，上传失败。 |
| `prompt_file_path` | `ServerArgs`:591 | `None` | 作用：批处理时提示词文件的路径。 默认与效果：默认空，不用文件。 交互与改写：一行一个提示或按该文件格式。 易错：编码不是 UTF-8。 |
| `model_paths` | `ServerArgs`:594 | `field(default_factory=dict)` | 作用：各组件路径，用于正确释放。 默认与效果：默认空字典。 交互与改写：由加载过程填。 易错：手改后和真实加载的路径不一致，释放错对象。 |
| `model_loaded` | `ServerArgs`:595 | `field(default_factory=lambda: {'transformer': True, 'vae': True, 'video_vae': True, 'audio_vae': True, 'video_dit': True, 'audio_dit': True, 'dual_tower_bridge': True})` | 作用：哪些组件已经加载，例如 transformer。 默认与效果：默认有一份初始字典。 交互与改写：内部状态。 易错：当它是用户配置。 |
| `boundary_ratio` | `ServerArgs`:608 | `None` | 作用：Wan2.2 一类 MoE 扩散的边界比例。 默认与效果：默认空。 交互与改写：只对该系列模型有意义。 易错：在别的 DiT 上设置。 |
| `disagg_role` | `ServerArgs`:611 | `RoleType.MONOLITHIC` | 作用：扩散池分离时本进程的角色：整体、编码、去噪或解码。 默认与效果：默认整体，不分离。 交互与改写：分离要成套启动。 易错：所有进程都是去噪，没有解码。 |
| `disagg_timeout` | `ServerArgs`:612 | `3600` | 作用：分离调用的总超时秒数。 默认与效果：默认 3600。 交互与改写：长视频要大于一步时间乘步数。 易错：默认一小时仍不够，或设得太短。 |
| `disagg_downstream_wait_timeout` | `ServerArgs`:613 | `1800` | 作用：等下游角色接活的超时。 默认与效果：默认 1800。 交互与改写：下游没起来就会等到超时。 易错：超时后请求丢了但上游还占着槽。 |
| `disagg_dispatch_policy` | `ServerArgs`:614 | `'round_robin'` | 作用：分离池把活分给下游的策略，默认轮询。 默认与效果：默认 round_robin。 交互与改写：和槽位上限一起决定会不会把一台打满。 易错：策略名写错。 |
| `disagg_mode` | `ServerArgs`:615 | `False` | 作用：是否启用扩散池分离。 默认与效果：默认 false。 交互与改写：为真时才读角色和各个 URL。 易错：只写了 URL 没开模式。 |
| `disagg_instance_id` | `ServerArgs`:616 | `0` | 作用：这个分离实例的编号。 默认与效果：默认 0。 交互与改写：多实例必须不同。 易错：都写成 0。 |
| `disagg_max_slots_per_instance` | `ServerArgs`:617 | `8` | 作用：每个实例最多同时接几个活。 默认与效果：默认 8。 交互与改写：视频很占显存时要调小。 易错：槽位大于显存能撑的并发。 |
| `disagg_transfer_redundancy` | `ServerArgs`:618 | `1.25` | 作用：分离传输缓冲相对数据量的冗余系数。 默认与效果：默认 1.25。 交互与改写：太小会在传输时缓冲不够。 易错：设成小于 1。 |
| `disagg_role_device` | `ServerArgs`:619 | `'auto'` | 作用：这个角色用什么设备。auto 按角色选。 默认与效果：默认 auto。 交互与改写：编码器可以在 CPU，去噪通常在 GPU。 易错：去噪角色写成 cpu。 |
| `disagg_transfer_backend` | `ServerArgs`:620 | `'auto'` | 作用：分离角色之间的传输后端。 默认与效果：默认 auto。 交互与改写：和 LLM 的 PD 后端不是同一个字段。 易错：写成 deepep。 |
| `disagg_transfer_pool_size` | `ServerArgs`:621 | `256 * 1024 * 1024` | 作用：传输池字节数。 默认与效果：默认 256MiB。 交互与改写：高分辨率要加大。 易错：池太小，传输阻塞。 |
| `disagg_transfer_pin_memory` | `ServerArgs`:622 | `'auto'` | 作用：传输缓冲是否锁页。auto 按后端。 默认与效果：默认 auto。 交互与改写：锁页占主机不可换出内存。 易错：主机紧仍强制锁页。 |
| `disagg_p2p_hostname` | `ServerArgs`:623 | `'127.0.0.1'` | 作用：点对点传输时对外公布的主机名。 默认与效果：默认 127.0.0.1。 交互与改写：跨机器必须改成可达地址。 易错：留着 127.0.0.1，别的机器连不回来。 |
| `disagg_ib_device` | `ServerArgs`:624 | `None` | 作用：分离传输的 InfiniBand 设备。 默认与效果：默认空，自动选。 交互与改写：和 LLM 的同名旗标不是一个进程。 易错：设备名带了端口后缀而后端不接受。 |
| `disagg_server_addr` | `ServerArgs`:625 | `None` | 作用：分离池控制面的地址。 默认与效果：默认空。 交互与改写：worker 靠它注册。 易错：地址不可达。 |
| `encoder_urls` | `ServerArgs`:626 | `None` | 作用：编码器角色的 URL 列表。 默认与效果：默认空。 交互与改写：分离模式才用。 易错：列表是空的，编码请求没人接。 |
| `prompt_encoder_urls` | `ServerArgs`:627 | `None` | 作用：提示编码器的 URL，和图像编码器分开时用。 默认与效果：默认空。 交互与改写：不分开时可以留空，走 encoder_urls。 易错：两边都空。 |
| `image_encoder_urls` | `ServerArgs`:628 | `None` | 作用：图像编码器 URL。 默认与效果：默认空。 交互与改写：图生视频必须有人编码图像。 易错：写成了去噪器地址。 |
| `denoiser_urls` | `ServerArgs`:629 | `None` | 作用：去噪器 URL。 默认与效果：默认空。 交互与改写：这是扩散的主计算。 易错：卡数和 denoiser_tp 不一致。 |
| `decoder_urls` | `ServerArgs`:630 | `None` | 作用：解码器 URL，把潜变量变成像素或视频。 默认与效果：默认空。 交互与改写：和 load_diffusion_decoder 是本地加载，这里是远程角色。 易错：两边都没配解码。 |
| `encoder_tp` | `ServerArgs`:631 | `None` | 作用：编码器角色的张量并行。 默认与效果：默认空，按该角色的卡数。 交互与改写：和主进程的 tp 无关。 易错：设得大于编码器那台机器的卡数。 |
| `denoiser_tp` | `ServerArgs`:632 | `None` | 作用：去噪器的张量并行。 默认与效果：默认空。 交互与改写：和 denoiser_sp 一起分卡。 易错：乘积超过去噪器的卡。 |
| `denoiser_sp` | `ServerArgs`:633 | `None` | 作用：去噪器序列并行度。 默认与效果：默认空。 交互与改写：ulysses 和 ring 是更具体的两种。 易错：三个都设了。 |
| `denoiser_ulysses` | `ServerArgs`:634 | `None` | 作用：去噪器的 Ulysses 度。 默认与效果：默认空。 交互与改写：头数要能被它整除。 易错：和 ring 同时大于 1。 |
| `denoiser_ring` | `ServerArgs`:635 | `None` | 作用：去噪器的 ring 度。 默认与效果：默认空。 交互与改写：同上。 易错：度数和卡数不匹配。 |
| `decoder_sp` | `ServerArgs`:636 | `None` | 作用：解码器的序列并行。 默认与效果：默认空。 交互与改写：解码器通常比 DiT 小，不必照抄去噪器的度数。 易错：照抄了一个解码器吃不下的度数。 |
| `pool_work_endpoint` | `ServerArgs`:637 | `None` | 作用：池模式下发任务的端点。 默认与效果：默认空。 交互与改写：和结果端点成对。 易错：只写了工作端点。 |
| `pool_result_endpoint` | `ServerArgs`:638 | `None` | 作用：池模式回收结果的端点。 默认与效果：默认空。 交互与改写：端口偏移见 DISAGG_RESULT_PORT_OFFSETS。 易错：结果端点和角色偏移算重。 |
| `log_level` | `ServerArgs`:641 | `'info'` | 作用：扩散进程的日志级别。 默认与效果：默认 info。 交互与改写：和 LLM 进程各自独立。 易错：debug 打满磁盘。 |
| `log_requests` | `ServerArgs`:642 | `False` | 作用：是否记录扩散请求。 默认与效果：默认 false。 交互与改写：级别由 log_requests_level 决定。 易错：打开级别 3，提示词全落盘。 |
| `log_requests_level` | `ServerArgs`:643 | `2` | 作用：0 到 3，越大约详细，3 是全部输入输出。 默认与效果：默认 2。 交互与改写：总开关没开则没有输出。 易错：没开总开关却调级别。 |
| `log_requests_format` | `ServerArgs`:644 | `'text'` | 作用：请求日志是 text 还是 json。 默认与效果：默认 text。 交互与改写：采集器要和格式一致。 易错：按 JSON 解析 text。 |
| `log_requests_target` | `ServerArgs`:645 | `None` | 作用：日志写到 stdout 或目录，可以多个。 默认与效果：默认空。 交互与改写：目录要可写。 易错：路径不存在。 |
| `uvicorn_access_log_exclude_prefixes` | `ServerArgs`:646 | `field(default_factory=list)` | 作用：这些路径前缀不打访问日志。 默认与效果：默认空列表。 交互与改写：健康检查应该排除。 易错：前缀没写斜杠，匹配不上。 |
| `enable_cache_report` | `ServerArgs`:647 | `False` | 作用：响应里报告扩散缓存命中。 默认与效果：默认 false。 交互与改写：和条件缓存是两件事。 易错：没开条件缓存，报告全是 0。 |
| `disable_conditioning_cache` | `ServerArgs`:648 | `False` | 作用：关掉文本或图像条件的缓存。 默认与效果：默认 false，允许缓存。 交互与改写：重复提示会重新编码。 易错：名字是否定式，设 true 是关掉缓存。 |
| `conditioning_cache_max_size_mb` | `ServerArgs`:649 | `512.0` | 作用：条件缓存最多多少 MiB。 默认与效果：默认 512。 交互与改写：设 0 等于没有容量。 易错：提示很长，缓存装不下仍以为会命中。 |
| `enable_trace` | `ServerArgs`:652 | `False` | 作用：打开扩散进程的追踪。 默认与效果：默认 false。 交互与改写：端点默认 localhost:4317。 易错：端点没有采集器。 |
| `otlp_traces_endpoint` | `ServerArgs`:653 | `'localhost:4317'` | 作用：OTLP 追踪地址。 默认与效果：默认 localhost:4317。 交互与改写：只在 enable_trace 时导出。 易错：写成了 http URL，协议不匹配。 |
| `otlp_service_name` | `ServerArgs`:654 | `None` | 作用：追踪里的服务名。 默认与效果：默认空，用实现默认。 交互与改写：多实例要不同名字。 易错：都叫一个名字。 |
| `srt_encoder_url` | `ServerArgs`:657 | `None` | 作用：用 SGLang LLM 服务做扩散的编码阶段时，那个服务的地址。 默认与效果：默认空，不用 LLM 编码。 交互与改写：连接和读取超时分开设。 易错：地址是扩散自己的 HTTP。 |
| `srt_encoder_connect_timeout` | `ServerArgs`:658 | `3.05` | 作用：连接 LLM 编码器的超时秒数。 默认与效果：默认 3.05。 交互与改写：连不上会快速失败。 易错：设得太大，故障时一直等。 |
| `srt_encoder_timeout` | `ServerArgs`:659 | `100` | 作用：等 LLM 编码器返回的超时秒数。 默认与效果：默认 100。 交互与改写：长提示编码可能超过它。 易错：设成和连接超时一样短。 |
| `pe_server_url` | `ServerArgs`:662 | `None` | 作用：提示增强模型所在的 SGLang 服务地址。 默认与效果：默认空，不做提示增强。 交互与改写：和 prompt_enhancer_config 一起用。 易错：增强服务没起，每条请求都失败。 |
| `prompt_enhancer_config` | `ServerArgs`:663 | `None` | 作用：提示增强的配置：用哪个模型、怎么改写提示。 默认与效果：默认空。 交互与改写：没有 pe_server_url 时配置无法执行。 易错：改写后的提示超过文本编码器长度。 |
