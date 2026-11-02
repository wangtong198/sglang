# 05 Server 选项大全

本文覆盖 v0.5.23（commit `297e54fd105e974cdf401e5b4d20253a22c16f90`）两套服务参数：

1. **LLM Runtime**：`sglang serve` / `python -m sglang.launch_server` 使用的 `ServerArgs`。字段定义在 `python/sglang/srt/arg_groups/fields/*.py`，由 `server_args.py` 的 `collect_input_fields` 组装，`add_cli_args_from_dataclass` 注册 CLI。下表由 AST 从这些文件抽出，**不手抄、不省略字段**。
2. **Diffusion Runtime**：`python/sglang/multimodal_gen/runtime/server_args/server_args.py` 的 dataclass `ServerArgs`（含 `DisaggServerArgsMixin`）。

## 怎么读这张表

| 列 | 含义 |
| --- | --- |
| CLI | 由字段名生成的 `--kebab-case`。括号里是别名、`无 CLI` 或「解析阶段可改写」。 |
| 默认 | 源码里的默认值。`（必填）` 表示 Struct 没有默认值。`（发布时计算）` 是 `Derived`，不是启动参数。 |
| 含义 | 字段上 `Arg.help` 或 `Derived.doc` 的原文。约束的权威实现在 `arg_groups/*_hook.py` 与 `validation_hook.py`，见 02、04。 |

LLM 字段合计 **584**（含派生字段与 `no_cli` 字段）。只统计带 CLI 的输入字段：**527**。

### 手工注册、不在 Struct 字段上的旗标

定义于 `python/sglang/srt/server_args.py` `ServerArgs.add_cli_args`：

| CLI | 作用 |
| --- | --- |
| `--config` | 从 YAML 合并 CLI。`prepare_server_args` 在 `parse_args` 前调用 `ConfigArgumentMerger`。 |
| `--sampling-backend` | 动态 choices：`SAMPLING_BACKEND_CHOICES`，环境变量打开 token oracle 时追加 `token_oracle`。默认仍是字段 `sampling_backend`。 |
| `--reasoning-parser` | choices 为 `auto` + `REASONING_PARSER_NAMES`（`parser/reasoning_parser_names.py`）。 |
| `--tool-call-parser` | choices 为 `auto` + `TOOL_CALL_PARSER_NAMES`。 |
| `--kv-canary-real-data` | choices 来自 `RealKvHashMode`。 |
| `--disable-cuda-graph` | 废弃。转到 `--cuda-graph-backend-{decode,prefill}=disabled`。 |
| `--enable-dp-attention` | 废弃。`--dp-size N --enable-dp-attention` 会被改写成 `attn_dp_size=N, dp_size=1`。计划 2026-12-31 后删除。 |
| `--monokernel-backend` | 废弃。用 `--enable-amd-monokernel`。 |
| `--enable-flashinfer-allreduce-fusion` | 废弃。用 `--flashinfer-allreduce-fusion-backend=auto`。 |

解析入口：`prepare_server_args`（`server_args.py`）只做 argparse；`ServerArgs.resolve_once` → `run_resolution_pipeline`（`arg_groups/pipeline.py`）才会改写默认值、做 assert。因此表中的默认值是**声明默认**，不是某个具体模型跑起来之后的最终值。DeepSeek-V4.1 会在 `model_overrides/deepseek_v4.py` 把 `attention_backend` 改成 `dsv4`、`page_size` 改成 256。

## LLM ServerArgs

### 模型与加载

命名空间 `model`，声明类 `Model`，文件 `python/sglang/srt/arg_groups/fields/model.py`。共 45 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `is_startup_weight_load_overlap` | --is-startup-weight-load-overlap (fn=sglang.srt.arg_groups.model_override_base.startup_weight_load_overlap_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Whether weight loading overlaps startup. |
| `shared_experts` | --shared-experts (fn=sglang.srt.configs.shared_experts.default_shared_experts_config; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Shared-expert layout of the model being constructed or executed. |
| `model_path` | --model-path (--model) | `str` | `（必填）` | — | The path of the model weights. This can be a local folder or a Hugging Face repo ID. |
| `load_format` | --load-format | `str` | `'auto'` | auto, pt, safetensors, instanttensor, npcache, dummy, sharded_state, presharded, gguf, expert_pack, bitsandbytes, mistral, layered, flash_rl, remote, remote_instance, fastsafetensors, private, runai_streamer | The format of the model weights to load. "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available. "pt" will load the weights in the pytorch bin format. "safetensors" will load the weights in the safetensors format. "instanttensor" will load Safetensors weights using InstantTensor's high-performance distributed loader. "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading. "dummy" will initialize the weights with random values, which is mainly for profiling."gguf" will load the weights in the gguf format. "expert_pack" is experimental and loads only the validated DeepSeek-V4-Flash-0731 MXFP4 or text-only Kimi-K3 Q2_K GGUF model with routed experts stored in an SSD expert pack. "bitsandbytes" will load the weights using bitsandbytes quantization."layered" loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller."presharded" performs a normal first-time load (with quantization), then dumps a per-rank/per-tensor sharded checkpoint with content deduplication into <model_path>/presharded/<parallelism+quant subfolder>/. Subsequent runs with the same parallelism+quantization config load directly from this presharded checkpoint and skip re-quantization. The dump directory must be on a shared filesystem across all ranks/nodes. Optional model_loader_extra_config roots: presharded_path (target) and draft_presharded_path (speculative draft); each replaces <model_path>/presharded and still gets a config subfolder appended. Use a writable path when model_path is read-only (e.g. HF cache mounts). |
| `model_loader_extra_config` | --model-loader-extra-config | `str` | `'{}'` | — | Extra config for model loader. This will be passed to the model loader corresponding to the chosen load_format. For load_format=presharded, JSON may include presharded_path (target cache root), draft_presharded_path (draft cache root), max_file_bytes, hash_num_threads, and verify_on_load. |
| `trust_remote_code` | --trust-remote-code | `bool` | `False` | — | Whether or not to allow for custom models defined on the Hub in their own modeling files. |
| `context_length` | --context-length | `Optional[int]` | `None` | — | — |
| `is_embedding` | --is-embedding | `bool` | `False` | — | Whether to use a CausalLM as an embedding model. |
| `revision` | --revision | `Optional[str]` | `None` | — | The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. |
| `model_impl` | --model-impl | `str` | `'auto'` | — | Which implementation of the model to use. * "auto" will try to use the SGLang implementation if it exists and fall back to the Transformers implementation if no SGLang implementation is available. * "sglang" will use the SGLang model implementation. * "transformers" will use the Transformers model * "mindspore" will use the MindSpore model implementation. |
| `model_config_parser` | --model-config-parser | `str` | `'auto'` | — | Which model-config parser to use. "auto" picks "mistral" via the is_mistral_model name heuristic, else "hf" (AutoConfig over config.json). Plugins can register additional parsers via @register_model_config_parser. |
| `json_model_override_args` | --json-model-override-args | `str` | `'{}'` | — | A dictionary in JSON string format used to override default model configurations. |
| `dtype` | --dtype (可被解析阶段改写) | `str` | `'auto'` | auto, half, float16, bfloat16, float, float32 | Data type for model weights and activations. * "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. * "half" for FP16. Recommended for AWQ quantization. * "float16" is the same as "half". * "bfloat16" for a balance between precision and range. * "float" is shorthand for FP32 precision. * "float32" for FP32 precision. |
| `quantization` | --quantization (可被解析阶段改写) | `Optional[str]` | `None` | awq, fp8, mxfp8, gptq, gptq_marlin, awq_marlin, bitsandbytes, gguf, modelopt, modelopt_fp8, modelopt_fp4, nvfp4_online, modelopt_mixed, petit_nvfp4, w8a8_int8, w8a8_fp8, moe_wna16, w4afp8, mxfp4, auto-round, auto-round-int8, compressed-tensors, modelslim, mxfp_w4a8, quark, quark_int4fp8_moe, quark_mxfp4, mlx_q4, mlx_q8, unquant, humming | The quantization method. |
| `quantization_param_path` | --quantization-param-path | `Optional[str]` | `None` | — | Path to the JSON file containing the KV cache scaling factors. This should generally be supplied, when KV cache dtype is FP8. Otherwise, KV cache scaling factors default to 1.0, which may cause accuracy issues. |
| `kv_cache_dtype` | --kv-cache-dtype (可被解析阶段改写) | `str` | `'auto'` | auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16, bfloat16, nvfp4, fp4_mx_block16 | Data type for kv cache storage. "auto" will use model data type. "bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and "fp8_e4m3" are supported for CUDA 11.8+. "mxfp8" is supported by the FA4 backend. "nvfp4" selects the NVFP4 FP4 E2M1 KV cache recipe; "fp4_mx_block16" selects the MX-style block-size-16 FP4 E2M1 KV cache recipe. Both require CUDA 12.8+ and PyTorch 2.8.0+ |
| `qsa_indexer_dtype` | --qsa-indexer-dtype | `str` | `'auto'` | auto, bfloat16, fp8_e4m3 | Storage dtype of the compressed QSA indexer cache and index query (Qwen4-Exp). "fp8_e4m3" stores them as plain e4m3 and scores blocks with fp8 GEMMs (CUDA SM90+). The main KV cache is unaffected. |
| `modelopt_quant` | --modelopt-quant | `Optional[Union[str, Dict]]` | `None` | — | The ModelOpt quantization configuration. Supported values: 'fp8', 'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. This requires the NVIDIA Model Optimizer library to be installed: pip install nvidia-modelopt |
| `modelopt_checkpoint_restore_path` | --modelopt-checkpoint-restore-path | `Optional[str]` | `None` | — | Path to restore a previously saved ModelOpt quantized checkpoint. If provided, the quantization process will be skipped and the model will be loaded from this checkpoint. |
| `modelopt_checkpoint_save_path` | --modelopt-checkpoint-save-path | `Optional[str]` | `None` | — | Path to save the ModelOpt quantized checkpoint after quantization. This allows reusing the quantized model in future runs. |
| `modelopt_export_path` | --modelopt-export-path | `Optional[str]` | `None` | — | Path to export the quantized model in HuggingFace format after ModelOpt quantization. The exported model can then be used directly with SGLang for inference. If not provided, the model will not be exported. |
| `quantize_and_serve` | --quantize-and-serve | `bool` | `False` | — | Quantize the model with ModelOpt and immediately serve it without exporting. This is useful for development and prototyping. For production, it's recommended to use separate quantization and deployment steps. |
| `rl_quant_profile` | --rl-quant-profile | `Optional[str]` | `None` | — | Path to the FlashRL quantization profile. Required when using --load-format flash_rl. |
| `startup_weight_load_mode` | --startup-weight-load-mode | `Literal['serial', 'overlap']` | `'serial'` | — | Control startup weight loading relative to CUDA graph capture. 'serial' preserves the existing startup order; 'overlap' stages checkpoint files while CUDA graphs are captured and commits the real weights afterward. |
| `custom_weight_loader` | --custom-weight-loader | `Optional[List[str]]` | `None` | — | The custom dataloader which used to update the model. Should be set with a valid import path, such as my_package.weight_load_func |
| `weight_loader_disable_mmap` | --weight-loader-disable-mmap | `bool` | `False` | — | Disable mmap while loading weight using safetensors. |
| `weight_loader_prefetch_checkpoints` | --weight-loader-prefetch-checkpoints | `bool` | `False` | — | Prefetch checkpoint files into OS page cache before loading. Each rank prefetches a fraction of the shards, reducing total network I/O on shared filesystems (NFS/Lustre) from N*checkpoint to 1*checkpoint. Recommended for models on network storage. When enabled, multi-threaded safetensors loading is disabled by default to avoid I/O oversubscription with the prefetch threads; set enable_multithread_load=true in --model-loader-extra-config to keep multi-threaded loading (e.g. on local NVMe where prefetch is a no-op). |
| `weight_loader_prefetch_num_threads` | --weight-loader-prefetch-num-threads | `int` | `4` | — | Number of threads per rank for checkpoint prefetching (default: 4). |
| `weight_loader_drop_cache_after_load` | --weight-loader-drop-cache-after-load | `bool` | `False` | — | Call posix_fadvise(DONTNEED) on each safetensors shard after loading it. |
| `remote_instance_weight_loader_seed_instance_ip` | --remote-instance-weight-loader-seed-instance-ip | `Optional[str]` | `None` | — | The ip of the seed instance for loading weights from remote instance. |
| `remote_instance_weight_loader_seed_instance_service_port` | --remote-instance-weight-loader-seed-instance-service-port | `Optional[int]` | `None` | — | The service port of the seed instance for loading weights from remote instance. |
| `remote_instance_weight_loader_send_weights_group_ports` | --remote-instance-weight-loader-send-weights-group-ports | `Optional[List[int]]` | `None` | — | The communication group ports for loading weights from remote instance. |
| `remote_instance_weight_loader_backend` | --remote-instance-weight-loader-backend | `Literal['transfer_engine', 'nccl', 'modelexpress']` | `'nccl'` | — | The backend for loading weights from remote instance. Can be 'transfer_engine', 'nccl', or 'modelexpress'. Default is 'nccl'. |
| `remote_instance_weight_loader_start_seed_via_transfer_engine` | --remote-instance-weight-loader-start-seed-via-transfer-engine | `bool` | `False` | — | Start seed server via transfer engine backend for remote instance weight loader. |
| `engine_info_bootstrap_port` | --engine-info-bootstrap-port | `int` | `6789` | — | Port for the engine info bootstrap server. Default is 6789. Must be set explicitly when running multiple instances on the same node. |
| `modelexpress_config` | --modelexpress-config | `Optional[str]` | `None` | — | JSON config for ModelExpress P2P weight loading. Keys: "url" (optional gRPC host:port override), "transport" ("nixl" or "transfer_engine"). Example: '{"url": "localhost:8001", "transport": "nixl"}' |
| `download_dir` | --download-dir | `Optional[str]` | `None` | — | Model download directory for huggingface. |
| `model_checksum` | --model-checksum | `Optional[str]` | `None` | — | Model file integrity verification. If provided without value, uses model-path as HF repo ID. Otherwise, provide checksums JSON file path or HuggingFace repo ID. |
| `delete_ckpt_after_loading` | --delete-ckpt-after-loading | `bool` | `False` | — | Delete the model checkpoint after loading the model. |
| `decrypted_config_file` | --decrypted-config-file | `Optional[str]` | `None` | — | The path of the decrypted config file. |
| `decrypted_draft_config_file` | --decrypted-draft-config-file | `Optional[str]` | `None` | — | The path of the decrypted draft config file. |
| `checkpoint_engine_wait_weights_before_ready` | --checkpoint-engine-wait-weights-before-ready | `bool` | `False` | — | If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests. |
| `weight_cache_mode` | --weight-cache-mode | `str` | `'off'` | off, daemon, client | Weight cache mode. 'off': normal disk loading. 'daemon': launch weight cache daemon (holds weights in GPU memory). Engine-spawned daemons are co-terminal with the engine and do NOT persist across restarts, so this alone does not speed up restart (the first start is slower). For fast recovery, run the standalone daemon (python -m sglang.srt.weight_cache.daemon) and connect with 'client'. 'client': connect to existing daemon and load via IPC. |
| `weight_cache_socket` | --weight-cache-socket | `Optional[str]` | `None` | — | Unix socket path for weight cache daemon (client mode).If not set, derives the path from SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE using the caller's physical GPU UUID. |
| `weight_cache_timeout` | --weight-cache-timeout | `int` | `1800` | — | Timeout in seconds for weight cache daemon readiness (default: 1800). |

### 并行（TP / PP / DP / EP / CP / DCP）

命名空间 `parallel`，声明类 `Parallel`，文件 `python/sglang/srt/arg_groups/fields/parallel.py`。共 73 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `nccl_port` | --nccl-port | `Optional[int]` | `None` | — | The port for NCCL distributed environment setup. Defaults to a random port. |
| `dist_timeout` | --dist-timeout | `Optional[int]` | `None` | — | Set timeout for torch.distributed initialization. |
| `dist_init_addr` | --dist-init-addr (--nccl-init-addr) | `Optional[str]` | `None` | — | The host address for initializing distributed backend (e.g., `192.168.0.2:25000`). |
| `gated_launch_port` | --gated-launch-port | `Optional[int]` | `None` | — | The port of the gated launch control server. When set, every rank blocks right after the distributed environment is initialized, before any sizable GPU allocation, until `POST /gate/activate` is sent to this port on the host of the first rank. This lets an external orchestrator defer the memory hungry part of startup to a safe window. Defaults to None, which disables the gate. |
| `nnodes` | --nnodes | `int` | `1` | — | The number of nodes. |
| `node_rank` | --node-rank | `int` | `0` | — | The node rank. |
| `tp_size` | --tp-size (--tensor-parallel-size) | `int` | `1` | — | The tensor parallelism size. |
| `dcp_size` | --dcp-size (--decode-context-parallel-size) | `int` | `1` | — | The decode context parallelism size. |
| `pp_size` | --pp-size (--pipeline-parallel-size) | `int` | `1` | — | The pipeline parallelism size. |
| `pp_max_micro_batch_size` | --pp-max-micro-batch-size | `Optional[int]` | `None` | — | The maximum micro batch size in pipeline parallelism. |
| `pp_async_batch_depth` | --pp-async-batch-depth | `int` | `0` | — | The async batch depth of pipeline parallelism. |
| `dp_size` | --dp-size (--data-parallel-size; 可被解析阶段改写) | `int` | `1` | — | The number of data-parallel replicas of the model. |
| `attn_dp_size` | --attn-dp-size (--attention-data-parallel-size; 可被解析阶段改写) | `int` | `1` | — | The attention data parallelism size: the number of data-parallel attention groups inside the TP group, one scheduler each, while the FFN stays tensor parallel. |
| `load_balance_method` | --load-balance-method | `str` | `'auto'` | auto, round_robin, follow_bootstrap_room, total_requests, total_tokens | The load balancing strategy for data parallelism. |
| `attn_cp_size` | --attn-cp-size (--attention-context-parallel-size; 可被解析阶段改写) | `int` | `1` | — | The attention context parallelism size. |
| `moe_dp_size` | --moe-dp-size (--moe-data-parallel-size) | `int` | `1` | — | The moe data parallelism size. |
| `dwdp_size` | --dwdp-size | `int` | `1` | — | DWDP (Distributed Weight Data Parallelism) group size. When > 1, MoE prefill uses weight prefetch instead of token all-to-all. Must equal tp_size. Only supported with --disaggregation-mode null or prefill. |
| `dcp_comm_backend` | --dcp-comm-backend (可被解析阶段改写) | `Optional[str]` | `None` | ag_rs, a2a, fi_a2a | Communication backend for the decode context-parallel (DCP) attention reduction: 'ag_rs' (AllGather + ReduceScatter), 'a2a' (one All-to-All exchange of output+LSE, then a local LSE combine), or 'fi_a2a' (FlashInfer MNNVL All-to-All kernel; requires Blackwell and a DCP group within one MNNVL domain). Unset resolves to 'fi_a2a' where supported, else 'a2a' on CUDA/ROCm/XPU, else 'ag_rs'. |
| `dcp_replicate_q_proj` | --dcp-replicate-q-proj (可被解析阶段改写) | `Optional[bool]` | `None` | — | For MLA decode context parallelism with the a2a/fi_a2a backend: replicate the Q projection so each DCP rank computes the full-head query locally (redundant projection compute), eliminating the per-layer head-dim all-gather of Q. Trades a small amount of extra GEMM for one fewer collective per layer. Use --no-dcp-replicate-q-proj to disable the model-specific default. |
| `enable_prefill_cp` | --enable-prefill-cp | `bool` | `False` | — | Enable context parallelism for the prefill phase. Select the layout with --cp-strategy. |
| `cp_strategy` | --cp-strategy | `Optional[str]` | `None` | zigzag, interleave | Sharding strategy for prefill CP. 'zigzag' assigns each rank one early and one late sequence block; 'interleave' assigns token indices modulo the CP size. |
| `enable_cp_tp_group_sharing` | --enable-cp-tp-group-sharing (无 CLI; 可被解析阶段改写) | `bool` | `False` | — | (Derived) CP-TP group sharing: prefill CP for hybrid linear-attention models where the CP group is the TP group; the residual stream, attention and indexer are CP-sharded while MoE and linear attention keep TP. Resolved from the model architecture. |
| `enable_dsa_cache_layer_split` | --enable-dsa-cache-layer-split | `bool` | `False` | — | Split DSA (DeepSeek Sparse Attention) GPU KV/indexer cache layers across context-parallel ranks to reduce per-rank KV memory. Currently only supported with the mooncake transfer backend (mooncake / mooncake_tcp); mori/nixl support will be added later by the community. |
| `enable_cp_decode_attn_tp` | --enable-cp-decode-attn-tp | `bool` | `False` | — | Enable attention tensor-parallel weight slicing during decode under context parallel (cp_size>1). Slices the replicated attention linears to the local CP partition, eliminating redundant decode GEMMs. |
| `enable_kv_cache_sharding` | --enable-kv-cache-sharding | `bool` | `False` | — | Shard at-rest KV cache storage at page granularity across the group that currently replicates it (attention-CP ranks for GQA models under prefill CP; attention-TP ranks for MLA models), multiplying the group's unique KV capacity. Prefill workers of PD disaggregation only. |
| `enable_dp_attention` | --enable-dp-attention (无 CLI; 可被解析阶段改写) | `bool` | `False` | — | Deprecated spelling of `attn_dp_size`: `--dp-size N --enable-dp-attention` resolves to `attn_dp_size = N`, `dp_size = 1`. TODO: remove after 2026-12-31. |
| `enable_dp_attention_local_control_broadcast` | --enable-dp-attention-local-control-broadcast | `bool` | `False` | — | With DP-attention, send control messages to every DP group leader and broadcast within attn_tp_group instead of the full tp_group. Eliminates a costly all-ranks gloo sync on every scheduler iteration. |
| `enable_dp_lm_head` | --enable-dp-lm-head (可被解析阶段改写) | `bool` | `False` | — | Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention. |
| `enable_tp_lm_head_all_to_all` | --enable-tp-lm-head-all-to-all (可被解析阶段改写) | `Optional[bool]` | `None` | — | Use all-to-all instead of TP all-gather followed by DP scatter for the TP-sharded LM head under DP attention. By default this is enabled only on decode-only PD nodes with pure DP attention (tp_size == attn_dp_size > 1 and attn_cp_size == 1), and disabled on prefill-only and colocated nodes. Pass --no-enable-tp-lm-head-all-to-all to opt out. The path is incompatible with --enable-dp-lm-head; batches without an equal padded row count fall back to the existing all-gather path. |
| `enable_attn_tp_input_scattered` | --enable-attn-tp-input-scattered (可被解析阶段改写) | `bool` | `False` | — | Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent. |
| `enable_shared_experts_attn_tp` | --enable-shared-experts-attn-tp | `bool` | `False` | — | Shard shared expert weights across the attention TP group when using an expert-parallel all-to-all backend. |
| `shared_experts_tp_size` | --shared-experts-tp-size | `Optional[int]` | `None` | — | Shared-expert TP size for Kimi-K3 with an expert-parallel all-to-all backend. Must divide attention TP size. Overrides --enable-shared-experts-attn-tp when set; 1 replicates the weights. |
| `enable_dense_mlp_attn_tp` | --enable-dense-mlp-attn-tp | `bool` | `False` | — | Shard dense MLP weights across the attention TP group under DP attention. |
| `enable_layernorm_sp` | --enable-layernorm-sp | `bool` | `False` | — | Enable Megatron-style sequence parallelism (arXiv:2205.05198) for the LayerNorm/residual regions under pure tensor parallelism: the row-parallel all-reduce becomes reduce-scatter + all-gather, so LayerNorm runs on sequence-sharded activations with no extra communication volume. Prefill only; supports Qwen3 dense and Muse Spark ModelOpt NVFP4 with TP4/EP4 or TP8/EP8 with FlashInfer A2A or MegaMoE. Muse Spark requires the FlashInfer TRT-LLM MoE runner and NVLink/NVSwitch. |
| `enable_megamoe_for_layernorm_sp` | --enable-megamoe-for-layernorm-sp | `bool` | `False` | — | Use fused MegaMoE for Muse Spark NVFP4 TP4/EP4 or TP8/EP8 LayerNorm-SP prefill, including BF16 excluded layers. Decode retains the FlashInfer TRT-LLM MoE runner and all-reduce, sharing expert weights with prefill. Requires --enable-layernorm-sp, DeepGEMM with post-expert normalization, and NVLink/NVSwitch. No FlashInfer A2A backend is required when enabled. |
| `disable_attn_tp_gather` | --disable-attn-tp-gather | `bool` | `False` | — | Disable scheduler-side attn_tp_gather (the upstream SP path that pads num_tokens to attn_tp_size and pre-allocates a gathered buffer). Use for models that manage SP scatter/gather at the model level (e.g., perform their own all_gather/reduce_scatter inside attention) and do not consume the upstream gathered_buffer. Without this, the cuda graph runner pads num_tokens to attn_tp_size, which can cause kernel autotuners to select wrong-sized variants at small batches. |
| `enable_p2p_check` | --enable-p2p-check | `bool` | `False` | — | Enable P2P check for GPU access, otherwise the p2p access is allowed by default. |
| `ep_size` | --ep-size (--expert-parallel-size, --ep; 可被解析阶段改写) | `int` | `1` | — | The expert parallelism size. |
| `moe_dense_tp_size` | --moe-dense-tp-size (可被解析阶段改写) | `Optional[int]` | `None` | — | TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports. |
| `ep_join_rank_offset` | --elastic-ep-join-rank-offset | `int` | `0` | — | Global rank offset of an elastic EP joining group. Scale joiners must set this to the current effective EP size. |
| `elastic_ep_initial_size` | --elastic-ep-initial-size | `Optional[int]` | `None` | — | EP size used to define the immutable per-rank expert storage layout. Scale joiners must use the primary deployment's launch-time EP size. |
| `max_ep_size` | --max-ep-size | `Optional[int]` | `None` | — | Maximum EP size the server can scale to at runtime. Pre-allocates active-rank state and backend buffers to this size. Defaults to the launch-time world size. |
| `attn_tp_size` | --attn-tp-size (fn=sglang.srt.runtime_context.attn_tp_size_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Attention tensor-parallel width: `tp_size` divided by the attention-DP and attention-CP dimensions. |
| `attn_dp_enabled` | --attn-dp-enabled (fn=sglang.srt.runtime_context.attn_dp_enabled_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Whether attention runs data parallel: `attn_dp_size` is wider than one, or this process is an elastic EP scale joiner, which joins an attention-DP deployment with a group of its own that may be one rank wide. |
| `num_dp_ranks` | --num-dp-ranks (fn=sglang.srt.runtime_context.num_dp_ranks_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | How many data-parallel ranks the deployment serves with, one scheduler each: `dp_size * attn_dp_size`. Elastic EP scale-up widens it as ranks join. |
| `attn_dcp_size` | --attn-dcp-size (fn=sglang.srt.runtime_context.attn_dcp_size_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Decode context-parallel width inside the attention TP group. |
| `moe_ep_size` | --moe-ep-size (fn=sglang.srt.runtime_context.moe_ep_size_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | MoE expert-parallel width, normalised from the configured value. |
| `moe_tp_size` | --moe-tp-size (fn=sglang.srt.runtime_context.moe_tp_size_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | MoE tensor-parallel width: what is left of `tp_size` after the expert and MoE-DP dimensions. |
| `dcp_enabled` | --dcp-enabled (fn=sglang.srt.runtime_context.dcp_enabled_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Whether decode context parallelism is in play: `dcp_size` is wider than one rank, which is exactly when the group gets built. |
| `tp_rank` | --tp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's place in the tensor-parallel group. |
| `pp_rank` | --pp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's place in the pipeline group. |
| `moe_ep_rank` | --moe-ep-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's place in the expert-parallel group. |
| `moe_dp_rank` | --moe-dp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's place in the MoE data-parallel group. |
| `moe_tp_rank` | --moe-tp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's place in the MoE tensor-parallel group. |
| `attn_tp_rank` | --attn-tp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's place in the attention tensor-parallel group. |
| `attn_cp_rank` | --attn-cp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's place in the attention context-parallel group. |
| `dcp_rank` | --dcp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's place in the decode context-parallel group. |
| `attn_dcp_rank` | --attn-dcp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | Decode context-parallel rank inside the attention TP group, zero where decode context parallelism is off. |
| `attn_dp_rank` | --attn-dp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's index in the attention-DP group, computed from `tp_rank` when `initialize_dp_attention` runs. |
| `dp_rank` | --dp-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | Which data-parallel replica this process serves, as the data parallel controller numbered them at spawn. `None` when there is no controller: unlike the other ranks it is a position in no group, which is why the spawn states it. |
| `launch_world_rank` | --launch-world-rank (派生，无 CLI) | `Derived` | `（发布时计算）` | — | This process's rank in the WORLD group as built. A scale-up does not renumber it. |
| `launch_world_size` | --launch-world-size (fn=sglang.srt.runtime_context.launch_world_size_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Width the WORLD group was built at -- what a scale-up leaves behind rather than updates. |
| `max_world_size` | --max-world-size (fn=sglang.srt.runtime_context.max_world_size_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Ranks the WORLD group has room for: `--max-ep-size` when set, otherwise the launch width. |
| `world_group` | --world-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The WORLD group. |
| `tp_group` | --tp-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The tensor-parallel group. |
| `pp_group` | --pp-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The pipeline group. |
| `moe_ep_group` | --moe-ep-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The expert-parallel group. |
| `moe_dp_group` | --moe-dp-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The MoE data-parallel group. |
| `moe_tp_group` | --moe-tp-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The MoE tensor-parallel group. |
| `attn_tp_group` | --attn-tp-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The attention tensor-parallel group. |
| `attn_cp_group` | --attn-cp-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The attention context-parallel group. |
| `shared_experts_tp_group` | --shared-experts-tp-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The shared-expert tensor-parallel group. |
| `dcp_group` | --dcp-group (派生，无 CLI) | `Derived` | `（发布时计算）` | — | The decode context-parallel group. |

### 显存、KV 与前缀缓存

命名空间 `memory`，声明类 `Memory`，文件 `python/sglang/srt/arg_groups/fields/memory.py`。共 31 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `radix_eviction_policy` | --radix-eviction-policy (可被解析阶段改写) | `str` | `'lru'` | lru, lfu, slru, priority, tlru | The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used, 'slru' stands for Segmented Least Recently Used, 'priority' evicts lower-priority requests first, and 'tlru' stands for Tail-Optimized LRU (arXiv:2510.15152), which evicts the part of a conversation that cannot affect tail TTFT before falling back to LRU. See https://docs.sglang.io/docs/advanced_features/radix_eviction_policy for what each policy optimizes for. |
| `_radix_eviction_policy_explicitly_set` | ---radix-eviction-policy-explicitly-set (无 CLI) | `bool` | `False` | — | The value alone cannot distinguish the default from an explicit LRU choice, which model-specific defaults must preserve. |
| `radix_eviction_policy_config` | --radix-eviction-policy-config | `Optional[Dict[str, Any]]` | `None` | — | Tuning parameters for --radix-eviction-policy, as a json object passed to the policy as keyword arguments. 'slru' takes protected_threshold (int, default 2), e.g. '{"protected_threshold": 4}'; 'tlru' takes threshold and next_prompt_estimate (ints, tokens), e.g. '{"threshold": 4096, "next_prompt_estimate": 512}'. An unrecognized key fails at startup, naming the key and the policy. See https://docs.sglang.io/docs/advanced_features/radix_eviction_policy#policy-parameters for the full parameter list. |
| `disable_radix_cache` | --disable-radix-cache (可被解析阶段改写) | `bool` | `False` | — | Disable RadixAttention for prefix caching. |
| `enable_page_major_kv_layout` | --enable-page-major-kv-layout | `bool` | `False` | — | Store the Mamba state and full/SWA KV caches as one entry per token slot, holding every layer's state, in a single page-granular buffer instead of one buffer per layer. Implied by --enable-unified-memory, and currently supported only with it. |
| `enable_unified_memory` | --enable-unified-memory | `bool` | `False` | — | Replace the statically-partitioned hybrid-model pools (full-attn KV + SWA/Mamba state) with one byte buffer split dynamically between sub-pools. Requires stride-aware attention, linear-attention and Mamba backends; the allowed backends and the supported PD-disaggregation and speculative-decoding configurations are validated at startup. |
| `enable_session_radix_cache` | --enable-session-radix-cache | `bool` | `False` | — | Track per-session references on UnifiedRadixCache KV: eviction consumes unreferenced entries before referenced ones, and closing a session only dereferences its KV. |
| `radix_cache_backend` | --radix-cache-backend | `Optional[str]` | `None` | — | Name of a radix-cache backend previously registered via register_radix_cache_backend. Omit this flag to use the built-in default cache selection chain. |
| `enable_hierarchical_cache` | --enable-hierarchical-cache (--enable-hicache) | `bool` | `False` | — | Enable hierarchical cache |
| `hicache_host_memory_mode` | --hicache-host-memory-mode | `str` | `'cache'` | cache, buffer_only | Whether host memory is a persistent HiCache tier (cache) or a transient staging buffer between GPU and the storage backend (buffer_only). buffer_only requires --hicache-storage-backend. |
| `hicache_ratio` | --hicache-ratio | `Optional[float]` | `None` | — | The ratio of the size of host KV cache memory pool to the size of device pool. Defaults to 2.0 in cache mode, 1.2 in buffer_only mode, or 0.2 for backup-only host-pool decode retraction. |
| `hicache_size` | --hicache-size | `int` | `0` | — | The size of host KV cache memory pool in gigabytes. Overrides --hicache-ratio in either host memory mode. |
| `hicache_host_memory_fraction` | --hicache-host-memory-fraction | `Optional[float]` | `0.8` | — | Fraction of the available host memory, bounded by visible cgroup memory.max/memory.high or v1 memory limits (after a 10 GiB reserve) that the HiCache host pools of all ranks on this machine may use. Applies only when neither --hicache-ratio nor --hicache-size is set: the default ratio is then reduced until the pools fit. Lower it when several engines share a memory cgroup. |
| `hicache_write_policy` | --hicache-write-policy | `str` | `'write_through'` | write_back, write_through, write_through_selective | The write policy of hierarchical cache. |
| `hicache_io_backend` | --hicache-io-backend | `str` | `'kernel'` | direct, kernel, kernel_ascend | The IO backend for KV cache transfer between CPU and GPU |
| `hicache_mem_layout` | --hicache-mem-layout | `str` | `'page_first'` | layer_first, page_first, page_first_direct, page_first_kv_split, page_head | The layout of host memory pool for hierarchical cache. |
| `hicache_storage_backend` | --hicache-storage-backend | `Optional[str]` | `None` | file, sim, mooncake, npu_memcache, hf3fs, nixl, aibrix, dynamic, eic, simm, mori, shm, tensorcast, seaweedfs | The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, npu_memcache, hf3fs, nixl, aibrix, tensorcast, seaweedfs. For dynamic backend, use --hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name). |
| `hicache_storage_prefetch_policy` | --hicache-storage-prefetch-policy | `str` | `'timeout'` | best_effort, wait_complete, timeout | Control when prefetching from the storage backend should stop. |
| `hicache_storage_backend_extra_config` | --hicache-storage-backend-extra-config | `Optional[str]` | `None` | — | A dictionary in JSON string format, or a string starting with a leading '@' and a config file in JSON/YAML/TOML format, containing extra configuration for the storage backend. |
| `hicache_storage_prefetch_retry_poll_interval` | --hicache-storage-prefetch-retry-poll-interval | `int` | `8` | — | Scheduling passes a queued request waits before its storage availability check is re-issued, when the prefetch found nothing and a backup may still be committing (under load the first check can run before it does). A re-issue that waits on staging or a moved match instead goes out on the next pass. Only passes that reach prefill scheduling count. 0 disables miss retries; known-hit deferrals are always re-issued. |
| `hicache_storage_prefetch_retry_max_attempts` | --hicache-storage-prefetch-retry-max-attempts | `int` | `8` | — | Storage availability re-issues a queued request may make, paced miss polls and immediate re-issues alike; past the cap it is admitted with whatever the device holds. 0 disables re-issues. |
| `enable_unified_cache_external_linker` | --enable-unified-cache-external-linker | `bool` | `False` | — | Link UnifiedRadixCache directly to an external KV store (direct L3), with no host cache tier. |
| `unified_cache_external_linker_backend` | --unified-cache-external-linker-backend | `str` | `'mooncake'` | mooncake, mori, dynamic | Storage backend for --enable-unified-cache-external-linker. |
| `unified_cache_external_linker_config` | --unified-cache-external-linker-config | `Optional[Dict[str, Any]]` | `None` | — | JSON configuration for the selected linker backend. Built-in backends accept storage options directly. The dynamic backend requires linker (class name), linker_module_path (Python module), and optional linker_extra_config (object). May be supplied while the linker is disabled. For built-in backends, replaces --hicache-storage-backend-extra-config when provided. The fallback to that flag for external linkers is deprecated; ordinary HiCache use of that flag is unaffected. |
| `enable_linker_mla_dedup` | --enable-linker-mla-dedup | `bool` | `False` | — | Load replicated MLA KV on rank 0 and broadcast each layer with the Mooncake linker. |
| `enable_hisparse` | --enable-hisparse | `bool` | `False` | — | Enable hierarchical sparse attention |
| `hisparse_config` | --hisparse-config (--hierarchical-sparse-attention-extra-config) | `Optional[str]` | `None` | — | A dictionary in JSON string format for hierarchical sparse attention configuration. Example: '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}' |
| `enable_lmcache` | --enable-lmcache | `bool` | `False` | — | Using LMCache as an alternative hierarchical cache solution |
| `lmcache_config_file` | --lmcache-config-file | `Optional[str]` | `None` | — | Path to the LMCache YAML configuration file |
| `enable_flexkv` | --enable-flexkv | `bool` | `False` | — | Route the default RadixCache through FlexKV's KVManager for host-tier (CPU / SSD / Remote) KV cache offload. Equivalent to --radix-cache-backend=flexkv but also participates in the auto-selection chain alongside --enable-lmcache. |
| `flexkv_config_file` | --flexkv-config-file | `Optional[str]` | `None` | — | Path to the FlexKV YAML / JSON configuration file. Equivalent to setting the FLEXKV_CONFIG_PATH environment variable. |

### 调度

命名空间 `schedule`，声明类 `Schedule`，文件 `python/sglang/srt/arg_groups/fields/schedule.py`。共 41 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `mem_fraction_static` | --mem-fraction-static | `Optional[float]` | `None` | — | The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors. |
| `max_running_requests` | --max-running-requests | `Optional[int]` | `None` | — | The maximum number of running requests. |
| `max_queued_requests` | --max-queued-requests | `Optional[int]` | `None` | — | The maximum number of queued requests. This option is ignored when using disaggregation-mode. |
| `max_total_tokens` | --max-total-tokens | `Optional[int]` | `None` | — | — |
| `chunked_prefill_size` | --chunked-prefill-size | `Optional[int]` | `None` | — | The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill. |
| `prefill_decode_interval` | --prefill-decode-interval (可被解析阶段改写) | `Optional[int]` | `None` | — | The number of decode rounds to run after a prefill batch before scheduling the next prefill. By default, this is disabled except for profiled Qwen3-VL serving configurations on Hopper. In data-parallel attention mode, the interval is synchronized across all DP ranks. Set to 0 to disable. |
| `prefill_decode_interval_min_bs` | --prefill-decode-interval-min-bs | `int` | `0` | — | Apply --prefill-decode-interval only while the running batch holds at least this many requests. Below it, prefill is admitted every round so the batch fills quickly. In data-parallel attention mode, each rank compares its own running batch. Set to 0 to always apply the interval. |
| `enable_dynamic_chunking` | --enable-dynamic-chunking | `bool` | `False` | — | Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks. |
| `max_prefill_tokens` | --max-prefill-tokens | `int` | `16384` | — | — |
| `prefill_max_requests` | --prefill-max-requests | `Optional[int]` | `None` | — | The maximum number of requests in a prefill batch. If not specified, there is no limit. |
| `schedule_policy` | --schedule-policy | `str` | `'fcfs'` | lpm, random, fcfs, dfs-weight, lof, priority, routing-key, hrrn, shortest-prefill-first | The scheduling policy of the requests. |
| `enable_priority_scheduling` | --enable-priority-scheduling | `bool` | `False` | — | Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default. |
| `disable_priority_preemption` | --disable-priority-preemption | `bool` | `False` | — | Disable priority scheduling preemption. |
| `default_priority_value` | --default-priority-value | `Optional[int]` | `None` | — | Default priority for requests without explicit priority. |
| `abort_on_priority_when_disabled` | --abort-on-priority-when-disabled | `bool` | `False` | — | If set, abort requests that specify a priority when priority scheduling is disabled. |
| `schedule_low_priority_values_first` | --schedule-low-priority-values-first | `bool` | `False` | — | If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first. |
| `priority_scheduling_preemption_threshold` | --priority-scheduling-preemption-threshold | `int` | `10` | — | Minimum difference in priorities for an incoming request to have to preempt running request(s). |
| `retraction_policy` | --retraction-policy | `str` | `'length'` | length, priority | The decode retraction policy to use when the KV cache is full. 'length' preserves the existing behavior and retracts short-output, long-input requests first. 'priority' retracts lower-priority requests first, using the same priority direction as priority scheduling. |
| `schedule_conservativeness` | --schedule-conservativeness | `float` | `1.0` | — | How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently. |
| `page_size` | --page-size (可被解析阶段改写) | `Optional[int]` | `None` | — | The number of tokens in a page. |
| `c128_page_size` | --c128-page-size | `int` | `16` | — | The physical page size of the NPU DSV4 C128 KV cache. Must be a positive multiple of 16. |
| `swa_full_tokens_ratio` | --swa-full-tokens-ratio (可被解析阶段改写) | `Optional[float]` | `None` | — | The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens. |
| `_swa_full_tokens_ratio_explicitly_set` | ---swa-full-tokens-ratio-explicitly-set (无 CLI) | `Optional[bool]` | `None` | — | Recorded by the cache hook; the effective field answers the fallback when unset. |
| `swa_prefix_tails` | --swa-prefix-tails | `Optional[int]` | `None` | — | When the SWA KV pool is sized from the request cap (DeepSeek-V4 family), how many radix-cached prefix tails it keeps room for. Each tail is one sliding window plus one page. Default: 4 x max_running_requests per attention-DP rank, multiplied by num_layers // draft_layers under encoder SWA bounded replay with DSpark (only draft layers are paged there); 0 when the radix cache is disabled. |
| `disable_hybrid_swa_memory` | --disable-hybrid-swa-memory (可被解析阶段改写) | `bool` | `False` | — | Disable the hybrid SWA memory pool. |
| `prefill_only_disable_kv_cache` | --prefill-only-disable-kv-cache | `bool` | `False` | — | Skip the physical KV cache allocation for embedding-mode prefill-only workloads. Currently only valid with --is-embedding, --chunked-prefill-size=-1, --disable-radix-cache, an FA prefill backend, and non-FP4 KV cache so the fa_skip_kv_cache path is active (no layer reads or writes the cache). Other prefill-only workloads such as scoring/MIS may benefit from this later once their attention paths stop using paged KV. Scheduler admission accounting is unchanged; per-layer K/V tensors are sized to (page_size, head_num, head_dim) placeholders so GPU memory is not wasted. |
| `disable_chunked_prefix_cache` | --disable-chunked-prefix-cache | `bool` | `False` | — | Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences. |
| `disable_overlap_schedule` | --disable-overlap-schedule (可被解析阶段改写) | `bool` | `False` | — | Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker. |
| `num_continuous_decode_steps` | --num-continuous-decode-steps | `int` | `1` | — | Run multiple continuous decoding steps to reduce scheduling overhead. This can potentially increase throughput but may also increase time-to-first-token latency. The default value is 1, meaning only run one decoding step at a time. |
| `scheduler_recv_interval` | --scheduler-recv-interval | `int` | `1` | — | The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this. |
| `enable_mixed_chunk` | --enable-mixed-chunk | `bool` | `False` | — | Enabling mixing prefill and decode in a batch when using chunked prefill. |
| `max_mamba_cache_size` | --max-mamba-cache-size | `Optional[int]` | `None` | — | The maximum size of the mamba cache. |
| `mamba_full_memory_ratio` | --mamba-full-memory-ratio (可被解析阶段改写) | `Optional[float]` | `None` | — | The ratio of mamba state memory to full kv cache memory. |
| `enable_prefill_delayer` | --enable-prefill-delayer | `bool` | `False` | — | Enable prefill delayer for DP attention to reduce idle time. |
| `prefill_delayer_max_delay_passes` | --prefill-delayer-max-delay-passes | `int` | `30` | — | Maximum forward passes to delay prefill. |
| `prefill_delayer_token_usage_low_watermark` | --prefill-delayer-token-usage-low-watermark | `Optional[float]` | `None` | — | Token usage low watermark for prefill delayer. |
| `prefill_delayer_forward_passes_buckets` | --prefill-delayer-forward-passes-buckets | `Optional[List[float]]` | `None` | — | Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added. |
| `prefill_delayer_wait_seconds_buckets` | --prefill-delayer-wait-seconds-buckets | `Optional[List[float]]` | `None` | — | Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added. |
| `prefill_delayer_queue_min_ratio` | --prefill-delayer-queue-min-ratio | `Optional[float]` | `None` | — | Opt-in to the adaptive queue-based delay trigger (independent of the slot-based one). Delays prefill until the waiting queue reaches min(running_req * ratio, prefill_max_requests), falling back to the observed max_prefill_bs when no request limit is set. Unset (default) keeps the original slot-only behavior. Typical: 0.1 ~ 0.5. |
| `prefill_delayer_max_delay_ms` | --prefill-delayer-max-delay-ms | `Optional[float]` | `None` | — | Wall-clock cap (ms) on a single queue-trigger delay; once exceeded, prefill is force-released to bound worst-case TTFT. Only consulted when --prefill-delayer-queue-min-ratio is set. Typical: 1000 ~ 5000; defaults to 5000 if unset. |
| `min_free_slots_delay` | --min-free-slots-delay | `Optional[int]` | `None` | — | Hold new prefills until at least N running-request slots have freed up, so they are admitted in one batch instead of one at a time. Useful when each admission is disproportionately expensive, e.g. speculative decoding with a separate draft prefill pass. An explicit value always wins, capped by max-running-requests (1 disables). When unset, DFlash workloads auto-enable the formula; other workloads stay disabled. Not supported with pipeline parallelism. |

### HTTP 服务、解析器与 API

命名空间 `serving`，声明类 `Serving`，文件 `python/sglang/srt/arg_groups/fields/serving.py`。共 64 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `enable_response_store` | --enable-response-store | `bool` | `False` | — | Enable in-memory Responses storage for retrieval, chaining, and background requests. Disabled by default; unsupported with prefill-decode disaggregation. Storage has no TTL or size limit. |
| `tokenizer_path` | --tokenizer-path | `Optional[str]` | `None` | — | The path of the tokenizer. |
| `tokenizer_mode` | --tokenizer-mode | `str` | `'auto'` | auto, slow | Tokenizer mode. 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer. |
| `tokenizer_backend` | --tokenizer-backend | `str` | `'huggingface'` | huggingface, fastokens | Tokenizer backend. 'huggingface' uses the default HuggingFace tokenizers library, and 'fastokens' uses the fastokens library for faster tokenization. Requires the fastokens package to be installed. |
| `tokenizer_worker_num` | --tokenizer-worker-num | `int` | `1` | — | The worker num of the tokenizer manager. |
| `detokenizer_worker_num` | --detokenizer-worker-num | `int` | `1` | — | The worker num of the detokenizer manager. |
| `skip_tokenizer_init` | --skip-tokenizer-init | `bool` | `False` | — | If set, skip init tokenizer and pass input_ids in generate request. |
| `host` | --host | `str` | `'127.0.0.1'` | — | The host of the HTTP server. |
| `port` | --port | `int` | `30000` | — | The port of the HTTP server. |
| `fastapi_root_path` | --fastapi-root-path | `str` | `''` | — | App is behind a path based routing proxy. |
| `enable_http_worker_reuseport` | --enable-http-worker-reuseport | `bool` | `False` | — | Give each Uvicorn HTTP worker its own SO_REUSEPORT listener. |
| `smg_grpc_mode` | --smg-grpc-mode | `bool` | `False` | — | Use the legacy SMG gRPC server (smg-grpc-servicer) instead of the HTTP server. Replaces the deprecated --grpc-mode. |
| `grpc_mode` | --grpc-mode | `bool` | `False` | — | (Deprecated, use --smg-grpc-mode) Legacy SMG gRPC server selector. |
| `grpc_port` | --grpc-port | `Optional[int]` | `None` | — | Port for the native gRPC server, started alongside HTTP. Setting this (or SGLANG_GRPC_PORT) enables the native gRPC server; it is off by default. Follower nodes expose GetServerInfo only. In legacy --smg-grpc-mode this is the SMG server port and defaults to --port + 10000. |
| `grpc_response_timeout_secs` | --grpc-response-timeout-secs | `int` | `300` | — | Timeout in seconds waiting for each native gRPC response chunk, including the first. Must be positive. Defaults to 300. |
| `grpc_worker_threads` | --grpc-worker-threads (无 CLI) | `Optional[int]` | `None` | — | Env-only (SGLANG_GRPC_WORKER_THREADS); a field so the projection sees it. |
| `sidecar` | --sidecar | `Optional[str]` | `None` | — | Start a locally managed sidecar against the native gRPC server. The selected module must expose main(argv) and read the resolved native gRPC endpoint from SGLANG_GRPC_ENDPOINT. Requires --grpc-port or SGLANG_GRPC_PORT. With SGLANG_RUST_SERVER, every node that hosts a Rust listener runs one sidecar, with SGLANG_GRPC_ENDPOINT set to that node's first listener (--grpc-port). |
| `sidecar_args` | --sidecar-args | `Optional[List[str]]` | `None` | — | JSON array passed to the selected sidecar module's main(argv) function. --sidecar-shutdown-timeout SECONDS is consumed by SGLang. |
| `skip_server_warmup` | --skip-server-warmup | `bool` | `False` | — | If set, skip warmup. |
| `warmups` | --warmups | `Optional[str]` | `None` | — | Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests |
| `enable_http2` | --enable-http2 | `bool` | `False` | — | Use Granian instead of Uvicorn as the ASGI server, enabling HTTP/1.1 and HTTP/2 auto-negotiation. Clients may use h2c (cleartext HTTP/2) or plain HTTP/1.1. Requires 'pip install sglang[http2]'. |
| `http2_max_concurrent_streams` | --http2-max-concurrent-streams | `int` | `200` | — | Maximum number of concurrent streams advertised on each HTTP/2 connection (1 to 2^32 - 1). Only applies with --enable-http2. |
| `http2_initial_connection_window_size` | --http2-initial-connection-window-size | `int` | `1024 * 1024` | — | Initial connection-level HTTP/2 receive window in bytes (1024 to 2^31 - 1). Only applies with --enable-http2. |
| `ssl_keyfile` | --ssl-keyfile | `Optional[str]` | `None` | — | The file path to the SSL key file. |
| `ssl_certfile` | --ssl-certfile | `Optional[str]` | `None` | — | The file path to the SSL certificate file. |
| `ssl_ca_certs` | --ssl-ca-certs | `Optional[str]` | `None` | — | The CA certificates file. |
| `ssl_keyfile_password` | --ssl-keyfile-password | `Optional[str]` | `None` | — | The password to decrypt the SSL keyfile. |
| `enable_ssl_refresh` | --enable-ssl-refresh | `bool` | `False` | — | Enable automatic SSL certificate hot-reloading when cert/key files change on disk. Requires --ssl-certfile and --ssl-keyfile. |
| `api_key` | --api-key | `Optional[str]` | `None` | — | Set API key of the server. It is also used in the OpenAI API compatible server. |
| `admin_api_key` | --admin-api-key | `Optional[str]` | `None` | — | Set admin API key for sensitive management endpoints (e.g. /hicache/storage-backend/clear). When set, admin endpoints require this key and do NOT accept --api-key. |
| `served_model_name` | --served-model-name | `Optional[str]` | `None` | — | Override the model name returned by the v1/models endpoint in OpenAI API server. |
| `weight_version` | --weight-version | `str` | `'default'` | — | Version identifier for the model weights. Defaults to 'default' if not specified. |
| `chat_template` | --chat-template | `Optional[str]` | `None` | — | The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server. |
| `trust_request_chat_template` | --trust-request-chat-template | `bool` | `False` | — | Allow a request to override the server chat template via its chat_template_kwargs. Off by default for safety. |
| `hf_chat_template_name` | --hf-chat-template-name | `Optional[str]` | `None` | — | When the HuggingFace tokenizer has multiple chat templates (e.g., 'default', 'tool_use', 'rag'), specify which named template to use. If not set, the first available template is used. |
| `completion_template` | --completion-template | `Optional[str]` | `None` | — | The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently. |
| `file_storage_path` | --file-storage-path | `Optional[str]` | `None` | — | The path of the file storage in backend. |
| `enable_cache_report` | --enable-cache-report | `bool` | `False` | — | Return number of cached tokens in usage.prompt_tokens_details for each openai request. |
| `return_input_ids` | --return-input-ids | `bool` | `False` | — | Return prompt (input) token ids on the response-level sglext extension for every chat completion request, as if return_input_ids_in_sglext were set on the request. |
| `return_output_ids` | --return-output-ids | `bool` | `False` | — | Return sampled output token ids on the response-level sglext extension for every chat completion request, as if return_output_ids_in_sglext were set on the request. |
| `reasoning_parser` | --reasoning-parser | `Optional[str]` | `None` | — | — |
| `default_chat_template_kwargs` | --default-chat-template-kwargs | `Optional[Dict[str, Any]]` | `None` | — | Default chat template kwargs applied to every request when not overridden per-request. Keys must match what the model's chat template expects (e.g. enable_thinking, thinking, reasoning_effort). Per-request chat_template_kwargs takes precedence. |
| `strip_thinking_cache` | --strip-thinking-cache | `bool` | `False` | — | Skip caching reasoning-model output (thinking + answer) in the radix tree on finish; keep only the prompt prefix. Opt-in: changes cache contents. |
| `enable_strict_thinking` | --enable-strict-thinking | `bool` | `False` | — | Enable strict token filtering during the thinking phase. Blocks model-specific excluded tokens (e.g., tool call markers) during reasoning. Requires a grammar backend that supports token filtering. |
| `tool_call_parser` | --tool-call-parser | `Optional[str]` | `None` | — | — |
| `tool_server` | --tool-server | `Optional[str]` | `None` | — | Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used. |
| `sampling_defaults` | --sampling-defaults | `str` | `'model'` | openai, model | Where to get default sampling parameters. 'openai' uses SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.). 'model' uses the model's generation_config.json to get the recommended sampling parameters if available. Default is 'model'. |
| `asr_max_buffer_seconds` | --asr-max-buffer-seconds | `int` | `60` | — | Maximum seconds of PCM audio the streaming ASR WebSocket handler will accumulate before closing the session with a buffer_overflow error. Guards against OOM when a client streams audio faster than inference can consume it. Default 60s. |
| `asr_max_concurrent_sessions` | --asr-max-concurrent-sessions | `int` | `32` | — | Maximum number of concurrent realtime ASR WebSocket sessions served by /v1/realtime. New connections beyond this cap are accepted, sent an error{code:too_many_sessions} frame, and closed. Default 32. |
| `preferred_sampling_params` | --preferred-sampling-params | `Optional[str]` | `None` | — | json-formatted sampling settings that will be returned in /get_model_info |
| `allow_auto_truncate` | --allow-auto-truncate | `bool` | `False` | — | Allow automatically truncating requests that exceed the maximum input length instead of returning an error. |
| `stream_interval` | --stream-interval | `int` | `1` | — | The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher |
| `batch_notify_size` | --batch-notify-size | `int` | `16` | — | Number of streaming notifications to batch before yielding to the event loop. Reduces asyncio wakeup overhead under high concurrency. |
| `stream_response_default_include_usage` | --stream-response-default-include-usage | `bool` | `False` | — | Include usage in every streaming response (even when stream_options is not specified). |
| `incremental_streaming_output` | --incremental-streaming-output | `bool` | `False` | — | Whether to output as a sequence of disjoint segments. |
| `enable_streaming_session` | --enable-streaming-session | `bool` | `False` | — | Enable streaming session mode and StreamingSession wrapper. |
| `constrained_json_whitespace_pattern` | --constrained-json-whitespace-pattern | `Optional[str]` | `None` | — | (outlines and llguidance backends only) Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [ ]* |
| `constrained_json_disable_any_whitespace` | --constrained-json-disable-any-whitespace | `bool` | `False` | — | (xgrammar and llguidance backends only) Enforce compact representation in JSON constrained output. |
| `constrained_json_max_whitespace_cnt` | --constrained-json-max-whitespace-cnt | `Optional[int]` | `None` | — | (xgrammar backend only) Max consecutive whitespace chars allowed in JSON constrained output. None means unbounded. |
| `enable_dynamic_batch_tokenizer` | --enable-dynamic-batch-tokenizer | `bool` | `False` | — | Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently. |
| `dynamic_batch_tokenizer_batch_size` | --dynamic-batch-tokenizer-batch-size | `int` | `32` | — | [Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer. |
| `dynamic_batch_tokenizer_batch_timeout` | --dynamic-batch-tokenizer-batch-timeout | `float` | `0.002` | — | [Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests. |
| `enable_tokenizer_batch_encode` | --enable-tokenizer-batch-encode | `bool` | `False` | — | Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds. |
| `disable_tokenizer_batch_decode` | --disable-tokenizer-batch-decode | `bool` | `False` | — | Disable batch decoding when decoding multiple completions. |

### 投机解码（MTP / EAGLE / DSPARK）

命名空间 `spec`，声明类 `Spec`，文件 `python/sglang/srt/arg_groups/fields/spec.py`。共 43 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `speculative_algorithm` | --speculative-algorithm | `Optional[str]` | `None` | — | Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO. Or any name registered via `SpeculativeAlgorithm.register`. |
| `uno_lora_path` | --uno-lora-path | `Optional[str]` | `None` | — | Path to the UNO draft LoRA checkpoint. |
| `speculative_draft_model_path` | --speculative-draft-model-path (--speculative-draft-model) | `Optional[str]` | `None` | — | The path of the draft model weights. This can be a local folder or a Hugging Face repo ID. |
| `speculative_boundary_reduction` | --speculative-boundary-reduction (无 CLI; 可被解析阶段改写) | `Literal['ar', 'rs', 'rsv', 'rs+rsv']` | `'rs+rsv'` | — | — |
| `speculative_draft_model_revision` | --speculative-draft-model-revision | `Optional[str]` | `None` | — | The specific draft model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. |
| `speculative_draft_load_format` | --speculative-draft-load-format | `Optional[str]` | `None` | auto, pt, safetensors, instanttensor, npcache, dummy, sharded_state, presharded, gguf, expert_pack, bitsandbytes, mistral, layered, flash_rl, remote, remote_instance, fastsafetensors, private, runai_streamer | The format of the draft model weights to load. If not specified, will use the same format as --load-format. Use 'dummy' to initialize draft model weights with random values for profiling. |
| `speculative_num_steps` | --speculative-num-steps | `Optional[int]` | `None` | — | The number of steps sampled from draft model in Speculative Decoding. |
| `speculative_eagle_topk` | --speculative-eagle-topk | `Optional[int]` | `None` | — | The number of tokens sampled from the draft model in eagle2 each step. |
| `speculative_num_draft_tokens` | --speculative-num-draft-tokens | `Optional[int]` | `None` | — | The number of tokens sampled from the draft model in Speculative Decoding. |
| `speculative_dflash_block_size` | --speculative-dflash-block-size | `Optional[int]` | `None` | — | DFLASH only. Block size (verify window length). Alias of --speculative-num-draft-tokens for DFLASH. |
| `speculative_domino_candidate_pool_size` | --speculative-domino-candidate-pool-size | `int` | `2048` | — | Domino only. Size of the approximate block-shared base-logit candidate pool. Set to 0 to score the full vocabulary. |
| `speculative_dspark_block_size` | --speculative-dspark-block-size | `Optional[int]` | `None` | — | DSPARK only. Draft block size gamma (number of proposed draft tokens). The verify window is gamma + 1, so this sets --speculative-num-draft-tokens = gamma + 1. Omit to auto-infer gamma from the draft checkpoint block_size. |
| `speculative_dspark_sps_table_path` | --speculative-dspark-sps-table-path | `Optional[str]` | `None` | — | DSPARK only. Path to a pre-profiled SPS cost table (JSON) built offline with sglang.benchmark.dspark_sps_profiler, consumed by the ragged-verify scheduler (cap-accept / compact). Omit for an uninitialized flat constant-SPS table: the budget degenerates to verify-all (zero throughput gain by itself). |
| `speculative_dspark_confidence_sts_path` | --speculative-dspark-confidence-sts-path | `Optional[str]` | `None` | — | DSPARK only. Optional path to a per-position STS (sequential temperature scaling) calibration JSON, fit offline with sglang.benchmark.dspark_sts_fit. Calibrates the confidence-head survival probabilities the ragged-verify scheduler consumes. Omit to use identity (no calibration); losslessness is unaffected either way. |
| `speculative_dspark_align_verify_tokens_to_graph_tier` | --speculative-dspark-align-verify-tokens-to-graph-tier | `bool` | `False` | — | DSPARK compact ragged-verify only. Fill the per-request verify lengths so the total verify-token count reaches the cuda-graph tier the forward is already padded to: round the dp-max scheduled total up to the captured token bucket and let the top-k allocator admit that many real draft tokens (confidence-ordered). This recovers the padding the forward pays for anyway -- both the cuda-graph bucket round-up and the dp cross-rank max -- turning it into extra real verification at the same step time. Off by default; when off the schedule is byte-for-byte unchanged. |
| `speculative_accept_threshold_single` | --speculative-accept-threshold-single | `float` | `1.0` | — | Accept a draft token if its probability in the target model is greater than this threshold. |
| `speculative_accept_threshold_acc` | --speculative-accept-threshold-acc | `float` | `1.0` | — | The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc). |
| `speculative_use_rejection_sampling` | --speculative-use-rejection-sampling | `bool` | `False` | — | Use rejection sampling for speculative decoding (requires topk=1). |
| `speculative_use_block_verification` | --speculative-use-block-verification | `bool` | `False` | — | Use block verification for EAGLE/EAGLE3/NEXTN (requires topk=1) or for sampled DFLASH/DSPARK drafts, on CUDA or ROCm. |
| `speculative_token_map` | --speculative-token-map | `Optional[str]` | `None` | — | The path of the draft model's small vocab table. |
| `speculative_attention_mode` | --speculative-attention-mode (可被解析阶段改写) | `str` | `'prefill'` | prefill, decode | Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'. |
| `speculative_draft_attention_backend` | --speculative-draft-attention-backend (可被解析阶段改写) | `Optional[str]` | `None` | — | Attention backend for speculative decoding drafting. |
| `speculative_dsa_topk_backend` | --speculative-dsa-topk-backend | `str` | `'sgl-kernel'` | sgl-kernel, torch, flashinfer | DSA indexer top-k backend for speculative draft workers. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false. |
| `speculative_draft_kv_cache_dtype` | --speculative-draft-kv-cache-dtype (可被解析阶段改写) | `Optional[str]` | `None` | auto, fp8_e5m2, fp8_e4m3, bf16, bfloat16 | KV cache dtype for the speculative draft model only. The draft pool is allocated with one slot per target token (draft and target share a slot index space), so for a small draft it can still rival the target pool: a 5-layer DFLASH draft costs 10240 bytes/token in bf16. Setting fp8_e4m3 halves the draft pool; the saving shows up as free device memory, so raise --mem-fraction-static to convert it into KV capacity. Default follows --kv-cache-dtype. |
| `speculative_draft_window_size` | --speculative-draft-window-size | `Optional[int]` | `None` | — | Sliding window size for the draft model. Honored by Llama EAGLE-3 (`LlamaForCausalLMEagle3`), DFLASH, and the built-in EAGLE/MTP draft-decode path on the Triton and FlashInfer draft backends; other EAGLE-3 backends (e.g. MLA-based drafters) silently ignore it. For Llama EAGLE-3, the drafter only attends to the most recent N keys (verifier hidden states + its own outputs); the verifier is unaffected. For DFLASH, the draft worker keeps a recent target-token window in its local KV cache (paged backends may retain up to one extra page on the left for alignment). For the built-in EAGLE/MTP draft, each draft-decode step attends to a --speculative-draft-sink-size sink plus the most recent N tokens, leaving the target verify pass unchanged; it is ignored (with a warning) if the draft model has a native sliding window of its own. Default is full attention/context. |
| `speculative_draft_sink_size` | --speculative-draft-sink-size | `Optional[int]` | `None` | — | Number of leading 'attention sink' tokens the draft always attends to, in addition to the --speculative-draft-window-size recent window (StreamingLLM-style). Honored only by the built-in EAGLE/MTP draft-decode path on the Triton and FlashInfer draft backends; the Llama EAGLE-3 and DFLASH windows ignore it. 0/unset => pure recent window. Requires --speculative-draft-window-size. |
| `speculative_moe_runner_backend` | --speculative-moe-runner-backend (可被解析阶段改写) | `Optional[str]` | `None` | auto, deep_gemm, triton, triton_kernel, flashinfer_trtllm, experimental_sgl_trtllm, flashinfer_trtllm_routed, flashinfer_cutlass, flashinfer_mxfp4, flashinfer_cutedsl, flashinfer_megamoe, cutlass, aiter, marlin, humming, experimental_sgl_marlin, lora_cutedsl, lora_triton, lora_marlin, hpc_ops, megamoe, intel_xpu | Choose the runner backend for MoE in speculative decoding. Defaults to --moe-runner-backend, except flashinfer_megamoe defaults the draft to flashinfer_trtllm (draft layers are typically unquantized, which flashinfer_megamoe's global-expert-id dispatch doesn't support). |
| `speculative_moe_a2a_backend` | --speculative-moe-a2a-backend (可被解析阶段改写) | `Optional[str]` | `None` | none, deepep, mooncake, nixl, mori, ascend_fuseep, flashinfer, megamoe, deepep_v2, pplx, ascend_tp | Choose the backend for MoE A2A in speculative decoding. Defaults to --moe-a2a-backend, except flashinfer_megamoe defaults the draft to none (draft layers are typically unquantized, which flashinfer_megamoe's global-expert-id dispatch doesn't support). |
| `speculative_enable_w4a4_mxfp4_megamoe` | --speculative-enable-w4a4-mxfp4-megamoe | `Optional[bool]` | `None` | — | Whether the draft model's MXFP4 MegaMoE layers use the W4A4 mxf4xmxf4 MMA type (see --enable-w4a4-mxfp4-megamoe). Pass --no-speculative-enable-w4a4-mxfp4-megamoe to keep the draft on fp8xfp4 (W4A8) while the target runs W4A4. Same as --enable-w4a4-mxfp4-megamoe if unset. A draft that runs MXFP4 MegaMoE with a different MMA type from the target may allocate an additional MegaMoE symmetric buffer after the KV pool is sized; lower --mem-fraction-static if CUDA-graph capture runs out of memory. |
| `speculative_draft_model_quantization` | --speculative-draft-model-quantization | `Optional[str]` | `None` | awq, fp8, mxfp8, gptq, gptq_marlin, awq_marlin, bitsandbytes, gguf, modelopt, modelopt_fp8, modelopt_fp4, nvfp4_online, modelopt_mixed, petit_nvfp4, w8a8_int8, w8a8_fp8, moe_wna16, w4afp8, mxfp4, auto-round, auto-round-int8, compressed-tensors, modelslim, mxfp_w4a8, quark, quark_int4fp8_moe, quark_mxfp4, mlx_q4, mlx_q8, unquant, humming | The quantization method for speculative model. |
| `_speculative_draft_quantization_explicitly_set` | ---speculative-draft-quantization-explicitly-set (无 CLI) | `Optional[bool]` | `None` | — | Internal provenance used after the public draft quantization inherits the target value. It is a dataclass field so ServerArgs round-trips preserve whether the user explicitly set the draft option; it has no CLI surface. |
| `enable_multi_layer_eagle` | --enable-multi-layer-eagle (可被解析阶段改写) | `bool` | `False` | — | Enable multi-layer Eagle speculative decoding. |
| `speculative_adaptive` | --speculative-adaptive | `bool` | `False` | — | Enable adaptive speculative decoding that dynamically adjusts num_steps based on acceptance rate. |
| `speculative_adaptive_config` | --speculative-adaptive-config | `Optional[str]` | `None` | — | Path to a JSON config file for adaptive speculative decoding tuning knobs. |
| `spec_trace_dir` | --spec-trace-dir | `Optional[str]` | `None` | — | Directory to write decoupled speculative decoding trace files. |
| `speculative_ngram_min_bfs_breadth` | --speculative-ngram-min-bfs-breadth | `int` | `1` | — | The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding. |
| `speculative_ngram_max_bfs_breadth` | --speculative-ngram-max-bfs-breadth | `int` | `10` | — | The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding. |
| `speculative_ngram_match_type` | --speculative-ngram-match-type | `Literal['BFS', 'PROB']` | `'BFS'` | — | The match type for cache tree. |
| `speculative_ngram_max_trie_depth` | --speculative-ngram-max-trie-depth | `int` | `18` | — | The max trie depth for ngram speculative decoding. |
| `speculative_ngram_capacity` | --speculative-ngram-capacity | `int` | `10 * 1000 * 1000` | — | The cache capacity for ngram speculative decoding. |
| `speculative_ngram_external_corpus_path` | --speculative-ngram-external-corpus-path | `Optional[str]` | `None` | — | Path to an external JSONL corpus to pre-load into SAM at startup. Additional corpora can be added at runtime via POST /add_external_corpus. |
| `speculative_ngram_external_sam_budget` | --speculative-ngram-external-sam-budget | `int` | `0` | — | Number of draft nodes reserved for the external SAM subtree in ngram speculative decoding. |
| `speculative_ngram_external_corpus_max_tokens` | --speculative-ngram-external-corpus-max-tokens | `int` | `10000000` | — | Fail startup if the tokenized external ngram corpus exceeds this many tokens. Tune this based on your CPU memory budget. |

### Prefill/Decode 分离

命名空间 `disagg`，声明类 `Disagg`，文件 `python/sglang/srt/arg_groups/fields/disagg.py`。共 30 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `decoupled_spec_bind_endpoint` | --decoupled-spec-bind-endpoint | `Optional[str]` | `None` | — | ZMQ endpoint this engine binds for its inbound channel in decoupled speculative decoding (verifier: result PULL; drafter: control PULL). |
| `decoupled_spec_connect_endpoints` | --decoupled-spec-connect-endpoints | `Optional[List[str]]` | `None` | — | Peer inbound (bind) endpoints to connect to, ordered by peer rank, for decoupled speculative decoding. |
| `decoupled_spec_rank` | --decoupled-spec-rank | `Optional[int]` | `None` | — | This engine's rank within its own role space (verifier-rank or drafter-rank) for decoupled speculative decoding. |
| `decoupled_spec_role` | --decoupled-spec-role | `Literal['null', 'verifier', 'drafter']` | `'null'` | — | Role in decoupled speculative decoding: 'null' disables it, 'verifier' runs the target/verify half, 'drafter' runs the draft half. |
| `disaggregation_mode` | --disaggregation-mode | `Literal['null', 'prefill', 'decode']` | `'null'` | — | Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated |
| `disaggregation_decode_allocation_policy` | --disaggregation-decode-allocation-policy | `Literal['early', 'prefill_complete']` | `'early'` | — | When to reserve per-request decode KV in disaggregated serving. prefill_complete retains KV on prefill until prefill finishes, then admits it on decode. Set the same policy on both roles. Default: early. |
| `disaggregation_transfer_backend` | --disaggregation-transfer-backend | `str` | `'mooncake'` | mooncake, nixl, ascend, fake, mori, mooncake_tcp | The backend for disaggregation transfer. Default is mooncake. |
| `disaggregation_enable_kv_checksum` | --disaggregation-enable-kv-checksum | `bool` | `False` | — | Compute an Adler-32 checksum over each request's KV pages on prefill and verify it on decode. Enable on both prefill and decode engines. A mismatch aborts the request, or raises in CI. With a decode-side radix cache the digest covers only the pages prefill sends. A prefill whose KV layout differs (TP width, PP stage, page size, dtype) is admitted unverified with one warning. Disabled by default. |
| `disaggregation_bootstrap_port` | --disaggregation-bootstrap-port | `int` | `8998` | — | Bootstrap server port on the prefill server. Default is 8998. |
| `disaggregation_ib_device` | --disaggregation-ib-device | `Optional[str]` | `None` | — | The InfiniBand devices for disaggregation transfer. Supports a single device (e.g., --disaggregation-ib-device mlx5_0), a shared comma-separated list (e.g., --disaggregation-ib-device mlx5_0,mlx5_1), a per-GPU JSON mapping (e.g., --disaggregation-ib-device '{"0": "mlx5_0,mlx5_1", "1": "mlx5_2"}'), or a path to a JSON file containing that mapping. Supported by Mooncake and NIXL UCX. Mapping keys are process-visible GPU IDs (after CUDA_VISIBLE_DEVICES); use HCA names without port suffixes for NIXL UCX. For NIXL UCX, a non-empty UCX_NET_DEVICES takes precedence over this option, including per-GPU mappings. Default is None, which preserves backend device selection. |
| `disaggregation_decode_enable_radix_cache` | --disaggregation-decode-enable-radix-cache | `bool` | `False` | — | Enable radix cache on decode server (PD mode). Caches KV prefixes to avoid redundant transfers. Incompatible with --enable-hisparse, --disaggregation-transfer-backend fake, and speculative algorithms other than EAGLE, EAGLE3, and DSPARK. |
| `disaggregation_decode_enable_offload_kvcache` | --disaggregation-decode-enable-offload-kvcache | `bool` | `False` | — | Enable async KV cache offloading on decode server (PD mode). |
| `disaggregation_decode_retraction_backup` | --disaggregation-decode-retraction-backup | `Optional[str]` | `None` | cpu_tensor, host_pool, none | Storage backend for KV preserved across PD decode retraction. 'cpu_tensor' uses per-request CPU tensors. 'host_pool' uses a reserved HiCache pool and does not fall back on exhaustion. 'none' keeps no backup: a retracted request is aborted with 503 for the client to retry. If omitted, the backend is inferred from the decode KV pool. |
| `num_reserved_decode_tokens` | --num-reserved-decode-tokens | `int` | `512` | — | Number of decode tokens that will have memory reserved when adding new request to the running batch. |
| `disaggregation_decode_extra_slots` | --disaggregation-decode-extra-slots | `Optional[int]` | `None` | — | Number of extra decode req_to_token slots pre-allocated for in-transfer requests (PD mode). If unset, defaults to 0 (or 2x the per-worker running batch for small batches). |
| `disaggregation_decode_polling_interval` | --disaggregation-decode-polling-interval | `int` | `1` | — | The interval to poll requests in decode server. Can be set to >1 to reduce the overhead of this. |
| `enable_pd_role_switch` | --enable-pd-role-switch | `bool` | `False` | — | Allow runtime prefill<->decode role switch via /pd_role_switch (PD mode). |
| `optimistic_prefill_attempts` | --optimistic-prefill-attempts | `int` | `0` | — | Number of optimistic prefill forward passes that skip the bootstrap wait. |
| `encoder_only` | --encoder-only | `bool` | `False` | — | For MLLM with an encoder, launch an encoder-only server |
| `language_only` | --language-only | `bool` | `False` | — | For VLM, load weights for the language model only. |
| `language_model_only` | --language-model-only | `bool` | `False` | — | Skip the multimodal encoder entirely: its weights are never loaded and the tower is never built, freeing that GPU memory for KV cache. Multimodal requests are rejected. Unlike --language-only this is a standalone mode, not part of encoder/decoder disaggregation. |
| `encoder_transfer_backend` | --encoder-transfer-backend | `str` | `'auto'` | auto, zmq_to_scheduler, zmq_to_tokenizer, mooncake | The backend for encoder disaggregation transfer. Auto selects a model- and TP-aware backend. |
| `encoder_urls` | --encoder-urls | `List[str]` | `msgspec.field(default_factory=list)` | — | List of encoder server urls. |
| `encoder_bootstrap_port` | --encoder-bootstrap-port | `int` | `8997` | — | Port for the EncoderBootstrapServer that runs in the language-only tokenizer manager process. Encoders register here, and language-only receivers fetch the current URL list from here. |
| `encoder_register_urls` | --encoder-register-urls | `List[str]` | `msgspec.field(default_factory=list)` | — | One or more EncoderBootstrapServer URLs to register this encoder with on startup, for dynamic encoder discovery. Example: --encoder-register-urls http://prefill0:8997 http://prefill1:8997. Used with --encoder-only servers. |
| `enable_adaptive_dispatch_to_encoder` | --enable-adaptive-dispatch-to-encoder | `bool` | `False` | — | When enabled, adaptively dispatch: multi-image requests go to encoder in language_only epd mode, single-image requests are processed locally. |
| `enable_pdmux` | --enable-pdmux | `bool` | `False` | — | Enable PD-Multiplexing, PD running on greenctx stream. |
| `pdmux_config_path` | --pdmux-config-path | `Optional[str]` | `None` | — | The path of the PD-Multiplexing config file. |
| `sm_group_num` | --sm-group-num | `int` | `8` | — | Number of sm partition groups. |
| `disaggregation_decode_host_receive_threshold` | --disaggregation-decode-host-receive-threshold | `float` | `1.0` | — | Device token usage fraction at which incoming KV is received in the decode retraction host pool, excluding evictable cache pages. Range [0, 1]; 1 disables host receive (default), 0 always uses host receive. Size with --hicache-size or --hicache-ratio; requires dense NHD MHA buffers (including separate paged SWA pools) and a transfer backend that supports host destinations. No built-in backend currently supports this. |

### LoRA

命名空间 `lora`，声明类 `Lora`，文件 `python/sglang/srt/arg_groups/fields/lora.py`。共 14 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `enable_lora` | --enable-lora | `Optional[bool]` | `None` | — | Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility. |
| `enable_lora_overlap_loading` | --enable-lora-overlap-loading | `Optional[bool]` | `None` | — | Enable asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters. |
| `max_lora_rank` | --max-lora-rank | `Optional[int]` | `None` | — | The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. |
| `lora_target_modules` | --lora-target-modules | `Optional[Union[set[str], List[str]]]` | `None` | SUPPORTED_LORA_TARGET_MODULES + [LORA_TARGET_ALL_MODULES] | The union set of all target modules where LoRA should be applied. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. If 'all' is specified, all supported modules will be targeted. |
| `lora_paths` | --lora-paths | `Optional[Union[dict[str, str], List[dict[str, str]], List[str], List[LoRARef]]]` | `None` | — | The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> \| <NAME>=<PATH> \| JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool} |
| `max_loaded_loras` | --max-loaded-loras | `Optional[int]` | `None` | — | If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `--max-loras-per-batch`. |
| `max_loras_per_batch` | --max-loras-per-batch | `int` | `8` | — | Maximum number of adapters for a running batch, include base-only request. |
| `lora_eviction_policy` | --lora-eviction-policy | `str` | `'lru'` | lru, fifo | LoRA adapter eviction policy when memory pool is full. 'lru': Least Recently Used (default, better cache efficiency). 'fifo': First-In-First-Out. |
| `lora_backend` | --lora-backend | `str` | `'csgmv'` | triton, csgmv, ascend, torch_native, triton_v2 | Choose the kernel backend for multi-LoRA serving. |
| `max_lora_chunk_size` | --max-lora-chunk-size | `Optional[int]` | `16` | 16, 32, 64, 128 | Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is 'csgmv'. Choosing a larger value might improve performance. |
| `experts_shared_outer_loras` | --experts-shared-outer-loras | `Optional[bool]` | `None` | — | Force shared outer LoRA mode for MoE models. When set, w1/w3 lora_A and w2 lora_B are shared across experts (expert_dim=1). Use --no-experts-shared-outer-loras to force disable. By default this is auto-detected from adapter weights. |
| `lora_use_virtual_experts` | --lora-use-virtual-experts | `bool` | `False` | — | Enable virtual expert computation for MoE models. When set, the model will use virtual expert computation. |
| `lora_strict_loading` | --lora-strict-loading | `bool` | `False` | — | Enable strict loading for LoRA adapters. When set, mismatched or missing keys in the adapter weights will raise an error. |
| `lora_drain_wait_threshold` | --lora-drain-wait-threshold | `float` | `0.0` | — | When any LoRA adapter request waits longer than this threshold (in seconds), the scheduler will selectively drain one running adapter to make room. This mitigates extreme tail latency under high or skewed workloads by preventing a small set of adapters from monopolizing batch slots. Set to 0 to disable draining (default). |

### 多模态

命名空间 `mm`，声明类 `Mm`，文件 `python/sglang/srt/arg_groups/fields/mm.py`。共 18 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `enable_multimodal` | --enable-multimodal | `Optional[bool]` | `None` | — | Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen |
| `mm_attention_backend` | --mm-attention-backend | `Optional[str]` | `None` | sdpa, fa3, fa4, triton_attn, ascend_attn, aiter_attn, flashinfer_cudnn, amx_attn, xpu_attn | Set multimodal attention backend. |
| `enable_broadcast_mm_inputs_process` | --enable-broadcast-mm-inputs-process | `bool` | `False` | — | Enable broadcast mm-inputs process in scheduler. |
| `enable_prefix_mm_cache` | --enable-prefix-mm-cache | `bool` | `False` | — | Enable prefix multimodal cache. Currently only supports mm-only. |
| `mm_enable_dp_encoder` | --mm-enable-dp-encoder (可被解析阶段改写) | `Optional[bool]` | `None` | — | Enabling data parallelism for mm encoder. The dp size will be set to the tp size automatically. |
| `mm_process_config` | --mm-process-config | `Optional[Dict[str, Any]]` | `None` | — | Multimodal preprocessing config, a json config contains keys: `image`, `video`, `audio` |
| `mm_processor_worker_num` | --mm-processor-worker-num | `int` | `0` | — | Number of threads for multimodal processor calls. 0 selects the model-specific default. Only processors with isolated-worker support can use more than one thread. |
| `mm_io_worker_num` | --mm-io-worker-num | `int` | `0` | — | Number of threads for multimodal data loading and decoding. 0 selects the model-specific default. SGLANG_IO_WORKERS remains supported as an environment override when this argument is 0. |
| `allowed_media_domains` | --allowed-media-domains | `List[str]` | `msgspec.field(default_factory=list)` | — | Restrict client-supplied HTTP(S) image, video, and audio URLs to these exact hostnames. Redirect destinations are checked against the same allowlist. When unset, remote media from any domain is allowed. |
| `media_url_max_file_size_mb` | --media-url-max-file-size-mb | `int` | `64` | — | Maximum size in MiB for one client-supplied remote media download. The limit is enforced while streaming; set to 0 to disable it. |
| `mm_preprocess_cache_size_mb` | --mm-preprocess-cache-size-mb (可被解析阶段改写) | `Optional[int]` | `None` | — | CPU memory budget for content-addressed multimodal preprocessing artifacts. Unset selects a model-specific default (256 MiB for Kimi-K3); 0 disables the cache. The budget is divided across tokenizer workers and does not reserve GPU memory. |
| `trust_mm_content_hashes` | --trust-mm-content-hashes | `bool` | `False` | — | Trust caller-provided multimodal SHA-256 content hashes. This can skip reading media on a hot metadata-cache hit; only enable it when the caller guarantees that hashes identify immutable media bytes. |
| `limit_mm_data_per_request` | --limit-mm-data-per-request | `Optional[Union[str, Dict[str, int]]]` | `None` | — | Limit the number of multimodal inputs per request. e.g. '{"image": 1, "video": 1, "audio": 1}' |
| `enable_mm_global_cache` | --enable-mm-global-cache | `bool` | `False` | — | Enable global multimodal embedding cache to skip redundant ViT inference. |
| `image_processor_backend` | --image-processor-backend | `Literal['auto', 'torchvision', 'pil']` | `'auto'` | — | Image processor backend. 'auto' lets Transformers select the best available backend. |
| `mm_global_cache_backend` | --mm-global-cache-backend | `str` | `'mooncake'` | mooncake | Storage backend for the multimodal global embedding cache. Used when --enable-mm-global-cache is set. |
| `disable_fast_image_processor` | --disable-fast-image-processor | `bool` | `False` | — | Deprecated. Use --image-processor-backend=pil instead. |
| `mm_feature_transport` | --mm-feature-transport (可被解析阶段改写) | `Optional[Literal['cpu', 'cuda_ipc', 'cuda_vmm']]` | `None` | — | Transport multimodal features through CPU memory, a bounded CUDA IPC pool, or a bounded CUDA VMM pool. Unset uses cpu except for validated multi-node GB200/GB300 MNNVL models, which use cuda_vmm when an IMEX channel is available. Select cuda_ipc explicitly for single-node GPU transport. GPU transports reserve SGLANG_MM_FEATURE_CACHE_MB (default 1024 MiB) on the base GPU and fall back to CPU transport when the pool is full. |

### 设备与进程启动

命名空间 `device`，声明类 `Device`，文件 `python/sglang/srt/arg_groups/fields/device.py`。共 13 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `device` | --device | `Optional[str]` | `None` | — | The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu', 'musa'). Defaults to auto-detection if not specified. |
| `base_gpu_id` | --base-gpu-id | `int` | `0` | — | The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine. |
| `gpu_id_step` | --gpu-id-step | `int` | `1` | — | The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,... |
| `gpu_id` | --gpu-id (派生，无 CLI) | `Derived` | `（发布时计算）` | — | Which device this process runs on. Nobody types it and nothing computes it from the configuration: the parent decides -- reindexing narrows the visible devices before the spawn, and Ray allocates from its own pool -- so the entry states it in the bundle it hands `publish`. `None` is an answer rather than an absence: most roles run on no device at all. |
| `random_seed` | --random-seed | `Optional[int]` | `None` | — | The random seed. |
| `mlx_enable_sampling` | --mlx-enable-sampling | `bool` | `False` | — | MLX backend only: sample decode tokens (temperature / top-k / top-p / min-p) instead of greedy argmax. Sampling runs inside the lazy MLX graph, so it works with the overlap scheduler; first tokens from prefill/extend are sampled too. Greedy requests keep exact argmax behavior. Also enables on the MLX path: grammar vocab masks and custom logit processors (these break decode chaining per step; custom processors run on pure-decode steps only), logit_bias, output logprobs (sampled token / top-k / token_ids; prompt input logprobs are not computed), NaN sanitization (SGLANG_SANITIZE_NAN_LOGITS), and per-request sampling_seed under --enable-deterministic-inference (deterministic within MLX only). Penalties are not applied. |
| `watchdog_timeout` | --watchdog-timeout | `float` | `300` | — | Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging. |
| `soft_watchdog_timeout` | --soft-watchdog-timeout | `Optional[float]` | `None` | — | Set soft watchdog timeout in seconds. If a forward batch takes longer than this, the server will dump information for debugging. |
| `sleep_on_idle` | --sleep-on-idle | `bool` | `False` | — | Reduce CPU usage when sglang is idle. |
| `use_ray` | --use-ray | `bool` | `False` | — | Use Ray actors for scheduler process management. |
| `custom_sigquit_handler` | --custom-sigquit-handler | `Optional[Callable]` | `None` | — | — |
| `numa_node` | --numa-node | `Optional[List[int]]` | `None` | — | Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. If unset, will be automatically detected on NUMA systems. |
| `gc_threshold` | --gc-threshold | `Optional[List[int]]` | `None` | — | Set the garbage collection thresholds (the collection frequency). Accepts 1 to 3 integers. |

### 可观测性

命名空间 `observability`，声明类 `Observability`，文件 `python/sglang/srt/arg_groups/fields/observability.py`。共 45 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `log_level` | --log-level | `str` | `'info'` | — | The logging level of all loggers. |
| `log_level_http` | --log-level-http | `Optional[str]` | `None` | — | The logging level of HTTP server. If not set, reuse --log-level by default. |
| `log_requests` | --log-requests | `bool` | `False` | — | Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level |
| `log_requests_level` | --log-requests-level | `int` | `2` | 0, 1, 2, 3 | 0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output. |
| `log_requests_format` | --log-requests-format | `str` | `'text'` | text, json | Format for request logging: 'text' (human-readable) or 'json' (structured) |
| `log_requests_target` | --log-requests-target | `Optional[List[str]]` | `None` | — | Target(s) for request logging: 'stdout' and/or directory path(s) for file output. Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. |
| `uvicorn_access_log_exclude_prefixes` | --uvicorn-access-log-exclude-prefixes | `List[str]` | `msgspec.field(default_factory=list)` | — | Exclude uvicorn access logs whose request path starts with any of these prefixes. Defaults to empty (disabled). Example: --uvicorn-access-log-exclude-prefixes /metrics /health |
| `crash_dump_folder` | --crash-dump-folder | `Optional[str]` | `None` | — | Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled. |
| `show_time_cost` | --show-time-cost | `bool` | `False` | — | Show time cost of custom marks. |
| `enable_metrics` | --enable-metrics | `bool` | `False` | — | Enable log prometheus metrics. |
| `smg_http_sidecar_port` | --smg-http-sidecar-port (--grpc-http-sidecar-port) | `Optional[int]` | `None` | — | Port for the HTTP sidecar server in legacy SMG gRPC mode (--smg-grpc-mode). Serves Prometheus metrics and profiling endpoints. Defaults to --port + 1. Not used in HTTP mode. |
| `enable_mfu_metrics` | --enable-mfu-metrics | `bool` | `False` | — | Enable estimated MFU-related prometheus metrics. |
| `enable_metrics_for_all_schedulers` | --enable-metrics-for-all-schedulers | `bool` | `False` | — | Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0. |
| `load_snapshot_publish_interval` | --load-snapshot-publish-interval | `int` | `15` | — | Publish load snapshot to shared memory every N decode iterations. Prefill and idle always publish immediately. |
| `tokenizer_metrics_custom_labels_header` | --tokenizer-metrics-custom-labels-header | `str` | `'x-custom-labels'` | — | Specify the HTTP header for passing custom labels for tokenizer metrics. |
| `tokenizer_metrics_allowed_custom_labels` | --tokenizer-metrics-allowed-custom-labels | `Optional[List[str]]` | `None` | — | The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set. |
| `extra_metric_labels` | --extra-metric-labels | `Optional[Dict[str, str]]` | `None` | — | The custom labels for metrics. e.g. '{"label1": "value1", "label2": "value2"}' |
| `bucket_time_to_first_token` | --bucket-time-to-first-token | `Optional[List[float]]` | `None` | — | The buckets of time to first token, specified as a list of floats. |
| `bucket_inter_token_latency` | --bucket-inter-token-latency | `Optional[List[float]]` | `None` | — | The buckets of inter-token latency, specified as a list of floats. |
| `bucket_e2e_request_latency` | --bucket-e2e-request-latency | `Optional[List[float]]` | `None` | — | The buckets of end-to-end request latency, specified as a list of floats. |
| `prompt_tokens_buckets` | --prompt-tokens-buckets | `Optional[List[str]]` | `None` | — | The buckets rule of prompt tokens. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> <value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500'). |
| `generation_tokens_buckets` | --generation-tokens-buckets | `Optional[List[str]]` | `None` | — | The buckets rule for generation tokens histogram. Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets [984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> <value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500'). |
| `gc_warning_threshold_secs` | --gc-warning-threshold-secs | `float` | `0.0` | — | The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable. |
| `decode_log_interval` | --decode-log-interval | `int` | `40` | — | The log and metrics reporting interval (in decode iterations) for decode batches. |
| `enable_request_time_stats_logging` | --enable-request-time-stats-logging | `bool` | `False` | — | Enable per request time stats logging |
| `kv_events_config` | --kv-events-config | `Optional[str]` | `None` | — | Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used. Runtime-load publishing for load-aware routers is a separate opt-in; see --load-publish-endpoint. Set replay_endpoint (e.g. tcp://*:5558) to let routers re-fetch dropped batches; /server_info advertises its port. |
| `load_publish_endpoint` | --load-publish-endpoint | `Optional[str]` | `None` | — | Opt in to the runtime-load PUB socket that load-aware routers subscribe to. Off by default (unset or 'off'). Use 'auto' to reserve the dp_size ports packed after the --kv-events-config range, or a wildcard-host TCP address (e.g. tcp://*:6000) to place it explicitly; rank r binds port+r and /server_info advertises the base under the kv_events block. Requires --kv-events-config to describe a publisher (routers discover the base through /server_info); startup fails if this is set without one, is not bindable, or overlaps the KV range. Note: 'auto' reserves 2*dp_size ports from the KV base — space co-hosted engines accordingly. The router-facing update cadence follows --load-snapshot-publish-interval (shared to avoid double-collecting the snapshot), so a large value there also staleness-caps this feed. |
| `enable_forward_pass_metrics` | --enable-forward-pass-metrics | `bool` | `False` | — | Enable per-iteration forward pass metrics via ZMQ IPC. External consumers (e.g. Dynamo planner) subscribe to the IPC endpoint exposed in server_args.forward_pass_metrics_ipc_name. |
| `forward_pass_metrics_worker_id` | --forward-pass-metrics-worker-id | `str` | `''` | — | — |
| `forward_pass_metrics_ipc_name` | --forward-pass-metrics-ipc-name | `Optional[str]` | `None` | — | — |
| `enable_trace` | --enable-trace | `bool` | `False` | — | Enable opentelemetry trace |
| `trace_modules` | --trace-modules | `str` | `'request'` | — | Select the components to trace. Available options are 'request' and 'mooncake'. Format: <module1 name>,<module2 name>,... |
| `otlp_traces_endpoint` | --otlp-traces-endpoint | `str` | `'localhost:4317'` | — | Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port> |
| `otlp_service_name` | --otlp-service-name | `Optional[str]` | `None` | — | Service name for OTLP traces (displayed as 'service.name' in trace backends). If unset, falls back to the OTEL_SERVICE_NAME env var, then to 'sglang'. |
| `export_metrics_to_file` | --export-metrics-to-file | `bool` | `False` | — | Export performance metrics for each request to local file (e.g. for forwarding to external systems). |
| `export_metrics_to_file_dir` | --export-metrics-to-file-dir | `Optional[str]` | `None` | — | Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled). |
| `stat_loggers` | --stat-loggers | `Optional[Dict[str, type]]` | `None` | — | Class-level DI for the five *MetricsCollector classes. Maps collector role (one of: "scheduler", "tokenizer", "storage", "radix_cache", "expert_dispatch") to a subclass of the matching base collector. The five instantiation sites read from this map and fall back to the base class. Class-object only (no CLI surface) since this exists for embedded use cases that pass a Python class directly. Default None preserves existing behavior. |
| `kv_canary` | --kv-canary | `str` | `'none'` | none, log, raise | KV cache canary mode. 'none' disables the canary (default). 'log' prints them while the server keeps running (production-safe). 'raise' fails the server on the first detected mismatch (CI lane). |
| `kv_canary_real_data` | --kv-canary-real-data | `str` | `'none'` | — | — |
| `kv_canary_sweep_interval` | --kv-canary-sweep-interval | `int` | `0` | — | Every N forward steps, run a full-pool sweep. |
| `debug_tensor_dump_output_folder` | --debug-tensor-dump-output-folder | `Optional[str]` | `None` | — | The output folder for dumping tensors. In Eagle mode, tensor outputs from draft and target models are stored in separate subdirectories ('draft' and 'target'). |
| `debug_tensor_dump_layers` | --debug-tensor-dump-layers | `Optional[List[int]]` | `None` | — | The layer ids to dump. Dump all layers if not specified. |
| `debug_tensor_dump_input_file` | --debug-tensor-dump-input-file | `Optional[str]` | `None` | — | The input filename for dumping tensors |
| `forward_hooks` | --forward-hooks | `Optional[List[dict[str, Any]]]` | `None` | — | JSON-formatted forward hook specifications to attach to the model. |
| `msprobe_dump_config` | --msprobe-dump-config | `Optional[str]` | `None` | — | The path of the JSON configuration file for msProbe. If specified, enables msProbe dump. |

### 执行特性开关

命名空间 `exec.features`，声明类 `ExecFeatures`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 27 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `enable_fp32_lm_head` | --enable-fp32-lm-head | `bool` | `False` | — | If set, the LM head outputs (logits) are in FP32. |
| `enable_tf32_matmul` | --enable-tf32-matmul (可被解析阶段改写) | `Optional[bool]` | `None` | — | Enable float32 matmuls to use TensorFloat32 precision for better performance (via torch.set_float32_matmul_precision). CUDA only. Defaults to the model's preference, otherwise disabled. |
| `enable_memory_saver` | --enable-memory-saver | `bool` | `False` | — | Allow saving memory using release_memory_occupation and resume_memory_occupation |
| `enable_weights_cpu_backup` | --enable-weights-cpu-backup | `bool` | `False` | — | Save model weights (both main model and draft model, if any) to CPU memory during release_weights_occupation and resume_weights_occupation |
| `enable_draft_weights_cpu_backup` | --enable-draft-weights-cpu-backup | `bool` | `False` | — | Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation |
| `enable_custom_logit_processor` | --enable-custom-logit-processor | `bool` | `False` | — | Enable users to pass custom logit processors to the server (disabled by default for security) |
| `enable_watermark` | --enable-watermark | `bool` | `False` | — | Enable Aaronson-Gumbel text watermarking. |
| `watermark_config` | --watermark-config | `Optional[str]` | `None` | — | Watermark settings as a JSON file path or an inline JSON object with optional key, key_b, context_window, mixing_probability, max_probability, default_enabled, and enforce_all fields. |
| `watermark_key` | --watermark-key (无 CLI; 可被解析阶段改写) | `Optional[str]` | `None` | — | Resolved from --watermark-config. |
| `watermark_key_b` | --watermark-key-b (无 CLI; 可被解析阶段改写) | `Optional[str]` | `None` | — | Resolved from --watermark-config. |
| `watermark_context_window` | --watermark-context-window (无 CLI; 可被解析阶段改写) | `int` | `4` | — | Resolved from --watermark-config. |
| `watermark_mixing_probability` | --watermark-mixing-probability (无 CLI; 可被解析阶段改写) | `float` | `0.5` | — | Resolved from --watermark-config. |
| `watermark_max_probability` | --watermark-max-probability (无 CLI; 可被解析阶段改写) | `float` | `1.0` | — | Resolved from --watermark-config. |
| `watermark_default_enabled` | --watermark-default-enabled (无 CLI; 可被解析阶段改写) | `bool` | `False` | — | Resolved from --watermark-config. |
| `watermark_enforce_all` | --watermark-enforce-all (无 CLI; 可被解析阶段改写) | `bool` | `False` | — | Resolved from --watermark-config. |
| `enable_return_hidden_states` | --enable-return-hidden-states | `bool` | `False` | — | Enable returning full hidden states with responses. Equivalent to `--return-hidden-states-mode full`. |
| `return_hidden_states_mode` | --return-hidden-states-mode | `Optional[str]` | `None` | last, full | Set the maximum hidden-state return mode supported by the server. `last` allows requests with return_hidden_states=False or `last`; `full` also allows return_hidden_states=True. |
| `enable_return_aux_hidden_states` | --enable-return-aux-hidden-states | `bool` | `False` | — | Capture auxiliary hidden states on the target model without running a speculative draft. --speculative-algorithm then only names the capture convention (EAGLE3, DFLASH or DSPARK) and is reset to None. Requests with return_hidden_states receive the concatenated aux hidden states in place of the last-layer hidden states. Sets --return-hidden-states-mode to full when unset. Requires pp_size == 1. |
| `aux_hidden_state_capture` | --aux-hidden-state-capture (无 CLI; 可被解析阶段改写) | `Optional[str]` | `None` | — | Resolved from --enable-return-aux-hidden-states and --speculative-algorithm. |
| `return_aux_hidden_state_layer_ids` | --return-aux-hidden-state-layer-ids | `Optional[List[int]]` | `None` | — | Target-model layer ids for the selected capture convention. These may differ from draft-checkpoint layer ids. Required for DFLASH and DSPARK; EAGLE3 defaults to the model's EAGLE3 layers. |
| `enable_return_routed_experts` | --enable-return-routed-experts | `bool` | `False` | — | Enable returning routed experts of each layer with responses. |
| `enable_return_indexer_topk` | --enable-return-indexer-topk | `bool` | `False` | — | Enable returning indexer topk indices of layers with indexer with responses. |
| `enable_encoder_swa_bounded_replay` | --enable-encoder-swa-bounded-replay | `bool` | `False` | — | DeepSeek-V4.1 encoder SWA bounded replay: cache Main KV and Indexer keys only, rebuild request-owned SWA windows on prefix hits. Experimental; CUDA and gfx950. |
| `enable_decoder_swa_bounded_replay` | --enable-decoder-swa-bounded-replay (可被解析阶段改写) | `Optional[bool]` | `None` | — | Enable DeepSeek-V4.1 decoder SWA bounded replay. Defaults on for supported configurations. Not numerically equivalent to full prefill; requests for prompt logprobs compute every prompt row. Use --no-enable-decoder-swa-bounded-replay to retain full prefill, e.g. for RL rollouts whose logprobs must match the trainer. |
| `sampling_mask_max_tokens` | --sampling-mask-max-tokens | `int` | `4096` | — | The maximum number of token IDs in a returned sampling mask. Requests are aborted if their realized sampling support exceeds this limit. Use the same value on disaggregated prefill and decode nodes; clients should set top_k below the limit to leave headroom for cutoff ties. |
| `disable_outlines_disk_cache` | --disable-outlines-disk-cache | `bool` | `False` | — | Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency. |
| `enable_mis` | --enable-mis | `bool` | `False` | — | Enable Multi-Item Scoring optimization. Combines query and multiple items into a single sequence for efficient batch processing. Requires --attention-backend flashinfer; auto-disables CUDA graph, radix cache, and chunked prefill. |

### Kernel、Attention、GEMM

命名空间 `exec.kernel`，声明类 `ExecKernel`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 29 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `attention_backend` | --attention-backend (可被解析阶段改写) | `Optional[str]` | `None` | triton, torch_native, flex_attention, dsa, nsa, qsa, dsv4, compressed, fa3, fa4, flashinfer, llada2_cfg_flashinfer, flashmla, trtllm_mla, cutedsl_mla, tokenspeed_mla, trtllm_mha, hpc_ops, minicpm_flashattn, minicpm_flashinfer, aiter, wave, intel_amx, ascend, intel_xpu | Choose the kernels for attention layers. |
| `decode_attention_backend` | --decode-attention-backend (可被解析阶段改写) | `Optional[str]` | `None` | triton, torch_native, flex_attention, dsa, nsa, qsa, dsv4, compressed, fa3, fa4, flashinfer, llada2_cfg_flashinfer, flashmla, trtllm_mla, cutedsl_mla, tokenspeed_mla, trtllm_mha, hpc_ops, minicpm_flashattn, minicpm_flashinfer, aiter, wave, intel_amx, ascend, intel_xpu | Choose the kernels for decode attention layers (have priority over --attention-backend). |
| `enable_lean_attention` | --enable-lean-attention | `Optional[bool]` | `None` | — | Enable Lean (Work-Centric) Attention decode kernel for long-context serving. When None (default), uses auto-gate that activates Lean for long contexts and falls back to standard kernel for short contexts. Set to True to force enable, False to force disable. |
| `enable_nvidia_megakernel` | --enable-nvidia-megakernel | `bool` | `False` | — | Enable the NVIDIA megakernel from the sgl-nvidia-mega package. Supported decode and verify steps use its fused kernels; other steps use the standard forward. An unsupported startup configuration raises an error listing the reasons. Experimental feature. |
| `enable_amd_monokernel` | --enable-amd-monokernel | `bool` | `False` | — | Enable the AMD monokernels from the sgl-amd-mono package. Supported decode and verify steps use persistent kernels; other steps use the standard forward. An unsupported startup configuration raises an error listing the reasons. Experimental feature. |
| `prefill_attention_backend` | --prefill-attention-backend (可被解析阶段改写) | `Optional[str]` | `None` | triton, torch_native, flex_attention, dsa, nsa, qsa, dsv4, compressed, fa3, fa4, flashinfer, llada2_cfg_flashinfer, flashmla, trtllm_mla, cutedsl_mla, tokenspeed_mla, trtllm_mha, hpc_ops, minicpm_flashattn, minicpm_flashinfer, aiter, wave, intel_amx, ascend, intel_xpu | Choose the kernels for prefill attention layers (have priority over --attention-backend). |
| `prefill_kv_cache_dequant_dtype` | --prefill-kv-cache-dequant-dtype (可被解析阶段改写) | `str` | `'auto'` | auto, nvfp4, fp8_e4m3 | Online dequantization dtype used by prefill attention when --kv-cache-dtype=nvfp4. 'nvfp4' reads the packed cache directly without additional dequantization; 'fp8_e4m3' dequantizes it into a temporary FP8 workspace. This does not change the stored KV-cache dtype. 'auto' selects NVFP4 without additional dequantization on SM100 and FP8 E4M3 otherwise. |
| `sampling_backend` | --sampling-backend (无 CLI; 可被解析阶段改写) | `Optional[str]` | `None` | — | Choose the kernels for sampling layers. |
| `grammar_backend` | --grammar-backend | `Optional[str]` | `None` | xgrammar, outlines, llguidance, none | Choose the backend for grammar-guided decoding. |
| `fp8_gemm_runner_backend` | --fp8-gemm-backend (可被解析阶段改写) | `str` | `'auto'` | auto, deep_gemm, flashinfer_trtllm, flashinfer_cutlass, flashinfer_deepgemm, flashinfer_cutedsl, cutlass, triton, aiter | Choose the runner backend for Blockwise FP8 GEMM operations. Options: 'auto' (default, auto-selects based on hardware; MXFP8 dense picks flashinfer_cutedsl on SM100/SM103 and FlashInfer CUTLASS on other supported Blackwell GPUs), 'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), 'flashinfer_trtllm' (optimal for Blackwell and low-latency), 'flashinfer_cutlass' (FlashInfer CUTLASS groupwise FP8 GEMM), 'flashinfer_cutedsl' (FlashInfer CuTe DSL MXFP8 GEMM on SM100/SM103), 'flashinfer_deepgemm' (Hopper SM90 only; uses swapAB optimization for small M dimensions in decoding), 'cutlass' (optimal for SM120 GPUs), 'triton' (fallback, widely compatible), 'aiter' (ROCm only). |
| `fp4_gemm_runner_backend` | --fp4-gemm-backend (可被解析阶段改写) | `str` | `'auto'` | auto, flashinfer_cudnn, flashinfer_cutedsl, flashinfer_cutlass, flashinfer_trtllm, marlin | Choose the runner backend for NVFP4 GEMM operations. Options: 'auto' (default; selects flashinfer_cutedsl on SM100, marlin on SM80-SM90, flashinfer_cutlass otherwise (including SM120)), 'flashinfer_cutlass' (FlashInfer CUTLASS backend), 'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), 'flashinfer_cutedsl' (FlashInfer CuTe DSL backend), 'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling), 'marlin' (weight-only W4A16 fallback for SM80+). |
| `bf16_gemm_backend` | --bf16-gemm-backend | `str` | `'auto'` | auto, cutedsl, gemv, torch | Choose the backend for unquantized BF16 GEMM operations. Options: 'auto' (default; selects 'cutedsl' on SM10x GPUs, except deterministic inference selects 'torch'; otherwise uses cuBLAS via torch.nn.functional.linear), 'cutedsl' (SGLang JIT CuTe DSL TGV BF16 GEMM on SM10x; dispatches between the allowlisted low-M Split-K kernel, the CuTe DSL kernel, and cuBLAS; set SGLANG_ENABLE_BF16_SPLITK_GEMM=0 to disable Split-K), 'gemv', 'torch' (always uses cuBLAS via torch.nn.functional.linear). |
| `dsa_prefill_backend` | --dsa-prefill-backend (可被解析阶段改写) | `Optional[str]` | `None` | flashmla_sparse, flashmla_sparse_q8, flashmla_kv, flashmla_auto, flashinfer_sparse_mla, triton_sparse_mla, fa3, tilelang, triton, aiter, trtllm, intel_xpu | DSA (DeepSeek Sparse Attention) prefill backend. If not specified, auto-detects based on hardware and kv_cache_dtype. |
| `dsv4_prefill_backend` | --dsv4-prefill-backend | `str` | `'auto'` | auto, flashmla_sparse, flashmla_sparse_q8 | DeepSeek-V4 sparse prefill backend. 'auto' and 'flashmla_sparse' use the existing BF16 sparse prefill path; 'flashmla_sparse_q8' enables the Q8KV8 sparse prefill path. |
| `dsa_triton_union` | --dsa-triton-union | `int` | `0` | 0, 2, 4 | Union group size for --dsa-prefill-backend triton_sparse_mla: 0 disables, 2 or 4 make G adjacent query tokens share one gathered index set (mathematically exact; an ownership mask restores each token's own softmax support). |
| `dsa_decode_backend` | --dsa-decode-backend (可被解析阶段改写) | `Optional[str]` | `None` | flashmla_sparse, flashmla_sparse_q8, flashmla_kv, flashmla_auto, flashinfer_sparse_mla, fa3, tilelang, triton, aiter, trtllm, intel_xpu | DSA (DeepSeek Sparse Attention) decode backend. If not specified, auto-detects based on hardware and kv_cache_dtype. |
| `dsv4_attn_backend` | --dsv4-attn-backend (可被解析阶段改写) | `str` | `'auto'` | auto, flashmla, flashmla_mega, trtllm | DeepSeek V4 attention backend. 'auto' (default) resolves to 'flashmla' where 'flashmla_mega' is unsupported or, outside PD disaggregation, where attention is tensor-parallel across ranks (attention TP > 1), and to 'flashmla_mega' otherwise. 'trtllm' (opt-in, SM100/SM103 with FP8 KV cache) switches the SWA/compressed KV pools to a uniform 512-dim FP8 layout and runs decode and sparse prefill through the flashinfer trtllm-gen sparse MLA kernel. 'flashmla_mega' (DeepSeek-V4.1 on SM100/SM103) runs the fused Q-RoPE + sparse attention + inverse-RoPE + MXFP8 FlashMLA kernel. The backend choice is shared by prefill and decode. |
| `dsa_paged_mqa_logits_backend` | --dsa-paged-mqa-logits-backend | `str` | `'auto'` | auto, deepgemm, cutedsl, aiter | DSA indexer paged MQA logits kernel backend. Options: 'auto' (default; DeepGEMM on CUDA, aiter on ROCm), 'deepgemm', 'cutedsl' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), 'aiter' (ROCm only). |
| `dsa_topk_backend` | --dsa-topk-backend (可被解析阶段改写) | `str` | `'sgl-kernel'` | sgl-kernel, torch, flashinfer | DSA indexer top-k backend for the target model. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false. |
| `disable_flashinfer_autotune` | --disable-flashinfer-autotune | `bool` | `False` | — | Disable FlashInfer autotuning. |
| `flashinfer_autotune_skip_ops` | --flashinfer-autotune-skip-ops | `Optional[List[str]]` | `None` | — | FlashInfer custom-op identifiers to skip during autotuning. Skipped ops use FlashInfer's heuristic fallback. |
| `triton_attention_reduce_in_fp32` | --triton-attention-reduce-in-fp32 | `bool` | `False` | — | Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16.This only affects Triton attention kernels. |
| `triton_attention_num_kv_splits` | --triton-attention-num-kv-splits | `int` | `8` | — | The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8. |
| `triton_attention_split_tile_size` | --triton-attention-split-tile-size | `Optional[int]` | `None` | — | The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference. |
| `flashinfer_mla_disable_ragged` | --flashinfer-mla-disable-ragged | `bool` | `False` | — | Not using ragged prefill wrapper when running flashinfer mla |
| `enable_fused_qk_norm_rope` | --enable-fused-qk-norm-rope (可被解析阶段改写) | `Optional[bool]` | `None` | — | Fuse q/k normalization and rope rotary embedding into one kernel. By default this is enabled on XPU and disabled elsewhere. Pass --enable-fused-qk-norm-rope or --no-enable-fused-qk-norm-rope to force it either way. |
| `enable_precise_embedding_interpolation` | --enable-precise-embedding-interpolation | `bool` | `False` | — | Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values. |
| `enable_deepseek_v4_fp4_indexer` | --enable-deepseek-v4-fp4-indexer | `bool` | `False` | — | Enable the experimental FP4 C4 indexer path for DeepSeek V4. Default keeps the existing indexer implementation. |
| `enable_dsa_fused_indexer` | --enable-dsa-fused-indexer (可被解析阶段改写) | `Optional[bool]` | `None` | — | Use the four-kernel fused DSA indexer decode path in place of the 12-launch aiter/torch chain. By default this is enabled wherever it is supported -- gfx950 (MI355X) with aiter preshuffle and an fp8 e4m3fn index cache -- and the runtime declines with a logged reason wherever any of that is missing. Pass --no-enable-dsa-fused-indexer to force the standard path. |

### Mamba / 线性注意力

命名空间 `exec.mamba`，声明类 `ExecMamba`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 19 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `enable_mamba_extra_buffer` | --enable-mamba-extra-buffer (fn=sglang.srt.arg_groups.model_override_base.mamba_extra_buffer_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Whether the hybrid-mamba radix cache keeps its extra state buffer: the radix cache is on and the strategy asks for one. |
| `enable_mamba_extra_buffer_lazy` | --enable-mamba-extra-buffer-lazy (fn=sglang.srt.arg_groups.overrides.mamba_extra_buffer_lazy_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | The lazy variant: the strategy is `extra_buffer_lazy` exactly. |
| `mamba_backend` | --mamba-backend | `str` | `'triton'` | triton, flashinfer | Choose the kernel backend for Mamba SSM operations. Default is 'triton'. Options: 'triton' (default), 'flashinfer' (requires FlashInfer with Mamba support). |
| `mamba_ssm_dtype` | --mamba-ssm-dtype | `Optional[str]` | `None` | float32, bfloat16, float16 | The data type of the SSM states in mamba cache. If not set, will be read from model config (mamba_ssm_dtype). |
| `mamba_max_states_per_path` | --mamba-max-states-per-path | `int` | `-1` | — | Maximum number of cached Mamba states retained per root-to-tail path (-1 means unlimited). When enabled, after each insert the shallowest eligible interior states beyond the cap are removed while their full KV remains. Tail, fork, and locked nodes are preserved. Must be -1 or a positive integer. |
| `enable_mamba_cache_stochastic_rounding` | --enable-mamba-cache-stochastic-rounding | `bool` | `False` | — | Enable stochastic rounding when writing FP16 Mamba SSM cache states. Requires --mamba-ssm-dtype float16 and CUDA. With --mamba-backend triton, requires SM100. |
| `mamba_cache_philox_rounds` | --mamba-cache-philox-rounds | `int` | `0` | — | Number of Philox rounds to use for stochastic rounding of FP16 Mamba SSM cache writes. Triton uses the Triton default when set to 0; FlashInfer uses 10 rounds when set to 0. |
| `mamba_radix_cache_strategy` | --mamba-radix-cache-strategy (可被解析阶段改写) | `str` | `'auto'` | auto, no_buffer, extra_buffer, extra_buffer_lazy | The strategy to use for mamba radix cache. |
| `uses_mamba_radix_cache` | --uses-mamba-radix-cache (无 CLI; 可被解析阶段改写) | `bool` | `False` | — | (Derived) whether the model routes through the hybrid-mamba radix cache handling; resolved from the model architecture, no CLI surface. |
| `mamba_track_interval` | --mamba-track-interval | `int` | `256` | — | The interval to track the mamba state during decode. |
| `enable_int8_mamba_checkpoint` | --enable-int8-mamba-checkpoint | `bool` | `False` | — | Store radix-cached linear-attn (mamba) states in int8 (separate checkpoint pool) for ~2x cached-prefix capacity at fixed memory. |
| `int8_mamba_ckpt_size` | --int8-mamba-ckpt-size | `Optional[int]` | `None` | — | Number of int8 mamba checkpoint slots (default: 2x the active mamba pool size). |
| `linear_attn_backend` | --linear-attn-backend | `str` | `'triton'` | triton, cutedsl, flashinfer, flashkda, nvidia_kda, ptx_kda, helion, intel_xpu | The default kernel backend for linear attention (GDN/KDA). Can be overridden per-mode by --linear-attn-decode-backend and --linear-attn-prefill-backend. The Helion backend is KDA-only. |
| `linear_attn_decode_backend` | --linear-attn-decode-backend | `Optional[str]` | `None` | triton, cutedsl, flashinfer, flashkda, nvidia_kda, ptx_kda, helion, intel_xpu | Override the kernel backend for linear attention decode. If not set, uses --linear-attn-backend. |
| `linear_attn_prefill_backend` | --linear-attn-prefill-backend | `Optional[str]` | `None` | triton, cutedsl, flashinfer, flashkda, nvidia_kda, ptx_kda, helion, intel_xpu | Override the kernel backend for linear attention prefill/extend. If not set, uses --linear-attn-backend; compatible SM100 GDN models may automatically select FlashInfer. |
| `linear_attn_verify_backend` | --linear-attn-verify-backend | `Optional[str]` | `None` | LINEAR_ATTN_KERNEL_BACKEND_CHOICES + ['nv_cutedsl'] | Override the kernel backend for linear attention speculative target-verify. If not set, follows the decode backend (flashinfer decode -> flashinfer verify, otherwise triton). KDA supports triton, nv_cutedsl, and flashinfer verify backends. |
| `enable_linear_replayssm` | --enable-linear-replayssm | `bool` | `False` | — | Enable the ReplaySSM buffered output-only linear-attn decode kernel. Primarily a GDN (scalar-gate) decode-bandwidth optimization (~1.2-1.5x at batch >= 64). KDA uses its selected Triton or Helion implementation, but its per-K gate ring is larger and ReplaySSM is typically slower than packed KDA decode; benchmark before enabling it. Requires the Triton linear-attn decode backend, or Helion for KDA, and --mamba-radix-cache-strategy no_buffer (the default). |
| `linear_replayssm_cache_len` | --linear-replayssm-cache-len | `int` | `16` | — | Ring-buffer length L for ReplaySSM linear-attn decode. The full recurrent state is flushed to HBM every L decode steps. |
| `enable_linear_replayssm_spec` | --enable-linear-replayssm-spec | `bool` | `False` | — | Enable the ReplaySSM spec-verify: fold-every-commit -- a per-slot raw-input window replaces the recurrent verify's per-draft full-state snapshots. GDN or KDA hybrid linear-attn models, linear-chain (--speculative-eagle-topk in {None, 1}) only. |

### CUDA Graph

命名空间 `exec.graph`，声明类 `ExecGraph`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 19 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `cuda_graph_config` | --cuda-graph-config | `Optional[CudaGraphConfig]` | `None` | — | Per-phase CUDA graph settings as JSON, e.g. '{"decode":{"backend":"full","max_bs":256},"prefill":{"backend":"breakable"}}'. Allowed backends per phase: full, breakable, disabled. JSON wins over the per-phase --cuda-graph-* convenience flags and over legacy flags. |
| `cuda_graph_backend_decode` | --cuda-graph-backend-decode | `Optional[Literal['full', 'breakable', 'disabled']]` | `None` | Backend.ALL | Backend for the decode phase. Folds into cuda_graph_config[decode].backend. |
| `cuda_graph_backend_prefill` | --cuda-graph-backend-prefill | `Optional[Literal['full', 'breakable', 'disabled']]` | `None` | Backend.ALL | Backend for the prefill phase. Folds into cuda_graph_config[prefill].backend. |
| `cuda_graph_max_bs_decode` | --cuda-graph-max-bs-decode | `Optional[int]` | `None` | — | Maximum batch size captured for the decode cuda graph. |
| `cuda_graph_max_bs_prefill` | --cuda-graph-max-bs-prefill | `Optional[int]` | `None` | — | Maximum batch size captured for the prefill cuda graph. |
| `cuda_graph_max_seq_len_prefill` | --cuda-graph-max-seq-len-prefill | `Optional[int]` | `None` | — | Longest sequence a prefill cuda graph replay admits; longer batches run eager prefill. Folds into cuda_graph_config[prefill].max_seq_len. |
| `cuda_graph_bs_decode` | --cuda-graph-bs-decode | `Optional[List[int]]` | `None` | — | Explicit list of batch sizes to capture for the decode cuda graph. |
| `cuda_graph_bs_prefill` | --cuda-graph-bs-prefill | `Optional[List[int]]` | `None` | — | Explicit list of batch sizes to capture for the prefill cuda graph. |
| `cuda_graph_prefill_max_context` | --cuda-graph-prefill-max-context (--context-bucket) | `Optional[int]` | `None` | — | — |
| `disable_prefill_cuda_graph` | --disable-prefill-cuda-graph | `bool` | `False` | — | Disable the prefill-phase CUDA graph. Convenience for --cuda-graph-backend-prefill=disabled. |
| `disable_decode_cuda_graph` | --disable-decode-cuda-graph | `bool` | `False` | — | Disable the decode-phase CUDA graph. Convenience for --cuda-graph-backend-decode=disabled. |
| `disable_cuda_graph` | --disable-cuda-graph (无 CLI) | `bool` | `False` | — | — |
| `disable_cuda_graph_padding` | --disable-cuda-graph-padding | `bool` | `False` | — | Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed. |
| `enable_profile_cuda_graph` | --enable-profile-cuda-graph | `bool` | `False` | — | Enable profiling of cuda graph capture. |
| `enable_cudagraph_gc` | --enable-cudagraph-gc | `bool` | `False` | — | Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process. |
| `debug_cuda_graph` | --debug-cuda-graph | `bool` | `False` | — | Enable debug/eager mode for CUDA graph using breakable CUDA graph. When enabled, graph breaks are inserted so every operation runs eagerly while still going through the CUDA graph capture / replay path. Useful for debugging CUDA graph capture / replay issues. |
| `enable_torch_compile` | --enable-torch-compile | `bool` | `False` | — | Optimize the model with torch.compile. Experimental feature. |
| `enable_torch_compile_debug_mode` | --enable-torch-compile-debug-mode | `bool` | `False` | — | Enable debug mode for torch compile |
| `torch_compile_max_bs` | --torch-compile-max-bs | `int` | `32` | — | Set the maximum batch size when using torch compile. |

### 集合通信

命名空间 `exec.comm`，声明类 `ExecComm`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 16 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `enable_layerwise_nvtx_marker` | --enable-layerwise-nvtx-marker | `bool` | `False` | — | Enable layerwise NVTX profiling annotations for the model. |
| `enable_nccl_nvls` | --enable-nccl-nvls | `bool` | `False` | — | Enable NCCL NVLS for prefill heavy requests when available. |
| `enable_symm_mem` | --enable-symm-mem (可被解析阶段改写) | `bool` | `False` | — | Enable NCCL symmetric memory for fast collectives. |
| `disable_custom_all_reduce` | --disable-custom-all-reduce (可被解析阶段改写) | `bool` | `False` | — | Disable the custom all-reduce kernel and fall back to NCCL. |
| `enable_mscclpp` | --enable-mscclpp | `bool` | `False` | — | Enable MSCCL++ for tuned AllReduce and AllGather messages, with NCCL fallback. |
| `enable_torch_symm_mem` | --enable-torch-symm-mem | `bool` | `False` | — | Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM100 supports world size 6, 8. |
| `enable_scattered_sconv` | --enable-scattered-sconv | `bool` | `False` | — | Inkling: replace the attention/MLP output all-reduce with a hidden-dimension reduce-scatter, run the channelwise output short convolution on the [T, H/P] shard, then all-gather before the residual add. This shards the convolution cache across tensor-parallel ranks without changing communication volume. |
| `pre_warm_nccl` | --pre-warm-nccl | `bool` | `False` | — | Pre-warm NCCL/RCCL communicators during startup to reduce P99 TTFT cold-start latency. Default: enabled for AMD/HIP (RCCL), disabled for NVIDIA/CUDA (NCCL). |
| `boundary_reduction` | --boundary-reduction (可被解析阶段改写) | `Literal['auto', 'ar', 'rs', 'rsv', 'rs+rsv']` | `'auto'` | — | Select FFN boundary reduction: ar uses all-reduce then token redistribution; rs and rsv permit fixed-size and variable-size reduce-scatter respectively; rs+rsv permits both (RSv first). Unsupported paths fall back to ar. The option applies to the FFN stages of decoders built with stage boundaries, where auto resolves the model default; other models ignore it. Required attention and MoE collectives and all-reduce fusion are unaffected. |
| `enable_quant_communications` | --enable-quant-communications | `Optional[bool]` | `False` | — | Enable INT8 quantization of TP communications (limited support). |
| `enable_flashinfer_allreduce_fusion` | --enable-flashinfer-allreduce-fusion (无 CLI) | `bool` | `False` | — | — |
| `enforce_disable_flashinfer_allreduce_fusion` | --enforce-disable-flashinfer-allreduce-fusion | `bool` | `False` | — | Enforce disable FlashInfer allreduce fusion. |
| `flashinfer_allreduce_fusion_backend` | --flashinfer-allreduce-fusion-backend (可被解析阶段改写) | `Optional[Literal['auto', 'trtllm', 'mnnvl', 'cutedsl']]` | `None` | — | Enable FlashInfer allreduce fusion and choose backend. Requires SM90 or SM10X NVIDIA GPUs. When unspecified, supported MiMo configurations use cutedsl; other supported models use auto. 'auto': choose mnnvl on Blackwell (SM100/SM103) systems (single- and multi-node) and trtllm on SM90 single-node systems. 'trtllm': available on single-node systems only. 'mnnvl': available on SM90 single-node systems and SM100/SM103 single-node or multi-node systems via MNNVL fabric. 'cutedsl': Blackwell-only bf16 MNNVL CuTe DSL backend; also fuses the MoE finalize and the shared-expert add into the collective when the MoE runner can defer them. Fuses allreduce with Residual + RMSNorm for supported MoE models. |
| `enable_aiter_allreduce_fusion` | --enable-aiter-allreduce-fusion (可被解析阶段改写) | `bool` | `False` | — | Enable Aiter AllReduce Fusion. |
| `disable_aiter_allreduce_fusion_in_prefill` | --disable-aiter-allreduce-fusion-in-prefill (可被解析阶段改写) | `bool` | `False` | — | Disable Aiter AllReduce Fusion for prefill batches (EXTEND / MIXED / SPLIT_PREFILL) while keeping it for decode. Only meaningful with --enable-aiter-allreduce-fusion. |
| `disable_aiter_allreduce_fusion_in_decode` | --disable-aiter-allreduce-fusion-in-decode (可被解析阶段改写) | `bool` | `False` | — | Disable Aiter AllReduce Fusion for decode batches (DECODE / TARGET_VERIFY / draft-extend / IDLE) while keeping it for prefill. Only meaningful with --enable-aiter-allreduce-fusion. |

### MoE 与 DeepEP / A2A

命名空间 `exec.moe`，声明类 `ExecMoe`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 41 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `is_ep_joiner` | --is-ep-joiner (fn=sglang.srt.arg_groups.model_override_base.ep_joiner_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Whether this process was launched as an elastic-EP joiner (scale or recover). |
| `is_ep_scale_joiner` | --is-ep-scale-joiner (fn=sglang.srt.arg_groups.model_override_base.ep_scale_joiner_of; 派生，无 CLI) | `Derived` | `（发布时计算）` | — | Whether it is a scale-up joiner specifically. |
| `enable_fused_moe_sum_all_reduce` | --enable-fused-moe-sum-all-reduce | `bool` | `False` | — | Enable fused moe triton and sum all reduce. |
| `moe_a2a_backend` | --moe-a2a-backend (可被解析阶段改写) | `Literal['none', 'deepep', 'mooncake', 'nixl', 'mori', 'ascend_fuseep', 'flashinfer', 'megamoe', 'deepep_v2', 'ascend_tp', 'pplx', 'flashinfer_megamoe']` | `'none'` | none, deepep, mooncake, nixl, mori, ascend_fuseep, flashinfer, megamoe, deepep_v2, pplx, ascend_tp, flashinfer_megamoe | Choose the backend for MoE A2A. |
| `enable_w4a4_mxfp4_megamoe` | --enable-w4a4-mxfp4-megamoe | `bool` | `False` | — | Enable the W4A4 MXFP4 MegaMoE path with DeepGEMM's mxf4xmxf4 MMA type. Use with --moe-a2a-backend megamoe. |
| `deepep_v2_mode` | --deepep-v2-mode | `Literal['direct', 'hybrid']` | `'direct'` | — | DeepEP v2 ElasticBuffer communication topology, fixed at server init: `direct` (single-node NVLink) or `hybrid` (multi-node scale-out). Layout/grouped-GEMM and the decode CUDA graph are chosen per batch by inference phase, independent of this knob; not equivalent to DeepEP v1 normal/low_latency. |
| `moe_runner_backend` | --moe-runner-backend (可被解析阶段改写) | `str` | `'auto'` | auto, deep_gemm, triton, triton_kernel, flashinfer_trtllm, experimental_sgl_trtllm, flashinfer_trtllm_routed, flashinfer_cutlass, flashinfer_mxfp4, flashinfer_cutedsl, flashinfer_megamoe, cutlass, aiter, marlin, humming, experimental_sgl_marlin, lora_cutedsl, lora_triton, lora_marlin, hpc_ops, megamoe, intel_xpu | Choose the runner backend for MoE. |
| `flashinfer_mxfp4_moe_precision` | --flashinfer-mxfp4-moe-precision | `Literal['default', 'bf16', 'fp8']` | `'default'` | — | Choose the computation precision of flashinfer mxfp4 moe. On SM90, `fp8` selects the Humming-style MXFP4-weight x FP8-activation path introduced by FlashInfer #3738 and requires FlashInfer >= 0.6.18. |
| `deepep_mode` | --deepep-mode | `Literal['auto', 'normal', 'low_latency']` | `'auto'` | — | Select the mode when enable DeepEP or MoriEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch. |
| `fuseep_mode` | --fuseep-mode | `Literal[1, 2]` | `2` | — | Select the mode when enable Ascend FuseEP MoE, 1 -> dispatch_gmm_combine_decode is executed；2 -> dispatch_ffn_combine is executed (support hybrid deployment when 2). |
| `deepep_dispatcher_output_dtype` | --deepep-dispatcher-output-dtype | `Literal['auto', 'bf16', 'fp8', 'int8', 'nvfp4']` | `'auto'` | — | Select DeepEP dispatcher output dtype |
| `flashinfer_a2a_dispatch_type` | --flashinfer-a2a-dispatch-type | `Optional[Literal['auto', 'bf16', 'nvfp4', 'mxfp8']]` | `None` | — | Select FlashInfer A2A dispatcher activation dtype. |
| `ep_num_redundant_experts` | --ep-num-redundant-experts | `int` | `0` | — | Allocate this number of redundant experts in expert parallel. |
| `ep_dispatch_algorithm` | --ep-dispatch-algorithm | `Optional[Literal['static', 'dynamic', 'fake', 'lp']]` | `None` | — | The algorithm to choose ranks for redundant experts in expert parallel. |
| `init_expert_location` | --init-expert-location | `str` | `'trivial'` | — | Initial location of EP experts. |
| `enable_eplb` | --enable-eplb | `bool` | `False` | — | Enable EPLB algorithm |
| `eplb_algorithm` | --eplb-algorithm | `str` | `'auto'` | — | Chosen EPLB algorithm |
| `eplb_rebalance_num_iterations` | --eplb-rebalance-num-iterations | `int` | `1000` | — | Number of iterations to automatically trigger a EPLB re-balance. |
| `eplb_rebalance_layers_per_chunk` | --eplb-rebalance-layers-per-chunk | `Optional[int]` | `None` | — | Number of layers to rebalance per forward pass. |
| `eplb_min_rebalancing_utilization_threshold` | --eplb-min-rebalancing-utilization-threshold | `float` | `1.0` | — | Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0]. |
| `expert_distribution_recorder_mode` | --expert-distribution-recorder-mode | `Optional[Literal['stat', 'stat_approx', 'per_pass', 'per_token']]` | `None` | — | Mode of expert distribution recorder. |
| `expert_distribution_recorder_buffer_size` | --expert-distribution-recorder-buffer-size | `Optional[int]` | `None` | — | Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer. |
| `expert_balancedness_report_mode` | --expert-balancedness-report-mode | `Literal['off', 'server_log', 'prometheus', 'both']` | `'off'` | — | Where to report expert balancedness. Options: off, server_log, prometheus, both. |
| `deepep_config` | --deepep-config | `Optional[str]` | `None` | — | Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path. |
| `elastic_ep_backend` | --elastic-ep-backend | `Literal[None, 'mooncake', 'nixl']` | `None` | none, mooncake, nixl | Specify the collective communication backend for elastic EP. Supports 'mooncake' and 'nixl'. |
| `enable_elastic_expert_backup` | --enable-elastic-expert-backup | `bool` | `False` | — | Enable elastic expert backup feature. |
| `mooncake_ib_device` | --mooncake-ib-device | `Optional[str]` | `None` | — | The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices (e.g., --mooncake-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when Mooncake Backend is enabled. |
| `enable_waterfill` | --enable-waterfill | `bool` | `False` | — | Enable Waterfill: dispatch the fused shared expert as an extra routed expert slot to the least-loaded EP rank. Supports DeepEP and MegaMOE MoE A2A backends, implicitly enables shared-expert fusion, and supports --deepep-mode auto, normal, or low_latency when used with DeepEP. Use auto or low_latency for production DeepEP decode so CUDA graph remains enabled. Supported on DeepSeek-V3/R1 with EP >= 2. |
| `ep_join_mode` | --elastic-ep-join-mode | `Optional[Literal['scale', 'recover']]` | `None` | scale, recover | Join mode for elastic EP. 'recover' rejoins an existing slot after a fault. 'scale' joins as a new rank beyond the original group size and requires --node-rank 1. |
| `elastic_ep_scale_timeout` | --elastic-ep-scale-timeout | `float` | `600` | — | Timeout in seconds for a pending elastic EP scale operation. |
| `disable_flashinfer_cutlass_moe_fp4_allgather` | --disable-flashinfer-cutlass-moe-fp4-allgather | `bool` | `False` | — | Disables quantize before all-gather for flashinfer cutlass moe. |
| `disable_shared_experts_fusion` | --disable-shared-experts-fusion (可被解析阶段改写) | `bool` | `False` | — | Disable the built-in shared experts fusion optimization for DeepSeek V3/R1. Note: Waterfill (--enable-waterfill) routes the shared expert as an extra MoE slot, so the shared expert is not separated from the MoE path when Waterfill is enabled. |
| `enforce_shared_experts_fusion` | --enforce-shared-experts-fusion | `bool` | `False` | — | Enforce shared experts fusion even when it would normally be disabled (e.g. under DeepEP). On a Quark MXFP4 Qwen3.5/Qwen3.8 checkpoint that excludes its shared expert from quantization, this also quantizes that shared expert to MXFP4 while loading so it can be fused, which runs its weights and its input activations at MXFP4 where the standalone path ran in BF16 - validate accuracy before using it. Mutually exclusive with --disable-shared-experts-fusion. |
| `enable_paged_experts` | --enable-paged-experts | `bool` | `False` | — | Keep only --paged-experts-num-resident routed experts per MoE layer on the GPU and page the rest from pinned host memory on demand. Single GPU; the experts must run on the triton or Marlin MoE runner (supported quantizations: see the Paged Experts docs). Decode CUDA graphs (full backend) are captured up to K // top_k requests; prefill graphs must be breakable or disabled. |
| `paged_experts_num_resident` | --paged-experts-num-resident | `Optional[int]` | `None` | — | Number of routed experts per MoE layer kept resident on the GPU. Required with --enable-paged-experts; at least the router's top-k. |
| `kt_weight_path` | --kt-weight-path | `Optional[str]` | `None` | — | [ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder. |
| `kt_method` | --kt-method | `str` | `'AMXINT4'` | — | [ktransformers parameter] Quantization formats for CPU execution. |
| `kt_cpuinfer` | --kt-cpuinfer | `Optional[int]` | `None` | — | [ktransformers parameter] The number of CPUInfer threads. |
| `kt_threadpool_count` | --kt-threadpool-count | `int` | `2` | — | [ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA). |
| `kt_num_gpu_experts` | --kt-num-gpu-experts | `Optional[int]` | `None` | — | [ktransformers parameter] The number of GPU experts. |
| `kt_max_deferred_experts_per_token` | --kt-max-deferred-experts-per-token | `Optional[int]` | `None` | — | [ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0. |

### Batch overlap

命名空间 `exec.overlap`，声明类 `ExecOverlap`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 3 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `enable_two_batch_overlap` | --enable-two-batch-overlap | `bool` | `False` | — | Enabling two micro batches to overlap. |
| `enable_single_batch_overlap` | --enable-single-batch-overlap | `bool` | `False` | — | Let computation and communication overlap within one micro batch. |
| `tbo_token_distribution_threshold` | --tbo-token-distribution-threshold | `float` | `0.48` | — | The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap. |

### 权重 offload

命名空间 `exec.offload`，声明类 `ExecOffload`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 8 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `cpu_offload_gb` | --cpu-offload-gb | `int` | `0` | — | How many GBs of RAM to reserve for CPU offloading. |
| `offload_group_size` | --offload-group-size | `int` | `-1` | — | Number of layers per group in offloading. |
| `offload_num_in_group` | --offload-num-in-group | `int` | `1` | — | Number of layers to be offloaded within a group. |
| `offload_prefetch_step` | --offload-prefetch-step | `int` | `1` | — | Steps to prefetch in offloading. |
| `offload_mode` | --offload-mode | `str` | `'cpu'` | — | Mode of offloading. |
| `ple_offload_embedding` | --ple-offload-embedding (可被解析阶段改写) | `Optional[bool]` | `None` | — | Offload Qwen4 PLE n-gram embedding weights to CPU pinned memory. Enabled by default for BF16 Qwen4-Exp on CUDA; use --no-ple-offload-embedding to disable. |
| `ple_offload_backend` | --ple-offload-backend | `str` | `'pinned'` | pinned, file | Host storage for the offloaded Qwen4 PLE n-gram table. 'pinned' (default) uses CPU pinned memory. 'file' maps a sparse file under --ple-offload-dir and lets the gather kernel read it directly; use it on unified-memory devices (e.g. GB10 / DGX Spark) where pinned host memory comes out of the same pool as the model weights. Requires a device that reports cudaDevAttrPageableMemoryAccessUsesHostPageTables. |
| `ple_offload_dir` | --ple-offload-dir | `Optional[str]` | `None` | — | Directory for the file-backed PLE table when --ple-offload-backend is 'file'. Defaults to $SGLANG_CACHE_DIR/ple/<model path>, one directory per checkpoint. The file is sparse and reused across restarts; put it on fast local storage (NVMe). |

### 扩散语言模型（dLLM）

命名空间 `exec.dllm`，声明类 `ExecDllm`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 3 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `dllm_algorithm` | --dllm-algorithm | `Optional[str]` | `None` | — | The diffusion LLM algorithm, such as LowConfidence. |
| `dllm_algorithm_config` | --dllm-algorithm-config | `Optional[str]` | `None` | — | The diffusion LLM algorithm configurations. Must be a YAML file. |
| `dllm_fdfo` | --dllm-fdfo | `bool` | `True` | — | Enable First-Done-First-Out (FDFO) scheduling for diffusion LLM inference. Enabled by default; use --no-dllm-fdfo to fall back to synchronous block scheduling. |

### 确定性推理

命名空间 `exec.deterministic`，声明类 `ExecDeterministic`，文件 `python/sglang/srt/arg_groups/fields/exec_.py`。共 2 项。

| 字段 | CLI | 类型 | 默认 | 可选值 | 含义 |
| --- | --- | --- | --- | --- | --- |
| `enable_deterministic_inference` | --enable-deterministic-inference | `bool` | `False` | — | Enable deterministic inference mode with batch invariant ops. |
| `rl_on_policy_target` | --rl-on-policy-target | `Optional[str]` | `None` | fsdp | The training system that SGLang needs to match for true on-policy. |

## Diffusion ServerArgs（多模态生成，独立于 LLM ServerArgs）

入口不走 `sglang.srt.server_args.ServerArgs`。类是 `@dataclasses.dataclass`，CLI 由其自身的 argparse 注册逻辑生成；下表列出全部声明字段（含 mixin），注释取字段正上方的 `#` 行。共 173 项。

| 字段 | 声明类 | 类型 | 默认 | 注释 |
| --- | --- | --- | --- | --- |
| `DISAGG_RESULT_PORT_OFFSETS` | `DisaggServerArgsMixin`:18 | `ClassVar[dict[RoleType, int]]` | `{RoleType.ENCODER: 1, RoleType.DENOISER: 2, RoleType.DECODER: 3, RoleType.PROMPT_ENCODER: 4, RoleType.IMAGE_ENCODER: 5}` | — |
| `model_path` | `ServerArgs`:299 | `str` | `（必填）` | Model and path configuration (for convenience) |
| `model_subfolder` | `ServerArgs`:300 | `str \| None` | `None` | — |
| `model_variant` | `ServerArgs`:301 | `str \| None` | `None` | — |
| `model_id` | `ServerArgs`:304 | `str \| None` | `None` | explicit model ID override (e.g. "Qwen-Image") |
| `served_model_name` | `ServerArgs`:307 | `str \| None` | `None` | served model name exposed via /v1/models and generation responses |
| `backend` | `ServerArgs`:310 | `Backend` | `Backend.AUTO` | Model backend (sglang native or diffusers) |
| `attention_backend` | `ServerArgs`:313 | `str` | `None` | Attention |
| `enable_attention_backend_autotune` | `ServerArgs`:317 | `bool` | `False` | Time the viable attention backends on the model's own tensors during warmup and keep the fastest per layer. Off by default: it has only been measured on sm90 and sm12x. |
| `attention_backend_config` | `ServerArgs`:318 | `addict.Dict \| None` | `None` | — |
| `component_attention_backends` | `ServerArgs`:319 | `dict[str, str] \| str \| None` | `field(default_factory=dict)` | — |
| `_requested_component_attention_backends` | `ServerArgs`:322 | `dict[str, str] \| None` | `field(default=None, repr=False, compare=False)` | — |
| `component_attention_backend_roles` | `ServerArgs`:328 | `dict[str, dict[str, str]]` | `field(default_factory=dict)` | Role-qualified overrides (``<component>.<role>``) live in their own map so ``component_attention_backends`` stays a plain component -> backend dict for every consumer that looks a component up by name. |
| `_requested_component_attention_backend_roles` | `ServerArgs`:331 | `dict[str, dict[str, str]] \| None` | `field(default=None, repr=False, compare=False)` | — |
| `cache_dit_config` | `ServerArgs`:334 | `str \| dict[str, Any] \| None` | `None` | — |
| `nccl_port` | `ServerArgs`:339 | `Optional[int]` | `None` | Distributed executor backend |
| `enable_nccl_nvls` | `ServerArgs`:340 | `bool` | `False` | — |
| `trust_remote_code` | `ServerArgs`:343 | `bool` | `False` | HuggingFace specific parameters |
| `revision` | `ServerArgs`:344 | `str \| None` | `None` | — |
| `num_gpus` | `ServerArgs`:347 | `int` | `1` | Parallelism |
| `performance_mode` | `ServerArgs`:348 | `str` | `'auto'` | — |
| `base_gpu_id` | `ServerArgs`:349 | `int` | `0` | — |
| `gpu_ids` | `ServerArgs`:350 | `list[int] \| None` | `None` | — |
| `numa_node` | `ServerArgs`:352 | `list[int] \| None` | `None` | i-th entry: NUMA node of the worker on local GPU i; None auto-detects |
| `nnodes` | `ServerArgs`:356 | `int` | `1` | cross-node: num_gpus is the total world size across all nodes; each node runs num_gpus // nnodes local GPU workers (mirrors srt's tp_size_per_node convention) |
| `node_rank` | `ServerArgs`:357 | `int` | `0` | — |
| `dist_init_addr` | `ServerArgs`:358 | `str \| None` | `None` | — |
| `tp_size` | `ServerArgs`:359 | `Optional[int]` | `None` | — |
| `sp_degree` | `ServerArgs`:360 | `Optional[int]` | `None` | — |
| `ulysses_degree` | `ServerArgs`:362 | `Optional[int]` | `None` | sequence parallelism |
| `ring_degree` | `ServerArgs`:363 | `Optional[int]` | `None` | — |
| `kv_gather_degree` | `ServerArgs`:367 | `Optional[int]` | `None` | rows split inside attention, exchanged with one K/V all-gather instead of Ulysses a2a or ring rotation; auto-assigned at sp_degree=2 when no SP degree is set explicitly |
| `sp_split_auto` | `ServerArgs`:369 | `bool` | `False` | whether the SP split was auto-assigned (lets layers fall back per call) |
| `dp_size` | `ServerArgs`:372 | `int` | `1` | data parallelism number of data parallelism groups |
| `enable_cfg_parallel` | `ServerArgs`:375 | `Optional[bool]` | `None` | number of gpu in a dp group cfg parallel (None = auto-decide based on num_gpus) |
| `cfg_parallel_degree` | `ServerArgs`:377 | `Optional[int]` | `None` | number of GPUs in each CFG parallel group (None = auto, 1 = disabled, N > 1 = enabled) |
| `encoder_parallel` | `ServerArgs`:382 | `str` | `'auto'` | encoder layout across a multi-rank replica: auto \| fold \| dp \| replicate (see --encoder-parallel); fold shards the weights at load time, so it is mutually exclusive with dp/replicate for the lifetime of the model |
| `hsdp_replicate_dim` | `ServerArgs`:384 | `int` | `1` | — |
| `hsdp_shard_dim` | `ServerArgs`:385 | `Optional[int]` | `None` | — |
| `dist_timeout` | `ServerArgs`:386 | `int \| None` | `3600` | — |
| `scheduler_rpc_timeout` | `ServerArgs`:387 | `int \| None` | `None` | — |
| `pipeline_config` | `ServerArgs`:389 | `PipelineConfig` | `field(default_factory=PipelineConfig, repr=False)` | — |
| `pipeline_class_name` | `ServerArgs`:392 | `str \| None` | `None` | Pipeline override |
| `lora_path` | `ServerArgs`:398 | `str \| None` | `None` | LoRA parameters (Wenxuan) prefer to keep it here instead of in pipeline config to not make it complicated. |
| `lora_nickname` | `ServerArgs`:399 | `str` | `'default'` | — |
| `lora_scale` | `ServerArgs`:400 | `float` | `1.0` | — |
| `lora_alpha` | `ServerArgs`:401 | `int \| None` | `None` | — |
| `lora_merge_mode` | `ServerArgs`:402 | `str` | `'auto'` | — |
| `lora_weight_name` | `ServerArgs`:403 | `str \| None` | `None` | — |
| `component_paths` | `ServerArgs`:406 | `dict[str, str]` | `field(default_factory=dict)` | Component path overrides (key = model_index.json component name, value = path) |
| `component_weights_paths` | `ServerArgs`:408 | `dict[str, str]` | `field(default_factory=dict)` | Exact weight-file overrides retain the base component configuration. |
| `component_direct_gpu_weight_loading` | `ServerArgs`:411 | `dict[str, bool]` | `field(default_factory=dict)` | Opt in one component to a loader-specific direct-GPU weight path. The existing --direct-gpu-weight-loading remains the primary-DiT control. |
| `component_quantizations` | `ServerArgs`:414 | `dict[str, str]` | `field(default_factory=dict)` | Explicit quantization override for one component. Self-describing checkpoints remain auto-detected and do not need this override. |
| `component_precisions` | `ServerArgs`:417 | `dict[str, str]` | `field(default_factory=dict)` | Exact load and execution precision overrides for components whose native loader advertises this capability. |
| `component_quantization_ignored_layers` | `ServerArgs`:419 | `dict[str, list[str]]` | `field(default_factory=dict)` | Component-local layer name patterns to skip during online quantization. |
| `load_diffusion_decoder` | `ServerArgs`:423 | `bool` | `False` | Optional LTX-2.5 decoder is large enough to load only when requested. |
| `transformer_weights_path` | `ServerArgs`:426 | `str \| None` | `None` | Pre-quantized transformer weights: safetensors file/directory or GGUF file. |
| `minimax_h3_adaln_cache_path` | `ServerArgs`:428 | `str \| None` | `None` | path to precomputed MiniMax H3 AdaLN outputs for inference-only serving. |
| `minimax_h3_adaln_online` | `ServerArgs`:430 | `bool` | `False` | Rebuild AdaLN outputs per request from the checkpoint, no sidecar needed. |
| `minimax_h3_adaln_plan_width` | `ServerArgs`:433 | `int` | `4` | Widest timestep plan the rebuild slab is sized for; see MINIMAX_H3_ADALN_MAX_PLAN_WIDTH. |
| `minimax_h3_adaln_host_cache_gb` | `ServerArgs`:438 | `float` | `8.0` | Pinned-host cache for built AdaLN plans (decimal GB, 0 disables). Plans evicted from the GPU slab swap back in from here instead of re-reading the 24.2 GiB checkpoint. Expert knobs (GPU slot count, fp32 rebuild) live in envs.py as SGLANG_DIFFUSION_MINIMAX_H3_ADALN_*. |
| `quantization` | `ServerArgs`:441 | `str \| None` | `None` | Explicit quantization method override (e.g. "mxfp8", "fp8", "modelslim"). When set, the transformer loader uses it instead of auto-detection. |
| `quantization_ignored_layers` | `ServerArgs`:443 | `list[str] \| None` | `None` | Layer name patterns to skip during online quantization |
| `lora_target_modules` | `ServerArgs`:447 | `list[str] \| None` | `None` | can restrict layers to adapt, e.g. ["q_proj"] Will adapt only q, k, v, o by default. |
| `component_residency` | `ServerArgs`:451 | `dict[str, str] \| list[str] \| str \| None` | `None` | CPU offload parameters Exact component keys or component groups mapped to a residency mode. |
| `cpu_offload_components` | `ServerArgs`:453 | `list[str] \| None` | `None` | Exact component keys from model_index.json, or a legacy component group. |
| `dit_cpu_offload` | `ServerArgs`:454 | `bool \| None` | `None` | — |
| `direct_gpu_weight_loading` | `ServerArgs`:456 | `bool` | `False` | trade checkpoint-loading peak memory for faster ordinary DiT startup |
| `dit_layerwise_offload` | `ServerArgs`:458 | `bool \| None` | `None` | if true, select the DiT layerwise group |
| `layerwise_offload_components` | `ServerArgs`:459 | `list[str] \| None` | `None` | — |
| `dit_offload_prefetch_size` | `ServerArgs`:460 | `float` | `0.0` | — |
| `dit_layerwise_resident_layers` | `ServerArgs`:462 | `float` | `0.0` | If set, keep this many DiT layers resident on GPU |
| `dit_layerwise_residency_policy` | `ServerArgs`:464 | `str` | `RESIDENCY_POLICY_LEADING` | Which layers those are: the leading ones, or spread evenly over the stack. |
| `dit_layerwise_residency_lifetime` | `ServerArgs`:467 | `str` | `RESIDENCY_LIFETIME_FORWARD` | How long they stay: while the DiT runs for a request, or for the life of the server. |
| `layerwise_prefetch_size` | `ServerArgs`:470 | `dict[str, float] \| str \| None` | `field(default_factory=dict)` | Per-component overrides of the four knobs above; an entry wins for that component. |
| `layerwise_resident_layers` | `ServerArgs`:471 | `dict[str, float] \| str \| None` | `field(default_factory=dict)` | — |
| `layerwise_residency_policy` | `ServerArgs`:474 | `dict[str, str] \| str \| None` | `field(default_factory=dict)` | — |
| `layerwise_residency_lifetime` | `ServerArgs`:477 | `dict[str, str] \| str \| None` | `field(default_factory=dict)` | — |
| `offload_during_compile` | `ServerArgs`:480 | `bool` | `True` | — |
| `text_encoder_cpu_offload` | `ServerArgs`:481 | `bool \| None` | `None` | — |
| `image_encoder_cpu_offload` | `ServerArgs`:482 | `bool \| None` | `None` | — |
| `vae_cpu_offload` | `ServerArgs`:483 | `bool \| None` | `False` | — |
| `use_fsdp_inference` | `ServerArgs`:484 | `bool \| None` | `None` | — |
| `pin_cpu_memory` | `ServerArgs`:485 | `bool` | `True` | — |
| `ltx2_two_stage_device_mode` | `ServerArgs`:486 | `str \| None` | `None` | — |
| `_explicit_arg_names` | `ServerArgs`:487 | `set[str]` | `field(default_factory=set, repr=False)` | — |
| `_required_resident_components` | `ServerArgs`:488 | `set[str]` | `field(default_factory=set, init=False, repr=False)` | — |
| `_fsdp_disabled_components` | `ServerArgs`:491 | `set[str]` | `field(default_factory=set, init=False, repr=False)` | — |
| `_component_layerwise_capabilities` | `ServerArgs`:494 | `dict[str, bool]` | `field(default_factory=dict, init=False, repr=False)` | — |
| `comfyui_mode` | `ServerArgs`:499 | `bool` | `False` | ComfyUI integration |
| `enable_torch_compile` | `ServerArgs`:502 | `bool` | `False` | Compilation |
| `regional_compile` | `ServerArgs`:503 | `bool` | `False` | — |
| `allow_cudnn_tf32` | `ServerArgs`:514 | `bool` | `True` | Breakable CUDA graph (BCG): capture the DiT forward as CUDA-graph segments split at attention modules (SP all-to-all / dynamic attention stay eager). Mutually exclusive with --enable-torch-compile and Cache-DiT; BCG takes priority when more than one is requested. PyTorch's two approximate numerics defaults, set explicitly per worker (see runtime/utils/numerics_policy.py). They are process-global, so they cannot follow a request's quality level; the defaults keep PyTorch's own values so the exact tier's speed is unchanged. |
| `allow_bf16_reduced_precision_reduction` | `ServerArgs`:515 | `bool` | `True` | — |
| `enable_breakable_cuda_graph` | `ServerArgs`:519 | `bool \| None` | `None` | BCG graphs are resolution-specific: every warmup resolution is captured at warmup so serving never triggers a fresh capture. None resolves in _adjust_breakable_cuda_graph_support (on for opted-in pipelines). |
| `bcg_text_buckets` | `ServerArgs`:525 | `list[int]` | `None` | Text/prompt sequence-length padding budget for BCG. Prompt-conditioning inputs are padded up to the smallest bucket that fits, so prompts of different lengths reuse one captured graph. Warmup captures one graph per bucket; a prompt longer than the largest bucket falls back to eager. ``None`` resolves to DEFAULT_BCG_TEXT_BUCKETS. |
| `enable_layerwise_nvtx_marker` | `ServerArgs`:528 | `bool` | `False` | NVTX profiling |
| `warmup_mode` | `ServerArgs`:537 | `str \| None` | `None` | Warmup is controlled by the canonical `warmup_mode` knob: one of WARMUP_MODES. - "off": no warmup. - "server": server-based warmup — a synthetic request right after the server is ready, before real traffic - "request": request-based warmup — warm on the first real request(s). This is a BENCHMARK aid. None is resolved by _adjust_warmup from the selected runtime features. |
| `warmup_resolutions` | `ServerArgs`:539 | `list[str]` | `None` | — |
| `warmup_num_frames` | `ServerArgs`:540 | `int \| None` | `None` | — |
| `warmup_steps` | `ServerArgs`:541 | `int` | `1` | — |
| `warmup_sampling_params` | `ServerArgs`:545 | `dict[str, Any] \| str \| None` | `None` | JSON overrides for the representative request shape used by synthetic warmup and automatic residency planning. Execution remains bounded by warmup_steps and the server warmup frame/area caps. |
| `disable_autocast` | `ServerArgs`:547 | `bool \| None` | `None` | — |
| `nunchaku_config` | `ServerArgs`:550 | `NunchakuSVDQuantArgs \| NunchakuConfig \| None` | `field(default_factory=NunchakuSVDQuantArgs, repr=False)` | Quantization / Nunchaku SVDQuant configuration |
| `kv_cache_quant_config` | `ServerArgs`:556 | `QVGKVQuantArgs` | `field(default_factory=QVGKVQuantArgs, repr=False)` | KV-cache quantization (Quant-VideoGen PRQ). Off by default; mirrors the SRT --kv-cache-dtype pattern (typed config, not a pile of env vars). |
| `master_port` | `ServerArgs`:561 | `int` | `30005` | Master port for distributed inference |
| `host` | `ServerArgs`:564 | `str \| None` | `'127.0.0.1'` | http server endpoint config |
| `port` | `ServerArgs`:565 | `int \| None` | `30000` | — |
| `http_graceful_shutdown_timeout` | `ServerArgs`:566 | `int \| None` | `None` | — |
| `enable_metrics` | `ServerArgs`:567 | `bool` | `False` | — |
| `webui` | `ServerArgs`:570 | `bool` | `False` | TODO: webui and their endpoint, check if webui_port is available. |
| `webui_port` | `ServerArgs`:571 | `int \| None` | `12312` | — |
| `scheduler_port` | `ServerArgs`:573 | `int` | `5555` | — |
| `scheduler_ports` | `ServerArgs`:575 | `list[int] \| None` | `None` | settled ingress ports, one per DP replica; None until ports are settled |
| `batching_mode` | `ServerArgs`:576 | `str` | `'dynamic'` | — |
| `batching_max_size` | `ServerArgs`:577 | `int` | `1` | — |
| `batching_delay_ms` | `ServerArgs`:578 | `float` | `0.0` | — |
| `batching_config` | `ServerArgs`:579 | `str \| None` | `None` | — |
| `enable_batching_metrics` | `ServerArgs`:580 | `bool` | `False` | — |
| `async_output_save` | `ServerArgs`:581 | `bool` | `False` | — |
| `sleep_on_idle` | `ServerArgs`:582 | `bool` | `False` | — |
| `strict_ports` | `ServerArgs`:585 | `bool` | `False` | Strict port mode: fail if requested port is unavailable instead of auto-selecting |
| `output_path` | `ServerArgs`:587 | `str \| None` | `'outputs/'` | — |
| `input_save_path` | `ServerArgs`:588 | `str \| None` | `'inputs/uploads'` | — |
| `prompt_file_path` | `ServerArgs`:591 | `str \| None` | `None` | Prompt text file for batch processing |
| `model_paths` | `ServerArgs`:594 | `dict[str, str]` | `field(default_factory=dict)` | model paths for correct deallocation |
| `model_loaded` | `ServerArgs`:595 | `dict[str, bool]` | `field(default_factory=lambda: {'transformer': True, 'vae': True, 'video_vae': True, 'audio_vae': True, 'video_dit': True, 'audio_dit': True, 'dual_tower_bridge': True})` | — |
| `boundary_ratio` | `ServerArgs`:608 | `float \| None` | `None` | MoE parameters used by Wan2.2 |
| `disagg_role` | `ServerArgs`:611 | `RoleType` | `RoleType.MONOLITHIC` | Disaggregation (pool mode only — launched via launch_pool_disagg_server()) |
| `disagg_timeout` | `ServerArgs`:612 | `int` | `3600` | — |
| `disagg_downstream_wait_timeout` | `ServerArgs`:613 | `int` | `1800` | — |
| `disagg_dispatch_policy` | `ServerArgs`:614 | `str` | `'round_robin'` | — |
| `disagg_mode` | `ServerArgs`:615 | `bool` | `False` | — |
| `disagg_instance_id` | `ServerArgs`:616 | `int` | `0` | — |
| `disagg_max_slots_per_instance` | `ServerArgs`:617 | `int` | `8` | — |
| `disagg_transfer_redundancy` | `ServerArgs`:618 | `float` | `1.25` | — |
| `disagg_role_device` | `ServerArgs`:619 | `Literal['auto', 'cpu', 'cuda']` | `'auto'` | — |
| `disagg_transfer_backend` | `ServerArgs`:620 | `Literal['auto', 'mock', 'mooncake']` | `'auto'` | — |
| `disagg_transfer_pool_size` | `ServerArgs`:621 | `int` | `256 * 1024 * 1024` | — |
| `disagg_transfer_pin_memory` | `ServerArgs`:622 | `Literal['auto', 'off', 'required']` | `'auto'` | — |
| `disagg_p2p_hostname` | `ServerArgs`:623 | `str` | `'127.0.0.1'` | — |
| `disagg_ib_device` | `ServerArgs`:624 | `str \| None` | `None` | — |
| `disagg_server_addr` | `ServerArgs`:625 | `str \| None` | `None` | — |
| `encoder_urls` | `ServerArgs`:626 | `str \| None` | `None` | — |
| `prompt_encoder_urls` | `ServerArgs`:627 | `str \| None` | `None` | — |
| `image_encoder_urls` | `ServerArgs`:628 | `str \| None` | `None` | — |
| `denoiser_urls` | `ServerArgs`:629 | `str \| None` | `None` | — |
| `decoder_urls` | `ServerArgs`:630 | `str \| None` | `None` | — |
| `encoder_tp` | `ServerArgs`:631 | `int \| None` | `None` | — |
| `denoiser_tp` | `ServerArgs`:632 | `int \| None` | `None` | — |
| `denoiser_sp` | `ServerArgs`:633 | `int \| None` | `None` | — |
| `denoiser_ulysses` | `ServerArgs`:634 | `int \| None` | `None` | — |
| `denoiser_ring` | `ServerArgs`:635 | `int \| None` | `None` | — |
| `decoder_sp` | `ServerArgs`:636 | `int \| None` | `None` | — |
| `pool_work_endpoint` | `ServerArgs`:637 | `str \| None` | `None` | — |
| `pool_result_endpoint` | `ServerArgs`:638 | `str \| None` | `None` | — |
| `log_level` | `ServerArgs`:641 | `str` | `'info'` | Logging |
| `log_requests` | `ServerArgs`:642 | `bool` | `False` | — |
| `log_requests_level` | `ServerArgs`:643 | `int` | `2` | — |
| `log_requests_format` | `ServerArgs`:644 | `str` | `'text'` | — |
| `log_requests_target` | `ServerArgs`:645 | `Optional[List[str]]` | `None` | — |
| `uvicorn_access_log_exclude_prefixes` | `ServerArgs`:646 | `list[str]` | `field(default_factory=list)` | — |
| `enable_cache_report` | `ServerArgs`:647 | `bool` | `False` | — |
| `disable_conditioning_cache` | `ServerArgs`:648 | `bool` | `False` | — |
| `conditioning_cache_max_size_mb` | `ServerArgs`:649 | `float` | `512.0` | — |
| `enable_trace` | `ServerArgs`:652 | `bool` | `False` | Tracing |
| `otlp_traces_endpoint` | `ServerArgs`:653 | `str` | `'localhost:4317'` | — |
| `otlp_service_name` | `ServerArgs`:654 | `str \| None` | `None` | — |
| `srt_encoder_url` | `ServerArgs`:657 | `str \| None` | `None` | SGLang backend for encoder stage |
| `srt_encoder_connect_timeout` | `ServerArgs`:658 | `int` | `3.05` | — |
| `srt_encoder_timeout` | `ServerArgs`:659 | `int` | `100` | — |
| `pe_server_url` | `ServerArgs`:662 | `str \| None` | `None` | SGLang server for PE model inference |
| `prompt_enhancer_config` | `ServerArgs`:663 | `str \| None` | `None` | — |
