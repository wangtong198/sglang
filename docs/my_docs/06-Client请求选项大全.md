# 06 Client 请求选项大全

每一行只讲这个请求字段。公开 JSON 在 HTTP 层被接住，再抄进 `GenerateReqInput` 或 `SamplingParams`。调度器最终看见的是 `Req`，不是这些类本身。
内部字段（下划线或 PrivateAttr）不列入本表。

## 原生 /generate

文件 `python/sglang/srt/managers/io_struct.py`。

### `GenerateReqInput`

定义于第 177 行，公开字段 72 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `rid` | `GenerateReqInput` | `field(default=None, kw_only=True)` | 作用：在 `GenerateReqInput` 里，`rid` 是请求 id。不传则服务生成。批里必须唯一。 默认与效果：默认空。 交互与改写：重复 id 会在规范化时抛错。 易错：一批里用了同一个 id。 |
| `session_id` | `GenerateReqInput` | `field(default=None, kw_only=True)` | 作用：在 `GenerateReqInput` 里，`session_id` 是同一会话的稳定 id。它不改写提示，和 session_params 不同。 默认与效果：默认空。 交互与改写：会话 radix 用它做引用。 易错：每条请求换一个 id，会话缓存用不上。 |
| `text` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`text` 是/generate 的提示文本。可以是一批字符串。 默认与效果：默认空。 交互与改写：和 input_ids、input_embeds 至少要有一种。 易错：三种同时给，规范化报错。 |
| `input_ids` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`input_ids` 是已经分好的 token id。单条是一维，批是二维。 默认与效果：默认空。 交互与改写：给了它就可以跳过文本分词。 易错：id 越界或用错词表。 |
| `input_embeds` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`input_embeds` 是直接给输入嵌入，代替 token。三维是一批。 默认与效果：默认空。 交互与改写：和 text、input_ids 三者只能有效提供一种。 易错：维度不是 hidden size。 |
| `image_data` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`image_data` 是图像输入，可以是路径、URL、base64，或它们的列表。批请求可以是列表的列表。 默认与效果：默认空。 交互与改写：服务会按 mm_hashes 或自己的哈希放进前缀键。 易错：URL 被域名白名单拒绝。 |
| `video_data` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`video_data` 是视频输入，路径、URL 或 base64。 默认与效果：默认空。 交互与改写：和 image_data 同一套多模态管道。 易错：文件过大，超过 media_url_max_file_size_mb。 |
| `audio_data` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`audio_data` 是音频输入，可以是路径、URL 或 base64。批处理时可以是列表。 默认与效果：默认空，表示没有音频。 交互与改写：和 image_data、video_data 一起进多模态处理器。 易错：URL 不在服务端允许的域名里。 |
| `mm_hashes` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`mm_hashes` 是调用方已经算好的特征哈希，用来和外部 KV 路由对齐。设置后不再对处理器输出重算哈希。 默认与效果：默认空，服务自己哈希。 交互与改写：必须和处理器输出一致，否则前缀键对不上。 易错：把文件 sha256 填进来，那是 content hash 不是特征哈希。 |
| `mm_content_hashes` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`mm_content_hashes` 是原始媒体字节的 sha256。不替换处理器输出的特征哈希。 默认与效果：默认空。 交互与改写：服务端 trust_mm_content_hashes 为真时才可能跳过读字节。 易错：哈希和字节不一致。 |
| `use_audio_in_video` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`use_audio_in_video` 是从视频里抽音频一起送进模型。 默认与效果：默认 false。 交互与改写：模型要同时有视觉和音频塔。 易错：纯视觉模型上打开。 |
| `video_config` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`video_config` 是这一请求的视频预处理配置。 默认与效果：默认空。 交互与改写：和 images_config 分开。 易错：键和处理器不一致。 |
| `sampling_params` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`sampling_params` 是/generate 内嵌的采样字典或按批的字典列表。 默认与效果：默认空，用 SamplingParams 默认值。 交互与改写：字段名要和 SamplingParams 一致。 易错：批的列表长度和提示条数不同。 |
| `watermark` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`watermark` 是这一请求是否加水印，或水印配置。服务端要先 enable_watermark。 默认与效果：默认空。 交互与改写：enforce_all 时请求关不掉。 易错：服务没开水印，字段被忽略。 |
| `return_logprob` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`return_logprob` 是/generate 是否返回对数概率。 默认与效果：默认空，当 false。 交互与改写：配合 logprob_start_len 和 top_logprobs_num。 易错：只开了这个，起点仍是 -1，没有提示段。 |
| `logprob_start_len` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`logprob_start_len` 是从提示的哪一位开始回 logprob。-1 只回输出。 默认与效果：默认 -1。 交互与改写：要提示 logprob 时设 0。 易错：留着 -1 却做 RL。 |
| `top_logprobs_num` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`top_logprobs_num` 是/generate 要的候选个数。 默认与效果：默认空。 交互与改写：和 logprob_start_len 一起决定范围。 易错：设得很大。 |
| `token_ids_logprob` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`token_ids_logprob` 是除了采样到的 token，还要这些 id 的 logprob。 默认与效果：默认空。 交互与改写：和 top_logprobs_num 一起返回。 易错：id 很多，响应巨大。 |
| `return_sampling_mask` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`return_sampling_mask` 是返回每个输出 token 的采样支持集。 默认与效果：默认 false。 交互与改写：支持集大小受服务端 sampling_mask_max_tokens 限制。 易错：top_k 大于上限，请求被中止。 |
| `sampling_logprobs_mode` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`sampling_logprobs_mode` 是采样 logprob 返回选中 token，还是整个支持集。不填且开了 mask 时用选中模式。 默认与效果：默认空。 交互与改写：取值 selected 或 support。 易错：写成了别的字符串。 |
| `return_text_in_logprobs` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_text_in_logprobs` 是logprob 里除了 id 还带解码后的文本。 默认与效果：默认 false。 交互与改写：文本按 skip_special_tokens 的规则解。 易错：特殊符被跳过，对不齐 id。 |
| `return_flat_raw_top_logprobs` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_flat_raw_top_logprobs` 是/generate 把提示段 top logprob 打成扁平数组加形状。 默认与效果：默认 false。 交互与改写：需要 return_logprob。 易错：没开 logprob。 |
| `return_flat_raw_output_top_logprobs` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_flat_raw_output_top_logprobs` 是把输出的 top logprob 打成扁平数组，放在 meta_info，choices 里的 top_logprobs 则为空。 默认与效果：默认 false。 交互与改写：chat 上还常配合 b64。 易错：两边都解析，得到空列表。 |
| `return_flat_raw_top_logprobs_b64` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_flat_raw_top_logprobs_b64` 是扁平数组再 base64。/generate 要先开上面的扁平标志；chat 默认就是 true。 默认与效果：chat 默认 true，generate 默认 false。 交互与改写：false 时返回普通 JSON 列表。 易错：没开扁平却要求 b64。 |
| `stream` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`stream` 是否流式。结束帧是 data: [DONE]。Ollama chat 默认 true。 默认与效果：多数默认 false。 交互与改写：间隔还受 stream_interval 影响。 易错：按非流式去读一个不关闭的流。 |
| `log_metrics` | `GenerateReqInput` | `True` | 作用：在 `GenerateReqInput` 里，`log_metrics` 是这条请求算不算进指标。健康检查会把它设为 false。 默认与效果：默认 true。 交互与改写：和 no_logs 不同，一个是指标一个是日志。 易错：把用户请求设成 false，监控缺数。 |
| `return_hidden_states` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_hidden_states` 是要隐状态。可以是布尔或 last。服务端还要开对应模式。 默认与效果：默认 false。 交互与改写：full 很占带宽。辅助隐状态打开时返回的是拼接 aux。 易错：服务模式是 last，请求要了全部。 |
| `return_routed_experts` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_routed_experts` 是返回每层选中的专家。 默认与效果：默认 false。 交互与改写：服务端开关也要开。起点由 routed_experts_start_len 决定。 易错：只在请求里开。 |
| `routed_experts_start_len` | `GenerateReqInput` | `0` | 作用：在 `GenerateReqInput` 里，`routed_experts_start_len` 是从哪个绝对位置开始返回专家路由，区间是到序列末尾。0 表示整段。必须落在 0 到提示长度。 默认与效果：默认 0。 交互与改写：只在 return_routed_experts 为真时有用。 易错：设得超过提示长度。 |
| `return_indexer_topk` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_indexer_topk` 是返回带 indexer 的层的 top-k 下标。 默认与效果：默认 false。 交互与改写：服务端 enable_return_indexer_topk 也要开。 易错：普通模型没有 indexer，结果为空。 |
| `modalities` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`modalities` 是声明图像、多图或视频等模态。 默认与效果：默认空，从数据推断。 交互与改写：和真正携带的 image_data 不一致时处理器会困惑。 易错：声明了视频却没给 video_data。 |
| `session_params` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`session_params` 是连续提示的会话参数，可能改写或重建提示。 默认与效果：默认空。 交互与改写：和 session_id 不是一回事。 易错：参数让服务重写了用户没预期的历史。 |
| `lora_path` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`lora_path` 是这条请求使用的适配器路径或名字。可以按批给列表。 默认与效果：默认空，用基座。 交互与改写：服务端要已经 enable_lora 并加载过。 易错：路径在服务进程上看不见。 |
| `lora_id` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`lora_id` 是适配器在服务端的 uid，通常由 tokenizer 填，而不是客户端猜。 默认与效果：默认空。 交互与改写：和 lora_path 二选一，path 会解析成 id。 易错：传了未加载的 id。 |
| `custom_logit_processor` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`custom_logit_processor` 是序列化后的自定义 logit 处理器。服务端必须开 enable_custom_logit_processor。 默认与效果：默认空。 交互与改写：等于允许请求在进程里执行代码，默认关是为了安全。 易错：服务没开开关，处理器被忽略。 |
| `positional_embed_overrides` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`positional_embed_overrides` 是在指定 token 位置替换嵌入。HTTP 侧常为了避开张量而标成 Any。 默认与效果：默认空。 交互与改写：长度和位置必须落在序列内。 易错：把整段 hidden 传上来，形状不对。 |
| `bootstrap_host` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`bootstrap_host` 是PD 分离时 prefill 的主机名，decode 靠它来要 KV。 默认与效果：默认空。 交互与改写：合部服务不需要。要和 bootstrap_port、room 一起用。 易错：写成了客户端自己的地址。 |
| `bootstrap_port` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`bootstrap_port` 是prefill 上 bootstrap 服务的端口，默认服务旗标是 8998。 默认与效果：默认空，表示用服务端配置的端口。 交互与改写：请求级可以覆盖。 易错：写成 HTTP 30000。 |
| `bootstrap_room` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`bootstrap_room` 是同一条对话在 PD 两侧的房间号，负载策略 follow_bootstrap_room 会把它送到同一组。 默认与效果：默认空。 交互与改写：和 routed_dp_rank 一起决定落点。 易错：每条请求随机 room，前缀永远对不齐。 |
| `bootstrap_pair_key` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`bootstrap_pair_key` 是给一对 prefill/decode 请求的配对键，比单用 room 更明确。 默认与效果：默认空。 交互与改写：外部编排生成，两边必须相同。 易错：只在一边填写。 |
| `decode_tp_size` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`decode_tp_size` 是告诉 prefill，对端 decode 的 TP 是多少，便于 KV 布局。 默认与效果：默认空。 交互与改写：和本机 tp 不一致时，校验和可能跳过。 易错：写错宽度，decode 无法解释传来的页。 |
| `routed_dp_rank` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`routed_dp_rank` 是外部路由器指定的注意力 DP 组。 默认与效果：默认空，由负载策略选。 交互与改写：旧名 data_parallel_rank 仍接受。 易错：指定的组号大于等于组数。 |
| `data_parallel_rank` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`data_parallel_rank` 是routed_dp_rank 的旧名。网关仍会塞这个字段。 默认与效果：默认空。 交互与改写：两边都写时以新字段为准或同时被接受。 易错：两个字段写成不同的组号。 |
| `disagg_prefill_dp_rank` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`disagg_prefill_dp_rank` 是提示 decode：KV 在哪一个 prefill 的 DP 组上。 默认与效果：默认空。 交互与改写：外部路由器填，应用客户端不要自己猜。 易错：组和真实存放 KV 的 rank 不一致。 |
| `routing_key` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`routing_key` 是给路由键调度策略用的键，相同键尽量进同一批。 默认与效果：默认空。 交互与改写：和前缀缓存不是同一个键。 易错：每个请求随机，策略失效。 |
| `conversation_id` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`conversation_id` 是用来把多条请求记成同一会话，便于追踪。 默认与效果：默认空。 交互与改写：和 session_id 不同，它不重建提示。 易错：以为它会自动带上历史消息。 |
| `http_worker_ipc` | `GenerateReqInput` | `field(default=None, kw_only=True)` | 作用：在 `GenerateReqInput` 里，`http_worker_ipc` 是多 tokenizer 时，这条请求属于哪个 HTTP worker 的回包通道。内部填写。 默认与效果：默认空。 交互与改写：客户端不要传。 易错：传了一个不存在的 IPC 名，结果没有人收。 |
| `background` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`background` 是Responses API 的请求放后台跑，先返回 id 再取结果。 默认与效果：默认 false。 交互与改写：需要服务端 enable_response_store，否则结果没处存。 易错：没开存储却发 background。 |
| `require_reasoning` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`require_reasoning` 是混合思考模型必须先走思考段。 默认与效果：默认 false。 交互与改写：要和严格思考、预算一起才强制。 易错：模型没有思考模板仍设为真。 |
| `max_thinking_tokens` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`max_thinking_tokens` 是思考阶段的 token 预算。要服务端严格思考模式才会强制截断。 默认与效果：默认空，不限制思考。 交互与改写：不开启 enable_strict_thinking 时只是元数据。 易错：以为预算生效，模型一直想。 |
| `priority` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`priority` 是请求优先级。服务端没开优先级调度时可能被忽略或拒绝。 默认与效果：默认空。 交互与改写：数值方向由 schedule_low_priority_values_first 决定。 易错：以为越大越优先，服务却是小的先跑。 |
| `extra_key` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`extra_key` 是额外的缓存命名空间，和 cache_salt 类似但给路由分类用。 默认与效果：默认空。 交互与改写：不同 extra_key 不共享前缀页。 易错：每个请求都随机，缓存命中率为 0。 |
| `no_logs` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`no_logs` 是这条请求不落日志，例如零数据保留。 默认与效果：默认 false。 交互与改写：和 log_metrics 是两件事。 易错：以为不打日志就不计费或不进指标。 |
| `custom_labels` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`custom_labels` 是给这条请求的指标打的标签。 默认与效果：默认空。 交互与改写：标签名还必须在服务端允许列表里。 易错：高基数字段当标签，指标爆炸。 |
| `return_bytes` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_bytes` 是图像类生成时返回字节。内部用途。 默认与效果：默认 false。 交互与改写：文本模型没有字节可回。 易错：文本请求设为真，响应类型错。 |
| `return_entropy` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_entropy` 是返回采样分布的熵。 默认与效果：默认 false。 交互与改写：需要保留分布，不能只留 argmax。 易错：贪心时熵没有意义仍打开。 |
| `return_prompt_token_ids` | `GenerateReqInput` | `False` | 作用：在 `GenerateReqInput` 里，`return_prompt_token_ids` 是返回提示的 token id，不必算 logprob。 默认与效果：默认 false。 交互与改写：和 return_logprob 独立。 易错：以为开了它就有概率。 |
| `external_trace_header` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`external_trace_header` 是把外部追踪上下文带进这次生成。 默认与效果：默认空。 交互与改写：服务端要开 trace 才会继续传播。 易错：头里的 trace id 格式不对。 |
| `received_time` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`received_time` 是请求到达服务的时间戳，用于延迟统计。通常由服务填写。 默认与效果：默认空。 交互与改写：客户端乱填会污染 TTFT。 易错：用了毫秒时间戳，服务当秒。 |
| `need_wait_for_mm_inputs` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`need_wait_for_mm_inputs` 是要等编码器把多模态特征送回再继续。 默认与效果：默认空。 交互与改写：编码器分离时由 tokenizer 填写。 易错：客户端设成 false，特征还没到就开始生成。 |
| `num_items_assigned` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`num_items_assigned` 是按模态分配给编码器的条目数，由 tokenizer 计算。 默认与效果：默认空。 交互与改写：和 encoder_urls 是同一时刻的快照。 易错：客户端自己填，和真实图片数不符。 |
| `encoder_urls` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`encoder_urls` 是这条请求要使用的编码器 URL 快照。 默认与效果：默认空。 交互与改写：由 tokenizer 在分配多模态条目时填上。 易错：客户端乱填，请求被派到不存在的编码器。 |
| `max_dynamic_patch` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`max_dynamic_patch` 是这一请求最多把图像切成多少动态块。 默认与效果：默认空，用模型默认。 交互与改写：图像和视频还可以分别再限。 易错：块数过大，视觉 token 超过 max_image_tokens。 |
| `min_dynamic_patch` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`min_dynamic_patch` 是动态切图至少多少块。 默认与效果：默认空。 交互与改写：不能大于 max。 易错：最小大于最大。 |
| `image_max_dynamic_patch` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`image_max_dynamic_patch` 是这一路图像最多切成多少动态块。 默认与效果：默认空，用模型默认。 交互与改写：和 max_dynamic_patch 是总开关和图像专用的关系。 易错：块数超过视觉 token 上限，被截断。 |
| `video_max_dynamic_patch` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`video_max_dynamic_patch` 是视频最多切多少动态块。 默认与效果：默认空。 交互与改写：和图像的块上限分开。 易错：块数把视觉 token 打满。 |
| `images_config` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`images_config` 是这一请求的图像预处理配置，覆盖服务端 mm_process_config。 默认与效果：默认空。 交互与改写：键要和处理器认识的一致。 易错：写成了视频配置的键。 |
| `mm_processor_kwargs` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`mm_processor_kwargs` 是这一请求传给多模态处理器的参数。 默认与效果：默认空。 交互与改写：覆盖服务端 mm_process_config。 易错：键名处理器不认识。 |
| `multi_item_delimiter_indices` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`multi_item_delimiter_indices` 是多条目打分时，条目之间分隔符的 token 位置。 默认与效果：默认空。 交互与改写：批是每个请求一个列表。 易错：位置越界。 |
| `token_indices_to_pool` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`token_indices_to_pool` 是在这些位置读 logit 或池化，而不是最后一个 token。 默认与效果：默认空。 交互与改写：批是每个请求一个列表。 易错：位置超过序列长度。 |
| `cache_salt` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`cache_salt` 是给前缀缓存加盐，相同 token 也不会命中另一盐的页。 默认与效果：默认空，不加盐。 交互与改写：和 extra_key、LoRA 名一起组成缓存键。 易错：把用户隐私写进盐里还打进日志。 |
| `kv_hints` | `GenerateReqInput` | `None` | 作用：在 `GenerateReqInput` 里，`kv_hints` 是编排器给的 KV 提示信封，应用客户端不要填。各存储后端只认自己的动作。 默认与效果：默认空。 交互与改写：批请求是每个请求一个信封。 易错：业务请求自己伪造 hint，和真实前缀不符。 |

## 原生 /encode 与 /classify

文件 `python/sglang/srt/managers/io_struct.py`。

### `EmbeddingReqInput`

定义于第 1204 行，公开字段 28 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `rid` | `EmbeddingReqInput` | `field(default=None, kw_only=True)` | 作用：在 `EmbeddingReqInput` 里，`rid` 是请求 id。不传则服务生成。批里必须唯一。 默认与效果：默认空。 交互与改写：重复 id 会在规范化时抛错。 易错：一批里用了同一个 id。 |
| `text` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`text` 是/generate 的提示文本。可以是一批字符串。 默认与效果：默认空。 交互与改写：和 input_ids、input_embeds 至少要有一种。 易错：三种同时给，规范化报错。 |
| `input_ids` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`input_ids` 是已经分好的 token id。单条是一维，批是二维。 默认与效果：默认空。 交互与改写：给了它就可以跳过文本分词。 易错：id 越界或用错词表。 |
| `input_embeds` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`input_embeds` 是直接给输入嵌入，代替 token。三维是一批。 默认与效果：默认空。 交互与改写：和 text、input_ids 三者只能有效提供一种。 易错：维度不是 hidden size。 |
| `image_data` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`image_data` 是图像输入，可以是路径、URL、base64，或它们的列表。批请求可以是列表的列表。 默认与效果：默认空。 交互与改写：服务会按 mm_hashes 或自己的哈希放进前缀键。 易错：URL 被域名白名单拒绝。 |
| `video_data` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`video_data` 是视频输入，路径、URL 或 base64。 默认与效果：默认空。 交互与改写：和 image_data 同一套多模态管道。 易错：文件过大，超过 media_url_max_file_size_mb。 |
| `audio_data` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`audio_data` 是音频输入，可以是路径、URL 或 base64。批处理时可以是列表。 默认与效果：默认空，表示没有音频。 交互与改写：和 image_data、video_data 一起进多模态处理器。 易错：URL 不在服务端允许的域名里。 |
| `embed_override_token_id` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`embed_override_token_id` 是嵌入时在这个 token 位置读取向量，而不是默认位置。 默认与效果：默认空。 交互与改写：和 embed_overrides 一起改读分点。 易错：id 不在序列里。 |
| `embed_overrides` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`embed_overrides` 是按位置覆盖嵌入向量。 默认与效果：默认空。 交互与改写：类型里可能含张量，HTTP 通常传列表。 易错：长度和序列不一致。 |
| `sampling_params` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`sampling_params` 是/generate 内嵌的采样字典或按批的字典列表。 默认与效果：默认空，用 SamplingParams 默认值。 交互与改写：字段名要和 SamplingParams 一致。 易错：批的列表长度和提示条数不同。 |
| `log_metrics` | `EmbeddingReqInput` | `True` | 作用：在 `EmbeddingReqInput` 里，`log_metrics` 是这条请求算不算进指标。健康检查会把它设为 false。 默认与效果：默认 true。 交互与改写：和 no_logs 不同，一个是指标一个是日志。 易错：把用户请求设成 false，监控缺数。 |
| `modalities` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`modalities` 是声明图像、多图或视频等模态。 默认与效果：默认空，从数据推断。 交互与改写：和真正携带的 image_data 不一致时处理器会困惑。 易错：声明了视频却没给 video_data。 |
| `is_cross_encoder_request` | `EmbeddingReqInput` | `False` | 作用：在 `EmbeddingReqInput` 里，`is_cross_encoder_request` 是这条嵌入请求按交叉编码器打分，而不是普通向量。 默认与效果：默认 false。 交互与改写：需要 token_type_ids 一类的成对输入。 易错：普通向量模型上设为真。 |
| `lora_path` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`lora_path` 是这条请求使用的适配器路径或名字。可以按批给列表。 默认与效果：默认空，用基座。 交互与改写：服务端要已经 enable_lora 并加载过。 易错：路径在服务进程上看不见。 |
| `lora_id` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`lora_id` 是适配器在服务端的 uid，通常由 tokenizer 填，而不是客户端猜。 默认与效果：默认空。 交互与改写：和 lora_path 二选一，path 会解析成 id。 易错：传了未加载的 id。 |
| `positional_embed_overrides` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`positional_embed_overrides` 是在指定 token 位置替换嵌入。HTTP 侧常为了避开张量而标成 Any。 默认与效果：默认空。 交互与改写：长度和位置必须落在序列内。 易错：把整段 hidden 传上来，形状不对。 |
| `routing_key` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`routing_key` 是给路由键调度策略用的键，相同键尽量进同一批。 默认与效果：默认空。 交互与改写：和前缀缓存不是同一个键。 易错：每个请求随机，策略失效。 |
| `http_worker_ipc` | `EmbeddingReqInput` | `field(default=None, kw_only=True)` | 作用：在 `EmbeddingReqInput` 里，`http_worker_ipc` 是多 tokenizer 时，这条请求属于哪个 HTTP worker 的回包通道。内部填写。 默认与效果：默认空。 交互与改写：客户端不要传。 易错：传了一个不存在的 IPC 名，结果没有人收。 |
| `background` | `EmbeddingReqInput` | `False` | 作用：在 `EmbeddingReqInput` 里，`background` 是Responses API 的请求放后台跑，先返回 id 再取结果。 默认与效果：默认 false。 交互与改写：需要服务端 enable_response_store，否则结果没处存。 易错：没开存储却发 background。 |
| `priority` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`priority` 是请求优先级。服务端没开优先级调度时可能被忽略或拒绝。 默认与效果：默认空。 交互与改写：数值方向由 schedule_low_priority_values_first 决定。 易错：以为越大越优先，服务却是小的先跑。 |
| `dimensions` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`dimensions` 是嵌入向量要截到多少维。 默认与效果：默认空，返回全维。 交互与改写：模型隐藏维必须不小于它。 易错：设得比隐藏维大。 |
| `return_pooled_hidden_states` | `EmbeddingReqInput` | `False` | 作用：在 `EmbeddingReqInput` 里，`return_pooled_hidden_states` 是嵌入接口返回池化后的向量，而不是每个 token 一个。 默认与效果：默认按接口。 交互与改写：dimensions 可以再截断。 易错：池化方式和模型不一致。 |
| `return_prompt_token_ids` | `EmbeddingReqInput` | `False` | 作用：在 `EmbeddingReqInput` 里，`return_prompt_token_ids` 是返回提示的 token id，不必算 logprob。 默认与效果：默认 false。 交互与改写：和 return_logprob 独立。 易错：以为开了它就有概率。 |
| `external_trace_header` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`external_trace_header` 是把外部追踪上下文带进这次生成。 默认与效果：默认空。 交互与改写：服务端要开 trace 才会继续传播。 易错：头里的 trace id 格式不对。 |
| `received_time` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`received_time` 是请求到达服务的时间戳，用于延迟统计。通常由服务填写。 默认与效果：默认空。 交互与改写：客户端乱填会污染 TTFT。 易错：用了毫秒时间戳，服务当秒。 |
| `multi_item_delimiter_indices` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`multi_item_delimiter_indices` 是多条目打分时，条目之间分隔符的 token 位置。 默认与效果：默认空。 交互与改写：批是每个请求一个列表。 易错：位置越界。 |
| `token_indices_to_pool` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`token_indices_to_pool` 是在这些位置读 logit 或池化，而不是最后一个 token。 默认与效果：默认空。 交互与改写：批是每个请求一个列表。 易错：位置超过序列长度。 |
| `decision_layout` | `EmbeddingReqInput` | `None` | 作用：在 `EmbeddingReqInput` 里，`decision_layout` 是决策接口里题目的排布，决定从哪几个 logit 读答案。 默认与效果：默认空。 交互与改写：和 token_indices_to_pool 一起指出读分位置。 易错：布局和实际提示对不上。 |

## sampling_params（/generate 内嵌，也是调度器真正消费的采样结构）

文件 `python/sglang/srt/sampling/sampling_params.py`。

### `SamplingParams`

定义于第 140 行，公开字段 33 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `max_new_tokens` | `SamplingParams` | `128` | 作用：在 `SamplingParams` 里，`max_new_tokens` 是采样器真正使用的生成长度。/generate 不传时是 128。 默认与效果：默认 128。 交互与改写：OpenAI 的长度字段会写进这里。 易错：HTTP 改了 max_tokens，这里仍是 128，说明转换没走到。 |
| `stop` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`stop` 是停止字符串。采样侧会拷到 stop_strs 再清空。 默认与效果：默认空。 交互与改写：和 stop_token_ids 一起生效。 易错：停止串出现在思考里。 |
| `stop_token_ids` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`stop_token_ids` 是遇到这些 id 就停。 默认与效果：默认空。 交互与改写：并进采样参数的集合。 易错：id 用错词表。 |
| `stop_regex` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`stop_regex` 是停止正则。 默认与效果：默认空。 交互与改写：拷进 stop_regex_strs。 易错：正则太宽，第一个 token 就停。 |
| `temperature` | `SamplingParams` | `1.0` | 作用：在 `SamplingParams` 里，`temperature` 是采样温度。越高越随机，接近 0 趋于贪心。 默认与效果：SamplingParams 默认 1.0；HTTP 空值会收成 1.0。 交互与改写：和 top_p、top_k 一起用。 易错：不传被当成贪心。 |
| `top_p` | `SamplingParams` | `1.0` | 作用：在 `SamplingParams` 里，`top_p` 是核采样。1 表示不截断。 默认与效果：默认 1。 交互与改写：通常在温度之后应用。 易错：设成 0。 |
| `top_k` | `SamplingParams` | `TOP_K_ALL` | 作用：在 `SamplingParams` 里，`top_k` 是只在概率最高的 k 个里采样。 默认与效果：默认不限制。 交互与改写：采样掩码上限要大于它。 易错：设成 0。 |
| `min_p` | `SamplingParams` | `0.0` | 作用：在 `SamplingParams` 里，`min_p` 是丢掉概率低于该值的 token。 默认与效果：默认 0，不丢。 交互与改写：在 top-k/top-p 之后再滤。 易错：接近 1 时几乎只剩一个 token。 |
| `frequency_penalty` | `SamplingParams` | `0.0` | 作用：在 `SamplingParams` 里，`frequency_penalty` 是按出现次数惩罚 token，减轻重复。 默认与效果：默认 0。 交互与改写：可与 presence、repetition 叠加。 易错：设得过大，句子被逼换词。 |
| `presence_penalty` | `SamplingParams` | `0.0` | 作用：在 `SamplingParams` 里，`presence_penalty` 是token 只要出现过就惩罚，不看次数。 默认与效果：默认 0。 交互与改写：可与 frequency 叠加。 易错：两个都很大，分布被压扁。 |
| `repetition_penalty` | `SamplingParams` | `1.0` | 作用：在 `SamplingParams` 里，`repetition_penalty` 是按重复缩放 logits。1 表示不惩罚。 默认与效果：默认 1。 交互与改写：和 frequency 的公式不同。 易错：设成 0。 |
| `min_new_tokens` | `SamplingParams` | `0` | 作用：在 `SamplingParams` 里，`min_new_tokens` 是至少生成这么多 token 才允许 EOS 停。 默认与效果：默认 0。 交互与改写：和 ignore_eos 不同，到数之后 EOS 仍有效。 易错：设得比 max 还大。 |
| `n` | `SamplingParams` | `1` | 作用：在 `SamplingParams` 里，`n` 是并行样本数。beam 时变成返回条数且不能超过束宽。 默认与效果：默认 1。 交互与改写：每条样本一份 KV。 易错：n 大于 beam_width。 |
| `beam_width` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`beam_width` 是大于 1 时这次请求变成 beam search。n 变成返回条数，且不能超过束宽。 默认与效果：默认空，不做 beam。 交互与改写：束搜索占用多份 KV。 易错：n 大于 beam_width。 |
| `json_schema` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`json_schema` 是用 JSON Schema 约束输出。 默认与效果：默认空。 交互与改写：会编译成语法，和 response_format 的 json_schema 是同一意图。 易错：schema 含远程 $ref，编译失败。 |
| `regex` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`regex` 是用正则约束输出。 默认与效果：默认空。 交互与改写：和 json_schema、ebnf 不要同时给。 易错：正则灾难性回溯。 |
| `ebnf` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`ebnf` 是用 EBNF 约束输出格式。 默认与效果：默认空。 交互与改写：和 json_schema、regex 同时给时通常只该有一个。 易错：语法有歧义，编译失败。 |
| `structural_tag` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`structural_tag` 是用结构化标签约束输出。 默认与效果：默认空。 交互与改写：和 json_schema 不要同时给。 易错：标签和解析器不配套。 |
| `ignore_eos` | `SamplingParams` | `False` | 作用：在 `SamplingParams` 里，`ignore_eos` 是为真时 EOS 不结束生成，直到达到长度上限。 默认与效果：默认 false。 交互与改写：和 stop 串仍会生效，除非一并清掉。 易错：打开后回答停不下来，打满 max_tokens。 |
| `skip_special_tokens` | `SamplingParams` | `True` | 作用：在 `SamplingParams` 里，`skip_special_tokens` 是解文本时跳过特殊符。 默认与效果：默认 true。 交互与改写：和 return_text_in_logprobs 一起决定 logprob 里的文本。 易错：关掉后特殊符泄露到用户可见文本。 |
| `spaces_between_special_tokens` | `SamplingParams` | `True` | 作用：在 `SamplingParams` 里，`spaces_between_special_tokens` 是特殊符之间是否加空格。 默认与效果：默认 true。 交互与改写：只影响 detokenize。 易错：和模板的空格规则不一致，看起来多了空格。 |
| `no_stop_trim` | `SamplingParams` | `False` | 作用：在 `SamplingParams` 里，`no_stop_trim` 是停止串命中后不要把它从返回文本里裁掉。 默认与效果：默认 false，会裁掉。 交互与改写：只影响返回文本，不影响停的位置。 易错：客户端自己再裁一次，多裁了正文。 |
| `stream_interval` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`stream_interval` 是每隔多少个 token 推一帧。 默认与效果：默认空，用服务端 stream_interval。 交互与改写：设得越大，客户端越久没增量。 易错：设成 0。 |
| `logit_bias` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`logit_bias` 是按 token 字符串 id 加减分。 默认与效果：默认空。 交互与改写：在采样前加到 logits。 易错：键不是十进制 id。 |
| `sampling_seed` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`sampling_seed` 是SamplingParams 里的种子。 默认与效果：默认空。 交互与改写：对应 HTTP 的 seed。 易错：只设种子，服务非确定性。 |
| `custom_params` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`custom_params` 是采样器的扩展参数字典。 默认与效果：默认空。 交互与改写：只有对应后端认识的键才生效。 易错：把标准温度写进这里而不是 temperature。 |
| `watermark` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`watermark` 是这一请求是否加水印，或水印配置。服务端要先 enable_watermark。 默认与效果：默认空。 交互与改写：enforce_all 时请求关不掉。 易错：服务没开水印，字段被忽略。 |
| `stop_strs` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`stop_strs` 是内部：从 stop 拷来的停止串。 默认与效果：默认空。 交互与改写：采样循环读的是这个而不是原始 stop。 易错：只改了 stop_strs，HTTP 下次仍写 stop。 |
| `stop_regex_strs` | `SamplingParams` | `None` | 作用：在 `SamplingParams` 里，`stop_regex_strs` 是内部：从 stop_regex 拷来的正则。 默认与效果：默认空。 交互与改写：normalize 之后才有。 易错：和 stop_regex 同时传不同值。 |
| `stop_str_max_len` | `SamplingParams` | `0` | 作用：在 `SamplingParams` 里，`stop_str_max_len` 是内部：停止字符串要回看的长度。 默认与效果：默认 0。 交互与改写：由 normalize 填。 易错：手改它。 |
| `stop_regex_max_len` | `SamplingParams` | `0` | 作用：在 `SamplingParams` 里，`stop_regex_max_len` 是内部：停止正则需要回看的最大长度。normalize 填写。 默认与效果：默认 0。 交互与改写：调用方不要传。 易错：手写一个太小的值，停不住。 |
| `is_normalized` | `SamplingParams` | `False` | 作用：在 `SamplingParams` 里，`is_normalized` 是SamplingParams 的内部标记，normalize 之后为真，避免再次重置停止串。 默认与效果：默认 false。 交互与改写：调用方不要传。 易错：手写成 true，停止串没有被填上。 |
| `ebnf_full_assistant` | `SamplingParams` | `False` | 作用：在 `SamplingParams` 里，`ebnf_full_assistant` 是内部标记，EBNF 是否覆盖整个助手回复。由 normalize 填写。 默认与效果：默认 false。 交互与改写：调用方不要手写。 易错：手写成 true，语法范围和模板不一致。 |

## OpenAI Chat Completions `/v1/chat/completions`

文件 `python/sglang/srt/entrypoints/openai/protocol.py`。

### `ChatCompletionRequest`

定义于第 909 行，公开字段 76 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `bootstrap_host` | `PDRoutingFields` | `None` | 作用：在 `ChatCompletionRequest` 里，`bootstrap_host` 是PD 分离时 prefill 的主机名，decode 靠它来要 KV。 默认与效果：默认空。 交互与改写：合部服务不需要。要和 bootstrap_port、room 一起用。 易错：写成了客户端自己的地址。 |
| `bootstrap_port` | `PDRoutingFields` | `None` | 作用：在 `ChatCompletionRequest` 里，`bootstrap_port` 是prefill 上 bootstrap 服务的端口，默认服务旗标是 8998。 默认与效果：默认空，表示用服务端配置的端口。 交互与改写：请求级可以覆盖。 易错：写成 HTTP 30000。 |
| `bootstrap_room` | `PDRoutingFields` | `None` | 作用：在 `ChatCompletionRequest` 里，`bootstrap_room` 是同一条对话在 PD 两侧的房间号，负载策略 follow_bootstrap_room 会把它送到同一组。 默认与效果：默认空。 交互与改写：和 routed_dp_rank 一起决定落点。 易错：每条请求随机 room，前缀永远对不齐。 |
| `routed_dp_rank` | `PDRoutingFields` | `None` | 作用：在 `ChatCompletionRequest` 里，`routed_dp_rank` 是外部路由器指定的注意力 DP 组。 默认与效果：默认空，由负载策略选。 交互与改写：旧名 data_parallel_rank 仍接受。 易错：指定的组号大于等于组数。 |
| `disagg_prefill_dp_rank` | `PDRoutingFields` | `None` | 作用：在 `ChatCompletionRequest` 里，`disagg_prefill_dp_rank` 是提示 decode：KV 在哪一个 prefill 的 DP 组上。 默认与效果：默认空。 交互与改写：外部路由器填，应用客户端不要自己猜。 易错：组和真实存放 KV 的 rank 不一致。 |
| `messages` | `ChatCompletionRequest` | `（必填）` | 作用：在 `ChatCompletionRequest` 里，`messages` 是对话列表。服务用 chat template 展开成 token。内容可以是文本或媒体块。 默认与效果：必填。 交互与改写：工具、思考、模板参数都读这里。 易错：role 写错或 content 结构不符合该模型模板。 |
| `model` | `ChatCompletionRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `ChatCompletionRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `frequency_penalty` | `ChatCompletionRequest` | `0.0` | 作用：在 `ChatCompletionRequest` 里，`frequency_penalty` 是按出现次数惩罚 token，减轻重复。 默认与效果：默认 0。 交互与改写：可与 presence、repetition 叠加。 易错：设得过大，句子被逼换词。 |
| `logit_bias` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`logit_bias` 是按 token 字符串 id 加减分。 默认与效果：默认空。 交互与改写：在采样前加到 logits。 易错：键不是十进制 id。 |
| `logprobs` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`logprobs` 是chat 是布尔；补全是要返回的个数。 默认与效果：chat 默认 false。 交互与改写：还要 top_logprobs 才有候选。 易错：类型用错。 |
| `top_logprobs` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`top_logprobs` 是每个位置返回几个候选的 logprob。 默认与效果：默认空。 交互与改写：要先开 logprobs。 易错：数量超过服务端上限。 |
| `max_tokens` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`max_tokens` 是旧的长度上限。chat 应改用 max_completion_tokens。补全接口默认常常是 16。Anthropic 用它且必须为正。 默认与效果：chat 默认空。 交互与改写：会抄进 max_new_tokens。 易错：补全没写它，只生成 16 个 token。 |
| `max_completion_tokens` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`max_completion_tokens` 是 chat 的生成上限，含可见输出和思考 token，不含输入。 默认与效果：默认空，回落到 SamplingParams 的 128 或服务端默认。 交互与改写：旧字段 max_tokens 已废弃。 易错：设得比思考还短，正文为空。 |
| `n` | `ChatCompletionRequest` | `1` | 作用：在 `ChatCompletionRequest` 里，`n` 是并行样本数。beam 时变成返回条数且不能超过束宽。 默认与效果：默认 1。 交互与改写：每条样本一份 KV。 易错：n 大于 beam_width。 |
| `presence_penalty` | `ChatCompletionRequest` | `0.0` | 作用：在 `ChatCompletionRequest` 里，`presence_penalty` 是token 只要出现过就惩罚，不看次数。 默认与效果：默认 0。 交互与改写：可与 frequency 叠加。 易错：两个都很大，分布被压扁。 |
| `response_format` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`response_format` 是输出格式，例如 json_object 或 json_schema。转写接口里则是 json、text、verbose。 默认与效果：默认空或 json。 交互与改写：会变成语法约束。 易错：schema 和格式类型不匹配。 |
| `seed` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`seed` 是请求级随机种子，对应 sampling_seed。 默认与效果：默认空。 交互与改写：确定性还要服务端开关。 易错：种子用了字符串。 |
| `stop` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`stop` 是停止字符串。采样侧会拷到 stop_strs 再清空。 默认与效果：默认空。 交互与改写：和 stop_token_ids 一起生效。 易错：停止串出现在思考里。 |
| `stream` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`stream` 是否流式。结束帧是 data: [DONE]。Ollama chat 默认 true。 默认与效果：多数默认 false。 交互与改写：间隔还受 stream_interval 影响。 易错：按非流式去读一个不关闭的流。 |
| `stream_options` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`stream_options` 是流式附加选项，例如最后是否带 usage。 默认与效果：默认空。 交互与改写：只在 stream 为真时生效。 易错：非流式也传了它。 |
| `temperature` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`temperature` 是采样温度。越高越随机，接近 0 趋于贪心。 默认与效果：SamplingParams 默认 1.0；HTTP 空值会收成 1.0。 交互与改写：和 top_p、top_k 一起用。 易错：不传被当成贪心。 |
| `top_p` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`top_p` 是核采样。1 表示不截断。 默认与效果：默认 1。 交互与改写：通常在温度之后应用。 易错：设成 0。 |
| `user` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`user` 是终端用户标识，只用于审计。 默认与效果：默认空。 交互与改写：不会自动变成缓存盐。 易错：把密钥写进 user。 |
| `tools` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`tools` 是可调用的工具定义列表。 默认与效果：默认空。 交互与改写：模板负责渲染，解析器负责拆调用。 易错：schema 不是 JSON Schema。 |
| `tool_choice` | `ChatCompletionRequest` | `'auto'` | 作用：在 `ChatCompletionRequest` 里，`tool_choice` 是工具选择：auto、required、none，或指定某个函数。 默认与效果：默认 auto。 交互与改写：required 时模型必须调用工具，语法会收紧。 易错：指定了不存在的函数名。 |
| `parallel_tool_calls` | `ChatCompletionRequest` | `True` | 作用：在 `ChatCompletionRequest` 里，`parallel_tool_calls` 是否允许同一轮并行多个工具调用。 默认与效果：默认 true。 交互与改写：和 max_tool_calls 一起限制工具。 易错：模型模板不支持并行，却开着它。 |
| `return_hidden_states` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_hidden_states` 是要隐状态。可以是布尔或 last。服务端还要开对应模式。 默认与效果：默认 false。 交互与改写：full 很占带宽。辅助隐状态打开时返回的是拼接 aux。 易错：服务模式是 last，请求要了全部。 |
| `return_routed_experts` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_routed_experts` 是返回每层选中的专家。 默认与效果：默认 false。 交互与改写：服务端开关也要开。起点由 routed_experts_start_len 决定。 易错：只在请求里开。 |
| `routed_experts_start_len` | `ChatCompletionRequest` | `0` | 作用：在 `ChatCompletionRequest` 里，`routed_experts_start_len` 是从哪个绝对位置开始返回专家路由，区间是到序列末尾。0 表示整段。必须落在 0 到提示长度。 默认与效果：默认 0。 交互与改写：只在 return_routed_experts 为真时有用。 易错：设得超过提示长度。 |
| `return_cached_tokens_details` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_cached_tokens_details` 是在 usage 里展开缓存命中了多少 token。 默认与效果：默认 false。 交互与改写：服务端还要能报告缓存。 易错：没开 radix，细节全是 0。 |
| `return_spec_tokens_details` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_spec_tokens_details` 是返回投机草稿的接受细节。 默认与效果：默认 false。 交互与改写：没有投机时没有细节。 易错：没开 DSPARK 却依赖这个字段。 |
| `return_prompt_token_ids` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_prompt_token_ids` 是返回提示的 token id，不必算 logprob。 默认与效果：默认 false。 交互与改写：和 return_logprob 独立。 易错：以为开了它就有概率。 |
| `return_token_ids` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_token_ids` 是响应里带输出 token id。 默认与效果：默认 false。 交互与改写：和 sglext 的输入 id 是两套。 易错：流式每帧都带完整列表而不是增量。 |
| `return_meta_info` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_meta_info` 是chat 响应里带上 SGLang 的 meta_info。 默认与效果：默认 false。 交互与改写：扁平 logprob 依赖它。 易错：没开它却要扁平 logprob。 |
| `return_input_ids_in_sglext` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_input_ids_in_sglext` 是在 SGLang 扩展字段里带回输入 id。 默认与效果：默认 false。 交互与改写：和 return_token_ids 不同，一个是输入一个是输出。 易错：两个都开，响应很大。 |
| `return_output_ids_in_sglext` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_output_ids_in_sglext` 是在扩展字段里带回输出 token id。 默认与效果：默认 false。 交互与改写：流式时每帧都可能带。 易错：带宽被 id 占满。 |
| `return_sampling_mask` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_sampling_mask` 是返回每个输出 token 的采样支持集。 默认与效果：默认 false。 交互与改写：支持集大小受服务端 sampling_mask_max_tokens 限制。 易错：top_k 大于上限，请求被中止。 |
| `sampling_logprobs_mode` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`sampling_logprobs_mode` 是采样 logprob 返回选中 token，还是整个支持集。不填且开了 mask 时用选中模式。 默认与效果：默认空。 交互与改写：取值 selected 或 support。 易错：写成了别的字符串。 |
| `return_flat_raw_output_top_logprobs` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`return_flat_raw_output_top_logprobs` 是把输出的 top logprob 打成扁平数组，放在 meta_info，choices 里的 top_logprobs 则为空。 默认与效果：默认 false。 交互与改写：chat 上还常配合 b64。 易错：两边都解析，得到空列表。 |
| `return_flat_raw_top_logprobs_b64` | `ChatCompletionRequest` | `True` | 作用：在 `ChatCompletionRequest` 里，`return_flat_raw_top_logprobs_b64` 是扁平数组再 base64。/generate 要先开上面的扁平标志；chat 默认就是 true。 默认与效果：chat 默认 true，generate 默认 false。 交互与改写：false 时返回普通 JSON 列表。 易错：没开扁平却要求 b64。 |
| `reasoning_effort` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`reasoning_effort` 是限制思考力度。可以是 none 到 max 的档位，或 0 到 0.99 的数。none 会把思考关掉。max 是 SGLang 扩展，不支持的模型当 high。 默认与效果：默认空，不限制。 交互与改写：harmony 路径不支持。 易错：档位字符串拼错。 |
| `thinking` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`thinking` 是思考配置，例如 {type: adaptive}。只有模板定义了思考模式的模型才读。 默认与效果：默认空。 交互与改写：Anthropic 用自己的 thinking 对象。 易错：模型没有思考模板，配置被忽略。 |
| `task` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`task` 是DeepSeek-V4 的快捷任务，例如 domain。设置后最后一条用户消息被当成一次性分类，并在生成前加上对应特殊 token。只有 dsv4/dsv41 编码器认它。 默认与效果：默认空。 交互与改写：其他模型忽略。 易错：在非 V4 上设置，没有任何特殊 token。 |
| `top_k` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`top_k` 是只在概率最高的 k 个里采样。 默认与效果：默认不限制。 交互与改写：采样掩码上限要大于它。 易错：设成 0。 |
| `min_p` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`min_p` 是丢掉概率低于该值的 token。 默认与效果：默认 0，不丢。 交互与改写：在 top-k/top-p 之后再滤。 易错：接近 1 时几乎只剩一个 token。 |
| `min_tokens` | `ChatCompletionRequest` | `0` | 作用：在 `ChatCompletionRequest` 里，`min_tokens` 是HTTP 层的最小生成长度，对应 min_new_tokens。 默认与效果：默认 0。 交互与改写：会抄进采样参数。 易错：和 max 写反。 |
| `regex` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`regex` 是用正则约束输出。 默认与效果：默认空。 交互与改写：和 json_schema、ebnf 不要同时给。 易错：正则灾难性回溯。 |
| `ebnf` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`ebnf` 是用 EBNF 约束输出格式。 默认与效果：默认空。 交互与改写：和 json_schema、regex 同时给时通常只该有一个。 易错：语法有歧义，编译失败。 |
| `repetition_penalty` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`repetition_penalty` 是按重复缩放 logits。1 表示不惩罚。 默认与效果：默认 1。 交互与改写：和 frequency 的公式不同。 易错：设成 0。 |
| `stop_token_ids` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`stop_token_ids` 是遇到这些 id 就停。 默认与效果：默认空。 交互与改写：并进采样参数的集合。 易错：id 用错词表。 |
| `stop_regex` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`stop_regex` 是停止正则。 默认与效果：默认空。 交互与改写：拷进 stop_regex_strs。 易错：正则太宽，第一个 token 就停。 |
| `no_stop_trim` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`no_stop_trim` 是停止串命中后不要把它从返回文本里裁掉。 默认与效果：默认 false，会裁掉。 交互与改写：只影响返回文本，不影响停的位置。 易错：客户端自己再裁一次，多裁了正文。 |
| `ignore_eos` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`ignore_eos` 是为真时 EOS 不结束生成，直到达到长度上限。 默认与效果：默认 false。 交互与改写：和 stop 串仍会生效，除非一并清掉。 易错：打开后回答停不下来，打满 max_tokens。 |
| `continue_final_message` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`continue_final_message` 是为真时接着最后一条助手消息往下写，而不是新开一轮。 默认与效果：默认 false。 交互与改写：模板不会再追加助手起始符。 易错：最后一条不是助手，续写位置错了。 |
| `skip_special_tokens` | `ChatCompletionRequest` | `True` | 作用：在 `ChatCompletionRequest` 里，`skip_special_tokens` 是解文本时跳过特殊符。 默认与效果：默认 true。 交互与改写：和 return_text_in_logprobs 一起决定 logprob 里的文本。 易错：关掉后特殊符泄露到用户可见文本。 |
| `lora_path` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`lora_path` 是这条请求使用的适配器路径或名字。可以按批给列表。 默认与效果：默认空，用基座。 交互与改写：服务端要已经 enable_lora 并加载过。 易错：路径在服务进程上看不见。 |
| `session_id` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`session_id` 是同一会话的稳定 id。它不改写提示，和 session_params 不同。 默认与效果：默认空。 交互与改写：会话 radix 用它做引用。 易错：每条请求换一个 id，会话缓存用不上。 |
| `session_params` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`session_params` 是连续提示的会话参数，可能改写或重建提示。 默认与效果：默认空。 交互与改写：和 session_id 不是一回事。 易错：参数让服务重写了用户没预期的历史。 |
| `separate_reasoning` | `ChatCompletionRequest` | `True` | 作用：在 `ChatCompletionRequest` 里，`separate_reasoning` 是把思考段从正文里拆开返回。 默认与效果：默认 true。 交互与改写：依赖 reasoning parser。 易错：解析器和模板不配套，拆错边界。 |
| `stream_reasoning` | `ChatCompletionRequest` | `True` | 作用：在 `ChatCompletionRequest` 里，`stream_reasoning` 是思考段是否也流式推送。 默认与效果：默认 true。 交互与改写：separate_reasoning 为假时思考和正文混在一起流。 易错：客户端只解析正文通道，丢掉思考流。 |
| `chat_template_kwargs` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`chat_template_kwargs` 是传给对话模板的额外参数，例如思考开关。 默认与效果：默认空。 交互与改写：服务端 default_chat_template_kwargs 是底，请求覆盖它。 易错：键名和模板里的变量不一致，被静默忽略。 |
| `max_dynamic_patch` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`max_dynamic_patch` 是这一请求最多把图像切成多少动态块。 默认与效果：默认空，用模型默认。 交互与改写：图像和视频还可以分别再限。 易错：块数过大，视觉 token 超过 max_image_tokens。 |
| `min_dynamic_patch` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`min_dynamic_patch` 是动态切图至少多少块。 默认与效果：默认空。 交互与改写：不能大于 max。 易错：最小大于最大。 |
| `use_audio_in_video` | `ChatCompletionRequest` | `False` | 作用：在 `ChatCompletionRequest` 里，`use_audio_in_video` 是从视频里抽音频一起送进模型。 默认与效果：默认 false。 交互与改写：模型要同时有视觉和音频塔。 易错：纯视觉模型上打开。 |
| `images_config` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`images_config` 是这一请求的图像预处理配置，覆盖服务端 mm_process_config。 默认与效果：默认空。 交互与改写：键要和处理器认识的一致。 易错：写成了视频配置的键。 |
| `video_config` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`video_config` 是这一请求的视频预处理配置。 默认与效果：默认空。 交互与改写：和 images_config 分开。 易错：键和处理器不一致。 |
| `mm_processor_kwargs` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`mm_processor_kwargs` 是这一请求传给多模态处理器的参数。 默认与效果：默认空。 交互与改写：覆盖服务端 mm_process_config。 易错：键名处理器不认识。 |
| `custom_logit_processor` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`custom_logit_processor` 是序列化后的自定义 logit 处理器。服务端必须开 enable_custom_logit_processor。 默认与效果：默认空。 交互与改写：等于允许请求在进程里执行代码，默认关是为了安全。 易错：服务没开开关，处理器被忽略。 |
| `custom_params` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`custom_params` 是采样器的扩展参数字典。 默认与效果：默认空。 交互与改写：只有对应后端认识的键才生效。 易错：把标准温度写进这里而不是 temperature。 |
| `watermark` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`watermark` 是这一请求是否加水印，或水印配置。服务端要先 enable_watermark。 默认与效果：默认空。 交互与改写：enforce_all 时请求关不掉。 易错：服务没开水印，字段被忽略。 |
| `input_ids` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`input_ids` 是已经分好的 token id。单条是一维，批是二维。 默认与效果：默认空。 交互与改写：给了它就可以跳过文本分词。 易错：id 越界或用错词表。 |
| `rid` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`rid` 是请求 id。不传则服务生成。批里必须唯一。 默认与效果：默认空。 交互与改写：重复 id 会在规范化时抛错。 易错：一批里用了同一个 id。 |
| `extra_key` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`extra_key` 是额外的缓存命名空间，和 cache_salt 类似但给路由分类用。 默认与效果：默认空。 交互与改写：不同 extra_key 不共享前缀页。 易错：每个请求都随机，缓存命中率为 0。 |
| `cache_salt` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`cache_salt` 是给前缀缓存加盐，相同 token 也不会命中另一盐的页。 默认与效果：默认空，不加盐。 交互与改写：和 extra_key、LoRA 名一起组成缓存键。 易错：把用户隐私写进盐里还打进日志。 |
| `priority` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`priority` 是请求优先级。服务端没开优先级调度时可能被忽略或拒绝。 默认与效果：默认空。 交互与改写：数值方向由 schedule_low_priority_values_first 决定。 易错：以为越大越优先，服务却是小的先跑。 |
| `data_parallel_rank` | `ChatCompletionRequest` | `None` | 作用：在 `ChatCompletionRequest` 里，`data_parallel_rank` 是routed_dp_rank 的旧名。网关仍会塞这个字段。 默认与效果：默认空。 交互与改写：两边都写时以新字段为准或同时被接受。 易错：两个字段写成不同的组号。 |

## OpenAI Completions `/v1/completions`

文件 `python/sglang/srt/entrypoints/openai/protocol.py`。

### `CompletionRequest`

定义于第 365 行，公开字段 56 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `bootstrap_host` | `PDRoutingFields` | `None` | 作用：在 `CompletionRequest` 里，`bootstrap_host` 是PD 分离时 prefill 的主机名，decode 靠它来要 KV。 默认与效果：默认空。 交互与改写：合部服务不需要。要和 bootstrap_port、room 一起用。 易错：写成了客户端自己的地址。 |
| `bootstrap_port` | `PDRoutingFields` | `None` | 作用：在 `CompletionRequest` 里，`bootstrap_port` 是prefill 上 bootstrap 服务的端口，默认服务旗标是 8998。 默认与效果：默认空，表示用服务端配置的端口。 交互与改写：请求级可以覆盖。 易错：写成 HTTP 30000。 |
| `bootstrap_room` | `PDRoutingFields` | `None` | 作用：在 `CompletionRequest` 里，`bootstrap_room` 是同一条对话在 PD 两侧的房间号，负载策略 follow_bootstrap_room 会把它送到同一组。 默认与效果：默认空。 交互与改写：和 routed_dp_rank 一起决定落点。 易错：每条请求随机 room，前缀永远对不齐。 |
| `routed_dp_rank` | `PDRoutingFields` | `None` | 作用：在 `CompletionRequest` 里，`routed_dp_rank` 是外部路由器指定的注意力 DP 组。 默认与效果：默认空，由负载策略选。 交互与改写：旧名 data_parallel_rank 仍接受。 易错：指定的组号大于等于组数。 |
| `disagg_prefill_dp_rank` | `PDRoutingFields` | `None` | 作用：在 `CompletionRequest` 里，`disagg_prefill_dp_rank` 是提示 decode：KV 在哪一个 prefill 的 DP 组上。 默认与效果：默认空。 交互与改写：外部路由器填，应用客户端不要自己猜。 易错：组和真实存放 KV 的 rank 不一致。 |
| `model` | `CompletionRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `CompletionRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `prompt` | `CompletionRequest` | `（必填）` | 作用：在 `CompletionRequest` 里，`prompt` 是补全或 Ollama 的原文，不经对话模板。 默认与效果：补全必填。 交互与改写：chat 不读它。 易错：把 messages 历史塞进一个字符串。 |
| `best_of` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`best_of` 是补全接口先采这么多条，再留 n 条更好的。 默认与效果：默认空。 交互与改写：需要 logprob 才有好坏可排。 易错：best_of 小于 n。 |
| `echo` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`echo` 是补全接口是否把提示原文回显在结果前面。 默认与效果：默认 false。 交互与改写：回显计入返回文本，不一定计入生成 token。 易错：客户端把 echo 的提示再拼一次。 |
| `frequency_penalty` | `CompletionRequest` | `0.0` | 作用：在 `CompletionRequest` 里，`frequency_penalty` 是按出现次数惩罚 token，减轻重复。 默认与效果：默认 0。 交互与改写：可与 presence、repetition 叠加。 易错：设得过大，句子被逼换词。 |
| `logit_bias` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`logit_bias` 是按 token 字符串 id 加减分。 默认与效果：默认空。 交互与改写：在采样前加到 logits。 易错：键不是十进制 id。 |
| `logprobs` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`logprobs` 是chat 是布尔；补全是要返回的个数。 默认与效果：chat 默认 false。 交互与改写：还要 top_logprobs 才有候选。 易错：类型用错。 |
| `max_tokens` | `CompletionRequest` | `16` | 作用：在 `CompletionRequest` 里，`max_tokens` 是旧的长度上限。chat 应改用 max_completion_tokens。补全接口默认常常是 16。Anthropic 用它且必须为正。 默认与效果：chat 默认空。 交互与改写：会抄进 max_new_tokens。 易错：补全没写它，只生成 16 个 token。 |
| `n` | `CompletionRequest` | `1` | 作用：在 `CompletionRequest` 里，`n` 是并行样本数。beam 时变成返回条数且不能超过束宽。 默认与效果：默认 1。 交互与改写：每条样本一份 KV。 易错：n 大于 beam_width。 |
| `presence_penalty` | `CompletionRequest` | `0.0` | 作用：在 `CompletionRequest` 里，`presence_penalty` 是token 只要出现过就惩罚，不看次数。 默认与效果：默认 0。 交互与改写：可与 frequency 叠加。 易错：两个都很大，分布被压扁。 |
| `seed` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`seed` 是请求级随机种子，对应 sampling_seed。 默认与效果：默认空。 交互与改写：确定性还要服务端开关。 易错：种子用了字符串。 |
| `stop` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`stop` 是停止字符串。采样侧会拷到 stop_strs 再清空。 默认与效果：默认空。 交互与改写：和 stop_token_ids 一起生效。 易错：停止串出现在思考里。 |
| `stream` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`stream` 是否流式。结束帧是 data: [DONE]。Ollama chat 默认 true。 默认与效果：多数默认 false。 交互与改写：间隔还受 stream_interval 影响。 易错：按非流式去读一个不关闭的流。 |
| `stream_options` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`stream_options` 是流式附加选项，例如最后是否带 usage。 默认与效果：默认空。 交互与改写：只在 stream 为真时生效。 易错：非流式也传了它。 |
| `suffix` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`suffix` 是补全接口在生成后再拼上的后缀，或 Ollama 的后缀。 默认与效果：默认空。 交互与改写：后缀不算模型生成。 易错：后缀含停止符，客户端自己截断。 |
| `temperature` | `CompletionRequest` | `1.0` | 作用：在 `CompletionRequest` 里，`temperature` 是采样温度。越高越随机，接近 0 趋于贪心。 默认与效果：SamplingParams 默认 1.0；HTTP 空值会收成 1.0。 交互与改写：和 top_p、top_k 一起用。 易错：不传被当成贪心。 |
| `top_p` | `CompletionRequest` | `1.0` | 作用：在 `CompletionRequest` 里，`top_p` 是核采样。1 表示不截断。 默认与效果：默认 1。 交互与改写：通常在温度之后应用。 易错：设成 0。 |
| `user` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`user` 是终端用户标识，只用于审计。 默认与效果：默认空。 交互与改写：不会自动变成缓存盐。 易错：把密钥写进 user。 |
| `return_hidden_states` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`return_hidden_states` 是要隐状态。可以是布尔或 last。服务端还要开对应模式。 默认与效果：默认 false。 交互与改写：full 很占带宽。辅助隐状态打开时返回的是拼接 aux。 易错：服务模式是 last，请求要了全部。 |
| `return_routed_experts` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`return_routed_experts` 是返回每层选中的专家。 默认与效果：默认 false。 交互与改写：服务端开关也要开。起点由 routed_experts_start_len 决定。 易错：只在请求里开。 |
| `routed_experts_start_len` | `CompletionRequest` | `0` | 作用：在 `CompletionRequest` 里，`routed_experts_start_len` 是从哪个绝对位置开始返回专家路由，区间是到序列末尾。0 表示整段。必须落在 0 到提示长度。 默认与效果：默认 0。 交互与改写：只在 return_routed_experts 为真时有用。 易错：设得超过提示长度。 |
| `return_cached_tokens_details` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`return_cached_tokens_details` 是在 usage 里展开缓存命中了多少 token。 默认与效果：默认 false。 交互与改写：服务端还要能报告缓存。 易错：没开 radix，细节全是 0。 |
| `return_spec_tokens_details` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`return_spec_tokens_details` 是返回投机草稿的接受细节。 默认与效果：默认 false。 交互与改写：没有投机时没有细节。 易错：没开 DSPARK 却依赖这个字段。 |
| `return_token_ids` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`return_token_ids` 是响应里带输出 token id。 默认与效果：默认 false。 交互与改写：和 sglext 的输入 id 是两套。 易错：流式每帧都带完整列表而不是增量。 |
| `top_k` | `CompletionRequest` | `-1` | 作用：在 `CompletionRequest` 里，`top_k` 是只在概率最高的 k 个里采样。 默认与效果：默认不限制。 交互与改写：采样掩码上限要大于它。 易错：设成 0。 |
| `min_p` | `CompletionRequest` | `0.0` | 作用：在 `CompletionRequest` 里，`min_p` 是丢掉概率低于该值的 token。 默认与效果：默认 0，不丢。 交互与改写：在 top-k/top-p 之后再滤。 易错：接近 1 时几乎只剩一个 token。 |
| `min_tokens` | `CompletionRequest` | `0` | 作用：在 `CompletionRequest` 里，`min_tokens` 是HTTP 层的最小生成长度，对应 min_new_tokens。 默认与效果：默认 0。 交互与改写：会抄进采样参数。 易错：和 max 写反。 |
| `json_schema` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`json_schema` 是用 JSON Schema 约束输出。 默认与效果：默认空。 交互与改写：会编译成语法，和 response_format 的 json_schema 是同一意图。 易错：schema 含远程 $ref，编译失败。 |
| `regex` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`regex` 是用正则约束输出。 默认与效果：默认空。 交互与改写：和 json_schema、ebnf 不要同时给。 易错：正则灾难性回溯。 |
| `ebnf` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`ebnf` 是用 EBNF 约束输出格式。 默认与效果：默认空。 交互与改写：和 json_schema、regex 同时给时通常只该有一个。 易错：语法有歧义，编译失败。 |
| `repetition_penalty` | `CompletionRequest` | `1.0` | 作用：在 `CompletionRequest` 里，`repetition_penalty` 是按重复缩放 logits。1 表示不惩罚。 默认与效果：默认 1。 交互与改写：和 frequency 的公式不同。 易错：设成 0。 |
| `stop_token_ids` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`stop_token_ids` 是遇到这些 id 就停。 默认与效果：默认空。 交互与改写：并进采样参数的集合。 易错：id 用错词表。 |
| `stop_regex` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`stop_regex` 是停止正则。 默认与效果：默认空。 交互与改写：拷进 stop_regex_strs。 易错：正则太宽，第一个 token 就停。 |
| `no_stop_trim` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`no_stop_trim` 是停止串命中后不要把它从返回文本里裁掉。 默认与效果：默认 false，会裁掉。 交互与改写：只影响返回文本，不影响停的位置。 易错：客户端自己再裁一次，多裁了正文。 |
| `ignore_eos` | `CompletionRequest` | `False` | 作用：在 `CompletionRequest` 里，`ignore_eos` 是为真时 EOS 不结束生成，直到达到长度上限。 默认与效果：默认 false。 交互与改写：和 stop 串仍会生效，除非一并清掉。 易错：打开后回答停不下来，打满 max_tokens。 |
| `skip_special_tokens` | `CompletionRequest` | `True` | 作用：在 `CompletionRequest` 里，`skip_special_tokens` 是解文本时跳过特殊符。 默认与效果：默认 true。 交互与改写：和 return_text_in_logprobs 一起决定 logprob 里的文本。 易错：关掉后特殊符泄露到用户可见文本。 |
| `lora_path` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`lora_path` 是这条请求使用的适配器路径或名字。可以按批给列表。 默认与效果：默认空，用基座。 交互与改写：服务端要已经 enable_lora 并加载过。 易错：路径在服务进程上看不见。 |
| `session_id` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`session_id` 是同一会话的稳定 id。它不改写提示，和 session_params 不同。 默认与效果：默认空。 交互与改写：会话 radix 用它做引用。 易错：每条请求换一个 id，会话缓存用不上。 |
| `session_params` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`session_params` 是连续提示的会话参数，可能改写或重建提示。 默认与效果：默认空。 交互与改写：和 session_id 不是一回事。 易错：参数让服务重写了用户没预期的历史。 |
| `response_format` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`response_format` 是输出格式，例如 json_object 或 json_schema。转写接口里则是 json、text、verbose。 默认与效果：默认空或 json。 交互与改写：会变成语法约束。 易错：schema 和格式类型不匹配。 |
| `custom_params` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`custom_params` 是采样器的扩展参数字典。 默认与效果：默认空。 交互与改写：只有对应后端认识的键才生效。 易错：把标准温度写进这里而不是 temperature。 |
| `custom_logit_processor` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`custom_logit_processor` 是序列化后的自定义 logit 处理器。服务端必须开 enable_custom_logit_processor。 默认与效果：默认空。 交互与改写：等于允许请求在进程里执行代码，默认关是为了安全。 易错：服务没开开关，处理器被忽略。 |
| `watermark` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`watermark` 是这一请求是否加水印，或水印配置。服务端要先 enable_watermark。 默认与效果：默认空。 交互与改写：enforce_all 时请求关不掉。 易错：服务没开水印，字段被忽略。 |
| `images_config` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`images_config` 是这一请求的图像预处理配置，覆盖服务端 mm_process_config。 默认与效果：默认空。 交互与改写：键要和处理器认识的一致。 易错：写成了视频配置的键。 |
| `mm_processor_kwargs` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`mm_processor_kwargs` 是这一请求传给多模态处理器的参数。 默认与效果：默认空。 交互与改写：覆盖服务端 mm_process_config。 易错：键名处理器不认识。 |
| `data_parallel_rank` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`data_parallel_rank` 是routed_dp_rank 的旧名。网关仍会塞这个字段。 默认与效果：默认空。 交互与改写：两边都写时以新字段为准或同时被接受。 易错：两个字段写成不同的组号。 |
| `rid` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`rid` 是请求 id。不传则服务生成。批里必须唯一。 默认与效果：默认空。 交互与改写：重复 id 会在规范化时抛错。 易错：一批里用了同一个 id。 |
| `extra_key` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`extra_key` 是额外的缓存命名空间，和 cache_salt 类似但给路由分类用。 默认与效果：默认空。 交互与改写：不同 extra_key 不共享前缀页。 易错：每个请求都随机，缓存命中率为 0。 |
| `cache_salt` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`cache_salt` 是给前缀缓存加盐，相同 token 也不会命中另一盐的页。 默认与效果：默认空，不加盐。 交互与改写：和 extra_key、LoRA 名一起组成缓存键。 易错：把用户隐私写进盐里还打进日志。 |
| `priority` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`priority` 是请求优先级。服务端没开优先级调度时可能被忽略或拒绝。 默认与效果：默认空。 交互与改写：数值方向由 schedule_low_priority_values_first 决定。 易错：以为越大越优先，服务却是小的先跑。 |
| `custom_labels` | `CompletionRequest` | `None` | 作用：在 `CompletionRequest` 里，`custom_labels` 是给这条请求的指标打的标签。 默认与效果：默认空。 交互与改写：标签名还必须在服务端允许列表里。 易错：高基数字段当标签，指标爆炸。 |

## OpenAI Embeddings / Classify / Score / Rerank / Responses

文件 `python/sglang/srt/entrypoints/openai/protocol.py`。

### `EmbeddingRequest`

定义于第 1416 行，公开字段 10 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `input` | `EmbeddingRequest` | `（必填）` | 作用：在 `EmbeddingRequest` 里，`input` 是嵌入、分类或 Responses 的输入，可以是字符串或 token。 默认与效果：默认必填。 交互与改写：和 messages 不要同时当成同一条对话。 易错：嵌套层数和接口要求不一致。 |
| `model` | `EmbeddingRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `EmbeddingRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `encoding_format` | `EmbeddingRequest` | `'float'` | 作用：在 `EmbeddingRequest` 里，`encoding_format` 是嵌入返回 float 还是 base64。 默认与效果：默认 float。 交互与改写：base64 更小但客户端要自己解码。 易错：写成不认识的格式名。 |
| `dimensions` | `EmbeddingRequest` | `None` | 作用：在 `EmbeddingRequest` 里，`dimensions` 是嵌入向量要截到多少维。 默认与效果：默认空，返回全维。 交互与改写：模型隐藏维必须不小于它。 易错：设得比隐藏维大。 |
| `user` | `EmbeddingRequest` | `None` | 作用：在 `EmbeddingRequest` 里，`user` 是终端用户标识，只用于审计。 默认与效果：默认空。 交互与改写：不会自动变成缓存盐。 易错：把密钥写进 user。 |
| `rid` | `EmbeddingRequest` | `None` | 作用：在 `EmbeddingRequest` 里，`rid` 是请求 id。不传则服务生成。批里必须唯一。 默认与效果：默认空。 交互与改写：重复 id 会在规范化时抛错。 易错：一批里用了同一个 id。 |
| `priority` | `EmbeddingRequest` | `None` | 作用：在 `EmbeddingRequest` 里，`priority` 是请求优先级。服务端没开优先级调度时可能被忽略或拒绝。 默认与效果：默认空。 交互与改写：数值方向由 schedule_low_priority_values_first 决定。 易错：以为越大越优先，服务却是小的先跑。 |
| `lora_path` | `EmbeddingRequest` | `None` | 作用：在 `EmbeddingRequest` 里，`lora_path` 是这条请求使用的适配器路径或名字。可以按批给列表。 默认与效果：默认空，用基座。 交互与改写：服务端要已经 enable_lora 并加载过。 易错：路径在服务进程上看不见。 |
| `embed_override_token_id` | `EmbeddingRequest` | `None` | 作用：在 `EmbeddingRequest` 里，`embed_override_token_id` 是嵌入时在这个 token 位置读取向量，而不是默认位置。 默认与效果：默认空。 交互与改写：和 embed_overrides 一起改读分点。 易错：id 不在序列里。 |
| `embed_overrides` | `EmbeddingRequest` | `None` | 作用：在 `EmbeddingRequest` 里，`embed_overrides` 是按位置覆盖嵌入向量。 默认与效果：默认空。 交互与改写：类型里可能含张量，HTTP 通常传列表。 易错：长度和序列不一致。 |

### `ClassifyRequest`

定义于第 1447 行，公开字段 5 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `model` | `ClassifyRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `ClassifyRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `input` | `ClassifyRequest` | `（必填）` | 作用：在 `ClassifyRequest` 里，`input` 是嵌入、分类或 Responses 的输入，可以是字符串或 token。 默认与效果：默认必填。 交互与改写：和 messages 不要同时当成同一条对话。 易错：嵌套层数和接口要求不一致。 |
| `user` | `ClassifyRequest` | `None` | 作用：在 `ClassifyRequest` 里，`user` 是终端用户标识，只用于审计。 默认与效果：默认空。 交互与改写：不会自动变成缓存盐。 易错：把密钥写进 user。 |
| `rid` | `ClassifyRequest` | `None` | 作用：在 `ClassifyRequest` 里，`rid` 是请求 id。不传则服务生成。批里必须唯一。 默认与效果：默认空。 交互与改写：重复 id 会在规范化时抛错。 易错：一批里用了同一个 id。 |
| `priority` | `ClassifyRequest` | `None` | 作用：在 `ClassifyRequest` 里，`priority` 是请求优先级。服务端没开优先级调度时可能被忽略或拒绝。 默认与效果：默认空。 交互与改写：数值方向由 schedule_low_priority_values_first 决定。 易错：以为越大越优先，服务却是小的先跑。 |

### `ScoringRequest`

定义于第 1482 行，公开字段 13 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `query` | `ScoringRequest` | `None` | 作用：在 `ScoringRequest` 里，`query` 是打分或重排的查询文本。 默认与效果：默认必填。 交互与改写：和 items 或 documents 成对。 易错：查询和文档用了不同语言但模型是单语。 |
| `items` | `ScoringRequest` | `None` | 作用：在 `ScoringRequest` 里，`items` 是打分接口的多个条目。 默认与效果：默认必填。 交互与改写：和 query 拼成 MIS 或逐条序列。 易错：条目数量超过服务端限制。 |
| `embed_override_token_id` | `ScoringRequest` | `None` | 作用：在 `ScoringRequest` 里，`embed_override_token_id` 是嵌入时在这个 token 位置读取向量，而不是默认位置。 默认与效果：默认空。 交互与改写：和 embed_overrides 一起改读分点。 易错：id 不在序列里。 |
| `query_embed_overrides` | `ScoringRequest` | `None` | 作用：在 `ScoringRequest` 里，`query_embed_overrides` 是打分时查询侧的嵌入覆盖。 默认与效果：默认空。 交互与改写：和 item 侧覆盖分开。 易错：只覆盖了一边。 |
| `item_embed_overrides` | `ScoringRequest` | `None` | 作用：在 `ScoringRequest` 里，`item_embed_overrides` 是打分接口里每个条目的嵌入覆盖。 默认与效果：默认空。 交互与改写：长度要和 items 一致。 易错：只给了 query 的覆盖，条目仍是原文。 |
| `label_token_ids` | `ScoringRequest` | `None` | 作用：在 `ScoringRequest` 里，`label_token_ids` 是打分时要读取的标签 token。 默认与效果：默认空。 交互与改写：和 token_indices_to_pool 一起决定读哪里的 logit。 易错：标签 id 不在词表里。 |
| `apply_softmax` | `ScoringRequest` | `False` | 作用：在 `ScoringRequest` 里，`apply_softmax` 是打分接口是否对分类 logits 做 softmax 再返回。 默认与效果：默认按接口。 交互与改写：不做则返回的是原始 logit，数值不能当概率比。 易错：客户端把 logit 当成 0 到 1 的分数。 |
| `temperature` | `ScoringRequest` | `1.0` | 作用：在 `ScoringRequest` 里，`temperature` 是采样温度。越高越随机，接近 0 趋于贪心。 默认与效果：SamplingParams 默认 1.0；HTTP 空值会收成 1.0。 交互与改写：和 top_p、top_k 一起用。 易错：不传被当成贪心。 |
| `return_token_logprobs` | `ScoringRequest` | `False` | 作用：在 `ScoringRequest` 里，`return_token_logprobs` 是打分接口返回标签 token 的 logprob，而不只是分数。 默认与效果：默认 false。 交互与改写：要和 label_token_ids 一起用。 易错：没指定标签 id。 |
| `item_first` | `ScoringRequest` | `False` | 作用：在 `ScoringRequest` 里，`item_first` 是打分时把条目放在查询前面。 默认与效果：默认按接口。 交互与改写：和提示模板的顺序必须一致。 易错：设反了，分数读在错误的 token 上。 |
| `return_pooled_hidden_states` | `ScoringRequest` | `False` | 作用：在 `ScoringRequest` 里，`return_pooled_hidden_states` 是嵌入接口返回池化后的向量，而不是每个 token 一个。 默认与效果：默认按接口。 交互与改写：dimensions 可以再截断。 易错：池化方式和模型不一致。 |
| `score_extraction_token` | `ScoringRequest` | `None` | 作用：在 `ScoringRequest` 里，`score_extraction_token` 是从哪个 token 的 logit 抽出打分。 默认与效果：默认按接口，常常是最后一个。 交互与改写：和 label_token_ids 二选一更清晰。 易错：该 token 没出现在序列里。 |
| `model` | `ScoringRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `ScoringRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |

### `DecisionRequest`

定义于第 1639 行，公开字段 8 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `input` | `DecisionRequest` | `（必填）` | 作用：在 `DecisionRequest` 里，`input` 是嵌入、分类或 Responses 的输入，可以是字符串或 token。 默认与效果：默认必填。 交互与改写：和 messages 不要同时当成同一条对话。 易错：嵌套层数和接口要求不一致。 |
| `images` | `DecisionRequest` | `factory` | 作用：在 `DecisionRequest` 里，`images` 是决策或 Ollama 请求里的图像列表。 默认与效果：默认空。 交互与改写：会收成 image_data。 易错：传了本地路径，服务进程看不见那个文件。 |
| `questions` | `DecisionRequest` | `（必填）` | 作用：在 `DecisionRequest` 里，`questions` 是决策接口的题目列表。 默认与效果：默认必填。 交互与改写：每题有自己的选项或是否题结构。 易错：题目 token 超过上下文。 |
| `temperature` | `DecisionRequest` | `1.0` | 作用：在 `DecisionRequest` 里，`temperature` 是采样温度。越高越随机，接近 0 趋于贪心。 默认与效果：SamplingParams 默认 1.0；HTTP 空值会收成 1.0。 交互与改写：和 top_p、top_k 一起用。 易错：不传被当成贪心。 |
| `chat_template_kwargs` | `DecisionRequest` | `factory` | 作用：在 `DecisionRequest` 里，`chat_template_kwargs` 是传给对话模板的额外参数，例如思考开关。 默认与效果：默认空。 交互与改写：服务端 default_chat_template_kwargs 是底，请求覆盖它。 易错：键名和模板里的变量不一致，被静默忽略。 |
| `prompt_format_version` | `DecisionRequest` | `None` | 作用：在 `DecisionRequest` 里，`prompt_format_version` 是决策接口提示格式的版本号。 默认与效果：默认按接口。 交互与改写：版本和题目布局必须配套。 易错：用了旧版本号，读分位置错了。 |
| `return_prompt_token_ids` | `DecisionRequest` | `False` | 作用：在 `DecisionRequest` 里，`return_prompt_token_ids` 是返回提示的 token id，不必算 logprob。 默认与效果：默认 false。 交互与改写：和 return_logprob 独立。 易错：以为开了它就有概率。 |
| `model` | `DecisionRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `DecisionRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |

### `V1RerankReqInput`

定义于第 1687 行，公开字段 5 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `query` | `V1RerankReqInput` | `...` | 作用：在 `V1RerankReqInput` 里，`query` 是打分或重排的查询文本。 默认与效果：默认必填。 交互与改写：和 items 或 documents 成对。 易错：查询和文档用了不同语言但模型是单语。 |
| `documents` | `V1RerankReqInput` | `...` | 作用：在 `V1RerankReqInput` 里，`documents` 是重排接口的候选文档列表。 默认与效果：默认必填。 交互与改写：和 query 成对。 易错：文档是 token 而接口要字符串。 |
| `instruct` | `V1RerankReqInput` | `None` | 作用：在 `V1RerankReqInput` 里，`instruct` 是重排时附加的指令，告诉模型按什么标准排序。 默认与效果：默认空。 交互与改写：和 query、documents 一起进提示。 易错：指令本身很长，把文档挤出上下文。 |
| `top_n` | `V1RerankReqInput` | `None` | 作用：在 `V1RerankReqInput` 里，`top_n` 是重排后返回前几条。 默认与效果：默认按接口。 交互与改写：不能大于文档数。 易错：大于 documents 长度。 |
| `return_documents` | `V1RerankReqInput` | `True` | 作用：在 `V1RerankReqInput` 里，`return_documents` 是重排结果是否把原文一并返回。 默认与效果：默认按接口。 交互与改写：文档很大时响应会膨胀。 易错：以为不返回文档就没计费。 |

### `ResponsesRequest`

定义于第 1924 行，公开字段 40 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `bootstrap_host` | `PDRoutingFields` | `None` | 作用：在 `ResponsesRequest` 里，`bootstrap_host` 是PD 分离时 prefill 的主机名，decode 靠它来要 KV。 默认与效果：默认空。 交互与改写：合部服务不需要。要和 bootstrap_port、room 一起用。 易错：写成了客户端自己的地址。 |
| `bootstrap_port` | `PDRoutingFields` | `None` | 作用：在 `ResponsesRequest` 里，`bootstrap_port` 是prefill 上 bootstrap 服务的端口，默认服务旗标是 8998。 默认与效果：默认空，表示用服务端配置的端口。 交互与改写：请求级可以覆盖。 易错：写成 HTTP 30000。 |
| `bootstrap_room` | `PDRoutingFields` | `None` | 作用：在 `ResponsesRequest` 里，`bootstrap_room` 是同一条对话在 PD 两侧的房间号，负载策略 follow_bootstrap_room 会把它送到同一组。 默认与效果：默认空。 交互与改写：和 routed_dp_rank 一起决定落点。 易错：每条请求随机 room，前缀永远对不齐。 |
| `routed_dp_rank` | `PDRoutingFields` | `None` | 作用：在 `ResponsesRequest` 里，`routed_dp_rank` 是外部路由器指定的注意力 DP 组。 默认与效果：默认空，由负载策略选。 交互与改写：旧名 data_parallel_rank 仍接受。 易错：指定的组号大于等于组数。 |
| `disagg_prefill_dp_rank` | `PDRoutingFields` | `None` | 作用：在 `ResponsesRequest` 里，`disagg_prefill_dp_rank` 是提示 decode：KV 在哪一个 prefill 的 DP 组上。 默认与效果：默认空。 交互与改写：外部路由器填，应用客户端不要自己猜。 易错：组和真实存放 KV 的 rank 不一致。 |
| `background` | `ResponsesRequest` | `False` | 作用：在 `ResponsesRequest` 里，`background` 是Responses API 的请求放后台跑，先返回 id 再取结果。 默认与效果：默认 false。 交互与改写：需要服务端 enable_response_store，否则结果没处存。 易错：没开存储却发 background。 |
| `include` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`include` 是Responses API 要在结果里额外带哪些字段。 默认与效果：默认空。 交互与改写：不认识的项会被忽略或报错，看适配层。 易错：把采样参数写进 include。 |
| `input` | `ResponsesRequest` | `（必填）` | 作用：在 `ResponsesRequest` 里，`input` 是嵌入、分类或 Responses 的输入，可以是字符串或 token。 默认与效果：默认必填。 交互与改写：和 messages 不要同时当成同一条对话。 易错：嵌套层数和接口要求不一致。 |
| `instructions` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`instructions` 是Responses 的系统级指令。 默认与效果：默认空。 交互与改写：会进模板的系统段。 易错：和 messages 里的 system 重复，两套指令冲突。 |
| `max_output_tokens` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`max_output_tokens` 是 Responses 的输出 token 上限。 默认与效果：默认空。 交互与改写：转换后变成 max_new_tokens。 易错：和 max_tool_calls 搞混。 |
| `max_tool_calls` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`max_tool_calls` 是Responses 最多允许几次工具调用。 默认与效果：默认空，不限制次数。 交互与改写：和 parallel_tool_calls 一个管次数一个管是否并行。 易错：设成 0，工具调用直接被禁。 |
| `metadata` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`metadata` 是请求附带的元数据，Anthropic 和 Responses 用来追踪，不参与采样。 默认与效果：默认空。 交互与改写：不要把采样参数塞进这里指望生效。 易错：体积很大，占响应和日志。 |
| `model` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `parallel_tool_calls` | `ResponsesRequest` | `True` | 作用：在 `ResponsesRequest` 里，`parallel_tool_calls` 是否允许同一轮并行多个工具调用。 默认与效果：默认 true。 交互与改写：和 max_tool_calls 一起限制工具。 易错：模型模板不支持并行，却开着它。 |
| `previous_response_id` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`previous_response_id` 是Responses 续接上一次响应。 默认与效果：默认空。 交互与改写：需要 response store 才能找到上一条。 易错：id 不存在。 |
| `reasoning` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`reasoning` 是Responses 的思考配置。 默认与效果：默认空。 交互与改写：和 chat 的 reasoning_effort 不是同一个对象。 易错：结构不符合该模型模板。 |
| `service_tier` | `ResponsesRequest` | `'auto'` | 作用：在 `ResponsesRequest` 里，`service_tier` 是Responses 的服务档位，SGLang 多半只接受不按档位改变数值。 默认与效果：默认空。 交互与改写：不认识的档位可能 400。 易错：以为它会改 GPU 优先级。 |
| `store` | `ResponsesRequest` | `True` | 作用：在 `ResponsesRequest` 里，`store` 是Responses 是否把结果存进 response store。 默认与效果：默认按接口。 交互与改写：需要服务端 enable_response_store。PD 不支持。 易错：store 为真但服务没开存储。 |
| `stream` | `ResponsesRequest` | `False` | 作用：在 `ResponsesRequest` 里，`stream` 是否流式。结束帧是 data: [DONE]。Ollama chat 默认 true。 默认与效果：多数默认 false。 交互与改写：间隔还受 stream_interval 影响。 易错：按非流式去读一个不关闭的流。 |
| `temperature` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`temperature` 是采样温度。越高越随机，接近 0 趋于贪心。 默认与效果：SamplingParams 默认 1.0；HTTP 空值会收成 1.0。 交互与改写：和 top_p、top_k 一起用。 易错：不传被当成贪心。 |
| `text` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`text` 是/generate 的提示文本。可以是一批字符串。 默认与效果：默认空。 交互与改写：和 input_ids、input_embeds 至少要有一种。 易错：三种同时给，规范化报错。 |
| `tool_choice` | `ResponsesRequest` | `'auto'` | 作用：在 `ResponsesRequest` 里，`tool_choice` 是工具选择：auto、required、none，或指定某个函数。 默认与效果：默认 auto。 交互与改写：required 时模型必须调用工具，语法会收紧。 易错：指定了不存在的函数名。 |
| `tools` | `ResponsesRequest` | `factory` | 作用：在 `ResponsesRequest` 里，`tools` 是可调用的工具定义列表。 默认与效果：默认空。 交互与改写：模板负责渲染，解析器负责拆调用。 易错：schema 不是 JSON Schema。 |
| `top_logprobs` | `ResponsesRequest` | `0` | 作用：在 `ResponsesRequest` 里，`top_logprobs` 是每个位置返回几个候选的 logprob。 默认与效果：默认空。 交互与改写：要先开 logprobs。 易错：数量超过服务端上限。 |
| `top_p` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`top_p` 是核采样。1 表示不截断。 默认与效果：默认 1。 交互与改写：通常在温度之后应用。 易错：设成 0。 |
| `truncation` | `ResponsesRequest` | `'disabled'` | 作用：在 `ResponsesRequest` 里，`truncation` 是Responses 输入过长时是否截断。 默认与效果：默认按接口。 交互与改写：和 allow_auto_truncate 是不同接口上的同类开关。 易错：截断丢了用户以为还在的历史。 |
| `user` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`user` 是终端用户标识，只用于审计。 默认与效果：默认空。 交互与改写：不会自动变成缓存盐。 易错：把密钥写进 user。 |
| `chat_template_kwargs` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`chat_template_kwargs` 是传给对话模板的额外参数，例如思考开关。 默认与效果：默认空。 交互与改写：服务端 default_chat_template_kwargs 是底，请求覆盖它。 易错：键名和模板里的变量不一致，被静默忽略。 |
| `request_id` | `ResponsesRequest` | `factory` | 作用：在 `ResponsesRequest` 里，`request_id` 是Responses 的请求 id。 默认与效果：默认空，服务生成。 交互与改写：和 rid 是不同接口上的同一意图。 易错：和 previous_response_id 写成同一个。 |
| `session_id` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`session_id` 是同一会话的稳定 id。它不改写提示，和 session_params 不同。 默认与效果：默认空。 交互与改写：会话 radix 用它做引用。 易错：每条请求换一个 id，会话缓存用不上。 |
| `priority` | `ResponsesRequest` | `0` | 作用：在 `ResponsesRequest` 里，`priority` 是请求优先级。服务端没开优先级调度时可能被忽略或拒绝。 默认与效果：默认空。 交互与改写：数值方向由 schedule_low_priority_values_first 决定。 易错：以为越大越优先，服务却是小的先跑。 |
| `extra_key` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`extra_key` 是额外的缓存命名空间，和 cache_salt 类似但给路由分类用。 默认与效果：默认空。 交互与改写：不同 extra_key 不共享前缀页。 易错：每个请求都随机，缓存命中率为 0。 |
| `cache_salt` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`cache_salt` 是给前缀缓存加盐，相同 token 也不会命中另一盐的页。 默认与效果：默认空，不加盐。 交互与改写：和 extra_key、LoRA 名一起组成缓存键。 易错：把用户隐私写进盐里还打进日志。 |
| `data_parallel_rank` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`data_parallel_rank` 是routed_dp_rank 的旧名。网关仍会塞这个字段。 默认与效果：默认空。 交互与改写：两边都写时以新字段为准或同时被接受。 易错：两个字段写成不同的组号。 |
| `frequency_penalty` | `ResponsesRequest` | `0.0` | 作用：在 `ResponsesRequest` 里，`frequency_penalty` 是按出现次数惩罚 token，减轻重复。 默认与效果：默认 0。 交互与改写：可与 presence、repetition 叠加。 易错：设得过大，句子被逼换词。 |
| `presence_penalty` | `ResponsesRequest` | `0.0` | 作用：在 `ResponsesRequest` 里，`presence_penalty` 是token 只要出现过就惩罚，不看次数。 默认与效果：默认 0。 交互与改写：可与 frequency 叠加。 易错：两个都很大，分布被压扁。 |
| `stop` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`stop` 是停止字符串。采样侧会拷到 stop_strs 再清空。 默认与效果：默认空。 交互与改写：和 stop_token_ids 一起生效。 易错：停止串出现在思考里。 |
| `top_k` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`top_k` 是只在概率最高的 k 个里采样。 默认与效果：默认不限制。 交互与改写：采样掩码上限要大于它。 易错：设成 0。 |
| `min_p` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`min_p` 是丢掉概率低于该值的 token。 默认与效果：默认 0，不丢。 交互与改写：在 top-k/top-p 之后再滤。 易错：接近 1 时几乎只剩一个 token。 |
| `repetition_penalty` | `ResponsesRequest` | `None` | 作用：在 `ResponsesRequest` 里，`repetition_penalty` 是按重复缩放 logits。1 表示不惩罚。 默认与效果：默认 1。 交互与改写：和 frequency 的公式不同。 易错：设成 0。 |

### `TokenizeRequest`

定义于第 1744 行，公开字段 10 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `model` | `TokenizeRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `TokenizeRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `prompt` | `TokenizeRequest` | `None` | 作用：在 `TokenizeRequest` 里，`prompt` 是补全或 Ollama 的原文，不经对话模板。 默认与效果：补全必填。 交互与改写：chat 不读它。 易错：把 messages 历史塞进一个字符串。 |
| `messages` | `TokenizeRequest` | `None` | 作用：在 `TokenizeRequest` 里，`messages` 是对话列表。服务用 chat template 展开成 token。内容可以是文本或媒体块。 默认与效果：必填。 交互与改写：工具、思考、模板参数都读这里。 易错：role 写错或 content 结构不符合该模型模板。 |
| `tools` | `TokenizeRequest` | `None` | 作用：在 `TokenizeRequest` 里，`tools` 是可调用的工具定义列表。 默认与效果：默认空。 交互与改写：模板负责渲染，解析器负责拆调用。 易错：schema 不是 JSON Schema。 |
| `tool_choice` | `TokenizeRequest` | `None` | 作用：在 `TokenizeRequest` 里，`tool_choice` 是工具选择：auto、required、none，或指定某个函数。 默认与效果：默认 auto。 交互与改写：required 时模型必须调用工具，语法会收紧。 易错：指定了不存在的函数名。 |
| `reasoning_effort` | `TokenizeRequest` | `None` | 作用：在 `TokenizeRequest` 里，`reasoning_effort` 是限制思考力度。可以是 none 到 max 的档位，或 0 到 0.99 的数。none 会把思考关掉。max 是 SGLang 扩展，不支持的模型当 high。 默认与效果：默认空，不限制。 交互与改写：harmony 路径不支持。 易错：档位字符串拼错。 |
| `continue_final_message` | `TokenizeRequest` | `False` | 作用：在 `TokenizeRequest` 里，`continue_final_message` 是为真时接着最后一条助手消息往下写，而不是新开一轮。 默认与效果：默认 false。 交互与改写：模板不会再追加助手起始符。 易错：最后一条不是助手，续写位置错了。 |
| `chat_template_kwargs` | `TokenizeRequest` | `None` | 作用：在 `TokenizeRequest` 里，`chat_template_kwargs` 是传给对话模板的额外参数，例如思考开关。 默认与效果：默认空。 交互与改写：服务端 default_chat_template_kwargs 是底，请求覆盖它。 易错：键名和模板里的变量不一致，被静默忽略。 |
| `add_special_tokens` | `TokenizeRequest` | `True` | 作用：在 `TokenizeRequest` 里，`add_special_tokens` 是分词时是否加上 BOS/EOS 等特殊符。 默认与效果：默认按分词器。 交互与改写：和 chat template 自己加的特殊符会重复。 易错：两边都加，序列多出一个 BOS。 |
| `add_generation_prompt` | `TokenizeRequest` | `None` | 作用：在 `TokenizeRequest` 里，`add_generation_prompt` 是为真时，tokenize 接口在套模板后追加「请模型开始回答」的标记。 默认与效果：默认按模板。 交互与改写：只影响 /tokenize，不直接生成。 易错：补全场景加了对话起始符，token 和真正补全不一致。 |

### `DetokenizeRequest`

定义于第 1804 行，公开字段 3 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `model` | `DetokenizeRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `DetokenizeRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `tokens` | `DetokenizeRequest` | `（必填）` | 作用：在 `DetokenizeRequest` 里，`tokens` 是反分词接口要解的 token id 列表。 默认与效果：默认必填。 交互与改写：用服务端当前分词器解。 易错：id 属于别的模型。 |
| `skip_special_tokens` | `DetokenizeRequest` | `True` | 作用：在 `DetokenizeRequest` 里，`skip_special_tokens` 是解文本时跳过特殊符。 默认与效果：默认 true。 交互与改写：和 return_text_in_logprobs 一起决定 logprob 里的文本。 易错：关掉后特殊符泄露到用户可见文本。 |

### `TranscriptionRequest`

定义于第 2487 行，公开字段 8 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `model` | `TranscriptionRequest` | `DEFAULT_MODEL_NAME` | 作用：在 `TranscriptionRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `language` | `TranscriptionRequest` | `None` | 作用：在 `TranscriptionRequest` 里，`language` 是转写的语言提示。 默认与效果：默认空，自动检测。 交互与改写：只影响 ASR 解码偏好。 易错：写成语言全名而接口要代码。 |
| `response_format` | `TranscriptionRequest` | `'json'` | 作用：在 `TranscriptionRequest` 里，`response_format` 是输出格式，例如 json_object 或 json_schema。转写接口里则是 json、text、verbose。 默认与效果：默认空或 json。 交互与改写：会变成语法约束。 易错：schema 和格式类型不匹配。 |
| `temperature` | `TranscriptionRequest` | `0.0` | 作用：在 `TranscriptionRequest` 里，`temperature` 是采样温度。越高越随机，接近 0 趋于贪心。 默认与效果：SamplingParams 默认 1.0；HTTP 空值会收成 1.0。 交互与改写：和 top_p、top_k 一起用。 易错：不传被当成贪心。 |
| `timestamp_granularities` | `TranscriptionRequest` | `None` | 作用：在 `TranscriptionRequest` 里，`timestamp_granularities` 是转写要词级还是段级时间戳。 默认与效果：默认空。 交互与改写：只有 verbose 格式才带时间戳。 易错：要了时间戳但格式是纯 json。 |
| `stream` | `TranscriptionRequest` | `False` | 作用：在 `TranscriptionRequest` 里，`stream` 是否流式。结束帧是 data: [DONE]。Ollama chat 默认 true。 默认与效果：多数默认 false。 交互与改写：间隔还受 stream_interval 影响。 易错：按非流式去读一个不关闭的流。 |
| `audio_data` | `TranscriptionRequest` | `None` | 作用：在 `TranscriptionRequest` 里，`audio_data` 是音频输入，可以是路径、URL 或 base64。批处理时可以是列表。 默认与效果：默认空，表示没有音频。 交互与改写：和 image_data、video_data 一起进多模态处理器。 易错：URL 不在服务端允许的域名里。 |
| `audio_duration_s` | `TranscriptionRequest` | `0.0` | 作用：在 `TranscriptionRequest` 里，`audio_duration_s` 是转写请求里音频的时长秒数，给计费用。内部字段也可由服务填。 默认与效果：默认 0。 交互与改写：usage 按秒向上取整。 易错：客户端乱填时长，账单和真实长度不符。 |

## Anthropic Messages

文件 `python/sglang/srt/entrypoints/anthropic/protocol.py`。

### `AnthropicMessagesRequest`

定义于第 370 行，公开字段 20 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `bootstrap_host` | `PDRoutingFields` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`bootstrap_host` 是PD 分离时 prefill 的主机名，decode 靠它来要 KV。 默认与效果：默认空。 交互与改写：合部服务不需要。要和 bootstrap_port、room 一起用。 易错：写成了客户端自己的地址。 |
| `bootstrap_port` | `PDRoutingFields` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`bootstrap_port` 是prefill 上 bootstrap 服务的端口，默认服务旗标是 8998。 默认与效果：默认空，表示用服务端配置的端口。 交互与改写：请求级可以覆盖。 易错：写成 HTTP 30000。 |
| `bootstrap_room` | `PDRoutingFields` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`bootstrap_room` 是同一条对话在 PD 两侧的房间号，负载策略 follow_bootstrap_room 会把它送到同一组。 默认与效果：默认空。 交互与改写：和 routed_dp_rank 一起决定落点。 易错：每条请求随机 room，前缀永远对不齐。 |
| `routed_dp_rank` | `PDRoutingFields` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`routed_dp_rank` 是外部路由器指定的注意力 DP 组。 默认与效果：默认空，由负载策略选。 交互与改写：旧名 data_parallel_rank 仍接受。 易错：指定的组号大于等于组数。 |
| `disagg_prefill_dp_rank` | `PDRoutingFields` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`disagg_prefill_dp_rank` 是提示 decode：KV 在哪一个 prefill 的 DP 组上。 默认与效果：默认空。 交互与改写：外部路由器填，应用客户端不要自己猜。 易错：组和真实存放 KV 的 rank 不一致。 |
| `model` | `AnthropicMessagesRequest` | `（必填）` | 作用：在 `AnthropicMessagesRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `messages` | `AnthropicMessagesRequest` | `（必填）` | 作用：在 `AnthropicMessagesRequest` 里，`messages` 是对话列表。服务用 chat template 展开成 token。内容可以是文本或媒体块。 默认与效果：必填。 交互与改写：工具、思考、模板参数都读这里。 易错：role 写错或 content 结构不符合该模型模板。 |
| `max_tokens` | `AnthropicMessagesRequest` | `（必填）` | 作用：在 `AnthropicMessagesRequest` 里，`max_tokens` 是旧的长度上限。chat 应改用 max_completion_tokens。补全接口默认常常是 16。Anthropic 用它且必须为正。 默认与效果：chat 默认空。 交互与改写：会抄进 max_new_tokens。 易错：补全没写它，只生成 16 个 token。 |
| `metadata` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`metadata` 是请求附带的元数据，Anthropic 和 Responses 用来追踪，不参与采样。 默认与效果：默认空。 交互与改写：不要把采样参数塞进这里指望生效。 易错：体积很大，占响应和日志。 |
| `stop_sequences` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`stop_sequences` 是Anthropic 的停止串。 默认与效果：默认空。 交互与改写：转成 stop。 易错：和 max_tokens 谁先到谁停。 |
| `stream` | `AnthropicMessagesRequest` | `False` | 作用：在 `AnthropicMessagesRequest` 里，`stream` 是否流式。结束帧是 data: [DONE]。Ollama chat 默认 true。 默认与效果：多数默认 false。 交互与改写：间隔还受 stream_interval 影响。 易错：按非流式去读一个不关闭的流。 |
| `system` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`system` 是系统提示。Anthropic 可以是字符串或内容块。 默认与效果：默认空。 交互与改写：会进模板的系统段。 易错：和 messages 里第一条 system 重复。 |
| `temperature` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`temperature` 是采样温度。越高越随机，接近 0 趋于贪心。 默认与效果：SamplingParams 默认 1.0；HTTP 空值会收成 1.0。 交互与改写：和 top_p、top_k 一起用。 易错：不传被当成贪心。 |
| `thinking` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`thinking` 是思考配置，例如 {type: adaptive}。只有模板定义了思考模式的模型才读。 默认与效果：默认空。 交互与改写：Anthropic 用自己的 thinking 对象。 易错：模型没有思考模板，配置被忽略。 |
| `tool_choice` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`tool_choice` 是工具选择：auto、required、none，或指定某个函数。 默认与效果：默认 auto。 交互与改写：required 时模型必须调用工具，语法会收紧。 易错：指定了不存在的函数名。 |
| `tools` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`tools` 是可调用的工具定义列表。 默认与效果：默认空。 交互与改写：模板负责渲染，解析器负责拆调用。 易错：schema 不是 JSON Schema。 |
| `top_k` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`top_k` 是只在概率最高的 k 个里采样。 默认与效果：默认不限制。 交互与改写：采样掩码上限要大于它。 易错：设成 0。 |
| `top_p` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`top_p` 是核采样。1 表示不截断。 默认与效果：默认 1。 交互与改写：通常在温度之后应用。 易错：设成 0。 |
| `output_config` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`output_config` 是Anthropic 的输出配置，例如 JSON 格式。SDK 会带上，模式必须接住。 默认与效果：默认空。 交互与改写：会转成 response_format 一类约束。 易错：格式类型不支持。 |
| `betas` | `AnthropicMessagesRequest` | `None` | 作用：在 `AnthropicMessagesRequest` 里，`betas` 是Anthropic 请求里的 beta 功能旗标列表，SDK 会带上，服务要能接住。 默认与效果：默认空。 交互与改写：不认识的 beta 应忽略而不是 400，除非该功能被要求。 易错：把 beta 当成采样参数。 |

### `AnthropicCountTokensRequest`

定义于第 350 行，公开字段 8 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `model` | `AnthropicCountTokensRequest` | `（必填）` | 作用：在 `AnthropicCountTokensRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `messages` | `AnthropicCountTokensRequest` | `（必填）` | 作用：在 `AnthropicCountTokensRequest` 里，`messages` 是对话列表。服务用 chat template 展开成 token。内容可以是文本或媒体块。 默认与效果：必填。 交互与改写：工具、思考、模板参数都读这里。 易错：role 写错或 content 结构不符合该模型模板。 |
| `system` | `AnthropicCountTokensRequest` | `None` | 作用：在 `AnthropicCountTokensRequest` 里，`system` 是系统提示。Anthropic 可以是字符串或内容块。 默认与效果：默认空。 交互与改写：会进模板的系统段。 易错：和 messages 里第一条 system 重复。 |
| `thinking` | `AnthropicCountTokensRequest` | `None` | 作用：在 `AnthropicCountTokensRequest` 里，`thinking` 是思考配置，例如 {type: adaptive}。只有模板定义了思考模式的模型才读。 默认与效果：默认空。 交互与改写：Anthropic 用自己的 thinking 对象。 易错：模型没有思考模板，配置被忽略。 |
| `tool_choice` | `AnthropicCountTokensRequest` | `None` | 作用：在 `AnthropicCountTokensRequest` 里，`tool_choice` 是工具选择：auto、required、none，或指定某个函数。 默认与效果：默认 auto。 交互与改写：required 时模型必须调用工具，语法会收紧。 易错：指定了不存在的函数名。 |
| `tools` | `AnthropicCountTokensRequest` | `None` | 作用：在 `AnthropicCountTokensRequest` 里，`tools` 是可调用的工具定义列表。 默认与效果：默认空。 交互与改写：模板负责渲染，解析器负责拆调用。 易错：schema 不是 JSON Schema。 |
| `output_config` | `AnthropicCountTokensRequest` | `None` | 作用：在 `AnthropicCountTokensRequest` 里，`output_config` 是Anthropic 的输出配置，例如 JSON 格式。SDK 会带上，模式必须接住。 默认与效果：默认空。 交互与改写：会转成 response_format 一类约束。 易错：格式类型不支持。 |
| `betas` | `AnthropicCountTokensRequest` | `None` | 作用：在 `AnthropicCountTokensRequest` 里，`betas` 是Anthropic 请求里的 beta 功能旗标列表，SDK 会带上，服务要能接住。 默认与效果：默认空。 交互与改写：不认识的 beta 应忽略而不是 400，除非该功能被要求。 易错：把 beta 当成采样参数。 |

## Ollama

文件 `python/sglang/srt/entrypoints/ollama/protocol.py`。

### `OllamaChatRequest`

定义于第 21 行，公开字段 7 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `model` | `OllamaChatRequest` | `（必填）` | 作用：在 `OllamaChatRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `messages` | `OllamaChatRequest` | `（必填）` | 作用：在 `OllamaChatRequest` 里，`messages` 是对话列表。服务用 chat template 展开成 token。内容可以是文本或媒体块。 默认与效果：必填。 交互与改写：工具、思考、模板参数都读这里。 易错：role 写错或 content 结构不符合该模型模板。 |
| `stream` | `OllamaChatRequest` | `True` | 作用：在 `OllamaChatRequest` 里，`stream` 是否流式。结束帧是 data: [DONE]。Ollama chat 默认 true。 默认与效果：多数默认 false。 交互与改写：间隔还受 stream_interval 影响。 易错：按非流式去读一个不关闭的流。 |
| `format` | `OllamaChatRequest` | `None` | 作用：在 `OllamaChatRequest` 里，`format` 是Ollama 的输出格式，json 或一个 JSON schema。 默认与效果：默认空。 交互与改写：会转成语法约束。 易错：schema 不是对象。 |
| `options` | `OllamaChatRequest` | `None` | 作用：在 `OllamaChatRequest` 里，`options` 是Ollama 的采样选项字典，例如 temperature、num_predict。 默认与效果：默认空。 交互与改写：会映射到 SamplingParams。 易错：用了 Ollama 独有键，SGLang 没实现。 |
| `keep_alive` | `OllamaChatRequest` | `None` | 作用：在 `OllamaChatRequest` 里，`keep_alive` 是Ollama 让模型在内存里再留多久，秒或带单位的字符串。 默认与效果：默认空，用服务端策略。 交互与改写：不是 SGLang 的显存比例。 易错：写成 0，模型立刻被卸。 |
| `think` | `OllamaChatRequest` | `None` | 作用：在 `OllamaChatRequest` 里，`think` 是Ollama 的思考开关，可以是布尔或 low/medium/high。 默认与效果：默认空。 交互与改写：会映射到思考档位。 易错：写成了 max，Ollama 这个字段不认。 |

### `OllamaGenerateRequest`

定义于第 59 行，公开字段 13 项。

| 字段 | 声明类 | 默认 | 含义 |
| --- | --- | --- | --- |
| `model` | `OllamaGenerateRequest` | `（必填）` | 作用：在 `OllamaGenerateRequest` 里，`model` 是要调用的模型名，要对上 served_model_name。可用 base:adapter 选 LoRA。 默认与效果：有的接口有占位默认。 交互与改写：对不上 /v1/models 会 404。 易错：写了磁盘路径，对外名字是另一个。 |
| `prompt` | `OllamaGenerateRequest` | `（必填）` | 作用：在 `OllamaGenerateRequest` 里，`prompt` 是补全或 Ollama 的原文，不经对话模板。 默认与效果：补全必填。 交互与改写：chat 不读它。 易错：把 messages 历史塞进一个字符串。 |
| `suffix` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`suffix` 是补全接口在生成后再拼上的后缀，或 Ollama 的后缀。 默认与效果：默认空。 交互与改写：后缀不算模型生成。 易错：后缀含停止符，客户端自己截断。 |
| `system` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`system` 是系统提示。Anthropic 可以是字符串或内容块。 默认与效果：默认空。 交互与改写：会进模板的系统段。 易错：和 messages 里第一条 system 重复。 |
| `template` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`template` 是Ollama 自定义模板。 默认与效果：默认空，用模型模板。 交互与改写：raw 为真时不用它。 易错：模板语法不是 Jinja。 |
| `context` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`context` 是Ollama generate 的上下文 token 列表，用来续写。 默认与效果：默认空。 交互与改写：和 prompt 一起组成输入。 易错：上下文用了另一套分词器的 id。 |
| `stream` | `OllamaGenerateRequest` | `True` | 作用：在 `OllamaGenerateRequest` 里，`stream` 是否流式。结束帧是 data: [DONE]。Ollama chat 默认 true。 默认与效果：多数默认 false。 交互与改写：间隔还受 stream_interval 影响。 易错：按非流式去读一个不关闭的流。 |
| `raw` | `OllamaGenerateRequest` | `False` | 作用：在 `OllamaGenerateRequest` 里，`raw` 是Ollama 为真时不套模板，原文直接补全。 默认与效果：默认 false。 交互与改写：和 template 字段冲突时以 raw 为准。 易错：raw 仍传了对话模板。 |
| `format` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`format` 是Ollama 的输出格式，json 或一个 JSON schema。 默认与效果：默认空。 交互与改写：会转成语法约束。 易错：schema 不是对象。 |
| `options` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`options` 是Ollama 的采样选项字典，例如 temperature、num_predict。 默认与效果：默认空。 交互与改写：会映射到 SamplingParams。 易错：用了 Ollama 独有键，SGLang 没实现。 |
| `keep_alive` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`keep_alive` 是Ollama 让模型在内存里再留多久，秒或带单位的字符串。 默认与效果：默认空，用服务端策略。 交互与改写：不是 SGLang 的显存比例。 易错：写成 0，模型立刻被卸。 |
| `images` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`images` 是决策或 Ollama 请求里的图像列表。 默认与效果：默认空。 交互与改写：会收成 image_data。 易错：传了本地路径，服务进程看不见那个文件。 |
| `think` | `OllamaGenerateRequest` | `None` | 作用：在 `OllamaGenerateRequest` 里，`think` 是Ollama 的思考开关，可以是布尔或 low/medium/high。 默认与效果：默认空。 交互与改写：会映射到思考档位。 易错：写成了 max，Ollama 这个字段不认。 |

公开字段合计 415。
