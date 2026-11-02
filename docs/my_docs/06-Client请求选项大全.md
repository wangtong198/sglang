# 06 Client 请求选项大全

范围是 v0.5.23 里客户端真正会送到 HTTP 层的请求体。原生接口的结构在 `python/sglang/srt/managers/io_struct.py`；OpenAI 兼容层在 `python/sglang/srt/entrypoints/openai/protocol.py`，由 `OpenAIServing*.handle_request` 转成 `GenerateReqInput` 后再进 `TokenizerManager.generate_request`。下表由 AST 抽取，父类字段（例如 `PDRoutingFields`）合并在子类表中，并标明声明类。

内部字段（名字以下划线开头，或类型里带 `PrivateAttr`）保留在表内并标成内部，避免漏掉服务端会读的状态，但它们不是公开 JSON 字段。

`SamplingParams` 后半段注释写明 “Internal fields” 的成员由 `normalize()` 填写，调用方应使用前半段 API 字段。

## 原生 /generate

文件 `python/sglang/srt/managers/io_struct.py`。

### `GenerateReqInput`

定义于 `python/sglang/srt/managers/io_struct.py` 第 177 行，含继承字段共 72 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `rid` | `GenerateReqInput` | `Optional[Union[str, List[str]]]` | `field(default=None, kw_only=True)` | Request ID(s). If omitted, generated during normalization. For batch requests, a string is expanded to per-item IDs using it as a prefix. |
| `session_id` | `GenerateReqInput` | `Optional[str]` | `field(default=None, kw_only=True)` | Stable identity shared by requests in the same session. Unlike session_params, this does not alter or reconstruct the prompt. |
| `text` | `GenerateReqInput` | `Optional[Union[List[str], str]]` | `None` | The input prompt. It can be a single prompt or a batch of prompts. |
| `input_ids` | `GenerateReqInput` | `Annotated[Optional[Union[List[List[int]], List[int]]], PlainValidator(validate_optional_list_i64_1d_2d)]` | `None` | The token ids for text. Use C-loop validator to replace Pydantic per-element type check for efficiency. |
| `input_embeds` | `GenerateReqInput` | `Optional[Union[List[List[List[float]]], List[List[float]]]]` | `None` | The embeddings for input_ids; one can specify either text or input_ids or input_embeds. |
| `image_data` | `GenerateReqInput` | `Optional[MultimodalDataInputFormat]` | `None` | The image input. It can be an image instance, file name, URL, or base64 encoded string. Can be formatted as: - Single image for a single request - List of images (one per request in a batch) - List of lists of images (multiple images per request) See also python/sglang/srt/utils.py:load_image for more details. |
| `video_data` | `GenerateReqInput` | `Optional[MultimodalDataInputFormat]` | `None` | The video input. Like image data, it can be a file name, a url, or base64 encoded string. |
| `audio_data` | `GenerateReqInput` | `Optional[MultimodalDataInputFormat]` | `None` | The audio input. Like image data, it can be a file name, a url, or base64 encoded string. |
| `mm_hashes` | `GenerateReqInput` | `Optional[Union[List[str], List[List[str]]]]` | `None` | Optional per-image hashes the caller has already computed (hex strings). Single request: one hash per image. Batch request: either one hash per request when each request has one image, or one list of hashes per request. When supplied, each MultimodalDataItem's `hash` is initialised from this list and `set_pad_value` skips the internal `hash_feature()` recompute, so the resulting `pad_value` is deterministic from the caller's hash. Intended for external KV routers that compute their own per-image hash for routing decisions and need sglang's prefix-cache key to align. When unset, behavior is unchanged (sglang hashes the processor feature tensor). |
| `mm_content_hashes` | `GenerateReqInput` | `Optional[Union[List[Optional[str]], List[List[Optional[str]]]]]` | `None` | Optional `sha256:<64-hex>` identities for the original media contents. Unlike mm_hashes, these identify processor inputs and never replace the processor-output feature hash used by the embedding/prefix cache. |
| `use_audio_in_video` | `GenerateReqInput` | `bool` | `False` | Whether to extract and process audio from video inputs. |
| `video_config` | `GenerateReqInput` | `Optional[Dict[str, Any]]` | `None` | Optional request-scoped video processor configuration. |
| `sampling_params` | `GenerateReqInput` | `Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]` | `None` | The sampling_params. See descriptions below. |
| `watermark` | `GenerateReqInput` | `Any` | `None` | — |
| `return_logprob` | `GenerateReqInput` | `Optional[Union[List[bool], bool]]` | `None` | Whether to return logprobs. |
| `logprob_start_len` | `GenerateReqInput` | `Optional[Union[List[int], int]]` | `None` | If return logprobs, the start location in the prompt for returning logprobs. By default, this value is "-1", which means it will only return logprobs for output tokens. |
| `top_logprobs_num` | `GenerateReqInput` | `Optional[Union[List[int], int]]` | `None` | If return logprobs, the number of top logprobs to return at each position. |
| `token_ids_logprob` | `GenerateReqInput` | `Optional[Union[List[List[int]], List[int]]]` | `None` | If return logprobs, the token ids to return logprob for. |
| `return_sampling_mask` | `GenerateReqInput` | `Optional[Union[List[bool], bool]]` | `None` | Whether to return each output token's sampling support and behavior logprob. |
| `sampling_logprobs_mode` | `GenerateReqInput` | `Optional[Union[List[Optional[SamplingLogprobsMode]], SamplingLogprobsMode]]` | `None` | Return either the selected token's behavior logprob or behavior logprobs for the full sampling support. When omitted, selected mode is used if return_sampling_mask is enabled. |
| `return_text_in_logprobs` | `GenerateReqInput` | `bool` | `False` | Whether to detokenize tokens in text in the returned logprobs. |
| `return_flat_raw_top_logprobs` | `GenerateReqInput` | `bool` | `False` | Return prompt top logprobs as flat arrays plus shape metadata instead of the nested per-position [logprob, token_id, text] lists. |
| `return_flat_raw_output_top_logprobs` | `GenerateReqInput` | `bool` | `False` | Return output top logprobs the same way, on the finished response only. |
| `return_flat_raw_top_logprobs_b64` | `GenerateReqInput` | `bool` | `False` | Base64-encode the flat arrays. Requires one of the two flags above. |
| `stream` | `GenerateReqInput` | `bool` | `False` | Whether to stream output. |
| `log_metrics` | `GenerateReqInput` | `bool` | `True` | Whether to log metrics for this request (e.g. health_generate calls do not log metrics) |
| `return_hidden_states` | `GenerateReqInput` | `Union[List[ReturnHiddenStatesMode], ReturnHiddenStatesMode]` | `False` | Whether to return hidden states |
| `return_routed_experts` | `GenerateReqInput` | `bool` | `False` | Whether to return captured routed experts |
| `routed_experts_start_len` | `GenerateReqInput` | `int` | `0` | Absolute start position for returned routings; response covers `[routed_experts_start_len, seqlen - 1)`. Must be in [0, prompt_tokens]. 0 = full sequence. |
| `return_indexer_topk` | `GenerateReqInput` | `bool` | `False` | — |
| `modalities` | `GenerateReqInput` | `Optional[List[str]]` | `None` | The modalities of the image data [image, multi-images, video] |
| `session_params` | `GenerateReqInput` | `Optional[Dict[str, Any]]` | `None` | Session info for continual prompting |
| `lora_path` | `GenerateReqInput` | `Optional[Union[List[Optional[str]], str]]` | `None` | The path to the LoRA adaptors |
| `lora_id` | `GenerateReqInput` | `Optional[Union[List[Optional[str]], str]]` | `None` | The uid of LoRA adaptors, should be initialized by tokenizer manager |
| `custom_logit_processor` | `GenerateReqInput` | `Optional[Union[List[Optional[str]], str]]` | `None` | Custom logit processor for advanced sampling control. Must be a serialized instance of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py Use the processor's `to_str()` method to generate the serialized string. |
| `positional_embed_overrides` | `GenerateReqInput` | `Any` | `None` | Embedding overrides to place at specific token positions. Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]] Typed as Any to avoid Pydantic/FastAPI schema errors (PositionalEmbeds contains torch.Tensor). |
| `bootstrap_host` | `GenerateReqInput` | `Optional[Union[List[Optional[str]], str]]` | `None` | For disaggregated inference |
| `bootstrap_port` | `GenerateReqInput` | `Optional[Union[List[Optional[int]], int]]` | `None` | — |
| `bootstrap_room` | `GenerateReqInput` | `Optional[Union[List[Optional[int]], int]]` | `None` | — |
| `bootstrap_pair_key` | `GenerateReqInput` | `Optional[Union[List[Optional[str]], str]]` | `None` | — |
| `decode_tp_size` | `GenerateReqInput` | `Optional[Union[List[Optional[int]], int]]` | `None` | — |
| `routed_dp_rank` | `GenerateReqInput` | `Optional[int]` | `None` | For DP routing — external router assigns a specific DP worker |
| `data_parallel_rank` | `GenerateReqInput` | `Optional[int]` | `None` | Deprecated alias for `routed_dp_rank`, still accepted because sgl-model-gateway's dp-aware mode injects this spelling into every request it forwards (DPAwareWorker::prepare_request), and the OpenAI entrypoints and Engine.generate() accept it as well. |
| `disagg_prefill_dp_rank` | `GenerateReqInput` | `Optional[int]` | `None` | For PD disagg — hint telling decode which prefill DP worker has the KV cache |
| `routing_key` | `GenerateReqInput` | `Optional[str]` | `None` | Routing key for routing-key schedule policy |
| `conversation_id` | `GenerateReqInput` | `Optional[str]` | `None` | Conversation id used for tracking requests |
| `http_worker_ipc` | `GenerateReqInput` | `Optional[str]` | `field(default=None, kw_only=True)` | Internal IPC endpoint of the HTTP/tokenizer worker that owns this request. Used to route outputs back in multi-tokenizer mode. |
| `background` | `GenerateReqInput` | `bool` | `False` | For background responses (OpenAI responses API) |
| `require_reasoning` | `GenerateReqInput` | `bool` | `False` | Require reasoning for the request (hybrid reasoning model only) |
| `max_thinking_tokens` | `GenerateReqInput` | `Optional[int]` | `None` | Per-request thinking budget. Requires strict thinking so the runtime can enforce the limit rather than silently treating it as metadata. |
| `priority` | `GenerateReqInput` | `Optional[int]` | `None` | Priority for the request |
| `extra_key` | `GenerateReqInput` | `Optional[Union[List[str], str]]` | `None` | Extra cache key for caller-defined request classification. |
| `no_logs` | `GenerateReqInput` | `bool` | `False` | Whether to disallow logging for this request (e.g. due to ZDR) |
| `custom_labels` | `GenerateReqInput` | `Optional[Dict[str, str]]` | `None` | For custom metric labels |
| `return_bytes` | `GenerateReqInput` | `bool` | `False` | (Internal) Whether to return bytes for image generation |
| `return_entropy` | `GenerateReqInput` | `bool` | `False` | Whether to return entropy |
| `return_prompt_token_ids` | `GenerateReqInput` | `bool` | `False` | Whether to return prompt token IDs without computing logprobs |
| `external_trace_header` | `GenerateReqInput` | `Optional[Dict[str, Any]]` | `None` | Propagates trace context via Engine.generate/async_generate |
| `received_time` | `GenerateReqInput` | `Optional[float]` | `None` | — |
| `need_wait_for_mm_inputs` | `GenerateReqInput` | `Optional[bool]` | `None` | For EPD-disaggregated inference |
| `num_items_assigned` | `GenerateReqInput` | `Optional[Dict[Modality, List[int]]]` | `None` | — |
| `encoder_urls` | `GenerateReqInput` | `Optional[List[str]]` | `None` | Snapshot of encoder URLs at the time tokenizer-side computed ``num_items_assigned``. |
| `max_dynamic_patch` | `GenerateReqInput` | `Optional[int]` | `None` | Multimodal tiling controls (extensions) |
| `min_dynamic_patch` | `GenerateReqInput` | `Optional[int]` | `None` | — |
| `image_max_dynamic_patch` | `GenerateReqInput` | `Optional[int]` | `None` | — |
| `video_max_dynamic_patch` | `GenerateReqInput` | `Optional[int]` | `None` | — |
| `images_config` | `GenerateReqInput` | `Optional[dict]` | `None` | For Unlimited-OCR |
| `mm_processor_kwargs` | `GenerateReqInput` | `Optional[dict]` | `None` | — |
| `multi_item_delimiter_indices` | `GenerateReqInput` | `Optional[Union[List[List[int]], List[int]]]` | `None` | Pre-computed delimiter indices for multi-item scoring. Batch-level: List[List[int]] (one per request). After __getitem__: List[int]. |
| `token_indices_to_pool` | `GenerateReqInput` | `Optional[Union[List[List[int]], List[int]]]` | `None` | Token positions for setwise pooling readout (CausalLM: label-token logprobs are read AT these positions instead of the last token). Batch-level: List[List[int]] (one per request). After __getitem__: List[int]. |
| `cache_salt` | `GenerateReqInput` | `Optional[Union[List[str], str]]` | `None` | Cache namespace used to isolate otherwise-identical prefixes. |
| `kv_hints` | `GenerateReqInput` | `Optional[Union[List[Optional[Union[Dict, KvHintsEnvelope]]], Dict, KvHintsEnvelope]]` | `None` | Versioned KV-hint envelope, set by a trusted orchestrator after worker selection and never by an application client. Passed through untouched to the HiCache storage backends, each of which reads only the action types it implements. Accepts a plain dict from HTTP/gRPC; normalization converts it to KvHintsEnvelope. A hint describes one request's prefix, so a batch carries one envelope per request. |

## 原生 /encode 与 /classify

文件 `python/sglang/srt/managers/io_struct.py`。

### `EmbeddingReqInput`

定义于 `python/sglang/srt/managers/io_struct.py` 第 1204 行，含继承字段共 28 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `rid` | `EmbeddingReqInput` | `Optional[Union[str, List[str]]]` | `field(default=None, kw_only=True)` | Request ID(s). If omitted, generated during normalization. For batch requests, a string is expanded to per-item IDs using it as a prefix. |
| `text` | `EmbeddingReqInput` | `Optional[Union[List[List[str]], List[str], str]]` | `None` | The input prompt. It can be a single prompt or a batch of prompts. |
| `input_ids` | `EmbeddingReqInput` | `Optional[Union[List[List[int]], List[int]]]` | `None` | The token ids for text; one can either specify text or input_ids. |
| `input_embeds` | `EmbeddingReqInput` | `Optional[Union[List[List[List[float]]], List[List[float]]]]` | `None` | Dummy input embeds for compatibility |
| `image_data` | `EmbeddingReqInput` | `Optional[MultimodalDataInputFormat]` | `None` | The image input. It can be an image instance, file name, URL, or base64 encoded string. Can be formatted as: - Single image for a single request - List of images (one per request in a batch) - List of lists of images (multiple images per request) See also python/sglang/srt/utils.py:load_image for more details. |
| `video_data` | `EmbeddingReqInput` | `Optional[MultimodalDataInputFormat]` | `None` | The video input. Like image data, it can be a file name, a url, or base64 encoded string. |
| `audio_data` | `EmbeddingReqInput` | `Optional[MultimodalDataInputFormat]` | `None` | The audio input. Like image data, it can be a file name, a url, or base64 encoded string. |
| `embed_override_token_id` | `EmbeddingReqInput` | `Optional[int]` | `None` | Placeholder token ID used to locate embedding override positions in input token IDs. |
| `embed_overrides` | `EmbeddingReqInput` | `Any` | `None` | Unresolved embedding overrides: per-input list of tensors. Position resolution happens in the tokenizer manager after tokenization. Shape: [num_inputs][num_replacements] where each entry is a torch.Tensor of [hidden_size]. Per-input entry may be None when only some inputs in a batch need overrides. Runtime type: Optional[List[Optional[List[torch.Tensor]]]] Typed as Any to avoid Pydantic/FastAPI schema errors (contains torch.Tensor). |
| `sampling_params` | `EmbeddingReqInput` | `Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]` | `None` | Dummy sampling params for compatibility |
| `log_metrics` | `EmbeddingReqInput` | `bool` | `True` | Whether to log metrics for this request (e.g. health_generate calls do not log metrics) |
| `modalities` | `EmbeddingReqInput` | `Optional[List[str]]` | `None` | The modalities of the image data [image, multi-images, video] |
| `is_cross_encoder_request` | `EmbeddingReqInput` | `bool` | `False` | For cross-encoder requests |
| `lora_path` | `EmbeddingReqInput` | `Optional[Union[List[Optional[str]], str]]` | `None` | The path to the LoRA adaptors |
| `lora_id` | `EmbeddingReqInput` | `Optional[Union[List[Optional[str]], str]]` | `None` | The uid of LoRA adaptors, should be initialized by tokenizer manager |
| `positional_embed_overrides` | `EmbeddingReqInput` | `Any` | `None` | Resolved embedding overrides with positions (set by tokenizer manager or score mixin). Runtime type: Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]] |
| `routing_key` | `EmbeddingReqInput` | `Optional[str]` | `None` | Routing key for routing-key schedule policy |
| `http_worker_ipc` | `EmbeddingReqInput` | `Optional[str]` | `field(default=None, kw_only=True)` | Internal IPC endpoint of the HTTP/tokenizer worker that owns this request. Used to route outputs back in multi-tokenizer mode. |
| `background` | `EmbeddingReqInput` | `bool` | `False` | For background responses (OpenAI responses API) |
| `priority` | `EmbeddingReqInput` | `Optional[int]` | `None` | Priority for the request |
| `dimensions` | `EmbeddingReqInput` | `Optional[int]` | `None` | The number of dimensions the resulting output embeddings should have. It is applicable for Matryoshka Embeddings. |
| `return_pooled_hidden_states` | `EmbeddingReqInput` | `bool` | `False` | Whether to return pooled hidden states (pre-head transformer output) |
| `return_prompt_token_ids` | `EmbeddingReqInput` | `bool` | `False` | Whether to return prompt token IDs without computing logprobs |
| `external_trace_header` | `EmbeddingReqInput` | `Optional[Dict[str, Any]]` | `None` | Propagates trace context via Engine.encode/async_encode |
| `received_time` | `EmbeddingReqInput` | `Optional[float]` | `None` | — |
| `multi_item_delimiter_indices` | `EmbeddingReqInput` | `Optional[Union[List[List[int]], List[int]]]` | `None` | Pre-computed delimiter indices for multi-item scoring. Batch-level: List[List[int]] (one per request). After __getitem__: List[int]. |
| `token_indices_to_pool` | `EmbeddingReqInput` | `Optional[Union[List[List[int]], List[int]]]` | `None` | Token positions the pooler reads out AT (head / hidden states), instead of the default LAST/CLS pooling. Generic multi-position primitive (setwise scoring today). Unlike MIS, pooling is AT the position (no delimiter - 1). Batch-level: List[List[int]] (one per request). After __getitem__: List[int]. |
| `decision_layout` | `EmbeddingReqInput` | `Optional[Union[List[List[int]], List[int]]]` | `None` | Question and option spans that a Clef checkpoint's joint schema head scores, as packed by layers.joint_schema_head.pack_decision_layout. Batch-level: List[List[int]] (one per request). After __getitem__: List[int]. |

## sampling_params（/generate 内嵌，也是调度器真正消费的采样结构）

文件 `python/sglang/srt/sampling/sampling_params.py`。

### `SamplingParams`

定义于 `python/sglang/srt/sampling/sampling_params.py` 第 140 行，含继承字段共 33 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `max_new_tokens` | `SamplingParams` | `Optional[int]` | `128` | --- API parameters (set by callers) --- |
| `stop` | `SamplingParams` | `Optional[Union[str, List[str]]]` | `None` | — |
| `stop_token_ids` | `SamplingParams` | `Optional[Set[int]]` | `None` | — |
| `stop_regex` | `SamplingParams` | `Optional[Union[str, List[str]]]` | `None` | — |
| `temperature` | `SamplingParams` | `float` | `1.0` | — |
| `top_p` | `SamplingParams` | `float` | `1.0` | — |
| `top_k` | `SamplingParams` | `int` | `TOP_K_ALL` | — |
| `min_p` | `SamplingParams` | `float` | `0.0` | — |
| `frequency_penalty` | `SamplingParams` | `float` | `0.0` | — |
| `presence_penalty` | `SamplingParams` | `float` | `0.0` | — |
| `repetition_penalty` | `SamplingParams` | `float` | `1.0` | — |
| `min_new_tokens` | `SamplingParams` | `int` | `0` | — |
| `n` | `SamplingParams` | `int` | `1` | — |
| `beam_width` | `SamplingParams` | `Optional[int]` | `None` | beam_width > 1 turns the request into a beam search request; n then means "number of returned sequences" rather than parallel samples (n <= beam_width). |
| `json_schema` | `SamplingParams` | `Optional[str]` | `None` | — |
| `regex` | `SamplingParams` | `Optional[str]` | `None` | — |
| `ebnf` | `SamplingParams` | `Optional[str]` | `None` | — |
| `structural_tag` | `SamplingParams` | `Optional[str]` | `None` | — |
| `ignore_eos` | `SamplingParams` | `bool` | `False` | — |
| `skip_special_tokens` | `SamplingParams` | `bool` | `True` | — |
| `spaces_between_special_tokens` | `SamplingParams` | `bool` | `True` | — |
| `no_stop_trim` | `SamplingParams` | `bool` | `False` | — |
| `stream_interval` | `SamplingParams` | `Optional[int]` | `None` | — |
| `logit_bias` | `SamplingParams` | `Optional[Dict[str, float]]` | `None` | — |
| `sampling_seed` | `SamplingParams` | `Optional[int]` | `None` | — |
| `custom_params` | `SamplingParams` | `Optional[Dict[str, CustomParamValue]]` | `None` | — |
| `watermark` | `SamplingParams` | `Optional[WatermarkRequestConfig]` | `None` | — |
| `stop_strs` | `SamplingParams` | `Optional[Union[str, List[str]]]` | `None` | --- Internal fields (populated by __post_init__ or normalize(), not API-facing) --- |
| `stop_regex_strs` | `SamplingParams` | `Optional[Union[str, List[str]]]` | `None` | — |
| `stop_str_max_len` | `SamplingParams` | `int` | `0` | — |
| `stop_regex_max_len` | `SamplingParams` | `int` | `0` | — |
| `is_normalized` | `SamplingParams` | `bool` | `False` | — |
| `ebnf_full_assistant` | `SamplingParams` | `bool` | `False` | — |

## OpenAI Chat Completions `/v1/chat/completions`

文件 `python/sglang/srt/entrypoints/openai/protocol.py`。

### `ChatCompletionRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 909 行，含继承字段共 83 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `bootstrap_host` | `PDRoutingFields` | `Optional[Union[List[str], str]]` | `None` | For PD disaggregation |
| `bootstrap_port` | `PDRoutingFields` | `Optional[Union[List[Optional[int]], int]]` | `None` | — |
| `bootstrap_room` | `PDRoutingFields` | `Optional[Union[List[int], int]]` | `None` | — |
| `routed_dp_rank` | `PDRoutingFields` | `Optional[int]` | `None` | For DP routing -- external router assigns a specific DP worker |
| `disagg_prefill_dp_rank` | `PDRoutingFields` | `Optional[int]` | `None` | For PD disagg -- hint telling decode which prefill DP worker has the KV cache |
| `_response_parser_prefix（内部）` | `ChatCompletionRequest` | `str` | `PrivateAttr(default='')` | — |
| `_response_parser_grammar_start（内部）` | `ChatCompletionRequest` | `Optional[str]` | `PrivateAttr(default=None)` | — |
| `_response_parser_json_response_format（内部）` | `ChatCompletionRequest` | `bool` | `PrivateAttr(default=False)` | — |
| `_response_parser_tool_grammar（内部）` | `ChatCompletionRequest` | `bool` | `PrivateAttr(default=False)` | — |
| `_format_add_generation_prompt（内部）` | `ChatCompletionRequest` | `Optional[bool]` | `PrivateAttr(default=None)` | — |
| `_format_add_special_tokens（内部）` | `ChatCompletionRequest` | `bool` | `PrivateAttr(default=False)` | — |
| `messages` | `ChatCompletionRequest` | `List[ChatCompletionMessageParam]` | `（必填）` | Ordered by official OpenAI API documentation https://platform.openai.com/docs/api-reference/chat/create |
| `model` | `ChatCompletionRequest` | `str` | `DEFAULT_MODEL_NAME` | Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax. |
| `frequency_penalty` | `ChatCompletionRequest` | `float` | `0.0` | — |
| `logit_bias` | `ChatCompletionRequest` | `Optional[Dict[str, float]]` | `None` | — |
| `logprobs` | `ChatCompletionRequest` | `bool` | `False` | — |
| `top_logprobs` | `ChatCompletionRequest` | `Optional[int]` | `None` | — |
| `max_tokens` | `ChatCompletionRequest` | `Optional[int]` | `None` | The maximum number of tokens that can be generated in the chat completion. |
| `max_completion_tokens` | `ChatCompletionRequest` | `Optional[int]` | `None` | The maximum number of completion tokens for a chat completion request, including visible output tokens and reasoning tokens. Input tokens are not included. |
| `n` | `ChatCompletionRequest` | `int` | `1` | — |
| `presence_penalty` | `ChatCompletionRequest` | `float` | `0.0` | — |
| `response_format` | `ChatCompletionRequest` | `Optional[Union[ResponseFormat, StructuralTagResponseFormat]]` | `None` | — |
| `seed` | `ChatCompletionRequest` | `Optional[int]` | `None` | — |
| `stop` | `ChatCompletionRequest` | `Optional[Union[str, List[str]]]` | `None` | — |
| `stream` | `ChatCompletionRequest` | `bool` | `False` | — |
| `stream_options` | `ChatCompletionRequest` | `Optional[StreamOptions]` | `None` | — |
| `temperature` | `ChatCompletionRequest` | `Optional[float]` | `None` | — |
| `top_p` | `ChatCompletionRequest` | `Optional[float]` | `None` | — |
| `user` | `ChatCompletionRequest` | `Optional[str]` | `None` | — |
| `tools` | `ChatCompletionRequest` | `Optional[List[Tool]]` | `None` | — |
| `_template_tools（内部）` | `ChatCompletionRequest` | `Optional[List[Dict[str, Any]]]` | `PrivateAttr(default=None)` | Set by /v1/responses when the request has namespace tools: the chat template renders these, while ``tools`` (flattened to ``<namespace>.<function>``) still drives tool-call parsing. |
| `tool_choice` | `ChatCompletionRequest` | `Union[ToolChoice, Literal['auto', 'required', 'none']]` | `'auto'` | — |
| `parallel_tool_calls` | `ChatCompletionRequest` | `bool` | `True` | — |
| `return_hidden_states` | `ChatCompletionRequest` | `Union[bool, Literal['last']]` | `False` | — |
| `return_routed_experts` | `ChatCompletionRequest` | `bool` | `False` | — |
| `routed_experts_start_len` | `ChatCompletionRequest` | `int` | `0` | — |
| `return_cached_tokens_details` | `ChatCompletionRequest` | `bool` | `False` | — |
| `return_spec_tokens_details` | `ChatCompletionRequest` | `bool` | `False` | — |
| `return_prompt_token_ids` | `ChatCompletionRequest` | `bool` | `False` | — |
| `return_token_ids` | `ChatCompletionRequest` | `bool` | `False` | — |
| `return_meta_info` | `ChatCompletionRequest` | `bool` | `False` | — |
| `return_input_ids_in_sglext` | `ChatCompletionRequest` | `bool` | `False` | — |
| `return_output_ids_in_sglext` | `ChatCompletionRequest` | `bool` | `False` | — |
| `return_sampling_mask` | `ChatCompletionRequest` | `bool` | `False` | — |
| `sampling_logprobs_mode` | `ChatCompletionRequest` | `Optional[Literal['selected', 'support']]` | `None` | — |
| `return_flat_raw_output_top_logprobs` | `ChatCompletionRequest` | `bool` | `False` | Flat output top logprobs in choices[].meta_info (requires return_meta_info); choices[].logprobs top_logprobs are then empty. |
| `return_flat_raw_top_logprobs_b64` | `ChatCompletionRequest` | `bool` | `True` | Base64 by default on chat, unlike /generate; false returns plain JSON lists. |
| `reasoning_effort` | `ChatCompletionRequest` | `ReasoningEffortType` | `None` | Constrains effort on reasoning for reasoning models. Accepts string levels ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'ultra', 'max') or a float in [0.0, 0.99] for fine-grained control. 'none' disables reasoning entirely, 'low' is the least effort, 'high' is the most effort. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response. 'none' defaults thinking and enable_thinking to false in chat_template_kwargs (unless explicitly overridden). Not supported in the harmony path.'max' is an sglang extension to the OpenAI schema for models that expose a maximum-effort tier above 'high'; models that don't support it treat it the same as 'high'. |
| `thinking` | `ChatCompletionRequest` | `Optional[Any]` | `None` | Thinking config such as {'type': 'adaptive'}; read only by models whose chat template defines a thinking mode, ignored otherwise. |
| `task` | `ChatCompletionRequest` | `Optional[Literal['action', 'query', 'authority', 'domain', 'title', 'read_url']]` | `None` | DeepSeek-V4 quick instruction task. When set, the last user/developer message is treated as a single-shot classification prompt and the corresponding task special token (e.g. `<｜domain｜>`) is appended before generation. Only honored by the dsv4/dsv41 chat encoders; ignored otherwise. |
| `top_k` | `ChatCompletionRequest` | `Optional[int]` | `None` | Extra parameters for SRT backend only and will be ignored by OpenAI models. |
| `min_p` | `ChatCompletionRequest` | `Optional[float]` | `None` | — |
| `min_tokens` | `ChatCompletionRequest` | `int` | `0` | — |
| `regex` | `ChatCompletionRequest` | `Optional[str]` | `None` | — |
| `ebnf` | `ChatCompletionRequest` | `Optional[str]` | `None` | — |
| `repetition_penalty` | `ChatCompletionRequest` | `Optional[float]` | `None` | — |
| `stop_token_ids` | `ChatCompletionRequest` | `Optional[List[int]]` | `None` | — |
| `stop_regex` | `ChatCompletionRequest` | `Optional[Union[str, List[str]]]` | `None` | — |
| `no_stop_trim` | `ChatCompletionRequest` | `bool` | `False` | — |
| `ignore_eos` | `ChatCompletionRequest` | `bool` | `False` | — |
| `continue_final_message` | `ChatCompletionRequest` | `bool` | `False` | — |
| `skip_special_tokens` | `ChatCompletionRequest` | `bool` | `True` | — |
| `lora_path` | `ChatCompletionRequest` | `Optional[Union[List[Optional[str]], Optional[str]]]` | `None` | — |
| `session_id` | `ChatCompletionRequest` | `Optional[str]` | `None` | — |
| `session_params` | `ChatCompletionRequest` | `Optional[Dict]` | `None` | — |
| `separate_reasoning` | `ChatCompletionRequest` | `bool` | `True` | — |
| `stream_reasoning` | `ChatCompletionRequest` | `bool` | `True` | — |
| `chat_template_kwargs` | `ChatCompletionRequest` | `Optional[Dict]` | `None` | — |
| `max_dynamic_patch` | `ChatCompletionRequest` | `Optional[int]` | `None` | SGLang multimodal controls (extensions) |
| `min_dynamic_patch` | `ChatCompletionRequest` | `Optional[int]` | `None` | — |
| `use_audio_in_video` | `ChatCompletionRequest` | `bool` | `False` | — |
| `images_config` | `ChatCompletionRequest` | `Optional[Dict]` | `None` | — |
| `video_config` | `ChatCompletionRequest` | `Optional[Dict]` | `None` | — |
| `mm_processor_kwargs` | `ChatCompletionRequest` | `Optional[Dict]` | `None` | Per-request processor overrides; only MiniCPM-V 4.7 reads them (downsample_mode, max_slice_nums), flattening images_/videos_kwargs into one request-wide profile. |
| `custom_logit_processor` | `ChatCompletionRequest` | `Optional[Union[List[Optional[str]], str]]` | `None` | Custom logit processor for advanced sampling control |
| `custom_params` | `ChatCompletionRequest` | `Optional[Dict]` | `None` | — |
| `watermark` | `ChatCompletionRequest` | `Optional[Dict[str, Any]]` | `None` | Per-request watermark settings: enabled, key, and context_window. A key implicitly enables watermarking. |
| `input_ids` | `ChatCompletionRequest` | `Optional[List[int]]` | `None` | Pre-computed prompt token IDs: when provided, bypasses chat template tokenization entirely. Messages are still used to derive stop tokens and tool_call_constraint. |
| `rid` | `ChatCompletionRequest` | `Optional[Union[List[str], str]]` | `None` | For request id |
| `extra_key` | `ChatCompletionRequest` | `Optional[Union[List[str], str]]` | `None` | Extra key for caller-defined request classification |
| `cache_salt` | `ChatCompletionRequest` | `Optional[Union[List[str], str]]` | `None` | Cache salt for request caching |
| `priority` | `ChatCompletionRequest` | `Optional[int]` | `None` | Priority for the request |
| `data_parallel_rank` | `ChatCompletionRequest` | `Optional[int]` | `None` | Deprecated: use routed_dp_rank instead |

## OpenAI Completions `/v1/completions`

文件 `python/sglang/srt/entrypoints/openai/protocol.py`。

### `CompletionRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 365 行，含继承字段共 56 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `bootstrap_host` | `PDRoutingFields` | `Optional[Union[List[str], str]]` | `None` | For PD disaggregation |
| `bootstrap_port` | `PDRoutingFields` | `Optional[Union[List[Optional[int]], int]]` | `None` | — |
| `bootstrap_room` | `PDRoutingFields` | `Optional[Union[List[int], int]]` | `None` | — |
| `routed_dp_rank` | `PDRoutingFields` | `Optional[int]` | `None` | For DP routing -- external router assigns a specific DP worker |
| `disagg_prefill_dp_rank` | `PDRoutingFields` | `Optional[int]` | `None` | For PD disagg -- hint telling decode which prefill DP worker has the KV cache |
| `model` | `CompletionRequest` | `str` | `DEFAULT_MODEL_NAME` | Ordered by official OpenAI API documentation https://platform.openai.com/docs/api-reference/completions/create Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax. |
| `prompt` | `CompletionRequest` | `Union[List[int], List[List[int]], str, List[str]]` | `（必填）` | — |
| `best_of` | `CompletionRequest` | `Optional[int]` | `None` | — |
| `echo` | `CompletionRequest` | `bool` | `False` | — |
| `frequency_penalty` | `CompletionRequest` | `float` | `0.0` | — |
| `logit_bias` | `CompletionRequest` | `Optional[Dict[str, float]]` | `None` | — |
| `logprobs` | `CompletionRequest` | `Optional[int]` | `None` | — |
| `max_tokens` | `CompletionRequest` | `int` | `16` | — |
| `n` | `CompletionRequest` | `int` | `1` | — |
| `presence_penalty` | `CompletionRequest` | `float` | `0.0` | — |
| `seed` | `CompletionRequest` | `Optional[int]` | `None` | — |
| `stop` | `CompletionRequest` | `Optional[Union[str, List[str]]]` | `None` | — |
| `stream` | `CompletionRequest` | `bool` | `False` | — |
| `stream_options` | `CompletionRequest` | `Optional[StreamOptions]` | `None` | — |
| `suffix` | `CompletionRequest` | `Optional[str]` | `None` | — |
| `temperature` | `CompletionRequest` | `float` | `1.0` | — |
| `top_p` | `CompletionRequest` | `float` | `1.0` | — |
| `user` | `CompletionRequest` | `Optional[str]` | `None` | — |
| `return_hidden_states` | `CompletionRequest` | `Union[bool, Literal['last']]` | `False` | — |
| `return_routed_experts` | `CompletionRequest` | `bool` | `False` | — |
| `routed_experts_start_len` | `CompletionRequest` | `int` | `0` | — |
| `return_cached_tokens_details` | `CompletionRequest` | `bool` | `False` | — |
| `return_spec_tokens_details` | `CompletionRequest` | `bool` | `False` | — |
| `return_token_ids` | `CompletionRequest` | `bool` | `False` | — |
| `top_k` | `CompletionRequest` | `int` | `-1` | Extra parameters for SRT backend only and will be ignored by OpenAI models. |
| `min_p` | `CompletionRequest` | `float` | `0.0` | — |
| `min_tokens` | `CompletionRequest` | `int` | `0` | — |
| `json_schema` | `CompletionRequest` | `Optional[str]` | `None` | — |
| `regex` | `CompletionRequest` | `Optional[str]` | `None` | — |
| `ebnf` | `CompletionRequest` | `Optional[str]` | `None` | — |
| `repetition_penalty` | `CompletionRequest` | `float` | `1.0` | — |
| `stop_token_ids` | `CompletionRequest` | `Optional[List[int]]` | `None` | — |
| `stop_regex` | `CompletionRequest` | `Optional[Union[str, List[str]]]` | `None` | — |
| `no_stop_trim` | `CompletionRequest` | `bool` | `False` | — |
| `ignore_eos` | `CompletionRequest` | `bool` | `False` | — |
| `skip_special_tokens` | `CompletionRequest` | `bool` | `True` | — |
| `lora_path` | `CompletionRequest` | `Optional[Union[List[Optional[str]], Optional[str]]]` | `None` | — |
| `session_id` | `CompletionRequest` | `Optional[str]` | `None` | — |
| `session_params` | `CompletionRequest` | `Optional[Dict]` | `None` | — |
| `response_format` | `CompletionRequest` | `Optional[Union[ResponseFormat, StructuralTagResponseFormat]]` | `None` | — |
| `custom_params` | `CompletionRequest` | `Optional[Dict]` | `None` | — |
| `custom_logit_processor` | `CompletionRequest` | `Optional[str]` | `None` | — |
| `watermark` | `CompletionRequest` | `Optional[Dict[str, Any]]` | `None` | Per-request watermark settings: enabled, key, and context_window. A key implicitly enables watermarking. |
| `images_config` | `CompletionRequest` | `Optional[Dict]` | `None` | — |
| `mm_processor_kwargs` | `CompletionRequest` | `Optional[Dict]` | `None` | Per-request processor overrides; only MiniCPM-V 4.7 reads them (downsample_mode, max_slice_nums), flattening images_/videos_kwargs into one request-wide profile. |
| `data_parallel_rank` | `CompletionRequest` | `Optional[int]` | `None` | Deprecated: use routed_dp_rank instead |
| `rid` | `CompletionRequest` | `Optional[Union[List[str], str]]` | `None` | For request id |
| `extra_key` | `CompletionRequest` | `Optional[Union[List[str], str]]` | `None` | Extra key for caller-defined request classification |
| `cache_salt` | `CompletionRequest` | `Optional[Union[List[str], str]]` | `None` | Cache salt for request caching |
| `priority` | `CompletionRequest` | `Optional[int]` | `None` | Priority for the request |
| `custom_labels` | `CompletionRequest` | `Optional[Dict[str, str]]` | `None` | For custom metric labels |

## OpenAI Embeddings / Classify / Score / Rerank / Responses

文件 `python/sglang/srt/entrypoints/openai/protocol.py`。

### `EmbeddingRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 1416 行，含继承字段共 10 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `input` | `EmbeddingRequest` | `EmbeddingInput` | `（必填）` | Ordered by official OpenAI API documentation https://platform.openai.com/docs/api-reference/embeddings/create |
| `model` | `EmbeddingRequest` | `str` | `DEFAULT_MODEL_NAME` | — |
| `encoding_format` | `EmbeddingRequest` | `str` | `'float'` | — |
| `dimensions` | `EmbeddingRequest` | `Optional[int]` | `None` | — |
| `user` | `EmbeddingRequest` | `Optional[str]` | `None` | — |
| `rid` | `EmbeddingRequest` | `Optional[Union[List[str], str]]` | `None` | The request id. |
| `priority` | `EmbeddingRequest` | `Optional[int]` | `None` | Priority for the request |
| `lora_path` | `EmbeddingRequest` | `Optional[Union[List[Optional[str]], Optional[str]]]` | `None` | LoRA adapter path(s) |
| `embed_override_token_id` | `EmbeddingRequest` | `Optional[int]` | `None` | Placeholder token id used to locate embedding override positions in input token IDs. |
| `embed_overrides` | `EmbeddingRequest` | `Optional[List[Optional[List[List[float]]]]]` | `None` | Per-input embedding overrides (null entries skip that input). Shape: [num_inputs][num_replacements][hidden_size] |

### `ClassifyRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 1447 行，含继承字段共 5 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `model` | `ClassifyRequest` | `str` | `DEFAULT_MODEL_NAME` | OpenAI-compatible classification request |
| `input` | `ClassifyRequest` | `ClassifyInput` | `（必填）` | — |
| `user` | `ClassifyRequest` | `Optional[str]` | `None` | — |
| `rid` | `ClassifyRequest` | `Optional[Union[List[str], str]]` | `None` | The request id. |
| `priority` | `ClassifyRequest` | `Optional[int]` | `None` | Priority for the request |

### `ScoringRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 1482 行，含继承字段共 13 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `query` | `ScoringRequest` | `Optional[Union[str, List[int]]]` | `None` | — |
| `items` | `ScoringRequest` | `Optional[Union[str, List[str], List[List[int]]]]` | `None` | — |
| `embed_override_token_id` | `ScoringRequest` | `Optional[int]` | `None` | Placeholder token id used to locate embedding override positions in query/items. |
| `query_embed_overrides` | `ScoringRequest` | `Optional[List[List[float]]]` | `None` | Query embedding overrides. |
| `item_embed_overrides` | `ScoringRequest` | `Optional[List[Optional[List[List[float]]]]]` | `None` | Per-item embedding overrides (null entries skip that item). |
| `label_token_ids` | `ScoringRequest` | `Optional[Union[List[int], List[List[int]]]]` | `None` | — |
| `apply_softmax` | `ScoringRequest` | `bool` | `False` | — |
| `temperature` | `ScoringRequest` | `float` | `1.0` | — |
| `return_token_logprobs` | `ScoringRequest` | `bool` | `False` | — |
| `item_first` | `ScoringRequest` | `bool` | `False` | — |
| `return_pooled_hidden_states` | `ScoringRequest` | `bool` | `False` | — |
| `score_extraction_token` | `ScoringRequest` | `Optional[str]` | `None` | Setwise readout: when set, the readout is taken AT every occurrence of this token in each `query + item` sequence instead of the last token, and `scores` is returned nested (one `[Nᵢ x num_labels]` matrix per item). SeqCls pools the head there; CausalLM reads label-token logprobs there. Both support batched and --enable-mis. |
| `model` | `ScoringRequest` | `str` | `DEFAULT_MODEL_NAME` | — |

### `DecisionRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 1639 行，含继承字段共 8 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `input` | `DecisionRequest` | `RequiredDecisionText` | `（必填）` | — |
| `images` | `DecisionRequest` | `List[DecisionImage]` | `factory` | — |
| `questions` | `DecisionRequest` | `List[DecisionQuestion]` | `（必填）` | — |
| `temperature` | `DecisionRequest` | `float` | `1.0` | Scales option probabilities only, not label_mass. |
| `chat_template_kwargs` | `DecisionRequest` | `Dict[str, Any]` | `factory` | Applied over the server defaults, with the template reasoning toggle off. |
| `prompt_format_version` | `DecisionRequest` | `Optional[int]` | `None` | Pins the server-owned prompt wording. A different served version is refused. |
| `return_prompt_token_ids` | `DecisionRequest` | `bool` | `False` | — |
| `model` | `DecisionRequest` | `str` | `DEFAULT_MODEL_NAME` | — |

### `V1RerankReqInput`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 1687 行，含继承字段共 5 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `query` | `V1RerankReqInput` | `RerankContent` | `...` | The query to match against documents. Can be a string (text-only) or a list of content parts for multimodal queries (text, image_url, video_url). |
| `documents` | `V1RerankReqInput` | `List[RerankContent]` | `...` | List of documents to rank. Each document can be a string (text-only) or a list of content parts for multimodal documents (text, image_url, video_url). |
| `instruct` | `V1RerankReqInput` | `Optional[str]` | `None` | The instruct to the reranker model. |
| `top_n` | `V1RerankReqInput` | `Optional[int]` | `None` | Maximum number of documents to return. Defaults to returning all documents. If specified value is greater than the total number of documents, all documents will be returned. |
| `return_documents` | `V1RerankReqInput` | `bool` | `True` | Whether to return documents in the response. Only included when set to true. |

### `ResponsesRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 1924 行，含继承字段共 45 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `bootstrap_host` | `PDRoutingFields` | `Optional[Union[List[str], str]]` | `None` | For PD disaggregation |
| `bootstrap_port` | `PDRoutingFields` | `Optional[Union[List[Optional[int]], int]]` | `None` | — |
| `bootstrap_room` | `PDRoutingFields` | `Optional[Union[List[int], int]]` | `None` | — |
| `routed_dp_rank` | `PDRoutingFields` | `Optional[int]` | `None` | For DP routing -- external router assigns a specific DP worker |
| `disagg_prefill_dp_rank` | `PDRoutingFields` | `Optional[int]` | `None` | For PD disagg -- hint telling decode which prefill DP worker has the KV cache |
| `_tool_state（内部）` | `ResponsesRequest` | `ResponseToolState` | `PrivateAttr(default_factory=ResponseToolState)` | — |
| `_response_parser_prefix（内部）` | `ResponsesRequest` | `str` | `PrivateAttr(default='')` | — |
| `_response_parser_grammar_start（内部）` | `ResponsesRequest` | `Optional[str]` | `PrivateAttr(default=None)` | — |
| `_response_parser_json_response_format（内部）` | `ResponsesRequest` | `bool` | `PrivateAttr(default=False)` | — |
| `_response_parser_tool_grammar（内部）` | `ResponsesRequest` | `bool` | `PrivateAttr(default=False)` | — |
| `background` | `ResponsesRequest` | `Optional[bool]` | `False` | Core OpenAI API fields (ordered by official documentation) |
| `include` | `ResponsesRequest` | `Optional[List[Literal['code_interpreter_call.outputs', 'computer_call_output.output.image_url', 'file_search_call.results', 'message.input_image.image_url', 'message.output_text.logprobs', 'reasoning.encrypted_content']]]` | `None` | — |
| `input` | `ResponsesRequest` | `Union[str, List[Dict[str, Any]], List[ResponseInputOutputItem]]` | `（必填）` | Preserve JSON input for downstream validation; SDK TypedDicts discard extra fields. |
| `instructions` | `ResponsesRequest` | `Optional[str]` | `None` | — |
| `max_output_tokens` | `ResponsesRequest` | `Optional[int]` | `None` | — |
| `max_tool_calls` | `ResponsesRequest` | `Optional[int]` | `None` | — |
| `metadata` | `ResponsesRequest` | `Optional[Dict[str, Any]]` | `None` | — |
| `model` | `ResponsesRequest` | `Optional[str]` | `None` | — |
| `parallel_tool_calls` | `ResponsesRequest` | `Optional[bool]` | `True` | — |
| `previous_response_id` | `ResponsesRequest` | `Optional[str]` | `None` | — |
| `reasoning` | `ResponsesRequest` | `Optional[ResponseReasoningParam]` | `None` | — |
| `service_tier` | `ResponsesRequest` | `Literal['auto', 'default', 'flex', 'scale', 'priority']` | `'auto'` | — |
| `store` | `ResponsesRequest` | `Optional[bool]` | `True` | — |
| `stream` | `ResponsesRequest` | `Optional[bool]` | `False` | — |
| `temperature` | `ResponsesRequest` | `Optional[float]` | `None` | — |
| `text` | `ResponsesRequest` | `Optional[ResponseTextConfig]` | `None` | — |
| `tool_choice` | `ResponsesRequest` | `Union[Literal['auto', 'required', 'none'], Dict[str, Any]]` | `'auto'` | — |
| `tools` | `ResponsesRequest` | `List[ResponseTool]` | `factory` | — |
| `top_logprobs` | `ResponsesRequest` | `Optional[int]` | `0` | — |
| `top_p` | `ResponsesRequest` | `Optional[float]` | `None` | — |
| `truncation` | `ResponsesRequest` | `Optional[Literal['auto', 'disabled']]` | `'disabled'` | — |
| `user` | `ResponsesRequest` | `Optional[str]` | `None` | — |
| `chat_template_kwargs` | `ResponsesRequest` | `Optional[Dict[str, Any]]` | `None` | Extra SGLang parameters |
| `request_id` | `ResponsesRequest` | `str` | `factory` | The request_id related to this request. If the caller does not set it, a random uuid will be generated. |
| `session_id` | `ResponsesRequest` | `Optional[str]` | `None` | — |
| `priority` | `ResponsesRequest` | `int` | `0` | Request priority |
| `extra_key` | `ResponsesRequest` | `Optional[str]` | `None` | Extra key for caller-defined request classification |
| `cache_salt` | `ResponsesRequest` | `Optional[str]` | `None` | Cache salt for request caching |
| `data_parallel_rank` | `ResponsesRequest` | `Optional[int]` | `None` | Deprecated: use routed_dp_rank instead |
| `frequency_penalty` | `ResponsesRequest` | `float` | `0.0` | SGLang sampling extras. ``None`` defers to ``--preferred-sampling-params``. |
| `presence_penalty` | `ResponsesRequest` | `float` | `0.0` | — |
| `stop` | `ResponsesRequest` | `Optional[Union[str, List[str]]]` | `None` | — |
| `top_k` | `ResponsesRequest` | `Optional[int]` | `None` | — |
| `min_p` | `ResponsesRequest` | `Optional[float]` | `None` | — |
| `repetition_penalty` | `ResponsesRequest` | `Optional[float]` | `None` | — |

### `TokenizeRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 1744 行，含继承字段共 10 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `model` | `TokenizeRequest` | `str` | `DEFAULT_MODEL_NAME` | — |
| `prompt` | `TokenizeRequest` | `Optional[Union[str, List[str]]]` | `None` | — |
| `messages` | `TokenizeRequest` | `Optional[List[ChatCompletionMessageParam]]` | `None` | — |
| `tools` | `TokenizeRequest` | `Optional[List[Tool]]` | `None` | — |
| `tool_choice` | `TokenizeRequest` | `Optional[Union[ToolChoice, Literal['auto', 'required', 'none']]]` | `None` | — |
| `reasoning_effort` | `TokenizeRequest` | `ReasoningEffortType` | `None` | — |
| `continue_final_message` | `TokenizeRequest` | `bool` | `False` | — |
| `chat_template_kwargs` | `TokenizeRequest` | `Optional[Dict]` | `None` | — |
| `add_special_tokens` | `TokenizeRequest` | `bool` | `True` | whether to add model-specific special tokens (e.g. BOS/EOS) during encoding. |
| `add_generation_prompt` | `TokenizeRequest` | `Optional[bool]` | `None` | With messages: render them as given, passing this to the chat template, and encode with add_special_tokens (default false). Omitted, messages are rendered as for a chat completion. |

### `DetokenizeRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 1804 行，含继承字段共 3 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `model` | `DetokenizeRequest` | `str` | `DEFAULT_MODEL_NAME` | — |
| `tokens` | `DetokenizeRequest` | `Union[List[int], List[List[int]]]` | `（必填）` | — |
| `skip_special_tokens` | `DetokenizeRequest` | `bool` | `True` | whether to exclude special tokens (e.g. padding or EOS) during decoding. |

### `TranscriptionRequest`

定义于 `python/sglang/srt/entrypoints/openai/protocol.py` 第 2487 行，含继承字段共 8 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `model` | `TranscriptionRequest` | `str` | `DEFAULT_MODEL_NAME` | — |
| `language` | `TranscriptionRequest` | `Optional[str]` | `None` | — |
| `response_format` | `TranscriptionRequest` | `str` | `'json'` | — |
| `temperature` | `TranscriptionRequest` | `float` | `0.0` | — |
| `timestamp_granularities` | `TranscriptionRequest` | `Optional[List[str]]` | `None` | — |
| `stream` | `TranscriptionRequest` | `bool` | `False` | — |
| `audio_data` | `TranscriptionRequest` | `Optional[bytes]` | `None` | Internal fields (not from API) |
| `audio_duration_s` | `TranscriptionRequest` | `float` | `0.0` | — |

## Anthropic Messages

文件 `python/sglang/srt/entrypoints/anthropic/protocol.py`。

### `AnthropicMessagesRequest`

定义于 `python/sglang/srt/entrypoints/anthropic/protocol.py` 第 370 行，含继承字段共 20 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `bootstrap_host` | `PDRoutingFields` | `Optional[Union[List[str], str]]` | `None` | For PD disaggregation |
| `bootstrap_port` | `PDRoutingFields` | `Optional[Union[List[Optional[int]], int]]` | `None` | — |
| `bootstrap_room` | `PDRoutingFields` | `Optional[Union[List[int], int]]` | `None` | — |
| `routed_dp_rank` | `PDRoutingFields` | `Optional[int]` | `None` | For DP routing -- external router assigns a specific DP worker |
| `disagg_prefill_dp_rank` | `PDRoutingFields` | `Optional[int]` | `None` | For PD disagg -- hint telling decode which prefill DP worker has the KV cache |
| `model` | `AnthropicMessagesRequest` | `str` | `（必填）` | — |
| `messages` | `AnthropicMessagesRequest` | `list[AnthropicMessage]` | `（必填）` | — |
| `max_tokens` | `AnthropicMessagesRequest` | `int` | `（必填）` | — |
| `metadata` | `AnthropicMessagesRequest` | `Optional[dict[str, Any]]` | `None` | — |
| `stop_sequences` | `AnthropicMessagesRequest` | `Optional[list[str]]` | `None` | — |
| `stream` | `AnthropicMessagesRequest` | `Optional[bool]` | `False` | — |
| `system` | `AnthropicMessagesRequest` | `Optional[Union[str, list[AnthropicContentBlock]]]` | `None` | — |
| `temperature` | `AnthropicMessagesRequest` | `Optional[float]` | `None` | — |
| `thinking` | `AnthropicMessagesRequest` | `Optional[AnthropicThinkingParam]` | `None` | — |
| `tool_choice` | `AnthropicMessagesRequest` | `Optional[AnthropicToolChoice]` | `None` | — |
| `tools` | `AnthropicMessagesRequest` | `Optional[list[AnthropicTool]]` | `None` | — |
| `top_k` | `AnthropicMessagesRequest` | `Optional[int]` | `None` | — |
| `top_p` | `AnthropicMessagesRequest` | `Optional[float]` | `None` | — |
| `output_config` | `AnthropicMessagesRequest` | `Optional[AnthropicOutputConfig]` | `None` | Claude 4.7 fields. The Anthropic SDK / Claude Code attach these even when targeting non-Anthropic backends, so the schema must accept them. |
| `betas` | `AnthropicMessagesRequest` | `Optional[list[str]]` | `None` | — |

### `AnthropicCountTokensRequest`

定义于 `python/sglang/srt/entrypoints/anthropic/protocol.py` 第 350 行，含继承字段共 8 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `model` | `AnthropicCountTokensRequest` | `str` | `（必填）` | — |
| `messages` | `AnthropicCountTokensRequest` | `list[AnthropicMessage]` | `（必填）` | — |
| `system` | `AnthropicCountTokensRequest` | `Optional[Union[str, list[AnthropicContentBlock]]]` | `None` | — |
| `thinking` | `AnthropicCountTokensRequest` | `Optional[AnthropicThinkingParam]` | `None` | — |
| `tool_choice` | `AnthropicCountTokensRequest` | `Optional[AnthropicToolChoice]` | `None` | — |
| `tools` | `AnthropicCountTokensRequest` | `Optional[list[AnthropicTool]]` | `None` | — |
| `output_config` | `AnthropicCountTokensRequest` | `Optional[AnthropicOutputConfig]` | `None` | Claude 4.7 / SDK-compatibility fields; betas is accepted but ignored. |
| `betas` | `AnthropicCountTokensRequest` | `Optional[list[str]]` | `None` | — |

## Ollama

文件 `python/sglang/srt/entrypoints/ollama/protocol.py`。

### `OllamaChatRequest`

定义于 `python/sglang/srt/entrypoints/ollama/protocol.py` 第 21 行，含继承字段共 7 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `model` | `OllamaChatRequest` | `str` | `（必填）` | — |
| `messages` | `OllamaChatRequest` | `List[OllamaMessage]` | `（必填）` | — |
| `stream` | `OllamaChatRequest` | `bool` | `True` | — |
| `format` | `OllamaChatRequest` | `Optional[Union[Literal['json'], Dict[str, Any]]]` | `None` | — |
| `options` | `OllamaChatRequest` | `Optional[Dict[str, Any]]` | `None` | — |
| `keep_alive` | `OllamaChatRequest` | `Optional[Union[float, str]]` | `None` | — |
| `think` | `OllamaChatRequest` | `Optional[Union[bool, Literal['low', 'medium', 'high']]]` | `None` | — |

### `OllamaGenerateRequest`

定义于 `python/sglang/srt/entrypoints/ollama/protocol.py` 第 59 行，含继承字段共 13 项。

| 字段 | 声明类 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- | --- |
| `model` | `OllamaGenerateRequest` | `str` | `（必填）` | — |
| `prompt` | `OllamaGenerateRequest` | `str` | `（必填）` | — |
| `suffix` | `OllamaGenerateRequest` | `Optional[str]` | `None` | — |
| `system` | `OllamaGenerateRequest` | `Optional[str]` | `None` | — |
| `template` | `OllamaGenerateRequest` | `Optional[str]` | `None` | — |
| `context` | `OllamaGenerateRequest` | `Optional[List[int]]` | `None` | — |
| `stream` | `OllamaGenerateRequest` | `bool` | `True` | — |
| `raw` | `OllamaGenerateRequest` | `bool` | `False` | — |
| `format` | `OllamaGenerateRequest` | `Optional[Union[Literal['json'], Dict[str, Any]]]` | `None` | — |
| `options` | `OllamaGenerateRequest` | `Optional[Dict[str, Any]]` | `None` | — |
| `keep_alive` | `OllamaGenerateRequest` | `Optional[Union[float, str]]` | `None` | — |
| `images` | `OllamaGenerateRequest` | `Optional[List[str]]` | `None` | — |
| `think` | `OllamaGenerateRequest` | `Optional[bool]` | `None` | — |

上表字段条目合计 427。
