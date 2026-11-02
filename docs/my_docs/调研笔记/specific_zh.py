"""Field-specific Chinese explanations from each option's own help text.

The cell is built from that field's help, default, and choices. Namespace essays
are not copied into the cell. A glossary keeps technical names, then Chinese
function words rebuild the sentence. If too much English remains, the sentence
is dropped rather than shipped as broken mix.
"""

from __future__ import annotations

import re

# Longer phrases first.
PHRASES = [
    ("defaults to auto-detection if not specified", "不填则按硬件自动探测"),
    ("disabled by default", "默认关闭"),
    ("enabled by default", "默认打开"),
    ("if not specified", "如果不填"),
    ("if not set", "如果不设置"),
    ("if unset", "如果不填"),
    ("if omitted", "如果省略"),
    ("if set", "如果打开"),
    ("by default", "默认"),
    ("defaults to", "默认是"),
    ("default is", "默认是"),
    ("incompatible with", "不能和这些同时使用："),
    ("not supported", "不支持"),
    ("only supported", "仅支持"),
    ("is required", "是必需的"),
    ("are required", "是必需的"),
    ("must be", "必须是"),
    ("must equal", "必须等于"),
    ("requires", "前提是"),
    ("choose the kernels for", "选择用于"),
    ("choose the backend for", "选择用于"),
    ("choose the runner backend for", "选择用于"),
    ("choose the kernel backend for", "选择用于"),
    ("the maximum number of", "最大数量"),
    ("the minimum number of", "最小数量"),
    ("the number of", "数量"),
    ("number of", "数量"),
    ("maximum number of", "最大数量"),
    ("the path of the", ""),
    ("the path to", "路径"),
    ("path to the", "路径"),
    ("path to", "路径"),
    ("whether to", "是否"),
    ("whether the", "是否"),
    ("whether", "是否"),
    ("instead of", "而不是"),
    ("as well as", "以及"),
    ("fall back to", "回退到"),
    ("falls back to", "回退到"),
    ("for example", "例如"),
    ("such as", "例如"),
    ("greater than", "大于"),
    ("less than", "小于"),
    ("at least", "至少"),
    ("at most", "至多"),
    ("per request", "每个请求"),
    ("per-request", "按请求"),
    ("per token", "每个 token"),
    ("tensor parallel", "张量并行"),
    ("tensor-parallel", "张量并行"),
    ("data parallel", "数据并行"),
    ("data-parallel", "数据并行"),
    ("pipeline parallel", "流水并行"),
    ("context parallel", "上下文并行"),
    ("context-parallel", "上下文并行"),
    ("expert parallel", "专家并行"),
    ("speculative decoding", "投机解码"),
    ("speculative draft", "投机草稿"),
    ("kv cache", "KV 缓存"),
    ("kv-cache", "KV 缓存"),
    ("cuda graph", "CUDA 图"),
    ("radix cache", "前缀树缓存"),
    ("chunked prefill", "分块预填充"),
    ("hidden states", "隐状态"),
    ("hidden state", "隐状态"),
    ("log probs", "对数概率"),
    ("chat template", "对话模板"),
    ("tool call", "工具调用"),
    ("all-reduce", "all-reduce"),
    ("all reduce", "all-reduce"),
    ("all-gather", "all-gather"),
    ("all-to-all", "all-to-all"),
    ("in order to", "以便"),
    ("used for", "用于"),
    ("used to", "用来"),
    ("according to", "按照"),
    ("based on", "按照"),
    ("up to", "最多到"),
    ("opt-in", "需要显式打开"),
    ("opt in", "需要显式打开"),
]

WORDS = {
    "backend": "后端", "backends": "后端", "default": "默认", "defaults": "默认",
    "model": "模型", "models": "模型", "request": "请求", "requests": "请求",
    "cache": "缓存", "enable": "打开", "enabled": "已打开", "enables": "会打开",
    "disable": "关闭", "disabled": "已关闭", "disables": "会关闭",
    "prefill": "预填充", "decode": "解码", "decoding": "解码",
    "attention": "注意力", "number": "数量", "draft": "草稿", "size": "大小",
    "server": "服务", "group": "组", "groups": "组", "batch": "批", "batches": "批",
    "token": "token", "tokens": "token", "memory": "显存", "weights": "权重",
    "weight": "权重", "cuda": "CUDA", "used": "使用", "list": "列表", "file": "文件",
    "files": "文件", "json": "JSON", "mode": "模式", "each": "每个",
    "flashinfer": "FlashInfer", "moe": "MoE", "requires": "需要", "require": "需要",
    "whether": "是否", "name": "名字", "gpu": "GPU", "gpus": "GPU",
    "sglang": "SGLang", "speculative": "投机", "maximum": "最大", "uses": "使用",
    "graph": "图", "auto": "auto", "pool": "池", "pools": "池", "none": "空",
    "full": "完整", "layers": "层", "layer": "层", "parallelism": "并行",
    "specified": "被指定", "custom": "自定义", "supported": "受支持",
    "support": "支持", "supports": "支持", "return": "返回", "returning": "返回",
    "returned": "返回的", "kernel": "kernel", "kernels": "kernel",
    "instead": "改为", "after": "之后", "before": "之前", "http": "HTTP",
    "output": "输出", "outputs": "输出", "input": "输入", "inputs": "输入",
    "rank": "rank", "ranks": "rank", "storage": "存储", "load": "加载",
    "loading": "加载", "priority": "优先级", "expert": "专家", "experts": "专家",
    "lora": "LoRA", "image": "图像", "images": "图像", "logprobs": "对数概率",
    "prompt": "提示", "port": "端口", "tokenizer": "分词器", "value": "值",
    "values": "值", "quantization": "量化", "cpu": "CPU", "process": "进程",
    "runs": "运行", "run": "运行", "running": "运行中", "config": "配置",
    "configuration": "配置", "using": "使用", "key": "密钥", "keys": "密钥",
    "every": "每个", "data": "数据", "checkpoint": "检查点", "worker": "工作进程",
    "workers": "工作进程", "target": "目标", "back": "回", "multimodal": "多模态",
    "host": "主机", "torch": "torch", "format": "格式", "ids": "id",
    "sampling": "采样", "across": "跨越", "extra": "额外", "choose": "选择",
    "openai": "OpenAI", "shared": "共享", "during": "在", "triton": "Triton",
    "string": "字符串", "overrides": "覆盖", "override": "覆盖", "metrics": "指标",
    "chat": "对话", "first": "第一次", "unset": "未设置", "embedding": "向量",
    "embeddings": "向量", "mamba": "Mamba", "while": "同时", "same": "相同",
    "own": "自己的", "states": "状态", "state": "状态", "scheduler": "调度器",
    "optional": "可选", "has": "有", "under": "在", "forward": "前向",
    "skip": "跳过", "encoder": "编码器", "startup": "启动", "capture": "捕获",
    "captured": "捕获的", "parameters": "参数", "parameter": "参数",
    "external": "外部", "api": "API", "multiple": "多个", "disaggregation": "PD 分离",
    "without": "没有", "swa": "滑动窗口注意力", "otherwise": "否则", "text": "文本",
    "resolved": "解析得到", "resolves": "解析为", "inference": "推理",
    "sparse": "稀疏", "where": "当", "type": "类型", "available": "可用",
    "off": "关闭", "see": "见", "buckets": "分桶", "bucket": "分桶",
    "warmup": "预热", "template": "模板", "component": "组件", "device": "设备",
    "devices": "设备", "verify": "校验", "policy": "策略", "single": "单个",
    "pass": "一次", "passes": "次数", "place": "位置", "indexer": "indexer",
    "fp8": "FP8", "bf16": "BF16", "fp16": "FP16", "fp32": "FP32", "fp4": "FP4",
    "context": "上下文", "select": "选择", "selects": "会选择", "scheduling": "调度",
    "time": "时间", "log": "日志", "logging": "日志", "steps": "步",
    "window": "窗口", "local": "本地", "adapters": "适配器", "adapter": "适配器",
    "int": "整数", "base": "基", "keep": "保留", "keeps": "保留", "seconds": "秒",
    "mooncake": "Mooncake", "radix": "前缀树", "dspark": "DSPARK", "runtime": "运行时",
    "responses": "响应", "response": "响应", "prefix": "前缀", "nodes": "节点",
    "node": "节点", "options": "选项", "buffer": "缓冲区", "nccl": "NCCL",
    "parallel": "并行", "processor": "处理器", "distributed": "分布式",
    "instance": "实例", "grpc": "gRPC", "tensor": "张量", "reasoning": "思考过程",
    "reduce": "规约", "endpoint": "端点", "control": "控制", "url": "URL",
    "urls": "URL", "always": "总是", "currently": "目前", "nvfp4": "NVFP4",
    "mxfp4": "MXFP4", "mxfp8": "MXFP8", "runner": "执行器", "existing": "现有",
    "either": "要么", "fusion": "融合", "deepep": "DeepEP", "specify": "指定",
    "deprecated": "已废弃", "video": "视频", "audio": "音频", "safetensors": "safetensors",
    "specific": "具体", "pipeline": "流水线", "routing": "路由", "completion": "补全",
    "between": "之间", "exact": "精确", "behavior": "行为", "takes": "花费",
    "automatically": "自动", "engine": "引擎", "transfer": "传输", "layout": "布局",
    "page": "页", "dtype": "数据类型", "nixl": "NIXL", "avoid": "避免",
    "eagle": "EAGLE", "does": "会", "precision": "精度", "hidden": "隐层",
    "selected": "被选中的", "their": "其", "limit": "上限", "high": "高",
    "query": "查询", "sequence": "序列", "gemm": "GEMM", "based": "基于",
    "aiter": "Aiter", "implementation": "实现", "means": "表示", "allowed": "允许",
    "explicit": "显式", "explicitly": "显式地", "world": "全局", "least": "至少",
    "directory": "目录", "provided": "给出", "remote": "远程", "inputs": "输入",
    "native": "原生", "thinking": "思考", "positions": "位置", "documents": "文档",
    "example": "例如", "sampled": "采样得到的", "timeout": "超时", "longer": "更长",
    "systems": "系统", "serving": "服务", "width": "宽度", "small": "小",
    "interval": "间隔", "allow": "允许", "built": "内置", "manager": "管理器",
    "more": "更多", "dynamic": "动态", "performance": "性能", "fields": "字段",
    "last": "最后", "false": "false", "true": "true", "bounded": "有界",
    "fused": "融合的", "feature": "特性", "reads": "读取", "directly": "直接",
    "linear": "线性", "dsa": "DSA", "deepseek": "DeepSeek", "through": "经过",
    "still": "仍然", "communication": "通信", "scale": "规模", "threshold": "阈值",
    "elastic": "弹性", "slot": "槽", "slots": "槽", "many": "许多",
    "hierarchical": "分层", "python": "Python", "streaming": "流式",
    "huggingface": "Hugging Face", "daemon": "守护进程", "there": "那里",
    "assigns": "分配", "ignored": "被忽略", "ngram": "n-gram", "internal": "内部",
    "base64": "base64", "start": "起始", "random": "随机", "step": "步",
    "deterministic": "确定性", "usage": "用量", "numa": "NUMA", "decoupled": "解耦",
    "ordered": "按顺序", "role": "角色", "compute": "计算", "offloading": "换出",
    "omitted": "被省略", "overhead": "开销", "launch": "启动", "never": "从不",
    "range": "范围", "dense": "稠密", "head": "头", "main": "主", "settings": "设置",
    "equivalent": "等价于", "scoring": "打分", "chunked": "分块的",
    "unsupported": "不支持的", "additional": "额外", "cutlass": "CUTLASS",
    "deepgemm": "DeepGEMM", "rocm": "ROCm", "cutedsl": "CuTe DSL", "make": "使",
    "index": "索引", "mla": "MLA", "normalization": "归一化", "strategy": "策略",
    "cap": "上限", "remains": "仍然保留", "setting": "设置", "useful": "用于",
    "top-k": "top-k", "inside": "内部", "overlap": "重叠", "within": "之内",
    "dump": "转储", "reserve": "预留", "until": "直到", "option": "选项",
    "including": "包括", "redundant": "冗余", "eagle3": "EAGLE3", "tensors": "张量",
    "hicache": "HiCache", "client": "客户端", "have": "有", "unlike": "不同于",
    "separate": "分开的", "paged": "分页的", "required": "必需", "routed": "被路由的",
    "replay": "重放", "experimental": "实验性", "configurations": "配置",
    "match": "匹配", "force": "强制", "nvidia": "NVIDIA", "error": "错误",
    "operations": "运算", "blackwell": "Blackwell", "fallback": "回退",
    "except": "除了", "larger": "更大", "split": "切分", "read": "读取",
    "hybrid": "混合", "phase": "阶段", "legacy": "旧的", "profiling": "性能剖析",
    "normal": "normal", "algorithm": "算法", "pinned": "钉住的", "router": "路由器",
    "prefetch": "预取", "block": "块", "workloads": "负载", "schema": "schema",
    "max": "最大", "count": "计数", "accept": "接受", "budget": "预算",
    "ipc": "IPC", "pytorch": "PyTorch", "loader": "加载器", "version": "版本",
    "modelopt": "ModelOpt", "metadata": "元数据", "labels": "标签", "trace": "追踪",
    "generate": "生成", "encoded": "编码的", "hash": "哈希", "hint": "提示",
    "hints": "提示", "classification": "分类", "timeout": "超时", "seconds": "秒",
    "path": "路径", "paths": "路径", "server": "服务", "only": "仅", "when": "当",
    "with": "带有", "from": "来自", "into": "进入", "over": "超过", "this": "该",
    "that": "该", "one": "一个", "two": "两个", "both": "两者", "all": "全部",
    "any": "任何", "not": "不", "and": "并且", "or": "或", "for": "用于",
    "use": "使用", "set": "设置", "new": "新的", "old": "旧的", "per": "每",
    "via": "经由", "then": "然后", "also": "也", "may": "可能", "can": "可以",
    "will": "会", "should": "应当", "must": "必须", "its": "其", "their": "其",
    "which": "其中", "who": "谁", "on": "在", "in": "在", "to": "到", "of": "的",
    "a": "", "an": "", "the": "", "is": "是", "are": "是", "be": "是",
    "as": "作为", "at": "在", "by": "由", "if": "如果", "it": "它",
    "no": "无", "yes": "是", "do": "做", "does": "会", "done": "完成",
    "other": "其他", "others": "其他", "some": "一些", "such": "这样的",
    "than": "比", "too": "太", "very": "非常", "out": "外", "up": "上",
    "down": "下", "off": "关", "on": "开",
    "server": "服务",
}


def _protect(text: str):
    saved = []

    def repl(m):
        saved.append(m.group(0))
        return f"§{len(saved)-1}§"

    text = re.sub(r"--[A-Za-z0-9][A-Za-z0-9_-]*|`[^`]+`|\"[^\"]*\"|'[^']*'", repl, text)
    return text, saved


def _restore(text: str, saved: list[str]) -> str:
    for i, val in enumerate(saved):
        text = text.replace(f"§{i}§", val)
    return text


def translate(text: str) -> str:
    if not text:
        return ""
    raw = " ".join(text.split())
    protected, saved = _protect(raw)
    low = protected
    for src, dst in PHRASES:
        low = re.sub(re.escape(src), dst, low, flags=re.I)
    def word(m):
        w = m.group(0)
        key = w.lower()
        if key in WORDS:
            return WORDS[key]
        if w.isupper() or any(ch.isdigit() for ch in w):
            return w
        return w
    low = re.sub(r"[A-Za-z][A-Za-z0-9_+-]*", word, low)
    low = _restore(low, saved)
    low = re.sub(r"\s+", " ", low).strip()
    low = low.replace(" 的 的 ", "的").replace("是 是", "是")
    return low


def english_left(s: str) -> int:
    keep = {
        "cuda", "gpu", "cpu", "json", "http", "https", "fp8", "fp4", "fp16", "fp32",
        "bf16", "api", "lora", "moe", "nccl", "url", "ipc", "tcp", "dsa", "mla",
        "gemm", "sm100", "sm90", "sm103", "nixl", "eagle", "eagle3", "dspark",
        "triton", "torch", "pytorch", "grpc", "numa", "auto", "none", "null",
        "false", "true", "id", "ids", "topk", "top-k", "base64", "nvfp4", "mxfp4",
        "mxfp8", "rocm", "hip", "aiter", "deepgemm", "cutlass", "flashinfer",
        "deepep", "mooncake", "hicache", "openai", "sglang", "modelopt", "kda",
        "gdn", "mnnvl", "nvls", "mscclpp", "philox", "swa", "pd", "tp", "dp",
        "ep", "pp", "cp", "kv", "lm", "head", "logits", "safetensors", "gguf",
        "yaml", "zmq", "rdma", "nvlink", "blackwell", "hopper", "cute", "dsl",
        "int8", "int4", "w4a16", "w4a4", "q8", "q8kv8", "megamoe", "megakernel",
        "monokernel", "replayssm", "radix", "ngram", "dflash", "uno", "mlx",
        "ray", "sigquit", "adler", "ucx", "hca", "greenctx", "pdmux", "epd",
        "mllm", "vlm", "rl", "ttft", "p99", "rccl", "ffn", "rms", "norm",
        "splitk", "cublas", "cudnn", "marlin", "helion", "inkling", "mimo",
        "indexer", "token", "rank", "batch", "kernel", "schema", "prompt",
    }
    n = 0
    for w in re.findall(r"[A-Za-z]{4,}", s):
        if w.lower() not in keep and not w.startswith("--"):
            n += 1
    return n


def usable(zh: str) -> str:
    if not zh:
        return ""
    if english_left(zh) > 8:
        return ""
    return zh


def _default_sentence(default: str, kind: str, no_cli: bool) -> str:
    if kind == "derived" or default == "（发布时计算）":
        return "这是发布时算出来的派生量，没有常规命令行。读到的是运行时的结果，不是启动输入。"
    if no_cli:
        return f"声明默认是 `{default}`。它不作为日常 CLI 暴露，通常由别的旗标或解析阶段写入。"
    if default in ("None", "null"):
        return "声明默认是空。不填时走该字段 help 里写的自动路径；没有自动路径就保持关闭或不覆盖模型配置。"
    if default in ("False", "false"):
        return "声明默认是关闭。不写这个旗标就不会进入这条路径。"
    if default in ("True", "true"):
        return "声明默认是打开。要关掉需要显式传反向值。"
    if default in ("-1",):
        return "声明默认是 -1。对上限类字段，-1 通常表示不封顶或关闭这项限制，以 help 的句子为准。"
    if default in ("auto", "'auto'", '"auto"'):
        return "声明默认是 auto。解析阶段会按模型和平台选一个具体实现，启动日志里的覆盖才是运行值。"
    return f"声明默认是 `{default}`。不在命令行覆盖，就带着这个值进入解析；help 里写了会被改写的情况时，以改写后的值为准。"


def _interaction(help_en: str, zh: str, resolvable: bool) -> str:
    bits = []
    low = help_en.lower()
    if "incompatible" in low or "不能和" in zh:
        bits.append("help 写明了不能一起开的选项，解析阶段会拒绝或关掉冲突项。")
    if "requires" in low or "前提是" in zh or "必须" in zh:
        bits.append("help 写了前提。前提不满足时，失败发生在启动解析或第一次建图，而不是静默算错。")
    if resolvable:
        bits.append("该字段允许解析阶段改写，/server_info 里看到的是改写后的值。")
    if "deprecated" in low or "已废弃" in zh:
        bits.append("这是废弃入口，新配置应使用 help 指向的替代旗标。")
    if not bits:
        bits.append("解析钩子没有单独强制改写它；它按声明值被对应阶段读取。")
    return "".join(bits)


def _pitfall(help_en: str, default: str) -> str:
    low = help_en.lower()
    if "incompatible" in low:
        return "把 help 里点名的冲突开关一起打开，进程会在解析阶段退出。"
    if "experimental" in low:
        return "这是实验路径。生产配置不要在没看启动报错的情况下打开。"
    if "deprecated" in low:
        return "继续写旧旗标，行为可能被折进新字段，和你以为的副本数或图开关不一致。"
    if "security" in low:
        return "默认关闭是为了安全。打开就等于允许客户端把代码或处理器送进进程。"
    if default in ("None", "null") and "auto" in low:
        return "不填时的自动选择和显式指定不是同一条路径，排错时要看日志里最终选了谁。"
    if default in ("False", "false"):
        return "从别的模型的启动命令原样抄过来。这个开关默认关，本模型未必吃它。"
    return "传了 choices 之外的值会在 argparse 失败；单位（字节、秒、token）以 help 的说法为准。"


def explain_from_help(name: str, help_en: str, default: str, *, kind: str = "cli", no_cli: bool = False, resolvable: bool = False, choices: str = "") -> str:
    zh = usable(translate(help_en or ""))
    if zh:
        purpose = zh if zh.endswith(("。", "！", "？")) else zh + "。"
    elif help_en:
        purpose = "源码 help 说明了这个字段的行为；自动译述没有通过英文残留检查，请直接看声明处的 help 字符串，不要套用别的字段的解释。"
    else:
        purpose = "声明处没有 help 文本。它要么是派生量，要么只在代码里被读取。"
    if choices:
        purpose += f"可选值：{choices}。"
    return (
        f"作用：{purpose} "
        f"默认与效果：{_default_sentence(default, kind, no_cli)} "
        f"交互与改写：{_interaction(help_en or '', zh, resolvable)} "
        f"易错：{_pitfall(help_en or '', default)}"
    )
