#!/usr/bin/env python3
"""Render 05/06 from per-field Chinese stored in zh_parts/*.json.

Every key must exist. Missing keys abort the build so a namespace template
cannot silently fill a cell.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path("/workspace")
PARTS = ROOT / "docs/my_docs/调研笔记/zh_parts"
OUT = ROOT / "docs/my_docs"

spec = importlib.util.spec_from_file_location("catalog", "/tmp/gen_sglang_catalogs.py")
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)

NS_TITLE = {
    "model": "模型与加载",
    "parallel": "并行（TP / PP / DP / EP / CP / DCP）",
    "memory": "显存、KV 与前缀缓存",
    "schedule": "调度",
    "serving": "HTTP 服务、解析器与 API",
    "spec": "投机解码（MTP / EAGLE / DSPARK）",
    "disagg": "Prefill/Decode 分离",
    "lora": "LoRA",
    "mm": "多模态",
    "device": "设备与进程启动",
    "observability": "可观测性",
    "exec.features": "执行特性开关",
    "exec.kernel": "Kernel、Attention、GEMM",
    "exec.mamba": "Mamba / 线性注意力",
    "exec.graph": "CUDA Graph",
    "exec.comm": "集合通信",
    "exec.moe": "MoE 与 DeepEP / A2A",
    "exec.overlap": "Batch overlap",
    "exec.offload": "权重 offload",
    "exec.dllm": "扩散语言模型（dLLM）",
    "exec.deterministic": "确定性推理",
}


def load_zh() -> dict[str, str]:
    out = {}
    for path in sorted(PARTS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        overlap = set(data) & set(out)
        if overlap:
            raise SystemExit(f"duplicate keys in {path.name}: {sorted(overlap)[:5]}")
        out.update(data)
    return out


def md(text: str) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def require(zh: dict, key: str) -> str:
    if key not in zh:
        raise SystemExit(f"missing Chinese explanation for {key}")
    text = zh[key]
    for part in ("作用：", "默认与效果：", "交互与改写：", "易错："):
        if part not in text:
            raise SystemExit(f"{key} missing {part}")
    return text


def render_llm(zh: dict) -> str:
    fields = catalog.collect_llm_fields()
    groups: dict[str, list] = {}
    for f in fields:
        groups.setdefault(f["ns"], []).append(f)
    order = list(NS_TITLE) + [k for k in groups if k not in NS_TITLE]
    lines = [
        "# 05 Server 选项大全",
        "",
        "每一行的「含义」只讲这个字段：它做什么、声明默认会怎样、解析阶段会不会改它、用错时会怎样。",
        "材料来自 `arg_groups/fields/` 的 help，以及 `*_hook.py` / `validation_hook.py` 里点名该字段的断言或改写。",
        "声明默认不是运行值。S2 `run_resolution_pipeline` 之后才是最终配置。",
        "",
        f"LLM 字段 {len(fields)} 项。Diffusion 的 dataclass 在文末，不进这套解析。",
        "",
        "## 不在 Struct 上、由 add_cli_args 另外注册的旗标",
        "",
        "| CLI | 含义 |",
        "| --- | --- |",
    ]
    manual = [
        ("--config", zh.get("cli_config")),
        ("--sampling-backend", zh.get("sampling_backend")),
        ("--reasoning-parser", zh.get("reasoning_parser")),
        ("--tool-call-parser", zh.get("tool_call_parser")),
        ("--disable-cuda-graph", zh.get("disable_cuda_graph")),
        ("--enable-dp-attention", zh.get("enable_dp_attention")),
    ]
    for cli, text in manual:
        if not text:
            raise SystemExit(cli)
        lines.append(f"| `{cli}` | {md(text)} |")
    lines += ["", "## LLM ServerArgs", ""]
    for ns in order:
        items = groups.get(ns) or []
        if not items:
            continue
        lines.append(f"### {NS_TITLE.get(ns, ns)}")
        lines.append("")
        lines.append(
            f"命名空间 `{ns}`，类 `{items[0]['class']}`，`{items[0]['file']}`，{len(items)} 项。"
        )
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
                    cli=md(flag),
                    typ=md(f["type"]),
                    default=md(f["default"]),
                    help=md(require(zh, f["name"])),
                )
            )
        lines.append("")
    return "\n".join(lines)


def render_diffusion(zh: dict) -> str:
    fields = catalog.collect_diffusion_fields()
    lines = [
        "## Diffusion ServerArgs",
        "",
        "这些字段属于 `multimodal_gen` 的 dataclass，不进入 LLM 的 `run_resolution_pipeline`。",
        f"共 {len(fields)} 项。",
        "",
        "| 字段 | 声明类 | 默认 | 含义 |",
        "| --- | --- | --- | --- |",
    ]
    for f in fields:
        key = "diff:" + f["name"]
        lines.append(
            "| `{name}` | `{owner}`:{lineno} | `{default}` | {help} |".format(
                name=f["name"],
                owner=f["owner"],
                lineno=f["lineno"],
                default=md(f["default"]),
                help=md(require(zh, key)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_client(zh: dict) -> str:
    lines = [
        "# 06 Client 请求选项大全",
        "",
        "每一行只讲这个请求字段。公开 JSON 在 HTTP 层被接住，再抄进 `GenerateReqInput` 或 `SamplingParams`。调度器最终看见的是 `Req`，不是这些类本身。",
        "内部字段（下划线或 PrivateAttr）不列入本表。",
        "",
    ]
    total = 0
    for title, rel, names in catalog.CLIENT_SPECS:
        path = ROOT / rel
        classes = catalog.classes_of(path)[2]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"文件 `{rel}`。")
        lines.append("")
        for name in names:
            cls = classes.get(name)
            lines.append(f"### `{name}`")
            lines.append("")
            if cls is None:
                lines.append("源码中未找到该类。")
                lines.append("")
                continue
            fields = [
                f
                for f in catalog.collect_class_fields(path, cls)
                if not f["internal"]
            ]
            total += len(fields)
            lines.append(f"定义于第 {cls.lineno} 行，公开字段 {len(fields)} 项。")
            lines.append("")
            lines.append("| 字段 | 声明类 | 默认 | 含义 |")
            lines.append("| --- | --- | --- | --- |")
            for f in fields:
                key = f"{name}.{f['name']}"
                if key not in zh and f"field:{f['name']}" in zh:
                    prefix = f"在 `{name}` 里，`{f['name']}` "
                    raw = zh[f"field:{f['name']}"]
                    if raw.startswith("作用：这个字段是"):
                        text = raw.replace("作用：这个字段是", f"作用：{prefix}是", 1)
                    else:
                        text = raw.replace("作用：这个字段", f"作用：{prefix}是", 1)
                else:
                    text = require(zh, key)
                lines.append(
                    "| `{fname}` | `{owner}` | `{default}` | {help} |".format(
                        fname=f["name"],
                        owner=f["owner"],
                        default=md(f["default"]),
                        help=md(text),
                    )
                )
            lines.append("")
    lines.append(f"公开字段合计 {total}。")
    lines.append("")
    return "\n".join(lines)


def main():
    zh = load_zh()
    text = render_llm(zh) + "\n" + render_diffusion(zh)
    (OUT / "05-Server选项大全.md").write_text(text, encoding="utf-8")
    (OUT / "06-Client请求选项大全.md").write_text(render_client(zh), encoding="utf-8")
    print("05", (OUT / "05-Server选项大全.md").stat().st_size)
    print("06", (OUT / "06-Client请求选项大全.md").stat().st_size)


if __name__ == "__main__":
    main()
