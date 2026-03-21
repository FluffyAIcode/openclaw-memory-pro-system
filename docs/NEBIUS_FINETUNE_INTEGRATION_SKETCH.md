# Chronos + Distiller → Nebius 微调：最小 JSONL 规范与集成草图

本文档与现有代码对齐：

- **`memora/distiller.py`**：`LoRADistiller.prepare_dataset()` 产出 `instruction` / `input` / `output` 三字段 JSONL（见 `memory/long_term/lora_training/dataset_*.jsonl`）。
- **`chronos/replay_buffer.py`**：持久化 **`EncodedMemory`**（`chronos/encoder.py`）为 JSONL，字段为 `facts`, `preferences`, `emotions`, `causal_links`, `importance`, `timestamp`, `raw_text`。

**Nebius 侧 API 路径、鉴权头、请求体字段名以 [Nebius 官方文档](https://docs.nebius.com/)（Token Factory / AI Studio / Serverless fine-tuning）为准；下文不写死 URL。**

---

## 1. 统一训练行：最小 JSONL 规范（`chronos_lora_row_v1`）

对外给云平台（或 Axolotl）的**每一行**建议统一为下列 **superset**，云平台不认识的字段可忽略或放进 `metadata`。

### 1.1 必填（与当前 `distiller` 兼容）

| 字段 | 类型 | 说明 |
|------|------|------|
| `instruction` | string | 任务说明；可与 distiller 一致，如「基于以下记忆上下文回答问题」。 |
| `input` | string | 上下文：摘要来源名、Chronos 拼接后的结构化文本、或空字符串。 |
| `output` | string | 模型应学习的回复/摘要正文；建议 ≤ 4k tokens，过长需截断或分条。 |

### 1.2 推荐（溯源与过滤）

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | string | `memora_digest` \| `chronos_replay` \| `memora_vector` 等。 |
| `source_id` | string | 如 digest 文件名 stem、或 replay 行 hash / timestamp。 |
| `importance` | number | 0.0–1.0，来自 `EncodedMemory.importance`，供云端或后处理加权采样。 |
| `timestamp` | string | ISO8601，与 Chronos 一致。 |

### 1.3 可选（Chronos 结构化字段展开）

若从 **`EncodedMemory`** 映射，可把结构化部分拼进 `input` 或单独保留（部分训练框架支持 `messages`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `facts` | string[] | 来自 `EncodedMemory.facts`。 |
| `preferences` | string[] | 同上。 |
| `emotions` | string[] | 同上。 |
| `causal_links` | string[] | 同上。 |
| `raw_text` | string | 原始记忆全文；**注意隐私**，上传前做脱敏策略。 |

### 1.4 示例行（单条 JSONL）

```json
{
  "instruction": "你是用户的长期记忆助手。根据下列结构化记忆片段，用简洁中文总结用户立场与可执行建议。",
  "input": "[facts] 用户倾向使用 JSONL 做小规模向量存储\n[preferences] 偏好无外部依赖的冷启动方案\n[importance] 0.82\n[ts] 2026-03-21T12:00:00",
  "output": "用户在向量库选型上偏好轻量、可自托管方案；小规模场景可优先 JSONL，并关注冷启动与依赖成本。",
  "source": "chronos_replay",
  "source_id": "2026-03-21T12:00:00_a1b2c3d4",
  "importance": 0.82,
  "timestamp": "2026-03-21T12:00:00+00:00"
}
```

### 1.5 与现有代码的映射关系

| 来源 | 映射方式 |
|------|-----------|
| **Distiller** `prepare_dataset()` | 已符合 `instruction` / `input` / `output`；可追加 `source="memora_digest"`, `source_id=digest_file.stem`, `importance=0.7`（常量或从文件名规则推断）。 |
| **ReplayBuffer** `EncodedMemory` | `output`：由规则或 LLM 将 `raw_text` + 结构化字段 **蒸馏** 成一条「助手应复述/应遵守」的文本（当前仓库无此步，需新增 `chronos/export_training.py` 或扩展 consolidator）。`input`：拼接 `facts/preferences/...` 模板字符串。 |

---

## 2. 集成草图：导出 → 上传 → 创建 Nebius 训练任务（伪代码）

以下使用占位符；**真实 endpoint、payload、轮询字段名以 Nebius 文档为准**。

```python
# nebius_finetune_sketch.py  —  伪代码，非可运行生产代码

import hashlib
import json
import os
import time
from pathlib import Path

# --- 配置占位：从环境变量读取，具体名字见 Nebius 控制台 / 文档 ---
NEBIUS_API_BASE = os.environ.get("NEBIUS_API_BASE", "")  # e.g. Token Factory / Studio base URL
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")

DISTILLER_OUTPUT = Path("memory/long_term/lora_training")  # memora.config.long_term_dir / lora_training
CHRONOS_BUFFER = Path("memory/chronos/replay_buffer.jsonl")  # 默认见 chronos/config.py：`base_dir / replay_buffer.jsonl`


def encoded_memory_to_row(m: dict) -> dict:
    """将 replay JSONL 中的一行（EncodedMemory dict）转为 chronos_lora_row_v1。"""
    ts = m.get("timestamp", "")
    sid = hashlib.sha256(f"{ts}:{m.get('raw_text','')[:200]}".encode()).hexdigest()[:12]
    structured = []
    for key in ("facts", "preferences", "emotions", "causal_links"):
        for item in m.get(key) or []:
            structured.append(f"[{key}] {item}")
    inp = "\n".join(structured) if structured else (m.get("raw_text") or "")[:1500]
    # output 应由 LLM 蒸馏生成；此处占位为「复述+约束」
    out = (m.get("raw_text") or "")[:2000]
    return {
        "instruction": "根据结构化记忆，生成一句可注入系统提示的用户偏好摘要。",
        "input": inp,
        "output": out,
        "source": "chronos_replay",
        "source_id": sid,
        "importance": m.get("importance", 0.5),
        "timestamp": ts,
    }


def merge_dataset(distiller_jsonl: Path, chronos_buffer_jsonl: Path, out_path: Path) -> Path:
    """合并 distiller 产出与 chronos buffer 导出为单一 JSONL。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if distiller_jsonl.exists():
        for line in distiller_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row.setdefault("source", "memora_digest")
                rows.append(row)
    if chronos_buffer_jsonl.exists():
        for line in chronos_buffer_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                rows.append(encoded_memory_to_row(m))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def nebius_upload_dataset(local_file: Path) -> str:
    """
    上传 JSONL，返回 dataset_id / uri。
    实现方式可能是：multipart upload、预签名 URL、或 CLI。
    详见 Nebius 文档「dataset upload」。
    """
    # import httpx
    # headers = {"Authorization": f"Bearer {NEBIUS_API_KEY}"}
    # resp = httpx.post(f"{NEBIUS_API_BASE}/.../datasets", headers=headers, files={...})
    # return resp.json()["dataset_id"]
    raise NotImplementedError("See Nebius docs: dataset upload API")


def nebius_create_finetune_job(
    dataset_id: str,
    base_model_id: str,
    *,
    lora_r: int = 16,
    lora_alpha: int = 16,
) -> str:
    """
    创建微调任务，返回 job_id。
    请求体通常包含：base_model、dataset_id、method=lora、超参、回调 webhook 等。
    详见 Nebius 文档「fine-tuning job」或 Token Factory training API。
    """
    # payload = {
    #     "base_model": base_model_id,
    #     "dataset_id": dataset_id,
    #     "training_type": "lora",
    #     "hyperparameters": {"lora_r": lora_r, "lora_alpha": lora_alpha, ...},
    # }
    # resp = httpx.post(f"{NEBIUS_API_BASE}/.../fine-tunes", json=payload, headers={...})
    # return resp.json()["job_id"]
    raise NotImplementedError("See Nebius docs: create fine-tune job")


def nebius_poll_job(job_id: str) -> dict:
    """轮询直到 succeeded / failed；返回含 adapter_uri 或 endpoint 的 dict。"""
    # while True:
    #     st = httpx.get(f"{NEBIUS_API_BASE}/.../jobs/{job_id}", headers={...}).json()
    #     if st["status"] in ("succeeded", "failed"):
    #         return st
    #     time.sleep(30)
    raise NotImplementedError("See Nebius docs: job status API")


def run_pipeline():
    merged = merge_dataset(
        DISTILLER_OUTPUT / "dataset_latest_placeholder.jsonl",  # 实际用 distiller 生成的文件名
        CHRONOS_BUFFER,
        DISTILLER_OUTPUT / "merged_for_nebius.jsonl",
    )
    dataset_id = nebius_upload_dataset(merged)
    job_id = nebius_create_finetune_job(
        dataset_id,
        base_model_id=os.environ.get("NEBIUS_BASE_MODEL", "meta-llama-3.1-8b-instruct"),
    )
    result = nebius_poll_job(job_id)
    # 将 result 中的 endpoint 或本地 adapter 路径写入 OpenClaw / Memory Server 配置 — 产品决策
    return result


if __name__ == "__main__":
    run_pipeline()
```

---

## 3. 与 `memora/distiller.py` 的衔接建议（最小改动）

1. **`prepare_dataset()`**  
   - 保持 `instruction` / `input` / `output`。  
   - 每行追加 `source`, `source_id`, `importance`（可选 `timestamp`）。

2. **`start_training()`**  
   - 将 `TODO: 对接 Axolotl / Unsloth` 旁路为：可选调用 `nebius_create_finetune_job`（或通过 subprocess 调用 Nebius CLI，若文档提供）。  
   - 失败时仍只写日志，不阻塞 Memora 主路径。

3. **Chronos**  
   - 新增薄模块 `chronos/training_export.py`：`export_jsonl_for_nebius(buffer_path, out_path)`，内部调用 `encoded_memory_to_row` 的实装版本（**建议 output 用 LLM 生成**，避免 `output == raw_text` 导致过拟合复述）。

---

## 4. 合规与安全（必读）

- 训练集可能含隐私：**上传前**做脱敏、最小必要字段、可选人工审核批次。  
- 基座模型与许可证：与 Nebius 可选模型列表一致；与 OpenClaw 当前对话模型（如 Grok）**无权重级连续关系**，属**侧车个性化模型**架构。

---

## 5. 参考链接（以官网为准）

- Nebius 文档根站：<https://docs.nebius.com/>  
- 站内搜索：`fine-tuning`、`Token Factory`、`LoRA`、`dataset`。

（具体路径随产品更名可能变化，集成时以当前文档为准。）
