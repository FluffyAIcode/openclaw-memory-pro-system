# OpenClaw Memory Pro System

**一句话：** 帮你把碎片化的想法、笔记、聊天记录自动整理成可搜索的长期记忆，并在合适的时候提醒你。

> An AI memory assistant that turns fragmented notes and conversations into searchable long-term memory, with automatic summarization, knowledge graph reasoning, and proactive reminders.

### 30 秒上手

```bash
pip install -e .                       # 安装
memory-cli server-start                # 启动（首次需等待 ~3 分钟加载模型）
memory-cli health                      # 确认就绪
memory-cli remember "今天学到了 X"      # 记住
memory-cli recall "X"                  # 回忆
memory-cli briefing                    # 今日简报
```

---

**目录：** [产品目标](#产品目标) · [当前已满足与差距](#当前已满足与差距) · [Architecture](#architecture) · [Quick Start](#quick-start)

## 产品目标

本仓库的**正式产品目标**如下（便于全文搜索：直接搜「产品目标」或下段关键词「碎片化」「非连续时间」）：

> 在当前信息环境下，支持**人**与 **Agent** 以**碎片化、发散化、非连续时间**的方式学习——断断续续、东一点西一点、往往没有明确的课题或课表；同时期望通过**长期时间积累**，逐步走向更**系统化**的理解、更**专业化**的深度，以及**可操作的**技能与规程。

**North star（英文对照）：** Support humans and agents in today’s information environment through **fragmented, divergent, non-continuous learning** (patchy inputs, no fixed curriculum), while aiming—via **sustained accumulation over time**—for **systematic** understanding, **domain depth**, and **actionable** skills and procedures.

各子系统（Memora、Chronos、MSA、Second Brain）历史上并非由同一份规格书一次写成，但**以上表述作为统一叙事**，用于对齐文档与路线图。

## Product vision

This section is the English mirror of **[产品目标](#产品目标)** above. The stack should feel like a personal/agent corpus that grows from noise toward usefulness.

## 当前已满足与差距

（英文小标题：**Vision alignment: satisfied vs. gaps** — 搜索「当前已满足」可定位本节。）

| Area | Status | Notes |
|------|--------|--------|
| **Fragmented ingest** | **Largely satisfied** | `remember`, daily `memory/*.md`, AutoIngestor, HTTP API; text-first (video/chat need export/transcription upstream). |
| **Persistence over time** | **Largely satisfied** | JSONL vector store, daily logs, bookmarks, ingestion state; dedup prevents runaway duplicates. |
| **Recall when needed** | **Largely satisfied** | Semantic search (`recall` / `search`), merged Memora+MSA, `deep-recall` for multi-hop; quality depends on corpus cleanliness and query. |
| **Light structure & “weaving”** | **Partial** | KG + relation extraction, digests, collision engine add edges and occasional insights—not guaranteed holistic “systemization.” |
| **Proactive nudges** | **Partial** | Scheduler + Telegram, collision on a timer; not fully **problem-** or **task-grounded**. |
| **Systematic + professional depth from accumulation alone** | **Not satisfied** | No automatic curriculum or domain mastery loop; depth comes from **how you use** recall + LLM, not from a finished “become expert” pipeline. |
| **Actionable skills (OpenClaw Skills, SOPs)** | **Not satisfied** | Memory store ≠ Skill registry; promoting a snippet to a **durable, invocable Skill** is still **manual / workflow-dependent** (e.g. `SKILL.md`). |
| **True parametric continual learning** | **Not satisfied** | Chronos tracks importance and personality artifacts; **EWC/LoRA paths are largely simulated**—see [STUBS.md](STUBS.md). |
| **“Learning quality” & intent (e.g. thought vs. share)** | **Not satisfied** | No first-class labels; optional metadata/tags only. |
| **Problem-triggered surfacing** | **Weak** | Strong when the user/agent issues a good query; **no** first-class “current task object” that always pulls the right slice of memory + skills. |

**Summary:** The system today is strongest as **durable capture + semantic retrieval + periodic inspiration and KG reasoning**. The **tail of the vision**—automatic professionalism and **automatic** skill crystallization from fragments—is **direction, not done**.

## Architecture

```
碎片输入 → [摄入 + 标签] → 统一语料库 (Memora 向量 + 可选长文 MSA)
                                    │
                          ┌─────────┼──────────┐
                          ↓         ↓          ↓
                      [KG 织网]  [蒸馏/总结]  [碰撞/联想]
                          │         │          │
                          └─────────┼──────────┘
                                    ↓
                             [技能注册表]
                            (Skill Registry)
                                    │
                          ┌─────────┼──────────┐
                          ↓         ↓          ↓
                     [问题驱动    [定时       [Nebius
                      召回+组装]   推送]       微调]
```

### Subsystem Roles

| Layer | Module | Role |
|-------|--------|------|
| **统一语料库** | **Memora** | Primary vector store (nomic-embed-text, JSONL). All content enters here. |
| | **MSA** | Optional document-level storage for long text (>500 chars). Sparse routing + multi-hop interleave. |
| **智能层** | **Second Brain** | KG weaving (relation extraction), distillation/summarization, collision/association. Reads from Memora/MSA, writes to KG + insights + digests. |
| **技能层** | **Skill Registry** | Versioned, lifecycle-managed skills (draft → active → deprecated). Bridge between memories and actionable capabilities. |
| **训练层** | **Chronos** | Training data export (distiller), replay buffer, personality profile generation, Nebius fine-tuning client. |

### Second Brain (Intelligence Layer)

Second Brain is the cognitive engine — it sits **above** the storage layer and provides value extraction:

1. **Digest / Distillation** — Periodic LLM-powered summarization of daily memories into long-term digests
2. **KG Weaving** — LLM-based relation extraction → Knowledge Graph (nodes: fact/decision/preference/goal/question; edges: supports/contradicts/extends/depends_on)
3. **Collision Engine** — 7-strategy inspiration generation (5 RAG-based + 2 KG-driven), adaptive weights
4. **Inference Engine** — Contradiction detection, absence reasoning, forward propagation, thread discovery
5. **Tracker** — Memory vitality scoring, dormancy detection, trend analysis
6. **Internalization** — High-maturity KG patterns → Chronos PERSONALITY.yaml

### Chronos (Training Pipeline)

Chronos has been refactored from simulated EWC/LoRA to a focused training pipeline:

1. **Encoder** — Structured memory encoding (facts, preferences, emotions, causal links)
2. **Replay Buffer** — Importance-weighted training candidate storage
3. **Distiller** — JSONL dataset generation from digests + buffer (chronos_lora_row_v1 format)
4. **Consolidator** — Periodic personality profile generation (PERSONALITY.yaml)
5. **Nebius Client** — Skeleton for cloud fine-tuning (upload → train → poll)

### Ingest Tags

Content is tagged at ingestion time to track cognitive intent:

| Tag | Meaning |
|-----|---------|
| `thought` | User's own thinking / analysis |
| `share` | Forwarded / curated content |
| `reference` | Factual reference material |
| `to_verify` | Unconfirmed, needs checking |

### Service Layer

| Component | Role |
|-----------|------|
| **Memory Hub** | Unified router — auto-selects subsystems based on content length, importance, and query type |
| **Memory Server** | Persistent HTTP daemon — keeps the embedding model loaded, exposes all operations as REST endpoints, async task queue for slow operations |
| **memory-cli** | Zero-dependency HTTP client — async polling with progress spinner for long-running tasks |
| **LLM Client** | Unified xAI Grok API wrapper — used by digest, interleave, consolidation, collision, and KG relation extraction |
| **Shared Embedder** | Singleton `nomic-ai/nomic-embed-text-v1.5` instance shared across Memora, MSA, and Second Brain's KG candidate pre-filtering |

## How It Works

### Ingestion Routing (Memory Hub)

```
Content arrives at hub.remember()
    │
    ├── < 100 words  ──────────────────────> Memora only
    ├── 100-500 words + high importance ───> Memora + Chronos
    └── > 500 words ───────────────────────> Memora + MSA + Chronos
    │
    ├── Always writes to memory/YYYY-MM-DD.md (daily log)
    └── Async: Second Brain KG relation extraction (if importance >= 0.4)
```

### Query Routing

- **`recall`** — Merged search across Memora (vector) and MSA (sparse routing), results deduplicated and ranked
- **`deep-recall`** — MSA multi-hop interleave: iteratively retrieves documents, generates intermediate answers via LLM, reformulates queries until the answer is sufficient
- **`search`** — Direct Memora vector similarity search

### Async Task Queue

Long-running operations (collide ~50s, deep-recall ~30s, consolidate ~20s) run asynchronously:

```
POST /second-brain/collide {"async": true}
  → 202 {"task_id": "abc123", "status": "running", "poll": "/task/abc123"}

GET /task/abc123
  → {"status": "running", "elapsed": 12.3}
  → {"status": "done", "elapsed": 48.7, "result": {...}}
```

The CLI handles this transparently with a polling spinner:

```
⠸ 灵感碰撞 ... 45s [running]
✓ 灵感碰撞 完成 (48s)
生成 3 条灵感
```

### Inspiration Collision (Second Brain)

Every 6 hours, the collision engine selects from **7 strategies** using adaptive weights (strategies that produce higher-rated insights get selected more often):

**RAG-based strategies:**
1. **Semantic Bridge** — Finds "distant relatives" (similarity 0.35–0.65) between memories
2. **Dormant Revival** — Pairs sleeping high-importance memories with recent active ones
3. **Temporal Echo** — Crosses today's memories with those from 7/30 days ago
4. **Chronos Cross-Reference** — Pairs Chronos structured memories with Memora entries
5. **Digest Bridge** — Crosses digest summaries with individual memories

**KG-driven strategies (fundamentally different from RAG):**
6. **Contradiction-Based** — Uses KG contradiction edges to surface decisions at risk, ranked by evidence balance
7. **Blind Spot-Based** — Uses KG absence reasoning to identify unexplored dimensions in important decisions

Each collision is evaluated by an LLM that scores novelty (1–5). Users can rate insights (`memory-cli rate <id> <1-5>`), and the system automatically adjusts strategy selection weights.

### Feedback Loop

```
Collision → Insight → User Rating → Strategy Weight Update → Better Collisions
```

Strategy weights persist in `memory/insights/strategy_weights.json`. Strategies with consistently high ratings get selected more often; low-rated strategies are de-prioritized.

## Quick Start

### 1. 安装

```bash
cd ~/.openclaw/workspace   # 或你的项目目录
pip install -e .           # 基础安装
pip install -e ".[embeddings]"  # 推荐：安装真实嵌入模型
```

**环境要求：** Python 3.9+, macOS (Mac mini M 系列) 或 Linux

### 2. 配置 (可选)

创建 `~/.openclaw/.env`：

```
XAI_API_KEY=your-xai-api-key-here
```

xAI Grok API 用于记忆总结、灵感碰撞、人格生成。**不配置也能用**，只是这些高级功能会降级为本地启发式方法。

### 3. 启动服务

```bash
memory-cli server-start    # 后台启动
memory-cli health          # 确认就绪（首次约 3 分钟加载嵌入模型）
```

### 4. 开始使用

```bash
memory-cli remember "今天学到了 Transformer 的多头注意力机制" --tag thought
memory-cli recall "注意力机制"
memory-cli briefing
memory-cli skills           # 查看技能
memory-cli --help           # 查看所有命令
```

The server loads the SentenceTransformer model once (~3s), then serves all requests with <100ms latency.

### Usage

```bash
# Remember something (auto-routes to appropriate subsystems + KG extraction)
memory-cli remember "用户偏好深色主题，不喜欢弹窗通知" -i 0.9

# Search memories
memory-cli recall "用户的UI偏好"

# Deep multi-hop reasoning
memory-cli deep-recall "根据用户过去的反馈，他们最可能接受什么样的新功能？"

# Second Brain — collision + KG reasoning
memory-cli collide                 # Run inspiration collision (7 strategies, adaptive weights)
memory-cli contradictions          # KG: find decisions with conflicting evidence
memory-cli blindspots              # KG: detect unexplored dimensions in decisions
memory-cli threads                 # KG: discover thought threads via community detection
memory-cli graph-status            # KG statistics
memory-cli rate <insight_id> <1-5> # Rate an insight to improve future collisions
memory-cli insight-stats           # View strategy performance

# System status
memory-cli status
memory-cli sb-report
memory-cli briefing                # Daily memory briefing
```

## Project Structure

```
├── memora/                  # 统一语料库 — RAG 向量存储
│   ├── collector.py         # Raw memory ingestion
│   ├── vectorstore.py       # JSONL-based vector store (nomic-embed-text)
│   ├── embedder.py          # Embedding with search_document/search_query prefixes
│   └── bridge.py            # Cross-system integration
├── msa/                     # 统一语料库 — 长文档存储 (Memory Sparse Attention)
│   ├── encoder.py           # Document chunking + routing key generation
│   ├── memory_bank.py       # Tiered storage (RAM keys, disk content)
│   ├── router.py            # Batch cosine similarity top-k selection
│   ├── interleave.py        # Multi-hop retrieval-generation loops
│   └── bridge.py            # Integration with cross-indexing to Memora
├── second_brain/            # 智能层 — KG 织网 / 蒸馏总结 / 碰撞联想
│   ├── digest.py            # LLM-powered daily summarization (moved from memora/)
│   ├── tracker.py           # Vitality scoring, dormancy detection, trends
│   ├── collision.py         # 7-strategy inspiration engine (5 RAG + 2 KG-driven)
│   ├── strategy_weights.py  # Adaptive collision strategy selection via ratings
│   ├── knowledge_graph.py   # KGNode/KGEdge + NetworkX DiGraph (insight storage)
│   ├── relation_extractor.py # LLM-based relation extraction + embedding pre-filter
│   ├── inference.py         # Contradiction detection, absence reasoning, propagation
│   ├── internalization.py   # Maturity tracking → Chronos PERSONALITY.yaml
│   └── bridge.py            # Unified interface
├── skill_registry/          # 技能层 — 从记忆到可操作技能
│   └── registry.py          # JSONL-backed skill store (draft → active → deprecated)
├── chronos/                 # 训练层 — Nebius 微调管线
│   ├── encoder.py           # Importance-weighted memory encoding
│   ├── replay_buffer.py     # Training candidate buffer
│   ├── distiller.py         # JSONL training dataset generation (moved from memora/)
│   ├── nebius_client.py     # Cloud fine-tuning client skeleton
│   ├── consolidator.py      # Personality profile generation (PERSONALITY.yaml)
│   └── bridge.py            # OpenClaw integration
├── memory_server.py         # HTTP daemon with async task queue
├── memory_hub.py            # Unified ingestion/query router (with tag support)
├── memory_cli.py            # Thin HTTP client with async polling
├── llm_client.py            # xAI Grok API wrapper
├── shared_embedder.py       # Singleton embedding model
├── setup.py                 # Package configuration
├── tests/                   # 474 unit tests across 17 files
│   ├── test_memora.py
│   ├── test_chronos.py       # distiller, nebius_client tests included
│   ├── test_msa.py
│   ├── test_second_brain.py
│   ├── test_skill_registry.py # NEW
│   ├── test_knowledge_graph.py
│   ├── test_inference.py
│   ├── test_relation_extractor.py
│   ├── test_internalization.py
│   ├── test_strategy_weights.py
│   ├── test_memory_server.py
│   ├── test_memory_hub.py
│   └── ...
├── skills/                  # OpenClaw skill manifests
│   ├── memora/SKILL.md
│   ├── chronos/SKILL.md
│   ├── msa/SKILL.md
│   └── second-brain/SKILL.md
├── AGENTS.md                # Agent behavior configuration
├── HEARTBEAT.md             # Periodic task definitions
└── STUBS.md                 # Documentation of all placeholder code
```

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -q

# With coverage
python3 -m pytest tests/ --cov=memora --cov=chronos --cov=msa --cov=second_brain --cov=skill_registry --cov=memory_server --cov=memory_hub -q
```

474 tests across 17 files, covering all subsystems (Memora, MSA, Second Brain KG/inference, Chronos training pipeline, Skill Registry), the server, the hub, and the CLI layer.

## Embedding Model

Uses [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (768-dim, Matryoshka support). The model is loaded once by the Memory Server and shared across Memora and MSA via the `shared_embedder` singleton. When the model is unavailable, a deterministic hash-based `MockEmbedder` serves as fallback (no semantic understanding).

For offline environments, pre-download the model and set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Known Limitations

See [STUBS.md](STUBS.md) for a full catalog of stubbed/simulated components. Key items:

- **Chronos Nebius Client**: Skeleton only — `upload_dataset`, `create_job`, `poll_job` raise `NotImplementedError`. See [NEBIUS_FINETUNE_INTEGRATION_SKETCH.md](docs/NEBIUS_FINETUNE_INTEGRATION_SKETCH.md) for the integration plan.
- **Chronos EWC/LoRA**: `ewc.py` and `dynamic_lora.py` still exist on disk but are **no longer imported** by any production code (deprecated since v0.0.3-beta).
- **Skill Registry**: Functional but no auto-promotion pipeline yet — skills must be manually added via `/skills/add`.
- **Daemon + MPS**: The daemon process forces CPU mode because macOS Metal services are unavailable after `setsid()`. Foreground mode uses MPS (Apple GPU) normally

## License

Private repository. All rights reserved.
