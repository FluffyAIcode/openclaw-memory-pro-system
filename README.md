# OpenClaw Memory Pro System

A production-grade cognitive architecture for the [OpenClaw](https://github.com/nicholasgasior/openclaw) AI agent framework. Four interconnected subsystems turn passive conversation logging into semantic retrieval, continual learning, document-level reasoning, and autonomous inspiration generation — with a **Knowledge Graph** inside Second Brain providing structured logical reasoning that RAG alone cannot achieve.

**目录：** [产品目标](#产品目标) · [当前已满足与差距](#当前已满足与差距) · [Architecture](#architecture) · [Quick Start](#quick-start)

## 产品目标

本仓库的**正式产品目标**如下（便于全文搜索：直接搜「产品目标」或下段关键词「碎片化」「非连续时间」）：

> 在当前信息环境下，支持**人**与 **Agent** 以**碎片化、发散化、非连续时间**的方式学习——断断续续、东一点西一点、往往没有明确的课题或课表；同时期望通过**长期时间积累**，逐步实现**系统化**的理解、更**专业化**的深度，以及**可操作的**技能的目标。

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
┌─────────────────────────────────────────────────────────────────┐
│                        memory-cli (HTTP)                        │
│               thin client with async polling + spinner          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP :18790
┌────────────────────────────▼────────────────────────────────────┐
│                      Memory Server (daemon)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Shared Embedder (SentenceTransformer, loaded once)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Async Task Manager (ThreadPool, 3 workers)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Memory Hub (router)                   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                  │   │
│  │  │  Memora   │ │ Chronos  │ │   MSA    │                  │   │
│  │  │  (RAG)   │ │  (CL)    │ │ (Sparse) │                  │   │
│  │  └──────────┘ └──────────┘ └──────────┘                  │   │
│  │  ┌───────────────────────────────────────────────────┐    │   │
│  │  │                  Second Brain                      │    │   │
│  │  │  ┌─────────────────────────────────────────────┐  │    │   │
│  │  │  │  Tracker (vitality, dormancy, trends)        │  │    │   │
│  │  │  └─────────────────────────────────────────────┘  │    │   │
│  │  │  ┌─────────────────────────────────────────────┐  │    │   │
│  │  │  │  Collision Engine (7 strategies, adaptive)   │  │    │   │
│  │  │  │  5 RAG-based + 2 KG-driven strategies       │  │    │   │
│  │  │  └──────────────────────┬──────────────────────┘  │    │   │
│  │  │                         │ read/write               │    │   │
│  │  │  ┌──────────────────────▼──────────────────────┐  │    │   │
│  │  │  │  Knowledge Graph (NetworkX DiGraph)          │  │    │   │
│  │  │  │  Nodes: fact/decision/preference/goal/question│  │    │   │
│  │  │  │  Edges: supports/contradicts/extends/...     │  │    │   │
│  │  │  │  Storage: memory/kg/nodes.jsonl + edges.jsonl│  │    │   │
│  │  │  └──────────────────────┬──────────────────────┘  │    │   │
│  │  │                         │ inference                │    │   │
│  │  │  ┌──────────────────────▼──────────────────────┐  │    │   │
│  │  │  │  Inference Engine                            │  │    │   │
│  │  │  │  → Contradiction Detection                   │  │    │   │
│  │  │  │  → Absence Reasoning (Blind Spots)           │  │    │   │
│  │  │  │  → Forward Propagation                       │  │    │   │
│  │  │  │  → Thread Discovery (Community Detection)    │  │    │   │
│  │  │  └──────────────────────┬──────────────────────┘  │    │   │
│  │  │                         │ mature patterns          │    │   │
│  │  │  ┌──────────────────────▼──────────────────────┐  │    │   │
│  │  │  │  Internalization → Chronos PERSONALITY.yaml  │  │    │   │
│  │  │  └─────────────────────────────────────────────┘  │    │   │
│  │  └───────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Four Memory Subsystems

| System | Purpose | Storage | Key Capability |
|--------|---------|---------|----------------|
| **Memora** | Snippet-level semantic search | Vector store (JSONL) | RAG with real embeddings, LLM-powered daily digests |
| **Chronos** | Continual learning | Replay buffer + EWC state | Importance-weighted memory encoding, personality profile generation |
| **MSA** | Document-level reasoning | Tiered (routing keys in RAM, content on disk) | Sparse attention routing, multi-hop interleave with LLM |
| **Second Brain** | Cognitive engine: tracking + collision + KG reasoning | Access log + insights JSONL + KG JSONL | Vitality tracking, 7-strategy inspiration collision, Knowledge Graph inference, internalization pipeline |

### Second Brain Internal Architecture

Second Brain is more than a memory tracker — it is the cognitive engine of the system. Its internal layers:

1. **Tracker** — Memory vitality scoring, dormancy detection, trend analysis
2. **Collision Engine** — 7-strategy inspiration generation (5 RAG-based + 2 KG-driven), with adaptive strategy weights
3. **Knowledge Graph** — Structured storage for insights and knowledge relationships (JSONL persistence via NetworkX DiGraph). Nodes represent knowledge units (facts, decisions, preferences, goals, questions); edges represent logical relationships (supports, contradicts, extends, depends_on, alternative_to, addresses)
4. **Inference Engine** — Operates on the KG to provide reasoning that RAG cannot:
   - **Contradiction Detection**: scans `contradicts` edges to find decisions with conflicting evidence
   - **Absence Reasoning**: generates expected dimensions for a decision, identifies which are missing (blind spots)
   - **Forward Propagation**: traces impact of new facts through `depends_on`/`supports` chains
   - **Thread Discovery**: community detection to auto-discover related knowledge clusters
5. **Internalization Manager** — Extracts high-maturity KG patterns and feeds them into Chronos for PERSONALITY.yaml generation (explicit KG knowledge → implicit agent behavior)

**Data flow within Second Brain:**

```
New memory ──→ Relation Extractor (LLM) ──→ KG nodes/edges (JSONL)
                                                    │
Collision Engine ←── reads KG for strategies 6 & 7 ─┘
       │
       ├── writes insights (JSONL)
       └── user rates insight → strategy weight update
                                                    │
Inference Engine ←── reads KG ──────────────────────┘
       │
       └── contradictions / blind spots / propagation alerts
                                                    │
Internalization Manager ←── reads mature KG nodes ──┘
       │
       └── patterns → Chronos consolidator → PERSONALITY.yaml
```

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

### Prerequisites

- Python 3.9+
- macOS (tested on Mac mini M-series) or Linux

### Installation

```bash
cd /path/to/openclaw-memory-pro-system

# Core install
pip install -e .

# With embedding model support (recommended)
pip install -e ".[embeddings]"

# Full install (includes torch, streamlit)
pip install -e ".[full]"
```

### Environment

Create `~/.openclaw/.env`:

```
XAI_API_KEY=your-xai-api-key-here
```

The LLM client (xAI Grok) powers digest summarization, multi-hop reasoning, personality generation, and inspiration collision. Without it, these features gracefully degrade to heuristic fallbacks.

### Start the Memory Server

```bash
# Daemon mode (recommended)
python3 memory_server.py --daemon

# Or foreground
python3 memory_server.py
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
├── memora/                  # RAG memory system
│   ├── collector.py         # Raw memory ingestion
│   ├── vectorstore.py       # JSONL-based vector store
│   ├── embedder.py          # Embedding with shared model fallback
│   ├── digest.py            # LLM-powered daily summarization
│   ├── distiller.py         # LoRA dataset preparation (training placeholder)
│   └── bridge.py            # Cross-system integration
├── chronos/                 # Continual learning system
│   ├── encoder.py           # Importance-weighted memory encoding
│   ├── replay_buffer.py     # Experience replay storage
│   ├── ewc.py               # Elastic Weight Consolidation
│   ├── dynamic_lora.py      # Virtual LoRA adapter management
│   ├── consolidator.py      # Periodic consolidation + personality generation
│   └── bridge.py            # OpenClaw integration
├── msa/                     # Memory Sparse Attention
│   ├── encoder.py           # Document chunking + routing key generation
│   ├── memory_bank.py       # Tiered storage (RAM keys, disk content)
│   ├── router.py            # Batch cosine similarity top-k selection
│   ├── interleave.py        # Multi-hop retrieval-generation loops
│   └── bridge.py            # Integration with cross-indexing to Memora
├── second_brain/            # Cognitive engine: tracking + collision + KG reasoning
│   ├── tracker.py           # Vitality scoring, dormancy detection, trends
│   ├── collision.py         # 7-strategy inspiration engine (5 RAG + 2 KG-driven)
│   ├── strategy_weights.py  # Adaptive collision strategy selection via ratings
│   ├── knowledge_graph.py   # KGNode/KGEdge + NetworkX DiGraph (insight storage)
│   ├── relation_extractor.py # LLM-based relation extraction + embedding pre-filter
│   ├── inference.py         # Contradiction detection, absence reasoning, propagation
│   ├── internalization.py   # Maturity tracking → Chronos PERSONALITY.yaml
│   └── bridge.py            # Unified interface
├── memory_server.py         # HTTP daemon with async task queue
├── memory_hub.py            # Unified ingestion/query router
├── memory_cli.py            # Thin HTTP client with async polling
├── llm_client.py            # xAI Grok API wrapper
├── shared_embedder.py       # Singleton embedding model
├── setup.py                 # Package configuration
├── tests/                   # 463 unit tests across 16 files
│   ├── test_memora.py
│   ├── test_chronos.py
│   ├── test_msa.py
│   ├── test_second_brain.py
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
python3 -m pytest tests/ --cov=memora --cov=chronos --cov=msa --cov=second_brain --cov=memory_server --cov=memory_hub -q
```

463 tests covering all four subsystems (including Second Brain's KG and inference engine), the server, the hub, and the CLI layer.

## Embedding Model

Uses [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (768-dim, Matryoshka support). The model is loaded once by the Memory Server and shared across Memora and MSA via the `shared_embedder` singleton. When the model is unavailable, a deterministic hash-based `MockEmbedder` serves as fallback (no semantic understanding).

For offline environments, pre-download the model and set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Known Limitations

See [STUBS.md](STUBS.md) for a full catalog of stubbed/simulated components. Key items:

- **Chronos EWC/LoRA**: Simulated without a real neural network — importance tracking and adapter management work, but no actual gradient-based learning occurs
- **Memora Distiller**: Dataset preparation is real, but LoRA fine-tuning is a placeholder (`TODO: integrate Axolotl/Unsloth`)
- **Daemon + MPS**: The daemon process forces CPU mode because macOS Metal services are unavailable after `setsid()`. Foreground mode uses MPS (Apple GPU) normally

## License

Private repository. All rights reserved.
