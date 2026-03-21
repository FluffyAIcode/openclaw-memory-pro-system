# OpenClaw Memory Pro System

A production-grade cognitive architecture for the [OpenClaw](https://github.com/nicholasgasior/openclaw) AI agent framework. Five interconnected subsystems — including a **Knowledge Graph cognitive engine** — turn passive conversation logging into semantic retrieval, continual learning, document-level reasoning, structured logical reasoning, and autonomous inspiration generation.

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
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ │   │
│  │  │  Memora   │ │ Chronos  │ │   MSA    │ │Second Brain │ │   │
│  │  │  (RAG)   │ │  (CL)    │ │ (Sparse) │ │(KG+Insights)│ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────┬──────┘ │   │
│  │                                                 │        │   │
│  │  ┌──────────────────────────────────────────────▼──────┐ │   │
│  │  │           Knowledge Graph (NetworkX DiGraph)         │ │   │
│  │  │  Nodes: fact/decision/preference/goal/question       │ │   │
│  │  │  Edges: supports/contradicts/extends/depends_on      │ │   │
│  │  │  → Contradiction Detection                           │ │   │
│  │  │  → Absence Reasoning (Blind Spot Detection)          │ │   │
│  │  │  → Forward Propagation (Impact Tracing)              │ │   │
│  │  │  → Thread Discovery (Community Detection)            │ │   │
│  │  └─────────────────────────┬────────────────────────────┘ │   │
│  │                            │ mature patterns              │   │
│  │                            ▼                              │   │
│  │                   Internalization Manager                  │   │
│  │                   → Chronos PERSONALITY.yaml               │   │
│  │                   → LoRA training data (future)            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Five Memory Subsystems

| System | Purpose | Storage | Key Capability |
|--------|---------|---------|----------------|
| **Memora** | Snippet-level semantic search | Vector store (JSONL) | RAG with real embeddings, LLM-powered daily digests |
| **Chronos** | Continual learning | Replay buffer + EWC state | Importance-weighted memory encoding, personality profile generation |
| **MSA** | Document-level reasoning | Tiered (routing keys in RAM, content on disk) | Sparse attention routing, multi-hop interleave with LLM |
| **Second Brain** | Active memory lifecycle + cognitive reasoning | Access log + insights + KG | Vitality tracking, inspiration collision, **Knowledge Graph** |
| **Knowledge Graph** | Structured logical reasoning | NetworkX DiGraph (JSONL) | Contradiction detection, absence reasoning, forward propagation |

### Knowledge Graph — What RAG Cannot Do

The Knowledge Graph is the core differentiator of this system. While RAG retrieves by cosine similarity, the KG reasons by typed logical relationships:

| Capability | RAG | Knowledge Graph |
|------------|-----|-----------------|
| Find similar content | Yes | — |
| Find logically related content (even if semantically distant) | No | **Yes** — graph traversal along typed edges |
| Detect contradictions | No | **Yes** — `contradicts` edges between facts and decisions |
| Discover blind spots (what hasn't been considered) | No | **Yes** — absence reasoning: expected dimensions minus covered dimensions |
| Trace impact of new information | No | **Yes** — forward propagation along `depends_on`/`supports` chains |
| Auto-discover thought threads | No | **Yes** — community detection on the graph |

**How it works:**
1. Every `/remember` call triggers async relation extraction (LLM + embedding pre-filter)
2. LLM identifies knowledge units (facts, decisions, preferences, goals, questions) and their logical relationships
3. The inference engine continuously monitors for contradictions, blind spots, and impact chains
4. High-maturity patterns (frequently accessed, high confidence, stable over time) feed into Chronos for PERSONALITY.yaml generation — the "internalization pipeline" from explicit (KG) to implicit (model behavior) memory

### Service Layer

| Component | Role |
|-----------|------|
| **Memory Hub** | Unified router — auto-selects subsystems based on content length, importance, and query type |
| **Memory Server** | Persistent HTTP daemon — keeps the embedding model loaded, exposes all operations as REST endpoints, async task queue for slow operations |
| **memory-cli** | Zero-dependency HTTP client — async polling with progress spinner for long-running tasks |
| **LLM Client** | Unified xAI Grok API wrapper — used by digest, interleave, consolidation, collision, and KG relation extraction |
| **Shared Embedder** | Singleton `nomic-ai/nomic-embed-text-v1.5` instance shared across Memora, MSA, and KG candidate pre-filtering |
| **Internalization Manager** | Bridges KG (explicit) to Chronos (implicit) — extracts mature patterns for personality profiles and future LoRA training |

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
    └── Async: KG relation extraction (if importance >= 0.4)
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

# Run inspiration collision (7 strategies, adaptive weights)
memory-cli collide

# Knowledge Graph — reasoning RAG cannot do
memory-cli contradictions          # Find decisions with conflicting evidence
memory-cli blindspots              # Detect unexplored dimensions in decisions
memory-cli threads                 # Discover thought threads via community detection
memory-cli graph-status            # KG statistics

# Feedback loop
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
├── second_brain/            # Cognitive engine + active memory lifecycle
│   ├── knowledge_graph.py   # KGNode/KGEdge + NetworkX DiGraph + JSONL persistence
│   ├── relation_extractor.py # LLM-based relation extraction + embedding pre-filter
│   ├── inference.py         # Contradiction detection, absence reasoning, propagation
│   ├── internalization.py   # Maturity tracking, pattern extraction, training data
│   ├── strategy_weights.py  # Adaptive collision strategy selection via ratings
│   ├── tracker.py           # Vitality scoring, dormancy detection, trends
│   ├── collision.py         # 7-strategy inspiration engine (5 RAG + 2 KG-driven)
│   └── bridge.py            # Unified interface
├── memory_server.py         # HTTP daemon with async task queue
├── memory_hub.py            # Unified ingestion/query router
├── memory_cli.py            # Thin HTTP client with async polling
├── llm_client.py            # xAI Grok API wrapper
├── shared_embedder.py       # Singleton embedding model
├── setup.py                 # Package configuration
├── tests/                   # 447 unit tests across 16 files
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

447 tests covering all subsystems, the Knowledge Graph, the inference engine, the server, the hub, and the CLI layer.

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
