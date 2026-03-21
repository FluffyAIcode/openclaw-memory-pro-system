# OpenClaw MindSystem

A multi-layered cognitive architecture for the [OpenClaw](https://github.com/nicholasgasior/openclaw) AI agent framework. Goes beyond passive conversation logging — four interconnected subsystems enable semantic retrieval, continual learning, document-level reasoning, and autonomous inspiration generation.

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
│  │  │  (RAG)   │ │  (CL)    │ │ (Sparse) │ │ (Insights)  │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └─────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Four Memory Subsystems

| System | Purpose | Storage | Key Capability |
|--------|---------|---------|----------------|
| **Memora** | Snippet-level semantic search | Vector store (JSONL) | RAG with real embeddings, LLM-powered daily digests |
| **Chronos** | Continual learning | Replay buffer + EWC state | Importance-weighted memory encoding, personality profile generation |
| **MSA** | Document-level reasoning | Tiered (routing keys in RAM, content on disk) | Sparse attention routing, multi-hop interleave with LLM |
| **Second Brain** | Active memory lifecycle | Access log + insights | Vitality tracking, dormancy detection, inspiration collision |

### Service Layer

| Component | Role |
|-----------|------|
| **Memory Hub** | Unified router — auto-selects subsystems based on content length, importance, and query type |
| **Memory Server** | Persistent HTTP daemon — keeps the embedding model loaded, exposes all operations as REST endpoints, async task queue for slow operations |
| **memory-cli** | Zero-dependency HTTP client — async polling with progress spinner for long-running tasks |
| **LLM Client** | Unified xAI Grok API wrapper — used by digest, interleave, consolidation, and collision |
| **Shared Embedder** | Singleton `nomic-ai/nomic-embed-text-v1.5` instance shared across Memora and MSA |

## How It Works

### Ingestion Routing (Memory Hub)

```
Content arrives at hub.remember()
    │
    ├── < 100 words  ──────────────────────> Memora only
    ├── 100-500 words + high importance ───> Memora + Chronos
    └── > 500 words ───────────────────────> Memora + MSA + Chronos
    │
    └── Always writes to memory/YYYY-MM-DD.md (daily log)
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

Every 6 hours, the collision engine runs 5 strategies against the full memory pool:

1. **Semantic Bridge** — Finds "distant relatives" (similarity 0.35–0.65) between memories
2. **Dormant Revival** — Pairs sleeping high-importance memories with recent active ones
3. **Temporal Echo** — Crosses today's memories with those from 7/30 days ago
4. **Chronos Cross-Reference** — Pairs encoded Chronos memories with different-source Memora entries, with Chronos context (facts, preferences, emotions) included in the LLM prompt
5. **Digest Bridge** — Crosses digest summaries with individual memories for macro-micro connections

Each collision is evaluated by an LLM that scores novelty (1–5) and generates actionable ideas. High-novelty insights (≥4) are automatically indexed back into Memora for future retrieval.

## Quick Start

### Prerequisites

- Python 3.9+
- macOS (tested on Mac mini M-series) or Linux

### Installation

```bash
cd /path/to/openclaw-mindsystem

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
# Remember something (auto-routes to appropriate subsystems)
memory-cli remember "用户偏好深色主题，不喜欢弹窗通知" -i 0.9

# Search memories
memory-cli recall "用户的UI偏好"

# Deep multi-hop reasoning
memory-cli deep-recall "根据用户过去的反馈，他们最可能接受什么样的新功能？"

# Run inspiration collision
memory-cli collide

# Check system status
memory-cli status

# View Second Brain report
memory-cli sb-report
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
├── second_brain/            # Active memory lifecycle
│   ├── tracker.py           # Vitality scoring, dormancy detection, trends
│   ├── collision.py         # 5-strategy inspiration engine
│   └── bridge.py            # Unified interface
├── memory_server.py         # HTTP daemon with async task queue
├── memory_hub.py            # Unified ingestion/query router
├── memory_cli.py            # Thin HTTP client with async polling
├── llm_client.py            # xAI Grok API wrapper
├── shared_embedder.py       # Singleton embedding model
├── setup.py                 # Package configuration
├── tests/                   # 332 unit tests across 11 files
│   ├── test_memora.py
│   ├── test_chronos.py
│   ├── test_msa.py
│   ├── test_second_brain.py
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

332 tests covering all subsystems, the server, the hub, and the CLI layer.

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
