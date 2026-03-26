# OpenClaw Memory Pro System

An AI memory assistant that turns fragmented notes and conversations into searchable long-term memory, auto-distills actionable skills via a closed-loop feedback pipeline, and proactively reminds you.

### Get Started in 30 Seconds

```bash
pip install -e .                       # install
memory-cli server-start                # start (first run takes ~3 min to load model)
memory-cli health                      # confirm ready
memory-cli remember "Learned X today"  # store a memory
memory-cli recall "X"                  # recall (4-layer Context Composer: skills > KG > evidence > conflicts)
memory-cli briefing                    # daily briefing
```

---

**Contents:** [Product Goal](#product-goal) · [Vision Alignment](#vision-alignment-satisfied-vs-gaps) · [Architecture](#architecture) · [Quick Start](#quick-start)

## Product Goal

Support humans and agents in today's information environment through **fragmented, divergent, non-continuous learning** (patchy inputs, no fixed curriculum), while aiming — via **sustained accumulation over time** — for **systematic** understanding, **domain depth**, and **actionable** skills and procedures.

The subsystems (Memora, Chronos, MSA, Second Brain) were not originally designed from a single specification, but the statement above serves as the **unified narrative** that aligns all documentation and roadmap decisions.

## Vision Alignment: Satisfied vs. Gaps

| Area | Status | Notes |
|------|--------|-------|
| **Fragmented ingest** | **Largely satisfied** | `remember`, daily `memory/*.md`, AutoIngestor, HTTP API; text-first (video/chat need export/transcription upstream). |
| **Persistence over time** | **Largely satisfied** | JSONL vector store, daily logs, bookmarks, ingestion state; dedup prevents runaway duplicates. MSA auto-syncs daily logs for cross-day reasoning. |
| **Recall when needed** | **Satisfied** | 4-layer Context Composer: **L1 Core Facts → L2 Concept Links → L3 Background → L4 Conflicts**. Hybrid retrieval (BM25+dense) → cross-encoder reranking → strategy-routed assembly. Deep-recall multi-hop. |
| **Light structure & weaving** | **Partial** | KG + relation extraction (with `structural_gain` scoring), digests (with `compression_value` scoring), collision engine; adds edges and insights — not guaranteed holistic systemization. |
| **Proactive nudges** | **Partial** | Scheduler + Telegram for briefing, collision, dormant check, contradiction scan, blindspot scan, skill proposals. Not fully task-grounded. |
| **Systematic + professional depth** | **Not satisfied** | No automatic curriculum or domain mastery loop; depth comes from **how you use** recall + LLM, not from a finished "become expert" pipeline. |
| **Actionable skills** | **Partially satisfied** | Skill Proposer auto-generates draft skills via 2-of-3 scoring rule. Skills have utility tracking, feedback loop, and low-utility auto-rewrite. Structured format (prerequisites, procedures, scenarios). v0.0.7: Skills now support `action_type` binding (`prompt_template`, `tool_call`, `webhook`) with auto-generated executable prompts. |
| **True parametric continual learning** | **Not satisfied** | Chronos tracks importance and personality artifacts; **EWC/LoRA paths are largely simulated** — see [STUBS.md](STUBS.md). Nebius client is skeleton only. |
| **Learning quality & intent** | **Partial** | Ingest tags (`thought`, `share`, `reference`, `to_verify`) supported; not yet used for differential processing. |
| **Problem-triggered surfacing** | **Partially satisfied** | Assembled recall returns skills + KG relations + evidence as layered context; session-context includes active skills. No first-class "current task object" yet. |

**Summary:** The system has progressed from **durable capture + retrieval** to a **closed-loop skill evolution pipeline** with production-grade recall: fragments → KG/Digest/Collision processing → auto-proposed skills → Context Composer (hybrid retrieval → reranking → 4-layer assembly) → usage feedback → skill refinement. v0.0.8 adds hybrid BM25+dense retrieval, cross-encoder reranking, Context Composer with strategy routing, KG node embeddings, and task-prefixed embedding for nomic-embed-text. The remaining gap is **true parametric learning** (Nebius fine-tuning).

## Architecture

```
Fragments → [Ingest + Tag] → Unified Corpus (Memora vectors + optional long-doc MSA)
                                        │
                              ┌─────────┼──────────┐
                              ↓         ↓          ↓
                          [KG Weave] [Distill]  [Collide]
                          structural compression  novelty
                           _gain      _value      (1-5)
                              │         │          │
                              └────┬────┼────┬─────┘
                                   │   ↓    │
                              [Skill Proposer]    ← triggered when 2-of-3 scores pass
                                   ↓
                            [Skill Registry]      ← utility tracking + feedback loop
                           (draft → active → deprecated)
                                   │
                         ┌─────────┼──────────┐
                         ↓         ↓          ↓
                    [Question-   [Scheduled  [Nebius
                     Driven       Push]       Fine-
                     Recall]                  Tuning]
                         │
             ┌───────────┼───────────┐
             ↓           ↓           ↓
         [Skills]   [KG Nodes]   [Evidence]     ← parallel raw retrieval
             │           │           │
             │      (embeddings)  (hybrid BM25+dense)
             │           │           │
             └─────── [Reranker] ───┘              ← cross-encoder reranking
                         │
                  [Context Composer]                ← strategy-routed 4-layer assembly
                  ┌──────┼──────┐──────┐
                  ↓      ↓      ↓      ↓
              L1 Core  L2 Links L3 BG  L4 Conflicts
                  │
                  ↓
        Use → Feedback → utility update → low-utility auto-rewrite  ← closed loop
```

### Subsystem Roles

| Layer | Module | Role |
|-------|--------|------|
| **Unified Corpus** | **Memora** | Hybrid vector store (dense cosine + BM25 sparse, RRF fusion). nomic-embed-text with task-prefixed encoding. JSONL-backed. All content enters here. |
| | **MSA** | Optional document-level storage for long text (>=100 words) or high importance (>=0.85). Sparse routing + LLM-powered multi-hop interleave. Daily logs auto-synced. |
| **Intelligence** | **Second Brain** | KG weaving (`structural_gain`) with node-level semantic embeddings, distillation/summarization (`compression_value`), collision/association (`novelty`). Reads from Memora/MSA, writes to KG + insights + digests. |
| | **Skill Proposer** | Auto-generates draft skills when 2-of-3 scores meet thresholds (KG >= 0.6, Digest >= 0.7, Collision >= 4). |
| **Skill** | **Skill Registry** | Versioned, lifecycle-managed skills with utility tracking. Feedback loop triggers LLM rewrite on low utility. Structured format: prerequisites / procedures / scenarios. Supports `action_type` binding (prompt_template / tool_call / webhook) with auto-generated executable prompts. |
| **Training** | **Chronos** | Training data export (distiller), replay buffer, personality profile generation, Nebius fine-tuning client (skeleton). |

### Second Brain (Intelligence Layer)

Second Brain is the cognitive engine — it sits **above** the storage layer and provides value extraction:

1. **Digest / Distillation** — Periodic LLM-powered summarization of daily memories into long-term digests. Each digest receives a `compression_value` score (0-1) measuring compression ratio, decision density, day coverage, and novelty.
2. **KG Weaving** — LLM-based relation extraction into a Knowledge Graph (nodes: fact/decision/preference/goal/question; edges: supports/contradicts/extends/depends_on). Each extraction receives a `structural_gain` score (0-1) measuring integration, contradiction discovery, question addressing, and community bridging.
3. **Collision Engine** — 7-strategy inspiration generation (5 RAG-based + 2 KG-driven), adaptive weights. Attention-aware anchor selection: extracts user's current focus topics from recent memories and biases collisions toward them with recency weighting. MSA chunks participate as individual entries in the collision pool.
4. **Skill Proposer** — Scans recent KG/Digest/Collision scores; when >=2 of 3 meet thresholds, auto-proposes a draft skill with content extracted from mature KG nodes, digest summaries, and high-novelty insights.
5. **Inference Engine** — Contradiction detection, absence reasoning, forward propagation, thread discovery.
6. **Tracker** — Memory vitality scoring, dormancy detection, trend analysis.
7. **Internalization** — High-maturity KG patterns exported to Chronos PERSONALITY.yaml.

### Skill Registry (Skill Evolution Layer)

Inspired by [Memento-Skills (arXiv:2603.18743)](https://arxiv.org/abs/2603.18743), the Skill Registry implements a closed-loop skill evolution mechanism:

| Feature | Description |
|---------|-------------|
| **Structured format** | Each skill has: `prerequisites`, `core knowledge`, `procedures`, `applicable scenarios`, `inapplicable scenarios` |
| **Utility tracking** | Every recall+use records `success` or `failure`; `utility_rate = successes / total` |
| **Auto-rewrite** | When `utility_rate < 30%` after >=3 uses, LLM rewrites the skill content, bumps version, resets counters |
| **Usage logging** | All feedback events written to `usage_log.jsonl` (query -> skill -> outcome) for future router training |
| **Vector routing** | Skills are matched via embedding similarity (shared_embedder), with keyword fallback |

### Chronos (Training Pipeline)

Chronos has been refactored from simulated EWC/LoRA to a focused training pipeline:

1. **Encoder** — Structured memory encoding (facts, preferences, emotions, causal links)
2. **Replay Buffer** — Importance-weighted training candidate storage
3. **Distiller** — JSONL dataset generation from digests + buffer (chronos_lora_row_v1 format)
4. **Consolidator** — Periodic personality profile generation (PERSONALITY.yaml)
5. **Nebius Client** — Skeleton for cloud fine-tuning (upload -> train -> poll)

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
| **Memory Hub** | Unified router — auto-selects subsystems based on content length and importance (>=0.85 triggers Chronos + MSA); recall delegates to Context Composer (4-layer assembly with strategy routing, cross-encoder reranking, conflict scan, token budget control); extensible post-remember/post-recall hooks |
| **Memory Server** | Persistent HTTP daemon — keeps embedding model loaded, exposes all operations as REST endpoints, async task queue, built-in scheduler + Telegram push |
| **memory-cli** | Zero-dependency HTTP client — async polling with progress spinner for long-running tasks |
| **LLM Client** | Multi-provider LLM router (OpenRouter primary, xAI fallback) — used by digest, interleave, consolidation, collision, KG extraction, skill rewrite, and attention focus extraction |
| **Shared Embedder** | Singleton `nomic-ai/nomic-embed-text-v1.5` with task-prefixed encoding (`search_document:` / `search_query:`). Shared across Memora, MSA, KG node embeddings, and skill routing |
| **Context Composer** | 4-stage recall assembly: strategy router (intent classification) → layered assembly (L1-L4 with MMR dedup) → quality gate → CJK-aware budget controller |
| **BM25 Index** | Zero-dependency sparse retrieval (Okapi BM25 with Chinese tokenization support). Fused with dense results via Reciprocal Rank Fusion |
| **Reranker** | Cross-encoder second-stage reranking (ms-marco-MiniLM-L-6-v2, 22M params). Falls back to pass-through when unavailable |

## How It Works

### Ingestion Routing (Memory Hub)

```
Content arrives at hub.remember()
    |
    +-- < 100 words, importance < 0.85 -> Memora only
    +-- >= 100 words  -----------------> Memora + MSA
    +-- importance >= 0.85  ------------> Memora + MSA + Chronos (auto-routed)
    |
    +-- Always writes to memory/YYYY-MM-DD.md (daily log)
    +-- Post-remember hooks: KG relation extraction, access tracking
    +-- Chronos: also available via explicit force_systems=["chronos"]
```

### Query Routing (Context Composer Recall)

**`recall`** uses a 4-stage pipeline to produce context-composed output:

```
query
  │
  ├─ Stage 1: Raw Retrieval (parallel)
  │    ├── Skill Registry (vector similarity → active skills)
  │    ├── Knowledge Graph (semantic node matching via cached embeddings)
  │    ├── Memora (hybrid: BM25 sparse + dense cosine → RRF fusion)
  │    └── MSA documents (sparse routing + LLM interleave)
  │
  ├─ Stage 2: Reranking
  │    └── Cross-encoder (ms-marco-MiniLM-L-6-v2) rescores evidence candidates
  │
  ├─ Stage 3: Conflict Scan
  │    └── KG contradiction edges + inference engine → real-time conflict detection
  │
  └─ Stage 4: Context Composer
       ├── Strategy Router: classify intent (factual/thinking/planning/review)
       ├── Layered Assembly with MMR dedup:
       │    L1 Core Facts     — skills + highest-relevance evidence
       │    L2 Concept Links  — KG relations + conceptual associations
       │    L3 Background     — long-term summaries, lower-rank evidence
       │    L4 Conflicts      — contradictions + warnings
       ├── Quality Gate: multi-dim scoring (relevance × recency × importance)
       └── Budget Controller: CJK-aware token estimation, dynamic rebalancing
            → trimmed to max_tokens (default 4000)
```

Strategy-specific token allocation:

| Intent | L1 Core | L2 Links | L3 Background | L4 Conflicts |
|--------|---------|----------|---------------|--------------|
| factual | 50% | 15% | 25% | 10% |
| thinking | 25% | 35% | 25% | 15% |
| planning | 30% | 25% | 20% | 25% |
| review | 20% | 15% | 55% | 10% |

- **`deep-recall`** — MSA multi-hop interleave + Memora context + matching skills
- **`search`** — Direct Memora hybrid search (BM25 + dense, with min_score filtering)

### Skill Evolution Pipeline

The complete information -> knowledge -> skill pipeline:

```
Fragments --> Memora + MSA --> KG Weaving (structural_gain)
                               Distillation (compression_value)
                               Collision (novelty)
                                    |
                              >= 2 scores pass threshold?
                                    | yes
                              Skill Proposer -> draft skill
                                    |
                              Skill Registry (promote -> active)
                                    |
                              Assembled Recall -> usage
                                    |
                              Feedback (success/failure)
                                    |
                              utility_rate update
                                    | < 30%?
                              LLM Auto-Rewrite -> v(n+1)
```

### Async Task Queue

Long-running operations (collide ~50s, deep-recall ~30s, consolidate ~20s) run asynchronously:

```
POST /second-brain/collide {"async": true}
  -> 202 {"task_id": "abc123", "status": "running", "poll": "/task/abc123"}

GET /task/abc123
  -> {"status": "running", "elapsed": 12.3}
  -> {"status": "done", "elapsed": 48.7, "result": {...}}
```

The CLI handles this transparently with a polling spinner:

```
[spinner] Collision ... 45s [running]
[done]    Collision complete (48s)
Generated 3 insights
```

### Inspiration Collision (Second Brain)

Every 6 hours, the collision engine selects from **7 strategies** using adaptive weights (strategies that produce higher-rated insights get selected more often):

**RAG-based strategies:**
1. **Semantic Bridge** — Finds "distant relatives" (similarity 0.35-0.65) between memories
2. **Dormant Revival** — Pairs sleeping high-importance memories with recent active ones
3. **Temporal Echo** — Crosses today's memories with those from 7/30 days ago
4. **Chronos Cross-Reference** — Pairs Chronos structured memories with Memora entries
5. **Digest Bridge** — Crosses digest summaries with individual memories

**KG-driven strategies (fundamentally different from RAG):**
6. **Contradiction-Based** — Uses KG contradiction edges to surface decisions at risk, ranked by evidence balance
7. **Blind Spot-Based** — Uses KG absence reasoning to identify unexplored dimensions in important decisions

**Attention Focus (v0.0.7):** Before each round, the engine extracts 3-5 focus keywords from the user's most recent memories (last 3 days). Anchor selection uses recency-weighted + attention-boosted probability — memories matching current focus topics are selected ~70x more often than old unrelated ones. The focus keywords are also injected into the LLM collision prompt to guide relevant insight generation.

Each collision is evaluated by an LLM that scores novelty (1-5). Users can rate insights (`memory-cli rate <id> <1-5>`), and the system automatically adjusts strategy selection weights.

### Feedback Loops

**Collision strategy feedback:**
```
Collision -> Insight -> User Rating -> Strategy Weight Update -> Better Collisions
```

**Skill utility feedback:**
```
Recall -> Skill Used -> Outcome (success/failure) -> Utility Update -> Auto-Rewrite if low
```

Strategy weights persist in `memory/insights/strategy_weights.json`. Skill usage logs persist in `memory/skills/usage_log.jsonl`.

## Quick Start

### 1. Install

```bash
cd ~/.openclaw/workspace   # or your project directory
pip install -e .           # basic install
pip install -e ".[embeddings]"  # recommended: install real embedding model
```

**Requirements:** Python 3.9+, macOS (Mac mini M-series) or Linux

### 2. Configure (Optional)

The LLM client auto-detects API keys in this order:
1. `OPENROUTER_API_KEY` env var
2. OpenClaw `auth-profiles.json` (openrouter:default)
3. `XAI_API_KEY` env var
4. OpenClaw `auth-profiles.json` (xai:default)
5. `~/.openclaw/.env` file

Or set manually:
```
OPENROUTER_API_KEY=sk-or-v1-...    # preferred (routes to deepseek-r1 by default)
XAI_API_KEY=xai-...                # fallback
```

The LLM API is used for memory summarization, MSA multi-hop reasoning, inspiration collision, attention focus extraction, personality generation, skill rewriting, and KG extraction. **The system works without it** — advanced features gracefully degrade to local heuristics.

### 3. Start the Server

```bash
memory-cli server-start    # start in background
memory-cli health          # confirm ready (first run takes ~3 min to load embedding model)
```

### 4. Start Using

```bash
memory-cli remember "Learned about Transformer multi-head attention today" --tag thought
memory-cli recall "attention mechanism"
memory-cli briefing
memory-cli skills           # list skills with utility stats
memory-cli --help           # see all commands
```

The server loads the SentenceTransformer model once (~3s), then serves all requests with <100ms latency.

### Usage

```bash
# Remember something (auto-routes to appropriate subsystems + KG extraction)
memory-cli remember "User prefers dark theme, dislikes popup notifications" -i 0.9

# Search memories (4-layer Context Composer: hybrid retrieval -> reranking -> assembly)
memory-cli recall "user UI preferences"

# Deep multi-hop reasoning (with skill context)
memory-cli deep-recall "Based on past feedback, what kind of new features would users accept?"

# Skills — lifecycle management + utility feedback
memory-cli skills                      # List all skills with utility stats
memory-cli skill-add "name" "content"  # Create a new skill
memory-cli skill-on <skill_id>         # Promote to active
memory-cli skill-off <skill_id>        # Deprecate
memory-cli skill-feedback <id> success # Record positive feedback
memory-cli skill-feedback <id> failure --context "reason" # Record negative feedback
memory-cli skill-usage                 # View usage statistics
memory-cli skill-propose               # Trigger auto skill proposal

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
memora/                      # Unified Corpus — hybrid vector store
    collector.py             # Raw memory ingestion
    vectorstore.py           # JSONL-based hybrid store (dense + BM25 sparse, RRF fusion)
    embedder.py              # Embedding with search_document/search_query prefixes
    bridge.py                # Cross-system integration (min_score filtering)
msa/                         # Unified Corpus — long-document store (Memory Sparse Attention)
    encoder.py               # Document chunking + routing key generation
    memory_bank.py           # Tiered storage (RAM keys, disk content)
    router.py                # Batch cosine similarity top-k selection
    interleave.py            # Multi-hop retrieval-generation loops (LLM-powered sufficiency + reformulation)
    bridge.py                # Integration with cross-indexing to Memora
second_brain/                # Intelligence Layer — KG / Digest / Collision / Skill Proposal
    digest.py                # LLM summarization + compression_value scoring
    tracker.py               # Vitality scoring, dormancy detection, trends
    collision.py             # 7-strategy inspiration engine (5 RAG + 2 KG-driven)
    strategy_weights.py      # Adaptive collision strategy selection via ratings
    knowledge_graph.py       # KGNode/KGEdge + NetworkX DiGraph + per-node semantic embeddings
    relation_extractor.py    # LLM relation extraction + structural_gain scoring
    skill_proposer.py        # 2-of-3 scoring -> auto draft skill proposal
    inference.py             # Contradiction detection, absence reasoning, propagation
    internalization.py       # Maturity tracking -> Chronos PERSONALITY.yaml
    bridge.py                # Unified interface (MSA chunks as collision material)
skill_registry/              # Skill Layer — closed-loop skill evolution
    registry.py              # Structured skills + utility tracking + feedback + auto-rewrite
chronos/                     # Training Layer — Nebius fine-tuning pipeline
    encoder.py               # Importance-weighted memory encoding
    replay_buffer.py         # Training candidate buffer
    distiller.py             # JSONL training dataset generation
    nebius_client.py         # Cloud fine-tuning client skeleton
    consolidator.py          # Personality profile generation (PERSONALITY.yaml)
    bridge.py                # OpenClaw integration
memory_server.py             # HTTP daemon + scheduler + Telegram push + async queue + KG embedding backfill
memory_hub.py                # Unified ingestion/query router + Context Composer recall pipeline
memory_cli.py                # Thin HTTP client with async polling
context_composer.py          # 4-stage context assembly: strategy router → layered assembly → quality gate → budget controller
bm25.py                      # Zero-dependency BM25 sparse retrieval (Chinese tokenization support)
reranker.py                  # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2, 22M params)
llm_client.py                # Multi-provider LLM router (OpenRouter / xAI)
shared_embedder.py           # Singleton embedding model
setup.py                     # Package configuration
tests/                       # 536 unit tests across 17 files
    test_memora.py
    test_chronos.py
    test_msa.py
    test_second_brain.py
    test_skill_registry.py   # Utility tracking, feedback, proposer tests
    test_knowledge_graph.py
    test_inference.py
    test_relation_extractor.py
    test_internalization.py
    test_strategy_weights.py
    test_memory_server.py
    test_memory_hub.py       # Context Composer recall, KG recall, skill recall tests
    test_doc_consistency.py  # Guards against doc/code drift
    test_cli_coverage.py
    ...
skills/                      # OpenClaw skill manifests
AGENTS.md                    # Agent behavior configuration
HEARTBEAT.md                 # Periodic task definitions
STUBS.md                     # Documentation of all placeholder code
```

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -q

# With coverage
python3 -m pytest tests/ --cov=memora --cov=chronos --cov=msa --cov=second_brain --cov=skill_registry --cov=memory_server --cov=memory_hub -q
```

536 tests across 17 files, covering all subsystems (Memora hybrid search, MSA, Second Brain KG/inference/skill proposer, Chronos training pipeline, Skill Registry with utility tracking), the server, the hub (Context Composer pipeline), and the CLI layer.

## Embedding Model

Uses [`nomic-ai/nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (768-dim, Matryoshka support) with **task-prefixed encoding**: `search_document:` prefix for stored content, `search_query:` prefix for queries — this asymmetric encoding significantly improves retrieval quality per the model's design. The model is loaded once by the Memory Server and shared across Memora, MSA, KG node embeddings, and skill routing via the `shared_embedder` singleton. KG nodes receive cached embeddings on first access (with full backfill on server startup). When the model is unavailable, a deterministic hash-based `MockEmbedder` serves as fallback (no semantic understanding).

For offline environments, pre-download the model and set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Known Limitations

See [STUBS.md](STUBS.md) for a full catalog of stubbed/simulated components. Key items:

- **Chronos Nebius Client**: Skeleton only — `upload_dataset`, `create_job`, `poll_job` raise `NotImplementedError`. See [NEBIUS_FINETUNE_INTEGRATION_SKETCH.md](docs/NEBIUS_FINETUNE_INTEGRATION_SKETCH.md) for the integration plan.
- **Chronos EWC/LoRA**: `ewc.py` and `dynamic_lora.py` still exist on disk but are **no longer imported** by any production code (deprecated since v0.0.3-beta).
- **Skill Registry**: Auto-proposal pipeline works. v0.0.7 adds `action_type` binding (`prompt_template` / `tool_call` / `webhook`) with auto-generated executable prompts. Full executable workflows (code generation + test scaffolding) as in [Memento-Skills](https://arxiv.org/abs/2603.18743) are not yet implemented. Behaviour-aligned router training requires more usage data accumulation.
- **Cross-Encoder Reranker**: Uses `ms-marco-MiniLM-L-6-v2` (22M params, English-optimized). Falls back to pass-through when model is unavailable. Chinese reranking quality may be limited.
- **Daemon + MPS**: The daemon process forces CPU mode because macOS Metal services are unavailable after `setsid()`. Foreground mode uses MPS (Apple GPU) normally.

## License

Private repository. All rights reserved.
