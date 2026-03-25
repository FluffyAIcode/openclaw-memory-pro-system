# Stubbed / Simulated Code in OpenClaw Memory System

This document catalogs every piece of production code that is **stubbed**, **simulated**,
or uses a **placeholder/fallback** instead of a real implementation. For each entry:

- **File** and **Line(s)** — where the stub lives
- **What's Stubbed** — the concept it replaces
- **Behavior** — what the stub actually does
- **To Make Real** — what is needed to replace it

---

## 1. Chronos — EWC Engine (Simulated Mode)

| | |
|-|-|
| **File** | `chronos/ewc.py` |
| **Lines** | 20-26 (torch import), 56-58 (mode switch), 106-120 (simulated impl) |
| **What's Stubbed** | Real Elastic Weight Consolidation on a neural network |
| **Behavior** | When `torch` is unavailable or no model is attached via `set_model()`, `_learn_simulated()` is called. It assigns heuristic "Fisher values" (`importance * 10.0`) and computes a fake EWC loss from importance scores. No gradient computation, no real parameter regularization. |
| **To Make Real** | Install `torch`, instantiate a real `torch.nn.Module`, call `ewc_engine.set_model(model)`. The `_learn_real()` / `_compute_fisher_real()` / `_ewc_loss_real()` methods (lines 78-102) contain the genuine Fisher-information implementation — they are **not stubs** themselves, they are simply unreachable without a model. |

---

## 2. Chronos — Dynamic LoRA (Virtual Adapters)

| | |
|-|-|
| **File** | `chronos/dynamic_lora.py` |
| **Lines** | 19-64 (entire class) |
| **What's Stubbed** | Real LoRA (Low-Rank Adaptation) weight matrices |
| **Behavior** | `DynamicLoRA` maintains a dictionary mapping `adapter_id → importance_score`. It tracks allocations and evictions purely by importance thresholds — no actual low-rank matrices are created, no model weights are modified. `merge()` simply clears the dict. |
| **To Make Real** | Integrate with a LoRA training framework (e.g. PEFT / Unsloth / Axolotl). Replace `self._adapters` dict with actual `LoraLayer` objects and implement weight manipulation in `allocate()` / `merge()`. |

---

## 3. Chronos — Memory Encoder (Heuristic Importance)

| | |
|-|-|
| **File** | `chronos/encoder.py` |
| **Lines** | `_auto_importance()`, `_extract_facts()`, `_extract_preferences()`, `_extract_emotions()`, `_extract_causal()` |
| **What's Stubbed** | LLM-powered content analysis and importance scoring |
| **Behavior** | Uses keyword matching (regex patterns for Chinese: `重要`, `决定`, `记住`, `喜欢`, `不喜欢`, `因为`, `所以`, etc.) and text-length heuristics to estimate importance and extract structured fields. No NLP model or LLM is invoked. `_extract_facts()` just returns each sentence as a "fact". |
| **To Make Real** | Call an LLM (via `llm_client.generate()`) with a prompt asking for structured extraction of facts, preferences, emotions, causal links, and an importance score. |

---

## 4. Memora — LoRA Distiller (Training Placeholder)

| | |
|-|-|
| **File** | `memora/distiller.py` |
| **Lines** | 50-59 (`start_training()`) |
| **What's Stubbed** | Actual LoRA fine-tuning |
| **Behavior** | `start_training()` calls `prepare_dataset()` (which does real JSONL generation from digest files), then logs the training parameters and returns `True` without performing any training. Line 57: `# TODO: 对接 Axolotl / Unsloth`. |
| **To Make Real** | Integrate with Axolotl, Unsloth, or HuggingFace PEFT to consume the generated JSONL dataset and produce LoRA weights. |

---

## 5. Memora — MockEmbedder (Hash-based Fallback)

| | |
|-|-|
| **File** | `memora/embedder.py` |
| **Lines** | 9-25 (`MockEmbedder` class) |
| **What's Stubbed** | Semantic text embedding |
| **Behavior** | Produces deterministic pseudo-random vectors seeded by the MD5 hash of input text. Vectors are normalized to unit length. These do not capture semantic meaning — cosine similarity between "king" and "queen" is random. |
| **Active When** | The `SentenceTransformerEmbedder` fails to load (model not cached, `sentence_transformers` not installed, etc.) and `shared_embedder` is not set by the Memory Server. When the Memory Server is running and the real model loads, `MockEmbedder` is NOT used. |
| **To Make Real** | Already real when `SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")` loads successfully. The `_create_embedder()` chain: shared_embedder → SentenceTransformer → MockEmbedder. |

---

## 6. MSA — MockEmbedder (Same Pattern)

| | |
|-|-|
| **File** | `msa/encoder.py` |
| **Lines** | 18-36 (`MockEmbedder` class) |
| **What's Stubbed** | Same as above — semantic embedding for MSA's chunk encoder |
| **Behavior** | Identical hash-based mock as Memora's. Used as fallback when the real model is unavailable. |
| **To Make Real** | Same as Memora — runs real model when Memory Server provides the shared embedder. |

---

## 7. ~~Memora — vLLM URL (Unused)~~ **RESOLVED**

Removed in v0.0.7. The dead `vllm_url` config field was deleted from `config.py` and `config.yaml`. LLM inference uses `llm_client.py` (xAI Grok API).

---

## 8. Memora — ZFS Integration (Disabled)

| | |
|-|-|
| **File** | `memora/zfs_integration.py` |
| **Lines** | 1-50 (entire file) |
| **What's Stubbed** | ZFS snapshot-based backup |
| **Behavior** | `ZFSIntegration` exists with working `check_zfs_available()` and `create_snapshot()` methods, but `enabled` defaults to `False` and `dataset` defaults to `None`. The config field `use_zfs_snapshot` is `false` in `config.yaml`. No code path enables it. |
| **To Make Real** | Set `enabled = True`, assign `dataset` to an actual ZFS dataset name, and add an integration point (e.g., call `zfs.create_snapshot()` after digest). |

---

## 9. ~~MSA — Interleave Sufficiency Check (Heuristic)~~ **UPGRADED**

Now uses `llm_client.generate()` for LLM-powered YES/NO sufficiency judgement.
Falls back to keyword heuristic when LLM is unavailable.

---

## 10. ~~MSA — Interleave Reformulation (String Concatenation)~~ **UPGRADED**

Now uses `llm_client.generate()` for intelligent query reformulation.
Falls back to string concatenation when LLM is unavailable.

---

## 11. Chronos — Personality Profile (LLM-dependent)

| | |
|-|-|
| **File** | `chronos/consolidator.py` |
| **Lines** | 94-140 (`_generate_personality_profile()`) |
| **What's Stubbed** | N/A — **this is real code** |
| **Behavior** | When `llm_client` is available and has an API key, this generates a real YAML personality profile using xAI Grok. When the LLM is unavailable, it **silently skips** and returns `False`. |
| **Note** | Not a stub per se, but the feature is **conditionally active**. Without `XAI_API_KEY`, this function is effectively disabled. |

---

## 12. Memory Server — `main()` Daemon Mode

| | |
|-|-|
| **File** | `memory_server.py` |
| **Lines** | 286-336 |
| **Coverage** | 0% (untested) |
| **Reason** | Contains `os.fork()` for daemonization and `HTTPServer.serve_forever()` — both require process-level operations that can't be meaningfully unit-tested. Tested via integration (launchd). |

---

## Summary Table

| # | Component | Type | Risk Level | Effort to Replace |
|---|-----------|------|------------|-------------------|
| 1 | EWC Engine (simulated) | Simulation | **High** — core CL claim | High (needs model + GPU) |
| 2 | Dynamic LoRA (virtual) | Simulation | **High** — core CL claim | High (needs PEFT integration) |
| 3 | Memory Encoder (heuristic) | Heuristic | Medium | Medium (LLM call) |
| 4 | LoRA Distiller (no training) | Placeholder | Medium | High (needs training infra) |
| 5 | Memora MockEmbedder | Fallback | Low (real model used normally) | Already real |
| 6 | MSA MockEmbedder | Fallback | Low (real model used normally) | Already real |
| 7 | vLLM URL | Dead config | None | Remove or wire in |
| 8 | ZFS Integration | Disabled | None | Enable + configure |
| 9 | Interleave sufficiency | Heuristic | Low | Easy (LLM call) |
| 10 | Interleave reformulation | Heuristic | Low | Easy (LLM call) |
| 11 | Personality Profile | Conditional | Low | Already real when API key present |
| 12 | Server daemon | Untestable | None | Integration-tested via launchd |
