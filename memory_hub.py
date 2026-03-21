"""
Memory Hub — Unified interface for OpenClaw's memory architecture.

Refactored architecture:
  - Memora   (primary storage: vector RAG, snippet-level search)
  - MSA      (optional long-document storage + multi-hop reasoning)
  - Chronos  (training buffer — not on default ingest path)
  - Second Brain  (intelligence layer: KG, digest, collision — not a storage system)

Ingest tags (intent):
  - thought:    user's own thinking / analysis
  - share:      forwarded / curated content
  - reference:  factual reference material
  - to_verify:  unconfirmed / needs checking
  - (none):     untagged legacy content
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).parent

VALID_TAGS = {"thought", "share", "reference", "to_verify"}


class MemoryHub:
    """
    Unified memory interface.

    Ingestion routing (simplified):
      - All text → Memora (primary vector store, dedup-aware)
      - Long text (>500 chars) → also MSA (document-level)
      - Chronos buffer → only on explicit force_systems=["chronos"]
      - Always writes daily file

    Query routing:
      - recall → merged Memora + MSA
      - deep-recall → MSA multi-hop + Memora context
    """

    def __init__(self):
        self._memora_bridge = None
        self._chronos_bridge = None
        self._msa_bridge = None

    @property
    def memora(self):
        if self._memora_bridge is None:
            from memora.bridge import bridge
            self._memora_bridge = bridge
        return self._memora_bridge

    @property
    def chronos(self):
        if self._chronos_bridge is None:
            from chronos.bridge import bridge
            self._chronos_bridge = bridge
        return self._chronos_bridge

    @property
    def msa(self):
        if self._msa_bridge is None:
            from msa.bridge import bridge
            self._msa_bridge = bridge
        return self._msa_bridge

    def remember(self, content: str, source: str = "openclaw",
                 importance: float = 0.7,
                 tag: Optional[str] = None,
                 doc_id: Optional[str] = None,
                 title: Optional[str] = None,
                 force_systems: Optional[List[str]] = None) -> Dict:
        """
        Smart ingestion with intent tagging.

        Args:
            content: The text to remember
            source: Origin channel (e.g. "telegram", "cursor", "auto_ingest")
            importance: 0.0-1.0 importance score
            tag: Intent tag — thought | share | reference | to_verify
            doc_id: Optional document ID for MSA
            title: Optional title for MSA
            force_systems: Override auto-routing, e.g. ["memora", "msa", "chronos"]
        """
        if tag and tag not in VALID_TAGS:
            logger.warning("Unknown tag '%s', ignoring", tag)
            tag = None

        word_count = max(len(content.split()), len(content) // 2)
        results = {"word_count": word_count, "systems_used": [], "tag": tag}

        if force_systems:
            systems = set(force_systems)
        else:
            systems = self._route_ingestion(word_count, importance)

        metadata = {
            "source": source,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
        }
        if tag:
            metadata["tag"] = tag

        if "memora" in systems:
            try:
                from memora.collector import collector
                from memora.vectorstore import vector_store

                entry = collector.collect(content, source=source, importance=importance)
                vector_store.add(content, metadata=metadata)
                results["memora"] = entry
                results["systems_used"].append("memora")
            except Exception as e:
                logger.warning("Memora ingestion failed: %s", e)

        if "msa" in systems:
            try:
                msa_meta = {"source": source}
                if title:
                    msa_meta["title"] = title
                if tag:
                    msa_meta["tag"] = tag
                msa_result = self.msa.ingest_and_save(
                    content, source=source, doc_id=doc_id,
                    metadata=msa_meta,
                    cross_index=("memora" not in systems),
                    write_daily=False)
                results["msa"] = msa_result
                results["systems_used"].append("msa")
            except Exception as e:
                logger.warning("MSA ingestion failed: %s", e)

        if "chronos" in systems:
            try:
                chronos_result = self.chronos.learn_and_save(
                    content, source=source, importance=importance,
                    write_daily=False)
                results["chronos"] = {"importance": chronos_result.importance}
                results["systems_used"].append("chronos")
            except Exception as e:
                logger.warning("Chronos ingestion failed: %s", e)

        self._write_daily(content, source, results["systems_used"], tag)
        results["systems_used"].append("daily_file")

        logger.info("Memory Hub: remembered %d words via %s (tag=%s)",
                     word_count, ", ".join(results["systems_used"]), tag)
        return results

    def recall(self, query: str, top_k: int = 8) -> Dict:
        """Merged search across Memora (snippet) + MSA (document-level)."""
        results = {"query": query, "memora": [], "msa": [], "merged": []}

        try:
            memora_results = self.memora.search_across(query, include_msa=False)
            for r in memora_results:
                r["system"] = "memora"
            results["memora"] = memora_results
        except Exception as e:
            logger.warning("Memora recall failed: %s", e)

        try:
            msa_result = self.msa.query_memory(query, top_k=min(top_k, 5))
            for doc in msa_result.get("results", []):
                results["msa"].append({
                    "content": "\n".join(doc["chunks"][:3]),
                    "score": round(doc["score"], 4),
                    "metadata": {"doc_id": doc["doc_id"], "title": doc["title"]},
                    "system": "msa",
                })
        except Exception as e:
            logger.warning("MSA recall failed: %s", e)

        merged = results["memora"] + results["msa"]
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        results["merged"] = merged[:top_k]
        return results

    def deep_recall(self, query: str, max_rounds: int = 3) -> Dict:
        """Multi-hop reasoning using MSA Memory Interleave + Memora context."""
        results = {"query": query}

        try:
            interleave = self.msa.interleave_query(query, max_rounds=max_rounds)
            results["interleave"] = interleave
        except Exception as e:
            logger.warning("MSA interleave failed: %s", e)
            results["interleave"] = None

        try:
            memora_results = self.memora.search_across(query, include_msa=False)
            results["memora_context"] = memora_results[:5]
        except Exception as e:
            results["memora_context"] = []

        return results

    def status(self) -> Dict:
        """Combined status from all systems."""
        st = {"systems": {}}

        try:
            from memora.vectorstore import vector_store
            st["systems"]["memora"] = {"entries": vector_store.count()}
        except Exception as e:
            st["systems"]["memora"] = {"error": str(e)}

        try:
            st["systems"]["chronos"] = self.chronos.status()
        except Exception as e:
            st["systems"]["chronos"] = {"error": str(e)}

        try:
            st["systems"]["msa"] = self.msa.status()
        except Exception as e:
            st["systems"]["msa"] = {"error": str(e)}

        return st

    def _route_ingestion(self, word_count: int, importance: float) -> set:
        """Simplified routing: Memora always; MSA for long text."""
        systems = {"memora"}
        if word_count >= 100:
            systems.add("msa")
        return systems

    def _write_daily(self, content: str, source: str,
                     systems: List[str], tag: Optional[str] = None):
        memory_dir = _WORKSPACE / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        today = memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        tags = "+".join(systems) if systems else "direct"
        tag_label = f" #{tag}" if tag else ""
        preview = content[:300] if len(content) > 300 else content
        with open(today, "a", encoding="utf-8") as f:
            f.write(f"\n### {datetime.now().strftime('%H:%M:%S')} "
                    f"[Hub/{tags}]{tag_label}\n{preview}\n")


hub = MemoryHub()
