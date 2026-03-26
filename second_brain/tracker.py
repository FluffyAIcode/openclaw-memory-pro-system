"""Memory Tracker — vitality scores, dormancy detection, trend analysis."""

import json
import logging
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()


class AccessRecord:
    __slots__ = ("memory_id", "content_hash", "timestamp", "query")

    def __init__(self, memory_id: str, content_hash: str, timestamp: str, query: str = ""):
        self.memory_id = memory_id
        self.content_hash = content_hash
        self.timestamp = timestamp
        self.query = query

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "content_hash": self.content_hash,
            "timestamp": self.timestamp,
            "query": self.query,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AccessRecord":
        return cls(
            memory_id=d["memory_id"],
            content_hash=d.get("content_hash", ""),
            timestamp=d["timestamp"],
            query=d.get("query", ""),
        )


class MemoryTracker:
    """Tracks memory access patterns and computes vitality scores."""

    def __init__(self, tracker_path: Path = None):
        self._path = tracker_path or config.tracker_path
        self._path.mkdir(parents=True, exist_ok=True)
        self._log_file = self._path / "access_log.jsonl"
        self._records: Optional[List[AccessRecord]] = None

    def _load(self) -> List[AccessRecord]:
        if self._records is not None:
            return self._records
        self._records = []
        if self._log_file.exists():
            with open(self._log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._records.append(AccessRecord.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError):
                            pass
        return self._records

    def track(self, memory_id: str, content_hash: str, query: str = ""):
        """Record an access event for a memory entry."""
        if query and self._NOISE_PATTERNS.search(query):
            return
        record = AccessRecord(
            memory_id=memory_id,
            content_hash=content_hash,
            timestamp=datetime.now().isoformat(),
            query=query,
        )
        records = self._load()
        records.append(record)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def vitality(self, importance: float, created_at: str,
                 memory_id: str = "", content_hash: str = "") -> float:
        """Compute vitality score for a memory entry.

        V = importance × decay(age) × activity_boost(hits)
          decay(age) = 0.5 + 0.5 × exp(-age_days / half_life)
          activity_boost(hits) = 1 + 0.3 × ln(hits + 1)
        """
        try:
            created = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            created = datetime.now()

        age_days = max((datetime.now() - created).total_seconds() / 86400, 0)
        half_life = config.vitality_half_life_days

        decay = 0.5 + 0.5 * math.exp(-age_days / half_life)
        hits = self._count_hits(memory_id, content_hash)
        boost = 1 + 0.3 * math.log(hits + 1)

        return round(importance * decay * boost, 4)

    def _count_hits(self, memory_id: str, content_hash: str) -> int:
        records = self._load()
        count = 0
        for r in records:
            if memory_id and r.memory_id == memory_id:
                count += 1
            elif content_hash and r.content_hash == content_hash:
                count += 1
        return count

    def find_dormant(self, entries: List[dict],
                     include_never_accessed: bool = False) -> List[dict]:
        """Find high-importance memories that were accessed before but neglected since.

        Args:
            entries: list of dicts with keys: content, timestamp, metadata, score
            include_never_accessed: if True, also include memories that were
                never accessed but are old enough (age > dormancy_age_days).
                Default False — only returns memories that *were* accessed
                at some point and then went dormant.
        """
        now = datetime.now()
        cutoff = now - timedelta(days=config.dormancy_age_days)
        dormant = []

        for e in entries:
            importance = e.get("importance",
                               e.get("metadata", {}).get("importance",
                                                          e.get("score", 0.5)))
            if importance < config.dormancy_importance_threshold:
                continue

            content_hash = str(hash(e.get("content", "")))
            last_access = self._last_access_time(content_hash)

            if last_access is None:
                if not include_never_accessed:
                    continue
                try:
                    created = datetime.fromisoformat(e.get("timestamp", ""))
                except (ValueError, TypeError):
                    continue
                age = (now - created).days
                if age < config.dormancy_age_days:
                    continue
                dormant.append({
                    **e,
                    "dormant_days": age,
                    "importance": importance,
                    "dormant_reason": "never_accessed",
                })
            elif last_access < cutoff:
                dormant.append({
                    **e,
                    "dormant_days": (now - last_access).days,
                    "importance": importance,
                    "dormant_reason": "stale",
                })

        dormant.sort(key=lambda x: x.get("importance", 0), reverse=True)
        return dormant

    _NOISE_PATTERNS = re.compile(
        r'(HEARTBEAT|heartbeat_ok|Read HEARTBEAT\.md|'
        r'Conversation info \(untrusted|Sender \(untrusted|'
        r'sender_id|message_id.*sender.*timestamp|'
        r'untrusted metadata|'
        r'quantum|量子|qubit)',
        re.IGNORECASE
    )

    def find_trends(self) -> List[dict]:
        """Find memories most frequently accessed in the recent window.

        Filters out system noise: heartbeat queries, Telegram metadata,
        benchmark test queries, and other non-user content.
        """
        records = self._load()
        cutoff = datetime.now() - timedelta(days=config.trend_window_days)

        counts: dict[str, dict] = {}
        for r in records:
            try:
                ts = datetime.fromisoformat(r.timestamp)
            except (ValueError, TypeError):
                continue
            if ts < cutoff:
                continue

            if r.query and self._NOISE_PATTERNS.search(r.query):
                continue

            key = r.content_hash or r.memory_id
            if key not in counts:
                counts[key] = {"id": r.memory_id, "hash": r.content_hash,
                               "hits": 0, "queries": []}
            counts[key]["hits"] += 1
            if r.query and r.query not in counts[key]["queries"]:
                counts[key]["queries"].append(r.query)

        trending = sorted(counts.values(), key=lambda x: x["hits"], reverse=True)
        return trending[:config.trend_top_k]

    def _last_access_time(self, content_hash: str) -> Optional[datetime]:
        records = self._load()
        latest = None
        for r in records:
            if r.content_hash == content_hash:
                try:
                    ts = datetime.fromisoformat(r.timestamp)
                    if latest is None or ts > latest:
                        latest = ts
                except (ValueError, TypeError):
                    pass
        return latest

    def stats(self) -> dict:
        records = self._load()
        if not records:
            return {"total_accesses": 0, "unique_memories": 0, "earliest": None, "latest": None}

        unique = set()
        for r in records:
            unique.add(r.content_hash or r.memory_id)

        return {
            "total_accesses": len(records),
            "unique_memories": len(unique),
            "earliest": records[0].timestamp if records else None,
            "latest": records[-1].timestamp if records else None,
        }


tracker = MemoryTracker()
