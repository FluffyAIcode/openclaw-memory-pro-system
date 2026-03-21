import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional

logger = logging.getLogger(__name__)


class EncodedMemory(BaseModel):
    """Structured representation of a raw memory."""
    facts: List[str]
    preferences: List[str]
    emotions: List[str]
    causal_links: List[str]
    importance: float
    timestamp: datetime
    raw_text: str

    def to_dict(self) -> dict:
        d = self.model_dump()
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EncodedMemory":
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


class MemoryEncoder:
    """
    Encodes raw text into structured EncodedMemory.

    Currently uses heuristic extraction rules.
    Upgrade path: plug in an LLM (Grok / local model) for deeper
    semantic extraction of facts, preferences, emotions, causal links.
    """

    def __init__(self):
        self._count = 0

    def encode(self, raw_text: str, importance: float = None) -> EncodedMemory:
        if importance is None:
            importance = self._estimate_importance(raw_text)

        encoded = EncodedMemory(
            facts=self._extract_facts(raw_text),
            preferences=self._extract_preferences(raw_text),
            emotions=self._extract_emotions(raw_text),
            causal_links=self._extract_causal(raw_text),
            importance=max(0.0, min(1.0, importance)),
            timestamp=datetime.now(),
            raw_text=raw_text,
        )
        self._count += 1
        logger.info("编码记忆 #%d (重要性: %.2f): %s...",
                     self._count, encoded.importance, raw_text[:50])
        return encoded

    def batch_encode(self, texts: List[str]) -> List[EncodedMemory]:
        return [self.encode(t) for t in texts]

    # ------------------------------------------------------------------
    # Heuristic helpers — intentionally simple; replace with LLM later
    # ------------------------------------------------------------------

    _HIGH_IMP_KW = [
        "重要", "记住", "决定", "关键", "必须", "偏好", "绝不", "核心",
        "important", "remember", "critical", "must", "prefer", "never",
    ]

    def _estimate_importance(self, text: str) -> float:
        score = 0.5
        low = text.lower()
        for kw in self._HIGH_IMP_KW:
            if kw in low:
                score += 0.08
        if len(text) > 200:
            score += 0.05
        if "?" in text or "？" in text:
            score -= 0.05
        return max(0.0, min(1.0, score))

    def _extract_facts(self, text: str) -> List[str]:
        return [text[:120]] if len(text) > 10 else [text]

    def _extract_preferences(self, text: str) -> List[str]:
        for kw in ("喜欢", "偏好", "prefer", "like", "want", "希望"):
            if kw in text.lower():
                return [text[:100]]
        return []

    def _extract_emotions(self, text: str) -> List[str]:
        _MAP = {
            "开心": "positive", "高兴": "positive", "happy": "positive",
            "失望": "negative", "沮丧": "negative", "sad": "negative",
            "焦虑": "anxious", "worried": "anxious",
        }
        return [lbl for kw, lbl in _MAP.items() if kw in text.lower()]

    def _extract_causal(self, text: str) -> List[str]:
        for kw in ("因为", "所以", "导致", "because", "therefore", "caused"):
            if kw in text.lower():
                return [text[:100]]
        return []


encoder = MemoryEncoder()
