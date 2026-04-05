"""Preference Learner — extracts structured preference tags from memory content (F-10).

Patterns cover English and Chinese expressions of likes/dislikes/habits.
Extracted preferences are stored alongside the Chronos encoder output
and can be queried by the Context Composer for suggestion/planning intents.

Persistence: memory/preferences.jsonl
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_WORKSPACE = Path(__file__).resolve().parent.parent
_PREFS_FILE = _WORKSPACE / "memory" / "preferences.jsonl"

PREFERENCE_PATTERNS = [
    (re.compile(r"i (love|like|prefer|enjoy|am into)\s+(.+)", re.IGNORECASE), "positive"),
    (re.compile(r"i (hate|dislike|don't like|can't stand|avoid)\s+(.+)", re.IGNORECASE), "negative"),
    (re.compile(r"my favorite (.+?) is (.+)", re.IGNORECASE), "favorite"),
    (re.compile(r"i always (.+)", re.IGNORECASE), "habit"),
    (re.compile(r"我(喜欢|爱|偏好|习惯|热衷于)\s*(.+)"), "positive"),
    (re.compile(r"我(讨厌|不喜欢|避免|受不了|不想)\s*(.+)"), "negative"),
    (re.compile(r"我最喜欢的(.+?)是(.+)"), "favorite"),
    (re.compile(r"我(总是|一直|每天都)\s*(.+)"), "habit"),
]


def extract_preferences(content: str) -> List[Dict]:
    """Extract structured preferences from free-text content."""
    prefs: List[Dict] = []
    for pattern, ptype in PREFERENCE_PATTERNS:
        match = pattern.search(content)
        if match:
            value = match.group(2).strip().rstrip("。.!！,，")[:120]
            if len(value) < 2:
                continue
            prefs.append({
                "type": ptype,
                "value": value,
                "raw": content[:200],
                "extracted_at": datetime.now().isoformat(),
            })
    return prefs


def store_preferences(prefs: List[Dict]):
    """Append preferences to the JSONL persistence file."""
    if not prefs:
        return
    _PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PREFS_FILE, "a", encoding="utf-8") as f:
        for p in prefs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def get_all_preferences() -> List[Dict]:
    """Load all stored preferences."""
    if not _PREFS_FILE.exists():
        return []
    prefs = []
    for line in _PREFS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                prefs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return prefs


def get_relevant_preferences(query: str, limit: int = 3) -> List[Dict]:
    """Return preferences relevant to the query (keyword overlap)."""
    all_prefs = get_all_preferences()
    if not all_prefs:
        return []

    q_lower = query.lower()
    scored = []
    for p in all_prefs:
        value_lower = p.get("value", "").lower()
        words = set(re.findall(r'\w+', value_lower))
        q_words = set(re.findall(r'\w+', q_lower))
        overlap = len(words & q_words)
        if overlap > 0:
            scored.append((overlap, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:limit]]
