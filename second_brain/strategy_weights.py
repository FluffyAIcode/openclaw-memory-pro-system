"""Strategy Weights — adaptive selection of collision strategies based on user feedback.

When a user rates an insight, the strategy that produced it gets its weight adjusted:
  - rating > 3: weight increases
  - rating < 3: weight decreases

Collision strategy selection becomes probabilistic, weighted by these scores.
This is the feedback loop that makes the system learn from the user.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()

_DEFAULT_STRATEGIES = [
    "semantic_bridge",
    "chronos_crossref",
    "digest_bridge",
    "dormant_revival",
    "temporal_echo",
    "contradiction_based",
    "blind_spot_based",
]


class InsightRating:
    __slots__ = ("insight_id", "strategy", "rating", "comment", "timestamp")

    def __init__(self, insight_id: str, strategy: str, rating: int,
                 comment: str = "", timestamp: str = ""):
        self.insight_id = insight_id
        self.strategy = strategy
        self.rating = rating
        self.comment = comment
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "insight_id": self.insight_id,
            "strategy": self.strategy,
            "rating": self.rating,
            "comment": self.comment,
            "timestamp": self.timestamp,
        }


class StrategyWeights:
    """Manages adaptive weights for collision strategies."""

    def __init__(self):
        self._insights_path = config.insights_path
        self._insights_path.mkdir(parents=True, exist_ok=True)
        self._weights_file = self._insights_path / "strategy_weights.json"
        self._ratings_file = self._insights_path / "ratings.jsonl"
        self._weights: Dict[str, float] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self):
        self._weights = {s: 1.0 for s in _DEFAULT_STRATEGIES}
        if self._weights_file.exists():
            try:
                with open(self._weights_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    if k in self._weights:
                        self._weights[k] = float(v)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load strategy weights: %s", e)

    def _save(self):
        with open(self._weights_file, "w", encoding="utf-8") as f:
            json.dump(self._weights, f, indent=2)

    def get_weights(self) -> Dict[str, float]:
        self._ensure_loaded()
        return dict(self._weights)

    def rate_insight(self, insight_id: str, strategy: str,
                     rating: int, comment: str = "") -> dict:
        """Record a rating and update the strategy weight."""
        rating = max(1, min(5, rating))
        self._ensure_loaded()

        record = InsightRating(insight_id, strategy, rating, comment)
        with open(self._ratings_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        if strategy in self._weights:
            delta = 0.1 * (rating - 3)
            self._weights[strategy] = max(0.1, self._weights[strategy] + delta)
            self._save()
            logger.info("Strategy '%s' weight updated: %.2f (rating=%d)",
                        strategy, self._weights[strategy], rating)

        return {
            "strategy": strategy,
            "new_weight": self._weights.get(strategy, 1.0),
            "rating": rating,
        }

    def select_strategies(self, n: int = 5,
                          available: List[str] = None) -> List[str]:
        """Select N strategies with probability proportional to weight."""
        self._ensure_loaded()
        pool = available or list(self._weights.keys())
        weights = [max(self._weights.get(s, 1.0), 0.1) for s in pool]
        total = sum(weights)
        probs = [w / total for w in weights]

        n = min(n, len(pool))
        selected = []
        remaining_pool = list(zip(pool, probs))

        for _ in range(n):
            if not remaining_pool:
                break
            names, ps = zip(*remaining_pool)
            ps_normalized = [p / sum(ps) for p in ps]
            choice = random.choices(names, weights=ps_normalized, k=1)[0]
            selected.append(choice)
            remaining_pool = [(nm, p) for nm, p in remaining_pool if nm != choice]

        return selected

    def stats(self) -> dict:
        self._ensure_loaded()
        ratings = self._load_ratings()
        strategy_ratings: Dict[str, List[int]] = {}
        for r in ratings:
            strategy_ratings.setdefault(r.strategy, []).append(r.rating)

        per_strategy = {}
        for s, rs in strategy_ratings.items():
            per_strategy[s] = {
                "count": len(rs),
                "avg_rating": round(sum(rs) / len(rs), 2),
            }

        return {
            "weights": dict(self._weights),
            "total_ratings": len(ratings),
            "per_strategy": per_strategy,
        }

    def _load_ratings(self) -> List[InsightRating]:
        ratings = []
        if self._ratings_file.exists():
            with open(self._ratings_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            ratings.append(InsightRating(**d))
                        except (json.JSONDecodeError, TypeError):
                            pass
        return ratings


strategy_weights = StrategyWeights()
