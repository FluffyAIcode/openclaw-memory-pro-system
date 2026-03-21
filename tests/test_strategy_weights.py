"""Tests for second_brain.strategy_weights module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from second_brain.strategy_weights import StrategyWeights, InsightRating


@pytest.fixture
def tmp_weights(tmp_path):
    with patch("second_brain.strategy_weights.config") as mock_cfg:
        mock_cfg.insights_path = tmp_path
        sw = StrategyWeights()
        sw._insights_path = tmp_path
        sw._weights_file = tmp_path / "strategy_weights.json"
        sw._ratings_file = tmp_path / "ratings.jsonl"
        return sw


class TestInsightRating:
    def test_create(self):
        r = InsightRating("ins001", "semantic_bridge", 4, "useful")
        assert r.insight_id == "ins001"
        assert r.strategy == "semantic_bridge"
        assert r.rating == 4
        assert r.timestamp != ""

    def test_to_dict(self):
        r = InsightRating("ins001", "semantic_bridge", 4)
        d = r.to_dict()
        assert d["insight_id"] == "ins001"
        assert d["rating"] == 4


class TestStrategyWeights:
    def test_initial_weights(self, tmp_weights):
        w = tmp_weights.get_weights()
        assert "semantic_bridge" in w
        assert "contradiction_based" in w
        assert "blind_spot_based" in w
        assert all(v == 1.0 for v in w.values())

    def test_rate_insight_increases_weight(self, tmp_weights):
        result = tmp_weights.rate_insight("ins001", "semantic_bridge", 5)
        assert result["new_weight"] > 1.0
        w = tmp_weights.get_weights()
        assert w["semantic_bridge"] > 1.0

    def test_rate_insight_decreases_weight(self, tmp_weights):
        result = tmp_weights.rate_insight("ins002", "temporal_echo", 1)
        assert result["new_weight"] < 1.0

    def test_rate_clamps_to_min(self, tmp_weights):
        for _ in range(50):
            tmp_weights.rate_insight("bad", "temporal_echo", 1)
        w = tmp_weights.get_weights()
        assert w["temporal_echo"] >= 0.1

    def test_rate_persists(self, tmp_weights):
        tmp_weights.rate_insight("ins001", "semantic_bridge", 5)
        sw2 = StrategyWeights()
        sw2._weights_file = tmp_weights._weights_file
        sw2._ratings_file = tmp_weights._ratings_file
        sw2._loaded = False
        sw2._ensure_loaded()
        assert sw2.get_weights()["semantic_bridge"] > 1.0

    def test_select_strategies(self, tmp_weights):
        selected = tmp_weights.select_strategies(3)
        assert len(selected) == 3
        assert len(set(selected)) == 3  # no duplicates

    def test_select_strategies_weighted(self, tmp_weights):
        for _ in range(10):
            tmp_weights.rate_insight("good", "contradiction_based", 5)
        for _ in range(10):
            tmp_weights.rate_insight("bad", "temporal_echo", 1)

        counts = {"contradiction_based": 0, "temporal_echo": 0}
        for _ in range(100):
            selected = tmp_weights.select_strategies(3)
            for s in selected:
                if s in counts:
                    counts[s] += 1

        assert counts["contradiction_based"] > counts["temporal_echo"]

    def test_stats(self, tmp_weights):
        tmp_weights.rate_insight("ins1", "semantic_bridge", 4)
        tmp_weights.rate_insight("ins2", "semantic_bridge", 5)
        tmp_weights.rate_insight("ins3", "temporal_echo", 2)
        s = tmp_weights.stats()
        assert s["total_ratings"] == 3
        assert "semantic_bridge" in s["per_strategy"]
        assert s["per_strategy"]["semantic_bridge"]["count"] == 2
        assert s["per_strategy"]["semantic_bridge"]["avg_rating"] == 4.5

    def test_ratings_file_created(self, tmp_weights):
        tmp_weights.rate_insight("ins1", "semantic_bridge", 4, "great insight")
        assert tmp_weights._ratings_file.exists()
        with open(tmp_weights._ratings_file) as f:
            line = json.loads(f.readline())
            assert line["rating"] == 4
            assert line["comment"] == "great insight"

    def test_rating_clamped_to_range(self, tmp_weights):
        result = tmp_weights.rate_insight("x", "semantic_bridge", 10)
        assert result["rating"] == 5
        result = tmp_weights.rate_insight("y", "semantic_bridge", -5)
        assert result["rating"] == 1

    def test_unknown_strategy_no_crash(self, tmp_weights):
        result = tmp_weights.rate_insight("z", "nonexistent_strategy", 4)
        assert result["strategy"] == "nonexistent_strategy"
