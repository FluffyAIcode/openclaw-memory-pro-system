"""Tests for second_brain v2 — multi-layer pool, 5 strategies, deep_collide."""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

WORKSPACE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE))


# ── Config ────────────────────────────────────────────────────

class TestSecondBrainConfig:
    def setup_method(self):
        import second_brain.config as cmod
        cmod._config_cache = None

    def test_load_config_defaults(self, tmp_path):
        with patch.dict(os.environ, {"SB_BASE_DIR": str(tmp_path)}, clear=False):
            import second_brain.config as cmod
            cmod._config_cache = None
            cfg = cmod.SecondBrainConfig(base_dir=tmp_path)
            assert cfg.vitality_half_life_days == 30.0
            assert cfg.collision_interval_hours == 6.0
            assert cfg.collisions_per_round == 3

    def test_ensure_dirs(self, tmp_path):
        from second_brain.config import SecondBrainConfig
        cfg = SecondBrainConfig(base_dir=tmp_path)
        cfg.ensure_dirs()
        assert (tmp_path / "tracker").is_dir()
        assert (tmp_path / "insights").is_dir()

    def test_validation_half_life(self):
        from second_brain.config import SecondBrainConfig
        with pytest.raises(ValueError, match="positive"):
            SecondBrainConfig(vitality_half_life_days=-1)

    def test_validation_bridge_range(self):
        from second_brain.config import SecondBrainConfig
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            SecondBrainConfig(semantic_bridge_low=1.5)

    def test_load_config_caching(self):
        import second_brain.config as cmod
        cmod._config_cache = None
        cfg1 = cmod.load_config()
        cfg2 = cmod.load_config()
        assert cfg1 is cfg2
        cmod._config_cache = None


# ── Tracker ───────────────────────────────────────────────────

class TestMemoryTracker:
    def _make_tracker(self, tmp_path):
        from second_brain.tracker import MemoryTracker
        return MemoryTracker(tracker_path=tmp_path / "tracker")

    def test_track_creates_log(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.track("mem1", "hash1", "test query")
        log_file = tmp_path / "tracker" / "access_log.jsonl"
        assert log_file.exists()
        records = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        assert len(records) == 1
        assert records[0]["memory_id"] == "mem1"

    def test_vitality_formula(self, tmp_path):
        t = self._make_tracker(tmp_path)
        v_new = t.vitality(importance=0.8, created_at=datetime.now().isoformat())
        assert 0.7 < v_new < 0.9
        v_old = t.vitality(importance=0.8, created_at=(datetime.now() - timedelta(days=90)).isoformat())
        assert v_old < v_new

    def test_vitality_with_hits(self, tmp_path):
        t = self._make_tracker(tmp_path)
        for _ in range(5):
            t.track("", "hash_x", "q")
        v_with = t.vitality(0.8, datetime.now().isoformat(), content_hash="hash_x")
        v_without = t.vitality(0.8, datetime.now().isoformat(), content_hash="hash_none")
        assert v_with > v_without

    def test_find_dormant(self, tmp_path):
        t = self._make_tracker(tmp_path)
        entries = [
            {"content": "important stuff", "timestamp": (datetime.now() - timedelta(days=30)).isoformat(),
             "metadata": {"importance": 0.9}, "importance": 0.9},
            {"content": "minor", "timestamp": datetime.now().isoformat(),
             "metadata": {"importance": 0.3}, "importance": 0.3},
        ]
        dormant = t.find_dormant(entries)
        assert len(dormant) == 1
        assert dormant[0]["content"] == "important stuff"

    def test_find_trends(self, tmp_path):
        t = self._make_tracker(tmp_path)
        t.track("m1", "h1", "alpha")
        t.track("m1", "h1", "beta")
        t.track("m2", "h2", "gamma")
        trends = t.find_trends()
        assert len(trends) >= 1
        assert trends[0]["hits"] >= 2

    def test_stats(self, tmp_path):
        t = self._make_tracker(tmp_path)
        assert t.stats()["total_accesses"] == 0
        t.track("m1", "h1")
        t.track("m2", "h2")
        s = t.stats()
        assert s["total_accesses"] == 2
        assert s["unique_memories"] == 2


# ── Collision Engine (v2 — multi-layer pool) ──────────────────

def _make_pool(memora_count=5, chronos_count=3, digest_count=2, msa_count=1):
    """Build a synthetic memory pool for testing."""
    now = datetime.now()
    pool = {
        "memora_vectors": [
            {"content": f"Memora entry {i}", "timestamp": (now - timedelta(days=i)).isoformat(),
             "importance": 0.5 + i * 0.05, "metadata": {}, "source_system": "memora"}
            for i in range(memora_count)
        ],
        "chronos_encoded": [
            {"content": f"Chronos memory {i}", "timestamp": (now - timedelta(days=i*3)).isoformat(),
             "importance": 0.7 + i * 0.05,
             "facts": [f"fact_{i}_a", f"fact_{i}_b"],
             "preferences": [f"prefer_{i}"] if i % 2 == 0 else [],
             "emotions": ["positive"] if i == 0 else [],
             "causal_links": [f"because_{i}"] if i == 1 else [],
             "source_system": "chronos"}
            for i in range(chronos_count)
        ],
        "digests": [
            {"content": f"# Digest summary {i}\n\nKey decision: chose approach {i}",
             "timestamp": (now - timedelta(days=7*i)).isoformat(),
             "importance": 0.9, "metadata": {"filename": f"digest_{i}.md"},
             "source_system": "digest"}
            for i in range(digest_count)
        ],
        "msa_docs": [
            {"content": f"MSA document {i} about topic {i}",
             "timestamp": (now - timedelta(days=2*i)).isoformat(),
             "importance": 0.7, "doc_id": f"doc_{i}", "chunk_count": 3,
             "metadata": {}, "source_system": "msa"}
            for i in range(msa_count)
        ],
    }
    return pool


class TestCollisionEngineV2:
    def test_parse_novelty(self):
        from second_brain.collision import _parse_novelty
        assert _parse_novelty("## 新颖度\n4") == 4
        assert _parse_novelty("no number here") == 3

    def test_parse_llm_output(self):
        from second_brain.collision import _parse_llm_output
        text = "## 联系发现\n编程相关\n\n## 灵感\n新框架\n\n## 新颖度\n4"
        conn, ideas, nov = _parse_llm_output(text)
        assert "编程" in conn
        assert "框架" in ideas
        assert nov == 4

    def test_insight_to_dict_with_sources(self):
        from second_brain.collision import Insight
        ins = Insight("a", "b", "chronos_crossref", "conn", "idea", 4,
                      source_a="chronos", source_b="memora")
        d = ins.to_dict()
        assert d["source_a"] == "chronos"
        assert d["source_b"] == "memora"

    def test_insight_to_markdown_shows_layer_names(self):
        from second_brain.collision import Insight
        ins = Insight("a", "b", "digest_bridge",
                      source_a="digest", source_b="memora")
        md = ins.to_markdown()
        assert "长期记忆摘要" in md
        assert "Memora向量库" in md

    def test_collide_round_too_few(self, tmp_path):
        from second_brain.collision import CollisionEngine
        eng = CollisionEngine()
        eng._insights_path = tmp_path / "insights"
        eng._insights_path.mkdir()
        pool = _make_pool(memora_count=0, chronos_count=0, digest_count=1, msa_count=0)
        result = eng.collide_round(pool)
        assert result == []

    def test_collide_round_with_pool(self, tmp_path):
        from second_brain.collision import CollisionEngine
        eng = CollisionEngine()
        eng._insights_path = tmp_path / "insights"
        eng._insights_path.mkdir()
        pool = _make_pool()

        mock_llm = MagicMock()
        mock_llm.is_available.return_value = False
        with patch.dict("sys.modules", {"llm_client": mock_llm}):
            with patch.dict("sys.modules", {"memora.vectorstore": MagicMock()}):
                cmod = sys.modules["second_brain.collision"]
                mock_tracker = MagicMock()
                mock_tracker.find_dormant.return_value = []
                old = cmod._tracker
                cmod._tracker = mock_tracker
                try:
                    insights = eng.collide_round(pool)
                finally:
                    cmod._tracker = old

    def test_chronos_crossref_strategy(self):
        from second_brain.collision import CollisionEngine
        eng = CollisionEngine()
        pool = _make_pool(chronos_count=3, memora_count=5)

        mock_vs = MagicMock()
        mock_vs.search.return_value = [
            {"content": "related memora entry", "score": 0.7,
             "timestamp": datetime.now().isoformat()},
        ]
        with patch.dict("sys.modules", {"memora.vectorstore": MagicMock(vector_store=mock_vs)}):
            all_flat = []
            for entries in pool.values():
                all_flat.extend(entries)
            result = eng._chronos_crossref(pool, all_flat)
            if result is not None:
                assert result[0].get("source_system") == "chronos"

    def test_digest_bridge_strategy(self):
        from second_brain.collision import CollisionEngine
        eng = CollisionEngine()
        pool = _make_pool(digest_count=2, memora_count=5)

        all_flat = []
        for entries in pool.values():
            all_flat.extend(entries)
        result = eng._digest_bridge(pool, all_flat)
        assert result is not None
        assert result[0]["source_system"] == "digest"

    def test_digest_bridge_no_digests(self):
        from second_brain.collision import CollisionEngine
        eng = CollisionEngine()
        pool = _make_pool(digest_count=0)
        result = eng._digest_bridge(pool, [])
        assert result is None

    def test_temporal_echo_with_pool(self):
        from second_brain.collision import CollisionEngine
        eng = CollisionEngine()
        now = datetime.now()
        pool = {
            "memora_vectors": [
                {"content": "today", "timestamp": now.isoformat(), "source_system": "memora"},
            ],
            "chronos_encoded": [
                {"content": "a week ago", "timestamp": (now - timedelta(days=7)).isoformat(),
                 "source_system": "chronos"},
            ],
            "digests": [],
            "msa_docs": [],
        }
        all_flat = pool["memora_vectors"] + pool["chronos_encoded"]
        result = eng._temporal_echo(pool, all_flat)
        assert result is not None

    def test_generate_insight_with_structured_data(self):
        from second_brain.collision import CollisionEngine
        eng = CollisionEngine()
        chronos_mem = {
            "content": "用户喜欢Python", "timestamp": "2025-01-01",
            "source_system": "chronos",
            "facts": ["Python是首选语言", "用户在做AI项目"],
            "preferences": ["喜欢简洁代码"],
            "emotions": ["positive"],
            "causal_links": [],
        }
        memora_mem = {
            "content": "今天学习了Rust", "timestamp": "2025-06-01",
            "source_system": "memora",
        }
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = "## 联系发现\nPython到Rust\n\n## 灵感\n用Rust加速\n\n## 新颖度\n5"
        with patch.dict("sys.modules", {"llm_client": mock_llm}):
            insight = eng._generate_insight(chronos_mem, memora_mem, "chronos_crossref")
            assert insight.source_a == "chronos"
            assert insight.source_b == "memora"
            assert insight.novelty == 5
            call_args = mock_llm.generate.call_args
            assert "事实" in call_args.kwargs.get("prompt", "") or "事实" in str(call_args)

    def test_save_insights(self, tmp_path):
        from second_brain.collision import CollisionEngine, Insight
        eng = CollisionEngine()
        eng._insights_path = tmp_path / "insights"
        eng._insights_path.mkdir()
        insights = [Insight("a", "b", "test", "conn", "idea", 4,
                            source_a="chronos", source_b="digest")]
        fp = eng.save_insights(insights)
        content = fp.read_text()
        assert "灵感碰撞" in content

    def test_index_high_novelty_with_cross_layer(self, tmp_path):
        from second_brain.collision import CollisionEngine, Insight
        eng = CollisionEngine()
        eng._insights_path = tmp_path
        ins = Insight("a", "b", "chronos_crossref", "cool", "idea", 5,
                      source_a="chronos", source_b="memora")
        mock_vs = MagicMock()
        with patch.dict("sys.modules",
                        {"memora": MagicMock(), "memora.vectorstore": MagicMock(vector_store=mock_vs)}):
            eng.index_high_novelty([ins])
            mock_vs.add.assert_called_once()
            call_kwargs = mock_vs.add.call_args
            metadata = call_kwargs[1]["metadata"] if "metadata" in call_kwargs[1] else call_kwargs[0][1]
            assert "cross_layers" in metadata or "chronos" in str(call_kwargs)


# ── Bridge v2 ─────────────────────────────────────────────────

def _get_bridge_mod():
    import second_brain.bridge
    return sys.modules["second_brain.bridge"]


class TestSecondBrainBridgeV2:
    def test_build_memory_pool_loads_all_layers(self, tmp_path):
        from second_brain.bridge import SecondBrainBridge
        bmod = _get_bridge_mod()
        br = SecondBrainBridge()

        br._load_memora_vectors = MagicMock(return_value=[{"content": "m1"}])
        br._load_chronos_memories = MagicMock(return_value=[{"content": "c1"}])
        br._load_digests = MagicMock(return_value=[{"content": "d1"}])
        br._load_msa_summaries = MagicMock(return_value=[{"content": "msa1"}])

        pool = br._build_memory_pool()
        assert len(pool["memora_vectors"]) == 1
        assert len(pool["chronos_encoded"]) == 1
        assert len(pool["digests"]) == 1
        assert len(pool["msa_docs"]) == 1

    def test_collide_returns_pool_stats(self):
        from second_brain.bridge import SecondBrainBridge
        bmod = _get_bridge_mod()
        br = SecondBrainBridge()

        pool = {"memora_vectors": [{"content": "a"}], "chronos_encoded": [],
                "digests": [], "msa_docs": []}
        br._build_memory_pool = MagicMock(return_value=pool)
        result = br.collide()
        assert "pool_stats" in result

    def test_collide_with_real_pool(self, tmp_path):
        from second_brain.bridge import SecondBrainBridge
        from second_brain.collision import Insight
        bmod = _get_bridge_mod()
        br = SecondBrainBridge()

        pool = _make_pool()
        br._build_memory_pool = MagicMock(return_value=pool)

        mock_engine = MagicMock()
        mock_engine.collide_round.return_value = [
            Insight("a", "b", "chronos_crossref", "conn", "idea", 4,
                    source_a="chronos", source_b="memora")
        ]
        mock_engine.save_insights.return_value = tmp_path / "insights.md"

        old_engine = bmod.engine
        bmod.engine = mock_engine
        try:
            result = br.collide()
            assert len(result["insights"]) == 1
            assert result["pool_stats"]["chronos_encoded"] == 3
        finally:
            bmod.engine = old_engine

    def test_deep_collide(self):
        from second_brain.bridge import SecondBrainBridge
        br = SecondBrainBridge()
        mock_msa = MagicMock()
        mock_msa.interleave_query.return_value = {
            "final_answer": "Deep insight about connections",
            "rounds": 2,
            "total_docs_used": 3,
            "doc_ids_used": ["d1", "d2", "d3"],
        }
        with patch.dict("sys.modules", {
            "msa.bridge": MagicMock(bridge=mock_msa),
            "memora.vectorstore": MagicMock(),
        }):
            result = br.deep_collide("test topic")
            assert result["answer"] is not None
            assert result["docs_used"] == 3

    def test_deep_collide_no_msa(self):
        from second_brain.bridge import SecondBrainBridge
        br = SecondBrainBridge()
        with patch.dict("sys.modules", {"msa.bridge": None}):
            result = br.deep_collide()
            assert result["answer"] is None

    def test_load_chronos_memories(self):
        from second_brain.bridge import SecondBrainBridge
        br = SecondBrainBridge()

        class FakeMem:
            raw_text = "test memory"
            timestamp = datetime.now()
            importance = 0.8
            facts = ["fact1"]
            preferences = ["pref1"]
            emotions = ["positive"]
            causal_links = ["because1"]

        mock_buf = MagicMock()
        mock_buf._buffer = [FakeMem()]
        with patch.dict("sys.modules", {
            "chronos": MagicMock(),
            "chronos.replay_buffer": MagicMock(replay_buffer=mock_buf),
        }):
            result = br._load_chronos_memories()
            assert len(result) == 1
            assert result[0]["facts"] == ["fact1"]
            assert result[0]["source_system"] == "chronos"

    def test_load_digests(self, tmp_path):
        from second_brain.bridge import SecondBrainBridge
        bmod = _get_bridge_mod()
        br = SecondBrainBridge()

        digest_dir = tmp_path / "memory" / "long_term"
        digest_dir.mkdir(parents=True)
        (digest_dir / "digest_2025-01-01_120000.md").write_text("# Test digest\n\nContent here")

        old_ws = bmod._WORKSPACE
        bmod._WORKSPACE = tmp_path
        try:
            result = br._load_digests()
            assert len(result) == 1
            assert result[0]["source_system"] == "digest"
            assert result[0]["importance"] == 0.9
        finally:
            bmod._WORKSPACE = old_ws

    def test_load_msa_summaries(self, tmp_path):
        from second_brain.bridge import SecondBrainBridge
        bmod = _get_bridge_mod()
        br = SecondBrainBridge()

        msa_dir = tmp_path / "memory" / "msa"
        msa_dir.mkdir(parents=True)
        content_dir = msa_dir / "content"
        content_dir.mkdir()

        routing = {"doc_id": "test_doc", "chunk_count": 3,
                    "metadata": {"title": "Test"}, "ingested_at": "2025-01-01"}
        (msa_dir / "routing_index.jsonl").write_text(json.dumps(routing) + "\n")
        (content_dir / "doc_test_doc.jsonl").write_text(
            json.dumps({"text": "First chunk of the document"}) + "\n"
        )

        old_ws = bmod._WORKSPACE
        bmod._WORKSPACE = tmp_path
        try:
            result = br._load_msa_summaries()
            assert len(result) == 1
            assert result[0]["source_system"] == "msa"
            assert "First chunk" in result[0]["content"]
        finally:
            bmod._WORKSPACE = old_ws

    def test_report_includes_pool_stats(self):
        from second_brain.bridge import SecondBrainBridge
        bmod = _get_bridge_mod()
        br = SecondBrainBridge()

        pool = _make_pool(memora_count=3, chronos_count=2)
        br._build_memory_pool = MagicMock(return_value=pool)

        mock_tracker = MagicMock()
        mock_tracker.find_dormant.return_value = []
        mock_tracker.find_trends.return_value = []
        mock_tracker.stats.return_value = {"total_accesses": 0, "unique_memories": 0,
                                            "earliest": None, "latest": None}
        mock_tracker.vitality.return_value = 0.7

        old = bmod.tracker
        bmod.tracker = mock_tracker
        try:
            result = br.report()
            assert "pool_stats" in result
            assert result["pool_stats"]["memora_vectors"] == 3
            assert result["pool_stats"]["chronos_encoded"] == 2
            assert result["total_memories"] > 0
        finally:
            bmod.tracker = old

    def test_status_v2(self):
        from second_brain.bridge import SecondBrainBridge
        bmod = _get_bridge_mod()
        br = SecondBrainBridge()

        pool = _make_pool()
        br._build_memory_pool = MagicMock(return_value=pool)

        mock_tracker = MagicMock()
        mock_tracker.stats.return_value = {"total_accesses": 5, "unique_memories": 3}
        old = bmod.tracker
        bmod.tracker = mock_tracker
        try:
            result = br.status()
            assert result["version"] == "2.0.0"
            assert "pool" in result
        finally:
            bmod.tracker = old

    def test_track_access(self):
        from second_brain.bridge import SecondBrainBridge
        bmod = _get_bridge_mod()
        br = SecondBrainBridge()
        mock_tracker = MagicMock()
        old = bmod.tracker
        bmod.tracker = mock_tracker
        try:
            br.track_access("id1", "content", "query")
            mock_tracker.track.assert_called_once()
        finally:
            bmod.tracker = old


# ── CLI ───────────────────────────────────────────────────────

class TestSecondBrainCLI:
    def test_cmd_collide(self, capsys):
        from second_brain.cli import cmd_collide
        with patch("second_brain.cli.bridge") as mock_br:
            mock_br.collide.return_value = {
                "insights": [{"strategy": "chronos_crossref", "novelty": 4,
                             "connection": "cool", "ideas": "idea",
                             "source_a": "chronos", "source_b": "memora"}],
                "message": "1 insight",
                "pool_stats": {"memora_vectors": 5, "chronos_encoded": 3},
            }
            cmd_collide(None)
            out = capsys.readouterr().out
            assert "1 insight" in out

    def test_main_no_command(self):
        from second_brain.cli import main
        with patch("sys.argv", ["second-brain"]), pytest.raises(SystemExit):
            main()


# ── Memory CLI integration ────────────────────────────────────

class TestMemoryCLISecondBrain:
    def test_cmd_collide(self, capsys):
        from memory_cli import cmd_collide
        with patch("memory_cli._post") as mock_post:
            mock_post.return_value = {
                "insights": [{"strategy": "digest_bridge", "novelty": 4,
                             "connection": "link", "ideas": "cool"}],
                "message": "1 条灵感",
            }
            cmd_collide(None)
            out = capsys.readouterr().out
            assert "1 条灵感" in out

    def test_cmd_deep_collide(self, capsys):
        from memory_cli import cmd_deep_collide
        mock_args = MagicMock()
        mock_args.topic = "AI creativity"
        with patch("memory_cli._post") as mock_post:
            mock_post.return_value = {
                "answer": "Deep insight about AI and creativity",
                "docs_used": 3,
                "rounds": 2,
            }
            cmd_deep_collide(mock_args)
            out = capsys.readouterr().out
            assert "Deep insight" in out
            assert "3 篇文档" in out

    def test_cmd_deep_collide_no_result(self, capsys):
        from memory_cli import cmd_deep_collide
        mock_args = MagicMock()
        mock_args.topic = ""
        with patch("memory_cli._post") as mock_post:
            mock_post.return_value = {"answer": None, "message": "不可用"}
            cmd_deep_collide(mock_args)
            out = capsys.readouterr().out
            assert "不可用" in out

    def test_cmd_sb_report(self, capsys):
        from memory_cli import cmd_sb_report
        with patch("memory_cli._get") as mock_get:
            mock_get.return_value = {
                "pool_stats": {"memora_vectors": 10, "chronos_encoded": 5,
                              "digests": 2, "msa_docs": 1},
                "total_memories": 18,
                "tracker_stats": {"total_accesses": 10, "unique_memories": 5},
                "avg_vitality": 0.65,
                "vitality_sample_size": 5,
                "dormant_count": 2,
                "dormant_top3": [
                    {"dormant_days": 20, "importance": 0.8, "content": "old",
                     "layer": "chronos"}
                ],
                "trending": [],
                "recent_insights_count": 3,
                "recent_insights": [{"date": "2025-01-01", "insight_count": 2}],
            }
            cmd_sb_report(None)
            out = capsys.readouterr().out
            assert "总访问 10 次" in out

    def test_cmd_sb_status(self, capsys):
        from memory_cli import cmd_sb_status
        with patch("memory_cli._get") as mock_get:
            mock_get.return_value = {"module": "second_brain", "version": "2.0.0",
                                      "pool": {"memora_vectors": 5}}
            cmd_sb_status(None)
            out = capsys.readouterr().out
            assert "second_brain" in out
