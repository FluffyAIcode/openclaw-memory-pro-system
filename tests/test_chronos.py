"""Tests for chronos/* — config, encoder, replay_buffer,
consolidator, system, bridge, trainer, distiller, nebius_client.

Note: EWC and DynamicLoRA tests removed — those modules are deprecated
(still exist on disk but no longer used in the pipeline).
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════
# chronos/config.py
# ═══════════════════════════════════════════════════════════

class TestChronosConfig:

    def test_load_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHRONOS_BASE_DIR", str(tmp_path / "chrono"))
        import chronos.config as cc
        cc._config_cache = None
        cfg = cc.load_config()
        assert cfg.base_dir == (tmp_path / "chrono").resolve()
        assert cfg.buffer_path == cfg.base_dir / "replay_buffer.jsonl"
        assert cfg.state_path == cfg.base_dir / "state"
        cc._config_cache = None

    def test_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHRONOS_BASE_DIR", str(tmp_path / "cc"))
        import chronos.config as cc
        cc._config_cache = None
        c1 = cc.load_config()
        c2 = cc.load_config()
        assert c1 is c2
        cc._config_cache = None

    def test_importance_validator(self):
        from chronos.config import ChronosConfig
        with pytest.raises(Exception):
            ChronosConfig(importance_threshold=-0.1)

    def test_hours_validator(self):
        from chronos.config import ChronosConfig
        with pytest.raises(Exception):
            ChronosConfig(consolidation_interval_hours=0)


# ═══════════════════════════════════════════════════════════
# chronos/encoder.py
# ═══════════════════════════════════════════════════════════

class TestMemoryEncoder:

    def test_encode_basic(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        mem = enc.encode("Hello world", importance=0.7)
        assert mem.raw_text == "Hello world"
        assert mem.importance == 0.7
        assert isinstance(mem.timestamp, datetime)

    def test_auto_importance(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        mem = enc.encode("重要的决定需要记住")
        assert mem.importance > 0.5

    def test_importance_clamped(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        mem = enc.encode("text", importance=1.5)
        assert mem.importance == 1.0
        mem2 = enc.encode("text", importance=-0.5)
        assert mem2.importance == 0.0

    def test_extract_facts(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        facts = enc._extract_facts("A short text")
        assert len(facts) == 1

    def test_extract_preferences(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        assert enc._extract_preferences("I prefer Python") != []
        assert enc._extract_preferences("Nothing here") == []

    def test_extract_emotions(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        assert "positive" in enc._extract_emotions("我很开心")
        assert enc._extract_emotions("neutral text") == []

    def test_extract_causal(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        assert enc._extract_causal("因为下雨所以带伞") != []
        assert enc._extract_causal("no cause") == []

    def test_batch_encode(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        results = enc.batch_encode(["a", "b"])
        assert len(results) == 2

    def test_encoded_memory_serialization(self):
        from chronos.encoder import MemoryEncoder, EncodedMemory
        enc = MemoryEncoder()
        mem = enc.encode("test", importance=0.8)
        d = mem.to_dict()
        assert isinstance(d["timestamp"], str)
        restored = EncodedMemory.from_dict(d)
        assert restored.raw_text == "test"
        assert restored.importance == 0.8

    def test_question_lowers_importance(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        q = enc.encode("这是什么？")
        s = enc.encode("这是声明。")
        assert q.importance <= s.importance

    def test_long_text_boosts_importance(self):
        from chronos.encoder import MemoryEncoder
        enc = MemoryEncoder()
        short = enc.encode("hi")
        long_ = enc.encode("a " * 150)
        assert long_.importance >= short.importance


# ═══════════════════════════════════════════════════════════
# chronos/replay_buffer.py
# ═══════════════════════════════════════════════════════════

class TestReplayBuffer:

    def _make_memory(self, text="t", imp=0.8):
        from chronos.encoder import MemoryEncoder
        return MemoryEncoder().encode(text, importance=imp)

    def test_add_and_size(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHRONOS_BASE_DIR", str(tmp_path))
        import chronos.config as cc
        cc._config_cache = None
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer()
        rb._buffer_path = tmp_path / "rb.jsonl"
        rb._buffer = []
        rb.add(self._make_memory(imp=0.9))
        assert rb.size() >= 1
        cc._config_cache = None

    def test_low_importance_sometimes_discarded(self, tmp_path):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._max_size = 100
        rb._importance_threshold = 0.9
        rb._buffer_path = None
        rb._buffer = []
        added = 0
        for _ in range(100):
            rb.add(self._make_memory(imp=0.1))
            added = rb.size()
        assert added < 100

    def test_sample_empty(self):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._buffer = []
        assert rb.sample() == []

    def test_sample_fewer_than_batch(self):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._buffer = [self._make_memory()]
        result = rb.sample(batch_size=5)
        assert len(result) == 1

    def test_sample_weighted(self):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._buffer = [self._make_memory(imp=0.9) for _ in range(20)]
        result = rb.sample(batch_size=5)
        assert len(result) == 5

    def test_sample_zero_weights(self):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._buffer = [self._make_memory(imp=0.0) for _ in range(5)]
        result = rb.sample(batch_size=3)
        assert len(result) == 3

    def test_eviction(self):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._max_size = 3
        rb._importance_threshold = 0.0
        rb._buffer_path = None
        rb._buffer = []
        for i in range(5):
            rb.add(self._make_memory(imp=0.1 * i))
        assert rb.size() <= 3

    def test_get_important_memories(self):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._buffer = [self._make_memory(imp=0.5), self._make_memory(imp=0.95)]
        important = rb.get_important_memories(threshold=0.9)
        assert len(important) == 1

    def test_stats_empty(self):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._buffer = []
        s = rb.stats()
        assert s["total"] == 0

    def test_stats_with_data(self):
        from chronos.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._buffer = [self._make_memory(imp=0.5), self._make_memory(imp=0.9)]
        s = rb.stats()
        assert s["total"] == 2
        assert s["max_importance"] >= s["min_importance"]

    def test_persistence_load(self, tmp_path):
        from chronos.replay_buffer import ReplayBuffer
        mem = self._make_memory(imp=0.8)
        buf_file = tmp_path / "buf.jsonl"
        buf_file.write_text(json.dumps(mem.to_dict()) + "\n")
        rb = ReplayBuffer.__new__(ReplayBuffer)
        rb._buffer_path = buf_file
        rb._buffer = []
        rb._max_size = 100
        rb._importance_threshold = 0.0
        rb._load()
        assert rb.size() == 1


# ═══════════════════════════════════════════════════════════
# chronos/consolidator.py (refactored — no EWC/LoRA)
# ═══════════════════════════════════════════════════════════

class TestConsolidator:

    def _make_memory(self, imp=0.9):
        from chronos.encoder import MemoryEncoder
        return MemoryEncoder().encode("test memory content", importance=imp)

    def test_should_consolidate_first_time(self):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._last = None
        mc._interval = timedelta(hours=6)
        assert mc.should_consolidate() is True

    def test_should_consolidate_interval_not_passed(self):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._last = datetime.now()
        mc._interval = timedelta(hours=6)
        assert mc.should_consolidate() is False

    def test_consolidate_skipped(self):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._last = datetime.now()
        mc._interval = timedelta(hours=6)
        mc._state_file = None
        result = mc.consolidate(force=False)
        assert result.get("skipped") is True

    def _get_cmod(self):
        return sys.modules["chronos.consolidator"]

    def test_consolidate_no_important(self, tmp_path):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._last = None
        mc._interval = timedelta(hours=6)
        mc._state_file = None
        mc._count = 0
        cmod = self._get_cmod()
        orig_rb = cmod.replay_buffer
        mock_rb = MagicMock()
        mock_rb.get_important_memories.return_value = []
        cmod.replay_buffer = mock_rb
        try:
            result = mc.consolidate(force=True)
        finally:
            cmod.replay_buffer = orig_rb
        assert result["consolidated"] == 0

    def test_consolidate_with_personality(self, tmp_path):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._last = None
        mc._interval = timedelta(hours=6)
        mc._state_file = tmp_path / "cons.json"
        mc._count = 0
        memories = [self._make_memory(imp=0.95)]

        cmod = self._get_cmod()
        orig_rb = cmod.replay_buffer
        mock_rb = MagicMock()
        mock_rb.get_important_memories.return_value = memories
        cmod.replay_buffer = mock_rb
        try:
            with patch("llm_client.is_available", return_value=True), \
                 patch("llm_client.generate", return_value="core_beliefs:\n  - test"):
                result = mc.consolidate(force=True)
        finally:
            cmod.replay_buffer = orig_rb
        assert result["consolidated"] == 1
        assert result["personality_profile_updated"] is True

    def test_personality_no_llm(self):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        memories = [self._make_memory()]
        with patch("llm_client.is_available", return_value=False):
            result = mc._generate_personality_profile(memories)
        assert result is False

    def test_personality_import_error(self):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        memories = [self._make_memory()]
        with patch.dict(sys.modules, {"llm_client": None}):
            result = mc._generate_personality_profile(memories)
        assert result is False

    def test_personality_llm_returns_none(self):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        memories = [self._make_memory()]
        with patch("llm_client.is_available", return_value=True), \
             patch("llm_client.generate", return_value=None):
            result = mc._generate_personality_profile(memories)
        assert result is False

    def test_personality_strips_code_fences(self, tmp_path):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        memories = [self._make_memory()]
        yaml_with_fences = "```yaml\ncore_beliefs:\n  - test\n```"
        with patch("llm_client.is_available", return_value=True), \
             patch("llm_client.generate", return_value=yaml_with_fences):
            result = mc._generate_personality_profile(memories)
        assert result is True
        profile = Path(__file__).parent.parent / "PERSONALITY.yaml"
        if profile.exists():
            content = profile.read_text()
            assert "```" not in content.split("\n\n", 1)[-1]

    def test_report(self):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._last = datetime.now()
        mc._count = 3
        cmod = self._get_cmod()
        orig_rb = cmod.replay_buffer
        cmod.replay_buffer = MagicMock()
        cmod.replay_buffer.stats.return_value = {"total": 5}
        try:
            rpt = mc.report()
        finally:
            cmod.replay_buffer = orig_rb
        assert rpt["consolidation_count"] == 3

    def test_save_and_load_last(self, tmp_path):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._state_file = tmp_path / "last.json"
        mc._last = datetime(2026, 1, 1, 12, 0)
        mc._count = 5
        mc._save_last()
        mc2 = MemoryConsolidator.__new__(MemoryConsolidator)
        mc2._state_file = tmp_path / "last.json"
        loaded = mc2._load_last()
        assert loaded.year == 2026

    def test_load_last_missing(self, tmp_path):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._state_file = tmp_path / "nonexist.json"
        loaded = mc._load_last()
        assert loaded is None

    def test_load_last_corrupt(self, tmp_path):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._state_file = tmp_path / "bad.json"
        mc._state_file.write_text("{bad json")
        loaded = mc._load_last()
        assert loaded is None

    def test_save_last_no_file(self):
        from chronos.consolidator import MemoryConsolidator
        mc = MemoryConsolidator.__new__(MemoryConsolidator)
        mc._state_file = None
        mc._last = None
        mc._count = 0
        mc._save_last()


# ═══════════════════════════════════════════════════════════
# chronos/system.py (refactored — no EWC/LoRA)
# ═══════════════════════════════════════════════════════════

class TestChronosSystem:

    def test_initialize(self):
        from chronos.system import ChronosSystem
        cs = ChronosSystem.__new__(ChronosSystem)
        cs._initialized = False
        cs._learn_count = 0
        cs.initialize()
        assert cs._initialized is True
        cs.initialize()  # idempotent

    def test_learn(self):
        import chronos.system as smod
        cs = smod.ChronosSystem.__new__(smod.ChronosSystem)
        cs._initialized = False
        cs._learn_count = 0
        orig = (smod.encoder, smod.replay_buffer, smod.consolidator)
        try:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = MagicMock(importance=0.8)
            smod.encoder = mock_enc
            mock_rb = MagicMock()
            smod.replay_buffer = mock_rb
            mock_cons = MagicMock()
            mock_cons.should_consolidate.return_value = False
            smod.consolidator = mock_cons
            cs.learn("text")
            assert cs._learn_count == 1
        finally:
            smod.encoder, smod.replay_buffer, smod.consolidator = orig

    def test_learn_triggers_consolidation(self):
        import chronos.system as smod
        cs = smod.ChronosSystem.__new__(smod.ChronosSystem)
        cs._initialized = True
        cs._learn_count = 0
        orig = (smod.encoder, smod.replay_buffer, smod.consolidator)
        try:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = MagicMock(importance=0.9)
            smod.encoder = mock_enc
            smod.replay_buffer = MagicMock()
            mock_cons = MagicMock()
            mock_cons.should_consolidate.return_value = True
            smod.consolidator = mock_cons
            cs.learn("important")
            mock_cons.consolidate.assert_called_once()
        finally:
            smod.encoder, smod.replay_buffer, smod.consolidator = orig

    def test_status(self):
        import importlib
        import chronos.system as smod
        cs = smod.ChronosSystem.__new__(smod.ChronosSystem)
        cs._initialized = True
        cs._learn_count = 5
        orig_rb = smod.replay_buffer
        try:
            mock_rb = MagicMock()
            mock_rb.size.return_value = 10
            smod.replay_buffer = mock_rb
            nbmod = importlib.import_module("chronos.nebius_client")
            orig_nb = nbmod.nebius_client
            mock_nb = MagicMock()
            mock_nb.status.return_value = {"configured": False}
            nbmod.nebius_client = mock_nb
            try:
                s = cs.status()
            finally:
                nbmod.nebius_client = orig_nb
        finally:
            smod.replay_buffer = orig_rb
        assert s["learn_count"] == 5
        assert s["buffer_size"] == 10

    def test_consolidate_delegates(self):
        import chronos.system as smod
        cs = smod.ChronosSystem.__new__(smod.ChronosSystem)
        orig = smod.consolidator
        smod.consolidator = MagicMock()
        smod.consolidator.consolidate.return_value = {"ok": True}
        try:
            assert cs.consolidate(force=True)["ok"] is True
        finally:
            smod.consolidator = orig

    def test_report_delegates(self):
        import chronos.system as smod
        cs = smod.ChronosSystem.__new__(smod.ChronosSystem)
        orig = smod.consolidator
        smod.consolidator = MagicMock()
        smod.consolidator.report.return_value = {"rpt": True}
        try:
            assert cs.report()["rpt"] is True
        finally:
            smod.consolidator = orig

    def test_export_training_data(self):
        import importlib
        import chronos.system as smod
        cs = smod.ChronosSystem.__new__(smod.ChronosSystem)
        dmod = importlib.import_module("chronos.distiller")
        orig = dmod.distiller
        mock_d = MagicMock()
        mock_d.prepare_merged.return_value = Path("/tmp/merged.jsonl")
        dmod.distiller = mock_d
        try:
            result = cs.export_training_data()
            assert "dataset_path" in result
        finally:
            dmod.distiller = orig


# ═══════════════════════════════════════════════════════════
# chronos/bridge.py
# ═══════════════════════════════════════════════════════════

class TestChronosBridge:

    def test_learn_and_save_with_daily(self, tmp_path):
        import chronos.bridge as bmod
        orig = bmod.chronos
        bmod.chronos = MagicMock()
        bmod.chronos.learn.return_value = MagicMock(importance=0.8)
        try:
            from chronos.bridge import ChronosBridge
            b = ChronosBridge(memory_dir=tmp_path)
            result = b.learn_and_save("content", write_daily=True)
            daily = list(tmp_path.glob("*.md"))
            assert len(daily) == 1
        finally:
            bmod.chronos = orig

    def test_learn_and_save_no_daily(self, tmp_path):
        import chronos.bridge as bmod
        orig = bmod.chronos
        bmod.chronos = MagicMock()
        bmod.chronos.learn.return_value = MagicMock(importance=0.8)
        try:
            from chronos.bridge import ChronosBridge
            b = ChronosBridge(memory_dir=tmp_path)
            result = b.learn_and_save("content", write_daily=False)
            daily = list(tmp_path.glob("*.md"))
            assert len(daily) == 0
        finally:
            bmod.chronos = orig

    def test_consolidate(self):
        import chronos.bridge as bmod
        orig = bmod.chronos
        bmod.chronos = MagicMock()
        bmod.chronos.consolidate.return_value = {"consolidated": 5}
        try:
            from chronos.bridge import ChronosBridge
            b = ChronosBridge()
            assert b.consolidate()["consolidated"] == 5
        finally:
            bmod.chronos = orig

    def test_status(self):
        import chronos.bridge as bmod
        orig = bmod.chronos
        bmod.chronos = MagicMock()
        bmod.chronos.status.return_value = {"ok": True}
        try:
            from chronos.bridge import ChronosBridge
            assert ChronosBridge().status()["ok"] is True
        finally:
            bmod.chronos = orig

    def test_report(self):
        import chronos.bridge as bmod
        orig = bmod.chronos
        bmod.chronos = MagicMock()
        bmod.chronos.report.return_value = {"report": True}
        try:
            from chronos.bridge import ChronosBridge
            assert ChronosBridge().report()["report"] is True
        finally:
            bmod.chronos = orig

    def test_export_training_data(self):
        import chronos.bridge as bmod
        orig = bmod.chronos
        bmod.chronos = MagicMock()
        bmod.chronos.export_training_data.return_value = {"dataset_path": "/tmp/x.jsonl"}
        try:
            from chronos.bridge import ChronosBridge
            assert "dataset_path" in ChronosBridge().export_training_data()
        finally:
            bmod.chronos = orig


# ═══════════════════════════════════════════════════════════
# chronos/trainer.py
# ═══════════════════════════════════════════════════════════

class TestChronosTrainer:

    def test_train_step(self):
        import chronos.trainer as tmod
        t = tmod.ChronosTrainer.__new__(tmod.ChronosTrainer)
        t._epoch = 0
        orig_c, orig_cons = tmod.chronos, tmod.consolidator
        tmod.chronos = MagicMock()
        tmod.chronos.learn.return_value = MagicMock()
        tmod.consolidator = MagicMock()
        try:
            t.train_step("text")
            assert t._epoch == 1
        finally:
            tmod.chronos, tmod.consolidator = orig_c, orig_cons

    def test_train_triggers_consolidation_every_5(self):
        import chronos.trainer as tmod
        t = tmod.ChronosTrainer.__new__(tmod.ChronosTrainer)
        t._epoch = 4
        orig_c, orig_cons = tmod.chronos, tmod.consolidator
        tmod.chronos = MagicMock()
        tmod.chronos.learn.return_value = MagicMock()
        mock_cons = MagicMock()
        tmod.consolidator = mock_cons
        try:
            t.train_step("text")
            mock_cons.consolidate.assert_called_once()
        finally:
            tmod.chronos, tmod.consolidator = orig_c, orig_cons

    def test_train_batch(self):
        import chronos.trainer as tmod
        t = tmod.ChronosTrainer.__new__(tmod.ChronosTrainer)
        t._epoch = 0
        orig_c, orig_cons = tmod.chronos, tmod.consolidator
        tmod.chronos = MagicMock()
        tmod.chronos.learn.return_value = MagicMock()
        tmod.consolidator = MagicMock()
        try:
            t.train(["a", "b"], epochs=1)
            assert tmod.chronos.learn.call_count == 2
        finally:
            tmod.chronos, tmod.consolidator = orig_c, orig_cons


# ═══════════════════════════════════════════════════════════
# chronos/distiller.py (new)
# ═══════════════════════════════════════════════════════════

class TestTrainingDistiller:

    def test_prepare_from_digests_empty(self, tmp_path):
        from chronos.distiller import TrainingDistiller
        d = TrainingDistiller(output_dir=tmp_path / "train")
        path = d.prepare_from_digests(digest_dir=tmp_path / "nonexist")
        assert path.parent.exists()

    def test_prepare_from_digests_with_files(self, tmp_path):
        from chronos.distiller import TrainingDistiller
        digest_dir = tmp_path / "digests"
        digest_dir.mkdir()
        (digest_dir / "digest_2026-03-21.md").write_text("# Test digest\nContent here.")
        d = TrainingDistiller(output_dir=tmp_path / "train")
        path = d.prepare_from_digests(digest_dir=digest_dir)
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["source"] == "second_brain_digest"

    def test_prepare_from_buffer(self, tmp_path):
        from chronos.distiller import TrainingDistiller
        buf = tmp_path / "buf.jsonl"
        mem = {
            "facts": ["fact1"], "preferences": [], "emotions": [],
            "causal_links": [], "importance": 0.8,
            "timestamp": "2026-03-21T12:00:00", "raw_text": "test content"
        }
        buf.write_text(json.dumps(mem) + "\n")
        d = TrainingDistiller(output_dir=tmp_path / "train")
        path = d.prepare_from_buffer(buffer_path=buf)
        assert path.exists()
        row = json.loads(path.read_text().strip())
        assert row["source"] == "chronos_replay"

    def test_encoded_memory_to_row(self):
        from chronos.distiller import TrainingDistiller
        m = {"facts": ["f"], "preferences": ["p"], "emotions": [],
             "causal_links": [], "importance": 0.7,
             "timestamp": "2026-01-01T00:00:00", "raw_text": "hello"}
        row = TrainingDistiller._encoded_memory_to_row(m)
        assert row["source"] == "chronos_replay"
        assert "[facts]" in row["input"]


# ═══════════════════════════════════════════════════════════
# chronos/nebius_client.py (new)
# ═══════════════════════════════════════════════════════════

class TestNebiusClient:

    def test_not_configured_by_default(self):
        from chronos.nebius_client import NebiusClient
        nc = NebiusClient()
        assert nc.is_configured is False

    def test_status(self):
        from chronos.nebius_client import NebiusClient
        nc = NebiusClient()
        s = nc.status()
        assert "configured" in s
        assert s["configured"] is False

    def test_upload_raises_not_configured(self, tmp_path):
        from chronos.nebius_client import NebiusClient
        nc = NebiusClient()
        with pytest.raises(RuntimeError, match="not set"):
            nc.upload_dataset(tmp_path / "data.jsonl")

    def test_create_job_raises_not_configured(self):
        from chronos.nebius_client import NebiusClient
        nc = NebiusClient()
        with pytest.raises(RuntimeError, match="not set"):
            nc.create_job("ds-123")
