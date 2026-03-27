"""Tests for CLI entry points — memora/cli.py, chronos/cli.py, msa/cli.py,
memory_hub_cli.py, and __main__.py files."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ═══════════════════════════════════════════════════════════
# memora/__main__.py & memora/cli.py
# ═══════════════════════════════════════════════════════════

class TestMemoraCLI:

    def test_no_command_shows_help(self, capsys):
        with patch("sys.argv", ["memora"]):
            from memora.cli import main
            main()

    def test_add_command(self):
        with patch("sys.argv", ["memora", "add", "test", "content", "-i", "0.8"]), \
             patch("memory_hub.hub") as mh, \
             patch("memora.cli.console"):
            mh.remember.return_value = {"word_count": 2, "systems_used": ["memora"]}
            from memora.cli import main
            main()
            mh.remember.assert_called_once()

    def test_search_no_results(self):
        with patch("sys.argv", ["memora", "search", "nothing"]), \
             patch("memora.vectorstore.vector_store") as mvs, \
             patch("memora.cli.console"):
            mvs.search.return_value = []
            from memora.cli import main
            main()

    def test_search_with_results(self):
        with patch("sys.argv", ["memora", "search", "python"]), \
             patch("memora.vectorstore.vector_store") as mvs, \
             patch("memora.cli.console"):
            mvs.search.return_value = [
                {"score": 0.9, "content": "Python is great"}
            ]
            from memora.cli import main
            main()

    def test_digest_command(self):
        with patch("sys.argv", ["memora", "digest", "--days", "3"]), \
             patch("second_brain.digest.digest_memories") as md, \
             patch("memora.cli.console"):
            from memora.cli import main
            main()
            md.assert_called_once()

    def test_status_command(self):
        with patch("sys.argv", ["memora", "status"]), \
             patch("memora.cli.console"), \
             patch("memora.vectorstore.vector_store") as mvs:
            mvs.count.return_value = 10
            from memora.cli import main
            main()

    def test_init_command(self):
        with patch("sys.argv", ["memora", "init"]), \
             patch("memora.cli.config") as cfg, \
             patch("memora.cli.console"):
            from memora.cli import main
            main()
            cfg.ensure_dirs.assert_called_once()


# ═══════════════════════════════════════════════════════════
# chronos/cli.py
# ═══════════════════════════════════════════════════════════

class TestChronosCLI:

    def test_no_command(self):
        with patch("sys.argv", ["chronos"]):
            from chronos.cli import main
            main()

    def test_learn_command(self):
        import chronos.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.learn_and_save.return_value = MagicMock(importance=0.8)
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["chronos", "learn", "test", "content"]), \
                 patch("chronos.cli.console"):
                cmod.main()
                mock_bridge.learn_and_save.assert_called_once()
        finally:
            cmod.bridge = orig_bridge

    def test_consolidate_command(self):
        import chronos.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.consolidate.return_value = {"consolidated": 5}
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["chronos", "consolidate"]), \
                 patch("chronos.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_status_command(self):
        import chronos.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.status.return_value = {
            "buffer_size": 10, "learn_count": 5,
            "nebius": {"configured": False}
        }
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["chronos", "status"]), \
                 patch("chronos.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_report_command(self):
        import chronos.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.report.return_value = {"report": True}
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["chronos", "report"]), \
                 patch("chronos.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_init_command(self):
        with patch("sys.argv", ["chronos", "init"]), \
             patch("chronos.cli.load_config") as lc, \
             patch("chronos.cli.console"):
            mock_cfg = MagicMock()
            lc.return_value = mock_cfg
            from chronos.cli import main
            main()
            mock_cfg.ensure_dirs.assert_called_once()


# ═══════════════════════════════════════════════════════════
# msa/cli.py
# ═══════════════════════════════════════════════════════════

class TestMSACLI:

    def test_no_command(self):
        with patch("sys.argv", ["msa"]):
            from msa.cli import main
            main()

    def test_ingest_text(self):
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.ingest_and_save.return_value = {"doc_id": "d1", "chunks": 3}
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["msa", "ingest", "some", "text"]), \
                 patch("msa.cli.console"):
                cmod.main()
                mock_bridge.ingest_and_save.assert_called_once()
        finally:
            cmod.bridge = orig_bridge

    def test_ingest_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("file content here")
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.ingest_and_save.return_value = {"doc_id": "d2", "chunks": 1}
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["msa", "ingest", "-f", str(f)]), \
                 patch("msa.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_ingest_no_content(self):
        with patch("sys.argv", ["msa", "ingest"]), \
             patch("msa.cli.console"):
            with pytest.raises(SystemExit):
                from msa.cli import main
                main()

    def test_query_command(self):
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.query_memory.return_value = {
            "total_results": 1, "total_documents": 2,
            "results": [{"rank": 1, "title": "T", "score": 0.9,
                         "chunk_count": 2, "chunks": ["c1", "c2"]}]
        }
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["msa", "query", "test"]), \
                 patch("msa.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_interleave_command(self):
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.interleave_query.return_value = {
            "rounds": 2, "total_docs_used": 3,
            "doc_ids_used": ["d1", "d2"], "final_answer": "answer"
        }
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["msa", "interleave", "complex", "question"]), \
                 patch("msa.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_remove_found(self):
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        cmod.bridge = MagicMock()
        cmod.bridge.remove.return_value = True
        try:
            with patch("sys.argv", ["msa", "remove", "d1"]), \
                 patch("msa.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_remove_not_found(self):
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        cmod.bridge = MagicMock()
        cmod.bridge.remove.return_value = False
        try:
            with patch("sys.argv", ["msa", "remove", "d1"]), \
                 patch("msa.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_status_command(self):
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        cmod.bridge = MagicMock()
        cmod.bridge.status.return_value = {
            "document_count": 5, "total_chunks": 20, "ingest_count": 10,
            "top_k": 5, "chunk_size": 512, "similarity_threshold": 0.3,
            "max_interleave_rounds": 3
        }
        try:
            with patch("sys.argv", ["msa", "status"]), \
                 patch("msa.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_report_command(self):
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        cmod.bridge = MagicMock()
        cmod.bridge.report.return_value = {"report": True}
        try:
            with patch("sys.argv", ["msa", "report"]), \
                 patch("msa.cli.console"):
                cmod.main()
        finally:
            cmod.bridge = orig_bridge

    def test_init_command(self):
        with patch("sys.argv", ["msa", "init"]), \
             patch("msa.cli.load_config") as lc, \
             patch("msa.cli.console"):
            mock_cfg = MagicMock()
            lc.return_value = mock_cfg
            from msa.cli import main
            main()
            mock_cfg.ensure_dirs.assert_called_once()

    def test_ingest_with_title(self):
        import msa.cli as cmod
        orig_bridge = cmod.bridge
        mock_bridge = MagicMock()
        mock_bridge.ingest_and_save.return_value = {"doc_id": "d5", "chunks": 1}
        cmod.bridge = mock_bridge
        try:
            with patch("sys.argv", ["msa", "ingest", "text", "-t", "My Title"]), \
                 patch("msa.cli.console"):
                cmod.main()
                call_kwargs = mock_bridge.ingest_and_save.call_args
                assert call_kwargs.kwargs.get("metadata", {}).get("title") == "My Title" or \
                       "title" in str(call_kwargs)
        finally:
            cmod.bridge = orig_bridge


# ═══════════════════════════════════════════════════════════
# memory_hub_cli.py
# ═══════════════════════════════════════════════════════════

class TestMemoryHubCLI:

    def test_no_command(self):
        with patch("sys.argv", ["memory-hub"]):
            from memory_hub_cli import main
            main()

    def test_remember_text(self):
        with patch("sys.argv", ["memory-hub", "remember", "hello", "world"]), \
             patch("memory_hub_cli.hub") as h, \
             patch("memory_hub_cli.console"):
            h.remember.return_value = {
                "word_count": 2, "systems_used": ["memora"],
            }
            from memory_hub_cli import main
            main()
            h.remember.assert_called_once()

    def test_remember_file(self, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("file content")
        with patch("sys.argv", ["memory-hub", "remember", "-f", str(f)]), \
             patch("memory_hub_cli.hub") as h, \
             patch("memory_hub_cli.console"):
            h.remember.return_value = {
                "word_count": 2, "systems_used": ["memora"],
            }
            from memory_hub_cli import main
            main()

    def test_remember_msa_result(self):
        with patch("sys.argv", ["memory-hub", "remember", "text"]), \
             patch("memory_hub_cli.hub") as h, \
             patch("memory_hub_cli.console"):
            h.remember.return_value = {
                "word_count": 1, "systems_used": ["msa"],
                "msa": {"doc_id": "d1", "chunks": 2}
            }
            from memory_hub_cli import main
            main()

    def test_remember_no_content(self):
        with patch("sys.argv", ["memory-hub", "remember"]), \
             patch("memory_hub_cli.console"):
            with pytest.raises(SystemExit):
                from memory_hub_cli import main
                main()

    def test_remember_with_systems(self):
        with patch("sys.argv", ["memory-hub", "remember", "text", "--systems", "memora,msa"]), \
             patch("memory_hub_cli.hub") as h, \
             patch("memory_hub_cli.console"):
            h.remember.return_value = {"word_count": 1, "systems_used": ["memora", "msa"]}
            from memory_hub_cli import main
            main()

    def test_recall_command(self):
        with patch("sys.argv", ["memory-hub", "recall", "query"]), \
             patch("memory_hub_cli.hub") as h, \
             patch("memory_hub_cli.console"):
            h.recall.return_value = {
                "memora": [{"content": "x", "score": 0.8, "metadata": {"title": "T"}}],
                "msa": [],
                "merged": [{"content": "x", "score": 0.8, "system": "memora",
                            "metadata": {"title": "T"}}],
            }
            from memory_hub_cli import main
            main()

    def test_deep_recall_with_interleave(self):
        with patch("sys.argv", ["memory-hub", "deep-recall", "complex", "question"]), \
             patch("memory_hub_cli.hub") as h, \
             patch("memory_hub_cli.console"):
            h.deep_recall.return_value = {
                "interleave": {"rounds": 2, "total_docs_used": 3,
                               "doc_ids_used": ["d1"], "final_answer": "answer"},
                "memora_context": [{"content": "ctx"}],
            }
            from memory_hub_cli import main
            main()

    def test_deep_recall_no_interleave(self):
        with patch("sys.argv", ["memory-hub", "deep-recall", "q"]), \
             patch("memory_hub_cli.hub") as h, \
             patch("memory_hub_cli.console"):
            h.deep_recall.return_value = {
                "interleave": None,
                "memora_context": [],
            }
            from memory_hub_cli import main
            main()

    def test_status_command(self):
        with patch("sys.argv", ["memory-hub", "status"]), \
             patch("memory_hub_cli.hub") as h, \
             patch("memory_hub_cli.console"):
            h.status.return_value = {
                "systems": {
                    "memora": {"entries": 5},
                    "chronos": {"error": "not available"},
                }
            }
            from memory_hub_cli import main
            main()


# ═══════════════════════════════════════════════════════════
# memory_cli.py remaining branches
# ═══════════════════════════════════════════════════════════

class TestMemoryCliMain:

    def test_main_no_args(self):
        with patch("sys.argv", ["memory-cli"]):
            from memory_cli import main
            with pytest.raises(SystemExit):
                main()

    def test_main_remember(self):
        with patch("sys.argv", ["memory-cli", "remember", "hello"]), \
             patch("memory_cli._post") as mp:
            mp.return_value = {"ok": True, "data": {"word_count": 1, "systems_used": ["memora"]}}
            from memory_cli import main
            main()

    def test_main_recall(self):
        with patch("sys.argv", ["memory-cli", "recall", "query"]), \
             patch("memory_cli._post") as mp:
            mp.return_value = {"ok": True, "data": {"merged": [{"score": 0.9, "content": "x", "system": "memora"}]}}
            from memory_cli import main
            main()

    def test_main_deep_recall(self):
        with patch("sys.argv", ["memory-cli", "deep-recall", "q"]), \
             patch("memory_cli._post") as mp:
            mp.return_value = {"ok": True, "data": {"interleave": {"rounds": 1, "total_docs_used": 1, "doc_ids_used": ["d1"], "final_answer": "a"}, "memora_context": []}}
            from memory_cli import main
            main()

    def test_main_status(self):
        with patch("sys.argv", ["memory-cli", "status"]), \
             patch("memory_cli._get") as mg:
            mg.return_value = {"ok": True, "data": {"systems": {"memora": {"entries": 5}}}}
            from memory_cli import main
            main()

    def test_main_health(self):
        with patch("sys.argv", ["memory-cli", "health"]), \
             patch("memory_cli._get") as mg:
            mg.return_value = {"ok": True, "status": "healthy"}
            from memory_cli import main
            main()


# ═══════════════════════════════════════════════════════════
# __main__.py entry points
# ═══════════════════════════════════════════════════════════

class TestMainEntryPoints:

    def test_memora_main(self):
        with patch("memora.cli.main") as m:
            import memora.__main__
            # Triggers the import which would call main() only if __name__ == "__main__"

    def test_msa_main(self):
        with patch("msa.cli.main"):
            import msa.__main__

    def test_chronos_main(self):
        with patch("chronos.cli.main"):
            import chronos.__main__
