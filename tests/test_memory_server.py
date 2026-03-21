"""Tests for memory_server.py

STUBS:
  - SentenceTransformer → FakeSentenceTransformer (conftest, autouse)
  - HTTPServer → not actually started (we test handler logic directly)
  - os.fork → mocked in daemon test
"""

import json
import os
import sys
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest


class TestLoadEnv:
    def test_load_env_file(self, tmp_path):
        env_file = tmp_path / ".openclaw" / ".env"
        env_file.parent.mkdir(parents=True)
        env_file.write_text("TEST_KEY=test_value\n")
        with patch("memory_server.Path.home", return_value=tmp_path):
            import memory_server
            memory_server._load_env()

    def test_hf_offline_setup(self):
        import memory_server
        memory_server._setup_hf_offline()
        assert os.environ.get("HF_HUB_OFFLINE") == "1"


class TestLoadSharedEmbedder:
    """STUB: SentenceTransformer → FakeSentenceTransformer"""

    def test_loads_successfully(self):
        import shared_embedder
        shared_embedder._instance = None
        import memory_server
        emb = memory_server._load_shared_embedder()
        assert emb is not None
        assert shared_embedder.get() is not None
        shared_embedder._instance = None

    def test_fallback_to_mock(self, monkeypatch):
        import shared_embedder
        shared_embedder._instance = None

        fake_st_module = MagicMock()
        fake_st_module.SentenceTransformer.side_effect = Exception("no model")
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

        import memory_server
        emb = memory_server._load_shared_embedder()
        assert emb is not None
        assert memory_server._EMBEDDER_TYPE == "MockEmbedder"
        shared_embedder._instance = None


class TestMemoryHandler:
    """Tests the HTTP handler logic via a mock request/response."""

    def _make_handler(self, method, path, body=None):
        import memory_server

        handler = MagicMock(spec=memory_server.MemoryHandler)
        handler.path = path
        handler.command = method
        handler.headers = {"Content-Length": str(len(body)) if body else "0"}

        if body:
            handler.rfile = BytesIO(body)
        else:
            handler.rfile = BytesIO(b"")

        responses = []

        def mock_respond(code, data):
            responses.append((code, data))

        handler._respond = mock_respond
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = BytesIO()
        handler.log_message = MagicMock()

        # Bind the real inner handler methods so the do_GET/do_POST
        # wrappers can delegate to them correctly on the mock.
        import types
        handler._handle_get = types.MethodType(
            memory_server.MemoryHandler._handle_get, handler)
        handler._handle_post = types.MethodType(
            memory_server.MemoryHandler._handle_post, handler)
        handler._dispatch_sync = types.MethodType(
            memory_server.MemoryHandler._dispatch_sync, handler)
        handler._dispatch_async = types.MethodType(
            memory_server.MemoryHandler._dispatch_async, handler)

        return handler, responses

    def test_health_endpoint(self):
        import memory_server
        handler, responses = self._make_handler("GET", "/health")
        memory_server.MemoryHandler.do_GET(handler)
        assert len(responses) == 1
        assert responses[0][0] == 200
        assert responses[0][1]["status"] == "ok"

    def test_unknown_get(self):
        import memory_server
        handler, responses = self._make_handler("GET", "/nonexistent")
        memory_server.MemoryHandler.do_GET(handler)
        assert responses[0][0] == 404

    def test_unknown_post(self):
        import memory_server
        handler, responses = self._make_handler("POST", "/nonexistent",
                                                 b'{}')
        memory_server.MemoryHandler.do_POST(handler)
        assert responses[0][0] == 404

    def test_invalid_json(self):
        import memory_server
        handler, responses = self._make_handler("POST", "/recall",
                                                 b'not json')
        memory_server.MemoryHandler.do_POST(handler)
        assert responses[0][0] == 400

    def test_status_endpoint(self):
        import memory_server
        mock_hub = MagicMock()
        mock_hub.status.return_value = {"systems": {}}
        memory_server._hub = mock_hub

        handler, responses = self._make_handler("GET", "/status")
        memory_server.MemoryHandler.do_GET(handler)
        assert responses[0][0] == 200
        memory_server._hub = None

    def test_recall_endpoint(self):
        import memory_server
        mock_hub = MagicMock()
        mock_hub.recall.return_value = {"merged": [], "memora": [], "msa": []}
        memory_server._hub = mock_hub

        body = json.dumps({"query": "test"}).encode()
        handler, responses = self._make_handler("POST", "/recall", body)
        memory_server.MemoryHandler.do_POST(handler)
        assert responses[0][0] == 200
        memory_server._hub = None

    def test_remember_endpoint(self):
        import memory_server
        mock_hub = MagicMock()
        mock_hub.remember.return_value = {"word_count": 5, "systems_used": ["memora"]}
        memory_server._hub = mock_hub

        body = json.dumps({"content": "test content"}).encode()
        handler, responses = self._make_handler("POST", "/remember", body)
        memory_server.MemoryHandler.do_POST(handler)
        assert responses[0][0] == 200
        memory_server._hub = None

    def test_deep_recall_endpoint(self):
        import memory_server
        mock_hub = MagicMock()
        mock_hub.deep_recall.return_value = {"interleave": None, "memora_context": []}
        memory_server._hub = mock_hub
        body = json.dumps({"query": "complex"}).encode()
        handler, responses = self._make_handler("POST", "/deep-recall", body)
        memory_server.MemoryHandler.do_POST(handler)
        assert responses[0][0] == 200
        memory_server._hub = None

    def test_search_endpoint(self):
        import memory_server
        mock_vs = MagicMock()
        mock_vs.search.return_value = [{"content": "x", "score": 0.9}]
        with patch.dict(sys.modules, {
            "memora.vectorstore": MagicMock(vector_store=mock_vs)
        }):
            body = json.dumps({"query": "test"}).encode()
            handler, responses = self._make_handler("POST", "/search", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200

    def test_add_endpoint(self):
        import memory_server
        mock_collector = MagicMock()
        mock_collector.collect.return_value = {"timestamp": "t", "content": "x"}
        mock_vs = MagicMock()
        with patch.dict(sys.modules, {
            "memora.collector": MagicMock(collector=mock_collector),
            "memora.vectorstore": MagicMock(vector_store=mock_vs)
        }):
            body = json.dumps({"content": "add this"}).encode()
            handler, responses = self._make_handler("POST", "/add", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200

    def test_digest_endpoint(self):
        import memory_server
        with patch("memora.digest.digest_memories", return_value=True):
            body = json.dumps({"days": 3}).encode()
            handler, responses = self._make_handler("POST", "/digest", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200

    def test_chronos_learn_endpoint(self):
        import memory_server
        import chronos.bridge as cbmod
        orig = cbmod.bridge
        cbmod.bridge = MagicMock()
        cbmod.bridge.learn_and_save.return_value = MagicMock(
            importance=0.8, timestamp=MagicMock(isoformat=lambda: "2026-01-01"))
        try:
            body = json.dumps({"content": "learn this"}).encode()
            handler, responses = self._make_handler("POST", "/chronos/learn", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200
        finally:
            cbmod.bridge = orig

    def test_chronos_consolidate_endpoint(self):
        import memory_server
        import chronos.bridge as cbmod
        orig = cbmod.bridge
        cbmod.bridge = MagicMock()
        cbmod.bridge.consolidate.return_value = {"consolidated": 3}
        try:
            body = json.dumps({}).encode()
            handler, responses = self._make_handler("POST", "/chronos/consolidate", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200
        finally:
            cbmod.bridge = orig

    def test_msa_ingest_endpoint(self):
        import memory_server
        import msa.bridge as mbmod
        orig = mbmod.bridge
        mbmod.bridge = MagicMock()
        mbmod.bridge.ingest_and_save.return_value = {"doc_id": "d1", "chunks": 2}
        try:
            body = json.dumps({"content": "ingest this"}).encode()
            handler, responses = self._make_handler("POST", "/msa/ingest", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200
        finally:
            mbmod.bridge = orig

    def test_msa_query_endpoint(self):
        import memory_server
        import msa.bridge as mbmod
        orig = mbmod.bridge
        mbmod.bridge = MagicMock()
        mbmod.bridge.query_memory.return_value = {"results": []}
        try:
            body = json.dumps({"query": "test"}).encode()
            handler, responses = self._make_handler("POST", "/msa/query", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200
        finally:
            mbmod.bridge = orig

    def test_msa_interleave_endpoint(self):
        import memory_server
        import msa.bridge as mbmod
        orig = mbmod.bridge
        mbmod.bridge = MagicMock()
        mbmod.bridge.interleave_query.return_value = {"answer": "a"}
        try:
            body = json.dumps({"query": "q"}).encode()
            handler, responses = self._make_handler("POST", "/msa/interleave", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200
        finally:
            mbmod.bridge = orig

    def test_respond_method(self):
        import memory_server
        handler, responses = self._make_handler("GET", "/health")
        memory_server.MemoryHandler._respond(handler, 200, {"test": True})
        assert handler.send_response.called
        written = handler.wfile.getvalue()
        assert b"test" in written

    def test_async_collide_returns_task_id(self):
        import memory_server
        body = json.dumps({"async": True}).encode()
        handler, responses = self._make_handler("POST", "/second-brain/collide", body)
        memory_server.MemoryHandler.do_POST(handler)
        assert responses[0][0] == 202
        assert "task_id" in responses[0][1]
        assert responses[0][1]["status"] == "running"

    def test_sync_mode_by_default(self):
        import memory_server
        mock_hub = MagicMock()
        mock_hub.recall.return_value = {"merged": [], "memora": [], "msa": []}
        memory_server._hub = mock_hub
        body = json.dumps({"query": "test"}).encode()
        handler, responses = self._make_handler("POST", "/recall", body)
        memory_server.MemoryHandler.do_POST(handler)
        assert responses[0][0] == 200
        assert "merged" in responses[0][1]
        memory_server._hub = None

    def test_task_polling(self):
        import memory_server
        task_id = memory_server._task_manager.submit(
            "test-task", lambda: {"result": "ok"})
        import time
        time.sleep(0.3)
        task = memory_server._task_manager.get(task_id)
        assert task is not None
        assert task["status"] == "done"
        assert task["result"] == {"result": "ok"}

    def test_task_error_handling(self):
        import memory_server
        def _fail():
            raise RuntimeError("boom")
        task_id = memory_server._task_manager.submit("fail-task", _fail)
        import time
        time.sleep(0.3)
        task = memory_server._task_manager.get(task_id)
        assert task["status"] == "error"
        assert "boom" in task["error"]

    def test_task_not_found(self):
        import memory_server
        handler, responses = self._make_handler("GET", "/task/nonexistent")
        memory_server.MemoryHandler.do_GET(handler)
        assert responses[0][0] == 404

    def test_tasks_list(self):
        import memory_server
        handler, responses = self._make_handler("GET", "/tasks")
        memory_server.MemoryHandler.do_GET(handler)
        assert responses[0][0] == 200
        assert "tasks" in responses[0][1]

    def test_task_cleanup(self):
        import memory_server
        task_id = memory_server._task_manager.submit(
            "cleanup-test", lambda: {"x": 1})
        import time
        time.sleep(0.3)
        memory_server._task_manager.cleanup(max_age=0)
        assert memory_server._task_manager.get(task_id) is None

    def test_briefing_endpoint(self):
        import memory_server
        mock_bridge = MagicMock()
        mock_bridge.daily_briefing.return_value = {
            "text": "briefing text", "date": "2026-03-14",
        }
        with patch.dict(sys.modules, {
            "second_brain": MagicMock(),
            "second_brain.bridge": MagicMock(bridge=mock_bridge),
        }):
            handler, responses = self._make_handler("GET", "/briefing")
            memory_server.MemoryHandler.do_GET(handler)
            assert responses[0][0] == 200
            assert responses[0][1]["text"] == "briefing text"

    def test_vitality_endpoint(self):
        import memory_server
        mock_bridge = MagicMock()
        mock_bridge.vitality_list.return_value = {
            "total": 10, "distribution": {"high": 3, "medium": 5, "low": 2},
        }
        with patch.dict(sys.modules, {
            "second_brain": MagicMock(),
            "second_brain.bridge": MagicMock(bridge=mock_bridge),
        }):
            handler, responses = self._make_handler("GET", "/vitality")
            memory_server.MemoryHandler.do_GET(handler)
            assert responses[0][0] == 200
            assert responses[0][1]["total"] == 10

    def test_dormant_endpoint(self):
        import memory_server
        mock_bridge = MagicMock()
        mock_bridge.list_dormant.return_value = {
            "count": 2, "memories": [{"content": "old"}],
        }
        with patch.dict(sys.modules, {
            "second_brain": MagicMock(),
            "second_brain.bridge": MagicMock(bridge=mock_bridge),
        }):
            handler, responses = self._make_handler("GET", "/dormant")
            memory_server.MemoryHandler.do_GET(handler)
            assert responses[0][0] == 200
            assert responses[0][1]["count"] == 2

    def test_inspect_endpoint(self):
        import memory_server
        mock_bridge = MagicMock()
        mock_bridge.memory_lifecycle.return_value = {
            "query": "test", "matches": [{"content": "found"}],
        }
        with patch.dict(sys.modules, {
            "second_brain": MagicMock(),
            "second_brain.bridge": MagicMock(bridge=mock_bridge),
        }):
            body = json.dumps({"query": "test"}).encode()
            handler, responses = self._make_handler("POST", "/inspect", body)
            memory_server.MemoryHandler.do_POST(handler)
            assert responses[0][0] == 200
            assert len(responses[0][1]["matches"]) == 1
