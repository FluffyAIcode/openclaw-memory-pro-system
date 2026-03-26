"""Tests for memory_cli.py

STUBS:
  - urllib.request.urlopen → mocked (no real HTTP server needed)
  - subprocess.Popen → mocked for server-start
  - os.kill → mocked for server-stop
"""

import json
import os
from unittest.mock import patch, MagicMock
from io import BytesIO

import pytest
import memory_cli


class FakeResponse:
    def __init__(self, data):
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _mock_urlopen(expected_response):
    return patch("memory_cli.urlopen", return_value=FakeResponse(expected_response))


class TestPost:
    """STUB: urlopen → FakeResponse"""

    def test_post_success(self):
        with _mock_urlopen({"result": "ok"}):
            result = memory_cli._post("/test", {"key": "val"})
            assert result == {"result": "ok"}

    def test_post_connection_refused(self):
        from urllib.error import URLError
        with patch("memory_cli.urlopen", side_effect=URLError("refused")):
            with pytest.raises(SystemExit):
                memory_cli._post("/test", {})


class TestGet:
    """STUB: urlopen → FakeResponse"""

    def test_get_success(self):
        with _mock_urlopen({"status": "ok"}):
            result = memory_cli._get("/health")
            assert result["status"] == "ok"


class TestCommands:
    """STUB: urlopen → FakeResponse for all HTTP calls"""

    def test_cmd_remember(self, capsys):
        args = MagicMock(content="test", source="cli", importance=0.7,
                         doc_id=None, title=None, tag=None)
        with _mock_urlopen({"word_count": 5, "systems_used": ["memora", "daily_file"]}):
            memory_cli.cmd_remember(args)
        out = capsys.readouterr().out
        assert "已记住" in out

    def test_cmd_remember_with_tag(self, capsys):
        args = MagicMock(content="test", source="cli", importance=0.7,
                         doc_id=None, title=None, tag="thought")
        with _mock_urlopen({"word_count": 5, "systems_used": ["memora"]}):
            memory_cli.cmd_remember(args)
        out = capsys.readouterr().out
        assert "#thought" in out

    def test_cmd_recall(self, capsys):
        args = MagicMock(query="test", top_k=5)
        with _mock_urlopen({
            "skills": [],
            "kg_relations": [{"description": "fact → supports → decision", "is_critical": False}],
            "evidence": [{"score": 0.9, "content": "result memory", "timestamp": "2026-03-26"}],
        }):
            memory_cli.cmd_recall(args)
        out = capsys.readouterr().out
        assert "result memory" in out

    def test_cmd_recall_empty(self, capsys):
        args = MagicMock(query="test", top_k=5)
        with _mock_urlopen({"skills": [], "kg_relations": [], "evidence": []}):
            memory_cli.cmd_recall(args)
        out = capsys.readouterr().out
        assert "没有找到" in out

    def test_cmd_deep_recall(self, capsys):
        args = MagicMock(query="test", max_rounds=3)
        with _mock_urlopen({
            "interleave": {
                "rounds": 2, "total_docs_used": 1,
                "final_answer": "deep answer here"
            },
            "memora_context": [{"score": 0.5, "content": "ctx"}]
        }):
            memory_cli.cmd_deep_recall(args)
        out = capsys.readouterr().out
        assert "deep answer" in out

    def test_cmd_deep_recall_no_interleave(self, capsys):
        args = MagicMock(query="test", max_rounds=3)
        with _mock_urlopen({"interleave": None, "memora_context": []}):
            memory_cli.cmd_deep_recall(args)
        out = capsys.readouterr().out
        assert "暂无结果" in out

    def test_cmd_search(self, capsys):
        args = MagicMock(query="test", top_k=3)
        with _mock_urlopen({"results": [
            {"score": 0.8, "content": "found it"}
        ]}):
            memory_cli.cmd_search(args)
        out = capsys.readouterr().out
        assert "found it" in out

    def test_cmd_add(self, capsys):
        args = MagicMock(content="new item", source="test", importance=0.7)
        with _mock_urlopen({"timestamp": "2026-01-01", "source": "test"}):
            memory_cli.cmd_add(args)
        out = capsys.readouterr().out
        assert "已添加" in out

    def test_cmd_digest(self, capsys):
        args = MagicMock(days=7)
        with _mock_urlopen({"success": True}):
            memory_cli.cmd_digest(args)
        out = capsys.readouterr().out
        assert "已生成" in out

    def test_cmd_status(self, capsys):
        args = MagicMock()
        with _mock_urlopen({
            "server": {"uptime_seconds": 10, "embedder": "ST"},
            "systems": {"memora": {"entries": 5}}
        }):
            memory_cli.cmd_status(args)
        out = capsys.readouterr().out
        assert "5 条记忆" in out

    def test_cmd_health(self, capsys):
        args = MagicMock()
        with _mock_urlopen({"status": "ok", "uptime_seconds": 10,
                            "embedder": "ST", "pid": 123}):
            memory_cli.cmd_health(args)
        out = capsys.readouterr().out
        assert "服务正常" in out


class TestSkillsCLI:
    """STUB: urlopen → FakeResponse"""

    def test_cmd_skills_empty(self, capsys):
        with _mock_urlopen({"skills": []}):
            memory_cli.cmd_skills(MagicMock())
        out = capsys.readouterr().out
        assert "还没有" in out

    def test_cmd_skills_list(self, capsys):
        with _mock_urlopen({"skills": [
            {"id": "abc", "name": "Python", "status": "active",
             "version": 2, "tags": ["code"]}
        ]}):
            memory_cli.cmd_skills(MagicMock())
        out = capsys.readouterr().out
        assert "Python" in out
        assert "code" in out

    def test_cmd_skill_add(self, capsys):
        args = MagicMock(content="desc", tags="a,b")
        args.name = "test"
        with _mock_urlopen({"id": "x", "name": "test"}):
            memory_cli.cmd_skill_add(args)
        out = capsys.readouterr().out
        assert "已创建" in out

    def test_cmd_skill_promote(self, capsys):
        args = MagicMock(skill_id="x")
        with _mock_urlopen({"id": "x", "name": "test"}):
            memory_cli.cmd_skill_promote(args)
        out = capsys.readouterr().out
        assert "已激活" in out

    def test_cmd_skill_deprecate(self, capsys):
        args = MagicMock(skill_id="x")
        with _mock_urlopen({"id": "x", "name": "test"}):
            memory_cli.cmd_skill_deprecate(args)
        out = capsys.readouterr().out
        assert "已废弃" in out

    def test_cmd_training_export(self, capsys):
        with _mock_urlopen({"dataset_path": "/tmp/data.jsonl"}):
            memory_cli.cmd_training_export(MagicMock())
        out = capsys.readouterr().out
        assert "已导出" in out


class TestServerManagement:
    """STUB: subprocess.Popen, os.kill"""

    def _save_workspace(self):
        self._orig = memory_cli._WORKSPACE

    def _restore_workspace(self):
        memory_cli._WORKSPACE = self._orig

    def test_server_start_not_running(self, tmp_path):
        self._save_workspace()
        try:
            memory_cli._WORKSPACE = tmp_path
            (tmp_path / "memory").mkdir()
            (tmp_path / "memory_server.py").write_text("")

            with patch("memory_cli.subprocess.Popen") as mock_popen:
                mock_proc = MagicMock()
                mock_proc.wait.return_value = 0
                mock_popen.return_value = mock_proc
                memory_cli.cmd_server_start(MagicMock())
                mock_popen.assert_called_once()
        finally:
            self._restore_workspace()

    def test_server_start_already_running(self, tmp_path, capsys):
        self._save_workspace()
        try:
            memory_cli._WORKSPACE = tmp_path
            pid_file = tmp_path / "memory" / "server.pid"
            pid_file.parent.mkdir(parents=True)
            pid_file.write_text(str(os.getpid()))

            memory_cli.cmd_server_start(MagicMock())
            out = capsys.readouterr().out
            assert "已在运行" in out
        finally:
            self._restore_workspace()

    def test_server_stop(self, tmp_path, capsys):
        self._save_workspace()
        try:
            memory_cli._WORKSPACE = tmp_path
            pid_file = tmp_path / "memory" / "server.pid"
            pid_file.parent.mkdir(parents=True)
            pid_file.write_text("99999")

            with patch("memory_cli.os.kill") as mock_kill:
                mock_kill.side_effect = OSError("no such process")
                memory_cli.cmd_server_stop(MagicMock())
            out = capsys.readouterr().out
            assert "已不存在" in out
        finally:
            self._restore_workspace()

    def test_server_stop_no_pid(self, tmp_path, capsys):
        self._save_workspace()
        try:
            memory_cli._WORKSPACE = tmp_path
            (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
            memory_cli.cmd_server_stop(MagicMock())
            out = capsys.readouterr().out
            assert "未运行" in out
        finally:
            self._restore_workspace()
