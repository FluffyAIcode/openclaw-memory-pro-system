"""Tests for llm_client.py — 100% coverage."""

import json
from unittest.mock import MagicMock, patch, mock_open

import pytest
import llm_client


class TestLoadApiKey:

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key-123")
        llm_client._api_key = None
        assert llm_client._load_api_key() == "test-key-123"

    def test_from_dotenv_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("XAI_API_KEY=file-key-456\n")
        with patch("llm_client.Path.home", return_value=tmp_path), \
             patch.object(llm_client, "_load_api_key") as real:
            # Call the real logic manually
            pass
        # Direct test of file parsing
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        llm_client._api_key = None
        with patch("llm_client.Path.home") as mock_home:
            mock_home.return_value = tmp_path
            (tmp_path / ".openclaw").mkdir(exist_ok=True)
            (tmp_path / ".openclaw" / ".env").write_text("XAI_API_KEY=from-file\n")
            key = llm_client._load_api_key()
            assert key == "from-file"

    def test_returns_none_when_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        llm_client._api_key = None
        with patch("llm_client.Path.home") as mock_home:
            mock_home.return_value = tmp_path
            assert llm_client._load_api_key() is None


class TestGetKey:

    def test_caches_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "cached-key")
        llm_client._api_key = None
        k1 = llm_client._get_key()
        k2 = llm_client._get_key()
        assert k1 == k2 == "cached-key"

    def teardown_method(self):
        llm_client._api_key = None


class TestGenerate:

    def test_no_api_key_returns_none(self, monkeypatch, tmp_path):
        llm_client._api_key = None
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch("llm_client.Path.home", return_value=tmp_path):
            llm_client._api_key = None
            result = llm_client.generate("hello")
            assert result is None

    def test_successful_call(self, monkeypatch):
        llm_client._api_key = "test-key"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Generated text"}}]
        }
        with patch("llm_client.requests.post", return_value=mock_resp) as mock_post:
            result = llm_client.generate("prompt", system="sys", max_tokens=100)
            assert result == "Generated text"
            call_args = mock_post.call_args
            body = call_args.kwargs["json"]
            assert body["model"] == "grok-4"
            assert len(body["messages"]) == 2
            assert body["messages"][0]["role"] == "system"

    def test_without_system_prompt(self, monkeypatch):
        llm_client._api_key = "test-key"
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        with patch("llm_client.requests.post", return_value=mock_resp) as mp:
            llm_client.generate("prompt")
            body = mp.call_args.kwargs["json"]
            assert len(body["messages"]) == 1

    def test_timeout_returns_none(self):
        llm_client._api_key = "test-key"
        import requests
        with patch("llm_client.requests.post", side_effect=requests.exceptions.Timeout):
            assert llm_client.generate("prompt") is None

    def test_request_error_returns_none(self):
        llm_client._api_key = "test-key"
        import requests
        with patch("llm_client.requests.post",
                   side_effect=requests.exceptions.ConnectionError):
            assert llm_client.generate("prompt") is None

    def test_json_parse_error_returns_none(self):
        llm_client._api_key = "test-key"
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"bad": "format"}
        with patch("llm_client.requests.post", return_value=mock_resp):
            assert llm_client.generate("prompt") is None

    def teardown_method(self):
        llm_client._api_key = None


class TestIsAvailable:

    def test_true_when_key_present(self):
        llm_client._api_key = "key"
        assert llm_client.is_available() is True

    def test_false_when_no_key(self, monkeypatch, tmp_path):
        llm_client._api_key = None
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch("llm_client.Path.home", return_value=tmp_path):
            llm_client._api_key = None
            assert llm_client.is_available() is False

    def teardown_method(self):
        llm_client._api_key = None
