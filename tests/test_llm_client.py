"""Tests for llm_client.py — provider resolution + generate + diagnostics."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests as req_lib
import llm_client


def _reset():
    llm_client._resolved = None


class TestResolveProvider:

    def teardown_method(self):
        _reset()

    def test_openrouter_env_var(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-test")
        key, url, model = llm_client._resolve_provider()
        assert key == "or-key-test"
        assert "openrouter" in url
        assert model == "deepseek/deepseek-r1"

    def test_xai_env_var(self, monkeypatch):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-key-test")
        with patch.object(llm_client, "_load_auth_profiles", return_value={}):
            key, url, model = llm_client._resolve_provider()
        assert key == "xai-key-test"
        assert "x.ai" in url
        assert model == "grok-4"

    def test_from_auth_profiles_openrouter(self, monkeypatch):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        profiles = {
            "profiles": {
                "openrouter:default": {"key": "or-from-file"},
                "xai:default": {"key": "xai-from-file"},
            }
        }
        with patch.object(llm_client, "_load_auth_profiles", return_value=profiles):
            key, url, _ = llm_client._resolve_provider()
        assert key == "or-from-file"
        assert "openrouter" in url

    def test_from_auth_profiles_xai_fallback(self, monkeypatch):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        profiles = {
            "profiles": {
                "openrouter:default": {},
                "xai:default": {"key": "xai-fb"},
            }
        }
        with patch.object(llm_client, "_load_auth_profiles", return_value=profiles):
            key, url, _ = llm_client._resolve_provider()
        assert key == "xai-fb"
        assert "x.ai" in url

    def test_from_dotenv_file(self, monkeypatch, tmp_path):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch.object(llm_client, "_load_auth_profiles", return_value={}):
            (tmp_path / ".openclaw").mkdir()
            (tmp_path / ".openclaw" / ".env").write_text("XAI_API_KEY=dotenv-key\n")
            with patch("llm_client.Path.home", return_value=tmp_path):
                key, url, _ = llm_client._resolve_provider()
        assert key == "dotenv-key"
        assert "x.ai" in url

    def test_none_when_nothing_found(self, monkeypatch, tmp_path):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch.object(llm_client, "_load_auth_profiles", return_value={}), \
             patch("llm_client.Path.home", return_value=tmp_path):
            key, _, _ = llm_client._resolve_provider()
        assert key is None


class TestGetProvider:

    def teardown_method(self):
        _reset()

    def test_caches_result(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "cached")
        k1, _, _ = llm_client._get_provider()
        k2, _, _ = llm_client._get_provider()
        assert k1 == k2 == "cached"


class TestGenerate:

    def teardown_method(self):
        _reset()

    def test_no_key_returns_none(self, monkeypatch, tmp_path):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch.object(llm_client, "_load_auth_profiles", return_value={}), \
             patch("llm_client.Path.home", return_value=tmp_path):
            assert llm_client.generate("hello") is None

    def test_successful_openrouter_call(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Generated"}}]
        }
        with patch("llm_client.requests.post", return_value=mock_resp) as mp:
            result = llm_client.generate("prompt", system="sys", max_tokens=100)
            assert result == "Generated"
            call_args = mp.call_args
            assert "openrouter" in call_args.args[0]
            body = call_args.kwargs["json"]
            assert body["model"] == "deepseek/deepseek-r1"
            assert len(body["messages"]) == 2
            headers = call_args.kwargs["headers"]
            assert "HTTP-Referer" in headers

    def test_without_system_prompt(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("llm_client.requests.post", return_value=mock_resp) as mp:
            llm_client.generate("prompt")
            body = mp.call_args.kwargs["json"]
            assert len(body["messages"]) == 1

    def test_custom_model(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("llm_client.requests.post", return_value=mock_resp) as mp:
            llm_client.generate("prompt", model="custom/model")
            assert mp.call_args.kwargs["json"]["model"] == "custom/model"

    def test_xai_no_referer_header(self, monkeypatch):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-k")
        with patch.object(llm_client, "_load_auth_profiles", return_value={}):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            with patch("llm_client.requests.post", return_value=mock_resp) as mp:
                llm_client.generate("prompt")
                headers = mp.call_args.kwargs["headers"]
                assert "HTTP-Referer" not in headers
                assert "x.ai" in mp.call_args.args[0]

    def test_timeout_returns_none(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch("llm_client.requests.post", side_effect=req_lib.exceptions.Timeout):
            assert llm_client.generate("prompt") is None

    def test_connection_error_returns_none(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch("llm_client.requests.post",
                   side_effect=req_lib.exceptions.ConnectionError):
            assert llm_client.generate("prompt") is None

    def test_bad_json_returns_none(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"bad": "format"}
        with patch("llm_client.requests.post", return_value=mock_resp):
            assert llm_client.generate("prompt") is None


class TestIsAvailable:

    def teardown_method(self):
        _reset()

    def test_true_when_key_present(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        assert llm_client.is_available() is True

    def test_false_when_no_key(self, monkeypatch, tmp_path):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch.object(llm_client, "_load_auth_profiles", return_value={}), \
             patch("llm_client.Path.home", return_value=tmp_path):
            assert llm_client.is_available() is False


class TestGetProviderInfo:

    def teardown_method(self):
        _reset()

    def test_returns_dict(self, monkeypatch):
        _reset()
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        info = llm_client.get_provider_info()
        assert info["provider"] == "openrouter"
        assert info["key_available"] is True
        assert "model" in info

    def test_xai_provider(self, monkeypatch):
        _reset()
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xk")
        with patch.object(llm_client, "_load_auth_profiles", return_value={}):
            info = llm_client.get_provider_info()
        assert info["provider"] == "xai"
