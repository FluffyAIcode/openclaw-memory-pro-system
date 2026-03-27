"""
Unified LLM client — routes through OpenRouter (primary) or xAI (fallback).

Used by:
  - second_brain/digest.py  (memory summarization/distillation)
  - msa/interleave.py (multi-hop intermediate generation)
  - chronos/consolidator.py (personality profile generation)

Key resolution order:
  1. OPENROUTER_API_KEY env var
  2. OpenClaw auth-profiles.json (openrouter:default)
  3. XAI_API_KEY env var
  4. OpenClaw auth-profiles.json (xai:default)
  5. ~/.openclaw/.env file
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_XAI_URL = "https://api.x.ai/v1/chat/completions"
_OPENROUTER_DEFAULT_MODEL = "deepseek/deepseek-r1"
_OPENROUTER_FAST_MODEL = "deepseek/deepseek-chat"
_XAI_DEFAULT_MODEL = "grok-4"

FAST_MODEL = _OPENROUTER_FAST_MODEL

_AUTH_PROFILES_PATH = (
    Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
)


def _load_auth_profiles() -> dict:
    if _AUTH_PROFILES_PATH.exists():
        try:
            return json.loads(_AUTH_PROFILES_PATH.read_text())
        except Exception:
            pass
    return {}


def _resolve_provider() -> Tuple[Optional[str], str, str]:
    """Return (api_key, api_url, default_model) for the best available provider."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key, _OPENROUTER_URL, _OPENROUTER_DEFAULT_MODEL

    profiles = _load_auth_profiles()
    or_profile = profiles.get("profiles", {}).get("openrouter:default", {})
    if or_profile.get("key"):
        return or_profile["key"], _OPENROUTER_URL, _OPENROUTER_DEFAULT_MODEL

    key = os.environ.get("XAI_API_KEY")
    if key:
        return key, _XAI_URL, _XAI_DEFAULT_MODEL

    xai_profile = profiles.get("profiles", {}).get("xai:default", {})
    if xai_profile.get("key"):
        return xai_profile["key"], _XAI_URL, _XAI_DEFAULT_MODEL

    env_path = Path.home() / ".openclaw" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("XAI_API_KEY="):
                return line.split("=", 1)[1].strip(), _XAI_URL, _XAI_DEFAULT_MODEL

    return None, _OPENROUTER_URL, _OPENROUTER_DEFAULT_MODEL


_resolved: Optional[Tuple[Optional[str], str, str]] = None


def _get_provider() -> Tuple[Optional[str], str, str]:
    global _resolved
    if _resolved is None:
        _resolved = _resolve_provider()
        provider_name = "OpenRouter" if _resolved[1] == _OPENROUTER_URL else "xAI"
        if _resolved[0]:
            logger.info("LLM client using %s (%s)", provider_name, _resolved[2])
        else:
            logger.warning("No LLM API key found — generation unavailable")
    return _resolved


_privacy_gateway = None


def _get_privacy_gateway():
    global _privacy_gateway
    if _privacy_gateway is None:
        try:
            from memory_security import PrivacyGateway
            _privacy_gateway = PrivacyGateway()
        except ImportError:
            pass
    return _privacy_gateway


def generate(
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    model: str = "",
    temperature: float = 0.7,
    timeout: int = 120,
) -> Optional[str]:
    """Call the configured LLM provider and return generated text.

    PII in prompt/system is masked before sending and restored in the response.
    Returns None if no API key is available or the call fails.
    """
    key, api_url, default_model = _get_provider()
    if not key:
        return None

    if not model:
        model = default_model

    gateway = _get_privacy_gateway()
    prompt_map: Dict[str, str] = {}
    sys_map: Dict[str, str] = {}
    if gateway:
        prompt, prompt_map = gateway.mask(prompt)
        if system:
            system, sys_map = gateway.mask(system)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if api_url == _OPENROUTER_URL:
        headers["HTTP-Referer"] = "https://openclaw.ai"
        headers["X-Title"] = "OpenClaw Memory System"

    try:
        resp = requests.post(
            api_url,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data["choices"][0]["message"]["content"]

        combined_map = {**prompt_map, **sys_map}
        if result and combined_map and gateway:
            result = gateway.unmask(result, combined_map)

        return result
    except requests.exceptions.Timeout:
        logger.error("LLM API call timed out after %ds (%s, model=%s)",
                      timeout, api_url, model)
        return None
    except requests.exceptions.RequestException as e:
        logger.error("LLM API call failed (%s): %s", api_url, e)
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error("LLM API response parse error: %s", e)
        return None


def is_available() -> bool:
    key, _, _ = _get_provider()
    return key is not None


def get_provider_info() -> dict:
    """Return current provider config for diagnostics."""
    key, api_url, default_model = _get_provider()
    provider = "openrouter" if api_url == _OPENROUTER_URL else "xai"
    return {
        "provider": provider,
        "api_url": api_url,
        "model": default_model,
        "key_available": key is not None,
    }
