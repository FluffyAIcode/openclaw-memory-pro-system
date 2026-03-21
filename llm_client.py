"""
Unified LLM client — calls xAI Grok API (OpenAI-compatible).

Used by:
  - second_brain/digest.py  (memory summarization/distillation)
  - msa/interleave.py (multi-hop intermediate generation)
  - chronos/consolidator.py (personality profile generation)

Loads XAI_API_KEY from environment or from ~/.openclaw/.env.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://api.x.ai/v1/chat/completions"
_DEFAULT_MODEL = "grok-4"


def _load_api_key() -> Optional[str]:
    key = os.environ.get("XAI_API_KEY")
    if key:
        return key

    env_path = Path.home() / ".openclaw" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("XAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


_api_key: Optional[str] = None


def _get_key() -> Optional[str]:
    global _api_key
    if _api_key is None:
        _api_key = _load_api_key()
    return _api_key


def generate(
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call xAI Grok API and return the generated text.

    Returns None if the API key is missing or the call fails.
    """
    key = _get_key()
    if not key:
        logger.warning("XAI_API_KEY not found — LLM generation unavailable")
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            _API_URL,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        logger.error("LLM API call timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error("LLM API call failed: %s", e)
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error("LLM API response parse error: %s", e)
        return None


def is_available() -> bool:
    return _get_key() is not None
