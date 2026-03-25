"""Documentation consistency tests — guard against doc/code drift."""

import re


def test_hub_docstring_matches_routing_threshold():
    from memory_hub import MemoryHub
    doc = MemoryHub.__doc__ or ""
    assert ("100 words" in doc or "≥100" in doc), (
        "MemoryHub docstring must mention the '100 words' MSA routing threshold"
    )
    assert "0.85" in doc, (
        "MemoryHub docstring must mention the 0.85 importance threshold for Chronos/MSA"
    )


def test_hub_docstring_mentions_three_layer_recall():
    from memory_hub import MemoryHub
    doc = MemoryHub.__doc__ or ""
    assert "skills" in doc.lower() and "kg" in doc.lower(), (
        "MemoryHub docstring must mention the three-layer recall (skills, KG, evidence)"
    )


def test_memora_config_no_vllm_url():
    from memora.config import MemoraConfig
    assert not hasattr(MemoraConfig, "vllm_url") or "vllm_url" not in MemoraConfig.model_fields, (
        "vllm_url should be removed from MemoraConfig (dead config)"
    )


def test_route_ingestion_uses_word_count_and_importance():
    """Ensure _route_ingestion uses both word_count and importance."""
    import inspect
    from memory_hub import MemoryHub
    source = inspect.getsource(MemoryHub._route_ingestion)
    assert "word_count" in source, "_route_ingestion must use word_count"
    assert "importance" in source, "_route_ingestion must use importance"
    assert "chronos" in source, "_route_ingestion must route to chronos for high importance"
