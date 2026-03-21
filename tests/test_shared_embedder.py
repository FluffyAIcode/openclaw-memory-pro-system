"""Tests for shared_embedder.py — 100% coverage."""

import shared_embedder


class TestSharedEmbedder:

    def setup_method(self):
        shared_embedder._instance = None

    def test_get_returns_none_initially(self):
        assert shared_embedder.get() is None

    def test_set_and_get(self):
        fake = object()
        shared_embedder.set(fake)
        assert shared_embedder.get() is fake

    def test_set_overwrites(self):
        a, b = object(), object()
        shared_embedder.set(a)
        shared_embedder.set(b)
        assert shared_embedder.get() is b

    def teardown_method(self):
        shared_embedder._instance = None
