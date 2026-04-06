import pytest
from src.cache.response_cache import ResponseCache


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(cache_dir=str(tmp_path / "test_cache"))


def test_cache_miss_returns_none(cache):
    result = cache.get("prompt_1", "context_1")
    assert result is None


def test_cache_set_and_get(cache):
    cache.set("prompt_1", "context_1", "response_text")
    result = cache.get("prompt_1", "context_1")
    assert result == "response_text"


def test_cache_different_keys_are_independent(cache):
    cache.set("prompt_1", "context_1", "response_a")
    cache.set("prompt_2", "context_2", "response_b")
    assert cache.get("prompt_1", "context_1") == "response_a"
    assert cache.get("prompt_2", "context_2") == "response_b"


def test_cache_stats(cache):
    cache.get("prompt_1", "context_1")  # miss
    cache.set("prompt_1", "context_1", "response_text")
    cache.get("prompt_1", "context_1")  # hit
    cache.get("prompt_2", "context_2")  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2


def test_cache_disabled():
    cache = ResponseCache(cache_dir="/tmp/unused", enabled=False)
    cache.set("prompt_1", "context_1", "response_text")
    result = cache.get("prompt_1", "context_1")
    assert result is None
