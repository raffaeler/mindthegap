"""Tests for the tool_call_id-keyed reasoning sidecar cache."""

from __future__ import annotations

from mindthegap.cache import ReasoningCache
from mindthegap.config import Settings
from mindthegap.transforms import (
    stitch_message,
    transform_request_body,
    unstitch_messages,
)

S = Settings()


def test_cache_round_trip_basic():
    c = ReasoningCache()
    c.put("call_abc", "the reasoning")
    assert c.get("call_abc") == "the reasoning"
    assert c.get("missing") is None


def test_cache_ignores_empty_keys_and_values():
    c = ReasoningCache()
    c.put("", "x")
    c.put("k", "")
    assert c.get("") is None
    assert c.get("k") is None


def test_cache_ttl_expiry():
    c = ReasoningCache(ttl_seconds=0.0)
    c.put("k", "v")
    assert c.get("k") is None


def test_cache_lru_eviction():
    c = ReasoningCache(max_entries=2)
    c.put("a", "1")
    c.put("b", "2")
    c.put("c", "3")
    assert c.get("a") is None
    assert c.get("b") == "2"
    assert c.get("c") == "3"


def test_stitch_indexes_reasoning_under_tool_call_ids():
    c = ReasoningCache()
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function"},
            {"id": "call_2", "type": "function"},
        ],
        "reasoning_content": "deep thoughts",
    }
    stitch_message(msg, S, cache=c)
    assert c.get("call_1") == "deep thoughts"
    assert c.get("call_2") == "deep thoughts"


def test_unstitch_recovers_reasoning_for_tool_call_assistant_with_null_content():
    """Reproduces the Copilot CLI failure: client persisted the assistant
    tool-call message with content=null, so the <think> block is gone, but
    the cache still has the reasoning indexed by tool_call_id.
    """
    c = ReasoningCache()
    c.put("call_xyz", "the original reasoning")
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_xyz", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call_xyz", "content": "result"},
    ]
    out = unstitch_messages(messages, S, "forward", cache=c)
    assert out[1]["reasoning_content"] == "the original reasoning"
    assert out[1]["content"] is None
    assert out[1]["tool_calls"] == [{"id": "call_xyz", "type": "function"}]


def test_unstitch_does_not_overwrite_existing_reasoning_content():
    c = ReasoningCache()
    c.put("call_1", "stale cached")
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1"}],
            "reasoning_content": "fresh inbound",
        }
    ]
    out = unstitch_messages(messages, S, "forward", cache=c)
    assert out[0]["reasoning_content"] == "fresh inbound"


def test_unstitch_prefers_inline_think_block_over_cache():
    """If the client preserved the <think> block we trust that over cache."""
    c = ReasoningCache()
    c.put("call_1", "cached reasoning")
    messages = [
        {
            "role": "assistant",
            "content": "<think>\ninline reasoning\n</think>\n\n",
            "tool_calls": [{"id": "call_1"}],
        }
    ]
    out = unstitch_messages(messages, S, "forward", cache=c)
    assert out[0]["reasoning_content"] == "inline reasoning"
    assert out[0]["content"] is None


def test_unstitch_drop_mode_skips_cache_lookup():
    c = ReasoningCache()
    c.put("call_1", "cached")
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
    ]
    out = unstitch_messages(messages, S, "drop", cache=c)
    assert "reasoning_content" not in out[0]


def test_transform_request_body_uses_cache_for_reasoner_model():
    c = ReasoningCache()
    c.put("call_1", "from cache")
    body = {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    }
    out = transform_request_body(body, S, cache=c)
    assert out["messages"][0]["reasoning_content"] == "from cache"
