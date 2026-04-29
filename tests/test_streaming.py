import json

import pytest

from oaipatch.config import Settings
from oaipatch.streaming import stitch_sse


async def _collect(gen):
    out = b""
    async for chunk in gen:
        out += chunk
    return out


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _aiter(chunks):
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_stream_reasoning_then_content_emits_think_tags():
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "let "}}]}),
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "me think"}}]}),
        _sse({"choices": [{"index": 0, "delta": {"content": "Hello"}}]}),
        _sse({"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    # Extract data: payloads
    data_lines = [line[5:].strip() for line in out.splitlines() if line.startswith("data:")]
    payloads = [json.loads(line) for line in data_lines if line != "[DONE]"]
    contents = [p["choices"][0]["delta"].get("content", "") for p in payloads]
    joined = "".join(c for c in contents if isinstance(c, str))
    assert "<think>\nlet me think" in joined
    assert "</think>\nHello world" in joined
    # reasoning_content must never leak downstream
    for p in payloads:
        assert "reasoning_content" not in p["choices"][0]["delta"]


@pytest.mark.asyncio
async def test_stream_no_reasoning_passthrough_content():
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"content": "abc"}}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "abc" in out
    assert "<think>" not in out


@pytest.mark.asyncio
async def test_stream_closes_think_on_finish_without_content():
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "only think"}}]}),
        _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "<think>" in out
    assert "</think>" in out


@pytest.mark.asyncio
async def test_stream_done_passthrough():
    settings = Settings()
    chunks = [b"data: [DONE]\n\n"]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "[DONE]" in out
