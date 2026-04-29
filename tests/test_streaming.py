import json

import pytest

from mindthegap.config import Settings
from mindthegap.streaming import stitch_sse


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
    assert "[[think]]  \nlet me think" in joined
    assert "[[/think]]\n\nHello world" in joined
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
    assert "[[think]]" not in out


@pytest.mark.asyncio
async def test_stream_closes_think_on_finish_without_content():
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "only think"}}]}),
        _sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "[[think]]" in out
    assert "[[/think]]" in out


@pytest.mark.asyncio
async def test_stream_done_passthrough():
    settings = Settings()
    chunks = [b"data: [DONE]\n\n"]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "[DONE]" in out


@pytest.mark.asyncio
async def test_stream_closes_think_before_done_when_finish_missing():
    # Reasoning starts, no real content, no finish_reason — just [DONE].
    # The proxy must still inject [[/think]] so the client persists a balanced
    # message; otherwise the next turn ships an unclosed [[think]] upstream.
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "abrupt"}}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    # Order matters: [[/think]] must appear before [DONE]
    assert out.index("[[/think]]") < out.index("[DONE]")
    assert "[[think]]" in out


@pytest.mark.asyncio
async def test_stream_closes_think_at_eof_without_done():
    # Upstream connection drops mid-reasoning: no finish_reason, no [DONE].
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "cut off"}}]}),
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "[[think]]" in out
    assert "[[/think]]" in out


@pytest.mark.asyncio
async def test_stream_emits_markdown_hard_break_before_close_when_no_trailing_newline():
    # Reasoning ends without a trailing \n. Proxy must insert a Markdown
    # hard line break ("  \n") before [[/think]] so renderers don't collapse
    # the bare \n into a space and put [[/think]] inline with the reasoning.
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "Answer: 3 balls."}}]}),
        _sse({"choices": [{"index": 0, "delta": {"content": "John"}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    payloads = [
        json.loads(line[5:].strip())
        for line in out.splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    ]
    joined = "".join(
        p["choices"][0]["delta"].get("content") or ""
        for p in payloads
        if isinstance(p["choices"][0]["delta"].get("content"), str)
    )
    assert "Answer: 3 balls.  \n[[/think]]\n\nJohn" in joined

    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "r"}}]}),
        _sse({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert out.count("[[/think]]") == 1


@pytest.mark.asyncio
async def test_stream_does_not_add_blank_line_when_reasoning_already_ends_in_newlines():
    # The closing [[/think]] must always sit on its own line with NO blank
    # line before it. Trailing \n already in the upstream reasoning must
    # not be supplemented with another \n by the proxy.
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "thinking done\n"}}]}),
        _sse({"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    payloads = [
        json.loads(line[5:].strip())
        for line in out.splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    ]
    joined = "".join(
        p["choices"][0]["delta"].get("content") or ""
        for p in payloads
        if isinstance(p["choices"][0]["delta"].get("content"), str)
    )
    assert "thinking done\n[[/think]]\n\nanswer" in joined
    # No double-newline (blank line) immediately before the closing tag:
    assert "\n\n[[/think]]" not in joined


@pytest.mark.asyncio
async def test_stream_handles_reasoning_split_across_chunks_with_trailing_newline():
    settings = Settings()
    chunks = [
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "part1"}}]}),
        _sse({"choices": [{"index": 0, "delta": {"reasoning_content": "\n\n"}}]}),
        _sse({"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    payloads = [
        json.loads(line[5:].strip())
        for line in out.splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    ]
    joined = "".join(
        p["choices"][0]["delta"].get("content") or ""
        for p in payloads
        if isinstance(p["choices"][0]["delta"].get("content"), str)
    )
    # Reasoning already ends with \n\n — proxy must not add yet another
    # newline before [[/think]]. (We can't retract the upstream's blank line,
    # but we must not make it worse.) After [[/think]] we always want a blank
    # line so Markdown renderers don't fold the next content onto the tag.
    assert "[[/think]]\n\nanswer" in joined
    assert "\n\n\n[[/think]]" not in joined
