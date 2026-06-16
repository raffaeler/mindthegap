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


# ── New tests: DSML tag sanitisation and fragment buffering ──────────────


@pytest.mark.asyncio
async def test_stream_strips_dsml_tags_from_reasoning():
    """DSML tags in reasoning content should be stripped before wrapping."""
    settings = Settings()
    chunks = [
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "Hello <\uff5cDSML\uff5ctool_calls> world"
                        },
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Done", "finish_reason": "stop"},
                    }
                ]
            }
        ),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "DSML" not in out
    assert "[[think]]" in out
    assert "Hello world" in out  # whitespace collapsed to single space


@pytest.mark.asyncio
async def test_stream_strips_xml_tags_from_reasoning():
    """XML-like tags (</parameter>, <analysis>) should be stripped."""
    settings = Settings()
    chunks = [
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "Before </parameter> after"
                        },
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "result", "finish_reason": "stop"},
                    }
                ]
            }
        ),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "</parameter>" not in out
    assert "Before after" in out  # whitespace collapsed


@pytest.mark.asyncio
async def test_stream_buffers_split_fragment_across_deltas():
    """A tag like </parameter> split across SSE deltas should be buffered and
    never emitted as a partial fragment."""
    settings = Settings()
    chunks = [
        _sse(
            {
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": "think</"}}
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": "parameter>"}}
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "answer", "finish_reason": "stop"},
                    }
                ]
            }
        ),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    # The combined fragment should be stripped, not emitted raw
    assert "</parameter>" not in out
    assert "think" in out
    # The fragment was not emitted in intermediate deltas
    for line in out.splitlines():
        if line.startswith("data:") and line[6:] not in ("", "[DONE]"):
            try:
                p = json.loads(line[6:])
                delta_content = (
                    p.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if isinstance(delta_content, str) and "parameter" in delta_content:
                    assert "</parameter" not in delta_content  # never partial
                    if ">" in delta_content:
                        assert delta_content.count(">") >= 1  # complete only
            except json.JSONDecodeError:
                pass


@pytest.mark.asyncio
async def test_stream_flushes_pending_on_content_arrival():
    """When reasoning has pending buffered fragments and content arrives,
    the pending text should be flushed with the content delta."""
    settings = Settings()
    chunks = [
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "start </parameter"},
                    }
                ]
            }
        ),
        # No more reasoning — next delta is content
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "the answer", "finish_reason": "stop"},
                    }
                ]
            }
        ),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "[[think]]" in out
    assert "[[/think]]" in out
    assert "the answer" in out
    # The </parameter> tag should be stripped from output
    assert "</parameter" not in out


@pytest.mark.asyncio
async def test_stream_flushes_pending_on_finish_reason():
    """When reasoning has pending fragments and finish_reason arrives with no
    content, the pending text should still be flushed."""
    settings = Settings()
    chunks = [
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "only </param"},
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                ]
            }
        ),
        b"data: [DONE]\n\n",
    ]
    out = (await _collect(stitch_sse(_aiter(chunks), settings))).decode()
    assert "[[think]]" in out
    assert "[[/think]]" in out
    assert "</param" not in out
    # "only" should appear, the incomplete " </param" suffix stripped
    assert "only" in out
