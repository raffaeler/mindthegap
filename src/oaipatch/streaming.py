"""SSE stitcher for chat.completion.chunk streams.

The upstream sends Server-Sent Events whose ``data:`` payload is JSON shaped
like ``{"choices": [{"index": int, "delta": {...}}], ...}``. For reasoner
models the early deltas carry ``reasoning_content`` instead of ``content``.
We rewrite them so the reasoning text appears inside ``content`` between
the configured think tags, then strip ``reasoning_content`` from every
forwarded chunk.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .config import Settings


@dataclass
class _ChoiceState:
    opened: bool = False
    closed: bool = False


@dataclass
class _StreamState:
    per_choice: dict[int, _ChoiceState] = field(default_factory=dict)

    def get(self, idx: int) -> _ChoiceState:
        st = self.per_choice.get(idx)
        if st is None:
            st = _ChoiceState()
            self.per_choice[idx] = st
        return st


def _rewrite_choice(
    choice: dict[str, Any],
    state: _StreamState,
    settings: Settings,
) -> dict[str, Any]:
    new_choice = dict(choice)
    delta = new_choice.get("delta")
    if not isinstance(delta, dict):
        return new_choice

    new_delta = dict(delta)
    reasoning = new_delta.pop("reasoning_content", None)
    content = new_delta.get("content")
    idx = choice.get("index", 0)
    if not isinstance(idx, int):
        idx = 0
    st = state.get(idx)

    pieces: list[str] = []
    if isinstance(reasoning, str) and reasoning:
        if not st.opened:
            pieces.append(f"{settings.think_tag_open}\n")
            st.opened = True
        pieces.append(reasoning)

    has_real_content = isinstance(content, str) and content != ""
    if has_real_content and st.opened and not st.closed:
        pieces.append(f"\n{settings.think_tag_close}\n")
        st.closed = True

    if has_real_content:
        pieces.append(content)  # type: ignore[arg-type]

    finish_reason = new_choice.get("finish_reason")
    if finish_reason and st.opened and not st.closed:
        pieces.append(f"\n{settings.think_tag_close}\n")
        st.closed = True

    if pieces:
        new_delta["content"] = "".join(pieces)
    elif "content" in new_delta and new_delta["content"] is None:
        # leave as-is (e.g. role-only delta)
        pass

    new_choice["delta"] = new_delta
    return new_choice


def _process_chunk(
    payload: dict[str, Any], state: _StreamState, settings: Settings
) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    new_payload = dict(payload)
    new_payload["choices"] = [
        _rewrite_choice(c, state, settings) if isinstance(c, dict) else c for c in choices
    ]
    return new_payload


async def stitch_sse(
    upstream: AsyncIterator[bytes],
    settings: Settings,
) -> AsyncIterator[bytes]:
    """Rewrite an upstream SSE byte stream so reasoning becomes inline content."""
    state = _StreamState()
    buffer = b""
    async for chunk in upstream:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            out = _process_line(line, state, settings)
            if out is not None:
                yield out + b"\n"
    if buffer:
        out = _process_line(buffer, state, settings)
        if out is not None:
            yield out


def _process_line(line: bytes, state: _StreamState, settings: Settings) -> bytes | None:
    stripped = line.rstrip(b"\r")
    if not stripped.startswith(b"data:"):
        return stripped
    data = stripped[5:].lstrip()
    if data == b"[DONE]" or data == b"":
        return stripped
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return stripped
    if not isinstance(payload, dict):
        return stripped
    new_payload = _process_chunk(payload, state, settings)
    return b"data: " + json.dumps(new_payload, ensure_ascii=False).encode("utf-8")
