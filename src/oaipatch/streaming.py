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
    # Trailing newlines already emitted at the tail of the reasoning stream.
    # Used to guarantee exactly one blank line (== 2 newlines) before
    # ``</think>`` without ever introducing extra blank lines inside the
    # think block when the upstream reasoning already ends with newlines.
    trailing_newlines: int = 0


@dataclass
class _StreamState:
    per_choice: dict[int, _ChoiceState] = field(default_factory=dict)

    def get(self, idx: int) -> _ChoiceState:
        st = self.per_choice.get(idx)
        if st is None:
            st = _ChoiceState()
            self.per_choice[idx] = st
        return st


def _count_trailing_newlines(s: str) -> int:
    n = 0
    for ch in reversed(s):
        if ch == "\n":
            n += 1
        else:
            break
    return n


def _close_padding(trailing: int) -> str:
    """Return the prefix to prepend to ``</think>`` so it always renders on
    its own line WITHOUT a blank line above it.

    We use a Markdown hard line break (two trailing spaces + ``\\n``) so the
    client's Markdown renderer doesn't collapse the bare ``\\n`` into a
    space and put the closing tag inline with the reasoning text.

    - ``trailing == 0``: emit ``"  \\n"`` (hard break)
    - ``trailing >= 1``: the upstream already wrote ``\\n`` at the tail, so
      we can't retroactively insert two spaces before it. The best we can do
      is emit nothing and accept the bare ``\\n`` (renderers will then show
      ``</think>`` flowed inline OR on its own line depending on parser).
    """
    return "" if trailing >= 1 else "  \n"


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
            st.trailing_newlines = 1
        pieces.append(reasoning)
        # Update trailing-newline tally based on the new piece.
        if reasoning.strip("\n") == "":
            st.trailing_newlines += len(reasoning)
        else:
            st.trailing_newlines = _count_trailing_newlines(reasoning)

    has_real_content = isinstance(content, str) and content != ""
    if has_real_content and st.opened and not st.closed:
        pieces.append(f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n")
        st.closed = True

    if has_real_content:
        pieces.append(content)  # type: ignore[arg-type]

    finish_reason = new_choice.get("finish_reason")
    if finish_reason and st.opened and not st.closed:
        pieces.append(f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n")
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
            for out in _process_line(line, state, settings):
                yield out + b"\n"
    if buffer:
        for out in _process_line(buffer, state, settings):
            yield out
    # Final safety net: if the upstream ended without ever closing an opened
    # <think> block (truncated stream, missing finish_reason, no [DONE]),
    # emit synthetic close chunks so the assistant message persisted by the
    # client always has a matching </think>.
    for idx, st in state.per_choice.items():
        if st.opened and not st.closed:
            yield _synthetic_close_chunk(idx, st, settings)
            st.closed = True


def _synthetic_close_chunk(idx: int, st: _ChoiceState, settings: Settings) -> bytes:
    delta = {"content": f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n"}
    payload = {"choices": [{"index": idx, "delta": delta}]}
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


def _process_line(line: bytes, state: _StreamState, settings: Settings) -> list[bytes]:
    stripped = line.rstrip(b"\r")
    if not stripped.startswith(b"data:"):
        return [stripped]
    data = stripped[5:].lstrip()
    if data == b"[DONE]":
        # Flush any unterminated <think> blocks before signalling end-of-stream
        # so the client's persisted message has a matching </think>.
        prelude: list[bytes] = []
        for idx, st in state.per_choice.items():
            if st.opened and not st.closed:
                prelude.append(_synthetic_close_chunk(idx, st, settings))
                st.closed = True
        return [*prelude, stripped]
    if data == b"":
        return [stripped]
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return [stripped]
    if not isinstance(payload, dict):
        return [stripped]
    new_payload = _process_chunk(payload, state, settings)
    return [b"data: " + json.dumps(new_payload, ensure_ascii=False).encode("utf-8")]
