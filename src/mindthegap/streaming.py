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

from .cache import ReasoningCache
from .config import Settings
from .sanitize import has_incomplete_fragment, sanitize_reasoning_text


def _strip_incomplete_fragment(text: str) -> str:
    """When *text* ends with an incomplete XML-like fragment (``<`` without
    ``>``), strip everything from the last ``<`` to the end.  Returns the
    cleaned text (may be empty)."""
    stripped = text.strip()
    if not has_incomplete_fragment(stripped):
        return text
    # Find the last '<' and drop it and everything after.
    last_lt = text.rfind("<")
    if last_lt >= 0:
        return text[:last_lt].rstrip()
    return text


@dataclass
class _ChoiceState:
    opened: bool = False
    closed: bool = False
    # Trailing newlines already emitted at the tail of the reasoning stream.
    # Used to guarantee exactly one blank line (== 2 newlines) before
    # ``</think>`` without ever introducing extra blank lines inside the
    # think block when the upstream reasoning already ends with newlines.
    trailing_newlines: int = 0
    # Full reasoning text accumulated across deltas, kept verbatim so we
    # can stash it in the sidecar cache keyed by the streamed tool_call
    # ids when the choice completes.
    reasoning_buffer: list[str] = field(default_factory=list)
    # Tool call ids observed in this choice's tool_calls deltas. The
    # ``id`` field is typically only present in the first delta for each
    # tool call \u2014 we record every one we see, deduplicated and
    # order-preserving.
    tool_call_ids: list[str] = field(default_factory=list)
    # Set once we've persisted reasoning into the cache for this choice
    # so we don't store the same text under the same ids more than once.
    cache_flushed: bool = False
    # Buffered reasoning text that appears to contain an incomplete XML/DSML
    # fragment (contains ``<`` but no ``>``).  The text is held here until the
    # next delta completes the tag or a flush event (content / finish_reason)
    # arrives.  Ported from llmhub's ``ChoiceStitchState.PendingReasoningText``.
    pending_reasoning_text: str = ""


@dataclass
class _StreamState:
    per_choice: dict[int, _ChoiceState] = field(default_factory=dict)

    def get(self, idx: int) -> _ChoiceState:
        st = self.per_choice.get(idx)
        if st is None:
            st = _ChoiceState()
            self.per_choice[idx] = st
        return st


# ── surrogate pair protection ─────────────────────────────────────────────
# CESU-8 encodes U+D800..U+DBFF as ED A0-BF 80-BF.  Splitting such a
# 3-byte sequence across two SSE output chunks would leave the downstream
# JSON parser with a lone/invalid surrogate.  We detect the incomplete
# prefix and buffer it for the next chunk.
_HIGH_SURROGATE_PREFIX = bytes([0xED, 0xA0])  # first two bytes of U+D800
_HIGH_SURROGATE_MAX_B2 = 0xBF


def _ends_with_high_surrogate_start(b: bytes) -> bool:
    """Return True when *b* ends with the first 2 bytes of a CESU-8
    high-surrogate encoding (0xED 0xA0–0xBF) without its third byte."""
    if len(b) < 2:
        return False
    return b[-2] == _HIGH_SURROGATE_PREFIX[0] and _HIGH_SURROGATE_PREFIX[1] <= b[-1] <= _HIGH_SURROGATE_MAX_B2


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


def _collect_tool_call_ids(delta: dict[str, Any], st: _ChoiceState) -> None:
    """Record any tool_call ids appearing in this delta's tool_calls list.

    OpenAI streams tool_calls as partial deltas; the ``id`` field is
    typically present only in the first delta for each call. We
    deduplicate while preserving order so we can later key the sidecar
    cache by the same ids the client will round-trip.
    """
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_id = tc.get("id")
        if isinstance(tc_id, str) and tc_id and tc_id not in st.tool_call_ids:
            st.tool_call_ids.append(tc_id)


def _flush_to_cache(st: _ChoiceState, cache: ReasoningCache | None) -> None:
    if cache is None or st.cache_flushed:
        return
    if not st.tool_call_ids or not st.reasoning_buffer:
        return
    reasoning_text = "".join(st.reasoning_buffer)
    if not reasoning_text:
        return
    for tc_id in st.tool_call_ids:
        cache.put(tc_id, reasoning_text)
    st.cache_flushed = True


def _rewrite_choice(
    choice: dict[str, Any],
    state: _StreamState,
    settings: Settings,
    cache: ReasoningCache | None = None,
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

    _collect_tool_call_ids(new_delta, st)

    pieces: list[str] = []
    if isinstance(reasoning, str) and reasoning:
        # Combine with any buffered pending fragment from a prior delta
        combined = st.pending_reasoning_text + reasoning
        sanitized = sanitize_reasoning_text(combined)
        incomplete = has_incomplete_fragment(combined)

        has_real_content = isinstance(content, str) and content != ""
        finish_reason = new_choice.get("finish_reason")
        has_finish = finish_reason is not None

        if has_real_content or has_finish:
            # Content or finish_reason arrived — flush everything now.
            if combined:
                st.reasoning_buffer.append(combined)
            if not st.opened and sanitized:
                pieces.append(f"{settings.think_tag_open}  \n")
                st.opened = True
                st.trailing_newlines = 1
            if sanitized:
                # Avoid prepending a bare newline if the reasoning itself ends
                # with newlines (keep trailing tally)
                pieces.append(sanitized)
                if sanitized.strip("\n") == "":
                    st.trailing_newlines += len(sanitized)
                else:
                    st.trailing_newlines = _count_trailing_newlines(sanitized)

            if has_real_content:
                pieces.append(
                    f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n"
                )
                st.closed = True
                pieces.append(content)  # type: ignore[arg-type]
            elif has_finish:
                pieces.append(
                    f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n"
                )
                st.closed = True

            st.pending_reasoning_text = ""

        elif incomplete:
            # Fragment looks like a split tag — buffer it for the next delta.
            st.pending_reasoning_text = combined
            # Remove reasoning_content from this delta so the client doesn't
            # see a partial, possibly malformed fragment.
            # (already popped from new_delta above)

        else:
            # Pure reasoning delta, no split detected — emit inline.
            st.reasoning_buffer.append(combined)
            if not st.opened and sanitized:
                pieces.append(f"{settings.think_tag_open}  \n")
                st.opened = True
                st.trailing_newlines = 1
            if sanitized:
                pieces.append(sanitized)
                if sanitized.strip("\n") == "":
                    st.trailing_newlines += len(sanitized)
                else:
                    st.trailing_newlines = _count_trailing_newlines(sanitized)
            st.pending_reasoning_text = ""

    else:
        # No reasoning in this delta, but we might have pending fragments.
        has_real_content = isinstance(content, str) and content != ""
        finish_reason = new_choice.get("finish_reason")
        has_finish = finish_reason is not None

        if has_real_content and st.pending_reasoning_text:
            # Flush pending reasoning before the real content.
            pending_sanitized = sanitize_reasoning_text(st.pending_reasoning_text)
            pending_sanitized = _strip_incomplete_fragment(pending_sanitized)
            if not st.opened and pending_sanitized:
                pieces.append(f"{settings.think_tag_open}  \n")
                st.opened = True
            if pending_sanitized:
                pieces.append(pending_sanitized)
            if st.opened:
                pieces.append(
                    f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n"
                )
                st.closed = True
            st.pending_reasoning_text = ""

        if has_real_content and st.opened and not st.closed:
            pieces.append(f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n")
            st.closed = True

        if has_real_content:
            pieces.append(content)  # type: ignore[arg-type]

        if finish_reason and st.opened and not st.closed:
            if st.pending_reasoning_text:
                pending_sanitized = sanitize_reasoning_text(st.pending_reasoning_text)
                pending_sanitized = _strip_incomplete_fragment(pending_sanitized)
                if pending_sanitized:
                    pieces.append(pending_sanitized)
                st.pending_reasoning_text = ""
            pieces.append(f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n")
            st.closed = True

        # Finish with pending but think never opened — still flush.
        if has_finish and st.pending_reasoning_text:
            pending_sanitized = sanitize_reasoning_text(st.pending_reasoning_text)
            pending_sanitized = _strip_incomplete_fragment(pending_sanitized)
            if not st.opened and pending_sanitized:
                pieces.append(f"{settings.think_tag_open}  \n")
                st.opened = True
                st.trailing_newlines = 1
            if pending_sanitized:
                pieces.append(pending_sanitized)
            if st.opened:
                pieces.append(
                    f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n"
                )
                st.closed = True
            st.pending_reasoning_text = ""

    # Once the choice has reached a terminal state, persist the reasoning
    # under every observed tool_call_id so follow-up turns can recover it.
    if finish_reason:
        _flush_to_cache(st, cache)

    if pieces:
        new_delta["content"] = "".join(pieces)
    elif "content" in new_delta and new_delta["content"] is None:
        # leave as-is (e.g. role-only delta)
        pass

    new_choice["delta"] = new_delta
    return new_choice


def _process_chunk(
    payload: dict[str, Any],
    state: _StreamState,
    settings: Settings,
    cache: ReasoningCache | None = None,
) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    new_payload = dict(payload)
    new_payload["choices"] = [
        _rewrite_choice(c, state, settings, cache=cache) if isinstance(c, dict) else c
        for c in choices
    ]
    return new_payload


async def stitch_sse(
    upstream: AsyncIterator[bytes],
    settings: Settings,
    cache: ReasoningCache | None = None,
) -> AsyncIterator[bytes]:
    """Rewrite an upstream SSE byte stream so reasoning becomes inline content."""
    state = _StreamState()
    buffer = b""
    pending_surrogate = b""  # buffered high-surrogate prefix (CESU-8 edge case)
    async for chunk in upstream:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            for out in _process_line(line, state, settings, cache=cache):
                out_b = pending_surrogate + out
                if _ends_with_high_surrogate_start(out_b):
                    pending_surrogate = out_b[-2:]
                    # Don't yield the partial surrogate tail yet — wait for the
                    # next output where the third byte of the CESU-8 sequence
                    # will complete the character.
                    if len(out_b) > 2:
                        yield out_b[:-2] + b"\n"
                else:
                    pending_surrogate = b""
                    yield out_b + b"\n"
    if buffer:
        for out in _process_line(buffer, state, settings, cache=cache):
            out_b = pending_surrogate + out
            yield out_b + b"\n"
            pending_surrogate = b""
    # Final safety net: if the upstream ended without ever closing an opened
    # <think> block (truncated stream, missing finish_reason, no [DONE]),
    # emit synthetic close chunks so the assistant message persisted by the
    # client always has a matching </think>.
    for idx, st in state.per_choice.items():
        if st.opened and not st.closed:
            yield _synthetic_close_chunk(idx, st, settings)
            st.closed = True
        # Last-chance flush in case finish_reason was never observed.
        _flush_to_cache(st, cache)
    # Flush any leftover surrogate prefix — this would only happen if the
    # upstream stream is truly malformed (lone surrogate at EOF).
    if pending_surrogate:
        yield pending_surrogate + b"\n"


def _synthetic_close_chunk(idx: int, st: _ChoiceState, settings: Settings) -> bytes:
    delta = {"content": f"{_close_padding(st.trailing_newlines)}{settings.think_tag_close}\n\n"}
    payload = {"choices": [{"index": idx, "delta": delta}]}
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


def _process_line(
    line: bytes,
    state: _StreamState,
    settings: Settings,
    cache: ReasoningCache | None = None,
) -> list[bytes]:
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
            _flush_to_cache(st, cache)
        return [*prelude, stripped]
    if data == b"":
        return [stripped]
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return [stripped]
    if not isinstance(payload, dict):
        return [stripped]
    new_payload = _process_chunk(payload, state, settings, cache=cache)
    return [b"data: " + json.dumps(new_payload, ensure_ascii=False).encode("utf-8")]
