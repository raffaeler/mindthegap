"""Pure stitch / unstitch transforms for chat.completions payloads."""

from __future__ import annotations

import re
from typing import Any

from .cache import ReasoningCache
from .config import Settings, UnstitchMode


def _think_pattern(settings: Settings) -> re.Pattern[str]:
    return re.compile(
        rf"^\s*{re.escape(settings.think_tag_open)}(.*?){re.escape(settings.think_tag_close)}\s*",
        re.DOTALL,
    )


def _unclosed_think_pattern(settings: Settings) -> re.Pattern[str]:
    """Match a leading ``<think>`` with no matching closing tag anywhere after.

    Captures everything from after the opening tag to the end of the string.
    Used to recover from truncated/streaming-broken assistant messages where
    the closing ``</think>`` was never emitted.
    """
    return re.compile(
        rf"^\s*{re.escape(settings.think_tag_open)}(?![\s\S]*{re.escape(settings.think_tag_close)})([\s\S]*)$",
    )


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    out: list[str] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            tc_id = tc.get("id")
            if isinstance(tc_id, str) and tc_id:
                out.append(tc_id)
    return out


def stitch_message(
    message: dict[str, Any],
    settings: Settings,
    cache: ReasoningCache | None = None,
) -> dict[str, Any]:
    """Move ``reasoning_content`` into ``content`` wrapped in think tags.

    Returns a new dict; the input is not mutated. ``reasoning_content`` is
    always removed from the output so vanilla clients don't see it.

    When the message also carries ``tool_calls`` and ``cache`` is provided,
    the original ``reasoning_content`` is additionally indexed under every
    ``tool_call_id`` so that follow-up requests (where clients typically
    drop the assistant ``content``) can recover it.
    """
    out = dict(message)
    reasoning = out.pop("reasoning_content", None)
    if reasoning is None or reasoning == "":
        return out
    if not isinstance(reasoning, str):
        return out
    if cache is not None:
        for tc_id in _tool_call_ids(out):
            cache.put(tc_id, reasoning)
    content = out.get("content")
    open_tag = settings.think_tag_open
    close_tag = settings.think_tag_close
    # Normalize the tail of the reasoning block: strip any trailing newlines
    # and append two spaces before the final \n. Those two trailing spaces
    # are a Markdown hard line break, which makes the closing </think>
    # render on its own line WITHOUT inserting a blank line above it.
    # (A bare \n would otherwise be collapsed to a space by the client's
    # Markdown renderer and the tag would appear inline with the reasoning.)
    reasoning_body = reasoning.rstrip("\n").rstrip()
    wrapped = f"{open_tag}  \n{reasoning_body}  \n{close_tag}\n\n"
    if isinstance(content, str) and content:
        out["content"] = wrapped + content
    else:
        # Preserve None (tool_calls case) by still attaching reasoning as content
        out["content"] = wrapped if content in (None, "") else content
    return out


def unstitch_messages(
    messages: list[dict[str, Any]],
    settings: Settings,
    mode: UnstitchMode,
    cache: ReasoningCache | None = None,
) -> list[dict[str, Any]]:
    """Extract leading ``<think>...</think>`` from assistant messages.

    ``mode`` controls behavior:
      - ``"forward"``: move extracted text into ``reasoning_content`` and
        strip the tags from ``content`` (the only sane choice for reasoner
        models that require ``reasoning_content`` on every prior turn).
      - ``"drop"``: strip the tags + text from ``content`` entirely.
      - ``"keep"``: leave the message untouched.

    When ``mode == "forward"`` and ``cache`` is provided, any assistant
    message that carries ``tool_calls`` but ends up without
    ``reasoning_content`` (because the client persisted ``content: null``
    for the tool-call turn and dropped the ``<think>`` block) gets its
    reasoning recovered from the sidecar cache, keyed by the first
    ``tool_call_id``. This is the workaround for clients (e.g. GitHub
    Copilot CLI) that strip non-standard text from tool-call assistant
    messages.
    """
    if mode == "keep":
        return [dict(m) for m in messages]

    pattern = _think_pattern(settings)
    unclosed = _unclosed_think_pattern(settings)
    out: list[dict[str, Any]] = []
    for msg in messages:
        new = dict(msg)
        if new.get("role") != "assistant":
            out.append(new)
            continue
        content = new.get("content")
        reasoning: str | None = None
        if isinstance(content, str) and content:
            match = pattern.match(content)
            if match:
                reasoning = match.group(1).strip("\n")
                stripped = content[match.end() :]
            else:
                # Recover from a truncated assistant message: leading
                # <think> without a matching </think> (upstream stream cut
                # off mid-reasoning). Treat the entire remainder as
                # reasoning so the forwarded request stays coherent for
                # reasoner models.
                unclosed_match = unclosed.match(content)
                if unclosed_match:
                    reasoning = unclosed_match.group(1).strip("\n")
                    stripped = ""
                else:
                    stripped = content
            # DeepSeek (and the OpenAI spec) expects assistant messages
            # that carry ``tool_calls`` to use ``content: null`` rather
            # than an empty string when there is no textual content. After
            # unstitching the leading ``<think>...</think>`` block the
            # residual content is often empty for tool-call messages, so
            # normalize it back to None.
            if stripped == "" and new.get("tool_calls"):
                new["content"] = None
            else:
                new["content"] = stripped
        # Sidecar recovery: if the assistant message has tool_calls but we
        # still don't have reasoning (either because the client dropped the
        # ``content`` of the tool-call turn entirely, or because no
        # ``<think>`` block was found), look it up by tool_call_id.
        if (
            mode == "forward"
            and reasoning is None
            and cache is not None
            and not isinstance(new.get("reasoning_content"), str)
            and new.get("tool_calls")
        ):
            for tc_id in _tool_call_ids(new):
                cached = cache.get(tc_id)
                if cached:
                    reasoning = cached
                    break
        if mode == "forward" and reasoning is not None:
            new["reasoning_content"] = reasoning
        out.append(new)
    return out


def transform_request_body(
    body: dict[str, Any],
    settings: Settings,
    cache: ReasoningCache | None = None,
) -> dict[str, Any]:
    """Apply unstitching to a /v1/chat/completions request body."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    model = body.get("model") if isinstance(body.get("model"), str) else None
    mode: UnstitchMode = (
        "forward" if settings.is_reasoner(model) else settings.unstitch_when_not_reasoner
    )
    new_body = dict(body)
    new_body["messages"] = unstitch_messages(messages, settings, mode, cache=cache)
    return new_body


def transform_response_body(
    body: dict[str, Any],
    settings: Settings,
    cache: ReasoningCache | None = None,
) -> dict[str, Any]:
    """Apply stitching to a non-streaming /v1/chat/completions response body."""
    choices = body.get("choices")
    if not isinstance(choices, list):
        return body
    new_choices: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            new_choices.append(choice)
            continue
        new_choice = dict(choice)
        msg = new_choice.get("message")
        if isinstance(msg, dict):
            new_choice["message"] = stitch_message(msg, settings, cache=cache)
        new_choices.append(new_choice)
    new_body = dict(body)
    new_body["choices"] = new_choices
    return new_body
