"""Pure stitch / unstitch transforms for chat.completions payloads."""

from __future__ import annotations

import re
from typing import Any

from .config import Settings, UnstitchMode


def _think_pattern(settings: Settings) -> re.Pattern[str]:
    return re.compile(
        rf"^\s*{re.escape(settings.think_tag_open)}(.*?){re.escape(settings.think_tag_close)}\s*",
        re.DOTALL,
    )


def stitch_message(message: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Move ``reasoning_content`` into ``content`` wrapped in think tags.

    Returns a new dict; the input is not mutated. ``reasoning_content`` is
    always removed from the output so vanilla clients don't see it.
    """
    out = dict(message)
    reasoning = out.pop("reasoning_content", None)
    if reasoning is None or reasoning == "":
        return out
    if not isinstance(reasoning, str):
        return out
    content = out.get("content")
    open_tag = settings.think_tag_open
    close_tag = settings.think_tag_close
    wrapped = f"{open_tag}\n{reasoning}\n{close_tag}\n"
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
) -> list[dict[str, Any]]:
    """Extract leading ``<think>...</think>`` from assistant messages.

    ``mode`` controls behavior:
      - ``"forward"``: move extracted text into ``reasoning_content`` and
        strip the tags from ``content`` (the only sane choice for reasoner
        models that require ``reasoning_content`` on every prior turn).
      - ``"drop"``: strip the tags + text from ``content`` entirely.
      - ``"keep"``: leave the message untouched.
    """
    if mode == "keep":
        return [dict(m) for m in messages]

    pattern = _think_pattern(settings)
    out: list[dict[str, Any]] = []
    for msg in messages:
        new = dict(msg)
        if new.get("role") != "assistant":
            out.append(new)
            continue
        content = new.get("content")
        if not isinstance(content, str) or not content:
            out.append(new)
            continue
        match = pattern.match(content)
        if not match:
            out.append(new)
            continue
        reasoning = match.group(1).strip("\n")
        stripped = content[match.end() :]
        new["content"] = stripped
        if mode == "forward":
            new["reasoning_content"] = reasoning
        out.append(new)
    return out


def transform_request_body(
    body: dict[str, Any],
    settings: Settings,
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
    new_body["messages"] = unstitch_messages(messages, settings, mode)
    return new_body


def transform_response_body(
    body: dict[str, Any],
    settings: Settings,
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
            new_choice["message"] = stitch_message(msg, settings)
        new_choices.append(new_choice)
    new_body = dict(body)
    new_body["choices"] = new_choices
    return new_body
