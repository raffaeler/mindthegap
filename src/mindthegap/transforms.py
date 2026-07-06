"""Pure stitch / unstitch transforms for chat.completions payloads."""

from __future__ import annotations

import re
from typing import Any

from .cache import ReasoningCache
from .config import Settings, UnstitchMode
from .sanitize import sanitize_reasoning_text


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
        # Also cache by function name — some clients (Android Studio)
        # rename tool_call_ids to the function name (e.g. "write_file").
        for tc in out.get("tool_calls", []):
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and fn_name:
                        cache.put(fn_name, reasoning)
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
    # Sanitise reasoning text: strip DSML/XML tags and normalise whitespace
    # before wrapping in think tags.  Ported from llmhub.
    reasoning_body = sanitize_reasoning_text(reasoning_body)
    wrapped = f"{open_tag}  \n{reasoning_body}  \n{close_tag}\n\n"
    if isinstance(content, str) and content:
        out["content"] = wrapped + content
    else:
        # Preserve None (tool_calls case) by still attaching reasoning as content
        out["content"] = wrapped if content in (None, "") else content
    return out


def _merge_fragmented_assistant_messages(
    messages: list[dict[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Merge consecutive assistant messages that don't carry tool_calls.

    Some clients (notably Android Studio's Java OpenAI client) split the
    stitched think-tagged content into word-level fragments, e.g.::

        [6] reasoning_content="  \\nThe"  content=""
        [7] content="user"
        [8] content="wants"
        ...
        [31] content="."
        [32] content="[[/think]]"

    This defeats the unstitch regex which expects contiguous
    ``[[think]]...[[/think]]`` in a single message.  We merge them back
    into one message; the unstitch logic handles extraction.
    """
    merged: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if msg.get("role") != "assistant" or msg.get("tool_calls"):
            merged.append(msg)
            i += 1
            continue

        # Collect consecutive assistant messages without tool_calls
        group = [msg]
        j = i + 1
        while j < n and messages[j].get("role") == "assistant" and not messages[j].get("tool_calls"):
            group.append(messages[j])
            j += 1

        if len(group) == 1:
            merged.append(msg)
            i = j
            continue

        # ── Merge the group ──────────────────────────────────────────
        merged_reasoning_parts: list[str] = []
        merged_content_parts: list[str] = []
        for m in group:
            rc = m.get("reasoning_content")
            if isinstance(rc, str) and rc:
                merged_reasoning_parts.append(rc)
            c = m.get("content")
            if isinstance(c, str) and c:
                merged_content_parts.append(c)

        merged_reasoning = "".join(merged_reasoning_parts)
        # Join content fragments with spaces (they're word-level splits)
        merged_content = " ".join(merged_content_parts)

        merged_msg = dict(group[0])
        # Preserve reasoning_content from fragments (may be partial).
        # The unstitch fallback will combine it with content-extracted reasoning.
        if merged_reasoning:
            merged_msg["reasoning_content"] = merged_reasoning
        elif "reasoning_content" in merged_msg:
            del merged_msg["reasoning_content"]
        merged_msg["content"] = merged_content

        merged.append(merged_msg)
        i = j

    return merged


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

    # ── Pre-process: merge fragmented assistant messages ───────────
    messages = _merge_fragmented_assistant_messages(messages, settings)

    pattern = _think_pattern(settings)
    unclosed = _unclosed_think_pattern(settings)
    out: list[dict[str, Any]] = []
    last_assistant_reasoning: str | None = None  # track reasoning for tool_calls copy
    for msg in messages:
        new = dict(msg)
        # Deepseek does not support developer role, so we convert it to system role.
        if new.get("role") == "developer":
            new["role"] = "system"
            out.append(new)
            continue
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
                    # Fallback: closing think tag without opening tag.
                    # Some clients (Android Studio) consume the [[think]]
                    # opening tag but preserve [[/think]].
                    close_tag = settings.think_tag_close
                    if close_tag in content and settings.think_tag_open not in content:
                        idx = content.index(close_tag)
                        extracted = content[:idx].strip()
                        # Combine with any reasoning_content already on the
                        # message (from prior merge of fragmented messages)
                        existing_rc = new.get("reasoning_content")
                        if isinstance(existing_rc, str) and existing_rc:
                            reasoning = existing_rc.rstrip() + " " + extracted if extracted else existing_rc
                        else:
                            reasoning = extracted
                        stripped = content[idx + len(close_tag):].strip()
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
            # Also try function name (Android Studio renames tool_call_ids)
            if reasoning is None:
                for tc in new.get("tool_calls", []):
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        if isinstance(fn, dict):
                            fn_name = fn.get("name")
                            if isinstance(fn_name, str) and fn_name:
                                cached = cache.get(fn_name)
                                if cached:
                                    reasoning = cached
                                    break
        # ── Final fallback: copy reasoning from preceding assistant msg ──
        # Some clients (Android Studio) split a single DeepSeek response
        # (reasoning + tool_calls) into separate assistant messages.
        # When the tool_calls message lacks reasoning_content, carry over
        # the reasoning from the immediately preceding assistant message.
        if (
            mode == "forward"
            and reasoning is None
            and last_assistant_reasoning is not None
            and new.get("tool_calls")
            and not isinstance(new.get("reasoning_content"), str)
        ):
            reasoning = last_assistant_reasoning
        if mode == "forward" and reasoning is not None:
            if (reasoning.count('\n') > 1) and (reasoning.count(' ') == 0):
                reasoning = reasoning.replace('\n', ' ')
            new["reasoning_content"] = reasoning
        # Track last reasoning_content for carryover to split tool_calls msgs
        if isinstance(new.get("reasoning_content"), str) and new["reasoning_content"]:
            last_assistant_reasoning = new["reasoning_content"]
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

    # Apply model-specific parameter overrides (temperature, top_p, etc.)
    params = settings.get_model_params(model)
    if params:
        new_body.update(params)

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
