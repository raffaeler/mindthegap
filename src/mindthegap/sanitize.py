"""Reasoning text sanitization — strips DSML/XML tags and normalizes whitespace.

Ported from llmhub's ``ReasoningContentJsonHelpers.SanitizeReasoningText`` and
``HasIncompleteReasoningFragment``. The upstream reasoning stream can contain
DSML tags (``<‖DSML‖tool_calls>``, ``</‖DSML‖invoke>``, ``<‖DSML‖parameter …>``)
and standard XML tags (``</parameter>``, ``<analysis>``, ``</invoke>``) that
would leak into the client-visible content and break conversation rendering.
"""

from __future__ import annotations

import re

# ── compiled patterns (built once) ──────────────────────────────────────────

# DSML tags use ‖ (U+FF5C fullwidth vertical bar) which standard XML regex misses.
_DSML_PATTERN = re.compile(r"</?\uff5cDSML\uff5c[A-Za-z0-9_]*(?:\s[^>]*)?>", re.IGNORECASE)

# Standard XML-like tags: </parameter>, <analysis>, </invoke>, etc.
_XML_TAG_PATTERN = re.compile(r"</?[A-Za-z][A-Za-z0-9_-]*>", re.IGNORECASE)

# Collapse runs of horizontal whitespace (spaces, tabs, form feeds, vertical tabs)
# into a single space.  Does NOT touch newlines.
_WHITESPACE_RUN_PATTERN = re.compile(r"[ \t\f\v]+")


def sanitize_reasoning_text(text: str | None) -> str:
    """Sanitize upstream reasoning text for client consumption.

    Three-step normalisation matching llmhub:

    1. Normalise line endings (``\\r\\n`` → ``\\n``, ``\\r`` → ``\\n``)
    2. Strip DSML tags — ``<‖DSML‖tool_calls>``, ``</‖DSML‖invoke>``,
       ``<‖DSML‖parameter name="..." string="true">`` etc.
    3. Strip standard XML-like tags — ``</parameter>``, ``<analysis>``,
       ``</invoke>``, etc.
    4. Collapse horizontal whitespace runs to a single space

    Returns an empty string when *text* is ``None`` or whitespace-only.
    """
    if text is None or text == "":
        return ""
    if text.strip() == "":
        return ""
    # Step 1 — normalise line endings
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    # Step 2 — strip DSML tags (‖ = U+FF5C)
    dsml_stripped = _DSML_PATTERN.sub("", normalised)
    # Step 3 — strip standard XML-like tags
    stripped = _XML_TAG_PATTERN.sub("", dsml_stripped)
    # Step 4 — collapse horizontal whitespace runs
    return _WHITESPACE_RUN_PATTERN.sub(" ", stripped)


def has_incomplete_fragment(text: str | None) -> bool:
    """Return ``True`` when *text* contains ``<`` but no ``>``.

    Used to detect tags (like ``</parameter>``) that were split across SSE
    delta boundaries.  When this returns ``True`` the caller should buffer the
    text and re-test on the next delta arrival instead of emitting it
    immediately.
    """
    if text is None or text == "":
        return False
    # Simple heuristic: a '<' without any '>' means there's an open tag that
    # hasn't been completed yet.  Empty / whitespace-only strings are not
    # considered fragments.
    stripped = text.strip()
    if not stripped:
        return False
    return "<" in stripped and ">" not in stripped
