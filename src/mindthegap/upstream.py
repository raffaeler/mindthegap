"""Upstream resolution — select the target upstream for each request.

Ported from llmhub's ``UpstreamResolver`` and ``UpstreamSelectionOptions``.
Resolution priority:

1. Request header (``X-Mindthegap-Upstream`` by default)
2. URL path prefix (first segment matches an upstream name)
3. ``model`` field in the JSON body matched against glob patterns
4. Configured default upstream
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import Settings, UpstreamOptions


@dataclass
class ResolvedUpstream:
    """The upstream selected for a request together with the path to forward."""

    name: str
    options: UpstreamOptions
    path: str  # rewritten path (stripped of any routing prefix)


def resolve_upstream(
    settings: Settings,
    path: str,
    body_bytes: bytes,
    request_headers: dict[str, str] | None = None,
) -> ResolvedUpstream | None:
    """Select an upstream for *path* considering headers, path prefix, model,
    and default.  Returns ``None`` when no upstream matches.

    *body_bytes* is the raw request body (used for model-based routing).
    *request_headers* is the incoming client headers (used for header-based
    routing).  Passing ``None`` skips header-based selection.
    """
    upstreams = settings.upstreams
    selection = settings.upstream_selection

    # 1. Header-based selection
    if request_headers and selection.header_name:
        hv = request_headers.get(selection.header_name.lower())
        if hv and hv in upstreams:
            return ResolvedUpstream(name=hv, options=upstreams[hv], path=path)

    # 2. Path-prefix-based selection
    if selection.path_prefix_enabled:
        stripped_path = path.lstrip("/")
        # Split off query string before extracting the first segment
        path_part, sep, query_part = stripped_path.partition("?")
        if path_part and "/" in path_part:
            first_seg = path_part.split("/", 1)[0]
            if first_seg and first_seg in upstreams:
                rest = path_part[len(first_seg):] or "/"
                if not rest.startswith("/"):
                    rest = "/" + rest
                if query_part:
                    rest += "?" + query_part
                return ResolvedUpstream(name=first_seg, options=upstreams[first_seg], path=rest)
        elif path_part and path_part in upstreams:
            rest = "/" if query_part else ""
            if query_part:
                rest += "?" + query_part
            return ResolvedUpstream(name=path_part, options=upstreams[path_part], path=rest)

    # 3. Model-based selection (glob matching)
    if body_bytes and selection.model_mapping:
        model = _try_read_model(body_bytes, selection.model_mapping_max_body_bytes)
        if model:
            upstream_name = _try_match_model(selection.model_mapping, model)
            if upstream_name and upstream_name in upstreams:
                return ResolvedUpstream(
                    name=upstream_name, options=upstreams[upstream_name], path=path
                )

    # 4. Default upstream
    if settings.default_upstream and settings.default_upstream in upstreams:
        return ResolvedUpstream(
            name=settings.default_upstream,
            options=upstreams[settings.default_upstream],
            path=path,
        )

    return None


# ── model parsing ───────────────────────────────────────────────────────────


def _try_read_model(body: bytes, max_scan_bytes: int) -> str | None:
    """Extract the ``model`` field value from a JSON request body.

    Only scans the first *max_scan_bytes* bytes.  Returns ``None`` when the
    body isn't JSON, the field is missing, or the value isn't a string.
    """
    scan_len = min(len(body), max_scan_bytes)
    if scan_len <= 0:
        return None
    try:
        data: Any = json.loads(body[:scan_len])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    model = data.get("model")
    return model if isinstance(model, str) and model else None


# ── glob matching ───────────────────────────────────────────────────────────


def _try_match_model(mapping: dict[str, str], model: str) -> str | None:
    """Match *model* against the keys in *mapping*.

    Exact matches win.  Otherwise the first glob key that matches (using
    ``*`` as the only wildcard) is returned.
    """
    # Exact match
    if model in mapping:
        return mapping[model]

    # Glob matching
    for pattern, upstream_name in mapping.items():
        if glob_match(pattern, model):
            return upstream_name

    return None


def glob_match(pattern: str, value: str) -> bool:
    """Simple glob match with ``*`` as the only wildcard.

    Matches llmhub's ``GlobMatch``: ``*`` acts as a greedy prefix/suffix/
    interior wildcard.  Anchoring is implicit: the pattern is a full-string
    pattern, meaning ``deepseek-*`` matches ``deepseek-chat`` but not
    ``x-deepseek-chat``.
    """
    if "*" not in pattern:
        return False

    parts = pattern.split("*")
    idx = 0
    for i, part in enumerate(parts):
        if not part:
            continue
        found = value.find(part, idx)
        if found < 0:
            return False
        # First part must match at position 0
        if i == 0 and found != 0:
            return False
        idx = found + len(part)

    # Last part must match at the end
    if parts and parts[-1] and not value.endswith(parts[-1]):
        return False
    return True
