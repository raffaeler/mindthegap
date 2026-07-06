"""FastAPI app exposing the stitch/unstitch proxy."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .cache import ReasoningCache
from .config import Settings, load_settings
from .streaming import stitch_sse
from .transforms import transform_request_body, transform_response_body
from .upstream import resolve_upstream

logger = logging.getLogger("mindthegap")

# Header values to redact in diagnostic dumps so secrets never reach the log.
_REDACT_HEADERS = {"authorization", "x-api-key", "api-key", "proxy-authorization"}


def _redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: ("<redacted>" if k.lower() in _REDACT_HEADERS else v) for k, v in headers.items()}


def _safe_decode(payload: bytes, limit: int = 65536) -> str:
    text = payload.decode("utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + f"... <truncated, total {len(text)} chars>"
    return text


def _summarize_messages(payload: bytes) -> str | None:
    """Return a compact per-message summary so we can always see the full
    message sequence (role / tool_calls / reasoning_content presence /
    content preview) even when the raw body would otherwise be truncated.

    Returns ``None`` if the payload isn't a JSON object with a ``messages``
    list — in which case the caller should just rely on the raw dump.
    """
    try:
        data = json.loads(payload.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    lines: list[str] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict):
            lines.append(f"  [{i}] <non-dict>")
            i += 1
            continue
        role = msg.get("role")
        content = msg.get("content")
        has_tool_calls = bool(msg.get("tool_calls"))
        has_reasoning = "reasoning_content" in msg
        reasoning_len = (
            len(msg["reasoning_content"])
            if has_reasoning and isinstance(msg["reasoning_content"], str)
            else 0
        )
        # Collapse consecutive assistant fragments (word-level splits from
        # misbehaving clients like Android Studio) into a single summary line.
        if (
            role == "assistant"
            and not has_tool_calls
            and not has_reasoning
            and isinstance(content, str)
            and len(content) < 32
        ):
            # Peek ahead to find the contiguous block
            j = i + 1
            collapsed_parts = [content]
            while j < len(messages):
                nxt = messages[j]
                if (
                    isinstance(nxt, dict)
                    and nxt.get("role") == "assistant"
                    and not nxt.get("tool_calls")
                    and "reasoning_content" not in nxt
                    and isinstance(nxt.get("content"), str)
                    and len(nxt["content"]) < 32
                ):
                    collapsed_parts.append(nxt["content"])
                    j += 1
                else:
                    break
            if len(collapsed_parts) > 2:
                joined = " ".join(collapsed_parts)
                preview = joined[:120].replace("\n", "\\n")
                lines.append(
                    f"  [{i}-{j - 1}] role=assistant tool_calls=False "
                    f"reasoning_content=no content=str({len(joined)}): {preview!r}"
                )
                i = j
                continue
        # ── normal single-message summary ──────────────────────────
        if isinstance(content, str):
            preview = content[:120].replace("\n", "\\n")
            content_desc = f"str({len(content)}): {preview!r}"
        elif content is None:
            content_desc = "None"
        else:
            content_desc = f"{type(content).__name__}"
        lines.append(
            f"  [{i}] role={role} tool_calls={has_tool_calls} "
            f"reasoning_content={'yes(' + str(reasoning_len) + ')' if has_reasoning else 'no'} "
            f"content={content_desc}"
        )
        i += 1
    return "\n".join(lines)


def _log_upstream_error(
    method: str,
    url: str,
    status: int,
    request_headers: Mapping[str, str],
    request_body: bytes,
    response_body: bytes,
) -> None:
    """Dump enough context to debug upstream rejections (4xx/5xx).

    Logged at WARNING so it surfaces without enabling DEBUG. Authorization
    and similar secret-bearing headers are redacted.
    """
    summary = _summarize_messages(request_body)
    logger.warning(
        "Upstream %s %s returned %d\n"
        "  request headers: %s\n"
        "  request messages summary:\n%s\n"
        "  request body: %s\n"
        "  response body: %s",
        method,
        url,
        status,
        # _redact_headers(request_headers),
        _redact_headers(request_headers),
        summary if summary is not None else "  <not a chat-completions JSON body>",
        _safe_decode(request_body),
        _safe_decode(response_body),
    )


# Hop-by-hop headers that must not be forwarded (RFC 7230 §6.1) plus a
# couple that httpx will recompute.
_HOP_BY_HOP = {
    "accept-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "content-encoding",
}


def _filter_headers(headers: Mapping[str, str] | Iterable[tuple[str, str]]) -> dict[str, str]:
    items: Iterable[tuple[str, str]]
    items = headers.items() if isinstance(headers, Mapping) else headers
    return {k: v for k, v in items if k.lower() not in _HOP_BY_HOP}


def _build_upstream_request(
    cfg: Settings,
    path: str,
    body_bytes: bytes,
    request_headers: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    """Resolve the upstream and return (upstream_url, amended_headers).

    When multi-upstream mode is active the upstream is selected via the
    configured routing rules (header → path prefix → model → default).
    Per-upstream API keys are injected when the client didn't supply one.
    Falls back to the legacy ``upstream_base_url`` for zero-config usage.
    """
    headers = _filter_headers(request_headers)
    headers["content-type"] = "application/json"

    if cfg.has_multi_upstream:
        resolved = resolve_upstream(
            cfg, path, body_bytes, request_headers=headers
        )
        if resolved is None:
            raise _NoUpstreamError("no upstream matches the request")

        base = resolved.options.base_url.rstrip("/")
        prefix = resolved.options.path_prefix or ""
        prefix = prefix.rstrip("/")
        rewritten = resolved.path if resolved.path.startswith("/") else f"/{resolved.path}"
        upstream_url = f"{base}{prefix}{rewritten}"

        # Override per-upstream API key
        api_key = resolved.options.api_key
        if api_key:
            key_header = resolved.options.api_key_header.lower()
            if key_header in {k.lower() for k in headers}:
                value = f"Bearer {api_key}" if resolved.options.api_key_as_bearer else api_key
                logger.info("Injecting header: %s (value redacted, length=%d) replacing original",
                             key_header, len(api_key))
                if key_header != resolved.options.api_key_header:
                    del headers[key_header]
                headers[resolved.options.api_key_header] = value

        return upstream_url, headers

    # Legacy single-upstream mode
    base = cfg.upstream_base_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    prefix = cfg.upstream_path_prefix.rstrip("/")
    if prefix and suffix != prefix and not suffix.startswith(prefix + "/"):
        suffix = prefix + suffix
    return f"{base}{suffix}", headers


class _NoUpstreamError(Exception):
    """Raised when no upstream can be resolved for a request."""
    pass


async def _chat_completions_handler(
    cfg: Settings,
    cache: ReasoningCache,
    request: Request,
    upstream_path: str,
) -> Response:
    """Shared handler for /v1/chat/completions and /chat/completions routes."""
    client: httpx.AsyncClient = request.app.state.client
    raw = await request.body()
    try:
        body: Any = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    new_body = transform_request_body(body, cfg, cache=cache)
    is_stream = bool(new_body.get("stream"))

    payload = json.dumps(new_body).encode("utf-8")
    try:
        upstream_url, headers = _build_upstream_request(
            cfg, upstream_path, raw, request.headers
        )
    except _NoUpstreamError:
        return JSONResponse(
            {"error": "no upstream matches the request"}, status_code=400
        )

    if is_stream:
        req = client.build_request("POST", upstream_url, headers=headers, content=payload)
        upstream_resp = await client.send(req, stream=True)
        if upstream_resp.status_code >= 400:
            err_body = await upstream_resp.aread()
            await upstream_resp.aclose()
            _log_upstream_error(
                "POST",
                upstream_url,
                upstream_resp.status_code,
                headers,
                payload,
                err_body,
            )
            return Response(
                content=err_body,
                status_code=upstream_resp.status_code,
                headers=_filter_headers(upstream_resp.headers),
            )

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for out in stitch_sse(upstream_resp.aiter_bytes(), cfg, cache=cache):
                    yield out
            finally:
                await upstream_resp.aclose()

        resp_headers = _filter_headers(upstream_resp.headers)
        return StreamingResponse(
            body_iter(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
        )

    upstream_resp = await client.post(upstream_url, headers=headers, content=payload)
    resp_headers = _filter_headers(upstream_resp.headers)
    if upstream_resp.status_code >= 400:
        _log_upstream_error(
            "POST",
            upstream_url,
            upstream_resp.status_code,
            headers,
            payload,
            upstream_resp.content,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )
    try:
        data = upstream_resp.json()
    except json.JSONDecodeError:
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )
    if isinstance(data, dict):
        data = transform_response_body(data, cfg, cache=cache)
    return JSONResponse(data, status_code=upstream_resp.status_code)


async def _passthrough_handler(
    cfg: Settings,
    request: Request,
    upstream_path: str,
) -> Response:
    """Shared handler for /v1/{path}, /{path}, and prefix-routing catch-all."""
    client: httpx.AsyncClient = request.app.state.client
    body = await request.body()
    try:
        upstream_url, headers = _build_upstream_request(
            cfg, upstream_path, body, request.headers
        )
    except _NoUpstreamError:
        return JSONResponse(
            {"error": "no upstream matches the request"}, status_code=400
        )
    upstream_resp = await client.request(
        request.method,
        upstream_url,
        headers=headers,
        params=dict(request.query_params),
        content=body if body else None,
    )
    if upstream_resp.status_code >= 400:
        _log_upstream_error(
            request.method,
            upstream_url,
            upstream_resp.status_code,
            headers,
            body,
            upstream_resp.content,
        )
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=_filter_headers(upstream_resp.headers),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    logging.basicConfig(level=cfg.log_level.upper())
    cache = ReasoningCache()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(cfg.request_timeout_s, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            app.state.client = client
            app.state.settings = cfg
            app.state.cache = cache
            yield

    app = FastAPI(title="mindthegap", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _chat_completions_handler(
            cfg, cache, request, "/v1/chat/completions"
        )

    @app.post("/chat/completions")
    async def chat_completions_no_prefix(request: Request) -> Response:
        return await _chat_completions_handler(
            cfg, cache, request, "/chat/completions"
        )

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def passthrough(path: str, request: Request) -> Response:
        return await _passthrough_handler(cfg, request, f"/v1/{path}")

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def catch_all(path: str, request: Request) -> Response:
        """Catch-all route: handles multi-upstream path-prefix routing
        (e.g. /deepseek/v1/...) and non-prefixed passthrough (e.g. /models).
        """
        return await _passthrough_handler(cfg, request, f"/{path}")

    return app
