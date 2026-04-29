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

from .config import Settings, load_settings
from .streaming import stitch_sse
from .transforms import transform_request_body, transform_response_body

logger = logging.getLogger("mindthegap")

# Header values to redact in diagnostic dumps so secrets never reach the log.
_REDACT_HEADERS = {"authorization", "x-api-key", "api-key", "proxy-authorization"}


def _redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: ("<redacted>" if k.lower() in _REDACT_HEADERS else v) for k, v in headers.items()}


def _safe_decode(payload: bytes, limit: int = 8192) -> str:
    text = payload.decode("utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + f"... <truncated, total {len(text)} chars>"
    return text


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
    logger.warning(
        "Upstream %s %s returned %d\n"
        "  request headers: %s\n"
        "  request body: %s\n"
        "  response body: %s",
        method,
        url,
        status,
        _redact_headers(request_headers),
        _safe_decode(request_body),
        _safe_decode(response_body),
    )


# Hop-by-hop headers that must not be forwarded (RFC 7230 §6.1) plus a
# couple that httpx will recompute.
_HOP_BY_HOP = {
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


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    logging.basicConfig(level=cfg.log_level.upper())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(cfg.request_timeout_s, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            app.state.client = client
            app.state.settings = cfg
            yield

    app = FastAPI(title="mindthegap", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.client
        raw = await request.body()
        try:
            body: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

        new_body = transform_request_body(body, cfg)
        is_stream = bool(new_body.get("stream"))

        upstream_url = cfg.upstream("/v1/chat/completions")
        headers = _filter_headers(request.headers)
        headers["content-type"] = "application/json"
        payload = json.dumps(new_body).encode("utf-8")

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
                    async for out in stitch_sse(upstream_resp.aiter_bytes(), cfg):
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
            data = transform_response_body(data, cfg)
        return JSONResponse(data, status_code=upstream_resp.status_code)

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def passthrough(path: str, request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.client
        upstream_url = cfg.upstream(f"/v1/{path}")
        headers = _filter_headers(request.headers)
        body = await request.body()
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

    return app
