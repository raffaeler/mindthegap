"""In-memory sidecar cache for ``reasoning_content`` keyed by tool_call_id.

Background:
    Some OpenAI-compatible clients (notably the GitHub Copilot CLI) persist
    assistant messages that carry ``tool_calls`` with ``content: null``,
    discarding any text that came alongside the tool calls. That defeats the
    proxy's normal trick of stuffing ``reasoning_content`` into ``content``
    inside ``<think>...</think>`` tags: on the next turn there is no
    ``<think>`` block to recover and DeepSeek rejects the request with
    ``reasoning_content must be passed back``.

    Tool call ids are mandatory in the OpenAI tool-calling protocol and are
    echoed verbatim by the client in the matching ``tool`` role messages on
    follow-up turns. We use them as a stable key to cache reasoning text
    server-side and re-inject it whenever the proxy sees an assistant
    message with tool_calls but no reasoning_content.

The cache is process-local, bounded (LRU eviction) and TTL'd, and uses a
single mutex \u2014 access is sub-microsecond and never blocks I/O.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


class ReasoningCache:
    def __init__(self, max_entries: int = 2048, ttl_seconds: float = 3600.0) -> None:
        self._lock = Lock()
        self._data: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_seconds

    def put(self, tool_call_id: str, reasoning: str) -> None:
        if not tool_call_id or not reasoning:
            return
        with self._lock:
            now = time.monotonic()
            self._data[tool_call_id] = (reasoning, now)
            self._data.move_to_end(tool_call_id)
            self._evict_locked(now)
            size = len(self._data)
        logger.info(
            "reasoning cache PUT id=%s len=%d size=%d",
            tool_call_id,
            len(reasoning),
            size,
        )

    def get(self, tool_call_id: str) -> str | None:
        if not tool_call_id:
            return None
        with self._lock:
            now = time.monotonic()
            entry = self._data.get(tool_call_id)
            if entry is None:
                logger.info("reasoning cache MISS id=%s", tool_call_id)
                return None
            reasoning, ts = entry
            if now - ts > self._ttl:
                del self._data[tool_call_id]
                logger.info("reasoning cache EXPIRED id=%s", tool_call_id)
                return None
            self._data.move_to_end(tool_call_id)
        logger.info("reasoning cache HIT id=%s len=%d", tool_call_id, len(reasoning))
        return reasoning

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def _evict_locked(self, now: float) -> None:
        # Drop expired entries first.
        expired = [k for k, (_, ts) in self._data.items() if now - ts > self._ttl]
        for k in expired:
            del self._data[k]
        # Then enforce the size cap (LRU: oldest first).
        while len(self._data) > self._max:
            self._data.popitem(last=False)
