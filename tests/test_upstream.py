"""Unit tests for upstream resolution."""

from __future__ import annotations

import json

import pytest

from mindthegap.config import Settings, UpstreamOptions, UpstreamSelectionOptions
from mindthegap.upstream import resolve_upstream, glob_match


def _make_settings(upstreams: dict | None = None, **kwargs) -> Settings:
    """Build a Settings with minimal upstreams for testing."""
    defaults = {
        "upstreams": upstreams or {},
        "upstream_selection": UpstreamSelectionOptions(),
        "default_upstream": None,
    }
    defaults.update(kwargs)
    return Settings.model_validate(defaults)


class TestGlobMatch:
    def test_exact_no_wildcard(self) -> None:
        assert glob_match("*", "anything") is True
        assert glob_match("no-wildcard", "no-wildcard") is False  # no * = no match

    def test_prefix_glob(self) -> None:
        assert glob_match("deepseek-*", "deepseek-chat") is True
        assert glob_match("deepseek-*", "deepseek-reasoner") is True
        assert glob_match("deepseek-*", "x-deepseek-chat") is False  # anchored

    def test_suffix_glob(self) -> None:
        assert glob_match("*-reasoner", "deepseek-reasoner") is True
        assert glob_match("*-reasoner", "openai-reasoner") is True
        assert glob_match("*-reasoner", "reasoner") is False  # need part before *

    def test_mid_glob(self) -> None:
        assert glob_match("gpt-*", "gpt-4o") is True
        assert glob_match("gpt-*mini", "gpt-4o-mini") is True

    def test_disable_pattern(self) -> None:
        # [X] prefix is a llmhub convention to disable a mapping — the
        # pattern intentionally doesn't match because the literal "[X]"
        # must appear at the start of the model name.
        assert glob_match("[X]gpt-*", "gpt-4o") is False
        assert glob_match("[X]gpt-*", "[X]gpt-4o") is True  # literal match


class TestResolveUpstream:
    def test_no_upstreams_returns_none(self) -> None:
        settings = _make_settings()
        assert resolve_upstream(settings, "/v1/chat/completions", b"{}") is None

    def test_default_upstream(self) -> None:
        settings = _make_settings(
            upstreams={"deepseek": UpstreamOptions(base_url="https://api.deepseek.com")},
            default_upstream="deepseek",
        )
        result = resolve_upstream(settings, "/v1/chat/completions", b"{}")
        assert result is not None
        assert result.name == "deepseek"

    def test_header_selection(self) -> None:
        settings = _make_settings(
            upstreams={
                "deepseek": UpstreamOptions(base_url="https://api.deepseek.com"),
                "openai": UpstreamOptions(base_url="https://api.openai.com"),
            },
            upstream_selection=UpstreamSelectionOptions(header_name="X-Mindthegap-Upstream"),
        )
        result = resolve_upstream(
            settings,
            "/v1/chat/completions",
            b"{}",
            request_headers={"x-mindthegap-upstream": "openai"},
        )
        assert result is not None
        assert result.name == "openai"

    def test_header_ignored_when_empty(self) -> None:
        settings = _make_settings(
            upstreams={"deepseek": UpstreamOptions(base_url="https://api.deepseek.com")},
            upstream_selection=UpstreamSelectionOptions(header_name=""),
        )
        # Should fall through to default
        result = resolve_upstream(
            settings, "/v1/chat/completions", b"{}",
            request_headers={"x-mindthegap-upstream": "anything"},
        )
        assert result is None  # no default set

    def test_path_prefix_selection(self) -> None:
        settings = _make_settings(
            upstreams={
                "deepseek": UpstreamOptions(base_url="https://api.deepseek.com"),
            },
            upstream_selection=UpstreamSelectionOptions(path_prefix_enabled=True),
        )
        result = resolve_upstream(settings, "/deepseek/v1/chat/completions", b"{}")
        assert result is not None
        assert result.name == "deepseek"
        assert result.path == "/v1/chat/completions"

    def test_path_prefix_preserves_query(self) -> None:
        settings = _make_settings(
            upstreams={"openai": UpstreamOptions(base_url="https://api.openai.com")},
            upstream_selection=UpstreamSelectionOptions(path_prefix_enabled=True),
        )
        result = resolve_upstream(settings, "/openai/v1/models?limit=10", b"{}")
        assert result is not None
        assert result.name == "openai"
        assert result.path == "/v1/models?limit=10"

    def test_model_glob_selection(self) -> None:
        settings = _make_settings(
            upstreams={
                "deepseek": UpstreamOptions(base_url="https://api.deepseek.com"),
                "openai": UpstreamOptions(base_url="https://api.openai.com"),
            },
            upstream_selection=UpstreamSelectionOptions(
                model_mapping={"deepseek-*": "deepseek", "gpt-*": "openai"},
            ),
        )
        body = json.dumps({"model": "deepseek-chat", "messages": []}).encode()
        result = resolve_upstream(settings, "/v1/chat/completions", body)
        assert result is not None
        assert result.name == "deepseek"

    def test_model_exact_match(self) -> None:
        settings = _make_settings(
            upstreams={
                "deepseek": UpstreamOptions(base_url="https://api.deepseek.com"),
            },
            upstream_selection=UpstreamSelectionOptions(
                model_mapping={"deepseek-chat": "deepseek"},
            ),
        )
        body = json.dumps({"model": "deepseek-chat"}).encode()
        result = resolve_upstream(settings, "/v1/chat/completions", body)
        assert result is not None
        assert result.name == "deepseek"

    def test_model_parsing_invalid_json(self) -> None:
        settings = _make_settings(
            upstreams={"deepseek": UpstreamOptions(base_url="https://api.deepseek.com")},
            upstream_selection=UpstreamSelectionOptions(
                model_mapping={"deepseek-*": "deepseek"},
            ),
        )
        result = resolve_upstream(settings, "/v1/chat/completions", b"not json")
        assert result is None  # falls to default, which is not set

    def test_priority_header_over_path(self) -> None:
        settings = _make_settings(
            upstreams={
                "deepseek": UpstreamOptions(base_url="https://api.deepseek.com"),
                "openai": UpstreamOptions(base_url="https://api.openai.com"),
            },
            upstream_selection=UpstreamSelectionOptions(
                header_name="X-Mindthegap-Upstream",
                path_prefix_enabled=True,
            ),
        )
        # Header says "openai", path says "deepseek" → header wins
        result = resolve_upstream(
            settings,
            "/deepseek/v1/chat/completions",
            b"{}",
            request_headers={"x-mindthegap-upstream": "openai"},
        )
        assert result is not None
        assert result.name == "openai"

    def test_priority_model_over_default(self) -> None:
        settings = _make_settings(
            upstreams={
                "deepseek": UpstreamOptions(base_url="https://api.deepseek.com"),
                "other": UpstreamOptions(base_url="https://other.api"),
            },
            upstream_selection=UpstreamSelectionOptions(
                model_mapping={"gpt-*": "other"},
            ),
            default_upstream="deepseek",
        )
        body = json.dumps({"model": "gpt-4o"}).encode()
        result = resolve_upstream(settings, "/v1/chat/completions", body)
        assert result is not None
        assert result.name == "other"
