"""Configuration loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

UnstitchMode = Literal["drop", "keep", "forward"]


class TlsConfig(BaseModel):
    enabled: bool = True
    cert_dir: str | None = None
    cert_file: str | None = None
    key_file: str | None = None
    san_dns: list[str] | None = None
    san_ip: list[str] | None = None
    validity_days: int = 3650
    renew_within_days: int = 30


class UpstreamOptions(BaseModel):
    """Configuration for a single named upstream server.

    Mirrors llmhub's ``UpstreamOptions``.
    """

    base_url: str = ""
    path_prefix: str | None = None
    api_key: str | None = None
    api_key_header: str = "Authorization"
    api_key_as_bearer: bool = True


class UpstreamSelectionOptions(BaseModel):
    """Rules controlling how the upstream for each request is chosen.

    Mirrors llmhub's ``UpstreamSelectionOptions``.
    """

    header_name: str = "X-Mindthegap-Upstream"
    path_prefix_enabled: bool = True
    model_mapping: dict[str, str] = Field(default_factory=dict)
    model_mapping_max_body_bytes: int = 256 * 1024


class Settings(BaseModel):
    # ── Multi-upstream support ──────────────────────────────────────────
    upstreams: dict[str, UpstreamOptions] = Field(default_factory=dict)
    upstream_selection: UpstreamSelectionOptions = Field(
        default_factory=UpstreamSelectionOptions
    )
    default_upstream: str | None = None

    # ── Legacy single-upstream (used when upstreams is empty) ──────────
    upstream_base_url: str = "https://api.deepseek.com"
    upstream_path_prefix: str = "/v1"

    # ── Server bind ────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    https_port: int = 3333
    http_port: int = 3300

    # ── Think tag configuration ─────────────────────────────────────────
    think_tag_open: str = "[[think]]"
    think_tag_close: str = "[[/think]]"

    # ── Model classification ───────────────────────────────────────────
    reasoner_models: list[str] = Field(
        default_factory=lambda: ["deepseek-reasoner", "deepseek-v4-pro", "kimi-2.7-code"]
    )
    unstitch_when_not_reasoner: UnstitchMode = "drop"

    # ── Model-specific parameter overrides ────────────────────────────
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # ── HTTP ───────────────────────────────────────────────────────────
    request_timeout_s: float = 600.0

    # ── Logging ────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── TLS ────────────────────────────────────────────────────────────
    tls: TlsConfig = Field(default_factory=TlsConfig)

    # ── Helpers ────────────────────────────────────────────────────────

    def upstream(self, path: str) -> str:
        """Build the full upstream URL for the legacy single-upstream mode."""
        base = self.upstream_base_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{base}{suffix}"

    def is_reasoner(self, model: str | None) -> bool:
        if not model:
            return False
        return model in self.reasoner_models

    def get_model_params(self, model: str | None) -> dict[str, Any]:
        """Return parameter overrides for *model* via exact then glob matching.

        Returns an empty dict when *model* is ``None`` or no pattern matches.
        """
        if not model or not self.model_params:
            return {}

        # Exact match
        if model in self.model_params:
            return self.model_params[model]

        # Glob matching (uses the same logic as upstream model_mapping)
        from .upstream import glob_match

        for pattern, params in self.model_params.items():
            if glob_match(pattern, model):
                return params

        return {}

    @property
    def has_multi_upstream(self) -> bool:
        """Return True when the new multi-upstream config is in use."""
        return bool(self.upstreams)


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings from a JSON file. Falls back to defaults when missing."""
    candidate: Path | None
    if path is not None:
        candidate = Path(path)
    elif env := os.environ.get("MINDTHEGAP_CONFIG"):
        candidate = Path(env)
    else:
        default = Path.cwd() / "config.json"
        candidate = default if default.exists() else None

    if candidate is None:
        return Settings()

    if not candidate.exists():
        raise FileNotFoundError(f"Config file not found: {candidate}")

    with candidate.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a JSON object: {candidate}")
    return Settings.model_validate(data)
