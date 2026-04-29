"""Configuration loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

UnstitchMode = Literal["drop", "keep", "forward"]


class TlsConfig(BaseModel):
    cert_dir: str | None = None
    cert_file: str | None = None
    key_file: str | None = None
    san_dns: list[str] | None = None
    san_ip: list[str] | None = None
    validity_days: int = 3650
    renew_within_days: int = 30


class Settings(BaseModel):
    upstream_base_url: str = "https://api.deepseek.com"
    host: str = "127.0.0.1"
    port: int = 3333
    think_tag_open: str = "<think>"
    think_tag_close: str = "</think>"
    reasoner_models: list[str] = Field(default_factory=lambda: ["deepseek-reasoner"])
    unstitch_when_not_reasoner: UnstitchMode = "drop"
    request_timeout_s: float = 600.0
    log_level: str = "INFO"
    tls: TlsConfig = Field(default_factory=TlsConfig)

    def upstream(self, path: str) -> str:
        base = self.upstream_base_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{base}{suffix}"

    def is_reasoner(self, model: str | None) -> bool:
        if not model:
            return False
        return model in self.reasoner_models


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
