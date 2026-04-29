"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import uvicorn

from .app import create_app
from .config import load_settings


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="oaipatch", description=__doc__)
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--host", help="Override bind host")
    parser.add_argument("--port", type=int, help="Override bind port")
    parser.add_argument("--log-level", help="Override log level (DEBUG, INFO, ...)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = load_settings(args.config)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.log_level:
        settings.log_level = args.log_level

    logging.basicConfig(level=settings.log_level.upper())
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
