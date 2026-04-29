"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import uvicorn

from . import __copyright__, __url__, __version__
from .app import create_app
from .config import load_settings
from .tls import ensure_cert, print_cert_reused, print_trust_instructions


def _print_banner() -> None:
    banner = (
        f"mindthegap {__version__} - "
        f"DeepSeek reasoning_content stitch/unstitch proxy\n"
        f"{__copyright__}. All rights reserved.\n"
        f"{__url__}"
    )
    print(banner, file=sys.stderr, flush=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mindthegap", description=__doc__)
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--host", help="Override bind host")
    parser.add_argument("--port", type=int, help="Override bind port")
    parser.add_argument("--log-level", help="Override log level (DEBUG, INFO, ...)")
    parser.add_argument("--cert-dir", help="Override directory for self-signed cert/key")
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
    if args.cert_dir:
        settings.tls.cert_dir = args.cert_dir

    logging.basicConfig(level=settings.log_level.upper())
    _print_banner()
    cert_path, key_path, generated = ensure_cert(settings)
    if generated:
        print_trust_instructions(cert_path, settings.host, settings.port)
    else:
        print_cert_reused(cert_path, settings.host, settings.port)

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
    )


if __name__ == "__main__":
    main()
