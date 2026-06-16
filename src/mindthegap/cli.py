"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from . import __copyright__, __url__, __version__
from .app import create_app
from .config import load_settings
from .tls import ensure_cert, print_cert_reused, print_trust_instructions

DESCRIPTION = """\
mindthegap - DeepSeek reasoning_content stitch/unstitch proxy.

Sits between an OpenAI-compatible client and a DeepSeek-compatible upstream,
preserving the reasoning_content field across multi-turn conversations by
stitching it into content tags ([[think]]...[[/think]]) and unstitching it
on the way back."""

EPILOG = f"""\
Examples:
  uv run mindthegap                                    # run with defaults (HTTPS, 127.0.0.1:3333)
  uv run mindthegap --config ./config.json              # use a specific config file
  uv run mindthegap --no-tls                            # plain HTTP on port 3300
  uv run mindthegap --no-tls --http-port 8080           # plain HTTP on port 8080
  uv run mindthegap --upstream deepseek                 # select upstream by name

mindthegap {__version__}  {__copyright__}  {__url__}"""


def _print_banner(use_tls: bool) -> None:
    protocol = "HTTPS" if use_tls else "HTTP"
    banner = (
        f"mindthegap {__version__} - "
        f"DeepSeek reasoning_content stitch/unstitch proxy ({protocol})\n"
        f"{__copyright__}. All rights reserved.\n"
        f"{__url__}"
    )
    print(banner, file=sys.stderr, flush=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mindthegap",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--host", help="Override bind host")
    parser.add_argument("--https-port", type=int, help="Override HTTPS bind port (default 3333)")
    parser.add_argument("--http-port", type=int, help="Override HTTP bind port (default 3300)")
    parser.add_argument("--log-level", help="Override log level (DEBUG, INFO, ...)")
    parser.add_argument("--cert-dir", help="Override directory for self-signed cert/key")
    parser.add_argument(
        "--upstream",
        help="Default upstream name (e.g. --upstream=deepseek or --upstream:deepseek)",
    )
    parser.add_argument(
        "--no-tls",
        action="store_true",
        default=None,
        help="Disable TLS/HTTPS and serve plain HTTP",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = load_settings(args.config)

    # ── If no config file was found and the example config exists, copy it ─
    _auto_copy_example_config(args.config)

    if args.host:
        settings.host = args.host
    if args.https_port:
        settings.https_port = args.https_port
    if args.http_port:
        settings.http_port = args.http_port
    if args.log_level:
        settings.log_level = args.log_level
    if args.cert_dir:
        settings.tls.cert_dir = args.cert_dir

    # ── TLS: CLI --no-tls overrides config tls.enabled ──────────────────
    if args.no_tls is not None:
        settings.tls.enabled = not args.no_tls
    use_tls = settings.tls.enabled
    bind_port = settings.https_port if use_tls else settings.http_port

    # ── Default upstream selection (ported from llmhub) ─────────────────
    if settings.has_multi_upstream:
        settings.default_upstream = _resolve_default_upstream(
            cli_upstream=args.upstream,
            upstream_names=list(settings.upstreams.keys()),
            configured_default=settings.default_upstream,
        )
        if settings.default_upstream is None:
            print(
                "error: no default upstream configured. Use --upstream=<name> "
                "or set 'default_upstream' in config.json.",
                file=sys.stderr,
            )
            sys.exit(1)

    logging.basicConfig(level=settings.log_level.upper())
    _print_banner(use_tls)

    # ── TLS setup ──────────────────────────────────────────────────────
    uvicorn_kwargs: dict = {}
    protocol = "http" if not use_tls else "https"
    if use_tls:
        cert_path, key_path, generated = ensure_cert(settings)
        if generated:
            print_trust_instructions(cert_path, settings.host, bind_port)
        else:
            print_cert_reused(cert_path, settings.host, bind_port)
        uvicorn_kwargs["ssl_certfile"] = str(cert_path)
        uvicorn_kwargs["ssl_keyfile"] = str(key_path)

    print(
        f"mindthegap: listening on {protocol}://{settings.host}:{bind_port}",
        file=sys.stderr,
        flush=True,
    )

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=bind_port,
        log_level=settings.log_level.lower(),
        **uvicorn_kwargs,
    )


def _resolve_default_upstream(
    cli_upstream: str | None,
    upstream_names: list[str],
    configured_default: str | None,
) -> str | None:
    """Resolve the default upstream at startup.

    Ported from llmhub's ``DefaultUpstreamResolver``.  Resolution order:

    1. CLI ``--upstream=<name>`` or ``--upstream:<name>``
    2. Config ``default_upstream``
    3. If only one upstream exists → auto-select
    4. If console is interactive → numbered menu
    5. Otherwise → ``None`` (caller should error)
    """
    names = list(dict.fromkeys(upstream_names))  # dedup, preserve order

    # 1. CLI argument
    if cli_upstream:
        match = next((n for n in names if n.lower() == cli_upstream.lower()), None)
        if match is None:
            print(
                f"Unknown upstream '{cli_upstream}'. Available: {', '.join(names)}",
                file=sys.stderr,
            )
            return None
        return match

    # 2. Config file
    if configured_default:
        return configured_default

    # 3. Single upstream — no choice needed
    if len(names) == 1:
        return names[0]

    if not names:
        print("No upstreams are configured.", file=sys.stderr)
        return None

    # 4. Interactive menu (only when console is a real TTY)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            f"No --upstream specified and console is not interactive. "
            f"Available: {', '.join(names)}",
            file=sys.stderr,
        )
        return None

    print("Select default upstream:", file=sys.stderr)
    for i, name in enumerate(names):
        print(f"  {i + 1}) {name}", file=sys.stderr)

    while True:
        try:
            choice = input("Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return None

        try:
            n = int(choice)
            if 1 <= n <= len(names):
                return names[n - 1]
        except ValueError:
            match = next((n for n in names if n.lower() == choice.lower()), None)
            if match:
                return match

        print("Invalid choice.", file=sys.stderr)


def _auto_copy_example_config(cli_config: str | None) -> None:
    """If no config file is found and the user didn't specify one, copy the
    example config so the project is ready to run out of the box."""
    if cli_config is not None:
        return  # user specified --config, don't interfere
    import os

    if os.environ.get("MINDTHEGAP_CONFIG"):
        return  # env var points elsewhere

    cwd = Path.cwd()
    config_path = cwd / "config.json"
    example_path = cwd / "config.example.json"

    if config_path.exists() or not example_path.exists():
        return

    try:
        shutil.copy2(example_path, config_path)
        print(
            f"mindthegap: created {config_path} from {example_path}. "
            f"Edit it to configure upstreams and TLS settings.",
            file=sys.stderr,
            flush=True,
        )
    except OSError as exc:
        print(
            f"mindthegap: could not create config.json from example ({exc}). "
            f"Running with defaults.",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
