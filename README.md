# mindthegap

A localhost stitch/unstitch HTTPS proxy that lets minimal OpenAI-compatible
clients (such as the GitHub Copilot CLI) talk to **DeepSeek reasoning models**
without losing the `reasoning_content` field across turns.

## Why

`deepseek-reasoner` (and DeepSeek V4 Pro) returns a `reasoning_content` field
alongside `content`. Most OpenAI-compatible clients drop it. When the client
sends the conversation back on the next turn, DeepSeek rejects the request
with HTTP 400 (`reasoning_content must be passed back`) and multi-turn chats
break.
The exact error is printed by the GitHub Copilot CLI:
```bash
✗ 400 The `reasoning_content` in the thinking mode must be passed back to the API. 
```

`mindthegap` sits between the client and DeepSeek and:

- **Stitches** every assistant response by folding `reasoning_content` into
  `content` wrapped in `[[think]]...[[/think]]` tags. The client persists this
  combined string in its local history without knowing anything special.
- **Unstitches** every outgoing request by extracting the leading
  `[[think]]...[[/think]]` block from each assistant message and moving it back
  into `reasoning_content` before forwarding upstream.

Streaming SSE responses are handled too: a per-choice state machine emits
the opening `[[think]]` tag, replays reasoning deltas as `content`, and emits
the closing `[[/think]]` when real content begins (or on `finish_reason`).

## Install

Requirements:

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (single-binary Python package manager)
- Git

### Linux / macOS

```bash
# 1. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone
git clone https://github.com/raffaeler/mindthegap.git
cd mindthegap

# 3. Create the virtualenv and install dependencies
uv sync

# 4. Copy the example config (edit if needed)
cp config.example.json config.json
```

### Windows (PowerShell)

```powershell
# 1. Install uv (skip if already installed)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone
git clone https://github.com/raffaeler/mindthegap.git
Set-Location mindthegap

# 3. Create the virtualenv and install dependencies
uv sync

# 4. Copy the example config (edit if needed)
Copy-Item config.example.json config.json
```

## Run

```bash
uv run mindthegap --config ./config.json
# equivalent: uv run python -m mindthegap
```

The proxy binds `127.0.0.1:3333` over **HTTPS** by default. Override at the
command line:

```
uv run mindthegap --host 127.0.0.1 --port 3333 --log-level INFO --cert-dir ./certs
```

You can also point at an alternate config via the `MINDTHEGAP_CONFIG`
environment variable.

## Endpoints

- `POST /v1/chat/completions` — full stitch/unstitch (JSON and SSE).
- `* /v1/{path}` — transparent passthrough for everything else
  (e.g. `GET /v1/models`).
- `GET /healthz` — liveness probe, returns `{"ok": true}`.

Quick health check:

```bash
curl --cacert ~/.config/mindthegap/cert.pem https://127.0.0.1:3333/healthz
```

Windows (PowerShell):

```powershell
curl.exe --cacert "$env:APPDATA\mindthegap\cert.pem" https://127.0.0.1:3333/healthz
```

## HTTPS/TLS certificate

A local certificate is necessary for HTTPS channel between the client (e.g. GitHub Copilot CLI) and the proxy. This project can:
- auto-generate a self-signed cert/key pair on first run, or use user-provided ones
- optionally include custom SAN entries (e.g. `localhost`, `127.0.0.x`, machine hostname) to avoid client validation issues
- automatically renew the cert when it's close to expiry

Depending on your security preferences, platform capabilities and the client software, you can choose one of several trust setup methods to make your clients trust the proxy's certificate:

- [TLS overview](docs/tls.md)
- [Certificate setup: global OS trust store](docs/certificates-global-trust.md)
- [Certificate setup: combined CA bundle](docs/certificates-combined-ca.md)
- [Certificate setup: per-client trust override](docs/certificates-per-client.md)
- [Certificate setup: browser / NSS-only trust](docs/certificates-browser-nss.md)

## Additional documentation

- [Troubleshooting](docs/troubleshooting.md)
- [Logging](docs/logging.md)
- [Configuration](docs/configuration.md)
- [Development](docs/development.md)

## License

MIT.
