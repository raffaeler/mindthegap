# mindthegap

> [!IMPORTANT]
> Configure your LLM client to point at **`https://localhost:3333`** (without `/v1`).
> For DeepSeek, also set `"upstream_path_prefix": ""` in `config.json` — see the
> [configuration guide](docs/configuration.md#quick-start--single-upstream-backward-compatible).

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
uv run mindthegap
# equivalent: uv run python -m mindthegap
```

If no `config.json` exists, mindthegap automatically copies `config.example.json`
so you can start right away. Edit the generated `config.json` to configure
upstreams and TLS settings.

The proxy binds `127.0.0.1` over **HTTPS** on port 3333 by default, or plain **HTTP** on port 3300
when TLS is disabled. Override at the command line:

```
uv run mindthegap --host 127.0.0.1 --https-port 3333 --log-level INFO --cert-dir ./certs
```

To serve plain HTTP (e.g. for development behind a reverse proxy):

```
uv run mindthegap --no-tls
uv run mindthegap --no-tls --http-port 8080
```

See all options with `uv run mindthegap --help`. You can also point at an
alternate config via the `MINDTHEGAP_CONFIG` environment variable.

## Endpoints

- `POST /v1/chat/completions` — full stitch/unstitch (JSON and SSE).
- `POST /chat/completions` — same as above, without the `/v1` prefix.
- `* /v1/{path}` — transparent passthrough for everything else
  (e.g. `GET /v1/models`).
- `* /{path}` — non-prefixed passthrough (e.g. `GET /models` forwards
  as `/v1/models` when `upstream_path_prefix` is `"/v1"`).
- `GET /healthz` — liveness probe, returns `{"ok": true}`.

The proxy accepts both `/v1`-prefixed and unprefixed paths so clients
can use `https://127.0.0.1:3333` as their base URL (without `/v1`).
When forwarding to the upstream, a configurable `upstream_path_prefix`
(default `"/v1"`) is prepended to unprefixed paths. Set it to `""` for
upstreams that don't use a `/v1` prefix (e.g. DeepSeek).

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
- auto-generate a self-signed **leaf/server** cert/key pair on first run, or use user-provided ones
- defaults to stable localhost-only SANs (`localhost`, `127.0.0.1`, `::1`) and lets you opt into additional SAN entries (for example a machine hostname) when needed
- automatically renew the cert when it's close to expiry
- automatically replace older incompatible auto-generated certs (for example CA-style certs rejected by newer Copilot CLI builds)

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

## Quick Start

Use this setup if you want `mindthegap` and a local client such as Copilot CLI
to work over HTTPS without certificate surprises.

1. Keep the proxy on `127.0.0.1` / `localhost`. Do not use your machine
   hostname unless you explicitly configure extra SANs.
2. Start the proxy once so it creates `~/.config/mindthegap/cert.pem`:

   ```bash
   uv run mindthegap --config ./config.json
   ```

3. Trust that exact cert globally:

   ```bash
   sudo install -m 0644 ~/.config/mindthegap/cert.pem /usr/local/share/ca-certificates/mindthegap.crt
   sudo update-ca-certificates
   ```

4. Run the proxy in a clean shell:

   ```bash
   unset SSL_CERT_FILE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS
   uv run mindthegap --config ./config.json
   ```

5. Point the client at `https://127.0.0.1:3333/v1`. For Copilot CLI:

   ```bash
   export COPILOT_PROVIDER_BASE_URL="https://127.0.0.1:3333/v1"
   export COPILOT_PROVIDER_API_KEY="sk-...your-deepseek-key..."
   export COPILOT_PROVIDER_TYPE="openai"
   export COPILOT_PROVIDER_WIRE_API="completions"
   export COPILOT_MODEL="deepseek-v4-pro"
   copilot
   ```

Rules that avoid most TLS pain:

- Never set `SSL_CERT_FILE` or `REQUESTS_CA_BUNDLE` to `cert.pem` alone in the
  proxy shell.
- Always connect to `127.0.0.1` or `localhost` unless you configured extra
  SANs.
- If `cert.pem` is ever regenerated, reinstall that exact file into the OS
  trust store.

If you want the certificate pair to be fully pinned and never auto-managed, set
both `tls.cert_file` and `tls.key_file` in `config.json`:

```json
{
  "tls": {
    "cert_file": "/home/you/.config/mindthegap/cert.pem",
    "key_file": "/home/you/.config/mindthegap/key.pem"
  }
}
```

When both are set, `mindthegap` uses those files as-is and does not
auto-generate them.
