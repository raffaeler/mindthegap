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
  `content` wrapped in `<think>...</think>` tags. The client persists this
  combined string in its local history without knowing anything special.
- **Unstitches** every outgoing request by extracting the leading
  `<think>...</think>` block from each assistant message and moving it back
  into `reasoning_content` before forwarding upstream.

Streaming SSE responses are handled too: a per-choice state machine emits
the opening `<think>` tag, replays reasoning deltas as `content`, and emits
the closing `</think>` when real content begins (or on `finish_reason`).

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (single-binary Python package manager)
- Git

## Install

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

## TLS / self-signed certificate

`mindthegap` always serves HTTPS — DeepSeek and the Copilot CLI both expect
a TLS endpoint. On first launch the proxy generates its **own** self-signed
certificate (RSA-2048, 10-year validity, with a SAN extension covering
`localhost`, your machine hostname, FQDN, `127.0.0.1` and `::1`) and stores
it under:

| OS              | Path                                                     |
| --------------- | -------------------------------------------------------- |
| Linux / macOS   | `$XDG_CONFIG_HOME/mindthegap/` (default `~/.config/mindthegap/`) |
| Windows         | `%APPDATA%\mindthegap\`                                    |

Files: `cert.pem` (public certificate) and `key.pem` (private key, mode
`0600` on POSIX). The cert is **reused** on subsequent launches and only
regenerated automatically when it:

- is missing or unparseable;
- expires within `tls.renew_within_days` (default 30 days);
- no longer covers the required SAN entries (e.g. you renamed the host).

Explicit `tls.cert_file` + `tls.key_file` in `config.json` disable
auto-generation entirely — bring your own cert and the proxy will leave it
alone.

### When does the trust prompt appear?

The proxy is deliberately quiet about TLS once you are set up:

- **First launch (or any time a new cert is generated)** — the full,
  copy-pasteable trust instructions for Linux, macOS and Windows are
  printed to stderr. Run them once.
- **Subsequent launches with the same cert** — only a short one-line note
  is printed. Most developers already have a trusted localhost cert
  installed and do not need to be re-prompted on every restart.

If you ever want to re-print the full instructions, delete `cert.pem` (and
`key.pem`) from the directory above and start the proxy again — it will
regenerate, re-print, and you can re-trust.

### Trusting the certificate

Pick whichever fits your client.

#### Linux — system-wide CA bundle (curl, Python `ssl`, etc.)

```bash
sudo cp ~/.config/mindthegap/cert.pem /usr/local/share/ca-certificates/mindthegap.crt
sudo update-ca-certificates
```

On Fedora/RHEL: copy to `/etc/pki/ca-trust/source/anchors/` then
`sudo update-ca-trust`.

#### Linux — Firefox / Chromium NSS DB (current user)

```bash
mkdir -p "$HOME/.pki/nssdb"
certutil -d "sql:$HOME/.pki/nssdb" -A -t "C,," -n mindthegap \
  -i ~/.config/mindthegap/cert.pem
```

#### macOS — login keychain

```bash
security add-trusted-cert -r trustRoot \
  -k "$HOME/Library/Keychains/login.keychain-db" \
  ~/.config/mindthegap/cert.pem
```

#### Windows — **Trusted Root Certification Authorities**

On Windows the cert **must** land in the Root CA store; the Personal store
is not consulted for server-auth validation.

PowerShell (current user, no admin):

```powershell
Import-Certificate -FilePath "$env:APPDATA\mindthegap\cert.pem" `
  -CertStoreLocation Cert:\CurrentUser\Root
```

PowerShell (machine-wide, **Administrator**):

```powershell
Import-Certificate -FilePath "$env:APPDATA\mindthegap\cert.pem" `
  -CertStoreLocation Cert:\LocalMachine\Root
```

GUI alternative: double-click `cert.pem` → *Install Certificate* → choose
*Current User* (or *Local Machine*) → *Place all certificates in the
following store* → **Trusted Root Certification Authorities**.

#### Per-process env vars (no OS trust changes)

Some clients ignore the OS trust store and read CA bundles from env vars.
This is the lightest-touch option and works well for one-off use.

Linux / macOS:

```bash
export NODE_EXTRA_CA_CERTS="$HOME/.config/mindthegap/cert.pem"
export SSL_CERT_FILE="$HOME/.config/mindthegap/cert.pem"
export REQUESTS_CA_BUNDLE="$HOME/.config/mindthegap/cert.pem"
```

Windows (PowerShell):

```powershell
$env:NODE_EXTRA_CA_CERTS = "$env:APPDATA\mindthegap\cert.pem"
$env:SSL_CERT_FILE       = "$env:APPDATA\mindthegap\cert.pem"
$env:REQUESTS_CA_BUNDLE  = "$env:APPDATA\mindthegap\cert.pem"
```

## Use with GitHub Copilot CLI

```bash
export NODE_EXTRA_CA_CERTS="$HOME/.config/mindthegap/cert.pem"     # trust the proxy
export COPILOT_PROVIDER_BASE_URL="https://127.0.0.1:3333/v1"
export COPILOT_PROVIDER_API_KEY="sk-...your-deepseek-key..."
export COPILOT_PROVIDER_TYPE="openai"
export COPILOT_MODEL="deepseek-reasoner"
copilot
```

Windows (PowerShell): swap each `export NAME=value` for `$env:NAME = "value"`
and use `$env:APPDATA\mindthegap\cert.pem` for the CA path.

The proxy passes the `Authorization` header straight through to DeepSeek.

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

## Logs

`mindthegap` runs under uvicorn and writes **all logs to stderr** — there is
no log file by default. Capture them with shell redirection:

```bash
# Linux / macOS
uv run mindthegap --config ./config.json 2> ~/.config/mindthegap/proxy.log

# Windows (PowerShell)
uv run mindthegap --config .\config.json 2> "$env:APPDATA\mindthegap\proxy.log"
```

Log verbosity is controlled by `log_level` in `config.json` or the
`--log-level` CLI flag (`DEBUG`, `INFO`, `WARNING`, `ERROR`). At `DEBUG`
you'll see request/response payloads and full SSE stream details, which is
useful when investigating stitch/unstitch issues.

## Configuration (`config.json`)

| Field | Default | Notes |
| --- | --- | --- |
| `upstream_base_url` | `https://api.deepseek.com` | Upstream OpenAI-compatible API root |
| `host` | `127.0.0.1` | Bind address |
| `port` | `3333` | Bind port |
| `think_tag_open` / `think_tag_close` | `<think>` / `</think>` | Tags wrapping the reasoning block |
| `reasoner_models` | `["deepseek-reasoner", "deepseek-v4-pro"]` | Models for which `reasoning_content` is forwarded upstream |
| `unstitch_when_not_reasoner` | `"drop"` | `drop` strips the block, `keep` leaves it inline, `forward` still sends it as `reasoning_content` |
| `request_timeout_s` | `600` | Upstream HTTP timeout (seconds) |
| `log_level` | `INFO` | Standard Python log level |
| `tls.cert_dir` | `null` | Directory for auto-generated cert/key. `null` → OS user-config dir. |
| `tls.cert_file` / `tls.key_file` | `null` | Explicit cert/key paths. When both set, no auto-generation. |
| `tls.san_dns` / `tls.san_ip` | `null` | Override the SAN entries. `null` → auto (`localhost` + hostname + FQDN; `127.0.0.1` + `::1`). |
| `tls.validity_days` | `3650` | Cert lifetime when generated. |
| `tls.renew_within_days` | `30` | Regenerate when expiry is closer than this. |

CLI flags `--config`, `--host`, `--port`, `--log-level`, `--cert-dir`
override the file. The config path can also be set via `MINDTHEGAP_CONFIG`.

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## Troubleshooting

- **`SELF_SIGNED_CERT_IN_CHAIN` / `unable to verify the first certificate`**
  — the cert is not trusted by the client. Re-run the trust step for your
  OS, or set `NODE_EXTRA_CA_CERTS` / `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`
  to point at `cert.pem`.
- **HTTP 400 `reasoning_content must be passed back`** — the client is
  bypassing the proxy. Verify the base URL really is
  `https://127.0.0.1:3333/v1`.
- **Cert keeps getting regenerated** — the SAN list changed (hostname
  rename) or `validity_days` is smaller than `renew_within_days`. Adjust
  the config.
- **Port 3333 already in use** — another instance is running, or pick a
  different port with `--port`.

## License

MIT.
