# oaipatch

A localhost stitch/unstitch proxy that lets minimal OpenAI-compatible clients
(such as the GitHub Copilot CLI) talk to **DeepSeek reasoning models** without
losing the `reasoning_content` field across turns.

## Why

`deepseek-reasoner` (and DeepSeek V4 Pro) returns a `reasoning_content` field
alongside `content`. Most OpenAI-compatible clients drop it. When the client
sends the conversation back on the next turn, DeepSeek rejects the request
with HTTP 400 ("reasoning_content must be passed back") and multi-turn chats
break.

`oaipatch` sits between the client and DeepSeek and:

- **Stitches** every assistant response by folding `reasoning_content` into
  `content` wrapped in `<think>...</think>` tags. The client persists this
  combined string in its local history without knowing anything special.
- **Unstitches** every outgoing request by extracting the leading
  `<think>...</think>` block from each assistant message and moving it back
  into `reasoning_content` before forwarding upstream.

Streaming SSE responses are handled too: a per-choice state machine emits
the opening `<think>` tag, replays reasoning deltas as `content`, and emits
the closing `</think>` when real content begins (or on `finish_reason`).

## Install

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
cp config.example.json config.json   # edit if needed
uv run oaipatch --config ./config.json
# or: uv run python -m oaipatch
```

The proxy binds `127.0.0.1:3333` by default. Health check: `GET /healthz`.

## Use with GitHub Copilot CLI

```bash
export COPILOT_PROVIDER_BASE_URL="http://127.0.0.1:3333/v1"
export COPILOT_PROVIDER_API_KEY="sk-...your-deepseek-key..."
export COPILOT_PROVIDER_TYPE="openai"
export COPILOT_MODEL="deepseek-reasoner"
copilot
```

The proxy passes the `Authorization` header straight through to DeepSeek.

## Configuration (`config.json`)

| Field | Default | Notes |
| --- | --- | --- |
| `upstream_base_url` | `https://api.deepseek.com` | Upstream OpenAI-compatible API root |
| `host` | `127.0.0.1` | Bind address |
| `port` | `3333` | Bind port |
| `think_tag_open` / `think_tag_close` | `<think>` / `</think>` | Tags wrapping the reasoning block |
| `reasoner_models` | `["deepseek-reasoner"]` | Models for which `reasoning_content` is forwarded upstream |
| `unstitch_when_not_reasoner` | `"drop"` | `drop` strips the block, `keep` leaves it, `forward` still sends it as `reasoning_content` |
| `request_timeout_s` | `600` | Upstream HTTP timeout (seconds) |
| `log_level` | `INFO` | Standard Python log level |

CLI flags `--config`, `--host`, `--port`, `--log-level` override the file.
The config path can also be set via `OAIPATCH_CONFIG`.

## Endpoints

- `POST /v1/chat/completions` — full stitch/unstitch (JSON and SSE).
- `* /v1/{path}` — transparent passthrough for everything else
  (e.g. `GET /v1/models`).
- `GET /healthz` — `{"ok": true}`.

## Development

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## License

MIT (or whatever you prefer — adjust as needed).
