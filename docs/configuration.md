# Configuration

`mindthegap` reads `config.json` from the current directory by default. If no
config file exists and `config.example.json` is present, mindthegap automatically
copies it on first run. You can also select a different config file with the
`MINDTHEGAP_CONFIG` environment variable or the `--config` CLI flag.

## Quick start — single upstream (backward compatible)

```json
{
  "upstream_base_url": "https://api.deepseek.com",
  "upstream_path_prefix": "",
  "httpsPort": 3333,
  "httpPort": 3300
}
```

All requests are forwarded to the same base URL. This is the simplest setup and
what was supported since v0.1.

> Set `upstream_path_prefix` to `""` (empty) when your upstream doesn't use a
> `/v1` prefix in its API paths (e.g. DeepSeek).  Leave it as `"/v1"` (the
> default) for OpenAI-style APIs.

## Multi-upstream mode

When you define the `upstreams` object, `upstream_base_url` is ignored and the
proxy selects an upstream per request based on routing rules.

```
upstreams:
  deepseek:
    base_url: https://api.deepseek.com
    path_prefix: /v1
    api_key: sk-...              # injected when client doesn't provide auth
    api_key_header: Authorization
    api_key_as_bearer: true

upstream_selection:
  header_name: X-Mindthegap-Upstream
  path_prefix_enabled: true
  model_mapping:
    deepseek-*: deepseek         # glob (*) wildcard — matches deepseek-chat, etc.
    "[X]gpt-*": openai           # [X] prefix disables the mapping (comment)
    gpt-*: openai
  model_mapping_max_body_bytes: 262144

default_upstream: deepseek       # CLI: --upstream=deepseek or --upstream:deepseek
```

### Upstream resolution order

1. **Request header** — `X-Mindthegap-Upstream: openai` (header name configurable)
2. **Path prefix** — `POST /openai/v1/chat/completions` → strips `/openai`, forwards to openai upstream
3. **Model field** — `"model": "gpt-4o"` matched against `model_mapping` globs
4. **Default** — configured `default_upstream`, or chosen interactively at startup

When no upstream matches, the proxy returns 400.

### Default upstream selection at startup

1. CLI `--upstream=<name>` or `--upstream:<name>`
2. Config `default_upstream`
3. If only one upstream is configured → auto-select
4. If console is interactive → numbered menu
5. Otherwise → error (startup aborts)

For example, with two upstreams and **no** `default_upstream` set, the proxy
prompts you to pick one at startup:

```json
{
  "upstreams": {
    "deepseek": {
      "base_url": "https://api.deepseek.com",
      "path_prefix": "",
      "api_key": "sk-deepseek-..."
    },
    "kimi": {
      "base_url": "https://api.moonshot.cn",
      "path_prefix": "/v1",
      "api_key": "sk-kimi-..."
    }
  },
  "upstream_selection": {
    "model_mapping": {
      "deepseek-v4-pro": "deepseek",
      "deepseek-v4-flash": "deepseek",
      "kimi-2.7-code": "kimi"
    }
  },
  "reasoner_models": [
    "deepseek-v4-pro", 
    "kimi-2.7-code"
  ]
}
```

You can also use `*` as a glob wildcard to match multiple models at once:

```json
{
  "upstream_selection": {
    "model_mapping": {
      "kimi-*": "kimi",
      "deepseek-v4-*": "deepseek"
    }
  }
}
```

When you start the proxy, a numbered menu appears:

```
$ uv run mindthegap
No default upstream configured.
Available upstreams:
  1) kimi
  2) deepseek
Choose default upstream (1-2):
```

After you select one, the proxy stores it for the session and continues
starting up. You can change the selection on subsequent runs by passing
`--upstream=<name>` on the command line.

## Configuration fields

| Field | Default | Notes |
| --- | --- | --- |
| `upstreams` | `{}` | Named upstream definitions (see **UpstreamOptions** below). When non-empty, `upstream_base_url` is ignored. |
| `upstream_selection` | `{}` | Routing rules (see **UpstreamSelectionOptions** below). |
| `default_upstream` | `null` | Default upstream name. Can also be set via `--upstream`. |
| `upstream_base_url` | `https://api.deepseek.com` | Legacy single-upstream base URL. Ignored when `upstreams` is non-empty. |
| `upstream_path_prefix` | `"/v1"` | Prepended to unprefixed paths in legacy single-upstream mode. Set to `""` for upstreams that don't use `/v1` (e.g. DeepSeek). |
| `host` | `127.0.0.1` | Bind address |
| `httpsPort` | `3333` | Bind port when TLS is enabled |
| `httpPort` | `3300` | Bind port when TLS is disabled (or `--no-tls` is used) |
| `think_tag_open` / `think_tag_close` | `[[think]]` / `[[/think]]` | Tags wrapping the reasoning block. The default uses double square brackets because some chat clients, including GitHub Copilot CLI, strip HTML-like `<think>` tags and their content during persistence. DSML tags (`<‖DSML‖…>`) and XML tags (`</parameter>`, `<analysis>`) are automatically stripped from reasoning text regardless of tag setting. |
| `reasoner_models` | `["deepseek-reasoner", "deepseek-v4-pro"]` | Models for which `reasoning_content` is forwarded upstream |
| `unstitch_when_not_reasoner` | `drop` | `drop` strips the block, `keep` leaves it inline, `forward` still sends it as `reasoning_content` |
| `request_timeout_s` | `600` | Upstream HTTP timeout in seconds |
| `log_level` | `INFO` | Standard Python log level |
| `tls.enabled` | `true` | Set to `false` to serve plain HTTP without TLS. Also controllable via `--no-tls`. |
| `tls.cert_dir` | `null` | Directory for auto-generated cert/key. `null` means the OS user-config dir. |
| `tls.cert_file` / `tls.key_file` | `null` | Explicit cert/key paths. When both are set, no auto-generation occurs. Recommended when the cert is trusted globally and must stay stable across restarts. |
| `tls.san_dns` / `tls.san_ip` | `null` | Override SAN entries. `null` means stable localhost defaults: `localhost`, `127.0.0.1`, and `::1`. |
| `tls.validity_days` | `3650` | Certificate lifetime when generated |
| `tls.renew_within_days` | `30` | Regenerate when expiry is closer than this |

### UpstreamOptions

| Field | Default | Notes |
| --- | --- | --- |
| `base_url` | `""` | Upstream API root (e.g. `https://api.openai.com`) |
| `path_prefix` | `null` | Prepended to forwarded paths when absent (e.g. `/v1`). `null` means no prefix. |
| `api_key` | `null` | Injected when the client doesn't provide an Authorization header |
| `api_key_header` | `Authorization` | Header name for the injected key |
| `api_key_as_bearer` | `true` | If true, value is `Bearer {api_key}` rather than raw |

### UpstreamSelectionOptions

| Field | Default | Notes |
| --- | --- | --- |
| `header_name` | `X-Mindthegap-Upstream` | Header used by clients to pick an upstream |
| `path_prefix_enabled` | `true` | Allow leading path segment to select an upstream |
| `model_mapping` | `{}` | Glob/exact map from request `model` field to upstream name |
| `model_mapping_max_body_bytes` | `262144` | Max body size scanned for the `model` field |

## TLS configuration examples

### Disable TLS (plain HTTP)

Useful for development or when running behind a reverse proxy that handles TLS:

```json
{
  "tls": {
    "enabled": false
  }
}
```

Or pass `--no-tls` on the command line for the same effect without editing config.

### Stable pinned certificate

If you want to trust the proxy cert globally and avoid later regeneration
surprises, pin the cert/key paths explicitly:

```json
{
  "tls": {
    "cert_file": "/home/you/.config/mindthegap/cert.pem",
    "key_file": "/home/you/.config/mindthegap/key.pem"
  }
}
```

That disables auto-generation entirely for those files.

## CLI overrides

These CLI flags override the config file:

- `--config`
- `--host`
- `--https-port` — override HTTPS bind port
- `--http-port` — override HTTP bind port
- `--log-level`
- `--cert-dir`
- `--upstream` (or `--upstream:<name>` / `--upstream=<name>`) — default upstream when multi-upstream mode is active
- `--no-tls` — disable TLS/HTTPS and serve plain HTTP (overrides `tls.enabled`)

Run `uv run mindthegap --help` to see all available options with examples.