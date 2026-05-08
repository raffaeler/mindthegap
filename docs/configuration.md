# Configuration

`mindthegap` reads `config.json` from the current directory by default. You can
also select a different config file with the `MINDTHEGAP_CONFIG` environment
variable or the `--config` CLI flag.

## Configuration fields

| Field | Default | Notes |
| --- | --- | --- |
| `upstream_base_url` | `https://api.deepseek.com` | Upstream OpenAI-compatible API root |
| `host` | `127.0.0.1` | Bind address |
| `port` | `3333` | Bind port |
| `think_tag_open` / `think_tag_close` | `[[think]]` / `[[/think]]` | Tags wrapping the reasoning block. The default uses double square brackets because some chat clients, including GitHub Copilot CLI, strip HTML-like `<think>` tags and their content during persistence. |
| `reasoner_models` | `["deepseek-reasoner", "deepseek-v4-pro"]` | Models for which `reasoning_content` is forwarded upstream |
| `unstitch_when_not_reasoner` | `drop` | `drop` strips the block, `keep` leaves it inline, `forward` still sends it as `reasoning_content` |
| `request_timeout_s` | `600` | Upstream HTTP timeout in seconds |
| `log_level` | `INFO` | Standard Python log level |
| `tls.cert_dir` | `null` | Directory for auto-generated cert/key. `null` means the OS user-config dir. |
| `tls.cert_file` / `tls.key_file` | `null` | Explicit cert/key paths. When both are set, no auto-generation occurs. |
| `tls.san_dns` / `tls.san_ip` | `null` | Override SAN entries. `null` means auto-detect localhost, hostname, FQDN, `127.0.0.1`, and `::1`. |
| `tls.validity_days` | `3650` | Certificate lifetime when generated |
| `tls.renew_within_days` | `30` | Regenerate when expiry is closer than this |

## CLI overrides

These CLI flags override the config file:

- `--config`
- `--host`
- `--port`
- `--log-level`
- `--cert-dir`