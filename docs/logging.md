# Logging

`mindthegap` runs under uvicorn and writes all logs to stderr. There is no log
file by default.

## Capture logs to a file

Linux / macOS:

```bash
uv run mindthegap --config ./config.json 2> ~/.config/mindthegap/proxy.log
```

Windows (PowerShell):

```powershell
uv run mindthegap --config .\config.json 2> "$env:APPDATA\mindthegap\proxy.log"
```

## Log levels

Set log verbosity in `config.json` via `log_level`, or override it with the
`--log-level` CLI flag.

Supported levels:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`

At `DEBUG`, the proxy logs request and response payloads plus full SSE stream
details, which is useful for stitch/unstitch debugging.