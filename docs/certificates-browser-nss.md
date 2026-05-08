# Certificate use case: browser / NSS-only trust

Use this when the client is a browser or another NSS-based tool and you only
need current-user browser trust.

## Certificate installation

```bash
mkdir -p "$HOME/.pki/nssdb"
certutil -d "sql:$HOME/.pki/nssdb" -A -t "C,," -n mindthegap \
  -i ~/.config/mindthegap/cert.pem
```

## Proxy terminal

```bash
uv run mindthegap --config ./config.json
```

## Copilot terminal

This setup does not help Copilot CLI by itself, because Copilot is Node-based,
not NSS-based. For Copilot, use either:

- [Global OS trust store](./certificates-global-trust.md)
- [Per-client trust override](./certificates-per-client.md)

## Other client terminal

Open the browser or NSS-based client normally after installing the cert into
the NSS database.