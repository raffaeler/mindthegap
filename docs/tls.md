# TLS / self-signed certificate

`mindthegap` always serves HTTPS. On first launch the proxy generates its own
self-signed certificate and reuses it on subsequent launches.

Certificate location:

| OS | Path |
| --- | --- |
| Linux / macOS | `$XDG_CONFIG_HOME/mindthegap/` (default `~/.config/mindthegap/`) |
| Windows | `%APPDATA%\mindthegap\` |

Generated files:

- `cert.pem`: public certificate
- `key.pem`: private key (`0600` on POSIX)

The cert is regenerated automatically only when it:

- is missing or unparseable;
- expires within `tls.renew_within_days`;
- no longer covers the required SAN entries.

If you set both `tls.cert_file` and `tls.key_file` in `config.json`, auto-
generation is disabled and `mindthegap` uses your provided files as-is.

## Trust prompt behavior

- First launch or new certificate: full trust instructions are printed.
- Reused certificate: only a short one-line note is printed.

If you want to re-print the full instructions, delete `cert.pem` and `key.pem`
from the configured cert directory and start the proxy again.

## Choose a trust model

Pick one setup and follow only that page:

- [Global OS trust store](./certificates-global-trust.md)
- [Combined CA bundle](./certificates-combined-ca.md)
- [Per-client trust override](./certificates-per-client.md)
- [Browser / NSS-only trust](./certificates-browser-nss.md)

## Copilot CLI default

For GitHub Copilot CLI, the safest default is:

- proxy terminal: do not set `SSL_CERT_FILE` or `REQUESTS_CA_BUNDLE`
- Copilot terminal: set `NODE_EXTRA_CA_CERTS` to `mindthegap`'s `cert.pem`

See [Per-client trust override](./certificates-per-client.md) for the exact
two-terminal setup.