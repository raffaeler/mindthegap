# TLS / self-signed certificate

`mindthegap` serves HTTPS by default. On first launch the proxy generates its own
self-signed leaf/server certificate and reuses it on subsequent launches.

To serve plain HTTP instead (e.g. for local development behind a reverse proxy
that handles TLS), set `tls.enabled` to `false` in `config.json` or pass
`--no-tls` on the command line.

Certificate location:

| OS | Path |
| --- | --- |
| Linux / macOS | `$XDG_CONFIG_HOME/mindthegap/` (default `~/.config/mindthegap/`) |
| Windows | `%APPDATA%\mindthegap\` |

Generated files:

- `cert.pem`: public certificate
- `key.pem`: private key (`0600` on POSIX)

Generated certificates are TLS server certificates (`CA:FALSE` with
`extendedKeyUsage = serverAuth`), not CA certificates. Recent GitHub Copilot
CLI builds reject CA certificates presented as HTTPS server certificates.

By default, auto-generated certs are stable for local use and include only:

- DNS: `localhost`
- IP: `127.0.0.1`
- IP: `::1`

Machine hostname / FQDN SANs are **not** added automatically. If you need to
connect to the proxy via another hostname or IP, set `tls.san_dns` /
`tls.san_ip` explicitly.

The cert is regenerated automatically only when it:

- is missing or unparseable;
- expires within `tls.renew_within_days`;
- no longer covers the required SAN entries;
- no longer matches the required server-certificate profile (for example an
  older auto-generated `CA:TRUE` cert from a previous release).

If you set both `tls.cert_file` and `tls.key_file` in `config.json`, auto-
generation is disabled and `mindthegap` uses your provided files as-is. In that
case, you are responsible for supplying a valid server certificate with the
needed SANs.

## Stable pinned-certificate setup

If you install `mindthegap`'s cert into the global OS trust store, and you want
to avoid invalidating that trust with a later regeneration, pin the exact cert
and key paths in `config.json`:

```json
{
  "tls": {
    "cert_file": "/home/you/.config/mindthegap/cert.pem",
    "key_file": "/home/you/.config/mindthegap/key.pem"
  }
}
```

With both fields set, `mindthegap` stops auto-generating and reusing SAN/expiry
rules for that pair and serves those files exactly as provided. This is the
recommended setup when you manually replace `cert.pem` or trust it globally.

## Trust prompt behavior

- First launch or new certificate: full trust instructions are printed.
- Reused certificate: only a short one-line note is printed.

If you want to re-print the full instructions, delete `cert.pem` and `key.pem`
from the configured cert directory and start the proxy again.

## Upgrading from older auto-generated certs

If you previously trusted an older `mindthegap` cert, and startup regenerates a
new one, reinstall the exact new `cert.pem` into your trust store (or refresh
the client env vars that point to it). Updating the file on disk alone does not
update the OS trust store automatically.

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