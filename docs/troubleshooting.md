# Troubleshooting

Most likely failures are listed first.

## 1. Copilot says `CAPIError: Connection error` and the debug log shows `CaUsedAsEndEntity`

Most likely cause:

The proxy is serving an older CA-style cert (`basicConstraints = CA:TRUE`).
Current Copilot CLI builds reject CA certificates when they are used directly
as HTTPS server certificates, so the TLS handshake fails before the proxy sees
an HTTP request.

Checks:

```bash
openssl x509 -in "$HOME/.config/mindthegap/cert.pem" -noout -ext basicConstraints -ext extendedKeyUsage
copilot --log-level debug --log-dir /tmp/copilot-debug -p "test" -s
rg -n 'CaUsedAsEndEntity' /tmp/copilot-debug
```

Fix:

- if you use `mindthegap`'s auto-generated certs, restart a current build and it
  will regenerate older CA-style certs automatically;
- if you use explicit `tls.cert_file` / `tls.key_file`, replace that cert with a
  proper leaf/server cert (`CA:FALSE`, `serverAuth`, correct SANs);
- if you want to prevent future surprise rotation, pin `tls.cert_file` and
  `tls.key_file` in `config.json`;
- reinstall the current `cert.pem` into the OS trust store (or refresh any
  client env vars that point to it), then restart the proxy and Copilot.

## 2. Copilot CLI says `ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]`

Most likely cause:

The Copilot shell does not trust `mindthegap`'s local certificate.

Checks:

```bash
curl --cacert ~/.config/mindthegap/cert.pem https://127.0.0.1:3333/healthz
NODE_EXTRA_CA_CERTS="$HOME/.config/mindthegap/cert.pem" \
node -e "require('https').get('https://127.0.0.1:3333/healthz', r => { console.log(r.statusCode); r.resume(); }).on('error', e => { console.error(e); process.exit(1) })"
```

Fix:

- install the certificate globally and run Copilot with no CA env vars, or
- set `NODE_EXTRA_CA_CERTS="$HOME/.config/mindthegap/cert.pem"` in the shell
  that launches `copilot`.

## 3. Copilot says `Request failed due to a transient API error. Retrying...` and the proxy logs `httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]`

Most likely cause:

The proxy shell has `SSL_CERT_FILE` or `REQUESTS_CA_BUNDLE` pointing to
`~/.config/mindthegap/cert.pem` alone. That makes Python trust only the local
proxy certificate and stop trusting DeepSeek's public certificate chain.

Checks:

```bash
env | rg '^(SSL_CERT_FILE|REQUESTS_CA_BUNDLE)='
uv run python -c "import httpx; print(httpx.get('https://api.deepseek.com', timeout=10.0).status_code)"
SSL_CERT_FILE="$HOME/.config/mindthegap/cert.pem" uv run python -c "import httpx; print(httpx.get('https://api.deepseek.com', timeout=10.0).status_code)"
```

If the plain call works and the overridden one fails, this is the problem.

Fix:

```bash
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE
uv run mindthegap --config ./config.json
```

If you really need env-based Python trust, use a combined CA bundle instead of
`cert.pem` alone.

## 4. `curl --cacert ... /healthz` works, but Copilot still fails

Most likely cause:

The `copilot` process is not running in the same shell where
`NODE_EXTRA_CA_CERTS` was exported, or a wrapper is dropping the env var.

Checks:

```bash
env | rg '^NODE_EXTRA_CA_CERTS='
NODE_EXTRA_CA_CERTS="$HOME/.config/mindthegap/cert.pem" copilot
```

Fix:

Launch `copilot` from the same terminal where you set
`NODE_EXTRA_CA_CERTS`, or use inline env assignment for the launch command.

## 5. The proxy starts, but clients fail only on hostname / SAN validation

Most likely cause:

The client is connecting to a host name or IP address that is not present in
the generated certificate's SAN list.

Checks:

```bash
openssl x509 -in "$HOME/.config/mindthegap/cert.pem" -noout -ext subjectAltName
```

Fix:

- use `https://127.0.0.1:3333` or `https://localhost:3333`, or
- delete `cert.pem` and `key.pem` so `mindthegap` regenerates them, or
- configure `tls.san_dns` / `tls.san_ip` explicitly in `config.json`.

## 6. The proxy prints trust instructions again on every restart

Most likely cause:

The cert is being regenerated because it is missing, unreadable, near expiry,
missing a required SAN entry, or no longer matches the required server-cert
profile.

Checks:

- confirm `cert.pem` and `key.pem` persist in the configured cert directory;
- check whether you changed `tls.san_dns` / `tls.san_ip`;
- inspect `openssl x509 -in "$HOME/.config/mindthegap/cert.pem" -noout -ext basicConstraints`;
- check proxy startup logs for the regeneration reason.

Fix:

Keep a stable cert directory, or provide explicit `tls.cert_file` and
`tls.key_file` paths if you want to manage the certificate yourself. If the cert
changed, reinstall the exact new `cert.pem` into your trust store.

## 7. DeepSeek returns `400 The reasoning_content in the thinking mode must be passed back`

Most likely cause:

Requests are reaching DeepSeek without the stitch/unstitch transform being
applied, or the client is bypassing `mindthegap` and talking to DeepSeek
directly.

Checks:

- confirm the client points to `https://127.0.0.1:3333/v1`;
- run the proxy at `--log-level DEBUG` and inspect the transformed payloads.

Fix:

Point the client at `mindthegap`, not at `https://api.deepseek.com`, and keep
the proxy in front of all chat-completions traffic.

## 8. Port 3333 is already in use

Most likely cause:

Another instance of `mindthegap` is already running, or another service is
bound to the same port.

Fix:

- stop the existing process, or
- run the proxy with a different port via `--https-port` (or `--http-port` when using `--no-tls`) and update the client base
  URL accordingly.