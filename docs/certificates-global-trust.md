# Certificate use case: global OS trust store

Use this when both the proxy and local clients should keep using the normal
system trust store, with no special CA environment variables.

For the least surprising long-term behavior, also pin the cert/key paths in
`config.json` so later proxy restarts do not auto-regenerate a different cert
and invalidate the trusted copy:

```json
{
  "tls": {
    "cert_file": "/home/you/.config/mindthegap/cert.pem",
    "key_file": "/home/you/.config/mindthegap/key.pem"
  }
}
```

This is especially important if you replace `cert.pem` manually or distribute
it to other trust stores.

## Certificate installation

Linux (Debian/Ubuntu):

```bash
sudo install -m 0644 ~/.config/mindthegap/cert.pem /usr/local/share/ca-certificates/mindthegap.crt
sudo update-ca-certificates
```

If `mindthegap` regenerates `cert.pem` later, rerun both commands so the OS
trust store is updated to the current exact certificate.

Linux (Fedora/RHEL): copy `cert.pem` to
`/etc/pki/ca-trust/source/anchors/` and then run:

```bash
sudo update-ca-trust
```

macOS:

```bash
security add-trusted-cert -r trustRoot \
  -k "$HOME/Library/Keychains/login.keychain-db" \
  ~/.config/mindthegap/cert.pem
```

Windows (current user, no admin):

```powershell
Import-Certificate -FilePath "$env:APPDATA\mindthegap\cert.pem" `
  -CertStoreLocation Cert:\CurrentUser\Root
```

Windows (machine-wide, Administrator):

```powershell
Import-Certificate -FilePath "$env:APPDATA\mindthegap\cert.pem" `
  -CertStoreLocation Cert:\LocalMachine\Root
```

Windows GUI alternative: double-click `cert.pem`, choose *Install Certificate*,
then place it in **Trusted Root Certification Authorities**.

## Proxy terminal

```bash
uv run mindthegap --config ./config.json
```

Do not set `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, or `NODE_EXTRA_CA_CERTS` for
the proxy in this mode.

## Copilot terminal

```bash
export COPILOT_PROVIDER_BASE_URL="https://127.0.0.1:3333/v1"
export COPILOT_PROVIDER_API_KEY="sk-...your-deepseek-key..."
export COPILOT_PROVIDER_TYPE="openai"
export COPILOT_MODEL="deepseek-reasoner"
copilot
```

## Other client terminal

If the client uses the OS trust store normally, point it at the proxy and do
not set any CA environment variables.