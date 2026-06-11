# Certificate use case: per-client trust override

Use this when you want the lightest-touch setup and the proxy itself should
keep using the default system/public CA roots.

## Certificate installation

No global installation is required. Reuse the generated `cert.pem` directly
from the `mindthegap` config directory. The generated cert must be a TLS server
certificate (`CA:FALSE`); newer Copilot CLI builds reject CA-style server certs.

## Proxy terminal

```bash
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE
uv run mindthegap --config ./config.json
```

Do not point `SSL_CERT_FILE` or `REQUESTS_CA_BUNDLE` at
`~/.config/mindthegap/cert.pem` in the proxy shell. Doing that replaces the
public CA roots and breaks outbound TLS to DeepSeek.

## Copilot terminal

```bash
export NODE_EXTRA_CA_CERTS="$HOME/.config/mindthegap/cert.pem"
export COPILOT_PROVIDER_BASE_URL="https://127.0.0.1:3333/v1"
export COPILOT_PROVIDER_API_KEY="sk-...your-deepseek-key..."
export COPILOT_PROVIDER_TYPE="openai"
export COPILOT_MODEL="deepseek-reasoner"
copilot
```

## Other client terminal

For Python clients that connect directly to `mindthegap`, do not use
`cert.pem` alone. Use a combined CA bundle instead, as described in
[Combined CA bundle](./certificates-combined-ca.md).