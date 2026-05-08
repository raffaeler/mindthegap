# Certificate use case: combined CA bundle

Use this when you cannot install the certificate globally but still want a CA
bundle that contains both the normal public roots and the local `mindthegap`
certificate.

## Certificate installation

Create a combined CA bundle by appending `mindthegap`'s `cert.pem` to the
default CA bundle used by your environment.

Linux / macOS example with Python `certifi`:

```bash
cat "$(uv run python -c 'import certifi; print(certifi.where())')" \
  "$HOME/.config/mindthegap/cert.pem" \
  > "$HOME/.config/mindthegap/combined-ca.pem"
```

If your platform uses a different CA bundle, replace the source path
accordingly.

## Proxy terminal

```bash
export SSL_CERT_FILE="$HOME/.config/mindthegap/combined-ca.pem"
export REQUESTS_CA_BUNDLE="$HOME/.config/mindthegap/combined-ca.pem"
uv run mindthegap --config ./config.json
```

This is safe because the combined bundle still contains the public roots needed
for `https://api.deepseek.com`.

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

Python clients that connect directly to `mindthegap` can use the same combined
bundle:

```bash
export SSL_CERT_FILE="$HOME/.config/mindthegap/combined-ca.pem"
export REQUESTS_CA_BUNDLE="$HOME/.config/mindthegap/combined-ca.pem"
python your_client.py
```