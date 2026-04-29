"""Self-signed TLS certificate management for mindthegap."""

from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import os
import platform
import socket
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from .config import Settings, TlsConfig

logger = logging.getLogger("mindthegap.tls")

CERT_FILENAME = "cert.pem"
KEY_FILENAME = "key.pem"


def _default_cert_dir() -> Path:
    """OS-appropriate user-config directory for the proxy."""
    if os.name == "nt" or platform.system() == "Windows":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "mindthegap"
        return Path.home() / "AppData" / "Roaming" / "mindthegap"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "mindthegap"
    return Path.home() / ".config" / "mindthegap"


def resolve_cert_paths(tls: TlsConfig) -> tuple[Path, Path]:
    """Return ``(cert_path, key_path)`` honoring overrides in ``tls``."""
    if tls.cert_file and tls.key_file:
        return Path(tls.cert_file).expanduser(), Path(tls.key_file).expanduser()
    cert_dir = Path(tls.cert_dir).expanduser() if tls.cert_dir else _default_cert_dir()
    return cert_dir / CERT_FILENAME, cert_dir / KEY_FILENAME


def _auto_san_dns() -> list[str]:
    names = ["localhost"]
    try:
        host = socket.gethostname()
        if host and host not in names:
            names.append(host)
    except OSError:
        pass
    try:
        fqdn = socket.getfqdn()
        if fqdn and fqdn != "localhost" and fqdn not in names:
            names.append(fqdn)
    except OSError:
        pass
    return names


def _auto_san_ip() -> list[str]:
    return ["127.0.0.1", "::1"]


def _build_san(tls: TlsConfig) -> tuple[list[str], list[str], x509.SubjectAlternativeName]:
    dns_names = tls.san_dns if tls.san_dns is not None else _auto_san_dns()
    ip_names = tls.san_ip if tls.san_ip is not None else _auto_san_ip()
    entries: list[x509.GeneralName] = [x509.DNSName(n) for n in dns_names]
    entries.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in ip_names)
    return dns_names, ip_names, x509.SubjectAlternativeName(entries)


def generate_self_signed(tls: TlsConfig, cert_path: Path, key_path: Path) -> None:
    """Generate a fresh RSA-2048 self-signed cert + key at the given paths."""
    cert_path.parent.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = key.public_key()

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "mindthegap local proxy"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "mindthegap"),
        ]
    )

    _, _, san = _build_san(tls)

    now = dt.datetime.now(dt.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=tls.validity_days))
        .add_extension(san, critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(public_key),
            critical=False,
        )
    )

    cert = builder.sign(private_key=key, algorithm=hashes.SHA256())

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)
    if os.name == "posix":
        try:
            os.chmod(key_path, 0o600)
            os.chmod(cert_path, 0o644)
        except OSError as exc:
            logger.warning("Could not set permissions on cert/key: %s", exc)


def _required_san(tls: TlsConfig) -> tuple[set[str], set[str]]:
    dns_names = tls.san_dns if tls.san_dns is not None else _auto_san_dns()
    ip_names = tls.san_ip if tls.san_ip is not None else _auto_san_ip()
    return set(dns_names), {str(ipaddress.ip_address(ip)) for ip in ip_names}


def _cert_needs_refresh(cert_path: Path, tls: TlsConfig) -> str | None:
    """Return a reason string if the cert must be regenerated, else None."""
    if not cert_path.exists():
        return "missing"
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    except ValueError as exc:
        return f"unparseable ({exc})"

    try:
        not_after = cert.not_valid_after_utc
    except AttributeError:  # pragma: no cover - cryptography < 42
        not_after = cert.not_valid_after.replace(tzinfo=dt.UTC)

    threshold = dt.datetime.now(dt.UTC) + dt.timedelta(days=tls.renew_within_days)
    if not_after <= threshold:
        return f"expires {not_after.isoformat()}"

    try:
        san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return "missing SAN extension"

    cert_dns = {name for name in san_ext.get_values_for_type(x509.DNSName)}
    cert_ip = {str(ip) for ip in san_ext.get_values_for_type(x509.IPAddress)}
    needed_dns, needed_ip = _required_san(tls)
    missing_dns = needed_dns - cert_dns
    missing_ip = needed_ip - cert_ip
    if missing_dns or missing_ip:
        return f"SAN missing: dns={sorted(missing_dns)}, ip={sorted(missing_ip)}"

    return None


def ensure_cert(settings: Settings) -> tuple[Path, Path, bool]:
    """Make sure cert+key exist and are valid.

    Returns ``(cert_path, key_path, generated)`` where ``generated`` is True
    when a fresh certificate was just written (new install, expiry, SAN
    mismatch, ...). When False the existing on-disk cert was reused and the
    user is presumed to have already trusted it.
    """
    cert_path, key_path = resolve_cert_paths(settings.tls)

    if settings.tls.cert_file and settings.tls.key_file:
        # User-supplied: do not touch.
        if not cert_path.exists() or not key_path.exists():
            raise FileNotFoundError(f"Configured TLS cert/key not found: {cert_path}, {key_path}")
        return cert_path, key_path, False

    reason = _cert_needs_refresh(cert_path, settings.tls)
    key_missing = not key_path.exists()
    if reason or key_missing:
        why = reason or "key file missing"
        logger.info("Generating self-signed cert (%s) at %s", why, cert_path)
        generate_self_signed(settings.tls, cert_path, key_path)
        return cert_path, key_path, True

    logger.info("Reusing existing self-signed cert at %s", cert_path)
    return cert_path, key_path, False


def print_trust_instructions(cert_path: Path, host: str, port: int) -> None:
    """Print the full block telling the user how to trust the cert.

    Only call this when a new cert was just generated. When an existing cert
    is being reused, prefer ``print_cert_reused`` to keep startup quiet.
    """
    cert = str(cert_path.resolve())
    base = f"https://{host}:{port}"
    msg = f"""
================================================================================
mindthegap is serving HTTPS at {base}
Self-signed certificate: {cert}

To trust this certificate so clients (Copilot CLI, curl, browsers) accept it:

  Linux (system-wide CA bundle, used by curl/python/etc.):
    sudo cp "{cert}" /usr/local/share/ca-certificates/mindthegap.crt
    sudo update-ca-certificates

  Linux (current user, NSS DB used by Firefox / Chrome):
    mkdir -p "$HOME/.pki/nssdb"
    certutil -d "sql:$HOME/.pki/nssdb" -A -t "C,," -n mindthegap -i "{cert}"

  macOS (login keychain, current user):
    security add-trusted-cert -r trustRoot \\
      -k "$HOME/Library/Keychains/login.keychain-db" "{cert}"

  macOS (system-wide, requires sudo):
    sudo security add-trusted-cert -d -r trustRoot \\
      -k /Library/Keychains/System.keychain "{cert}"

  Windows (PowerShell as Administrator, machine-wide):
    Import-Certificate -FilePath "{cert}" `
      -CertStoreLocation Cert:\\LocalMachine\\Root

  Windows (current user, no admin needed):
    Import-Certificate -FilePath "{cert}" `
      -CertStoreLocation Cert:\\CurrentUser\\Root

Some tools ignore the OS trust store and read CA bundles from env vars.
For Copilot CLI (Node-based) and many SDKs, set:

  Linux/macOS:
    export NODE_EXTRA_CA_CERTS="{cert}"
    export SSL_CERT_FILE="{cert}"
    export REQUESTS_CA_BUNDLE="{cert}"

  Windows (PowerShell):
    $env:NODE_EXTRA_CA_CERTS = "{cert}"
    $env:SSL_CERT_FILE       = "{cert}"
    $env:REQUESTS_CA_BUNDLE  = "{cert}"

Then restart the client so it picks up the new env / trust store.
================================================================================
"""
    print(msg, file=sys.stderr, flush=True)


def print_cert_reused(cert_path: Path, host: str, port: int) -> None:
    """Short startup note when reusing an already-installed certificate.

    Most developers trust the proxy cert exactly once and then forget about
    it; spamming the full instruction block on every restart adds noise.
    The full block can still be re-printed on demand by deleting the cert
    (it will be regenerated and treated as new) or by inspecting this file.
    """
    cert = str(cert_path.resolve())
    msg = (
        f"mindthegap: HTTPS at https://{host}:{port} (reusing cert at {cert}). "
        f"If clients complain about TLS, re-run the trust steps from the README "
        f"or delete the cert to regenerate and re-print full instructions."
    )
    print(msg, file=sys.stderr, flush=True)
