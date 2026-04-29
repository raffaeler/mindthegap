import os
from pathlib import Path

import pytest
from cryptography import x509

from mindthegap.config import Settings, TlsConfig
from mindthegap.tls import ensure_cert, generate_self_signed


def _load_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _san_dns(cert: x509.Certificate) -> list[str]:
    ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return list(ext.get_values_for_type(x509.DNSName))


def _san_ip(cert: x509.Certificate) -> list[str]:
    ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return [str(ip) for ip in ext.get_values_for_type(x509.IPAddress)]


def test_generate_self_signed_writes_cert_with_required_sans(tmp_path: Path):
    tls = TlsConfig(san_dns=["localhost", "myhost"], san_ip=["127.0.0.1", "::1"])
    cert_path = tmp_path / "c.pem"
    key_path = tmp_path / "k.pem"
    generate_self_signed(tls, cert_path, key_path)
    assert cert_path.exists() and key_path.exists()

    cert = _load_cert(cert_path)
    assert set(_san_dns(cert)) >= {"localhost", "myhost"}
    assert set(_san_ip(cert)) == {"127.0.0.1", "::1"}

    bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert bc.ca is True
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    from cryptography.x509.oid import ExtendedKeyUsageOID

    assert ExtendedKeyUsageOID.SERVER_AUTH in list(eku)

    # not_valid_after in the future
    import datetime as dt

    assert cert.not_valid_after_utc > dt.datetime.now(dt.UTC)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only file mode check")
def test_key_file_mode_is_0600_on_posix(tmp_path: Path):
    tls = TlsConfig()
    cert_path = tmp_path / "c.pem"
    key_path = tmp_path / "k.pem"
    generate_self_signed(tls, cert_path, key_path)
    mode = key_path.stat().st_mode & 0o777
    assert mode == 0o600


def _settings_with_dir(tmp_path: Path, **tls_kwargs: object) -> Settings:
    return Settings(tls=TlsConfig(cert_dir=str(tmp_path), **tls_kwargs))  # type: ignore[arg-type]


def test_ensure_cert_creates_when_missing(tmp_path: Path):
    s = _settings_with_dir(tmp_path)
    cert_path, key_path, generated = ensure_cert(s)
    assert cert_path.exists() and key_path.exists()
    assert generated is True
    mtime = cert_path.stat().st_mtime

    # Calling again reuses the existing files.
    cert_path2, _, generated2 = ensure_cert(s)
    assert cert_path2 == cert_path
    assert cert_path.stat().st_mtime == mtime
    assert generated2 is False


def test_ensure_cert_regenerates_when_san_missing(tmp_path: Path):
    s1 = _settings_with_dir(tmp_path, san_dns=["localhost"], san_ip=["127.0.0.1"])
    cert_path, _, gen1 = ensure_cert(s1)
    assert gen1 is True
    assert "myhost.example" not in _san_dns(_load_cert(cert_path))
    first_serial = _load_cert(cert_path).serial_number

    s2 = _settings_with_dir(
        tmp_path,
        san_dns=["localhost", "myhost.example"],
        san_ip=["127.0.0.1"],
    )
    cert_path2, _, gen2 = ensure_cert(s2)
    assert cert_path2 == cert_path
    assert gen2 is True
    cert = _load_cert(cert_path2)
    assert "myhost.example" in _san_dns(cert)
    assert cert.serial_number != first_serial


def test_ensure_cert_regenerates_when_near_expiry(tmp_path: Path):
    s = _settings_with_dir(tmp_path, validity_days=1, renew_within_days=30)
    cert_path, _, gen1 = ensure_cert(s)
    assert gen1 is True
    first_serial = _load_cert(cert_path).serial_number

    # Same settings: validity (1d) is within the renew window (30d) so it
    # should be regenerated on the next call.
    cert_path2, _, gen2 = ensure_cert(s)
    assert gen2 is True
    assert _load_cert(cert_path2).serial_number != first_serial


def test_ensure_cert_explicit_paths_must_exist(tmp_path: Path):
    s = Settings(
        tls=TlsConfig(
            cert_file=str(tmp_path / "missing.pem"),
            key_file=str(tmp_path / "missing.key"),
        )
    )
    with pytest.raises(FileNotFoundError):
        ensure_cert(s)
