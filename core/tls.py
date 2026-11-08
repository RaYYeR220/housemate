"""Self-signed TLS for the LAN hub (phone browsers only give mic access to secure origins)."""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import socket
from pathlib import Path

__all__ = ["CERT_FILE", "KEY_FILE", "ensure_self_signed_cert", "lan_hosts", "lan_ipv4s"]

CERT_FILE = "hub-cert.pem"
KEY_FILE = "hub-key.pem"
VALIDITY_DAYS = 825
RENEW_BEFORE_DAYS = 30


def lan_ipv4s() -> list[str]:
    """Best-effort list of this machine's non-loopback IPv4 addresses."""
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    # The routing table's pick for a LAN peer; connect() on UDP sends no packet.
    for probe in ("192.168.0.1", "10.255.255.255"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 1))
            found.append(s.getsockname()[0])
        except OSError:
            pass
        finally:
            s.close()
    out: list[str] = []
    for ip in found:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if addr.is_loopback or addr.is_unspecified or addr.is_link_local:
            continue
        if ip not in out:
            out.append(ip)
    return out


def lan_hosts() -> list[str]:
    """Names and addresses the hub certificate should cover."""
    hosts = lan_ipv4s()
    try:
        name = socket.gethostname()
        if name:
            hosts += [name, f"{name}.local"]
    except OSError:
        pass
    hosts += ["localhost", "127.0.0.1"]
    seen: list[str] = []
    for h in hosts:
        if h and h not in seen:
            seen.append(h)
    return seen


def _fingerprint(cert) -> str:
    from cryptography.hazmat.primitives import hashes

    return cert.fingerprint(hashes.SHA256()).hex(":").upper()


def _covered(cert, hosts: list[str]) -> bool:
    from cryptography import x509

    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return False
    have = {str(ip) for ip in ext.get_values_for_type(x509.IPAddress)}
    have |= {n.lower() for n in ext.get_values_for_type(x509.DNSName)}
    want = {h.lower() for h in hosts}
    return want <= have


def ensure_self_signed_cert(data_dir: Path, hosts: list[str]) -> tuple[Path, Path, str]:
    """Return (cert, key, sha256 fingerprint), creating or renewing the pair as needed.

    The existing pair is reused while it is valid for 30+ more days and its SAN still
    covers every host given; otherwise (e.g. the LAN IP changed) a new one is made.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = data_dir / CERT_FILE, data_dir / KEY_FILE
    now = dt.datetime.now(dt.timezone.utc)

    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            fresh = cert.not_valid_after_utc - now > dt.timedelta(days=RENEW_BEFORE_DAYS)
            if fresh and _covered(cert, hosts):
                return cert_path, key_path, _fingerprint(cert)
        except ValueError:
            pass  # unreadable: regenerate

    key = ec.generate_private_key(ec.SECP256R1())
    san: list[x509.GeneralName] = []
    for h in hosts:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            san.append(x509.DNSName(h))
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Housemate hub")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )

    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key_bytes)
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path, _fingerprint(cert)
