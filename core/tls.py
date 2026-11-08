"""Self-signed TLS for the LAN hub (phone browsers only give mic access to secure origins)."""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import re
import socket
from pathlib import Path

__all__ = ["CERT_FILE", "KEY_FILE", "ensure_self_signed_cert", "hub_urls", "lan_hosts", "lan_ipv4s", "rank_lan_ipv4s"]

CERT_FILE = "hub-cert.pem"
KEY_FILE = "hub-key.pem"
VALIDITY_DAYS = 825
MAX_SANS = 48  # names and addresses kept in the certificate (new ones first)
RENEW_BEFORE_DAYS = 30


# Adapters a phone on the home Wi-Fi cannot reach: VPNs, VM host-only networks, containers.
_VIRTUAL_NAME = re.compile(
    r"virtualbox|vbox|vmware|vmnet|vethernet|hyper-v|wsl|docker|podman|virbr|^br-|^veth|"
    r"tailscale|zerotier|^zt|wireguard|^wg\d|^utun|^tun\d|^tap\d|npcap|bluetooth",
    re.I,
)
_CGNAT = ipaddress.ip_network("100.64.0.0/10")  # Tailscale and carrier-grade NAT
_VIRTUAL_NETS = (ipaddress.ip_network("192.168.56.0/24"),)  # VirtualBox's default host-only network
_HOME_NETS = (ipaddress.ip_network("192.168.0.0/16"), ipaddress.ip_network("10.0.0.0/8"))
_OTHER_PRIVATE = ipaddress.ip_network("172.16.0.0/12")
_ROUTE_PROBES = ("1.1.1.1", "192.168.0.1", "10.255.255.255")  # default route, then LAN routes


def _interface_addresses() -> list[tuple[str, str]]:
    """``(interface name, IPv4)`` for every interface that is up; names are "" when unknown."""
    found: list[tuple[str, str]] = []
    try:
        import psutil  # optional: present wherever Hermes is installed

        stats = psutil.net_if_stats()
        for name, addrs in psutil.net_if_addrs().items():
            if name in stats and not stats[name].isup:
                continue
            found += [(name, a.address) for a in addrs if a.family == socket.AF_INET]
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(("", info[4][0]))
    except OSError:
        pass
    return found


def _route_ips() -> list[str]:
    """The source address the routing table picks for the internet and for LAN peers. ``connect()``
    on a UDP socket sends no packet."""
    picked: list[str] = []
    for probe in _ROUTE_PROBES:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 1))
            picked.append(s.getsockname()[0])
        except OSError:
            pass
        finally:
            s.close()
    return picked


def rank_lan_ipv4s(interfaces: list[tuple[str, str]], route_ips: list[str]) -> list[str]:
    """This machine's IPv4 addresses, the one a phone on the home network most likely reaches first.

    1. a private address (192.168/16, 10/8, 172.16/12) the default route uses, on a real adapter;
    2. other 192.168/16 and 10/8 addresses; 3. other 172.16/12 ones (often WSL, Docker);
    4. other routable ones; 5. virtual adapters (VirtualBox, Hyper-V/WSL, VPNs, containers);
    6. CGNAT 100.64/10 (Tailscale). Loopback, link-local and multicast are left out.
    """
    names: dict[str, str] = {}
    order: list[str] = []
    for name, ip in [*interfaces, *(("", ip) for ip in route_ips)]:
        if ip not in names:
            order.append(ip)
            names[ip] = name or ""
        elif name and not names[ip]:
            names[ip] = name
    routed = set(route_ips)

    def tier(ip: str, addr: ipaddress.IPv4Address) -> int:
        virtual = bool(_VIRTUAL_NAME.search(names[ip])) or any(addr in net for net in _VIRTUAL_NETS)
        if addr in _CGNAT:
            return 6
        if virtual:
            return 5
        if addr.is_private and ip in routed:
            return 1
        if any(addr in net for net in _HOME_NETS):
            return 2
        if addr in _OTHER_PRIVATE:
            return 3
        return 4

    ranked: list[tuple[int, int, str]] = []
    for position, ip in enumerate(order):
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if addr.version != 4 or addr.is_loopback or addr.is_unspecified or addr.is_link_local or addr.is_multicast:
            continue
        ranked.append((tier(ip, addr), position, ip))
    return [ip for _, _, ip in sorted(ranked)]


def lan_ipv4s() -> list[str]:
    """This machine's non-loopback IPv4 addresses, best first for the QR code and the CLI."""
    return rank_lan_ipv4s(_interface_addresses(), _route_ips())


def hub_urls(scheme: str, port: int, advertise_host: str = "") -> list[str]:
    """The URLs a satellite can open, best first (the QR code and ``hermes housemate pair`` use the
    first): ``platforms.housemate.advertise_host`` when set, the ranked LAN addresses, localhost."""
    hosts = [h for h in (advertise_host, *lan_ipv4s()) if h]
    urls = [f"{scheme}://{h}:{port}" for h in dict.fromkeys(hosts)]
    return urls + [f"{scheme}://localhost:{port}"]


def lan_hosts(extra: list[str] | None = None) -> list[str]:
    """Names and addresses the hub certificate should cover (``extra`` first: an advertised host)."""
    hosts = [h for h in (extra or []) if h] + lan_ipv4s()
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


def _san_names(cert) -> list[str]:
    from cryptography import x509

    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return []
    return [str(ip) for ip in ext.get_values_for_type(x509.IPAddress)] + list(ext.get_values_for_type(x509.DNSName))


def _load_key(key_path: Path):
    """The hub's existing EC key, reused when the certificate is renewed or gains a name."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    try:
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError):
        return None
    return key if isinstance(key, ec.EllipticCurvePrivateKey) else None


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

    known: list[str] = []  # names the current certificate covers: kept, so a VPN toggle is no flip-flop
    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            fresh = cert.not_valid_after_utc - now > dt.timedelta(days=RENEW_BEFORE_DAYS)
            if fresh and _covered(cert, hosts):
                return cert_path, key_path, _fingerprint(cert)
            known = _san_names(cert)
        except ValueError:
            pass  # unreadable: regenerate

    key = _load_key(key_path) or ec.generate_private_key(ec.SECP256R1())
    hosts = list(dict.fromkeys([*hosts, *known]))[:MAX_SANS]
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
