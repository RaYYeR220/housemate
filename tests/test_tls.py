import ipaddress

from cryptography import x509

from housemate.core.tls import ensure_self_signed_cert, lan_hosts


def _sans(cert_path):
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    ips = {str(ip) for ip in ext.get_values_for_type(x509.IPAddress)}
    names = set(ext.get_values_for_type(x509.DNSName))
    return cert, ips | names


def test_cert_generated_with_sans_and_fingerprint(tmp_path):
    cert_path, key_path, fp = ensure_self_signed_cert(tmp_path, ["192.168.1.20", "myhost", "localhost"])
    assert cert_path.exists() and key_path.exists()
    cert, sans = _sans(cert_path)
    assert {"192.168.1.20", "myhost", "localhost"} <= sans
    assert (cert.not_valid_after_utc - cert.not_valid_before_utc).days <= 825
    hex_part = fp.replace(":", "")
    assert len(hex_part) == 64 and int(hex_part, 16) >= 0
    assert b"PRIVATE KEY" in key_path.read_bytes()


def test_cert_reused_when_hosts_unchanged(tmp_path):
    first = ensure_self_signed_cert(tmp_path, ["192.168.1.20", "localhost"])
    second = ensure_self_signed_cert(tmp_path, ["localhost", "192.168.1.20"])
    assert first[2] == second[2]


def test_cert_regenerated_when_lan_ip_changed(tmp_path):
    first = ensure_self_signed_cert(tmp_path, ["192.168.1.20", "localhost"])
    second = ensure_self_signed_cert(tmp_path, ["192.168.1.77", "localhost"])
    assert first[2] != second[2]
    _, sans = _sans(second[0])
    assert "192.168.1.77" in sans


def test_lan_hosts_include_loopback_names():
    hosts = lan_hosts()
    assert "localhost" in hosts and "127.0.0.1" in hosts
    for h in hosts:
        try:
            assert ipaddress.ip_address(h).version == 4
        except ValueError:
            assert h  # a hostname
