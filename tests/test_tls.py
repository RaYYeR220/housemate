import ipaddress

from cryptography import x509

from housemate.core import tls
from housemate.core.tls import ensure_self_signed_cert, lan_hosts, rank_lan_ipv4s


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


# --- which address the QR code and the CLI advertise (review M4) ---------------------------------

# What a Windows laptop with Tailscale, VirtualBox and WSL reports (the review's machine).
LAPTOP = [
    ("Tailscale", "100.79.155.122"),
    ("Ethernet 3", "192.168.56.1"),  # VirtualBox host-only (default range, generic name)
    ("Wi-Fi", "192.168.0.17"),
    ("vEthernet (WSL (Hyper-V firewall))", "172.27.144.1"),
    ("Ethernet", "169.254.135.162"),
    ("Loopback Pseudo-Interface 1", "127.0.0.1"),
]


def test_default_route_wifi_address_comes_first():
    ranked = rank_lan_ipv4s(LAPTOP, route_ips=["192.168.0.17"])
    assert ranked[0] == "192.168.0.17"
    # virtual adapters, then the CGNAT/Tailscale address, last; link-local and loopback never
    assert ranked == ["192.168.0.17", "192.168.56.1", "172.27.144.1", "100.79.155.122"]


def test_ranking_without_interface_names():
    """getaddrinfo order from the review: Tailscale, VirtualBox, Wi-Fi, WSL."""
    unnamed = [("", ip) for ip in ("100.79.155.122", "192.168.56.1", "192.168.0.17", "172.27.144.1")]
    ranked = rank_lan_ipv4s(unnamed, route_ips=["192.168.0.17"])
    assert ranked[0] == "192.168.0.17"
    assert ranked[-1] == "100.79.155.122"
    assert ranked.index("172.27.144.1") < ranked.index("192.168.56.1")  # host-only range is virtual


def test_private_ranges_order_without_a_route_hint():
    found = [("eth1", "172.20.1.2"), ("eth2", "10.0.0.5"), ("eth3", "192.168.1.9"), ("eth4", "203.0.113.7")]
    assert rank_lan_ipv4s(found, route_ips=[]) == ["10.0.0.5", "192.168.1.9", "172.20.1.2", "203.0.113.7"]


def test_a_vpn_default_route_does_not_win():
    """A full-tunnel VPN (Tailscale exit node) owns the default route: still not the LAN address."""
    ranked = rank_lan_ipv4s(LAPTOP, route_ips=["100.79.155.122"])
    assert ranked[0] == "192.168.0.17"


def test_a_corporate_172_network_on_the_default_route_wins():
    found = [("vEthernet (Default Switch)", "172.30.48.1"), ("Ethernet", "172.16.4.20"), ("docker0", "172.17.0.1")]
    assert rank_lan_ipv4s(found, route_ips=["172.16.4.20"])[0] == "172.16.4.20"


def test_lan_ipv4s_uses_interfaces_and_routing_and_keeps_all_for_the_cert(monkeypatch):
    monkeypatch.setattr(tls, "_interface_addresses", lambda: list(LAPTOP))
    monkeypatch.setattr(tls, "_route_ips", lambda: ["192.168.0.17"])
    assert tls.lan_ipv4s()[0] == "192.168.0.17"
    hosts = tls.lan_hosts()
    assert {"192.168.0.17", "192.168.56.1", "172.27.144.1", "100.79.155.122"} <= set(hosts)
    assert "169.254.135.162" not in hosts


def test_advertised_host_goes_first(monkeypatch):
    monkeypatch.setattr(tls, "_interface_addresses", lambda: list(LAPTOP))
    monkeypatch.setattr(tls, "_route_ips", lambda: ["192.168.0.17"])
    assert tls.lan_hosts(extra=["hub.home.arpa"])[0] == "hub.home.arpa"


def test_hub_urls_put_an_advertised_host_first(monkeypatch):
    monkeypatch.setattr(tls, "_interface_addresses", lambda: list(LAPTOP))
    monkeypatch.setattr(tls, "_route_ips", lambda: ["192.168.0.17"])
    assert tls.hub_urls("https", 8742) == [
        "https://192.168.0.17:8742", "https://192.168.56.1:8742", "https://172.27.144.1:8742",
        "https://100.79.155.122:8742", "https://localhost:8742",
    ]
    assert tls.hub_urls("https", 8742, "hub.home.arpa")[:2] == ["https://hub.home.arpa:8742", "https://192.168.0.17:8742"]


def test_config_advertise_host():
    from housemate.core.config import HousemateConfig

    assert HousemateConfig.from_extra({}).advertise_host == ""
    assert HousemateConfig.from_extra({"advertise_host": " 192.168.0.17 "}).advertise_host == "192.168.0.17"
    bad = HousemateConfig.from_extra({"advertise_host": "http://x/<y>"})
    assert bad.advertise_host == "" and bad.warnings


def test_cert_keeps_old_addresses_so_a_vpn_toggle_does_not_flip_it(tmp_path):
    """Review L11: a new address adds to the certificate (same key); switching back is covered, so the
    phones do not see a new certificate every time a VPN or adapter comes and goes."""
    first = ensure_self_signed_cert(tmp_path, ["192.168.1.20", "localhost"])
    key_before = first[1].read_bytes()
    second = ensure_self_signed_cert(tmp_path, ["192.168.1.20", "100.80.1.2", "localhost"])
    assert second[2] != first[2]
    assert second[1].read_bytes() == key_before
    third = ensure_self_signed_cert(tmp_path, ["192.168.1.20", "localhost"])  # VPN off again
    assert third[2] == second[2]
    _, sans = _sans(third[0])
    assert {"192.168.1.20", "100.80.1.2", "localhost"} <= sans
