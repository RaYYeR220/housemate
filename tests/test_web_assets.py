"""The satellite web app as shipped: every file present, served by the hub with the right type,
nothing fetched from outside the hub at runtime, inside the size budget, a valid manifest."""

import json
import re
import sqlite3
from pathlib import Path

import aiohttp
import pytest

from housemate.core.hub import Hub
from housemate.core.rooms import RoomRegistry
from housemate.core.store import HousemateStore

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
FONTS = WEB / "fonts"

REQUIRED = [
    "index.html", "app.js", "ui.js", "face.js", "player.js", "mic-worklet.js", "style.css",
    "manifest.webmanifest", "icon.svg", "chime.wav", "fonts/archivo-expanded-500.woff2", "fonts/OFL.txt",
]
TEXT_SUFFIXES = {".html", ".js", ".css", ".webmanifest", ".svg", ".json"}
BUDGET = 150 * 1024  # everything but the fonts
FONT_BUDGET = 120 * 1024

URL_RE = re.compile(r"\b(?:https?|wss?|ftp)://[A-Za-z0-9\[]")
SVG_NS_RE = re.compile(r'xmlns(?::\w+)?="http://www\.w3\.org/[^"]*"')
PROTO_RELATIVE_RE = re.compile(r"""(?:src|href|action)\s*=\s*["']//|url\(\s*["']?//|from\s+["']//|import\(\s*["']//""")


def _web_files() -> list[Path]:
    return sorted(p for p in WEB.rglob("*") if p.is_file())


def _text_files() -> list[Path]:
    return [p for p in _web_files() if p.suffix.lower() in TEXT_SUFFIXES]


def test_required_files_present():
    missing = [name for name in REQUIRED if not (WEB / name).is_file()]
    assert not missing, missing


@pytest.mark.parametrize("path", _text_files(), ids=lambda p: p.relative_to(WEB).as_posix())
def test_no_external_urls(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".svg":
        text = SVG_NS_RE.sub("", text)  # an XML namespace name is never fetched
    assert not URL_RE.findall(text), URL_RE.findall(text)
    assert not PROTO_RELATIVE_RE.search(text)


def test_local_references_resolve():
    """Every src/href in the page, every url() in the CSS and every module import is a file in web/."""
    refs: list[tuple[Path, str]] = []
    html = (WEB / "index.html").read_text(encoding="utf-8")
    refs += [(WEB / "index.html", r) for r in re.findall(r'(?:src|href)="([^"#]+)"', html)]
    css = (WEB / "style.css").read_text(encoding="utf-8")
    refs += [(WEB / "style.css", r) for r in re.findall(r"url\(\s*['\"]?([^'\")]+)", css)]
    for js in WEB.glob("*.js"):
        text = js.read_text(encoding="utf-8")
        refs += [(js, r) for r in re.findall(r"""(?:import|from)\s*\(?\s*['"]([^'"]+)['"]""", text)]
        refs += [(js, r) for r in re.findall(r"""(?:addModule|fetch)\(\s*['"]([^'"]+)['"]""", text)]
    assert refs
    for source, ref in refs:
        assert (source.parent / ref).resolve().is_file(), f"{source.name} -> {ref}"


def test_size_budget():
    files = _web_files()
    app = sum(p.stat().st_size for p in files if FONTS not in p.parents)
    fonts = sum(p.stat().st_size for p in files if FONTS in p.parents)
    assert app <= BUDGET, f"web/ without fonts is {app} bytes (budget {BUDGET})"
    assert fonts <= FONT_BUDGET, f"web/fonts is {fonts} bytes (budget {FONT_BUDGET})"


def test_font_is_open_licensed():
    licence = (FONTS / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL Open Font License" in licence and "Archivo" in licence
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "fonts/archivo-expanded-500.woff2" in css
    assert (FONTS / "archivo-expanded-500.woff2").read_bytes()[:4] == b"wOF2"


def test_manifest_is_valid():
    m = json.loads((WEB / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert m["name"] and m["short_name"] and len(m["short_name"]) <= 12
    assert m["start_url"] and m["display"] in ("fullscreen", "standalone", "minimal-ui", "browser")
    for key in ("theme_color", "background_color"):
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", m[key]), key
    assert m["icons"]
    for icon in m["icons"]:
        assert (WEB / icon["src"]).is_file() and icon["sizes"] and icon["type"].startswith("image/")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'rel="manifest" href="manifest.webmanifest"' in html


def test_page_is_a_module_app_with_labelled_controls():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert '<script type="module" src="app.js">' in html
    assert '<html lang="en">' in html and 'name="viewport"' in html
    runtime_labelled = {"rname", "door"}  # ui.js sets their text and aria-label from the hub's rooms
    for attrs, inner in re.findall(r"<button\b([^>]*)>(.*?)</button>", html, re.S):
        ident = re.search(r'id="([^"]+)"', attrs)
        text = re.sub(r"<[^>]+>", "", inner).strip()
        assert "aria-label=" in attrs or text or (ident and ident.group(1) in runtime_labelled), attrs
    worklet = (WEB / "mic-worklet.js").read_text(encoding="utf-8")
    assert "registerProcessor('pcm16k'" in worklet and "OUT_RATE = 16000" in worklet and "FRAME = 320" in worklet
    app = (WEB / "app.js").read_text(encoding="utf-8")
    for flag in ("echoCancellation: true", "noiseSuppression: true", "autoGainControl: true"):
        assert flag in app
    assert "playback.done" in app and "wakeLock" in app


@pytest.fixture
async def hub():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    h = Hub(HousemateStore(conn), RoomRegistry(), host="127.0.0.1", port=0, cert=None, key=None, web_root=WEB, tls=False)
    await h.start()
    yield h
    await h.stop()


TYPES = {
    "": "text/html", "index.html": "text/html", "app.js": "text/javascript", "mic-worklet.js": "text/javascript",
    "style.css": "text/css", "manifest.webmanifest": "application/manifest+json", "icon.svg": "image/svg+xml",
    "chime.wav": "audio/wav", "fonts/archivo-expanded-500.woff2": "font/woff2",
}


async def test_hub_serves_every_file(hub):
    async with aiohttp.ClientSession() as http:
        for rel in [p.relative_to(WEB).as_posix() for p in _web_files()] + [""]:
            async with http.get(f"{hub.url}/{rel}") as resp:
                assert resp.status == 200, rel
                body = await resp.read()
                if rel in TYPES:
                    assert resp.content_type == TYPES[rel], rel
                assert resp.headers.get("X-Content-Type-Options") == "nosniff"
                if rel:
                    assert body == (WEB / rel).read_bytes(), rel
