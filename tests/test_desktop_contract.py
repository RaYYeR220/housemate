"""Static checks on the Desktop half (``desktop/plugin.js``): it stays inside the Desktop plugin SDK.

Catalog rule 8 and the loader's import allowlist: only ``@hermes/plugin-sdk``, ``react`` and
``react/jsx-runtime`` are importable; no eval / new Function, no MutationObserver, no reaching
into the app's markup (``[data-…]`` selectors, ``document.querySelector``). When Hermes is
importable, its own admission lint (``hermes plugins validate``'s desktop surface check) runs too.
"""

import re
from pathlib import Path

import pytest

from housemate import desktop_api
from housemate.core import desk

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_JS = ROOT / "desktop" / "plugin.js"
ALLOWED = {"@hermes/plugin-sdk", "react", "react/jsx-runtime"}


def source() -> str:
    return PLUGIN_JS.read_text(encoding="utf-8")


def code_only(text: str) -> str:
    """The source with comments blanked (strings kept), so prose can't hide or fake a hit."""
    pattern = re.compile(r"(\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)|/\*.*?\*/|(?<![:\w])//[^\n]*", re.S)
    return pattern.sub(lambda m: m.group(1) or "\n" * m.group(0).count("\n"), text)


def test_only_sdk_and_react_imports():
    code = code_only(source())
    static = re.findall(r"^\s*import\s+(?:[^'\"]*?\s+from\s+)?['\"]([^'\"]+)['\"]", code, re.M)
    assert static, "plugin.js imports nothing?"
    assert set(static) <= ALLOWED, set(static) - ALLOWED
    assert not re.search(r"\bimport\s*\(", code), "dynamic import"
    assert not re.search(r"\brequire\s*\(", code), "require()"


@pytest.mark.parametrize("pattern, what", [
    (r"(?<![\w$.])eval\s*\(", "eval"),
    (r"\bnew\s+Function\s*\(", "new Function"),
    (r"\bFunction\s*\(\s*['\"`]", "Function constructor"),
    (r"\bMutationObserver\b", "MutationObserver"),
    (r"\[data-", "a [data-…] selector"),
    (r"\bdocument\.(?:querySelector(?:All)?|getElementsBy\w+|getElementById)\s*\(", "a document query"),
    (r"\bdocument\.body\b", "document.body"),
    (r"\bhermesDesktop\b", "the raw native bridge"),
    (r"createElement\(\s*['\"]script", "script injection"),
    (r"\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML", "raw HTML writes"),
    (r"\bfetch\s*\(|XMLHttpRequest|new\s+WebSocket\b", "own network calls (use host.request / ctx.rest)"),
    (r"\blocalStorage\b|\bsessionStorage\b", "raw storage (use ctx.storage)"),
])
def test_no_reach_outside_the_sdk(pattern, what):
    hits = [i + 1 for i, line in enumerate(code_only(source()).splitlines()) if re.search(pattern, line)]
    assert not hits, f"{what} on line(s) {hits}"


def test_no_hardcoded_colours():
    """Theme tokens only: no hex / rgb() / named colours in styles or canvas paint."""
    code = code_only(source())
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b(?![-\w])", code.replace("'#'", "")), "hex colour"
    assert not re.search(r"\brgba?\(\s*\d", code), "rgb() colour"
    assert not re.search(r"['\"](?:black|white|red|blue|green|yellow|orange|gray|grey)['\"]", code), "named colour"


def test_no_jsx_syntax_and_a_default_export():
    code = code_only(source())
    assert "export default" in code
    assert not re.search(r"return\s*\(\s*<[A-Za-z]", code), "JSX syntax does not parse in a disk plugin"
    assert re.search(r"\bid:\s*PLUGIN_ID\b", code) and "const PLUGIN_ID = 'housemate'" in code


def test_contributes_page_nav_chip_and_palette():
    code = code_only(source())
    for area in ("ROUTES_AREA", "SIDEBAR_NAV_AREA", "STATUSBAR_AREAS.right", "PALETTE_AREA"):
        assert re.search(rf"area:\s*{re.escape(area)}\b", code), area
    assert "const ROUTE = '/rooms'" in code
    assert "label: 'Rooms', codicon: 'home'" in code


def test_every_sdk_name_used_is_imported():
    """A component used in jsx() but not imported is a ReferenceError at render (SDK pitfall)."""
    code = code_only(source())
    imported = set(re.findall(r"\b([A-Z_][A-Za-z_]*)\b", re.search(r"import \{(.*?)\} from '@hermes/plugin-sdk'", code, re.S).group(1)))
    used = set(re.findall(r"jsxs?\(\s*([A-Z][A-Za-z]*)\b", code))
    local = set(re.findall(r"^function ([A-Z][A-Za-z]*)\b", code, re.M))
    runtime = set(re.findall(r"\b([A-Z][A-Za-z]*)\b", re.search(r"import \{(.*?)\} from 'react/jsx-runtime'", code, re.S).group(1)))
    assert used - imported - local - runtime == set()


def test_wire_names_match_the_python_side():
    code = code_only(source())
    assert f"const EVENT = 'plugin.housemate.{desk.EVENT}'" in code
    assert "const RPC_PREFIX = 'plugin.housemate.rooms.'" in code
    called = set(re.findall(r"store\.act\('([a-z_]+)'", code)) | {"state"}
    assert {name.split(".", 1)[1] for name in desktop_api.RPC_METHODS} == called


def test_hermes_desktop_surface_lint_passes():
    """The same lint ``hermes plugins validate`` runs (needs a Hermes >= 0.23 importable)."""
    lint = pytest.importorskip("hermes_cli.plugin_validate_desktop")
    assert lint.desktop_surface_hits(ROOT) == []
    assert lint.desktop_surface_hits(ROOT, lint.desktop_review_findings) == []


def test_pairing_is_approved_by_the_typed_code_only():
    """Review M3: names and rooms of pairing requests come from the requester, so the page never shows
    a request's code and approves only with the code the owner typed from the device, always in a
    dialog (no "confirm once, then one click" shortcut)."""
    code = code_only(source())
    assert not re.search(r"children:[^\n]*\bp\.code\b", code)
    assert "approvedOnce" not in code
    assert "store.act('pair_approve', room ? { code, room } : { code })" in code
    assert "if (!confirmSetter) return" in code
