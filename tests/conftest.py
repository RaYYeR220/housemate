import importlib.util
import sys
from pathlib import Path

# Make the plugin importable as the package ``housemate`` (the repo root is the
# plugin root, the same layout Hermes loads from ~/.hermes/plugins/housemate).
ROOT = Path(__file__).resolve().parents[1]
if ROOT.name == "housemate":
    sys.path.insert(0, str(ROOT.parent))
else:  # a checkout under another folder name (a git worktree): load this one as ``housemate``
    _spec = importlib.util.spec_from_file_location(
        "housemate", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["housemate"] = _pkg
    _spec.loader.exec_module(_pkg)
