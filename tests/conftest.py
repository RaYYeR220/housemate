import sys
from pathlib import Path

# Make the plugin importable as the package ``housemate`` (the repo root is the
# plugin root, the same layout Hermes loads from ~/.hermes/plugins/housemate).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
