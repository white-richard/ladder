import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CODEBASE = SRC / "codebase"

for path in (SRC, CODEBASE):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
