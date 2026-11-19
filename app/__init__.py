"""Root forwarder package for SENTINEL app."""
import sys
from pathlib import Path

_sentinel_dir = Path(__file__).resolve().parent.parent / "sentinel"
_sentinel_app = _sentinel_dir / "app"

__path__ = [str(_sentinel_app), str(Path(__file__).resolve().parent)]

if str(_sentinel_dir) not in sys.path:
    sys.path.insert(0, str(_sentinel_dir))
