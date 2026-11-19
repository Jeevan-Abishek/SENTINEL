"""Root forwarder for app.main."""
import sys
from pathlib import Path

_sentinel = Path(__file__).resolve().parent.parent / "sentinel"
if str(_sentinel) not in sys.path:
    sys.path.insert(0, str(_sentinel))

from app.api.main import app, manager, lifespan

__all__ = ["app", "manager", "lifespan"]

