"""
SENTINEL App Configuration Re-export
Allows backward-compatible imports:
    from app.config import settings, Settings, get_settings
    from config import settings, Settings, get_settings
"""

import sys
from pathlib import Path

# Ensure sentinel root is in sys.path
_sentinel_dir = Path(__file__).resolve().parent.parent
if str(_sentinel_dir) not in sys.path:
    sys.path.insert(0, str(_sentinel_dir))

from config import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
