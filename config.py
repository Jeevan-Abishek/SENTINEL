"""
SENTINEL Workspace Root Configuration Module
Provides direct access to application settings:
    from config import settings, Settings, get_settings
"""

import sys
from pathlib import Path

# Add sentinel directory and workspace root to sys.path
_workspace_dir = Path(__file__).resolve().parent
_sentinel_dir = _workspace_dir / "sentinel"

for _p in [str(_sentinel_dir), str(_workspace_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sentinel.config import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
