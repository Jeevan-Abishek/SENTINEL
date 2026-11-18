"""
SENTINEL App Configuration Re-export
Allows backward-compatible imports:
    from app.config import settings, Settings, get_settings
    from config import settings, Settings, get_settings
"""

import sys
import importlib.util
from pathlib import Path

_sentinel_dir = Path(__file__).resolve().parent.parent

# Robustly load the root config module avoiding any self-referencing circular loop
_root_config_path = _sentinel_dir / "config.py"
if not _root_config_path.exists():
    _root_config_path = _sentinel_dir.parent / "config.py"

_spec = importlib.util.spec_from_file_location("_root_sentinel_config", str(_root_config_path))
_root_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_cfg)

Settings = _root_cfg.Settings
get_settings = _root_cfg.get_settings
settings = _root_cfg.settings

__all__ = ["Settings", "get_settings", "settings"]
