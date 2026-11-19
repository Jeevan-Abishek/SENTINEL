"""
SENTINEL Workspace Root Database Entrypoint.
Enables running from workspace root:
    from database import get_db, init_db, engine, Base
"""

import sys
from pathlib import Path

_sentinel_dir = Path(__file__).resolve().parent / "sentinel"
if str(_sentinel_dir) not in sys.path:
    sys.path.insert(0, str(_sentinel_dir))

from sentinel.database import *

