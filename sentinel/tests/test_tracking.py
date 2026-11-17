"""Tests for Phase 3: Person Tracking with ByteTrack."""
import sys
from pathlib import Path
_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from unit.test_tracking import *
