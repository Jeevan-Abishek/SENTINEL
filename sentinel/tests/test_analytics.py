"""Tests for Phase 4: Crowd Counting, Density, Movement, Flow, and Bottleneck Analytics."""
import sys
from pathlib import Path
_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from unit.test_analytics import *
