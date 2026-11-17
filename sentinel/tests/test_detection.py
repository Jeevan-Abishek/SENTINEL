"""Tests for Phase 2: Person Detection with YOLO."""
import sys
from pathlib import Path
_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from unit.test_detection import *
from unit.test_detection_real import *
