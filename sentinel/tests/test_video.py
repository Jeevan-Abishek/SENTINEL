"""Tests for Phase 1 & 5: Video Ingestion, Validation, and Preprocessing."""
import sys
from pathlib import Path
_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from unit.test_video_validation import *
from unit.test_video_processing import *
