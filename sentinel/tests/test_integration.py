"""Tests for Phase 11: End-to-End Pipeline Integration."""
import sys
from pathlib import Path
_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from integration.test_end_to_end_pipeline import *
