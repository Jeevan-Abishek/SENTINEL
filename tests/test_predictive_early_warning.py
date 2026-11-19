"""Root test forwarder for 5-minute predictive early warnings."""
import sys
from pathlib import Path

_workspace = Path(__file__).resolve().parent.parent
_sentinel = _workspace / "sentinel"
for p in [str(_workspace), str(_sentinel)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sentinel.tests.unit.test_predictive_early_warning import *

