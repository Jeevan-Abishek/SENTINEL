"""SENTINEL Root Test Suite Configuration."""
import sys
import platform

# Prevent Python 3.14 WMI query crash (0xc0000008) in Windows sandbox environments
try:
    platform._wmi = None
    def _mock_wmi_query(*a, **k):
        raise OSError("not supported")
    platform._wmi_query = _mock_wmi_query
except Exception:
    pass

from pathlib import Path

_workspace = Path(__file__).resolve().parent.parent
_sentinel = _workspace / "sentinel"
for p in [str(_workspace), str(_sentinel)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import asyncio
import numpy as np

asyncio_default_fixture_loop_scope = "function"
pytest_asyncio_mode = "auto"

@pytest.fixture
def sample_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def sample_frame_gray():
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)

