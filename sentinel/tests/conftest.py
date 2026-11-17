"""
Pytest configuration for SENTINEL tests.
"""

import sys
from pathlib import Path

# Add project root and sentinel root to path
_sentinel = Path(__file__).resolve().parent.parent
_workspace = _sentinel.parent
for p in [str(_sentinel), str(_workspace)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import asyncio
import numpy as np


# Configure pytest-asyncio
asyncio_default_fixture_loop_scope = "function"
pytest_asyncio_mode = "auto"


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_gray():
    """Create a sample grayscale frame for testing."""
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)
