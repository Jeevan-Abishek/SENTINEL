"""
Pytest configuration for SENTINEL tests.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# Configure pytest-asyncio
pytest_asyncio_mode = "auto"


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    import numpy as np
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_gray():
    """Create a sample grayscale frame for testing."""
    import numpy as np
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)