"""
SENTINEL Root Entrypoint.
Enables running directly from the sentinel directory:
    python main.py
    uvicorn main:app --reload
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from app.config import settings
from app.api.main import app, manager, lifespan

__all__ = ["app", "manager", "lifespan"]

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )

