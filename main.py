"""
SENTINEL Workspace Root Entrypoint.
Enables running from the workspace root:
    python main.py
    uvicorn main:app --reload
"""

import sys
from pathlib import Path

_sentinel_dir = Path(__file__).resolve().parent / "sentinel"
if str(_sentinel_dir) not in sys.path:
    sys.path.insert(0, str(_sentinel_dir))

import uvicorn
from config import settings
from app.api.main import app, manager, lifespan

__all__ = ["app", "manager", "lifespan"]

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )

