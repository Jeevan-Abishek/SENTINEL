"""
SENTINEL Application Main Entrypoint.
Re-exports the FastAPI app from app.api.main for standard uvicorn invocations:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn
from app.config import settings
from app.api.main import app, manager, lifespan

__all__ = ["app", "manager", "lifespan"]

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )

