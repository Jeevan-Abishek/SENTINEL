"""
API Routes Module.
Unifies all REST subrouters for events, zones, drones, alerts, analytics, and video.
"""

from app.api.routes import router as api_router

__all__ = ["api_router"]

