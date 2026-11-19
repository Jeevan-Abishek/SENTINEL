"""
SENTINEL Dashboard API Client Service.
Provides high-performance, non-blocking HTTP connectivity between the Streamlit dashboard
and the FastAPI backend with fast circuit-breakers and seamless standalone fallback.
"""

import time
import requests
from typing import Dict, Any, List, Optional
from loguru import logger

from config import settings


class SentinelAPIClient:
    """Client for communicating with the SENTINEL FastAPI Backend."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 0.35):
        self.base_url = base_url or f"http://localhost:{settings.api_port}"
        self.timeout = timeout
        self._last_probe_time: float = 0.0
        self._is_online_cached: bool = False
        self._probe_ttl: float = 15.0

    def is_online(self) -> bool:
        """Probes the FastAPI /health endpoint with 15s cached result to prevent lag."""
        now = time.perf_counter()
        if now - self._last_probe_time < self._probe_ttl:
            return self._is_online_cached

        self._last_probe_time = now
        try:
            r = requests.get(f"{self.base_url}/health", timeout=0.25)
            self._is_online_cached = (r.status_code == 200)
        except Exception:
            self._is_online_cached = False

        return self._is_online_cached

    def get_health(self) -> Optional[Dict[str, Any]]:
        if not self.is_online():
            return None
        try:
            r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        if not self.is_online():
            return None
        try:
            r = requests.get(f"{self.base_url}/api/v1/status", timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def get_zones(self) -> List[Dict[str, Any]]:
        if not self.is_online():
            return []
        try:
            r = requests.get(f"{self.base_url}/api/v1/zones", timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else data.get("zones", [])
            return []
        except Exception:
            return []

    def get_analytics_summary(self) -> Dict[str, Any]:
        if not self.is_online():
            return {}
        try:
            r = requests.get(f"{self.base_url}/api/v1/analytics/summary", timeout=self.timeout)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def get_flow_metrics(self) -> Dict[str, Any]:
        if not self.is_online():
            return {}
        try:
            r = requests.get(f"{self.base_url}/api/v1/analytics/flow", timeout=self.timeout)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def get_bottlenecks(self) -> Dict[str, Any]:
        if not self.is_online():
            return {}
        try:
            r = requests.get(f"{self.base_url}/api/v1/analytics/bottlenecks", timeout=self.timeout)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def get_risk_current(self) -> Dict[str, Any]:
        if not self.is_online():
            return {}
        try:
            r = requests.get(f"{self.base_url}/api/v1/risk/current", timeout=self.timeout)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.is_online():
            return []
        try:
            r = requests.get(f"{self.base_url}/api/v1/alerts", params={"limit": limit}, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else data.get("alerts", [])
            return []
        except Exception:
            return []

    def acknowledge_alert(self, alert_id: str, operator_id: str = "operator_chief") -> bool:
        if not self.is_online():
            return False
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/alerts/{alert_id}/acknowledge",
                json={"operator_id": operator_id},
                timeout=1.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def get_drones(self) -> List[Dict[str, Any]]:
        if not self.is_online():
            return []
        try:
            r = requests.get(f"{self.base_url}/api/v1/drones", timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else data.get("drones", [])
            return []
        except Exception:
            return []

    def get_live_drone_telemetry(self) -> Dict[str, Any]:
        if not self.is_online():
            return {}
        try:
            r = requests.get(f"{self.base_url}/api/v1/drones/telemetry/live", timeout=self.timeout)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def connect_drone(self, rtsp_url: str, telemetry_port: int = 14550) -> Dict[str, Any]:
        if not self.is_online():
            return {"status": "offline_mode", "connected": False}
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/drones/connect",
                json={"rtsp_url": rtsp_url, "telemetry_port": telemetry_port},
                timeout=1.5,
            )
            return r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def disconnect_drone(self, drone_id: str = "drone_alpha") -> Dict[str, Any]:
        if not self.is_online():
            return {"status": "offline_mode"}
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/drones/disconnect",
                json={"drone_id": drone_id},
                timeout=1.0,
            )
            return r.json() if r.status_code == 200 else {"status": "error"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}


api_client = SentinelAPIClient()

__all__ = ["SentinelAPIClient", "api_client"]

