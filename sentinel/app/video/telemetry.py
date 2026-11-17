from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum
import json


class DroneStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    FLYING = "flying"
    HOVERING = "hovering"
    LANDED = "landed"


@dataclass
class GPSCoordinate:
    latitude: float
    longitude: float
    altitude: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"lat": self.latitude, "lon": self.longitude, "alt": self.altitude}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "GPSCoordinate":
        return cls(
            latitude=data.get("lat", 0.0),
            longitude=data.get("lon", 0.0),
            altitude=data.get("alt", 0.0),
        )


@dataclass
class GimbalOrientation:
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"pitch": self.pitch, "roll": self.roll, "yaw": self.yaw}


@dataclass
class DroneTelemetry:
    timestamp: datetime
    position: GPSCoordinate
    gimbal: GimbalOrientation
    velocity: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    battery_level: float = 100.0
    flight_mode: str = "unknown"
    status: DroneStatus = DroneStatus.CONNECTED
    raw_data: Dict[str, Any] = field(default_factory=dict)
    is_simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "position": self.position.to_dict(),
            "gimbal": self.gimbal.to_dict(),
            "velocity": self.velocity,
            "battery_level": self.battery_level,
            "flight_mode": self.flight_mode,
            "status": self.status.value,
            "raw_data": self.raw_data,
            "is_simulated": self.is_simulated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DroneTelemetry":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            position=GPSCoordinate.from_dict(data["position"]),
            gimbal=GimbalOrientation(**data.get("gimbal", {})),
            velocity=data.get("velocity", {"x": 0.0, "y": 0.0, "z": 0.0}),
            battery_level=data.get("battery_level", 100.0),
            flight_mode=data.get("flight_mode", "unknown"),
            status=DroneStatus(data.get("status", "connected")),
            raw_data=data.get("raw_data", {}),
            is_simulated=data.get("is_simulated", False),
        )


class TelemetryBuffer:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: list[DroneTelemetry] = []

    def add(self, telemetry: DroneTelemetry) -> None:
        self._buffer.append(telemetry)
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)

    def get_latest(self) -> Optional[DroneTelemetry]:
        return self._buffer[-1] if self._buffer else None

    def get_at_time(self, timestamp: datetime) -> Optional[DroneTelemetry]:
        if not self._buffer:
            return None
        return min(self._buffer, key=lambda t: abs((t.timestamp - timestamp).total_seconds()))

    def get_range(self, start: datetime, end: datetime) -> list[DroneTelemetry]:
        return [t for t in self._buffer if start <= t.timestamp <= end]

    def clear(self) -> None:
        self._buffer.clear()


class MockTelemetryProvider:
    def __init__(self, base_position: GPSCoordinate = None):
        self.base_position = base_position or GPSCoordinate(0.0, 0.0, 100.0)
        self.current_position = self.base_position
        self.current_gimbal = GimbalOrientation(-90.0, 0.0, 0.0)

    def generate_telemetry(self) -> DroneTelemetry:
        import random
        self.current_position = GPSCoordinate(
            latitude=self.base_position.latitude + random.uniform(-0.0001, 0.0001),
            longitude=self.base_position.longitude + random.uniform(-0.0001, 0.0001),
            altitude=self.base_position.altitude + random.uniform(-1.0, 1.0),
        )
        self.current_gimbal = GimbalOrientation(
            pitch=-90.0 + random.uniform(-5.0, 5.0),
            roll=random.uniform(-2.0, 2.0),
            yaw=random.uniform(0.0, 360.0),
        )

        return DroneTelemetry(
            timestamp=datetime.now(),
            position=self.current_position,
            gimbal=self.current_gimbal,
            velocity={"x": random.uniform(-2.0, 2.0), "y": random.uniform(-2.0, 2.0), "z": 0.0},
            battery_level=random.uniform(80.0, 100.0),
            flight_mode="GUIDED",
            status=DroneStatus.FLYING,
            is_simulated=True,
        )


class TelemetrySynchronizer:
    """Synchronizes video frame timestamps with drone telemetry data."""

    def __init__(
        self,
        telemetry_buffer: Optional[TelemetryBuffer] = None,
        mock_provider: Optional[MockTelemetryProvider] = None,
        max_time_skew_seconds: float = 0.5,
    ):
        self.buffer = telemetry_buffer or TelemetryBuffer()
        self.mock_provider = mock_provider
        self.max_time_skew_seconds = max_time_skew_seconds
        self.start_wall_time = datetime.now()

    def sync_frame(self, frame_timestamp_sec: float) -> DroneTelemetry:
        """Find the matching telemetry for a video frame timestamp (seconds from start).

        If real telemetry within max_time_skew_seconds exists, returns it.
        Otherwise falls back to simulated telemetry clearly marked with is_simulated=True.
        """
        target_time = self.start_wall_time + timedelta(seconds=frame_timestamp_sec)
        closest = self.buffer.get_at_time(target_time)

        if closest and abs((closest.timestamp - target_time).total_seconds()) <= self.max_time_skew_seconds:
            return closest

        # Fallback to simulated telemetry
        if self.mock_provider:
            sim = self.mock_provider.generate_telemetry()
            sim.timestamp = target_time
            sim.is_simulated = True
            return sim

        # Default synthetic telemetry fallback
        return DroneTelemetry(
            timestamp=target_time,
            position=GPSCoordinate(latitude=0.0, longitude=0.0, altitude=30.0),
            gimbal=GimbalOrientation(pitch=-90.0, roll=0.0, yaw=0.0),
            battery_level=100.0,
            flight_mode="SIMULATED",
            status=DroneStatus.FLYING,
            is_simulated=True,
        )