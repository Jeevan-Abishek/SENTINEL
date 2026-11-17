"""Generic Drone Video & Telemetry Stream Adapter.

Provides a unified interface bridging video streams (RTSP, recorded drone MP4, or test stream),
telemetry data ingestion (with automatic fallback to simulated telemetry), and camera motion
compensation for drone aerial surveillance.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
from loguru import logger

from app.video.adapter import VideoAdapter, VideoFrame, VideoInfo, create_video_adapter
from app.video.telemetry import (
    DroneTelemetry,
    DroneStatus,
    TelemetryBuffer,
    MockTelemetryProvider,
    TelemetrySynchronizer,
)
from app.video.camera_motion import CameraMotionCompensator, CameraMotionResult


@dataclass
class DroneFrameBundle:
    """Synchronized package of video frame, drone telemetry, and camera motion."""
    frame: VideoFrame
    telemetry: DroneTelemetry
    camera_motion: CameraMotionResult
    is_telemetry_simulated: bool


class DroneStreamAdapter:
    """High-level drone interface combining video input, telemetry sync, and camera motion."""

    def __init__(
        self,
        video_source: str,
        frame_skip: int = 1,
        enable_motion_compensation: bool = True,
        telemetry_buffer: Optional[TelemetryBuffer] = None,
        mock_telemetry: bool = True,
    ):
        self.video_source = video_source
        self.frame_skip = frame_skip
        self.enable_motion_compensation = enable_motion_compensation
        
        # Initialize video adapter
        self.video_adapter: VideoAdapter = create_video_adapter(video_source, frame_skip=frame_skip)
        
        # Telemetry handling
        self.telemetry_buffer = telemetry_buffer or TelemetryBuffer()
        self.mock_provider = MockTelemetryProvider() if mock_telemetry else None
        self.telemetry_sync = TelemetrySynchronizer(
            telemetry_buffer=self.telemetry_buffer,
            mock_provider=self.mock_provider,
        )
        
        # Motion compensator
        self.motion_compensator = CameraMotionCompensator() if enable_motion_compensation else None
        
        # Status
        self._connected = False
        self._frames_served = 0
        self._reconnect_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected and self.video_adapter.is_connected

    async def connect(self) -> bool:
        """Connect to video source and telemetry interface."""
        logger.info(f"Connecting DroneStreamAdapter to {self.video_source}")
        success = await self.video_adapter.connect()
        if success:
            self._connected = True
            if self.motion_compensator:
                self.motion_compensator.reset()
            logger.info("DroneStreamAdapter connected successfully")
        else:
            self._connected = False
            logger.warning("DroneStreamAdapter failed to connect to video source")
        return success

    async def disconnect(self) -> None:
        """Disconnect and release resources."""
        await self.video_adapter.disconnect()
        self._connected = False
        if self.motion_compensator:
            self.motion_compensator.reset()
        logger.info("DroneStreamAdapter disconnected")

    async def get_next_frame(self) -> Optional[DroneFrameBundle]:
        """Fetch the next video frame synchronized with telemetry and camera motion."""
        if not self.is_connected:
            return None

        video_frame = await self.video_adapter.get_frame()
        if video_frame is None:
            return None

        self._frames_served += 1

        # Synchronize telemetry
        telemetry = self.telemetry_sync.sync_frame(video_frame.timestamp)

        # Estimate camera motion
        if self.motion_compensator:
            motion = self.motion_compensator.estimate_motion(video_frame.frame)
        else:
            motion = CameraMotionResult(dx=0.0, dy=0.0, is_valid=True)

        return DroneFrameBundle(
            frame=video_frame,
            telemetry=telemetry,
            camera_motion=motion,
            is_telemetry_simulated=telemetry.is_simulated,
        )

    async def get_video_info(self) -> Optional[VideoInfo]:
        """Return metadata about the underlying video stream."""
        return await self.video_adapter.get_info()

    def feed_external_telemetry(self, telemetry: DroneTelemetry) -> None:
        """Feed real telemetry received from a drone SDK / telemetry link into the buffer."""
        telemetry.is_simulated = False
        self.telemetry_buffer.add(telemetry)

    def get_status(self) -> Dict[str, Any]:
        """Return operational status of the drone stream."""
        latest_tel = self.telemetry_buffer.get_latest()
        return {
            "source": self.video_source,
            "connected": self.is_connected,
            "frames_served": self._frames_served,
            "motion_compensation_enabled": self.enable_motion_compensation,
            "has_real_telemetry": latest_tel is not None and not latest_tel.is_simulated,
            "latest_altitude_m": latest_tel.position.altitude if latest_tel else None,
            "latest_battery_pct": latest_tel.battery_level if latest_tel else None,
        }

