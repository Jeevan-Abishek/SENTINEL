"""
Stream Manager Module.
Coordinates multiple active video and drone streams, handling reconnection and metrics.
"""

from typing import Dict, Optional, Any, List
import asyncio
from loguru import logger

from app.ingestion.video_source import VideoSource, VideoFrame, VideoInfo
from app.ingestion.file_source import FileVideoSource
from app.ingestion.rtsp_source import RTSPVideoSource
from app.ingestion.drone_adapter import DroneStreamAdapter, DroneFrameBundle
from app.video.adapter import create_video_adapter, MockVideoAdapter


class StreamManager:
    """Manages active video sources, health tracking, and frame dispatching."""

    def __init__(self):
        self._streams: Dict[str, VideoSource] = {}
        self._drone_adapters: Dict[str, DroneStreamAdapter] = {}
        self._active_status: Dict[str, Dict[str, Any]] = {}

    def register_stream(self, stream_id: str, adapter: VideoSource) -> None:
        """Register an initialized video adapter under a stream ID."""
        self._streams[stream_id] = adapter
        self._active_status[stream_id] = {
            "stream_id": stream_id,
            "connected": adapter.is_connected,
            "frames_read": 0,
            "errors": 0,
        }
        logger.info(f"Registered stream: {stream_id}")

    def register_drone_stream(self, drone_id: str, adapter: DroneStreamAdapter) -> None:
        """Register a high-level drone stream adapter."""
        self._drone_adapters[drone_id] = adapter
        logger.info(f"Registered drone stream: {drone_id}")

    async def connect_stream(self, stream_id: str) -> bool:
        """Connect to the specified stream."""
        if stream_id not in self._streams:
            logger.warning(f"Stream {stream_id} not registered")
            return False
        adapter = self._streams[stream_id]
        success = await adapter.connect()
        if stream_id in self._active_status:
            self._active_status[stream_id]["connected"] = success
        return success

    async def disconnect_stream(self, stream_id: str) -> None:
        """Disconnect and release a stream."""
        if stream_id in self._streams:
            await self._streams[stream_id].disconnect()
            if stream_id in self._active_status:
                self._active_status[stream_id]["connected"] = False
            logger.info(f"Disconnected stream: {stream_id}")

    async def get_next_frame(self, stream_id: str) -> Optional[VideoFrame]:
        """Fetch the next available frame from a video stream."""
        if stream_id not in self._streams:
            return None
        adapter = self._streams[stream_id]
        frame = await adapter.get_frame()
        if frame and stream_id in self._active_status:
            self._active_status[stream_id]["frames_read"] += 1
        elif frame is None and stream_id in self._active_status:
            self._active_status[stream_id]["errors"] += 1
        return frame

    def get_stream(self, stream_id: str) -> Optional[VideoSource]:
        return self._streams.get(stream_id)

    def get_drone_stream(self, drone_id: str) -> Optional[DroneStreamAdapter]:
        return self._drone_adapters.get(drone_id)

    def list_streams(self) -> List[Dict[str, Any]]:
        return list(self._active_status.values())

    async def disconnect_all(self) -> None:
        for sid in list(self._streams.keys()):
            await self.disconnect_stream(sid)
        for did, dadapter in list(self._drone_adapters.items()):
            await dadapter.disconnect()
        self._streams.clear()
        self._drone_adapters.clear()
        self._active_status.clear()


stream_manager = StreamManager()

__all__ = ["StreamManager", "stream_manager"]

