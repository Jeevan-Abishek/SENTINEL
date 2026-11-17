"""
SENTINEL Video Ingestion Package.
Provides video input sources, RTSP, generic drone stream adapters, and stream coordination.
"""

from app.ingestion.video_source import VideoSource, VideoAdapter, VideoFrame, VideoInfo
from app.ingestion.file_source import FileVideoSource, FileVideoAdapter
from app.ingestion.rtsp_source import RTSPVideoSource, RTSPVideoAdapter
from app.ingestion.drone_adapter import DroneStreamAdapter, DroneFrameBundle
from app.ingestion.stream_manager import StreamManager, stream_manager

__all__ = [
    "VideoSource",
    "VideoAdapter",
    "VideoFrame",
    "VideoInfo",
    "FileVideoSource",
    "FileVideoAdapter",
    "RTSPVideoSource",
    "RTSPVideoAdapter",
    "DroneStreamAdapter",
    "DroneFrameBundle",
    "StreamManager",
    "stream_manager",
]

