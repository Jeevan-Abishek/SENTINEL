"""
RTSP Video Source Module.
Handles live network RTSP video streaming from drone receivers.
"""

from typing import Optional
from loguru import logger

from app.video.adapter import RTSPVideoAdapter, VideoFrame, VideoInfo

# Alias matching master architecture
RTSPVideoSource = RTSPVideoAdapter

__all__ = ["RTSPVideoAdapter", "RTSPVideoSource"]

