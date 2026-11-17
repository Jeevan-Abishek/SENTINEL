"""
File Video Source Module.
Handles video ingestion from local recorded drone files.
"""

from typing import Optional
import cv2
from loguru import logger

from app.video.adapter import FileVideoAdapter, VideoFrame, VideoInfo

# Alias matching master architecture
FileVideoSource = FileVideoAdapter

__all__ = ["FileVideoAdapter", "FileVideoSource"]

