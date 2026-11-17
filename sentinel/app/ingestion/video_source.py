"""
Video Source Base Module.
Defines the standard interface for all SENTINEL video ingestion adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from loguru import logger

from app.video.adapter import VideoFrame, VideoInfo, VideoAdapter

# Aliases matching master architecture naming
VideoSource = VideoAdapter

__all__ = ["VideoFrame", "VideoInfo", "VideoSource", "VideoAdapter"]

