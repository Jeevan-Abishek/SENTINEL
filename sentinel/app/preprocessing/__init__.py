"""
SENTINEL Preprocessing Package.
Provides video frame normalization, image quality auditing, and camera motion compensation.
"""

from app.preprocessing.frame_processor import FramePreprocessor, FrameProcessor, ProcessedFrame, FrameQualityChecker
from app.preprocessing.camera_motion import CameraMotionCompensator, CameraMotionResult

__all__ = [
    "FramePreprocessor",
    "FrameProcessor",
    "ProcessedFrame",
    "FrameQualityChecker",
    "CameraMotionCompensator",
    "CameraMotionResult",
]

