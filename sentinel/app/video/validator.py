"""
SENTINEL Video Validation and Ingestion Utilities
Provides robust validation, metadata extraction, and error handling for video sources.
"""

from pathlib import Path
from typing import Optional, Tuple, Set
import cv2
from loguru import logger
from app.video.adapter import VideoInfo

SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB limit for safety


def is_supported_video_format(filename: str) -> bool:
    """Check if file extension corresponds to a supported video format."""
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_VIDEO_EXTENSIONS


def validate_video_file(file_path: str) -> Tuple[bool, Optional[str], Optional[VideoInfo]]:
    """
    Validates a video file for existence, size, header integrity, and OpenCV decode capability.
    
    Returns:
        (is_valid, error_message, video_info)
    """
    path = Path(file_path)
    
    if not path.exists():
        msg = f"Video file does not exist: {file_path}"
        logger.warning(msg)
        return False, msg, None

    if not path.is_file():
        msg = f"Path is not a regular file: {file_path}"
        logger.warning(msg)
        return False, msg, None

    file_size = path.stat().st_size
    if file_size == 0:
        msg = f"Video file is empty (0 bytes): {file_path}"
        logger.warning(msg)
        return False, msg, None

    if file_size > MAX_FILE_SIZE_BYTES:
        msg = f"Video file exceeds maximum allowed size of 2GB: {file_size} bytes"
        logger.warning(msg)
        return False, msg, None

    if not is_supported_video_format(path.name):
        msg = f"Unsupported video file extension: {path.suffix}. Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        logger.warning(msg)
        return False, msg, None

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            msg = f"Failed to open video file (corrupted or unsupported codec): {file_path}"
            logger.warning(msg)
            return False, msg, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((codec_int >> (8 * i)) & 0xFF) for i in range(4)]).strip()

        if width <= 0 or height <= 0:
            msg = f"Corrupted video header: invalid dimensions {width}x{height}"
            logger.warning(msg)
            return False, msg, None

        # Attempt to read at least one frame to confirm decodability
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            msg = f"Unable to decode initial frame from video: {file_path}"
            logger.warning(msg)
            return False, msg, None

        effective_fps = fps if fps > 0 else 30.0
        effective_count = frame_count if frame_count > 0 else 1
        duration = effective_count / effective_fps

        info = VideoInfo(
            width=width,
            height=height,
            fps=effective_fps,
            frame_count=effective_count,
            duration=duration,
            codec=codec if codec else "UNKNOWN",
        )
        return True, None, info

    finally:
        cap.release()

