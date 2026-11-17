"""
Unit tests for Video Ingestion & Validation (Phase 1)
"""

import pytest
from pathlib import Path
import numpy as np
import cv2

from app.video.validator import (
    validate_video_file,
    is_supported_video_format,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from app.video.adapter import FileVideoAdapter, VideoInfo


class TestVideoValidation:
    """Test suite for video validation and metadata extraction."""

    def test_supported_video_extensions(self):
        assert is_supported_video_format("drone_flight.mp4") is True
        assert is_supported_video_format("surveillance.AVI") is True
        assert is_supported_video_format("stream.mkv") is True
        assert is_supported_video_format("document.pdf") is False
        assert is_supported_video_format("script.py") is False
        assert is_supported_video_format("image.jpg") is False

    def test_missing_file(self):
        is_valid, err, info = validate_video_file("non_existent_flight.mp4")
        assert is_valid is False
        assert "does not exist" in err
        assert info is None

    def test_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty_stream.mp4"
        empty_file.write_bytes(b"")
        is_valid, err, info = validate_video_file(str(empty_file))
        assert is_valid is False
        assert "empty (0 bytes)" in err
        assert info is None

    def test_unsupported_format(self, tmp_path):
        txt_file = tmp_path / "stream_log.txt"
        txt_file.write_text("not a video")
        is_valid, err, info = validate_video_file(str(txt_file))
        assert is_valid is False
        assert "Unsupported video file extension" in err
        assert info is None

    def test_corrupted_video_file(self, tmp_path):
        corrupted = tmp_path / "corrupted.mp4"
        corrupted.write_bytes(b"garbage byte stream that is not valid mp4 container")
        is_valid, err, info = validate_video_file(str(corrupted))
        assert is_valid is False
        assert err is not None
        assert info is None

    def test_valid_video_metadata_and_frames(self, tmp_path):
        # Create a real temporary 10-frame video with OpenCV
        test_video_path = tmp_path / "test_valid.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(test_video_path), fourcc, 10.0, (320, 240))
        for i in range(10):
            frame = np.full((240, 320, 3), i * 20, dtype=np.uint8)
            out.write(frame)
        out.release()

        is_valid, err, info = validate_video_file(str(test_video_path))
        assert is_valid is True
        assert err is None
        assert isinstance(info, VideoInfo)
        assert info.width == 320
        assert info.height == 240
        assert info.fps == 10.0
        assert info.frame_count == 10
        assert info.duration == 1.0


class TestFileVideoAdapterIntegration:
    """Test reading and frame extraction using FileVideoAdapter with real video."""

    @pytest.mark.asyncio
    async def test_file_adapter_real_video(self, tmp_path):
        test_video_path = tmp_path / "adapter_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(test_video_path), fourcc, 20.0, (640, 480))
        for i in range(15):
            frame = np.full((480, 640, 3), 100, dtype=np.uint8)
            out.write(frame)
        out.release()

        adapter = FileVideoAdapter(str(test_video_path), frame_skip=1)
        connected = await adapter.connect()
        assert connected is True
        assert adapter.is_connected is True
        assert adapter.info.width == 640
        assert adapter.info.height == 480

        frame = await adapter.get_frame()
        assert frame is not None
        assert frame.width == 640
        assert frame.height == 480
        assert frame.frame.shape == (480, 640, 3)

        await adapter.disconnect()
        assert adapter.is_connected is False

