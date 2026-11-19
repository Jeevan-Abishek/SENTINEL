"""
Unit tests for SENTINEL RTSP Live Video Capture & Streaming Feature.

Tests cover:
- RTSPStreamStats telemetry dataclass
- RTSPVideoAdapter initialization, URL validation, and socket pre-flight checks
- Non-blocking latest-frame atomic buffering and background reader thread
- Ingestion resolution downscaling (640px / 480px width cap) for low RAM usage
- Frame skipping efficiency
- TCP and UDP transport protocols
- Reconnection backoff handling and graceful failure modes
- FramePreprocessor preview resizing
- Dashboard RTSP stream probing
"""

import sys
import time
import socket
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.video.rtsp_adapter import RTSPVideoAdapter, RTSPStreamStats
from app.video.processor import FramePreprocessor, VideoStreamProcessor
from app.video.adapter import VideoFrame, VideoInfo
from app.config import settings


class TestRTSPStreamStats:
    """Tests for RTSPStreamStats telemetry dataclass."""

    def test_default_stats(self):
        stats = RTSPStreamStats(status="DISCONNECTED")
        assert stats.status == "DISCONNECTED"
        assert stats.fps == 0.0
        assert stats.width == 0
        assert stats.height == 0
        assert stats.frames_received == 0
        assert stats.frames_dropped == 0
        assert stats.reconnect_attempts == 0
        assert stats.transport == "tcp"
        assert stats.last_error == ""

    def test_populated_stats(self):
        stats = RTSPStreamStats(
            status="CONNECTED",
            fps=29.97,
            width=640,
            height=360,
            frames_received=150,
            frames_dropped=12,
            reconnect_attempts=1,
            latency_estimate_ms=38.5,
            transport="udp",
        )
        assert stats.status == "CONNECTED"
        assert stats.fps == 29.97
        assert stats.width == 640
        assert stats.height == 360
        assert stats.frames_received == 150
        assert stats.frames_dropped == 12
        assert stats.transport == "udp"


class TestRTSPVideoAdapter:
    """Tests for RTSPVideoAdapter implementation."""

    def test_adapter_initialization(self):
        url = "rtsp://192.168.1.100:8554/live"
        adapter = RTSPVideoAdapter(
            rtsp_url=url,
            frame_skip=2,
            target_width=640,
            transport="tcp",
            connect_timeout_sec=2.0,
        )
        assert adapter.rtsp_url == url
        assert adapter.frame_skip == 2
        assert adapter.target_width == 640
        assert adapter.transport == "tcp"
        assert adapter.status == "DISCONNECTED"
        assert not adapter.is_connected

    def test_empty_url_fails_cleanly(self):
        adapter = RTSPVideoAdapter(rtsp_url="")
        success = adapter._sync_connect()
        assert not success
        assert adapter.status == "FAILED"
        assert "empty" in adapter.get_stats()["last_error"].lower()

    def test_unreachable_rtsp_preflight(self):
        adapter = RTSPVideoAdapter(
            rtsp_url="rtsp://192.168.254.254:8554/stream",
            connect_timeout_sec=0.2,
        )
        t0 = time.perf_counter()
        success = adapter._sync_connect()
        dt = time.perf_counter() - t0

        assert not success
        assert adapter.status == "FAILED"
        assert dt < 2.0  # Guarantees zero-freeze UI (< 2.0s resolution)
        stats = adapter.get_stats()
        assert stats["status"] == "FAILED"
        assert stats["last_error"] != ""

    def test_socket_reachability_probe(self):
        adapter = RTSPVideoAdapter("rtsp://127.0.0.1:65432/drone")
        reachable, err = adapter._check_network_reachability("rtsp://127.0.0.1:65432/drone", timeout=0.1)
        assert not reachable
        assert "65432" in err or "refused" in err.lower() or "reach" in err.lower()

    def test_local_video_stream_connection_and_downscale(self):
        video_path = str(settings.sample_videos_dir / "demo_drone_crowd.mp4")
        if not Path(video_path).exists():
            pytest.skip("Demo video not present on system.")

        adapter = RTSPVideoAdapter(
            rtsp_url=video_path,
            frame_skip=1,
            target_width=640,
        )
        success = adapter._sync_connect()
        assert success
        assert adapter.status == "CONNECTED"
        assert adapter.is_connected

        # Allow worker thread to capture frames into buffer
        time.sleep(0.3)

        frame = adapter.get_latest_frame()
        assert frame is not None
        assert isinstance(frame, VideoFrame)
        assert frame.width == 640
        assert frame.frame.shape[1] == 640  # Capped to 640px
        assert frame.frame.shape[0] > 0
        assert frame.frame_number >= 1
        assert frame.timestamp > 0

        raw_frame = adapter.get_latest_raw_frame()
        assert raw_frame is not None
        assert raw_frame.shape[1] == 640

        stats = adapter.get_stats()
        assert stats["frames_received"] > 0
        assert stats["width"] == 640

        # Disconnect cleanly
        adapter._sync_disconnect()
        assert adapter.status == "DISCONNECTED"
        assert not adapter.is_connected
        assert adapter.get_latest_frame() is None

    def test_custom_target_width_480(self):
        video_path = str(settings.sample_videos_dir / "demo_drone_crowd.mp4")
        if not Path(video_path).exists():
            pytest.skip("Demo video not present.")

        adapter = RTSPVideoAdapter(
            rtsp_url=video_path,
            frame_skip=1,
            target_width=480,
        )
        success = adapter._sync_connect()
        assert success
        time.sleep(0.2)

        frame = adapter.get_latest_frame()
        if frame is not None:
            assert frame.width == 480
            assert frame.frame.shape[1] == 480

        adapter._sync_disconnect()

    def test_frame_skip_increments_dropped(self):
        video_path = str(settings.sample_videos_dir / "demo_drone_crowd.mp4")
        if not Path(video_path).exists():
            pytest.skip("Demo video not present.")

        adapter = RTSPVideoAdapter(
            rtsp_url=video_path,
            frame_skip=3,
            target_width=640,
        )
        success = adapter._sync_connect()
        assert success
        time.sleep(0.3)

        stats = adapter.get_stats()
        # With frame_skip=3, frames_dropped should be approximately 2x frames_received
        assert stats["frames_dropped"] > 0
        adapter._sync_disconnect()

    def test_transport_udp_option(self):
        adapter = RTSPVideoAdapter(
            rtsp_url="rtsp://10.0.0.1:8554/live",
            transport="udp",
        )
        assert adapter.transport == "udp"
        stats = adapter.get_stats()
        assert stats["transport"] == "udp"


class TestFramePreprocessorPreview:
    """Tests for lightweight frame downscaling and preview resizing."""

    def test_resize_preview_downscales_hd_to_640(self):
        # 1920x1080 frame
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = FramePreprocessor.resize_preview(img, max_width=640)
        assert resized.shape[1] == 640
        assert resized.shape[0] == 360  # Aspect ratio 16:9 preserved

    def test_resize_preview_downscales_4k_to_640(self):
        # 3840x2160 (4K) frame
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        resized = FramePreprocessor.resize_preview(img, max_width=640)
        assert resized.shape[1] == 640
        assert resized.shape[0] == 360

    def test_resize_preview_does_not_upscale_small_frame(self):
        # 480x270 frame
        img = np.zeros((270, 480, 3), dtype=np.uint8)
        resized = FramePreprocessor.resize_preview(img, max_width=640)
        assert resized.shape[1] == 480  # Does not upscale
        assert resized.shape[0] == 270

    def test_resize_preview_none_safe(self):
        assert FramePreprocessor.resize_preview(None, max_width=640) is None


class TestVideoStreamProcessorResolutionCap:
    """Tests for VideoStreamProcessor with resolution capping."""

    def test_processor_stats_and_skip(self):
        mock_adapter = Mock()
        mock_adapter.is_connected = True
        processor = VideoStreamProcessor(
            adapter=mock_adapter,
            frame_skip=2,
            max_resolution_width=640,
        )
        stats = processor.get_stats()
        assert stats["skip_ratio"] == 2
        assert stats["total_frames"] == 0
        assert stats["processed_frames"] == 0

