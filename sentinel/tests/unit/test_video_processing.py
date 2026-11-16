"""
Unit tests for SENTINEL Phase 1 - Video Processing Module

Tests cover:
- Video adapter creation and connection
- Frame extraction from mock adapter
- Frame preprocessing (resize, normalize, letterbox)
- Frame quality checking
- Video stream processing pipeline
- Error handling for invalid videos
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.video.adapter import (
    VideoAdapter,
    VideoFrame,
    VideoInfo,
    FileVideoAdapter,
    RTSPVideoAdapter,
    MockVideoAdapter,
    create_video_adapter,
)
from app.video.processor import (
    FramePreprocessor,
    FrameQualityChecker,
    VideoStreamProcessor,
    ProcessedFrame,
)


class TestVideoFrame:
    """Tests for VideoFrame dataclass."""
    
    def test_video_frame_creation(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vf = VideoFrame(frame=frame, frame_number=1, timestamp=0.033, width=640, height=480)
        
        assert vf.frame_number == 1
        assert vf.timestamp == 0.033
        assert vf.width == 640
        assert vf.height == 480
        assert vf.frame.shape == (480, 640, 3)


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""
    
    def test_video_info_creation(self):
        info = VideoInfo(width=1920, height=1080, fps=30.0, frame_count=900, duration=30.0, codec="H264")
        
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30.0
        assert info.frame_count == 900
        assert info.duration == 30.0
        assert info.codec == "H264"


class TestMockVideoAdapter:
    """Tests for MockVideoAdapter."""
    
    @pytest.mark.asyncio
    async def test_mock_adapter_connect(self):
        adapter = MockVideoAdapter(width=640, height=480, fps=30.0, frame_count=100)
        result = await adapter.connect()
        
        assert result is True
        assert adapter.is_connected is True
        assert adapter.info is not None
        assert adapter.info.width == 640
        assert adapter.info.height == 480
        assert adapter.info.fps == 30.0
        assert adapter.info.frame_count == 100
    
    @pytest.mark.asyncio
    async def test_mock_adapter_get_frame(self):
        adapter = MockVideoAdapter(width=640, height=480, fps=30.0, frame_count=10)
        await adapter.connect()
        
        frame = await adapter.get_frame()
        
        assert frame is not None
        assert isinstance(frame, VideoFrame)
        assert frame.frame_number == 1
        assert frame.width == 640
        assert frame.height == 480
        assert frame.frame.shape == (480, 640, 3)
        assert frame.frame.dtype == np.uint8
    
    @pytest.mark.asyncio
    async def test_mock_adapter_multiple_frames(self):
        adapter = MockVideoAdapter(width=640, height=480, fps=30.0, frame_count=5)
        await adapter.connect()
        
        frames = []
        for _ in range(5):
            frame = await adapter.get_frame()
            if frame:
                frames.append(frame)
        
        assert len(frames) == 5
        assert frames[0].frame_number == 1
        assert frames[4].frame_number == 5
    
    @pytest.mark.asyncio
    async def test_mock_adapter_exhaustion(self):
        adapter = MockVideoAdapter(width=640, height=480, fps=30.0, frame_count=2)
        await adapter.connect()
        
        frame1 = await adapter.get_frame()
        frame2 = await adapter.get_frame()
        frame3 = await adapter.get_frame()
        
        assert frame1 is not None
        assert frame2 is not None
        assert frame3 is None  # Exhausted
    
    @pytest.mark.asyncio
    async def test_mock_adapter_disconnect(self):
        adapter = MockVideoAdapter()
        await adapter.connect()
        await adapter.disconnect()
        
        assert adapter.is_connected is False


class TestFileVideoAdapter:
    """Tests for FileVideoAdapter."""
    
    @pytest.mark.asyncio
    async def test_file_adapter_invalid_path(self):
        adapter = FileVideoAdapter("/nonexistent/path/video.mp4")
        result = await adapter.connect()
        
        assert result is False
        assert adapter.is_connected is False
    
    @pytest.mark.asyncio
    async def test_file_adapter_invalid_file(self, tmp_path):
        # Create a non-video file
        invalid_file = tmp_path / "not_a_video.txt"
        invalid_file.write_text("not a video")
        
        adapter = FileVideoAdapter(str(invalid_file))
        result = await adapter.connect()
        
        assert result is False
        assert adapter.is_connected is False


class TestCreateVideoAdapter:
    """Tests for create_video_adapter factory function."""
    
    def test_create_file_adapter(self):
        adapter = create_video_adapter("/path/to/video.mp4")
        assert isinstance(adapter, FileVideoAdapter)
    
    def test_create_rtsp_adapter(self):
        adapter = create_video_adapter("rtsp://example.com/stream")
        assert isinstance(adapter, RTSPVideoAdapter)
    
    def test_create_rtsps_adapter(self):
        adapter = create_video_adapter("rtsps://example.com/stream")
        assert isinstance(adapter, RTSPVideoAdapter)
    
    def test_create_mock_adapter(self):
        adapter = create_video_adapter("mock")
        assert isinstance(adapter, MockVideoAdapter)


class TestFramePreprocessor:
    """Tests for FramePreprocessor."""
    
    def test_preprocess_letterbox(self):
        preprocessor = FramePreprocessor(target_width=640, target_height=640, keep_aspect_ratio=True)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(frame)
        
        assert isinstance(result, ProcessedFrame)
        assert result.processed.shape == (1, 3, 640, 640)  # NCHW format
        assert result.scale_x > 0
        assert result.scale_y > 0
        assert result.original.shape == (480, 640, 3)
    
    def test_preprocess_no_letterbox(self):
        preprocessor = FramePreprocessor(target_width=640, target_height=480, keep_aspect_ratio=False)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(frame)
        
        assert result.processed.shape == (1, 3, 480, 640)
        assert result.pad_left == 0
        assert result.pad_top == 0
    
    def test_preprocess_different_aspect_ratios(self):
        preprocessor = FramePreprocessor(target_width=640, target_height=640, keep_aspect_ratio=True)
        
        # Wide frame
        wide_frame = np.random.randint(0, 255, (480, 800, 3), dtype=np.uint8)
        result = preprocessor.preprocess(wide_frame)
        assert result.processed.shape == (1, 3, 640, 640)
        
        # Tall frame
        tall_frame = np.random.randint(0, 255, (800, 480, 3), dtype=np.uint8)
        result = preprocessor.preprocess(tall_frame)
        assert result.processed.shape == (1, 3, 640, 640)
    
    def test_denormalize_boxes(self):
        preprocessor = FramePreprocessor(target_width=640, target_height=640, keep_aspect_ratio=True)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = preprocessor.preprocess(frame)
        
        # Create test boxes in normalized coordinates
        boxes = np.array([
            [100, 100, 200, 200],  # x1, y1, x2, y2
            [300, 150, 400, 250],
        ], dtype=np.float32)
        
        denormalized = preprocessor.denormalize_boxes(boxes, processed)
        
        assert denormalized.shape == boxes.shape
        # Boxes should be within original frame bounds
        assert np.all(denormalized[:, 0] >= 0)
        assert np.all(denormalized[:, 1] >= 0)
        assert np.all(denormalized[:, 2] <= frame.shape[1])
        assert np.all(denormalized[:, 3] <= frame.shape[0])


class TestFrameQualityChecker:
    """Tests for FrameQualityChecker."""
    
    def test_quality_checker_normal_frame(self):
        checker = FrameQualityChecker()
        # Create a normal frame with good contrast
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        result = checker.check_quality(frame)
        
        assert "brightness" in result
        assert "contrast" in result
        assert "blur_score" in result
        assert "quality_score" in result
        assert "passed" in result
        assert 0 <= result["quality_score"] <= 1
    
    def test_quality_checker_dark_frame(self):
        checker = FrameQualityChecker(min_brightness=50)
        # Create a very dark frame
        frame = np.full((480, 640, 3), 10, dtype=np.uint8)
        
        result = checker.check_quality(frame)
        
        assert result["is_dark"] == True
        assert result["quality_score"] < 1.0
    
    def test_quality_checker_bright_frame(self):
        checker = FrameQualityChecker(max_brightness=200)
        # Create a very bright frame
        frame = np.full((480, 640, 3), 250, dtype=np.uint8)
        
        result = checker.check_quality(frame)
        
        assert result["is_bright"] == True
        assert result["quality_score"] < 1.0
    
    def test_quality_checker_low_contrast(self):
        checker = FrameQualityChecker(min_contrast=20)
        # Create a low contrast frame
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        
        result = checker.check_quality(frame)
        
        assert result["is_low_contrast"] == True
        assert result["quality_score"] < 1.0
    
    def test_quality_checker_blurry_frame(self):
        checker = FrameQualityChecker(blur_threshold=500)
        # Create a blurry frame (uniform color)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        
        result = checker.check_quality(frame)
        
        assert result["is_blurry"] == True
        assert result["blur_score"] < 500


class TestVideoStreamProcessor:
    """Tests for VideoStreamProcessor."""
    
    @pytest.mark.asyncio
    async def test_process_stream_mock(self):
        adapter = MockVideoAdapter(width=640, height=480, fps=30.0, frame_count=10)
        await adapter.connect()
        
        processor = VideoStreamProcessor(adapter, frame_skip=1, enable_quality_check=True)
        
        frames_processed = 0
        async for frame_data in processor.process_stream():
            assert "frame" in frame_data
            assert "processed" in frame_data
            assert "quality" in frame_data
            assert isinstance(frame_data["frame"], VideoFrame)
            assert isinstance(frame_data["processed"], ProcessedFrame)
            assert frame_data["quality"] is not None
            frames_processed += 1
            if frames_processed >= 5:
                break
        
        assert frames_processed == 5
        stats = processor.get_stats()
        assert stats["processed_frames"] == 5
    
    @pytest.mark.asyncio
    async def test_process_stream_with_frame_skip(self):
        adapter = MockVideoAdapter(width=640, height=480, fps=30.0, frame_count=20)
        await adapter.connect()
        
        processor = VideoStreamProcessor(adapter, frame_skip=2, enable_quality_check=False)
        
        frames_processed = 0
        async for frame_data in processor.process_stream():
            frames_processed += 1
            if frames_processed >= 3:
                break
        
        # With frame_skip=2, we process every 2nd frame
        # But the counter increments for every frame read
        stats = processor.get_stats()
        assert stats["skip_ratio"] == 2


class TestVideoAdapterErrorHandling:
    """Tests for error handling in video adapters."""
    
    @pytest.mark.asyncio
    async def test_file_adapter_handles_corrupted_video(self, tmp_path):
        # Create a corrupted video file
        corrupted = tmp_path / "corrupted.mp4"
        corrupted.write_bytes(b"not a valid video file")
        
        adapter = FileVideoAdapter(str(corrupted))
        result = await adapter.connect()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_frame_when_not_connected(self):
        adapter = MockVideoAdapter()
        # Don't connect
        
        frame = await adapter.get_frame()
        assert frame is None
    
    @pytest.mark.asyncio
    async def test_get_info_before_connect(self):
        adapter = MockVideoAdapter(width=640, height=480, fps=30.0, frame_count=100)
        
        info = await adapter.get_info()
        
        assert info is not None
        assert info.width == 640
        assert info.height == 480


class TestVideoProcessingIntegration:
    """Integration tests for the complete video processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_mock_video(self):
        """Test the complete pipeline: adapter -> processor -> frames."""
        adapter = MockVideoAdapter(width=1280, height=720, fps=30.0, frame_count=30)
        await adapter.connect()
        
        processor = VideoStreamProcessor(adapter, frame_skip=3, enable_quality_check=True)
        
        frame_count = 0
        total_quality = 0.0
        
        async for frame_data in processor.process_stream():
            frame = frame_data["frame"]
            processed = frame_data["processed"]
            quality = frame_data["quality"]
            
            # Verify frame properties
            assert frame.width == 1280
            assert frame.height == 720
            assert frame.frame.shape == (720, 1280, 3)
            
            # Verify processed frame
            assert processed.processed.shape == (1, 3, 640, 640)  # Default target size
            assert processed.scale_x > 0
            assert processed.scale_y > 0
            
            # Verify quality
            assert 0 <= quality["quality_score"] <= 1
            total_quality += quality["quality_score"]
            
            frame_count += 1
            if frame_count >= 5:
                break
        
        assert frame_count == 5
        avg_quality = total_quality / frame_count
        assert avg_quality > 0.5  # Mock frames should have decent quality
    
    @pytest.mark.asyncio
    async def test_pipeline_with_different_frame_sizes(self):
        """Test pipeline handles different input resolutions."""
        for width, height in [(1920, 1080), (1280, 720), (640, 480), (320, 240)]:
            adapter = MockVideoAdapter(width=width, height=height, fps=30.0, frame_count=10)
            await adapter.connect()
            
            processor = VideoStreamProcessor(adapter, frame_skip=1, enable_quality_check=False)
            
            async for frame_data in processor.process_stream():
                processed = frame_data["processed"]
                # All should be resized to target size
                assert processed.processed.shape == (1, 3, 640, 640)
                break  # Just test first frame
            
            await adapter.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])