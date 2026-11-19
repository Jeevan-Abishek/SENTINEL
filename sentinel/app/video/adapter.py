from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, AsyncGenerator
import cv2
import numpy as np
from loguru import logger


@dataclass
class VideoFrame:
    frame: np.ndarray
    frame_number: int
    timestamp: float
    width: int
    height: int


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str


class VideoAdapter(ABC):
    def __init__(self, source: str, frame_skip: int = 1):
        self.source = source
        self.frame_skip = frame_skip
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None
        self._connected = False
        self._frame_count = 0

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def get_frame(self) -> Optional[VideoFrame]:
        pass

    @abstractmethod
    async def get_info(self) -> VideoInfo:
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def info(self) -> Optional[VideoInfo]:
        return self._info


class FileVideoAdapter(VideoAdapter):
    def __init__(self, file_path: str, frame_skip: int = 1):
        super().__init__(file_path, frame_skip)
        self.file_path = file_path

    async def connect(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.file_path)
            if not self._cap.isOpened():
                logger.error(f"Failed to open video file: {self.file_path}")
                return False

            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            codec_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])

            duration = frame_count / fps if fps > 0 else 0

            self._info = VideoInfo(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec,
            )
            self._connected = True
            logger.info(f"Connected to video file: {self.file_path}")
            logger.info(f"Video info: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")
            return True
        except Exception as e:
            logger.error(f"Error connecting to video file: {e}")
            return False

    async def disconnect(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._connected = False
        logger.info(f"Disconnected from video file: {self.file_path}")

    async def get_frame(self) -> Optional[VideoFrame]:
        if not self._connected or not self._cap:
            return None

        for _ in range(self.frame_skip):
            ret, frame = self._cap.read()
            if not ret:
                return None
            self._frame_count += 1

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._frame_count += 1
        timestamp = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        return VideoFrame(
            frame=frame,
            frame_number=self._frame_count,
            timestamp=timestamp,
            width=self._info.width if self._info else frame.shape[1],
            height=self._info.height if self._info else frame.shape[0],
        )

    async def get_info(self) -> VideoInfo:
        if not self._info:
            await self.connect()
        return self._info

    async def seek(self, frame_number: int) -> bool:
        if not self._cap:
            return False
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    async def seek_time(self, timestamp: float) -> bool:
        if not self._cap:
            return False
        return self._cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)


# Re-export resilient RTSPVideoAdapter from dedicated rtsp_adapter module
from app.video.rtsp_adapter import RTSPVideoAdapter, RTSPStreamStats


class MockVideoAdapter(VideoAdapter):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        frame_count: int = 1000,
        frame_skip: int = 1,
    ):
        super().__init__("mock", frame_skip)
        self.mock_width = width
        self.mock_height = height
        self.mock_fps = fps
        self.mock_frame_count = frame_count

    async def connect(self) -> bool:
        self._info = VideoInfo(
            width=self.mock_width,
            height=self.mock_height,
            fps=self.mock_fps,
            frame_count=self.mock_frame_count,
            duration=self.mock_frame_count / self.mock_fps,
            codec="MOCK",
        )
        self._connected = True
        logger.info(f"Mock video adapter connected: {self.mock_width}x{self.mock_height} @ {self.mock_fps}fps")
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Mock video adapter disconnected")

    async def get_frame(self) -> Optional[VideoFrame]:
        if not self._connected or self._frame_count >= self.mock_frame_count:
            return None

        # Skip frames according to frame_skip
        for _ in range(self.frame_skip - 1):
            self._frame_count += 1
            if self._frame_count >= self.mock_frame_count:
                return None

        # Generate the actual frame to return
        self._frame_count += 1
        if self._frame_count > self.mock_frame_count:
            return None

        frame = np.random.randint(0, 255, (self.mock_height, self.mock_width, 3), dtype=np.uint8)
        timestamp = self._frame_count / self.mock_fps

        return VideoFrame(
            frame=frame,
            frame_number=self._frame_count,
            timestamp=timestamp,
            width=self.mock_width,
            height=self.mock_height,
        )

    async def get_info(self) -> VideoInfo:
        if not self._info:
            await self.connect()
        return self._info


def create_video_adapter(source: str, frame_skip: int = 1, **kwargs) -> VideoAdapter:
    if source.startswith("rtsp://") or source.startswith("rtsps://"):
        return RTSPVideoAdapter(source, frame_skip, **kwargs)
    elif source == "mock":
        return MockVideoAdapter(**kwargs)
    else:
        return FileVideoAdapter(source, frame_skip)