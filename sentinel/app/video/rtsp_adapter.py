"""Resilient and low-latency RTSP Live Video Capture Adapter for SENTINEL.

Optimized for drone aerial surveillance streams with non-blocking latest-frame retrieval,
OpenCV FFmpeg TCP/UDP transport configuration, automatic exponential backoff reconnection,
and lightweight resolution downscaling (640px target) for low-RAM Windows CPU environments.
"""

import os
import time
import math
import socket
import threading
import asyncio
from urllib.parse import urlparse
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import cv2
import numpy as np
from loguru import logger

from app.video.adapter import VideoAdapter, VideoFrame, VideoInfo
from app.config import settings


@dataclass
class RTSPStreamStats:
    """Telemetry and operational statistics for an active RTSP stream."""
    status: str  # 'DISCONNECTED', 'CONNECTING', 'CONNECTED', 'RECONNECTING', 'FAILED'
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frames_received: int = 0
    frames_dropped: int = 0
    reconnect_attempts: int = 0
    latency_estimate_ms: float = 0.0
    last_error: str = ""
    transport: str = "tcp"


class RTSPVideoAdapter(VideoAdapter):
    """Resilient RTSP live video adapter with background non-blocking frame capture."""

    def __init__(
        self,
        rtsp_url: str,
        frame_skip: int = 1,
        target_width: int = 640,
        transport: str = "tcp",  # 'tcp' or 'udp'
        connect_timeout_sec: float = 3.0,
        read_timeout_sec: float = 3.0,
        max_reconnect_attempts: int = 10,
        initial_backoff_sec: float = 1.0,
        max_backoff_sec: float = 5.0,
    ):
        super().__init__(rtsp_url, frame_skip)
        self.rtsp_url = rtsp_url.strip()
        self.target_width = target_width
        self.transport = transport.lower()
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.read_timeout_sec = float(read_timeout_sec)
        self.max_reconnect_attempts = int(max_reconnect_attempts)
        self.initial_backoff_sec = float(initial_backoff_sec)
        self.max_backoff_sec = float(max_backoff_sec)

        # Threading and latest-frame buffer (prevents queue buildup & latency lag)
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_frame: Optional[VideoFrame] = None
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._last_frame_timestamp: float = 0.0

        # Stream statistics
        self._status = "DISCONNECTED"
        self._frames_received = 0
        self._frames_dropped = 0
        self._reconnect_attempts = 0
        self._last_error = ""
        self._fps_measured = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_frame_count = 0

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._connected and self._status == "CONNECTED"

    # -------------------------------------------------------------------------
    # Connection Lifecycle
    # -------------------------------------------------------------------------
    async def connect(self) -> bool:
        """Connect to the RTSP live stream asynchronously without blocking event loops."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_connect)

    def _sync_connect(self) -> bool:
        """Synchronous connect implementation with socket pre-flight check."""
        if not self.rtsp_url:
            self._last_error = "RTSP URL cannot be empty."
            self._status = "FAILED"
            return False

        self._status = "CONNECTING"
        self._stop_event.clear()

        # Step 1: Pre-flight network reachability check if standard network RTSP
        if self.rtsp_url.startswith("rtsp://") or self.rtsp_url.startswith("rtsps://"):
            reachable, err = self._check_network_reachability(self.rtsp_url, timeout=self.connect_timeout_sec)
            if not reachable:
                self._last_error = err
                self._status = "FAILED"
                logger.warning(f"RTSP pre-flight reachability failed for '{self.rtsp_url}': {err}")
                return False

        # Step 2: Open OpenCV VideoCapture with FFmpeg low-latency flags
        success = self._open_capture()
        if not success:
            self._status = "FAILED"
            return False

        # Step 3: Launch background reader thread for non-blocking frame retrieval
        self._status = "CONNECTED"
        self._connected = True
        self._reader_thread = threading.Thread(
            target=self._stream_reader_worker,
            daemon=True,
            name=f"RTSPReader-{id(self)}",
        )
        self._reader_thread.start()
        logger.info(f"RTSP Live Stream connected & reader worker active: {self.rtsp_url}")
        return True

    def _open_capture(self) -> bool:
        """Initialize OpenCV VideoCapture with low-latency FFmpeg parameters."""
        try:
            # Configure FFmpeg transport and latency optimizations
            ffmpeg_opts = f"rtsp_transport;{self.transport}|fflags;nobuffer|max_delay;500000|stimeout;{int(self.connect_timeout_sec * 1000000)}"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_opts

            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Fallback for local files if CAP_FFMPEG fails to open
            if not cap.isOpened() and not self.rtsp_url.startswith(("rtsp://", "rtsps://")):
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set hardware timeout properties if supported
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.connect_timeout_sec * 1000))
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout_sec * 1000))

            if not cap.isOpened():
                self._last_error = f"OpenCV VideoCapture failed to open '{self.rtsp_url}'."
                logger.warning(self._last_error)
                cap.release()
                return False

            # Query stream metadata
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.target_width
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(self.target_width * 0.5625)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            fps = fps if (0 < fps <= 120) else 30.0

            # Apply downscaled target resolution if applicable
            if self.target_width and w > self.target_width:
                aspect = h / max(w, 1)
                final_w = self.target_width
                final_h = int(self.target_width * aspect)
            else:
                final_w, final_h = w, h

            self._info = VideoInfo(
                width=final_w,
                height=final_h,
                fps=fps,
                frame_count=0,
                duration=0.0,
                codec=f"RTSP-{self.transport.upper()}",
            )
            self._cap = cap
            return True

        except Exception as e:
            self._last_error = f"Exception opening RTSP stream: {e}"
            logger.error(self._last_error)
            return False

    def _check_network_reachability(self, url: str, timeout: float = 2.0) -> Tuple[bool, str]:
        """Perform fast TCP socket probe to verify host:port reachability."""
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or (554 if parsed.scheme == "rtsp" else 80)
            if not host:
                return False, f"Invalid RTSP URL: missing host ({url})"

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                sock.connect((host, port))
                sock.close()
                return True, "Reachable"
            except socket.timeout:
                return False, f"Connection to {host}:{port} timed out ({timeout:.1f}s)"
            except ConnectionRefusedError:
                return False, f"Connection refused at {host}:{port}"
            except Exception as e:
                return False, f"Cannot reach {host}:{port} ({e})"
        except Exception as e:
            return False, f"URL parse error: {e}"

    async def disconnect(self) -> None:
        """Disconnect and release camera stream cleanly."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_disconnect)

    def _sync_disconnect(self) -> None:
        """Synchronous disconnection and thread cleanup."""
        self._stop_event.set()
        self._connected = False
        self._status = "DISCONNECTED"

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._latest_frame = None
            self._latest_raw_frame = None

        logger.info(f"RTSP Live Stream disconnected: {self.rtsp_url}")

    # -------------------------------------------------------------------------
    # Background Stream Reader Worker
    # -------------------------------------------------------------------------
    def _stream_reader_worker(self) -> None:
        """Dedicated background thread reading frames from OpenCV into 1-frame atomic buffer.
        Discards stale queued frames so consumer receives the freshest live view (< 50ms latency).
        """
        consecutive_failures = 0
        skip_counter = 0

        while not self._stop_event.is_set():
            if not self._cap or not self._cap.isOpened():
                self._handle_reconnect()
                continue

            try:
                # Fast grab of the next frame
                grabbed = self._cap.grab()
                if not grabbed:
                    # Check if EOF reached on a seekable stream (e.g., video file loop)
                    frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) if self._cap else 0
                    if frame_count > 0:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        grabbed = self._cap.grab()

                    if not grabbed:
                        consecutive_failures += 1
                        if consecutive_failures >= 10:
                            logger.warning(f"RTSP stream grab failed {consecutive_failures} times. Initiating reconnect.")
                            self._handle_reconnect()
                            consecutive_failures = 0
                        time.sleep(0.01)
                        continue

                consecutive_failures = 0
                skip_counter += 1

                # Frame skip handling
                if skip_counter % self.frame_skip != 0:
                    self._frames_dropped += 1
                    continue

                # Retrieve the actual image array
                ret, frame = self._cap.retrieve()
                if not ret or frame is None:
                    continue

                # Downscale immediately to target width cap (640px) to keep RAM < 145 MB
                orig_h, orig_w = frame.shape[:2]
                if self.target_width and orig_w > self.target_width:
                    scale = self.target_width / orig_w
                    new_w = self.target_width
                    new_h = int(orig_h * scale)
                    proc_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    proc_frame = frame
                    new_w, new_h = orig_w, orig_h

                self._frames_received += 1
                now = time.time()
                self._frame_count += 1

                # Measured FPS
                self._fps_frame_count += 1
                if (now - self._last_fps_calc_time) >= 1.0:
                    self._fps_measured = round(self._fps_frame_count / (now - self._last_fps_calc_time), 1)
                    self._fps_frame_count = 0
                    self._last_fps_calc_time = now

                vf = VideoFrame(
                    frame=proc_frame,
                    frame_number=self._frame_count,
                    timestamp=now,
                    width=new_w,
                    height=new_h,
                )

                # Atomically update freshest frame
                with self._lock:
                    self._latest_frame = vf
                    self._latest_raw_frame = proc_frame
                    self._last_frame_timestamp = now

            except Exception as e:
                logger.warning(f"Exception in RTSP reader worker: {e}")
                self._handle_reconnect()

    def _handle_reconnect(self) -> None:
        """Handle automatic exponential backoff reconnection."""
        if self._stop_event.is_set():
            return

        self._status = "RECONNECTING"
        self._reconnect_attempts += 1
        delay = min(self.initial_backoff_sec * (1.5 ** (self._reconnect_attempts - 1)), self.max_backoff_sec)
        logger.info(f"RTSP Reconnecting (Attempt {self._reconnect_attempts}) in {delay:.1f}s...")

        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

        time.sleep(delay)

        if self._stop_event.is_set():
            return

        opened = self._open_capture()
        if opened:
            self._status = "CONNECTED"
            self._reconnect_attempts = 0
            logger.info("RTSP Reconnection successful.")
        else:
            if self._reconnect_attempts >= self.max_reconnect_attempts:
                self._status = "FAILED"
                self._last_error = f"Max reconnect attempts ({self.max_reconnect_attempts}) reached."

    # -------------------------------------------------------------------------
    # Frame Retrieval Methods
    # -------------------------------------------------------------------------
    async def get_frame(self) -> Optional[VideoFrame]:
        """Asynchronously retrieve the latest frame from the buffer."""
        return self.get_latest_frame()

    def get_latest_frame(self) -> Optional[VideoFrame]:
        """Non-blocking synchronous retrieval of the latest live frame."""
        with self._lock:
            return self._latest_frame

    def get_latest_raw_frame(self) -> Optional[np.ndarray]:
        """Non-blocking retrieval of raw OpenCV image array (BGR)."""
        with self._lock:
            return self._latest_raw_frame

    async def get_info(self) -> VideoInfo:
        if not self._info:
            await self.connect()
        return self._info or VideoInfo(width=640, height=360, fps=30.0, frame_count=0, duration=0.0, codec="RTSP")

    def get_stats(self) -> Dict[str, Any]:
        """Return operational telemetry statistics for the RTSP live link."""
        return {
            "status": self._status,
            "is_connected": self.is_connected,
            "fps": self._fps_measured,
            "width": self._info.width if self._info else 0,
            "height": self._info.height if self._info else 0,
            "frames_received": self._frames_received,
            "frames_dropped": self._frames_dropped,
            "reconnect_attempts": self._reconnect_attempts,
            "transport": self.transport,
            "last_error": self._last_error,
            "target_width": self.target_width,
            "frame_skip": self.frame_skip,
        }
