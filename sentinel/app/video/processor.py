"""
SENTINEL — Fast, Clean, and Lightweight Frame Preprocessing Pipeline.

Strictly optimized for low-RAM Windows CPU environments (~145 MB footprint target).
Provides:
1. Frame skipping (FRAME_SKIP) to throttle CPU load during live streams.
2. Aspect-ratio-preserving resize & padding/letterbox with fast interpolation
   (cv2.INTER_AREA when downscaling, cv2.INTER_LINEAR when upscaling).
3. Optional/toggleable color conversion (BGR <-> RGB) and low-cost CLAHE contrast
   enhancement for low-light or glare drone surveillance (disabled by default).
4. Safe frame validation, zero memory leaks, and accurate bounding box denormalization.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
from loguru import logger

from app.config import settings


@dataclass
class ProcessedFrame:
    """Standardized preprocessed frame container for detection and tracking pipeline."""
    original: np.ndarray
    processed: np.ndarray  # NCHW normalized float32 tensor: (1, 3, H, W)
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    orig_w: int = 0
    orig_h: int = 0
    new_w: int = 0
    new_h: int = 0
    frame_index: int = 0
    timestamp: float = 0.0
    color_format: str = "RGB"
    processing_time_ms: float = 0.0
    clahe_applied: bool = False

    def __post_init__(self):
        if self.orig_w == 0 and self.original is not None:
            self.orig_h, self.orig_w = self.original.shape[:2]


class FramePreprocessor:
    """Fast, memory-efficient CPU frame preprocessor for drone surveillance video."""

    def __init__(
        self,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        keep_aspect_ratio: bool = True,
        frame_skip: Optional[int] = None,
        enable_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        color_format: str = "RGB",
    ):
        self.target_width = target_width or settings.processing_width
        self.target_height = target_height or settings.processing_height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.frame_skip = max(1, frame_skip or settings.frame_skip)
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_grid_size = clahe_grid_size
        self.color_format = color_format.upper()

        # Cache CLAHE instance if enabled to avoid re-allocation per frame
        self._clahe: Optional[cv2.CLAHE] = None
        if self.enable_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size,
            )

    # -------------------------------------------------------------------------
    # Frame Validation & Skipping
    # -------------------------------------------------------------------------
    @staticmethod
    def validate_frame(frame: Optional[np.ndarray]) -> Tuple[bool, str]:
        """Validates frame integrity without memory copy.
        Returns (is_valid, reason).
        """
        if frame is None:
            return False, "Frame is None"
        if not isinstance(frame, np.ndarray):
            return False, f"Frame must be np.ndarray, got {type(frame).__name__}"
        if frame.size == 0:
            return False, "Frame array is empty (size 0)"
        if frame.ndim not in (2, 3):
            return False, f"Frame must have 2 or 3 dimensions, got {frame.ndim}"
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return False, f"Invalid dimensions: height={h}, width={w}"
        return True, "Valid frame"

    def should_skip(self, frame_index: int) -> bool:
        """Returns True if the frame should be skipped according to frame_skip setting."""
        if self.frame_skip <= 1:
            return False
        return (frame_index % self.frame_skip) != 0

    # -------------------------------------------------------------------------
    # Static Resizing Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def resize_preview(image: Optional[np.ndarray], max_width: int = 640) -> Optional[np.ndarray]:
        """Downscale frame keeping aspect ratio if width exceeds max_width.
        Uses cv2.INTER_AREA for crisp, fast downscaling on CPU.
        Returns None safely if input is None or empty.
        """
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if w <= max_width:
            return image
        scale = max_width / float(w)
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def resize_aspect(
        image: np.ndarray,
        target_width: int = 640,
        max_height: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resize image preserving aspect ratio with metadata for box mapping.
        Uses INTER_AREA when downscaling and INTER_LINEAR when upscaling.
        """
        if image is None or image.size == 0:
            raise ValueError("Input frame is None or empty")
        orig_h, orig_w = image.shape[:2]
        scale = target_width / float(orig_w)
        new_w = target_width
        new_h = int(orig_h * scale)

        if max_height and new_h > max_height:
            scale = max_height / float(orig_h)
            new_w = int(orig_w * scale)
            new_h = max_height

        interp = cv2.INTER_AREA if (new_w < orig_w or new_h < orig_h) else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        meta = {
            "orig_w": orig_w,
            "orig_h": orig_h,
            "new_w": new_w,
            "new_h": new_h,
            "scale_x": scale,
            "scale_y": scale,
            "pad_left": 0,
            "pad_top": 0,
        }
        return resized, meta

    # -------------------------------------------------------------------------
    # Contrast & Light Normalization (CLAHE)
    # -------------------------------------------------------------------------
    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Lightweight CPU-vectorized contrast enhancement for low-light or harsh glare drone shots.
        Operates on Luminance (L) channel in LAB color space to preserve natural chromaticity.
        """
        if image is None or image.size == 0:
            return image

        if len(image.shape) == 2 or image.shape[2] == 1:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            return clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_enhanced = clahe.apply(l_channel)

        merged = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # -------------------------------------------------------------------------
    # Core Pipeline Preprocessing
    # -------------------------------------------------------------------------
    def preprocess(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> ProcessedFrame:
        """Preprocesses input frame with validation, optional CLAHE, aspect-fit/letterbox,
        and normalization into float32 NCHW tensor (1, 3, H, W).
        """
        is_valid, err_msg = self.validate_frame(frame)
        if not is_valid:
            raise ValueError(f"Invalid frame passed to FramePreprocessor: {err_msg}")

        t0 = time.perf_counter()
        original_height, original_width = frame.shape[:2]

        # 1. Optional low-light / glare contrast enhancement
        work_frame = frame
        clahe_applied = False
        if self.enable_clahe:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit,
                    tileGridSize=self.clahe_grid_size,
                )
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                work_frame = self._clahe.apply(frame)
            else:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                l_enhanced = self._clahe.apply(l_channel)
                work_frame = cv2.cvtColor(cv2.merge((l_enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
            clahe_applied = True

        # 2. Aspect-ratio preserving resize & padding
        if self.keep_aspect_ratio:
            processed_img, scale_x, scale_y, pad_left, pad_top = self._letterbox_resize(
                work_frame, self.target_width, self.target_height
            )
        else:
            interp = cv2.INTER_AREA if (self.target_width < original_width or self.target_height < original_height) else cv2.INTER_LINEAR
            processed_img = cv2.resize(work_frame, (self.target_width, self.target_height), interpolation=interp)
            scale_x = self.target_width / original_width
            scale_y = self.target_height / original_height
            pad_left = 0
            pad_top = 0

        # 3. Color conversion (OpenCV BGR -> RGB)
        if self.color_format == "RGB" and len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        # 4. Normalize to NCHW float32 tensor
        processed_tensor = self._normalize(processed_img)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return ProcessedFrame(
            original=frame,
            processed=processed_tensor,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_left=pad_left,
            pad_top=pad_top,
            orig_w=original_width,
            orig_h=original_height,
            new_w=self.target_width,
            new_h=self.target_height,
            frame_index=frame_index,
            timestamp=timestamp,
            color_format=self.color_format,
            processing_time_ms=dt_ms,
            clahe_applied=clahe_applied,
        )

    def _letterbox_resize(
        self, image: np.ndarray, target_width: int, target_height: int
    ) -> Tuple[np.ndarray, float, float, int, int]:
        """Letterbox resize with grey padding (114, 114, 114) preserving aspect ratio."""
        height, width = image.shape[:2]

        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_width, new_height), interpolation=interp)

        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        pad_right = target_width - new_width - pad_left
        pad_bottom = target_height - new_height - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return padded, scale, scale, pad_left, pad_top

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Convert uint8 HWC image to normalized float32 NCHW tensor."""
        normalized = image.astype(np.float32) / 255.0
        if normalized.ndim == 3:
            normalized = np.transpose(normalized, (2, 0, 1))
            normalized = np.expand_dims(normalized, axis=0)
        elif normalized.ndim == 2:
            normalized = np.expand_dims(normalized, axis=(0, 1))
        return normalized

    # -------------------------------------------------------------------------
    # Bounding Box Coordinate Mapping
    # -------------------------------------------------------------------------
    def denormalize_boxes(
        self, boxes: np.ndarray, processed_frame: ProcessedFrame
    ) -> np.ndarray:
        """Map bounding boxes [x1, y1, x2, y2] from padded processed tensor back to original frame coordinates."""
        if len(boxes) == 0:
            return boxes

        denormalized = boxes.copy().astype(np.float32)
        denormalized[:, [0, 2]] -= processed_frame.pad_left
        denormalized[:, [1, 3]] -= processed_frame.pad_top
        denormalized[:, [0, 2]] /= processed_frame.scale_x
        denormalized[:, [1, 3]] /= processed_frame.scale_y

        denormalized[:, 0] = np.clip(denormalized[:, 0], 0, processed_frame.original.shape[1])
        denormalized[:, 1] = np.clip(denormalized[:, 1], 0, processed_frame.original.shape[0])
        denormalized[:, 2] = np.clip(denormalized[:, 2], 0, processed_frame.original.shape[1])
        denormalized[:, 3] = np.clip(denormalized[:, 3], 0, processed_frame.original.shape[0])

        return denormalized

    @staticmethod
    def map_boxes_to_original(boxes: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        """Map boxes [x1, y1, x2, y2] from resized image back using metadata dictionary."""
        if len(boxes) == 0:
            return boxes
        mapped = boxes.copy().astype(np.float32)
        pad_left = meta.get("pad_left", 0)
        pad_top = meta.get("pad_top", 0)
        scale_x = meta.get("scale_x", 1.0)
        scale_y = meta.get("scale_y", 1.0)
        orig_w = meta.get("orig_w", 1920)
        orig_h = meta.get("orig_h", 1080)

        mapped[:, [0, 2]] = (mapped[:, [0, 2]] - pad_left) / scale_x
        mapped[:, [1, 3]] = (mapped[:, [1, 3]] - pad_top) / scale_y

        mapped[:, 0] = np.clip(mapped[:, 0], 0, orig_w)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, orig_h)
        mapped[:, 2] = np.clip(mapped[:, 2], 0, orig_w)
        mapped[:, 3] = np.clip(mapped[:, 3], 0, orig_h)
        return mapped


class FrameQualityChecker:
    """Fast heuristics to identify dark, overexposed, low-contrast, or blurry video frames."""

    def __init__(
        self,
        min_brightness: float = 10.0,
        max_brightness: float = 240.0,
        min_contrast: float = 10.0,
        blur_threshold: float = 100.0,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.blur_threshold = blur_threshold

    def check_quality(self, frame: np.ndarray) -> dict:
        """Analyzes frame brightness, contrast, and Laplacian blur score."""
        if frame is None or frame.size == 0:
            return {
                "brightness": 0.0,
                "contrast": 0.0,
                "blur_score": 0.0,
                "is_dark": True,
                "is_bright": False,
                "is_low_contrast": True,
                "is_blurry": True,
                "quality_score": 0.0,
                "passed": False,
            }

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        is_dark = brightness < self.min_brightness
        is_bright = brightness > self.max_brightness
        is_low_contrast = contrast < self.min_contrast
        is_blurry = blur_score < self.blur_threshold

        quality_score = 1.0
        if is_dark or is_bright:
            quality_score *= 0.7
        if is_low_contrast:
            quality_score *= 0.8
        if is_blurry:
            quality_score *= 0.6

        return {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_low_contrast": is_low_contrast,
            "is_blurry": is_blurry,
            "quality_score": float(quality_score),
            "passed": bool(quality_score > 0.5),
        }


class VideoStreamProcessor:
    """Async generator processing video frames with frame-skipping, resolution capping,
    quality verification, and model tensor preprocessing.
    """

    def __init__(
        self,
        adapter,
        frame_skip: Optional[int] = None,
        max_resolution_width: int = 640,
        enable_quality_check: bool = True,
        enable_clahe: bool = False,
    ):
        self.adapter = adapter
        self.frame_skip = max(1, frame_skip or settings.frame_skip)
        self.max_resolution_width = max_resolution_width
        self.preprocessor = FramePreprocessor(
            target_width=max_resolution_width,
            target_height=max_resolution_width,
            frame_skip=self.frame_skip,
            enable_clahe=enable_clahe,
        )
        self.quality_checker = FrameQualityChecker() if enable_quality_check else None
        self._frame_counter = 0
        self._processed_counter = 0
        self._skipped_counter = 0

    async def process_stream(self):
        """Async generator yielding processed frame packets."""
        async for frame_data in self._frame_generator():
            yield frame_data

    async def _frame_generator(self):
        while self.adapter.is_connected:
            video_frame = await self.adapter.get_frame()
            if video_frame is None:
                break

            self._frame_counter += 1

            if self.preprocessor.should_skip(self._frame_counter):
                self._skipped_counter += 1
                continue

            # Safe frame validation
            is_valid, msg = FramePreprocessor.validate_frame(video_frame.frame)
            if not is_valid:
                logger.warning(f"Dropping corrupted frame {self._frame_counter}: {msg}")
                continue

            # Resolution cap (640px) right after frame capture for low CPU/RAM consumption
            if self.max_resolution_width and video_frame.frame.shape[1] > self.max_resolution_width:
                video_frame.frame = FramePreprocessor.resize_preview(video_frame.frame, max_width=self.max_resolution_width)
                video_frame.width = video_frame.frame.shape[1]
                video_frame.height = video_frame.frame.shape[0]

            quality_info = None
            if self.quality_checker:
                quality_info = self.quality_checker.check_quality(video_frame.frame)
                if not quality_info["passed"]:
                    logger.warning(
                        f"Frame {video_frame.frame_number} quality check failed: {quality_info}"
                    )

            processed = self.preprocessor.preprocess(
                video_frame.frame,
                frame_index=self._frame_counter,
                timestamp=video_frame.timestamp,
            )

            self._processed_counter += 1

            yield {
                "frame": video_frame,
                "processed": processed,
                "quality": quality_info,
            }

    def get_stats(self) -> dict:
        """Returns processing statistics."""
        return {
            "total_frames": self._frame_counter,
            "processed_frames": self._processed_counter,
            "skipped_frames": self._skipped_counter,
            "skip_ratio": self.frame_skip,
            "max_resolution_width": self.max_resolution_width,
        }
