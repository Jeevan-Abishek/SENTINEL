import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger

from app.config import settings


@dataclass
class ProcessedFrame:
    original: np.ndarray
    processed: np.ndarray
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int


class FramePreprocessor:
    def __init__(
        self,
        target_width: int = None,
        target_height: int = None,
        keep_aspect_ratio: bool = True,
    ):
        self.target_width = target_width or settings.processing_width
        self.target_height = target_height or settings.processing_height
        self.keep_aspect_ratio = keep_aspect_ratio

    def preprocess(self, frame: np.ndarray) -> ProcessedFrame:
        original_height, original_width = frame.shape[:2]

        if self.keep_aspect_ratio:
            processed, scale_x, scale_y, pad_left, pad_top = self._letterbox_resize(
                frame, self.target_width, self.target_height
            )
        else:
            processed = cv2.resize(frame, (self.target_width, self.target_height))
            scale_x = self.target_width / original_width
            scale_y = self.target_height / original_height
            pad_left = 0
            pad_top = 0

        processed = self._normalize(processed)

        return ProcessedFrame(
            original=frame,
            processed=processed,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_left=pad_left,
            pad_top=pad_top,
        )

    def _letterbox_resize(
        self, image: np.ndarray, target_width: int, target_height: int
    ) -> Tuple[np.ndarray, float, float, int, int]:
        height, width = image.shape[:2]

        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

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
        normalized = image.astype(np.float32) / 255.0
        normalized = np.transpose(normalized, (2, 0, 1))
        normalized = np.expand_dims(normalized, axis=0)
        return normalized

    def denormalize_boxes(
        self, boxes: np.ndarray, processed_frame: ProcessedFrame
    ) -> np.ndarray:
        if len(boxes) == 0:
            return boxes

        denormalized = boxes.copy()
        denormalized[:, [0, 2]] -= processed_frame.pad_left
        denormalized[:, [1, 3]] -= processed_frame.pad_top
        denormalized[:, [0, 2]] /= processed_frame.scale_x
        denormalized[:, [1, 3]] /= processed_frame.scale_y

        denormalized[:, 0] = np.clip(denormalized[:, 0], 0, processed_frame.original.shape[1])
        denormalized[:, 1] = np.clip(denormalized[:, 1], 0, processed_frame.original.shape[0])
        denormalized[:, 2] = np.clip(denormalized[:, 2], 0, processed_frame.original.shape[1])
        denormalized[:, 3] = np.clip(denormalized[:, 3], 0, processed_frame.original.shape[0])

        return denormalized


class FrameQualityChecker:
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        contrast = np.std(gray)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

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
            "brightness": float(brightness),
            "contrast": float(contrast),
            "blur_score": float(blur_score),
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_low_contrast": is_low_contrast,
            "is_blurry": is_blurry,
            "quality_score": quality_score,
            "passed": quality_score > 0.5,
        }


class VideoStreamProcessor:
    def __init__(
        self,
        adapter,
        frame_skip: int = None,
        enable_quality_check: bool = True,
    ):
        self.adapter = adapter
        self.frame_skip = frame_skip or settings.frame_skip
        self.preprocessor = FramePreprocessor()
        self.quality_checker = FrameQualityChecker() if enable_quality_check else None
        self._frame_counter = 0
        self._processed_counter = 0

    async def process_stream(self):
        async for frame_data in self._frame_generator():
            yield frame_data

    async def _frame_generator(self):
        while self.adapter.is_connected:
            video_frame = await self.adapter.get_frame()
            if video_frame is None:
                break

            self._frame_counter += 1

            if self._frame_counter % self.frame_skip != 0:
                continue

            quality_info = None
            if self.quality_checker:
                quality_info = self.quality_checker.check_quality(video_frame.frame)
                if not quality_info["passed"]:
                    logger.warning(
                        f"Frame {video_frame.frame_number} quality check failed: {quality_info}"
                    )

            processed = self.preprocessor.preprocess(video_frame.frame)

            self._processed_counter += 1

            yield {
                "frame": video_frame,
                "processed": processed,
                "quality": quality_info,
            }

    def get_stats(self) -> dict:
        return {
            "total_frames": self._frame_counter,
            "processed_frames": self._processed_counter,
            "skip_ratio": self.frame_skip,
        }