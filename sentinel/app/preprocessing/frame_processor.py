"""
Frame Preprocessor Module.
Handles frame resizing, normalization, contrast enhancement, and quality checking.
"""

from app.video.processor import FramePreprocessor, ProcessedFrame, FrameQualityChecker

# Alias matching master architecture
FrameProcessor = FramePreprocessor

__all__ = ["FramePreprocessor", "FrameProcessor", "ProcessedFrame", "FrameQualityChecker"]

